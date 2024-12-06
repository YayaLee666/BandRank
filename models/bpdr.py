import torch
import torch.nn.functional as F
from torch_scatter import scatter
from models.time import TimeEncoder
from models.mlp import MLP
import torch.nn as nn
from torch.fft import fft, ifft


class FreMLP(nn.Module):
    def __init__(self, input_dim, output_dim, N=3, sparsity_threshold=0.01):
        super(FreMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.N = N 
        self.sparsity_threshold = sparsity_threshold
        self.scale = 0.02

        # alpha_k parameters for each frequency band
        self.alpha_k = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(N)])

        # r1, i1, rb1, ib1 parameters for frequency transformation
        self.r1 = nn.ParameterList([nn.Parameter(self.scale * torch.randn(input_dim, output_dim)) for _ in range(N)])
        self.i1 = nn.ParameterList([nn.Parameter(self.scale * torch.randn(input_dim, output_dim)) for _ in range(N)])
        self.rb1 = nn.ParameterList([nn.Parameter(self.scale * torch.randn(output_dim)) for _ in range(N)])
        self.ib1 = nn.ParameterList([nn.Parameter(self.scale * torch.randn(output_dim)) for _ in range(N)])

        # Linear layer to combine output
        self.linear = nn.Linear(output_dim * N, output_dim)

    def calculate_Qk(self, k, energy_sum):
        factor = (2 * (k + 1) - 1) / (2 * self.N)
        Q_k = self.alpha_k[k] * factor * energy_sum
        return Q_k

    def create_band_pass_filter(self, x_fft):
        B, L = x_fft.shape

        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        energy_sum = energy.sum()
        

        Q_k_list = [self.calculate_Qk(k, energy_sum) for k in range(self.N)]

        masks = []
        for k in range(self.N):
            Q_k = Q_k_list[k]

            b_k = (1 / self.alpha_k[k]) * energy_sum / (self.N * 2)

            distances = torch.abs(energy - Q_k)
            mask = (energy >= (Q_k - b_k)) & (energy <= (Q_k + b_k))

            masks.append(mask)

        return masks

    def forward(self, x):
        if not torch.is_complex(x):
            x = torch.view_as_complex(torch.stack((x.float(), torch.zeros_like(x).float()), dim=-1))

        x_fft = fft(x)
        
        freq_masks = self.create_band_pass_filter(x_fft)
        freq_ffts = [x_fft * mask.unsqueeze(-1).expand_as(x_fft) for mask in freq_masks]

        outputs = []
        for k in range(self.N):
            freq_fft = freq_ffts[k]

            o_real = torch.einsum('...n,nd->...d', freq_fft.real, self.r1[k]) - \
                     torch.einsum('...n,nd->...d', freq_fft.imag, self.i1[k]) + self.rb1[k]
            o_imag = torch.einsum('...n,nd->...d', freq_fft.imag, self.r1[k]) + \
                     torch.einsum('...n,nd->...d', freq_fft.real, self.i1[k]) + self.ib1[k]

            y_fft = torch.stack([F.relu(o_real), F.relu(o_imag)], dim=-1)
            y_fft = F.softshrink(y_fft, lambd=self.sparsity_threshold)
            y_fft = torch.view_as_complex(y_fft)

            outputs.append(y_fft)

        y_fft_concat = torch.cat(outputs, dim=-1)
        y = ifft(y_fft_concat)
        y = self.linear(y.real)

        return y



class BPDRLayer(nn.Module):
    def __init__(self, emb_dim, edge_attr_size, edge_time_emb_size, reduce, N=3):
        super(BPDRLayer, self).__init__()
        self.emb_dim = emb_dim
        self.edge_attr_size = edge_attr_size
        self.edge_time_emb_size = edge_time_emb_size
        self.reduce = reduce
        self.N = N
        self.msg_function = FreMLP(self.emb_dim + self.edge_attr_size + self.edge_time_emb_size, self.emb_dim, N=N)
        self.combine_linear = nn.Linear(2 * self.emb_dim, self.emb_dim, bias=True)
        self.linear = nn.Linear(self.emb_dim, self.emb_dim, bias=True)
        self.layer_norm = nn.LayerNorm(self.emb_dim, elementwise_affine=True, eps=1e-5)

    def forward(self, hidden, edge_index, edge_attr, edge_time_emb, boundary_condition):
        num_nodes = hidden.size(0)
        assert hidden.shape == boundary_condition.shape

        hidden = hidden.float()  
        edge_attr = edge_attr.float()  
        edge_time_emb = edge_time_emb.float()

        # Process msg_function
        msg_input = torch.cat((hidden[edge_index[0]], edge_attr, edge_time_emb), dim=1)
        msg = self.msg_function(msg_input)

        if not torch.is_complex(msg):
            msg = torch.view_as_complex(torch.stack((msg, torch.zeros_like(msg)), dim=-1))

        msg_real = torch.view_as_real(msg).flatten(start_dim=-2)
        msg = self.combine_linear(msg_real)

        msg_aug = torch.cat([msg, boundary_condition], dim=0)
        self_loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        self_loop_index = self_loop_index.unsqueeze(0).repeat(2, 1)
        idx_aug = torch.cat([edge_index[1], self_loop_index[1]])

        out = scatter(msg_aug, idx_aug, dim=0, reduce=self.reduce, dim_size=hidden.size(0))
        out = self.linear(out)
        out = F.dropout(F.relu(self.layer_norm(out)), p=0.1, training=self.training)

        return out


class BPDRNet(nn.Module):
    def __init__(self, emb_dim, edge_attr_size, edge_time_emb_dim=128, num_layers=3, use_fourier_features=True, use_id_label=True, device=None):
        super(BPDRNet, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.edge_attr_size = edge_attr_size
        self.num_layers = num_layers
        self.use_fourier_features = use_fourier_features
        self.use_id_label = use_id_label
        print("bool using_label_diffusion:", self.use_id_label)
        self.nbf_layers = nn.ModuleList()

        if edge_time_emb_dim > 0:
            self.edge_time_emb_dim = edge_time_emb_dim
            self.time_encoder = TimeEncoder(dim=self.edge_time_emb_dim)
        else:
            self.edge_time_emb_dim = 0
            self.time_encoder = None

        self.indicator_embedding = nn.Parameter(torch.rand(1, emb_dim), requires_grad=True)

        for layer in range(num_layers):
            self.nbf_layers.append(BPDRLayer(self.emb_dim, self.edge_attr_size, self.edge_time_emb_dim, reduce="sum"))

        self.mlp = MLP(emb_dim)

    def forward(self, batch):
        batch_sources_idx = batch.source + batch.ptr[:-1]
        batch_bc = torch.zeros(batch.num_nodes, self.emb_dim, device=self.device)

        if self.use_id_label:
            batch_bc[batch_sources_idx] = self.indicator_embedding

        if batch.edge_index.nelement() == 0 and batch.edge_attr.nelement() == 0:
            return batch_bc
        if self.edge_time_emb_dim > 0:
            edge_time_embeddings = self.time_encoder(batch.edge_time.to(self.device))
        else:
            edge_time_embeddings = None

        for layer in range(self.num_layers):
            if layer == 0:
                h = self.nbf_layers[layer](batch_bc, batch.edge_index.to(self.device), batch.edge_attr.to(self.device),
                                           edge_time_embeddings, batch_bc)
            else:
                h = self.nbf_layers[layer](h, batch.edge_index.to(self.device), batch.edge_attr.to(self.device),
                                           edge_time_embeddings, batch_bc)
        return h

    def predict_proba(self, edge_repr):
        edge_repr = edge_repr.float()
        previous, prob = self.mlp(edge_repr)
        return previous.squeeze(), prob.squeeze()
