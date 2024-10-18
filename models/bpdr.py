import torch
import torch.nn.functional as F
from torch_scatter import scatter
from models.time import TimeEncoder
from models.mlp import MLP
import torch.nn as nn
from torch.fft import fft, ifft


class FreMLP(nn.Module):
    def __init__(self, input_dim, output_dim, sparsity_threshold=0.01):
        super(FreMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity_threshold = sparsity_threshold
        self.scale = 0.02

        # Low frequency parameters
        self.r1_low = nn.Parameter(self.scale * torch.randn(input_dim, output_dim))
        self.i1_low = nn.Parameter(self.scale * torch.randn(input_dim, output_dim))
        self.rb1_low = nn.Parameter(self.scale * torch.randn(output_dim))
        self.ib1_low = nn.Parameter(self.scale * torch.randn(output_dim))

        # Mid frequency parameters
        self.r1_mid = nn.Parameter(self.scale * torch.randn(input_dim, output_dim))
        self.i1_mid = nn.Parameter(self.scale * torch.randn(input_dim, output_dim))
        self.rb1_mid = nn.Parameter(self.scale * torch.randn(output_dim))
        self.ib1_mid = nn.Parameter(self.scale * torch.randn(output_dim))

        # High frequency parameters
        self.r1_high = nn.Parameter(self.scale * torch.randn(input_dim, output_dim))
        self.i1_high = nn.Parameter(self.scale * torch.randn(input_dim, output_dim))
        self.rb1_high = nn.Parameter(self.scale * torch.randn(output_dim))
        self.ib1_high = nn.Parameter(self.scale * torch.randn(output_dim))

        self.linear = nn.Linear(output_dim * 3, output_dim)

    def create_band_pass_filter(self, x_fft):
        B, L = x_fft.shape

        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        min_energy = energy.min(dim=0, keepdim=True)[0]
        max_energy = energy.max(dim=0, keepdim=True)[0]
        threshold_param1 = (2 * min_energy + max_energy) / 3
        threshold_param2 = (min_energy + 2 * max_energy) / 3

        low_freq_mask = (energy <= threshold_param1).float()
        mid_freq_mask = ((energy > threshold_param1) & (energy <= threshold_param2)).float()
        high_freq_mask = (energy > threshold_param2).float()

        return low_freq_mask, mid_freq_mask, high_freq_mask

    def forward(self, x):
        if not torch.is_complex(x):
            x = torch.view_as_complex(torch.stack((x.float(), torch.zeros_like(x).float()), dim=-1))


        x_fft = fft(x)
        low_freq_mask, mid_freq_mask, high_freq_mask = self.create_band_pass_filter(x_fft)
        low_freq_mask = low_freq_mask.unsqueeze(-1).expand_as(x_fft)
        mid_freq_mask = mid_freq_mask.unsqueeze(-1).expand_as(x_fft)
        high_freq_mask = high_freq_mask.unsqueeze(-1).expand_as(x_fft)

        low_freq_fft = x_fft * low_freq_mask
        mid_freq_fft = x_fft * mid_freq_mask
        high_freq_fft = x_fft * high_freq_mask

        o_real_fft_low = torch.einsum('...n,nd->...d', low_freq_fft.real, self.r1_low) - \
                         torch.einsum('...n,nd->...d', low_freq_fft.imag, self.i1_low) + self.rb1_low
        o_imag_fft_low = torch.einsum('...n,nd->...d', low_freq_fft.imag, self.r1_low) + \
                         torch.einsum('...n,nd->...d', low_freq_fft.real, self.i1_low) + self.ib1_low
        y_fft_low = torch.stack([F.relu(o_real_fft_low), F.relu(o_imag_fft_low)], dim=-1)
        y_fft_low = F.softshrink(y_fft_low, lambd=self.sparsity_threshold)
        y_fft_low = torch.view_as_complex(y_fft_low)

        o_real_fft_mid = torch.einsum('...n,nd->...d', mid_freq_fft.real, self.r1_mid) - \
                         torch.einsum('...n,nd->...d', mid_freq_fft.imag, self.i1_mid) + self.rb1_mid
        o_imag_fft_mid = torch.einsum('...n,nd->...d', mid_freq_fft.imag, self.r1_mid) + \
                         torch.einsum('...n,nd->...d', mid_freq_fft.real, self.i1_mid) + self.ib1_mid
        y_fft_mid = torch.stack([F.relu(o_real_fft_mid), F.relu(o_imag_fft_mid)], dim=-1)
        y_fft_mid = F.softshrink(y_fft_mid, lambd=self.sparsity_threshold)
        y_fft_mid = torch.view_as_complex(y_fft_mid)

        o_real_fft_high = torch.einsum('...n,nd->...d', high_freq_fft.real, self.r1_high) - \
                          torch.einsum('...n,nd->...d', high_freq_fft.imag, self.i1_high) + self.rb1_high
        o_imag_fft_high = torch.einsum('...n,nd->...d', high_freq_fft.imag, self.r1_high) + \
                          torch.einsum('...n,nd->...d', high_freq_fft.real, self.i1_high) + self.ib1_high
        y_fft_high = torch.stack([F.relu(o_real_fft_high), F.relu(o_imag_fft_high)], dim=-1)
        y_fft_high = F.softshrink(y_fft_high, lambd=self.sparsity_threshold)
        y_fft_high = torch.view_as_complex(y_fft_high)

        y_fft_concat = torch.cat([y_fft_low, y_fft_mid, y_fft_high], dim=-1)
        y = ifft(y_fft_concat)
        y = self.linear(y.real)

        return y


class BPDRLayer(nn.Module):
    def __init__(self, emb_dim, edge_attr_size, edge_time_emb_size, reduce):
        super(BPDRLayer, self).__init__()
        self.emb_dim = emb_dim
        self.edge_attr_size = edge_attr_size
        self.edge_time_emb_size = edge_time_emb_size
        self.reduce = reduce
        self.msg_function = FreMLP(self.emb_dim + self.edge_attr_size + self.edge_time_emb_size, self.emb_dim)
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
