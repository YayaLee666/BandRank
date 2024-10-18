import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class TimeEncoder(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim, use_fourier_features=True):
        super(TimeEncoder, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.use_fourier_features = use_fourier_features
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output

