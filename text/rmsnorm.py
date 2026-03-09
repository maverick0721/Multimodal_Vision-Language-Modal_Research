import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self,dim):

        super().__init__()

        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self,x):

        norm = x.pow(2).mean(-1,keepdim=True)

        return x * torch.rsqrt(norm+1e-6) * self.scale