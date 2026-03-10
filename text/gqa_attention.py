import torch
import torch.nn as nn

class GQAAttention(nn.Module):

    def __init__(self,dim):

        super().__init__()

        self.q = nn.Linear(dim,dim)
        self.k = nn.Linear(dim,dim//4)
        self.v = nn.Linear(dim,dim//4)

        self.out = nn.Linear(dim,dim)

    def forward(self,x):

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = torch.softmax(
            q @ k.transpose(-1,-2)/q.size(-1)**0.5,
            dim=-1
        )

        return self.out(attn @ v)