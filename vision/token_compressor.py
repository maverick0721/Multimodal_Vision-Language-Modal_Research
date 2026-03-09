import torch
import torch.nn as nn

class TokenCompressor(nn.Module):

    def __init__(self,compressed=32,dim=768):

        super().__init__()

        self.query = nn.Parameter(torch.randn(compressed,dim))

        self.attn = nn.MultiheadAttention(dim,8,batch_first=True)

    def forward(self,x):

        B = x.size(0)

        q = self.query.unsqueeze(0).repeat(B,1,1)

        out,_ = self.attn(q,x,x)

        return out