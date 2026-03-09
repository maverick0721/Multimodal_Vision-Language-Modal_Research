import torch
import torch.nn as nn

class VisionAttention(nn.Module):

    def __init__(self,dim,heads):

        super().__init__()

        self.heads = heads
        self.scale = (dim//heads)**-0.5

        self.qkv = nn.Linear(dim,dim*3)
        self.proj = nn.Linear(dim,dim)

    def forward(self,x):

        B,N,D = x.shape

        qkv = self.qkv(x).chunk(3,-1)

        q,k,v = [
            t.view(B,N,self.heads,D//self.heads).transpose(1,2)
            for t in qkv
        ]

        attn = (q @ k.transpose(-2,-1))*self.scale
        attn = attn.softmax(-1)

        out = attn @ v

        out = out.transpose(1,2).reshape(B,N,D)

        return self.proj(out)