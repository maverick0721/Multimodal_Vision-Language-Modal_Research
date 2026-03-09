import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):

    def __init__(self,img=224,patch=16,dim=768):

        super().__init__()

        self.proj = nn.Conv2d(3,dim,patch,patch)

        n = (img//patch)**2

        self.cls = nn.Parameter(torch.randn(1,1,dim))
        self.pos = nn.Parameter(torch.randn(1,n+1,dim))

    def forward(self,x):

        B = x.size(0)

        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)

        cls = self.cls.expand(B,-1,-1)

        x = torch.cat([cls,x],1)

        return x + self.pos