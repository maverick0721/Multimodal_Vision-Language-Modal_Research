import torch.nn as nn
from .rmsnorm import RMSNorm
from .gqa_attention import GQAAttention
from .moe_ffn import MoE

class DecoderLayer(nn.Module):

    def __init__(self,dim):

        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.attn = GQAAttention(dim)
        self.ffn = MoE(dim)

    def forward(self,x):

        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x