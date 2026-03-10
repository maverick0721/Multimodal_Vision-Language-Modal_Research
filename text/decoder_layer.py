import torch.nn as nn

from .gqa_attention import GQAAttention
from .cross_attention import CrossAttention
from .ffn import FFN
from .rmsnorm import RMSNorm


class DecoderLayer(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.attn = GQAAttention(dim)

        self.norm2 = RMSNorm(dim)
        self.cross = CrossAttention(dim)

        self.norm3 = RMSNorm(dim)
        self.ffn = FFN(dim)

    def forward(self, x, vision):

        x = x + self.attn(self.norm1(x))

        x = x + self.cross(self.norm2(x), vision)

        x = x + self.ffn(self.norm3(x))

        return x