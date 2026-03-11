import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

from .rmsnorm import RMSNorm
from .gqa_attention import GQAAttention
from .cross_attention import CrossAttention
from .moe_ffn import MoEFFN


class DecoderLayer(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)

        self.attn = GQAAttention(dim)
        self.cross = CrossAttention(dim)
        self.ffn = MoEFFN(dim)

    def _tensor_only(self, out):
        # Some modules return (tensor, aux)
        if isinstance(out, tuple):
            return out[0]
        return out

    def forward(self, x, vision):

        # ---- self attention ----
        def self_attn_block(y):
            out = self.attn(self.norm1(y))
            return self._tensor_only(out)

        if self.training and torch.is_grad_enabled():
            attn_out = checkpoint.checkpoint(
                self_attn_block,
                x,
                use_reentrant=False
            )
        else:
            attn_out = self_attn_block(x)

        residual = x
        x = residual + attn_out


        # ---- cross attention ----
        def cross_attn_block(y):
            out = self.cross(self.norm2(y), vision)
            return self._tensor_only(out)

        if self.training and torch.is_grad_enabled():
            cross_out = checkpoint.checkpoint(
                cross_attn_block,
                x,
                use_reentrant=False
            )
        else:
            cross_out = cross_attn_block(x)

        residual = x
        x = residual + cross_out


        # ---- feedforward / MoE ----
        def ffn_block(y):
            out = self.ffn(self.norm3(y))
            return self._tensor_only(out)

        if self.training and torch.is_grad_enabled():
            ffn_out = checkpoint.checkpoint(
                ffn_block,
                x,
                use_reentrant=False
            )
        else:
            ffn_out = ffn_block(x)

        residual = x
        x = residual + ffn_out

        return x