import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .attention import VisionAttention


class TransformerBlock(nn.Module):

    def __init__(self, dim, heads):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = VisionAttention(dim, heads)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def _attn_block(self, x):
        return self.attn(self.norm1(x))

    def _ffn_block(self, x):
        return self.ffn(self.norm2(x))

    def forward(self, x):

        # ---- attention ----
        if self.training and torch.is_grad_enabled():

            attn_out = checkpoint.checkpoint(
                self._attn_block,
                x,
                use_reentrant=False
            )

        else:
            attn_out = self._attn_block(x)

        x = x + attn_out

        # ---- feed forward ----
        if self.training and torch.is_grad_enabled():

            ffn_out = checkpoint.checkpoint(
                self._ffn_block,
                x,
                use_reentrant=False
            )

        else:
            ffn_out = self._ffn_block(x)

        x = x + ffn_out

        return x