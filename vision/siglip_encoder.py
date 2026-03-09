import torch.nn as nn
from .patch_embedding import PatchEmbedding
from .transformer_block import TransformerBlock

class SigLipEncoder(nn.Module):

    def __init__(self,dim=768,depth=12,heads=12):

        super().__init__()

        self.patch = PatchEmbedding()

        self.layers = nn.ModuleList(
            [TransformerBlock(dim,heads) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self,x):

        x = self.patch(x)

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)