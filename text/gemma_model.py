import torch
import torch.nn as nn

from .decoder_layer import DecoderLayer


class GemmaModel(nn.Module):

    def __init__(self, vocab, dim=768, depth=12):

        super().__init__()

        self.vocab = vocab
        self.dim = dim

        # token embedding
        self.embed = nn.Embedding(vocab, dim)

        # transformer layers
        self.layers = nn.ModuleList(
            [DecoderLayer(dim) for _ in range(depth)]
        )

        # final norm
        self.norm = nn.LayerNorm(dim)

        # LM head
        self.head = nn.Linear(dim, vocab, bias=False)

        # weight tying (GPT style)
        self.head.weight = self.embed.weight


    def forward(self, tokens, vision):

        # tokens shape: [B, T]
        x = self.embed(tokens)

        # transformer blocks
        for layer in self.layers:
            x = layer(x, vision)

        hidden_states = self.norm(x)

        logits = self.head(hidden_states)

        return hidden_states, logits