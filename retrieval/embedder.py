import torch
import torch.nn as nn
from transformers import AutoTokenizer


class SimpleEmbedder(nn.Module):

    def __init__(self, vocab=50257, dim=768):

        super().__init__()

        self.emb = nn.Embedding(vocab, dim)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, texts):

        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=16, return_tensors="pt")
        ids = encoded["input_ids"].to(self.emb.weight.device)

        return self.emb(ids).mean(dim=1)