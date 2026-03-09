import torch

def pad_tokens(tokens,max_len):

    tokens = tokens[:max_len]

    pad = [0]*(max_len-len(tokens))

    return torch.tensor(tokens+pad)