import torch

def top_p(logits,p=0.9,temp=1.0):

    logits = logits / temp

    probs = torch.softmax(logits,-1)

    sorted_probs,sorted_idx = torch.sort(
        probs,
        descending=True
    )

    cumulative = torch.cumsum(sorted_probs,-1)

    mask = cumulative - sorted_probs > p

    sorted_probs[mask] = 0

    sorted_probs /= sorted_probs.sum()

    token = torch.multinomial(sorted_probs,1)

    return sorted_idx[token]