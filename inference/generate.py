import torch
from .sampling import top_p


@torch.no_grad()
def generate(model,image,tokens,steps=50):

    for _ in range(steps):

        logits = model(image,tokens)

        next_token = top_p(
            logits[:,-1,:]
        )

        tokens = torch.cat(
            [tokens,next_token],
            dim=1
        )

    return tokens