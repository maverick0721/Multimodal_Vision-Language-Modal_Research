import torch

def siglip_loss(image_emb,text_emb):

    image_emb = torch.nn.functional.normalize(
        image_emb,
        dim=-1
    )

    text_emb = torch.nn.functional.normalize(
        text_emb,
        dim=-1
    )

    logits = image_emb @ text_emb.T

    labels = torch.eye(
        logits.size(0),
        device=logits.device
    )

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels
    )

    return loss