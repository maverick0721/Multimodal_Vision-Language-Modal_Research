import torch.nn.functional as F

class Trainer:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def step(self, images, tokens, labels):

        outputs = self.model(images, tokens)
        logits = outputs[0]

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()