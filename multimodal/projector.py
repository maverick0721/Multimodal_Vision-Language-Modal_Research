import torch.nn as nn

class ImageProjector(nn.Module):

    def __init__(self,vision_dim,text_dim):

        super().__init__()

        self.proj = nn.Linear(
            vision_dim,
            text_dim
        )

    def forward(self,x):

        return self.proj(x)