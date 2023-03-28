import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.linear1 = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)

        return x
