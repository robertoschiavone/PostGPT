import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.linear1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear2 = nn.Linear(4 * n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x
