import torch.nn as nn

from FeedForward import FeedForward
from MultiHeadAttention import MultiHeadAttention


# transformer block: communication followed by computation
class Block(nn.Module):
    # n_embed: embedding dimension
    # n_head: the number of heads we'd like
    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, n_embed, head_size, block_size)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)

        return x
