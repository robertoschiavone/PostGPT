import torch
import torch.nn as nn

from Head import Head


# multiple heads of self-attention in parallel
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embed, head_size, block_size):
        super().__init__()
        heads = [Head(n_embed, head_size, block_size) for _ in range(num_heads)]
        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)

        return x
