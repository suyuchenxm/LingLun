import torch
import torch.nn as nn
import math


class ScaledDotProductAttentionWithRelative(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttentionWithRelative, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, relative_positions, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        rel_scores = torch.einsum("bhld,bhmd->bhlm", Q, relative_positions)
        scores = scores + rel_scores

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)

        return output, attention
