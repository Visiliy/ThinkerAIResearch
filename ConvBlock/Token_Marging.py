import torch.nn as nn
import torch.nn.functional as F


class TokenMerging(nn.Module):
    def __init__(self, dim, reduction_ratio=2):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim * reduction_ratio, dim)

    def forward(self, x):
        b, s, d = x.shape
        if s % self.reduction_ratio != 0:
            pad_len = self.reduction_ratio - (s % self.reduction_ratio)
            x = F.pad(x, (0, 0, 0, pad_len))
            s = s + pad_len

        x = x.view(b, s // self.reduction_ratio, self.reduction_ratio * d)
        x = self.linear(x)
        return self.norm(x)