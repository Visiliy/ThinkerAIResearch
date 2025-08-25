import torch.nn as nn


class FastKernelCompression(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.compression = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, s, c = x.size()
        se = self.global_pool(x.transpose(1, 2)).view(b, c)
        se = self.compression(se).view(b, 1, c)
        return x * se