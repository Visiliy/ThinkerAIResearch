import torch
import torch.nn as nn


class LinearDynamicInceptionBlock(nn.Module):
    def __init__(self, channels, kernel_sizes=[1, 3, 5], feature_dims=[32, 64, 32], dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList()

        for ks, fd in zip(kernel_sizes, feature_dims):
            branch = nn.Sequential(
                nn.Linear(channels, fd),
                nn.GELU(),
                nn.Conv1d(fd, fd, ks, padding=ks // 2, groups=fd),
                nn.GELU(),
                nn.Linear(fd, channels // len(kernel_sizes)),
                nn.Dropout(dropout)
            )
            self.branches.append(branch)

        self.avg_pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(channels, channels // len(kernel_sizes)),
            nn.GELU()
        )

        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)

        avg_out = self.avg_pool_branch(x.transpose(1, 2)).transpose(1, 2)
        branch_outputs.append(avg_out)

        out = torch.cat(branch_outputs, dim=-1)
        return self.norm(out)
