import torch
import torch.nn as nn


class LinearDynamicInceptionBlock(nn.Module):
    def __init__(self, channels, kernel_sizes=[1, 3, 5], feature_dims=[32, 64, 32], dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList()

        for ks, fd in zip(kernel_sizes, feature_dims):
            branch = nn.ModuleList([
                nn.Linear(channels, fd),
                nn.GELU(),
                nn.Conv1d(fd, fd, ks, padding=ks // 2, groups=fd),
                nn.GELU(),
                nn.Linear(fd, channels // len(kernel_sizes)),
                nn.Dropout(dropout)
            ])
            self.branches.append(branch)

        self.avg_pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // len(kernel_sizes)),
            nn.GELU()
        )

        total_output_dim = (channels // len(kernel_sizes)) * (len(kernel_sizes) + 1)  # +1 для avg_pool_branch
        self.norm = nn.LayerNorm(total_output_dim)

    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            linear1_out = branch[0](x)
            gelu1_out = branch[1](linear1_out)
            
            conv_input = gelu1_out.transpose(1, 2)
            conv_out = branch[2](conv_input)
            gelu2_out = branch[3](conv_out)
            
            linear2_input = gelu2_out.transpose(1, 2)
            linear2_out = branch[4](linear2_input)
            dropout_out = branch[5](linear2_out)
            
            branch_outputs.append(dropout_out)

        x_t = x.transpose(1, 2)
        avg_pooled = self.avg_pool_branch[0](x_t)
        avg_flattened = self.avg_pool_branch[1](avg_pooled)
        avg_linear = self.avg_pool_branch[2](avg_flattened)
        avg_out = self.avg_pool_branch[3](avg_linear)
        avg_out = avg_out.unsqueeze(1).expand(-1, x.size(1), -1)
        
        branch_outputs.append(avg_out)

        out = torch.cat(branch_outputs, dim=-1)
        return self.norm(out)
