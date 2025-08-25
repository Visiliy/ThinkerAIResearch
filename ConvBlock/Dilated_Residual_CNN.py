import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedDilatedResidual(nn.Module):
    def __init__(self, channels, dilations=[1, 2, 4, 8], dropout=0.1, use_glu=True):
        super().__init__()
        self.dilations = dilations
        self.use_glu = use_glu

        self.conv_branches = nn.ModuleList()
        self.gate_branches = nn.ModuleList() if use_glu else None

        for dilation in dilations:
            padding = (3 - 1) * dilation // 2

            conv = nn.Conv1d(channels, channels, 3,
                             padding=padding, dilation=dilation)
            self.conv_branches.append(conv)

            if use_glu:
                gate = nn.Conv1d(channels, channels, 3,
                                 padding=padding, dilation=dilation)
                self.gate_branches.append(gate)

        self.branch_weights = nn.Parameter(torch.ones(len(dilations)))
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Fusion layer
        self.fusion = nn.Linear(channels * len(dilations), channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x_t = x.transpose(1, 2)

        branch_outputs = []
        for i, dilation in enumerate(self.dilations):
            if self.use_glu:
                # Gated Linear Unit
                conv_out = self.conv_branches[i](x_t)
                gate = torch.sigmoid(self.gate_branches[i](x_t))
                branch_out = conv_out * gate
            else:
                branch_out = self.conv_branches[i](x_t)

            branch_out = branch_out.transpose(1, 2)
            branch_outputs.append(branch_out)

        weights = F.softmax(self.branch_weights / self.temperature, dim=0)
        weighted_outputs = [out * weight for out, weight in zip(branch_outputs, weights)]

        combined = torch.cat(weighted_outputs, dim=-1)
        fused = self.fusion(combined)
        fused = self.activation(fused)
        fused = self.dropout(fused)


        output = fused + residual
        return self.norm(output)