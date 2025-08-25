import torch.nn as nn


class LinearParameterizationKernel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, feature_dim=32):
        super().__init__()
        self.feature_dim = feature_dim

        self.U = nn.Linear(in_channels, feature_dim, bias=False)
        self.V = nn.Linear(feature_dim, out_channels, bias=False)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.U(x)
        x = self.activation(x)
        x = self.V(x)
        return x
