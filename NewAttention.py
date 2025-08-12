import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenCompressor(nn.Module):
    def __init__(self, input_dim, min_tokens=4, init_compression=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.min_tokens = min_tokens

        self.compression_logit = nn.Parameter(
            torch.logit(torch.tensor(float(init_compression))))

        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)

        self.attention = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Softmax(dim=1)
        )

    def get_compression_ratio(self):
        return torch.sigmoid(self.compression_logit)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        compression = self.get_compression_ratio()
        target_len = max(self.min_tokens,
                         int(seq_len * compression.item()))

        if target_len >= seq_len:
            return x

        keys = self.key_proj(x)
        values = self.value_proj(x)

        attn_weights = self.attention(keys)

        grid = torch.linspace(0, 1, target_len, device=x.device)
        grid = grid.view(1, target_len, 1).expand(batch_size, -1, -1)

        pos = torch.linspace(0, 1, seq_len, device=x.device)
        pos = pos.view(1, 1, seq_len).expand(batch_size, target_len, -1)

        distances = torch.abs(grid - pos)
        kernel = torch.exp(-distances * 10)

        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-6)

        weighted_kernel = kernel * attn_weights.squeeze(-1).unsqueeze(1)
        norm = weighted_kernel.sum(dim=-1, keepdim=True)
        weighted_kernel = weighted_kernel / (norm + 1e-6)

        compressed = torch.bmm(weighted_kernel, values)

        return compressed


class SmartSequenceCompressor(nn.Module):
    def __init__(self, input_dim, num_layers=3, min_tokens=8):
        super().__init__()
        self.layers = nn.ModuleList([
            TokenCompressor(input_dim,
                            min_tokens=max(min_tokens, 2 ** (num_layers - i)))
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
