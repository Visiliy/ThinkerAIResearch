import torch
import torch.nn as nn


class MultiHeadInfluence(nn.Module):

    def __init__(self, mbed_dim, num_heads, device, batch_first) -> None:
        super().__init__()
        self.mbed_dim = mbed_dim
        self.num_heads = num_heads
        self.device = device
        self.batch_first = batch_first
        self.head_dim = mbed_dim // num_heads

        self.qwery1 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)
        self.qwery2 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)

        self.key1 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)
        self.key2 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)

        self.value1 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)
        self.value2 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)
        self.value3 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)

        self.out_linear1 = nn.Linear(mbed_dim, mbed_dim)
        self.out_linear2 = nn.Linear(mbed_dim, mbed_dim)
        self.out_linear3 = nn.Linear(mbed_dim, mbed_dim)

    def forward(self, X, Y, Z):
        pass


class EncoderBlock(nn.Module):

    def __init__(self, device, embed_dim) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=12, device=device,
                                               batch_first=True)
        self.normalization = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim * 4),
            nn.GELU(),
            nn.Linear(in_features=embed_dim * 4, out_features=embed_dim)
        )

    def forward(self):
        pass


class DecoderBlock(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self):
        pass
