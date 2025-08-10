import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadInfluence(nn.Module):

    def __init__(self, mbed_dim, num_heads, device) -> None:
        super().__init__()
        self.mbed_dim = mbed_dim
        self.num_heads = num_heads
        self.device = device
        self.head_dim = mbed_dim // num_heads

        self.qwery1 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)
        self.qwery2 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)

        self.key1 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)
        self.key2 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)

        self.value1 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)
        self.value2 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)
        self.value3 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)

    def split_heads(self, X, num_heads, head_dim):
        if X.dim() == 2:
            X = X.unsqueeze(0)
        batch_size, seq_len, embed_dim = X.size()
        X = X.view(batch_size, seq_len, num_heads, head_dim)
        return X.transpose(1, 2)

    def combine_heads(self, X):
        batch_size, num_heads, seq_len, head_dim = X.size()
        X = X.transpose(1, 2).contiguous()
        return X.view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, X, Y, Z):
        sh = X.shape
        if len(sh) == 3:
            X = X.squeeze(0)
            Y = Y.squeeze(0)
            Z = Z.squeeze(0)

        Q1 = self.split_heads(self.qwery1(X), head_dim=self.head_dim, num_heads=self.num_heads)
        Q2 = self.split_heads(self.qwery2(X), head_dim=self.head_dim, num_heads=self.num_heads)

        K1 = self.split_heads(self.key1(Y), head_dim=self.head_dim, num_heads=self.num_heads)
        K2 = self.split_heads(self.key2(Y), head_dim=self.head_dim, num_heads=self.num_heads)

        V1 = self.split_heads(self.value1(Z), head_dim=self.head_dim, num_heads=self.num_heads)
        V2 = self.split_heads(self.value2(Z), head_dim=self.head_dim, num_heads=self.num_heads)
        V3 = self.split_heads(self.value3(Z), head_dim=self.head_dim, num_heads=self.num_heads)

        result1 = torch.matmul(Q1.transpose(-2, -1), V1.transpose(-2, -1).T)
        result1 = torch.matmul(result1.transpose(-2, -1), K1.transpose(-2, -1))
        result1 = torch.matmul(result1.transpose(-2, -1), V2.tarnspose(-2, -1).T) / torch.sqrt(self.head_dim)
        result1 = F.softmax(result1)
        result1 = torch.matmul(result1, Q2.transpose(-2, -1))
        Q_new = self.combine_heads(result1)
        return Q_new


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
