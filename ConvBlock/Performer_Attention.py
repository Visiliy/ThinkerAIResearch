import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearPerformerAttention(nn.Module):
    def __init__(self, dim, heads=8, feature_dim=256, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.feature_dim = feature_dim
        self.head_dim = dim // heads

        self.proj_matrix = nn.Parameter(torch.randn(heads, self.head_dim, feature_dim))
        nn.init.orthogonal_(self.proj_matrix)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        q_proj = torch.einsum('bhnd,hdf->bhnf', q, self.proj_matrix)
        k_proj = torch.einsum('bhnd,hdf->bhnf', k, self.proj_matrix)

        q_proj = F.elu(q_proj) + 1
        k_proj = F.elu(k_proj) + 1

        k_v = torch.einsum('bhnf,bhnd->bhfd', k_proj, v)
        attention_out = torch.einsum('bhnf,bhfd->bhnd', q_proj, k_v)

        k_proj_sum = k_proj.sum(dim=2, keepdim=True)
        z = 1.0 / (torch.einsum('bhnf,bhf->bhn', q_proj, k_proj_sum.squeeze(2)) + 1e-8)
        attention_out = attention_out * z.unsqueeze(-1)

        attention_out = attention_out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(attention_out)
