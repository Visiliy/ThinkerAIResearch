import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLocalAttention(nn.Module):
    def __init__(self, dim, window_size=7, heads=8, feature_dim=64, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.heads = heads
        self.feature_dim = feature_dim

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        self.proj_matrix = nn.Parameter(torch.randn(dim // heads, feature_dim))
        nn.init.orthogonal_(self.proj_matrix)

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        output = torch.zeros_like(q)

        for center in range(n):
            start = max(0, center - self.window_size // 2)
            end = min(n, center + self.window_size // 2 + 1)
            window_size = end - start

            q_center = q[:, :, center:center + 1, :]
            k_window = k[:, :, start:end, :]
            v_window = v[:, :, start:end, :]

            q_proj = torch.einsum('bhnd,df->bhnf', q_center, self.proj_matrix)
            k_proj = torch.einsum('bhnd,df->bhnf', k_window, self.proj_matrix)

            q_proj = F.elu(q_proj) + 1
            k_proj = F.elu(k_proj) + 1

            k_v = torch.einsum('bhnf,bhnd->bhfd', k_proj, v_window)
            attn_out = torch.einsum('bhnf,bhfd->bhnd', q_proj, k_v)

            k_proj_sum = k_proj.sum(dim=2, keepdim=True)
            z = 1.0 / (torch.einsum('bhnf,bhf->bhn', q_proj, k_proj_sum.squeeze(2)) + 1e-8)
            attn_out = attn_out * z.unsqueeze(-1)

            output[:, :, center:center + 1, :] = attn_out

        output = output.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(output)
