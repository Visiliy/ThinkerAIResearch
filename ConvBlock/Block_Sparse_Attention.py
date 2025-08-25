import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBlockSparseAttention(nn.Module):
    def __init__(self, dim, block_size=32, heads=8, feature_dim=128, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.block_size = block_size
        self.feature_dim = feature_dim

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        self.proj_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(dim // heads, feature_dim))
            for _ in range(heads)
        ])
        for param in self.proj_matrices:
            nn.init.orthogonal_(param)

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        output = torch.zeros_like(q)

        for i in range(0, n, self.block_size):
            end_idx = min(i + self.block_size, n)
            block_size = end_idx - i

            q_block = q[:, :, i:end_idx, :]
            k_block = k[:, :, i:end_idx, :]
            v_block = v[:, :, i:end_idx, :]

            for head_idx in range(h):
                proj = self.proj_matrices[head_idx]
                q_proj = torch.einsum('bnd,df->bnf', q_block[:, head_idx], proj)
                k_proj = torch.einsum('bnd,df->bnf', k_block[:, head_idx], proj)

                q_proj = F.elu(q_proj) + 1
                k_proj = F.elu(k_proj) + 1

                k_v = torch.einsum('bnf,bnd->bfd', k_proj, v_block[:, head_idx])
                attn_out = torch.einsum('bnf,bfd->bnd', q_proj, k_v)

                z = 1.0 / (torch.einsum('bnf,bf->bn', q_proj, k_proj.sum(1)) + 1e-8)
                attn_out = attn_out * z.unsqueeze(-1)

                output[:, head_idx, i:end_idx, :] = attn_out

        output = output.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(output)

