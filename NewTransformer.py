import torch
import torch.nn as nn
from torch.nn import functional as F


class NewTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # Multi-Head Attention
        x_attn = x.transpose(0, 1)
        attn_out, _ = self.attention(x_attn, x_attn, x_attn, key_padding_mask=mask)
        x = x + self.dropout(attn_out.transpose(0, 1))
        x = self.norm1(x)

        # FFN
        x = x + self.dropout(self.ffn(x))
        x = self.norm2(x)
        return x


class QRTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, embed_dim))
        self.blocks = nn.ModuleList([NewTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        self.qr_proj = nn.Linear(embed_dim, embed_dim)
        self.R = nn.Parameter(torch.triu(torch.randn(embed_dim, embed_dim)))

    def forward(self, input_ids, mask=None):
        batch_size, seq_len = input_ids.shape
        x = self.token_emb(input_ids) + self.pos_emb[:, :seq_len]

        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)

        x_proj = self.qr_proj(x)
        Q = torch.linalg.qr(x_proj, mode="reduced").Q
        logits = torch.matmul(Q, torch.triu(self.R))

        logits = F.linear(logits, self.token_emb.weight)
        return logits


if __name__ == "__main__":
    model = QRTransformer(1000, 128, 8, 12)
    input_ids = torch.randint(0, 1000, (1, 1024))
    logits = model(input_ids)
    print(logits.shape)