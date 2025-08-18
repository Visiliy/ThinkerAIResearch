import torch
import torch.nn as nn
import torch.nn.functional as F


class NewTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout_attention = nn.Dropout(0.1)
        self.dropout_ffn = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        x_permuted = x.transpose(0, 1)

        attention_output, _ = self.attention(x_permuted, x_permuted, x_permuted, key_padding_mask=mask)
        attention_output = self.dropout_attention(attention_output)
        x_permuted = x_permuted + attention_output
        x = self.norm1(x_permuted.transpose(0, 1))


        ffn_output = self.ffn(x)
        ffn_output = self.dropout_ffn(ffn_output)
        x = x + ffn_output
        x = self.norm2(x)

        return x


class NewTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.position_emb = nn.Parameter(torch.zeros(1, 1024, embed_dim))
        self.blocks = nn.ModuleList([NewTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, mask=None):
        batch_size, seq_len = input_ids.size()
        x = self.token_emb(input_ids) + self.position_emb[:, :seq_len]

        for block in self.blocks:
            x = block(x, mask)

        x = self.final_norm(x)
        logits = self.output_proj(x)
        return logits
