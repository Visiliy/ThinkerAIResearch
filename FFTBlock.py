import torch
import torch.nn as nn
import torch.fft as fft
import math


class SPECTREFFTBlock(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff=None,
                 num_heads=8,
                 dropout=0.1,
                 max_seq_len=8192,
                 use_learnable_freq_weights=True):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.use_learnable_freq_weights = use_learnable_freq_weights

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        if use_learnable_freq_weights:
            self.freq_weights = nn.Parameter(torch.ones(1, num_heads, 1, self.head_dim))

        self.pos_embeddings = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.scale = math.sqrt(self.head_dim)

    def fft_attention(self, x):

        batch_size, seq_len, d_model = x.shape

        Q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        Q_fft = fft.fft(Q, dim=-2)
        K_fft = fft.fft(K, dim=-2)
        V_fft = fft.fft(V, dim=-2)

        if self.use_learnable_freq_weights:
            Q_fft = Q_fft * self.freq_weights
            K_fft = K_fft * self.freq_weights

        attention_weights = torch.conj(Q_fft) * K_fft
        attention_weights = attention_weights / self.scale

        attention_scores = fft.ifft(attention_weights, dim=-2).real
        attention_scores = torch.softmax(attention_scores, dim=-2)

        output_fft = attention_scores.unsqueeze(-1) * V_fft
        output = fft.ifft(output_fft, dim=-2).real

        output = output.transpose(1, 2)
        output = output.contiguous().view(batch_size, seq_len, d_model)

        return output

    def forward(self, x):

        seq_len = x.size(1)
        pos_emb = self.pos_embeddings[:, :seq_len, :]
        x = x + pos_emb

        attn_output = self.fft_attention_v2(x)
        attn_output = self.output_proj(attn_output)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x
