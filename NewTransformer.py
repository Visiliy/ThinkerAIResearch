import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import TextDataset
from get_data import get_data


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

    def forward(self, x, pad_mask=None, future_mask=None):
        x_attn = x.transpose(0, 1)
        attn_out, _ = self.attention(
            x_attn, x_attn, x_attn,
            key_padding_mask=pad_mask,
            attn_mask=future_mask
        )
        x = x + self.dropout(attn_out.transpose(0, 1))
        x = self.norm1(x)

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

    def forward(self, input_ids, pad_mask=None, future_mask=None):
        batch_size, seq_len = input_ids.shape
        x = self.token_emb(input_ids) + self.pos_emb[:, :seq_len]

        for block in self.blocks:
            x = block(x, pad_mask=pad_mask, future_mask=future_mask)
        x = self.norm(x)

        x_proj = self.qr_proj(x)
        Q = torch.linalg.qr(x_proj, mode="reduced").Q
        logits = torch.matmul(Q, torch.triu(self.R))

        logits = F.linear(logits, self.token_emb.weight)
        return logits


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    path = "/home/user/.cache/kagglehub/datasets/kkhubiev/russian-financial-news/versions/3/RussianFinancialNews/news_descriptions/news_description_LLama3_8b.json"

    dataset = TextDataset(get_data(path), tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QRTransformer(
        vocab_size=vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=6
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            pad_mask = (inputs == tokenizer.pad_token_id)
            seq_len = inputs.size(1)
            future_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)

            optimizer.zero_grad()
            logits = model(inputs, mask=pad_mask, future_mask=future_mask)

            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)

            loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=tokenizer.pad_token_id
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(dataloader)}")