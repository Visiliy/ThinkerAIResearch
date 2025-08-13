import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadWordAttention(nn.Module):

    def __init__(self, num_head, embed_dim):
        super().__init__()
        self.head_dim = embed_dim // num_head
        self.num_head = num_head

        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.linear3 = nn.Linear(embed_dim, embed_dim)

        self.linear_out = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, X, num_heads, head_dim):
        if X.dim() == 3:
            X = X.squeeze(0)
        batch_size, embed_dim = X.size()
        X = X.view(batch_size, num_heads, head_dim)
        return X

    def combine_heads(self, X):
        batch_size, num_heads, head_dim = X.size()
        return X.view(batch_size, num_heads * head_dim)


    def forward(self, word, text_v):
        split_world1 = self.split_heads(self.linear1(word), head_dim=self.head_dim, num_heads=self.num_head)
        split_world3 = self.split_heads(self.linear3(word), head_dim=self.head_dim, num_heads=self.num_head)
        split_text_v = self.split_heads(self.linear2(text_v), head_dim=self.head_dim, num_heads=self.num_head)

        result = torch.matmul(split_world1, split_text_v.transpose(-2, -1))
        result = result / self.head_dim ** 0.5
        result = F.softmax(result)
        return self.linear_out(self.combine_heads(torch.matmul(result, split_world3)))


class ThinkingHeadsBlock(nn.Module):

    def __init__(self, seq_size, embed_dim, num_head):
        super().__init__()
        self.head_dim = embed_dim // num_head
        self.matrix = nn.Parameter(torch.randn((seq_size, 1)))
        self.attention = MultiHeadWordAttention(embed_dim=embed_dim, num_head=num_head)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim * 4),
            nn.GELU(),
            nn.Linear(in_features=embed_dim * 4, out_features=embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim, embed_dim)

        self.attention_dropout = nn.Dropout(0.1)
        self.ffn_dropout = nn.Dropout(0.1)

        self.linear = nn.Linear((embed_dim // num_head), embed_dim)

        self.thought_scorer = nn.Linear(self.head_dim, 1)
        self.linear_out = nn.Linear(self.head_dim, embed_dim)

    def split_heads(self, X, num_heads, head_dim):
        if X.dim() == 3:
            X = X.squeeze(0)
        batch_size, embed_dim = X.size()
        X = X.view(batch_size, num_heads, head_dim)
        return X

    def forward(self, word, text):
        text_vector = torch.matmul(text.transpose(-2, -1), self.matrix).transpose(-2, -1)

        attention = self.attention(word, text_vector)
        attention = self.attention_dropout(attention)

        norm1 = self.norm1(attention + word + text_vector)

        mlp = self.mlp(norm1)
        mlp = self.ffn_dropout(mlp)

        norm2 = self.norm2(mlp + norm1)

        heads = self.split_heads(norm2, num_heads=12, head_dim=(norm2.shape[-1] // 12))
        thought_scores = self.thought_scorer(heads).squeeze(-1)
        selected_head_idx = thought_scores.argmax(dim=-1, keepdim=True)
        selected_thought = heads.gather(1, selected_head_idx.unsqueeze(-1).expand(-1, -1, self.head_dim)).squeeze(1)

        return self.linear_out(selected_thought)


class Conscience(nn.Module):

    def __init__(self, seq_size, embed_dim, num_head, out_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, out_dim)
        self.blocks = nn.ModuleList([ThinkingHeadsBlock(seq_size=seq_size, embed_dim=embed_dim, num_head=num_head) for _ in range(4)])

    def forward(self, word, text):
        for layer in self.blocks:
            word = layer(word, text)
        word = word.squeeze(0)
        word = self.linear(word)
        return word.unsqueeze(0)
    

embedding = torch.randn((1, 132))
text = torch.randn((20, 132))
model = Conscience(20, 132, 12, 64)
word = model(embedding, text)
print(word.shape)
