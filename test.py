import torch
import torch.nn as nn
from random import random
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
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_head
        self.num_head = num_head
        self.max_heads = 12 

        self.sigmoid_param1 = nn.Parameter(torch.rand(1))
        self.sigmoid_param2 = nn.Parameter(torch.rand(1))
        self.m1 = nn.Parameter(torch.randn(self.max_heads, embed_dim))
        self.m2 = nn.Linear(embed_dim, 1) 
        self.mix_layer = nn.Linear(embed_dim, embed_dim)
        self.matrix = nn.Parameter(torch.randn(seq_size, 1))

        self.matrix = nn.Parameter(torch.randn((seq_size, 1)))
        self.attention = MultiHeadWordAttention(embed_dim=embed_dim, num_head=num_head)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim * 4),
            nn.GELU(),
            nn.Linear(in_features=self.embed_dim * 4, out_features=self.embed_dim)
        )

        self.norm1 = nn.LayerNorm(self.embed_dim, self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim, self.embed_dim)

        self.attention_dropout = nn.Dropout(0.1)
        self.ffn_dropout = nn.Dropout(0.1)

        self.thought_scorer = nn.Linear(self.head_dim, 1)
        self.linear_out = nn.Linear(self.embed_dim, self.embed_dim)


    def forward(self, word, text):
        batch_size = word.size(0)

        h = torch.round(12 * torch.sigmoid(self.sigmoid_param1)).clamp(1, 12).long()
        k = torch.round(12 * torch.sigmoid(self.sigmoid_param2)).clamp(1, h.item()).long()

        text_vector = torch.matmul(text.transpose(-2, -1), self.matrix).transpose(-2, -1)

        doubled_tensor = word.repeat(1, h.item(), 1) 
        doubled_tensor = doubled_tensor * self.m1[:h.item()]

        k_scores = self.m2(doubled_tensor)
        k_probs = torch.sigmoid(k_scores)
        _, k_indices = torch.topk(k_probs, k=k.item(), dim=1)
        K_heads = doubled_tensor.gather(1, k_indices.expand(-1, -1, self.embed_dim))

        L = torch.mean(self.mix_layer(K_heads), dim=1, keepdim=True)


        attention = self.attention(L, text_vector)
        attention = self.attention_dropout(attention)

        norm1 = self.norm1(attention + L + text_vector)

        mlp = self.mlp(norm1)
        mlp = self.ffn_dropout(mlp)

        norm2 = self.norm2(mlp + norm1)

        return self.linear_out(norm2)


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
