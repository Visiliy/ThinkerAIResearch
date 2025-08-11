import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadInfluence(nn.Module):

    def __init__(self, mbed_dim, num_heads) -> None:
        super().__init__()
        self.mbed_dim = mbed_dim
        self.num_heads = num_heads
        self.head_dim = mbed_dim // num_heads

        self.qwery1 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)
        self.qwery2 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)

        self.key1 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)
        self.key2 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)

        self.value1 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)
        self.value2 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)
        self.value3 = nn.Linear(in_features=mbed_dim, out_features=mbed_dim)

        self.linear_out1 = nn.Linear(mbed_dim, mbed_dim)
        self.linear_out2 = nn.Linear(mbed_dim, mbed_dim)
        self.linear_out3 = nn.Linear(mbed_dim, mbed_dim)

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

        result = torch.matmul(Q1, V1.transpose(-2, -1))
        result = torch.matmul(result, K1)
        result = torch.matmul(result, V2.transpose(-2, -1)) / torch.sqrt(self.head_dim)
        result = F.softmax(result)

        result1 = torch.matmul(result, Q2)
        Q_new = self.combine_heads(result1)

        result2 = torch.matmul(result, K2)
        K_new = self.combine_heads(result2)

        result3 = torch.matmul(result, V3)
        V_new = self.combine_heads(result3)

        return self.linear_out1(Q_new), self.linear_out2(K_new), self.linear_out3(V_new)


class DecoderBlock(nn.Module):

    def __init__(self, device, embed_dim) -> None:
        super().__init__()
        self.device = device
        self.attention1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=12, device=device,
                                               batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=12, device=device,
                                               batch_first=True)
        self.attention3 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=12, device=device,
                                               batch_first=True)

        self.influence = MultiHeadInfluence(mbed_dim=embed_dim, num_heads=12)

        self.normalization1_1 = nn.LayerNorm(embed_dim)
        self.normalization1_2 = nn.LayerNorm(embed_dim)
        self.normalization1_3 = nn.LayerNorm(embed_dim)

        self.normalization2_1 = nn.LayerNorm(embed_dim)
        self.normalization2_2 = nn.LayerNorm(embed_dim)
        self.normalization2_3 = nn.LayerNorm(embed_dim)

        self.normalization3_1 = nn.LayerNorm(embed_dim)
        self.normalization3_2 = nn.LayerNorm(embed_dim)
        self.normalization3_3 = nn.LayerNorm(embed_dim)

        self.mlp1 = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim * 4),
            nn.GELU(),
            nn.Linear(in_features=embed_dim * 4, out_features=embed_dim)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim * 4),
            nn.GELU(),
            nn.Linear(in_features=embed_dim * 4, out_features=embed_dim)
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim * 4),
            nn.GELU(),
            nn.Linear(in_features=embed_dim * 4, out_features=embed_dim)
        )

        self.attn_dropout1 = nn.Dropout(0.1)
        self.attn_dropout2 = nn.Dropout(0.1)
        self.attn_dropout3 = nn.Dropout(0.1)

        self.ffn_dropout1 = nn.Dropout(0.1)
        self.ffn_dropout2 = nn.Dropout(0.1)
        self.ffn_dropout3 = nn.Dropout(0.1)

        self.influence_dropout1 = nn.Dropout(0.1)
        self.influence_dropout2 = nn.Dropout(0.1)
        self.influence_dropout3 = nn.Dropout(0.1)

    def forward(self, X, Y, Z):
        if X.dim() == 3:
            X.squeeze(0)
            Y.squeeze(0)
            Z.squeeze(0)
        seq_len = X.shape[0]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device) * float('-inf'), diagonal=1)

        att1, _ = self.attention1(X, X, X)
        att1 = self.attn_dropout1(att1)
        norm1_1 = self.normalization1_1(att1 + X)

        att2, _ = self.attention2(Y, Y, Y)
        att2 = self.attn_dropout2(att2)
        norm1_2 = self.normalization1_2(att2 + Y, attn_mask=mask)

        att3, _ = self.attention3(Z, Z, Z)
        att3 = self.attn_dropout3(att3)
        norm1_3 = self.normalization1_3(att3 + Z, attn_mask=mask)

        Q, K, V = self.influence(norm1_1, norm1_2, norm1_3)

        Q = self.influence_dropout1(Q)
        K = self.influence_dropout2(K)
        V = self.influence_dropout3(V)

        norm2_1 = self.normalization2_1(Q + norm1_1)
        norm2_2 = self.normalization2_2(K + norm1_2)
        norm2_3 = self.normalization2_3(V + norm1_3)

        mlp1 = self.mlp1(norm2_1)
        mlp2 = self.mlp2(norm2_2)
        mlp3 = self.mlp3(norm2_3)

        mlp1 = self.ffn_dropout1(mlp1)
        mlp2 = self.ffn_dropout2(mlp2)
        mlp3 = self.ffn_dropout3(mlp3)

        return self.normalization3_1(mlp1 + norm2_1), self.normalization3_2(mlp2 + norm2_2), self.normalization3_3(mlp3 + norm2_3)


class ALT(nn.Module):

    def __init__(self, embed_dim, out_size, divie):
        super().__init__()

        self.influence = MultiHeadInfluence(embed_dim=embed_dim, num_heads=12)
        self.decoderes = nn.ModuleList([DecoderBlock(embed_dim=embed_dim, device=divie) for _ in range(10)])

        self.mlp1 = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim * 4),
            nn.GELU(),
            nn.Linear(in_features=embed_dim * 4, out_features=embed_dim)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim * 4),
            nn.GELU(),
            nn.Linear(in_features=embed_dim * 4, out_features=embed_dim)
        )

        self.normalization1_1 = nn.LayerNorm(embed_dim)
        self.normalization1_2 = nn.LayerNorm(embed_dim)

        self.normalization2_1 = nn.LayerNorm(embed_dim)
        self.normalization2_2 = nn.LayerNorm(embed_dim)

        self.influence_dropout1 = nn.Dropout(0.1)
        self.influence_dropout2 = nn.Dropout(0.1)

        self.ffn_dropout1 = nn.Dropout(0.1)
        self.ffn_dropout2 = nn.Dropout(0.1)

        self.token_predictor1 = nn.Linear(embed_dim, out_size)
        self.token_predictor2 = nn.Linear(embed_dim, out_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, X, Y, Z):
        for layer in self.decoderes:
            X, Y, Z = layer(X, Y, Z)
        _, inf1, inf2 = self.influence(X, Y, Z)

        inf1 = self.influence_dropout1(Y)
        inf2 = self.influence_dropout2(Z)

        norm1 = self.normalization1_1(inf1 + Y)
        norm2 = self.normalization1_2(inf2 + Z)

        mlp1 = self.ffn_dropout1(self.mlp1(norm1))
        mlp2 = self.ffn_dropout2(self.mlp2(norm2))

        norm2_1 = self.normalization2_1(mlp1 + norm1)
        norm2_2 = self.normalization2_2(mlp2 + norm2)

        token_predict1 = self.token_predictor1(norm2_1)
        token_predict2 = self.token_predictor2(norm2_2)
        return token_predict1, token_predict2
