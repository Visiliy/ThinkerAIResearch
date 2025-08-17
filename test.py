import torch
import torch.nn as nn
import torch.nn.functional as F


class Influence(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.linear3 = nn.Linear(embed_dim, embed_dim)

        self.out_linear1 = nn.Linear(embed_dim, embed_dim)
        self.out_linear2 = nn.Linear(embed_dim, embed_dim)
        self.out_linear3 = nn.Linear(embed_dim, embed_dim)

    def apply_permute(self, tensor_lists):
        new_tensor_lists = []
        for tensor, one, two, three in tensor_lists:
            tensor = tensor.permute(one, two, three)
            new_tensor_lists.append(tensor)
        return new_tensor_lists

    def forward(self, Q, K, V):
        Q, K, V = self.apply_permute([[Q, 2, 0, 1], [K, 2, 0, 1], [V, 2, 0, 1]])

        Q = self.linear1(Q)
        K = self.linear2(K)
        V = self.linear3(V)

        Q, K, V = self.apply_permute([[Q, 1, 2, 0], [K, 1, 2, 0], [V, 1, 2, 0]])

        return self.out_linear1(Q), self.out_linear2(K), self.out_linear3(V)
