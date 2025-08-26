import torch.nn as nn
from ConvBlock import DynamicPooling, LinearPerformerAttention, LinearParameterizationKernel, FastKernelCompression, LinearBlockSparseAttention, OptimizedDilatedResidual, TokenMerging, LinearLocalAttention, LinearDynamicInceptionBlock


class ConvBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        
        self.dynamic_pooling = DynamicPooling(output_size=None, mode='adaptive')
        self.performer_attention = LinearPerformerAttention(dim, heads, feature_dim=256, dropout=dropout)
        self.parameterization_kernel = LinearParameterizationKernel(dim, dim, feature_dim=32)
        self.kernel_compression = FastKernelCompression(dim, reduction_ratio=4)
        self.block_sparse_attention = LinearBlockSparseAttention(dim, block_size=32, heads=heads, feature_dim=128, dropout=dropout)
        self.dilated_residual_cnn = OptimizedDilatedResidual(dim, dilations=[1, 2, 4, 8], dropout=dropout, use_glu=True)
        self.token_merging = TokenMerging(dim, reduction_ratio=2)
        self.local_attention = LinearLocalAttention(dim, window_size=7, heads=heads, feature_dim=64, dropout=dropout)
        self.dynamic_inception_block = LinearDynamicInceptionBlock(dim, kernel_sizes=[1, 3, 5], feature_dims=[32, 64, 32], dropout=dropout)

        self.inception_proj = nn.Linear((dim // 3) * 4, dim)

        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = self.dynamic_inception_block(x)
        x = self.inception_proj(x)
        x = self.local_attention(x)
        x = self.token_merging(x)
        x = self.dilated_residual_cnn(x)
        x = self.block_sparse_attention(x)
        x = self.kernel_compression(x)
        x = self.parameterization_kernel(x)
        x = self.performer_attention(x)
        x = self.dynamic_pooling(x)
        
        return self.final_norm(x)