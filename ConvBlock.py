import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPooling(nn.Module):
    def __init__(self, output_size=None, mode='adaptive'):
        super().__init__()
        self.output_size = output_size
        self.mode = mode

        if mode == 'adaptive':
            self.pool = nn.AdaptiveAvgPool1d(output_size)
        elif mode == 'learnable':
            self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x = x.transpose(1, 2)

        if self.mode == 'adaptive':
            pooled = self.pool(x)
        elif self.mode == 'learnable':
            avg_pool = F.adaptive_avg_pool1d(x, self.output_size)
            max_pool = F.adaptive_max_pool1d(x, self.output_size)
            pooled = self.alpha * avg_pool + (1 - self.alpha) * max_pool
        else:
            seq_len = x.size(2)
            if self.output_size is None:
                pool_size = max(8, seq_len // 4)
                pooled = F.adaptive_avg_pool1d(x, pool_size)
            else:
                pooled = F.adaptive_avg_pool1d(x, self.output_size)

        return pooled.transpose(1, 2)


class LinearPerformerAttention(nn.Module):
    def __init__(self, dim, heads=8, feature_dim=256, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.feature_dim = feature_dim
        self.head_dim = dim // heads

        self.proj_matrix = nn.Parameter(torch.randn(heads, self.head_dim, feature_dim))
        nn.init.orthogonal_(self.proj_matrix)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        q_proj = torch.einsum('bhnd,bhdf->bhnf', q, self.proj_matrix)
        k_proj = torch.einsum('bhnd,bhdf->bhnf', k, self.proj_matrix)

        q_proj = F.elu(q_proj) + 1
        k_proj = F.elu(k_proj) + 1

        k_v = torch.einsum('bhnf,bhnd->bhfd', k_proj, v)
        attention_out = torch.einsum('bhnf,bhfd->bhnd', q_proj, k_v)

        z = 1.0 / (torch.einsum('bhnf,bhfn->bhn', q_proj, k_proj.transpose(2, 3)) + 1e-8)
        attention_out = attention_out * z.unsqueeze(-1)

        attention_out = attention_out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(attention_out)


class LinearParameterizationKernel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, feature_dim=32):
        super().__init__()
        self.feature_dim = feature_dim

        self.U = nn.Linear(in_channels, feature_dim, bias=False)
        self.V = nn.Linear(feature_dim, out_channels, bias=False)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.U(x)
        x = self.activation(x)
        x = self.V(x)
        return x


class FastKernelCompression(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.compression = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, s, c = x.size()
        se = self.global_pool(x.transpose(1, 2)).view(b, c)
        se = self.compression(se).view(b, 1, c)
        return x * se


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


class OptimizedDilatedResidual(nn.Module):
    def __init__(self, channels, dilations=[1, 2, 4, 8], dropout=0.1, use_glu=True):
        super().__init__()
        self.dilations = dilations
        self.use_glu = use_glu

        self.conv_branches = nn.ModuleList()
        self.gate_branches = nn.ModuleList() if use_glu else None

        for dilation in dilations:
            padding = (3 - 1) * dilation // 2

            conv = nn.Conv1d(channels, channels, 3,
                             padding=padding, dilation=dilation)
            self.conv_branches.append(conv)

            if use_glu:
                gate = nn.Conv1d(channels, channels, 3,
                                 padding=padding, dilation=dilation)
                self.gate_branches.append(gate)

        self.branch_weights = nn.Parameter(torch.ones(len(dilations)))
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Fusion layer
        self.fusion = nn.Linear(channels * len(dilations), channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x_t = x.transpose(1, 2)

        branch_outputs = []
        for i, dilation in enumerate(self.dilations):
            if self.use_glu:
                conv_out = self.conv_branches[i](x_t)
                gate = torch.sigmoid(self.gate_branches[i](x_t))
                branch_out = conv_out * gate
            else:
                branch_out = self.conv_branches[i](x_t)

            branch_out = branch_out.transpose(1, 2)
            branch_outputs.append(branch_out)

        weights = F.softmax(self.branch_weights / self.temperature, dim=0)
        weighted_outputs = [out * weight for out, weight in zip(branch_outputs, weights)]

        combined = torch.cat(weighted_outputs, dim=-1)
        fused = self.fusion(combined)
        fused = self.activation(fused)
        fused = self.dropout(fused)

        output = fused + residual
        return self.norm(output)


class TokenMerging(nn.Module):
    def __init__(self, dim, reduction_ratio=2):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim * reduction_ratio, dim)

    def forward(self, x):
        b, s, d = x.shape
        if s % self.reduction_ratio != 0:
            pad_len = self.reduction_ratio - (s % self.reduction_ratio)
            x = F.pad(x, (0, 0, 0, pad_len))
            s = s + pad_len

        x = x.view(b, s // self.reduction_ratio, self.reduction_ratio * d)
        x = self.linear(x)
        return self.norm(x)


class LinearLocalAttention(nn.Module):
    def __init__(self, dim, window_size=7, heads=8, feature_dim=64, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.heads = heads
        self.feature_dim = feature_dim

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        self.proj_matrix = nn.Parameter(torch.randn(dim // heads, feature_dim))
        nn.init.orthogonal_(self.proj_matrix)

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        output = torch.zeros_like(q)

        for center in range(n):
            start = max(0, center - self.window_size // 2)
            end = min(n, center + self.window_size // 2 + 1)
            window_size = end - start

            q_center = q[:, :, center:center + 1, :]
            k_window = k[:, :, start:end, :]
            v_window = v[:, :, start:end, :]

            q_proj = torch.einsum('bhnd,df->bhnf', q_center, self.proj_matrix)
            k_proj = torch.einsum('bhnd,df->bhnf', k_window, self.proj_matrix)

            q_proj = F.elu(q_proj) + 1
            k_proj = F.elu(k_proj) + 1

            k_v = torch.einsum('bhnf,bhnd->bhfd', k_proj, v_window)
            attn_out = torch.einsum('bhnf,bhfd->bhnd', q_proj, k_v)

            z = 1.0 / (torch.einsum('bhnf,bhfn->bhn', q_proj, k_proj.transpose(2, 3)) + 1e-8)
            attn_out = attn_out * z.unsqueeze(-1)

            output[:, :, center:center + 1, :] = attn_out

        output = output.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(output)


class LinearDynamicInceptionBlock(nn.Module):
    def __init__(self, channels, kernel_sizes=[1, 3, 5], feature_dims=[32, 64, 32], dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList()

        for ks, fd in zip(kernel_sizes, feature_dims):
            branch = nn.Sequential(
                nn.Linear(channels, fd),
                nn.GELU(),
                nn.Conv1d(fd, fd, ks, padding=ks // 2, groups=fd),
                nn.GELU(),
                nn.Linear(fd, channels // len(kernel_sizes)),
                nn.Dropout(dropout)
            )
            self.branches.append(branch)

        self.avg_pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(channels, channels // len(kernel_sizes)),
            nn.GELU()
        )

        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)

        avg_out = self.avg_pool_branch(x.transpose(1, 2)).transpose(1, 2)
        branch_outputs.append(avg_out)

        out = torch.cat(branch_outputs, dim=-1)
        return self.norm(out)


class ConvBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        
        # Инициализация всех блоков согласно диаграмме
        self.dynamic_pooling = DynamicPooling(output_size=None, mode='adaptive')
        self.performer_attention = LinearPerformerAttention(dim, heads, feature_dim=256, dropout=dropout)
        self.parameterization_kernel = LinearParameterizationKernel(dim, dim, feature_dim=32)
        self.kernel_compression = FastKernelCompression(dim, reduction_ratio=4)
        self.block_sparse_attention = LinearBlockSparseAttention(dim, block_size=32, heads=heads, feature_dim=128, dropout=dropout)
        self.dilated_residual_cnn = OptimizedDilatedResidual(dim, dilations=[1, 2, 4, 8], dropout=dropout, use_glu=True)
        self.token_merging = TokenMerging(dim, reduction_ratio=2)
        self.local_attention = LinearLocalAttention(dim, window_size=7, heads=heads, feature_dim=64, dropout=dropout)
        self.dynamic_inception_block = LinearDynamicInceptionBlock(dim, kernel_sizes=[1, 3, 5], feature_dims=[32, 64, 32], dropout=dropout)

        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = self.dynamic_pooling(x)
        x = self.performer_attention(x)
        x = self.parameterization_kernel(x)
        x = self.kernel_compression(x)
        x = self.block_sparse_attention(x)
        x = self.dilated_residual_cnn(x)
        x = self.token_merging(x)
        x = self.local_attention(x)
        x = self.dynamic_inception_block(x)
        
        return self.final_norm(x)