"""
Floorplan Diffusion Model Architecture v3

v2 -> v3 變動：
  1. BlockEncoder 輸入 9 -> 13 dims（feat_dim 由 config 帶入，不需改這裡）
  2. ConnectivityBiasedAttention 新增 group_bias 輸入：
       除了 b2b 連線權重，同 MIB / cluster 組的 block 也加一個可學習的 attention bias，
       讓同組 block 在 self-attention 裡互相增強，模型更容易學到
       「同組要協調尺寸（MIB）或位置（cluster）」。
"""
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ConnectivityBiasedAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, use_group_bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_group_bias = use_group_bias

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.bias_proj = nn.Linear(1, n_heads, bias=False)
        if use_group_bias:
            # 同組 bias：輸入是 0/1，投影成每個 head 一個可學習的偏置量
            self.group_bias_proj = nn.Linear(1, n_heads, bias=False)

    def forward(self, x, conn_weights, mask=None, group_bias=None):
        # type: (torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]) -> torch.Tensor
        B, N, _ = x.shape
        Q = self.w_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # b2b 連線 bias
        conn_bias = torch.log1p(conn_weights).unsqueeze(-1)
        conn_bias = self.bias_proj(conn_bias)
        conn_bias = conn_bias.permute(0, 3, 1, 2)
        attn = attn + conn_bias

        # group bias（v3 新增）：同 MIB / cluster 組
        if self.use_group_bias and group_bias is not None:
            gb = group_bias.unsqueeze(-1)             # (B, N, N, 1)
            gb = self.group_bias_proj(gb)             # (B, N, N, n_heads)
            gb = gb.permute(0, 3, 1, 2)               # (B, n_heads, N, N)
            attn = attn + gb

        if mask is not None:
            pad_mask = ~mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.w_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout=0.1, use_group_bias=True):
        super().__init__()
        self.attn = ConnectivityBiasedAttention(d_model, n_heads, dropout, use_group_bias)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, conn_weights, mask=None, group_bias=None):
        x = x + self.attn(self.norm1(x), conn_weights, mask, group_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class BlockEncoder(nn.Module):
    def __init__(self, feat_dim, d_model, n_heads, n_layers, dim_ff, dropout=0.1,
                 use_group_bias=True):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_ff, dropout, use_group_bias)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, block_features, conn_weights, mask=None, group_bias=None):
        x = self.feature_proj(block_features)
        for layer in self.layers:
            x = layer(x, conn_weights, mask, group_bias)
        return self.norm(x)


class Denoiser(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dim_ff, dropout=0.1,
                 use_group_bias=True):
        super().__init__()
        self.state_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_ff, dropout, use_group_bias)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    def forward(self, noisy_state, cond_emb, t, conn_weights, mask=None, group_bias=None):
        h = self.state_proj(noisy_state)
        h = h + cond_emb
        t_emb = self.time_embed(t)
        h = h + t_emb.unsqueeze(1)
        for layer in self.layers:
            h = layer(h, conn_weights, mask, group_bias)
        h = self.norm(h)
        return self.output_proj(h)


class FloorplanDiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        use_gb = getattr(config, "use_group_attention_bias", True)
        self.encoder = BlockEncoder(
            feat_dim=config.block_feat_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.encoder_layers,
            dim_ff=config.dim_feedforward,
            dropout=config.dropout,
            use_group_bias=use_gb,
        )
        self.denoiser = Denoiser(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.denoiser_layers,
            dim_ff=config.dim_feedforward,
            dropout=config.dropout,
            use_group_bias=use_gb,
        )

    def forward(self, noisy_state, block_features, conn_weights, t, mask=None,
                group_bias=None):
        cond_emb = self.encoder(block_features, conn_weights, mask, group_bias)
        noise_pred = self.denoiser(noisy_state, cond_emb, t, conn_weights, mask, group_bias)
        if mask is not None:
            noise_pred = noise_pred * mask.unsqueeze(-1).float()
        return noise_pred