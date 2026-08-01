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


class RMSNorm(nn.Module):
    """
    v5.9（實驗用）：QK-normalization 用——對 Q/K 在算 attention logits 之前
    先各自做 RMSNorm，避免 logits 隨訓練過程無界增長（Dehghani et al. 2023
    "Scaling Vision Transformers to 22B Parameters"；Qwen3/Gemma3 等近期
    LLM 都採用）。跟一般拿來穩定殘差流的 LayerNorm 不同，這裡是專門套在
    每個 head 的 Q/K 向量上（維度 d_k），不是套在 d_model 維度上。
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


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


class CoordFourierEmbedding(nn.Module):
    """
    v5.10（實驗用）：座標用的多頻率 Fourier 編碼，NeRF 風格（Mildenhall
    et al. 2020），跟"Chip Placement with Diffusion Models"（ICML 2025，
    直接同領域的晶片巨集擺放 diffusion 論文）的做法一致——他們的 ablation
    顯示拿掉這個座標編碼會讓合法擺放率下降（0.960→0.949），作者說這能
    提升「精確擺放小物件」的能力。

    不能直接借用現有的 SinusoidalEmbedding：那個類別的頻率表是為了
    timestep t 這種大範圍整數（T=1000）設計的，最低頻的分量在整個
    T=1000 範圍才轉一圈；直接套用在正規化到大約 [0,1] 範圍的座標上，
    最低頻分量連四分之一個週期都轉不到，等於沒提供有效的位置區分能力。
    這裡改用頻率隨 2^k 成長（k=0..n_freqs-1）的做法，最粗的頻帶剛好讓
    座標範圍轉一整圈，愈高頻愈能分辨座標上的細微差異——這是連續、有界
    座標值的標準編碼方式。
    """
    def __init__(self, n_freqs=16):
        super().__init__()
        self.n_freqs = n_freqs
        freq_bands = 2.0 ** torch.arange(n_freqs) * math.pi
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def forward(self, p):
        # p: 任意形狀 (...,)，回傳 (..., 2*n_freqs)
        p = p.float().unsqueeze(-1) * self.freq_bands
        return torch.cat([torch.sin(p), torch.cos(p)], dim=-1)


class ConnectivityBiasedAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, use_group_bias=True, use_qk_norm=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_group_bias = use_group_bias
        self.use_qk_norm = use_qk_norm

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.bias_proj = nn.Linear(1, n_heads, bias=False)
        if use_group_bias:
            # 同組 bias：輸入是 0/1，投影成每個 head 一個可學習的偏置量
            self.group_bias_proj = nn.Linear(1, n_heads, bias=False)
        if use_qk_norm:
            # v5.9：這個 attention 的 logits 是 QK^T/sqrt(d_k) 再疊加兩個
            # additive bias（b2b 連線、同組），三項加總沒有任何機制防止
            # logits 隨訓練過程變大——QK-norm 對每個 head 的 Q/K 各自套
            # RMSNorm，把這個問題摀住。
            self.q_norm = RMSNorm(self.d_k)
            self.k_norm = RMSNorm(self.d_k)

    def forward(self, x, conn_weights, mask=None, group_bias=None):
        # type: (torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]) -> torch.Tensor
        B, N, _ = x.shape
        Q = self.w_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        if self.use_qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

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
    def __init__(self, d_model, n_heads, dim_ff, dropout=0.1, use_group_bias=True, use_qk_norm=False):
        super().__init__()
        self.attn = ConnectivityBiasedAttention(d_model, n_heads, dropout, use_group_bias, use_qk_norm)
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
                 use_group_bias=True, use_qk_norm=False):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_ff, dropout, use_group_bias, use_qk_norm)
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
                 use_group_bias=True, use_qk_norm=False, use_coord_sincos=False,
                 coord_n_freqs=16, use_self_cond=False):
        super().__init__()
        self.use_coord_sincos = use_coord_sincos
        self.use_self_cond = use_self_cond
        self.state_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        if use_self_cond:
            # v5.18（Self-Conditioning, Chen et al. 2022 "Analog Bits"）：
            # 把上一步（或訓練時以 50% 機率算出的）x0_pred 當額外輸入，用
            # 獨立的小 MLP 投影後以加法疊加在 state_proj 的輸出上——跟
            # v5.10 coord_sincos 同一種疊加方式，不改 state_proj 本身的
            # input dim（維持跟舊 checkpoint 的 state_proj 權重形狀相容）。
            # 沒有自我調節資訊時（訓練 50% 機率、推論第一步）传入零向量，
            # 這個分支此時輸出等於 self_cond_proj(0)，是一個固定、可學習
            # 的 bias，不是恆零。
            self.self_cond_proj = nn.Sequential(
                nn.Linear(3, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        if use_coord_sincos:
            # v5.10：x/y 各自算一份 Fourier 座標編碼，串起來投影回 d_model，
            #用加法疊加在 state_proj 的輸出上（跟 cond_emb/t_emb 同一種
            # additive 疊加方式）。見 CoordFourierEmbedding docstring。
            self.coord_embed_fn = CoordFourierEmbedding(n_freqs=coord_n_freqs)
            self.coord_proj = nn.Sequential(
                nn.Linear(4 * coord_n_freqs, d_model),   # 2 座標 x 2*n_freqs
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
            TransformerBlock(d_model, n_heads, dim_ff, dropout, use_group_bias, use_qk_norm)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    def forward(self, noisy_state, cond_emb, t, conn_weights, mask=None, group_bias=None,
                self_cond=None):
        h = self.state_proj(noisy_state)
        h = h + cond_emb
        if self.use_self_cond:
            if self_cond is None:
                self_cond = torch.zeros_like(noisy_state)
            h = h + self.self_cond_proj(self_cond)
        if self.use_coord_sincos:
            x_coord = noisy_state[..., 0]
            y_coord = noisy_state[..., 1]
            coord_feat = torch.cat(
                [self.coord_embed_fn(x_coord), self.coord_embed_fn(y_coord)], dim=-1)
            h = h + self.coord_proj(coord_feat)
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
        use_qk_norm = getattr(config, "use_qk_norm", False)
        self.encoder = BlockEncoder(
            feat_dim=config.block_feat_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.encoder_layers,
            dim_ff=config.dim_feedforward,
            dropout=config.dropout,
            use_group_bias=use_gb,
            use_qk_norm=use_qk_norm,
        )
        self.denoiser = Denoiser(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.denoiser_layers,
            dim_ff=config.dim_feedforward,
            dropout=config.dropout,
            use_group_bias=use_gb,
            use_qk_norm=use_qk_norm,
            use_coord_sincos=getattr(config, "use_coord_sincos", False),
            coord_n_freqs=getattr(config, "coord_n_freqs", 16),
            use_self_cond=getattr(config, "use_self_cond", False),
        )

    def forward(self, noisy_state, block_features, conn_weights, t, mask=None,
                group_bias=None, self_cond=None):
        cond_emb = self.encoder(block_features, conn_weights, mask, group_bias)
        noise_pred = self.denoiser(noisy_state, cond_emb, t, conn_weights, mask, group_bias,
                                   self_cond=self_cond)
        if mask is not None:
            noise_pred = noise_pred * mask.unsqueeze(-1).float()
        return noise_pred