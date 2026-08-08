"""
Gaussian Diffusion Process v3 — 支援 soft constraint loss

v2 -> v3 變動：
  training_loss 新增 soft constraint 懲罰項（可微，從預測的 x0 計算）：
    - MIB:      同組 block 的 log_r 應一致（同尺寸 <=> 同 area 同 aspect）
    - boundary: 該貼邊的 block，對應座標應接近邊界（0 或 1）
    - cluster:  同組 block 的中心應彼此靠近
  這些懲罰讓模型在去噪的同時學到 soft 傾向，而非只靠 conditioning。

  推理用的 ddim_sample_constrained 沿用 v2（hard constraint inpainting），
  但多接一個 group_bias 參數傳給 model。
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianDiffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device

        betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.concatenate([[1.0], alphas_cumprod[:-1]])

        self.betas = self._to_tensor(betas)
        self.alphas = self._to_tensor(alphas)
        self.alphas_cumprod = self._to_tensor(alphas_cumprod)
        self.alphas_cumprod_prev = self._to_tensor(alphas_cumprod_prev)
        self.sqrt_alphas_cumprod = self._to_tensor(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = self._to_tensor(np.sqrt(1.0 - alphas_cumprod))

        self.sqrt_recip_alphas = self._to_tensor(np.sqrt(1.0 / alphas))
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = self._to_tensor(posterior_var)
        self.posterior_log_variance = self._to_tensor(
            np.log(np.maximum(posterior_var, 1e-20))
        )

    def _to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def to(self, device):
        self.device = device
        for attr in [
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas", "posterior_variance", "posterior_log_variance",
        ]:
            setattr(self, attr, getattr(self, attr).to(device, non_blocking=True))
        return self

    def _extract(self, schedule, t, shape):
        batch_size = t.shape[0]
        out = schedule.gather(0, t)
        return out.view(batch_size, *([1] * (len(shape) - 1)))

    # -- Forward process --
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        x_t = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    # v5.20: epsilon-prediction／v-prediction 共用的還原介面。model 的
    # 原始輸出 `model_out` 依 `prediction_type` 有不同意義：
    #   "epsilon"（預設）：model_out 就是 eps_pred，x0_pred 用標準公式反推
    #                      （= 舊行為，逐位元不變）。
    #   "v"：model_out 是 v = sqrt(ᾱ_t)·eps − sqrt(1−ᾱ_t)·x0（Salimans &
    #        Ho, 2022），跟 x_t = sqrt(ᾱ_t)·x0 + sqrt(1−ᾱ_t)·eps 聯立解出
    #        x0_pred = sqrt(ᾱ_t)·x_t − sqrt(1−ᾱ_t)·v，
    #        eps_pred = sqrt(1−ᾱ_t)·x_t + sqrt(ᾱ_t)·v。
    # 所有下游程式碼（DDIM 更新公式、soft constraint loss 的 x0 重建、
    # self-conditioning 的 x0 估計）都只吃這組 (x0_pred, eps_pred) 標準
    # 介面，不需要各自知道 model 實際在預測什麼——這是唯一需要依
    # prediction_type 分支的地方。
    def _recover_x0_and_eps(self, x_t, model_out, alpha_bar_t, prediction_type="epsilon"):
        sqrt_ab = torch.sqrt(alpha_bar_t)
        sqrt_1mab = torch.sqrt(1 - alpha_bar_t)
        if prediction_type == "v":
            x0_pred = sqrt_ab * x_t - sqrt_1mab * model_out
            eps_pred = sqrt_1mab * x_t + sqrt_ab * model_out
        else:
            eps_pred = model_out
            x0_pred = (x_t - sqrt_1mab * eps_pred) / sqrt_ab
        return x0_pred, eps_pred

    # ----------------------------------------------------------------
    # Soft constraint 懲罰（可微，作用在預測的 x0 上）
    # x0_pred: (B, N, 3) = (x_norm, y_norm, log_r)
    # ----------------------------------------------------------------
    @staticmethod
    def _group_variance_loss(values, group_ids, mask):
        """
        對每個 batch、每個非零 group，計算組內 values 的變異數總和。

        values:    (B, N) 要算變異數的值（如 log_r 或 x_norm）
        group_ids: (B, N) long，0 表無組
        mask:      (B, N) float，0/1（padding=0）

        回傳 (B,) tensor：每個 sample 自己的 sum over (gid>0) of
        var_within_group（v5.8 改成逐 sample 回傳，之前是整個 batch
        加總成一個 scalar——呼叫端想要舊行為時自己再 `.sum()`／`.mean()`
        即可，`.mean()` 在數學上完全等於舊版的 `.sum() / B`，不是近似）。

        實作：用 scatter_add 一次跨 batch 算完。
        """
        B, N = values.shape
        device = values.device
        # 把 group_ids 映射到 [0, G_max]，0 仍代表「無組」
        # max() 在 GPU 上會引起 sync，但只在這個 batch 算一次而已
        G_max = int(group_ids.max().item()) + 1   # +1 因為 0 也算一個 bucket
        if G_max <= 1:
            return values.new_zeros(B)   # 整個 batch 都沒組

        # 對每個 (b, gid) 算 sum, sum_sq, count
        # flat index = b * G_max + gid
        flat_idx = (torch.arange(B, device=device).unsqueeze(1) * G_max +
                    group_ids).reshape(-1)               # (B*N,)
        v_flat = (values * mask).reshape(-1)             # padding 不貢獻
        v2_flat = (values * values * mask).reshape(-1)
        m_flat = mask.reshape(-1)                        # 對 count 用

        n_bins = B * G_max
        sum_v = values.new_zeros(n_bins).scatter_add_(0, flat_idx, v_flat)
        sum_v2 = values.new_zeros(n_bins).scatter_add_(0, flat_idx, v2_flat)
        cnt = values.new_zeros(n_bins).scatter_add_(0, flat_idx, m_flat)

        # 變異數 = E[X^2] - E[X]^2，但只在 cnt >= 2 的 bucket 才有意義；
        # 同時排除 gid=0（每個 batch 的 bucket 0 都是「無組」聚集區，跳過）
        sum_v = sum_v.view(B, G_max)
        sum_v2 = sum_v2.view(B, G_max)
        cnt = cnt.view(B, G_max)

        # 排除 gid=0
        sum_v = sum_v[:, 1:]
        sum_v2 = sum_v2[:, 1:]
        cnt = cnt[:, 1:]

        # 安全 divide
        safe_cnt = cnt.clamp(min=1)
        mean = sum_v / safe_cnt
        var = sum_v2 / safe_cnt - mean * mean
        var = var.clamp(min=0)   # 浮點誤差可能讓 var 微負

        # 只算 cnt >= 2 的組
        weight = (cnt >= 2).float()
        return (var * weight).sum(dim=1)   # (B,)

    def _soft_constraint_loss(self, x0_pred, batch, weights, t_weight=None):
        """
        回傳 (total, (mib, cluster, boundary, overlap))，total 為 scalar，
        後四項為 detached scalar（純粹給 log 用，跨 epoch 比較同一種
        定義的 loss 大小，不受 t_weight 影響）。
        向量化版：用 scatter_add 跨整個 batch 計算組內變異數，
        移除原本的 `for b in range(B): for gid in unique:` 雙重迴圈
        以及每次 `gid.item()` 造成的 GPU→CPU 同步。

        t_weight（v5.8，實驗用）：None（預設）= 完全比照 v5.7 以前的算法
        （mib/cluster/overlap 三項在數學上跟舊版逐 batch 加總再除以 B
        完全等價，不是近似；boundary 項刻意保留原本「整個 batch 所有貼邊
        block 一起 pooled 平均」的算法，不動）。
        給 (B,) tensor 時 = 每個 sample 各自的 t 加權權重（見 training_loss
        呼叫處說明），四個 soft loss 全部先各自攤成 per-sample (B,)，
        再依權重做加權平均——這時候 boundary 項的池化方式會變成「每個
        sample 各自平均、sample 間再平均」，跟 t_weight=None 時的「整個
        batch 逐 block 平均」語意不同（前者每個 sample 權重相等，後者每個
        貼邊 block 權重相等），這是支援逐樣本 t 加權必須付出的代價，但也
        代表 t_weight=None 這個分支被刻意設計成跟這個功能加入之前的訓練
        行為完全一致，方便做「加不加 t 權重」的單一變因 A/B。
        """
        device = x0_pred.device
        B, N, _ = x0_pred.shape

        x = x0_pred[:, :, 0]       # (B, N)
        y = x0_pred[:, :, 1]
        log_r = x0_pred[:, :, 2]

        mask = batch["mask"].to(device, non_blocking=True).float()              # (B, N)
        mib_group = batch["mib_group"].to(device, non_blocking=True)            # (B, N) long
        cluster_group = batch["cluster_group"].to(device, non_blocking=True)
        bnd = batch["boundary_code"].to(device, non_blocking=True)

        # ---- MIB: 同組 log_r 一致 → 組內變異數總和，(B,) per-sample ----
        mib_loss_ps = self._group_variance_loss(log_r, mib_group, mask)

        # ---- cluster: 同組中心靠近 → 組內 x 變異 + y 變異，(B,) per-sample ----
        cluster_loss_ps = (self._group_variance_loss(x, cluster_group, mask) +
                           self._group_variance_loss(y, cluster_group, mask))

        # ---- boundary: 該貼邊的 block 對應座標接近 0 或 1 ----
        b_left   = ((bnd & 1) > 0).float() * mask
        b_right  = ((bnd & 2) > 0).float() * mask
        b_top    = ((bnd & 4) > 0).float() * mask
        b_bottom = ((bnd & 8) > 0).float() * mask
        b_num = (b_left   * (x - 0.0) ** 2 +
                 b_right  * (x - 1.0) ** 2 +
                 b_top    * (y - 1.0) ** 2 +
                 b_bottom * (y - 0.0) ** 2)                # (B, N)
        b_cnt = b_left + b_right + b_top + b_bottom        # (B, N)

        # ---- v4.0: Overlap loss ----
        # 用 x0_pred 算每對 block 的 bbox 重疊面積（可微）。
        # state = (x_norm, y_norm, log_r)，配合 areas 推出 (w, h)，再算 pair-wise overlap。
        # 全部用 normalized 座標，整個 batch 向量化（O(B * N^2)）。
        #
        # v4.1 NaN 修正：
        #   (a) x/y/log_r 進 overlap 前 clamp 範圍。早期 t 大時 x0_pred 可能 inf。
        #   (b) sqrt(0) 反向梯度 = inf。對 areas 加 epsilon 避免梯度爆炸，
        #       而不是只 clamp forward 數值。
        #   (c) 強制升 fp32 算（AMP fp16 在 pair-wise N^2 累積容易溢位）。
        overlap_loss_ps = x0_pred.new_zeros(B)
        area_loss_ps = x0_pred.new_zeros(B)
        if "areas_norm" in batch:
            areas = batch["areas_norm"].to(device, non_blocking=True).float()    # (B, N)
            # 強制升 fp32（從 AMP 的 fp16 升回來）
            with torch.cuda.amp.autocast(enabled=False):
                x32 = x.float().clamp(-2.0, 2.0)             # 防 inf
                y32 = y.float().clamp(-2.0, 2.0)
                log_r32 = log_r.float().clamp(-3.0, 3.0)
                # log_r → w, h（w * h = area）
                r = torch.exp(log_r32)                       # clamp 後 r ∈ [exp(-3), exp(3)] ≈ [0.05, 20]
                # padding 的 areas 是 0，sqrt(0) 反向 = inf 會炸。
                # 對 areas 加 epsilon（這對 forward 數值影響 < 1e-6，但 backward 梯度安全）
                eps = 1e-8
                w_blk = torch.sqrt(areas * r + eps)
                h_blk = torch.sqrt(areas / r + eps)

                # Pair-wise (i, j) 重疊面積
                x_left  = x32 - w_blk / 2; x_right = x32 + w_blk / 2
                y_bot   = y32 - h_blk / 2; y_top   = y32 + h_blk / 2

                ovx = torch.minimum(x_right[:, :, None], x_right[:, None, :]) - \
                      torch.maximum(x_left[:, :, None],  x_left[:, None, :])
                ovy = torch.minimum(y_top[:, :, None],   y_top[:, None, :]) - \
                      torch.maximum(y_bot[:, :, None],   y_bot[:, None, :])
                overlap_area = torch.relu(ovx) * torch.relu(ovy)    # (B, N, N)

                # 只算 mask=1 的 pair（也排除 padding），且只取上三角
                m_pair = mask[:, :, None] * mask[:, None, :]
                tri_upper = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool),
                                       diagonal=1).float().unsqueeze(0)
                valid_pair = m_pair * tri_upper

                overlap_sum = (overlap_area * valid_pair).sum(dim=(1, 2))    # (B,)
                # v4.2: 除以 k（有效 block 數）而不是 n_pair（k*(k-1)/2）。
                # 之前除 n_pair 讓 loss 值太小（~0.004），模型收不到明顯壓力。
                # 除 k 讓 loss 量級提高到 ~0.1（跟 MSE 同量級），配合 lambda=1.0
                # 貢獻約 50%。同時「大 k sample 貢獻大」，因為大 k 天然重疊 pair 多。
                k_per_sample = mask.sum(dim=1).clamp(min=1)                  # (B,)
                overlap_loss_ps = overlap_sum / k_per_sample                 # (B,)

                # v5.29: Packing density（bbox 面積）loss，重用上面已經算好
                # 的 x_left/x_right/y_bot/y_top（fp32、clamp 過、對 padding
                # 安全）。padding 位置（mask=0）在算 min 前設 +inf、算 max
                # 前設 -inf，避免無意義座標污染 bbox 範圍。
                inf = float("inf")
                mask_bool = mask > 0.5
                bbox_xmin = torch.where(mask_bool, x_left, x_left.new_full((), inf)).min(dim=1).values
                bbox_xmax = torch.where(mask_bool, x_right, x_right.new_full((), -inf)).max(dim=1).values
                bbox_ymin = torch.where(mask_bool, y_bot, y_bot.new_full((), inf)).min(dim=1).values
                bbox_ymax = torch.where(mask_bool, y_top, y_top.new_full((), -inf)).max(dim=1).values
                area_pred_ps = (bbox_xmax - bbox_xmin) * (bbox_ymax - bbox_ymin)   # (B,)

                total_block_area_ps = (areas * mask).sum(dim=1)             # (B,)，跟預測無關的固定值
                # area_pred_ps 恆 >= total_block_area_ps（bbox 不可能比所有
                # block 面積總和還小），這裡的 -1.0 純粹讓 loss=0 對應
                # 100% packing density，不影響梯度方向。
                area_loss_ps = area_pred_ps / total_block_area_ps.clamp(min=eps) - 1.0

        if t_weight is None:
            # ---- 跟 v5.8 之前完全一致的算法（加 area 項時，lambda_area
            # 預設 0.0，加法上完全不影響改動前的行為）----
            mib_loss = mib_loss_ps.mean()
            cluster_loss = cluster_loss_ps.mean()
            boundary_loss = b_num.sum() / b_cnt.sum().clamp(min=1)
            overlap_loss = overlap_loss_ps.mean()
            area_loss = area_loss_ps.mean()
            total = (weights["lambda_mib"] * mib_loss +
                     weights["lambda_cluster"] * cluster_loss +
                     weights["lambda_boundary"] * boundary_loss +
                     weights.get("lambda_overlap", 0.0) * overlap_loss +
                     weights.get("lambda_area", 0.0) * area_loss)
            return total, (mib_loss.detach(), cluster_loss.detach(),
                           boundary_loss.detach(), overlap_loss.detach(), area_loss.detach())

        # ---- t 加權：五項都先攤成 per-sample，再依 t_weight 加權平均 ----
        boundary_loss_ps = b_num.sum(dim=1) / b_cnt.sum(dim=1).clamp(min=1)   # (B,)
        total_ps = (weights["lambda_mib"] * mib_loss_ps +
                    weights["lambda_cluster"] * cluster_loss_ps +
                    weights["lambda_boundary"] * boundary_loss_ps +
                    weights.get("lambda_overlap", 0.0) * overlap_loss_ps +
                    weights.get("lambda_area", 0.0) * area_loss_ps)           # (B,)
        w = t_weight.clamp(min=0.0)
        total = (total_ps * w).sum() / w.sum().clamp(min=1e-8)
        return total, (mib_loss_ps.mean().detach(), cluster_loss_ps.mean().detach(),
                       boundary_loss_ps.mean().detach(), overlap_loss_ps.mean().detach(),
                       area_loss_ps.mean().detach())

    # ----------------------------------------------------------------
    # Training loss
    # ----------------------------------------------------------------
    def training_loss(self, model, batch, soft_weights=None, min_snr_gamma=None,
                       use_self_cond=False, prediction_type="epsilon"):
        """
        Args:
            model: FloorplanDiffusionModel
            batch: dataloader 的一個 batch（dict），需含
                   state, block_features, conn_weights, group_bias, mask,
                   mib_group, cluster_group, boundary_code
            soft_weights: None 表示這步不加 soft loss（warmup 用）；
                          否則為 dict(lambda_mib, lambda_cluster, lambda_boundary,
                          lambda_overlap, weight_soft_loss_by_alpha_bar)
            min_snr_gamma: None（預設）= 主要去噪 MSE 不加權，跟 v5.9 之前
                          完全一樣的算法（整個 batch 攤平算一個 loss，不看
                          各 sample 有效 block 數差異）。給浮點數時 = Min-SNR-
                          gamma 加權（Hang et al., "Efficient Diffusion
                          Training via Min-SNR Weighting Strategy," ICCV
                          2023）：每個 sample 依自己的 SNR_t = ᾱ_t/(1-ᾱ_t)
                          算權重 w = min(SNR_t, gamma)/SNR_t。

                          注意方向跟 v5.8 的 weight_soft_loss_by_alpha_bar
                          剛好相反，不要搞混：這裡 t 小（雜訊少、SNR 遠大於
                          gamma）時 w 被壓得很小（實測 t=1 時 w≈0.001），
                          t 中大（SNR ≤ gamma，實測約 t≳150 之後）時 w=1、
                          完全不壓。原因是 epsilon-prediction 這種「noise-
                          space 權重恆定為 1」的 loss，換算到 x0 重建誤差的
                          隱含權重正比於 SNR(t)——t 小的步驟 SNR 極大，雖然
                          「本來就很好學」（幾乎沒雜訊，預測誤差對 x0 影響
                          極小），卻在原始 loss 裡分到不成比例地大的梯度
                          預算，把「t 大、真正要學怎麼從雜訊生出全域結構」
                          這些更難、更重要的步驟給比下去，導致不同 t 之間
                          的最佳梯度方向互相衝突、拖慢收斂。Min-SNR 把 t 小
                          這端的權重「封頂」在 gamma，讓訓練力氣更公平地
                          分給 t 大的步驟，論文報告 3.4x 收斂加速。

                          跟 v5.8 的差異：v5.8 是「downweight 不可信的 t 大
                          樣本」（x0_pred 反推在 t 大時數值不穩定，作用在
                          從 x0_pred 算出的輔助 soft constraint loss 上）；
                          這裡是「downweight 已經佔太多梯度預算的 t 小樣本」
                          （作用在主要的 noise-prediction MSE 上，跟數值
                          穩不穩定無關，是梯度預算的重新分配）——方向相反、
                          道理也不同，只是剛好都用「依 t 調整 loss 貢獻」
                          這個手段，不要因為手段像就以為兩者在做同一件事。
                          gamma 用論文預設值 5，不當成需要另外掃的超參數
                          （見呼叫處 config.py:
                          min_snr_gamma）。
            use_self_cond: False（預設）= 不做 self-conditioning，`model`
                          forward 一次，`self_cond=None`（Denoiser 內部視為
                          零向量），跟改動前完全等價。True 時每個 batch 以
                          50% 機率額外做一次 no-grad forward（用零向量當
                          self_cond）取得 x0_pred 估計、detach 後當這個
                          batch 真正 forward 的 self_cond；另外 50% 機率
                          維持零向量、省下那次多的 forward（訓練時平均
                          1.5x forward 次數）。見 model.py: Denoiser 的
                          self_cond_proj 說明。
            prediction_type: "epsilon"（預設）= model 預測 noise，跟改動前
                          完全等價。"v" 時改預測
                          v = sqrt(ᾱ_t)·eps − sqrt(1−ᾱ_t)·x0（Salimans &
                          Ho, 2022），主要 MSE loss 的目標從 `noise` 換成
                          這個 v-target；soft constraint loss／
                          self-conditioning 用到的 x0_pred 重建也一併換成
                          v-prediction 對應公式（見
                          `GaussianDiffusion._recover_x0_and_eps`）。不改
                          架構、不新增可學習參數，但輸出的意義不同，
                          舊 checkpoint 不能跟新 prediction_type 混用。
        Returns:
            loss (scalar), info (dict)

        v5.8（實驗用，timestep 加權）：soft constraint loss（mib/cluster/
        boundary/overlap）都是從 x0_pred 反推出來的，這個反推
        `x0_pred = (x_t - sqrt(1-ᾱ_t)·noise_pred) / sqrt(ᾱ_t)` 在 t 大
        （雜訊多，ᾱ_t 接近 0）時數值上很不穩定，x0_pred 這時候基本上不可信
        （`_soft_constraint_loss` 裡的 overlap 分支有 clamp 純粹是防
        NaN/inf，不是說這個範圍內的值就可信）。如果 `soft_weights` 裡
        `weight_soft_loss_by_alpha_bar=True`，就把 ᾱ_t（本來就要算來重建
        x0_pred，不需要多算）當每個 sample 的權重傳給
        `_soft_constraint_loss`，讓 t 小（雜訊少、x0_pred 可信）的樣本
        在 soft loss 裡佔比較大的份量，t 大的樣本自動被壓低——不新增任何
        需要另外調的超參數。預設關閉（見 `config.py:
        weight_soft_loss_by_alpha_bar`），開啟後跟關閉時的訓練行為在
        mib/cluster/overlap 三項數學上完全等價、只有 boundary 項的 pooling
        方式會變（見 `_soft_constraint_loss` docstring），方便單獨驗證這一
        個改動的效果。
        """
        device = next(model.parameters()).device
        x0 = batch["state"].to(device, non_blocking=True)
        block_features = batch["block_features"].to(device, non_blocking=True)
        conn_weights = batch["conn_weights"].to(device, non_blocking=True)
        group_bias = batch["group_bias"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=device)
        x_t, noise = self.q_sample(x0, t)
        # 提前算：v-target（若 prediction_type=="v"）、soft loss 的 t_weight、
        # self-conditioning 的 x0 重建都要用，只算一次
        alpha_bar_t = self._extract(self.alphas_cumprod, t, x_t.shape)

        if prediction_type == "v":
            target = torch.sqrt(alpha_bar_t) * noise - torch.sqrt(1 - alpha_bar_t) * x0
        else:
            target = noise

        self_cond = None
        if use_self_cond and torch.rand(()).item() < 0.5:
            with torch.no_grad():
                model_out_sc = model(x_t, block_features, conn_weights, t, mask, group_bias,
                                     self_cond=None)
                x0_pred_sc, _ = self._recover_x0_and_eps(x_t, model_out_sc, alpha_bar_t,
                                                          prediction_type)
            self_cond = x0_pred_sc.detach()

        noise_pred = model(x_t, block_features, conn_weights, t, mask, group_bias,
                           self_cond=self_cond)

        # 主要去噪 MSE loss（masked）——`noise_pred` 是 model 的原始輸出，
        # `target` 已經依 prediction_type 換成正確的訓練目標
        mask_expanded = mask.unsqueeze(-1).float()
        if min_snr_gamma is None:
            # 跟 Min-SNR 加入之前完全一樣：整個 batch 攤平一起算
            mse = F.mse_loss(noise_pred * mask_expanded, target * mask_expanded, reduction="sum")
            mse = mse / mask_expanded.sum().clamp(min=1)
        else:
            sq_err = (noise_pred - target) ** 2 * mask_expanded             # (B, N, 3)
            per_sample_num = sq_err.sum(dim=(1, 2))                         # (B,)
            per_sample_cnt = mask_expanded.sum(dim=(1, 2)).clamp(min=1)     # (B,)
            per_sample_mse = per_sample_num / per_sample_cnt                # (B,)

            alpha_bar_flat = alpha_bar_t.view(B)
            snr = alpha_bar_flat / (1.0 - alpha_bar_flat).clamp(min=1e-8)
            min_snr_w = torch.clamp(snr, max=min_snr_gamma) / snr.clamp(min=1e-8)
            mse = (per_sample_mse * min_snr_w).mean()

        info = {"mse": mse.detach()}
        loss = mse

        # soft constraint loss（從預測 x0 算）
        if soft_weights is not None:
            x0_pred, _ = self._recover_x0_and_eps(x_t, noise_pred, alpha_bar_t, prediction_type)

            t_weight = None
            if soft_weights.get("weight_soft_loss_by_alpha_bar", False):
                t_weight = alpha_bar_t.view(B)   # (B,)，見上方 docstring

            soft_loss, (ml, cl, bl, ol, al) = self._soft_constraint_loss(
                x0_pred, batch, soft_weights, t_weight=t_weight)
            loss = loss + soft_loss
            info.update({"soft": soft_loss.detach(), "mib": ml, "cluster": cl,
                         "boundary": bl, "overlap": ol, "area": al})

        return loss, info

    # -- 無約束 DDIM sampling（向後相容）--
    @torch.no_grad()
    def ddim_sample(self, model, shape, block_features, conn_weights,
                    mask=None, ddim_steps=50, eta=0.0, group_bias=None,
                    mib_group=None):
        return self.ddim_sample_constrained(
            model, shape, block_features, conn_weights, mask,
            ddim_steps=ddim_steps, eta=eta,
            gt_state=None, fixed_mask=None, preplaced_mask=None,
            group_bias=group_bias, mib_group=mib_group,
        )

    # -- 約束感知 DDIM sampling --
    @torch.no_grad()
    def ddim_sample_constrained(
        self, model, shape, block_features, conn_weights,
        mask=None, ddim_steps=50, eta=0.0,
        gt_state=None, fixed_mask=None, preplaced_mask=None,
        group_bias=None,
        mib_group=None,                # (B, N) long, 0=無；用於最後 10% 步驟硬投影
        mib_project_ratio=0.9,         # 進度超過這個比例後啟用 MIB 投影
    ):
        device = block_features.device
        B = shape[0]

        has_constraints = (gt_state is not None and
                           fixed_mask is not None and
                           preplaced_mask is not None)

        # MIB 引導：把同組的 log_r（state 的 dim 2）強制投影到組內平均
        # 只在後段啟用，前段讓 diffusion 自由探索。
        has_mib_guide = mib_group is not None and (mib_group > 0).any()
        if has_mib_guide:
            mib_group = mib_group.to(device)

        step_size = self.T // ddim_steps
        timesteps = list(range(0, self.T, step_size))[::-1]
        n_steps = len(timesteps)

        if has_constraints:
            inpaint_noise = torch.randn(shape, device=device)

        x = torch.randn(shape, device=device)

        for i, t_cur in enumerate(timesteps):
            t = torch.full((B,), t_cur, device=device, dtype=torch.long)
            noise_pred = model(x, block_features, conn_weights, t, mask, group_bias)

            alpha_bar_t = self._extract(self.alphas_cumprod, t, x.shape)

            if i + 1 < len(timesteps):
                t_prev_val = timesteps[i + 1]
                t_prev = torch.full((B,), t_prev_val, device=device, dtype=torch.long)
                alpha_bar_prev = self._extract(self.alphas_cumprod, t_prev, x.shape)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)

            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * noise_pred
            noise = torch.randn_like(x) if t_cur > 0 else 0
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

            if has_constraints:
                x_known = torch.sqrt(alpha_bar_prev) * gt_state + \
                          torch.sqrt(1 - alpha_bar_prev) * inpaint_noise

                pre_mask = preplaced_mask.unsqueeze(-1).float()
                x = x * (1 - pre_mask) + x_known * pre_mask

                fix_only = (fixed_mask & ~preplaced_mask).unsqueeze(-1).float()
                x_dim2_known = x.clone()
                x_dim2_known[:, :, 2] = x_known[:, :, 2]
                x = x * (1 - fix_only) + x_dim2_known * fix_only

            # ---- MIB 引導：最後 10% 步驟強制統一同組 log_r ----
            # 進度 progress = i / n_steps，0 開始、1 結束。
            # 進度超過 mib_project_ratio 才啟用。
            if has_mib_guide and (i / max(n_steps - 1, 1)) >= mib_project_ratio:
                x = self._project_mib_log_r(x, mib_group, fixed_mask, preplaced_mask)

        return x

    def _project_mib_log_r(self, x, mib_group, fixed_mask, preplaced_mask):
        """
        把同 MIB 組的 log_r（dim 2）投影到組內平均。
        若組內含 fixed/preplaced 成員，以它們的 log_r 為基準（不動）；
        其他成員拉過去。若整組都不是 fixed，取組內平均。

        x: (B, N, 3)
        mib_group: (B, N) long, 0 = 無組
        fixed_mask, preplaced_mask: (B, N) bool, 可為 None
        """
        B, N, _ = x.shape
        device = x.device

        # 哪些位置「不可動 log_r」：preplaced 或 fixed 都鎖形狀（含 log_r）
        if fixed_mask is not None:
            locked = fixed_mask.to(device)
        else:
            locked = torch.zeros(B, N, dtype=torch.bool, device=device)
        if preplaced_mask is not None:
            locked = locked | preplaced_mask.to(device)

        log_r = x[:, :, 2].clone()
        G_max = int(mib_group.max().item()) + 1
        if G_max <= 1:
            return x

        # 為每組計算目標 log_r。優先用 locked 成員的值；若無，用組內平均（含所有成員）。
        # 用 scatter_add 跨 batch 一次處理。
        flat_idx = (torch.arange(B, device=device).unsqueeze(1) * G_max +
                    mib_group).reshape(-1)            # (B*N,)
        log_r_flat = log_r.reshape(-1)

        # 1) locked 成員的 log_r 平均（每組）
        locked_f = locked.float().reshape(-1)
        sum_locked = log_r.new_zeros(B * G_max).scatter_add_(
            0, flat_idx, log_r_flat * locked_f)
        cnt_locked = log_r.new_zeros(B * G_max).scatter_add_(0, flat_idx, locked_f)

        # 2) 全體成員的 log_r 平均（fallback）
        sum_all = log_r.new_zeros(B * G_max).scatter_add_(0, flat_idx, log_r_flat)
        cnt_all = log_r.new_zeros(B * G_max).scatter_add_(
            0, flat_idx, torch.ones_like(log_r_flat))

        # 目標值 = 有 locked 用 locked 平均，否則用全體平均
        has_locked = cnt_locked > 0
        target = torch.where(
            has_locked,
            sum_locked / cnt_locked.clamp(min=1),
            sum_all / cnt_all.clamp(min=1),
        )                                              # (B*G_max,)

        # 取出每個 block 對應的目標 log_r
        target_per_block = target[flat_idx].view(B, N)

        # 套用：只對「屬於某組(gid>0) 且 非 locked」的位置改值
        is_mib_member = (mib_group > 0)
        update_mask = is_mib_member & ~locked

        new_log_r = torch.where(update_mask, target_per_block, log_r)
        x = x.clone()
        x[:, :, 2] = new_log_r
        return x

    # ============================================================
    # v3.8: GNN-style force-guided inference helpers
    # ============================================================
    # 設計：每個力回傳 delta (B, N, 2)，作用在 normalized (x, y) 上。
    # 力不動 log_r（形狀由 MIB clamp + 訓練時學到的 shape head 決定）。
    # 力強度全部設為可調 hyperparameter，主迴圈會 clip 總位移避免發散。

    @torch.no_grad()
    def _force_pin(self, x, pin_targets, pin_weights, mask, strength=0.02):
        """
        Pin Force：把每個 block 中心往「該 block 連到的 pin 加權中心」拉。

        x:           (B, N, 3)
        pin_targets: (B, N, 2) - normalized 目標座標（每個 block 的加權 pin 中心）
        pin_weights: (B, N)    - 每個 block 的總 pin weight（無連線 = 0）
        mask:        (B, N)
        strength:    每步最大位移比例
        """
        if pin_targets is None or pin_weights is None:
            return None
        cur = x[:, :, :2]                                  # (B, N, 2)
        delta = (pin_targets - cur) * strength             # 朝目標移動 strength 比例
        # 只有有 pin 連線的 block 才施力
        w = (pin_weights * mask).unsqueeze(-1)             # (B, N, 1)
        return delta * (w > 0).float()

    @torch.no_grad()
    def _force_grouping(self, x, grouping_group, mask, strength=0.015):
        """
        Grouping Force：同 grouping 組的 block 中心被拉到「組內中心」。
        用 scatter_add 跨 batch 一次處理。
        """
        if grouping_group is None or not (grouping_group > 0).any():
            return None
        B, N, _ = x.shape
        device = x.device

        G_max = int(grouping_group.max().item()) + 1
        if G_max <= 1:
            return None

        flat_idx = (torch.arange(B, device=device).unsqueeze(1) * G_max +
                    grouping_group).reshape(-1)            # (B*N,)
        cx = x[:, :, 0].reshape(-1)                        # (B*N,)
        cy = x[:, :, 1].reshape(-1)
        m_flat = mask.reshape(-1)

        n_bins = B * G_max
        sum_x = x.new_zeros(n_bins).scatter_add_(0, flat_idx, cx * m_flat)
        sum_y = x.new_zeros(n_bins).scatter_add_(0, flat_idx, cy * m_flat)
        cnt = x.new_zeros(n_bins).scatter_add_(0, flat_idx, m_flat)

        safe = cnt.clamp(min=1)
        target_x = (sum_x / safe)[flat_idx].view(B, N)
        target_y = (sum_y / safe)[flat_idx].view(B, N)

        # 只對「在某組」的 block 施力
        is_member = (grouping_group > 0).float() * mask
        dx = (target_x - x[:, :, 0]) * strength * is_member
        dy = (target_y - x[:, :, 1]) * strength * is_member
        return torch.stack([dx, dy], dim=-1)               # (B, N, 2)

    @torch.no_grad()
    def _force_boundary_nudge(self, x, boundary_code, mask, areas, strength=0.05):
        """
        Boundary Nudge：對有 boundary 約束的 block，往「當前 layout bbox 邊」推。
        v3.9: 改用 layout bbox 邊（不是 normalized canvas [0,1] 邊），
              否則 block 會被推到 pin 範圍之外。

        目標：block 的對應邊（不是中心）貼到 layout bbox 對應邊。
            LEFT  → block.x_left = bbox.x_min  → cx = bbox.x_min + w/2
            RIGHT → block.x_right = bbox.x_max → cx = bbox.x_max - w/2
            TOP   → block.y_top = bbox.y_max   → cy = bbox.y_max - h/2
            BOTTOM→ block.y_bot = bbox.y_min   → cy = bbox.y_min + h/2
        """
        if boundary_code is None or not (boundary_code > 0).any():
            return None

        # 推 w, h（normalized）
        r = torch.exp(x[:, :, 2]).clamp(min=0.1, max=10.0)
        w = torch.sqrt(areas * r)
        h = torch.sqrt(areas / r)
        cx = x[:, :, 0]
        cy = x[:, :, 1]

        # 當前 layout bbox：對每個 batch 取所有 valid block 的 min/max
        # (B, N) -> (B,) 4 個值
        very_large = torch.tensor(1e9, device=x.device, dtype=x.dtype)
        masked_xl = torch.where(mask > 0, cx - w/2, very_large)
        masked_xr = torch.where(mask > 0, cx + w/2, -very_large)
        masked_yb = torch.where(mask > 0, cy - h/2, very_large)
        masked_yt = torch.where(mask > 0, cy + h/2, -very_large)
        bb_xmin = masked_xl.min(dim=1, keepdim=True).values    # (B, 1)
        bb_xmax = masked_xr.max(dim=1, keepdim=True).values
        bb_ymin = masked_yb.min(dim=1, keepdim=True).values
        bb_ymax = masked_yt.max(dim=1, keepdim=True).values

        b_left   = ((boundary_code & 1) > 0).float() * mask
        b_right  = ((boundary_code & 2) > 0).float() * mask
        b_top    = ((boundary_code & 4) > 0).float() * mask
        b_bottom = ((boundary_code & 8) > 0).float() * mask

        # 每個 block 的目標 cx, cy = bbox 邊 ± w/h 的一半
        target_cx_left   = bb_xmin + w/2
        target_cx_right  = bb_xmax - w/2
        target_cy_bottom = bb_ymin + h/2
        target_cy_top    = bb_ymax - h/2

        dx = strength * (b_left * (target_cx_left - cx) + b_right * (target_cx_right - cx))
        dy = strength * (b_bottom * (target_cy_bottom - cy) + b_top * (target_cy_top - cy))
        return torch.stack([dx, dy], dim=-1)

    @torch.no_grad()
    def _force_repulsion(self, x, areas, mask, canvas_aspect=1.0, strength=0.05):
        """
        Direct Repulsion：重疊的 block 互相推開。
        把 (x_norm, y_norm, log_r) + areas 轉成 normalized (cx, cy, w, h)，
        對每對重疊 block 算 overlap rect，依重疊方向施反向力。

        areas:         (B, N) normalized to [0,1] (or arbitrary, 只看比例)
        canvas_aspect: w/h，用來把 normalized log_r 轉回 (w_norm, h_norm)
                       (因為 state 的 (x,y) 已 normalized 到 [0,1]，w/h 也用 [0,1])

        簡化版：假設 areas 已 normalized 成「占 canvas 面積比例」，
        且 log_r 直接給 w/h ratio。w_norm * h_norm = area_norm。
        """
        B, N, _ = x.shape
        device = x.device

        # 從 state 推 w, h（normalized 形式）
        # area = w * h，log_r = log(w/h) -> w = sqrt(area * r), h = sqrt(area / r)
        r = torch.exp(x[:, :, 2]).clamp(min=0.1, max=10.0)
        w = torch.sqrt(areas * r)                          # (B, N)
        h = torch.sqrt(areas / r)                          # (B, N)
        cx = x[:, :, 0]
        cy = x[:, :, 1]

        # Pair-wise overlap 計算
        # (B, N, N) 矩陣
        cx_i, cx_j = cx[:, :, None], cx[:, None, :]
        cy_i, cy_j = cy[:, :, None], cy[:, None, :]
        w_i, w_j = w[:, :, None], w[:, None, :]
        h_i, h_j = h[:, :, None], h[:, None, :]

        # i 右邊 - j 左邊 = overlap_x（正值代表 x 軸有重疊）
        ovx = torch.minimum(cx_i + w_i / 2, cx_j + w_j / 2) - \
              torch.maximum(cx_i - w_i / 2, cx_j - w_j / 2)
        ovy = torch.minimum(cy_i + h_i / 2, cy_j + h_j / 2) - \
              torch.maximum(cy_i - h_i / 2, cy_j - h_j / 2)
        overlap = (ovx.clamp(min=0) * ovy.clamp(min=0))    # (B, N, N)

        # mask：只算 i!=j 且兩個都在 mask 內
        m_pair = mask[:, :, None] * mask[:, None, :]
        eye = torch.eye(N, device=device).unsqueeze(0)
        m_pair = m_pair * (1 - eye)
        overlap = overlap * m_pair

        # 方向：從 j 推 i (i 離開 j)。dx_ij = cx_i - cx_j，再 normalize
        dxd = cx_i - cx_j
        dyd = cy_i - cy_j
        norm = (dxd * dxd + dyd * dyd).sqrt().clamp(min=1e-6)
        # 力的大小正比於 overlap，方向是 (dxd, dyd) / norm
        force_mag = overlap * strength                     # (B, N, N)
        fx_pair = force_mag * dxd / norm
        fy_pair = force_mag * dyd / norm

        # 對每個 i，累加所有 j 的貢獻
        fx = fx_pair.sum(dim=2)                            # (B, N)
        fy = fy_pair.sum(dim=2)
        return torch.stack([fx, fy], dim=-1)

    @torch.no_grad()
    def _apply_forces_clipped(self, x, deltas, preplaced_mask, fixed_mask,
                              max_step=0.05, clamp_bbox=None):
        """
        把多個 (B, N, 2) delta 加總，clip 每個 block 的 step norm <= max_step，
        套用到 x 的前兩維 (cx, cy)，但跳過 preplaced（位置鎖定）。
        Fixed-only 不鎖位置只鎖形狀，所以仍允許移動。

        v3.9: 加 clamp_bbox 參數，套用力後把 block 中心 clamp 到 [bbox] 內，
              避免 boundary nudge/repulsion 把 block 推出 pin 範圍。
        clamp_bbox: (x_min, y_min, x_max, y_max) in normalized coord, or None
        """
        if not deltas:
            return x
            return x
        total = sum(deltas)                                # (B, N, 2)
        # Clip norm per block
        step_norm = (total ** 2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        scale = (max_step / step_norm).clamp(max=1.0)
        total = total * scale
        # Mask out preplaced
        if preplaced_mask is not None:
            total = total * (1 - preplaced_mask.unsqueeze(-1).float())
        new = x.clone()
        new[:, :, 0] = x[:, :, 0] + total[:, :, 0]
        new[:, :, 1] = x[:, :, 1] + total[:, :, 1]

        # v3.9: clamp 中心到 bbox 內（pin bbox 為主）。
        # 不過 preplaced 我們不動（避免破壞 hard constraint），用之前那個 mask。
        if clamp_bbox is not None:
            xmin, ymin, xmax, ymax = clamp_bbox
            clamped = new.clone()
            clamped[:, :, 0] = new[:, :, 0].clamp(min=xmin, max=xmax)
            clamped[:, :, 1] = new[:, :, 1].clamp(min=ymin, max=ymax)
            if preplaced_mask is not None:
                pm = preplaced_mask.unsqueeze(-1).float()
                # preplaced 維持 new 原值（其實就是 inpainting 過的 gt 值）
                new[:, :, :2] = new[:, :, :2] * pm + clamped[:, :, :2] * (1 - pm)
            else:
                new = clamped
        return new

    @torch.no_grad()
    def _compute_pin_targets(self, p2b_edges_per_batch, pins_pos_per_batch,
                             N, B, device, canvas_offsets):
        """
        對每個 batch sample，把 p2b edges 聚合成 (B, N, 2) 加權 pin 中心 +
        (B, N) total weight。 pin 位置用 normalized [0,1] 座標。

        p2b_edges_per_batch: list of B elements; 每個是 list of (p, b, w)
        pins_pos_per_batch:  list of B elements; 每個是 (n_pins, 2) array (raw coord)
        canvas_offsets:      list of (x_off, y_off, w_canvas, h_canvas)，用來
                             normalize pin 座標。
        """
        targets = torch.zeros(B, N, 2, device=device)
        weights = torch.zeros(B, N, device=device)
        for b in range(B):
            edges = p2b_edges_per_batch[b]
            pins = pins_pos_per_batch[b]
            if pins is None or len(edges) == 0:
                continue
            xo, yo, cw, ch = canvas_offsets[b]
            for p, blk, w_ in edges:
                if blk >= N or p >= len(pins): continue
                px = (pins[p, 0] - xo) / cw
                py = (pins[p, 1] - yo) / ch
                targets[b, blk, 0] += w_ * px
                targets[b, blk, 1] += w_ * py
                weights[b, blk] += w_
        # 歸一化加權平均
        valid = weights > 0
        targets[..., 0] = torch.where(valid, targets[..., 0] / weights.clamp(min=1e-8), targets[..., 0])
        targets[..., 1] = torch.where(valid, targets[..., 1] / weights.clamp(min=1e-8), targets[..., 1])
        return targets, weights

    # ============================================================
    # v3.8: Main force-guided sampler
    # ============================================================
    @torch.no_grad()
    def ddim_sample_with_forces(
        self, model, shape, block_features, conn_weights,
        mask=None, ddim_steps=100, eta=0.0,
        # Hard constraints
        gt_state=None, fixed_mask=None, preplaced_mask=None,
        group_bias=None,
        mib_group=None,
        # Soft-as-force
        grouping_group=None,
        boundary_code=None,
        pin_targets=None,                  # (B, N, 2) normalized, 已預先算好
        pin_weights=None,                  # (B, N)
        areas_norm=None,                   # (B, N) normalized areas（佔 canvas 比例）
        # Best-of-N
        n_renoise_steps=70,                # 跑前 70 步、選 best、re-noise 後重跑
        select_metric_fn=None,             # callable(x_state, ...) -> (B,) tensor (low=好)
        # Post-repel
        post_repel_steps=30,               # v3.9: 50→30，減少時間
        # 力的強度
        pin_force_strength=0.02,
        grouping_force_strength=0.015,
        boundary_nudge_strength=0.05,
        repulsion_strength=0.05,
        max_step_per_iter=0.05,
        # 各機制窗口
        mib_clamp_until_t=20,
        pin_force_until_t=20,
        grouping_until_t=30,
        repulsion_from_t=50,               # v3.9: 30→50，repulsion 提早啟動（讓 overlap 改善）
        boundary_from_t=20,
        # v3.9: pin bbox clamp 上下限，套力後把 block 中心拉回 bbox 內
        clamp_bbox=None,                   # (xmin, ymin, xmax, ymax) normalized
        use_amp=False,                     # 只把 model forward 包進 autocast(fp16)，其餘算術維持 fp32
        # v5.12: 純推論端實驗，見下方 docstring 說明
        force_confidence_power=0.0,
        # v5.13: 純推論端實驗，見下方 docstring 說明
        resample_temperature=None,
        # v5.14: 純推論端實驗，見下方 docstring 說明
        repaint_resample_steps=1,
        # v5.18: 只有 model 是用 use_self_cond=True 訓練出來的才能開，
        # 見下方 docstring 說明
        use_self_cond=False,
        # v5.20: 必須跟 model 訓練時的 config.prediction_type 一致，見下方
        # docstring 說明——不是可以自由選的推論端選項，通常由呼叫方
        # （inference.py: generate_floorplan）從 model 的 config 自動帶入
        prediction_type="epsilon",
        # v5.30: 純推論端實驗，見下方 docstring 說明
        score_from_x0_pred=False,
    ):
        """
        GNN-style force-guided diffusion sampler。

        反向擴散每一步順序：
          1. DDIM reverse (model forward + x0_pred + DDIM update)
          2. Hard inpainting (preplaced/fixed shape)
          3. MIB clamp (if t >= mib_clamp_until_t)
          4. 累積各 force delta，clip 總位移，apply
          5. (在 t == renoise_at_t 時做 Best-of-N + re-noise)

        然後 Post-repel 階段（純物理迴圈、無 model）。

        force_confidence_power (v5.12，預設 0.0 = 關閉，跟改動前完全等價):
            四個 force（pin/grouping/repulsion/boundary）各自的強度目前是
            在各自的 t 窗口內（`*_until_t`／`*_from_t`）套用固定常數。這個
            參數讓有效強度改乘上 `alpha_bar_t ** power`（`alpha_bar_t` 是
            這一步的信噪比，隨 t 從 999 降到 0 單調從接近 0 升到 1，代表
            x0_pred 這時有多可信）：power=0 時 `alpha_bar_t**0=1`，等於
            完全不變；power>0 時，每個力剛進入自己的窗口時強度接近 0，
            隨著取樣越接近 t=0（x0_pred 越可信）平滑增強到接近原本設定的
            滿強度，而不是一進窗口就是滿強度的瞬間跳變。窗口本身（何時
            開始/結束）跟滿強度的數值都完全沿用 v5.0 已驗證的預設，這個
            機制只改窗口「內部」怎麼分配強度。

        resample_temperature (v5.13，預設 None = 關閉，跟改動前完全等價):
            現有的 best-of-N 機制在 `n_renoise_steps`（預設 70%）那個
            checkpoint，是把 N 個候選裡分數最好（`select_metric_fn` 最低，
            目前用 overlap 近似）的「唯一一個」複製到全部 N 個 batch slot、
            再各自加噪聲重跑剩下 30% 的步驟——等於把所有候選硬性收斂成同一
            個贏家的變體，其餘 N-1 個候選（就算只比贏家差一點點）直接丟棄。
            這是 SMC/Feynman-Kac 類文獻裡「resampling」步驟的一個退化特例
            （全部權重收斂到單一 particle），文獻發現這種硬性 collapse 通常
            不如「依分數做加權重新抽樣、保留多個較好候選（可以重複抽中同一
            個，但不是全部都收斂成同一個）」有效。

            `resample_temperature=None` 時完全維持舊行為（`argmin` 硬選
            +複製）。給正浮點數時，改成：先把分數做 z-score 正規化
            （`(scores - min) / (std + eps)`，讓 temperature 的意義不受
            不同案例 overlap 絕對量級影響），再用
            `softmax(-normalized_scores / temperature)` 當機率、
            `torch.multinomial` 做「取後放回」的加權重抽樣決定新的 N 個
            batch slot 各自複製哪個候選——temperature 越小，行為越接近舊的
            `argmin` 硬選；越大，越接近均勻隨機（不看分數）。重抽後一樣
            對每個 slot 各自加獨立噪聲重跑剩下的步驟，保持既有的「re-noise
            以維持多樣性」機制不變。

        repaint_resample_steps (v5.14，預設 1 = 關閉，跟改動前完全等價):
            RePaint（Lugmayr et al., CVPR 2022）的 inpainting 技巧。現有的
            hard inpainting（見下方「Hard inpainting」區塊）每一步都把
            preplaced/fixed 的已知區域強制貼回它們各自的已知加噪版本，但
            這個「貼回去」只在當前這一步發生一次——已知區域跟自由生成區域
            之間的資訊只透過下一步的 attention 慢慢傳遞，容易在兩者交界處
            留下不協調的痕跡（本專案裡 preplaced/fixed block 附近的
            boundary 違規偏高，跟這個已知的 DDPM inpainting 缺陷方向一致）。

            RePaint 的修法：在同一個 t，做完一次完整 reverse step 後，先不
            急著往下一個 t 走，而是把結果「往回加噪聲跳回」同一個 t（用跟
            DDPM 前向過程一致的邊際分布公式），再重跑一次同樣的 reverse
            step——重複 `repaint_resample_steps` 次（最後一次的結果才真的
            往下一步走）。每次重跑，已知區域都會被重新拉回它自己的已知值，
            讓自由生成區域有更多機會在同一個雜訊量級上跟已知區域對齊，才
            繼續往下個、雜訊更低的 t 前進。`repaint_resample_steps=1`
            時完全跳過這個機制（跟改動前逐位元相同）；只有存在
            preplaced/fixed 硬限制（`has_constraints=True`）時才會啟用，
            沒有硬限制的樣本不受影響、也不用付出額外計算成本。代價：
            model forward 次數約略乘上這個倍數（`repaint_resample_steps=2`
            時 diffusion 部分的算力成本大約翻倍）。

        use_self_cond (v5.18，預設 False = 關閉，跟改動前完全等價):
            只有用 `config.use_self_cond=True` 訓練出來的 checkpoint 才能
            開這個——把上一步算出的 x0_pred 當自我調節訊號傳給下一步的
            model forward（第一步跟每次 re-noise checkpoint 之後視為零
            向量，見 model.py: Denoiser 的 self_cond_proj 說明）。訓練/
            推論的自我調節資訊來源不同（訓練是 50% 機率的額外 no-grad
            forward 估計，推論是上一步真的算出的結果）符合 Chen et al.
            2022 原始設計——訓練時只是在「近似」推論時真正會發生的情況。

        prediction_type (v5.20，預設 "epsilon" = 關閉，跟改動前完全等價):
            model 的原始輸出要當 epsilon 還是 v 解讀（見
            `GaussianDiffusion._recover_x0_and_eps`）。**必須**跟 checkpoint
            訓練時的 `config.prediction_type` 一致，否則整個 DDIM 更新
            公式會用錯誤的 x0_pred/eps_pred，生成結果會是垃圾——不是可以
            自由調的推論端超參數，一般由 `inference.py: generate_floorplan`
            從 model 自己的 config 自動帶入，不需要呼叫方手動記得設定
            （不像 `use_self_cond` 是獨立於 config 的推論端行為開關）。

        score_from_x0_pred (v5.30，預設 False = 關閉，跟改動前完全等價):
            best-of-N 的 re-noise checkpoint（`n_renoise_steps`，預設
            70% 進度）目前用 `select_metric_fn(x)` 幫 N 個候選打分數，
            但 `x` 是這一步 DDIM 更新算出來的**下一個雜訊 state**，不是
            model 對「乾淨最終佈局」的估計——`_one_diffusion_step` 內部
            其實已經算出這個估計（`x0_pred`，DDIM 更新公式本身就是從它
            推出 `x` 的），只是原本沒有被傳出來給 `select_metric_fn` 用。

            v5.13 測過「把硬性 argmin 選王者改成依分數 softmax 加權重
            抽樣」，30 樣本沒有訊號（四組 temperature 全部變差樣本數 >
            變好樣本數），診斷認為是「中途分數不夠準」。但 v5.13 一直
            都是用雜訊 `x` 算分數，從沒測過「分數本身是不是可以更準」
            這個更基礎的變因——`score_from_x0_pred=True` 時把
            `select_metric_fn` 的輸入從 `x` 換成 `x0_pred`，其餘完全
            不變（`_select_metric` 的 overlap 公式本身不用改，因為兩者
            形狀/語意相同，都是 `(B, N, 3)` 的 x/y/log_r）。可以跟
            `resample_temperature` 正交組合測試：只換評分輸入
            （`resample_temperature=None`，維持硬性 argmin）、或評分
            輸入+軟性重抽樣一起換，分開驗證兩個變因各自的貢獻。
        """
        device = block_features.device
        B = shape[0]

        has_constraints = (gt_state is not None and
                           fixed_mask is not None and
                           preplaced_mask is not None)
        has_mib = mib_group is not None and (mib_group > 0).any()

        step_size = self.T // ddim_steps
        timesteps = list(range(0, self.T, step_size))[::-1]
        n_steps = len(timesteps)
        # n_renoise_steps 是「跑前幾步後 select & re-noise」
        # 例：ddim_steps=100, n_renoise_steps=70 → 在 step index 70 時 select
        renoise_idx = n_renoise_steps if n_renoise_steps < n_steps else None

        if has_constraints:
            inpaint_noise = torch.randn(shape, device=device)

        x = torch.randn(shape, device=device)
        renoise_done = False

        def _one_diffusion_step(x, i, t_cur, self_cond):
            """跑單一 reverse step + 套所有 mid-step 機制。
            回傳 (x, next_self_cond, x0_pred)——next_self_cond 是這步的
            x0_pred（use_self_cond=True 時給下一步當自我調節輸入，否則
            恆為 None）；x0_pred（v5.30）則不論 use_self_cond 為何都會
            回傳，給 resample checkpoint 當評分輸入用（見
            score_from_x0_pred docstring）。"""
            t = torch.full((B,), t_cur, device=device, dtype=torch.long)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
                    noise_pred = model(x, block_features, conn_weights, t, mask, group_bias,
                                       self_cond=self_cond)
                noise_pred = noise_pred.float()
            else:
                noise_pred = model(x, block_features, conn_weights, t, mask, group_bias,
                                   self_cond=self_cond)
            alpha_bar_t = self._extract(self.alphas_cumprod, t, x.shape)
            if i + 1 < len(timesteps):
                t_prev_val = timesteps[i + 1]
                t_prev = torch.full((B,), t_prev_val, device=device, dtype=torch.long)
                alpha_bar_prev = self._extract(self.alphas_cumprod, t_prev, x.shape)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)

            # v5.20: 把 model 的原始輸出（epsilon 或 v，取決於
            # prediction_type）統一還原成 (x0_pred, eps_pred)，下面的 DDIM
            # 更新公式維持不變、只吃這組標準介面
            x0_pred, eps_pred = self._recover_x0_and_eps(x, noise_pred, alpha_bar_t,
                                                          prediction_type)
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps_pred
            noise = torch.randn_like(x) if t_cur > 0 else 0
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

            # ---- Hard inpainting ----
            if has_constraints:
                x_known = torch.sqrt(alpha_bar_prev) * gt_state + \
                          torch.sqrt(1 - alpha_bar_prev) * inpaint_noise
                pre_mask = preplaced_mask.unsqueeze(-1).float()
                x = x * (1 - pre_mask) + x_known * pre_mask
                fix_only = (fixed_mask & ~preplaced_mask).unsqueeze(-1).float()
                x_dim2 = x.clone(); x_dim2[:, :, 2] = x_known[:, :, 2]
                x = x * (1 - fix_only) + x_dim2 * fix_only

            # ---- MIB clamp (t >= 20 才做，即前 80%) ----
            if has_mib and t_cur >= mib_clamp_until_t:
                x = self._project_mib_log_r(x, mib_group, fixed_mask, preplaced_mask)

            # ---- 累積 forces ----
            mask_f = mask.float() if mask is not None else torch.ones(B, shape[1], device=device)
            if force_confidence_power != 0.0:
                conf_w = float(self.alphas_cumprod[t_cur]) ** force_confidence_power
            else:
                conf_w = 1.0
            deltas = []
            if t_cur >= pin_force_until_t and pin_targets is not None:
                d = self._force_pin(x, pin_targets, pin_weights, mask_f,
                                    pin_force_strength * conf_w)
                if d is not None: deltas.append(d)
            if t_cur >= grouping_until_t:
                d = self._force_grouping(x, grouping_group, mask_f,
                                         grouping_force_strength * conf_w)
                if d is not None: deltas.append(d)
            if t_cur <= repulsion_from_t and areas_norm is not None:
                d = self._force_repulsion(x, areas_norm, mask_f,
                                          strength=repulsion_strength * conf_w)
                if d is not None: deltas.append(d)
            if t_cur <= boundary_from_t and areas_norm is not None:
                d = self._force_boundary_nudge(x, boundary_code, mask_f, areas_norm,
                                               boundary_nudge_strength * conf_w)
                if d is not None: deltas.append(d)

            if deltas:
                x = self._apply_forces_clipped(x, deltas, preplaced_mask, fixed_mask,
                                               max_step=max_step_per_iter,
                                               clamp_bbox=clamp_bbox)
            next_self_cond = x0_pred.detach() if use_self_cond else None
            return x, next_self_cond, x0_pred

        def _repaint_jump_back(x_prev, t_from_val, t_to_val):
            """v5.14: 把 x_prev（在 t_from 這個雜訊量級）往回加噪聲跳到
            雜訊更多的 t_to（t_to > t_from），公式跟既有 re-noise 檢查點
            用的邊際分布一致（DDPM 前向過程 t_from -> t_to 的解析解）。"""
            ab_from = self._extract(self.alphas_cumprod,
                                    torch.full((B,), t_from_val, device=device, dtype=torch.long),
                                    x_prev.shape)
            ab_to = self._extract(self.alphas_cumprod,
                                  torch.full((B,), t_to_val, device=device, dtype=torch.long),
                                  x_prev.shape)
            ratio = ab_to / ab_from
            noise = torch.randn_like(x_prev)
            return torch.sqrt(ratio) * x_prev + torch.sqrt(1 - ratio) * noise

        # ============= 主迴圈：兩段（含 Best-of-N） =============
        # v5.18: self_cond 是「上一步」的 x0_pred，跨 step 累積傳遞，
        # use_self_cond=False 時恆為 None（Denoiser 內部視為零向量，
        # 跟改動前完全等價）。
        self_cond = None
        last_x0_pred = None
        i = 0
        while i < len(timesteps):
            t_cur = timesteps[i]
            if repaint_resample_steps > 1 and has_constraints:
                for r in range(repaint_resample_steps):
                    x_next, self_cond, last_x0_pred = _one_diffusion_step(x, i, t_cur, self_cond)
                    is_last = (r == repaint_resample_steps - 1)
                    if not is_last:
                        x = _repaint_jump_back(x_next, timesteps[i + 1] if i + 1 < len(timesteps) else 0, t_cur)
                    else:
                        x = x_next
            else:
                x, self_cond, last_x0_pred = _one_diffusion_step(x, i, t_cur, self_cond)

            # Best-of-N + re-noise 檢查點
            # v3.9: 改成「短第二段」——re-noise 到 select 那個時間點，從 i+1 繼續，
            # 不再從頭跑完 100 步。總 step = n_renoise + (n_steps - n_renoise) = n_steps
            # 比舊版（n_renoise + n_steps）省一段 model forward。
            if (renoise_idx is not None and i == renoise_idx - 1 and not renoise_done
                and select_metric_fn is not None):
                # v5.30（預設 False，跟改動前完全等價）：score_from_x0_pred=True
                # 時餵 model 這一步對「乾淨最終佈局」的估計（x0_pred），而不是
                # 還帶著雜訊的 x——見該參數 docstring 與 CHANGELOG v5.30 的
                # 根因分析（v5.13 用雜訊 x 算分數，訊號不夠可靠）。
                score_input = last_x0_pred if score_from_x0_pred else x
                scores = select_metric_fn(score_input)     # (B,) 低 = 好
                if resample_temperature is None:
                    best_idx = int(scores.argmin().item())
                    # 複製 best 到所有 batch slot
                    x_best = x[best_idx:best_idx+1].expand_as(x).clone()
                else:
                    # v5.13: 加權重抽樣（取後放回），取代硬性收斂成單一贏家
                    z = (scores - scores.min()) / (scores.std() + 1e-6)
                    probs = torch.softmax(-z / resample_temperature, dim=0)
                    resample_idx = torch.multinomial(probs, B, replacement=True)
                    x_best = x[resample_idx].clone()
                # Re-noise 到 select 時間點（不是 t=T），加少量噪聲
                t_mid = timesteps[i]                       # 剛跑完 i 步，當前 t = timesteps[i]
                alpha_mid = self._extract(self.alphas_cumprod,
                                          torch.full((B,), t_mid, device=device, dtype=torch.long),
                                          x.shape)
                noise = torch.randn_like(x_best)
                x = torch.sqrt(alpha_mid) * x_best + torch.sqrt(1 - alpha_mid) * noise
                renoise_done = True
                # v5.18: re-noise 換了 batch 內容（複製/重抽）也換了雜訊
                # 量級，上一步的 self_cond 對應的是舊的 batch 身分，直接
                # 沿用會誤導模型，重置成零向量（下一步視為「第一步」）
                self_cond = None
                # i 繼續往下走，不重置
            i += 1

        # ============= Post-repel 階段 =============
        # 純物理：只用 Direct Repulsion + Boundary Nudge，沒 model
        if post_repel_steps > 0 and areas_norm is not None:
            mask_f = mask.float() if mask is not None else torch.ones(B, shape[1], device=device)
            for _ in range(post_repel_steps):
                deltas = []
                d = self._force_repulsion(x, areas_norm, mask_f,
                                          strength=repulsion_strength)
                if d is not None: deltas.append(d)
                d = self._force_boundary_nudge(x, boundary_code, mask_f, areas_norm,
                                               boundary_nudge_strength)
                if d is not None: deltas.append(d)
                if deltas:
                    x = self._apply_forces_clipped(x, deltas, preplaced_mask, fixed_mask,
                                                   max_step=max_step_per_iter,
                                                   clamp_bbox=clamp_bbox)
                # 期間仍維持 hard constraints
                if has_constraints:
                    pre_mask = preplaced_mask.unsqueeze(-1).float()
                    x = x * (1 - pre_mask) + gt_state * pre_mask
                    fix_only = (fixed_mask & ~preplaced_mask).unsqueeze(-1).float()
                    x_dim2 = x.clone(); x_dim2[:, :, 2] = gt_state[:, :, 2]
                    x = x * (1 - fix_only) + x_dim2 * fix_only
                if has_mib:
                    x = self._project_mib_log_r(x, mib_group, fixed_mask, preplaced_mask)

        return x

    # ============================================================
    # v4.2: EDM-style Heun sampler (Karras et al. 2022)
    # ============================================================
    # 與 ddim_sample_with_forces 平行存在，共用所有 force helper。
    #
    # 三個關鍵改變 vs DDIM：
    # 1. Heun 2 階求解器：每步 2 次 model forward，截斷誤差 O(h^3) 而不是 O(h^2)
    # 2. 時間步分佈：sigma_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho, rho=7
    #    集中在小 sigma（後期），因為小 sigma 對輸出品質影響最大
    # 3. sigma-t 換算：用 DDPM 的 sigma_t = sqrt((1-alpha_bar_t) / alpha_bar_t)，
    #    找最接近的 t 傳給 model（因為 model 訓練時看的是離散 t）
    #
    # 你的 model 輸出 noise epsilon，EDM 需要 D(x; sigma) = clean sample。
    # 換算：D_theta(x; sigma) = (x - sigma * epsilon) / sqrt(1 + sigma^2)
    # 推導：EDM 空間下 x_t = x_0 + sigma * n（n ~ N(0, I)），我們 model 訓練是 x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*epsilon。
    # 兩者對應：sqrt(alpha_bar) = 1/sqrt(1+sigma^2)，sqrt(1-alpha_bar) = sigma/sqrt(1+sigma^2)。
    # 因此在 EDM 座標 x_edm 下要先 x_ddpm = x_edm / sqrt(1+sigma^2) 再 forward，反算 D_theta 見程式碼。

    @torch.no_grad()
    def _sigma_to_t(self, sigma):
        """把 EDM 的 sigma 找最接近的 DDPM t 值。用 vectorized 二分搜。"""
        # sigma_at_t = sqrt((1 - alpha_bar_t) / alpha_bar_t), t=0..T-1，單調遞增
        sigma_at_t = torch.sqrt((1.0 - self.alphas_cumprod) / self.alphas_cumprod)
        # 對每個 sigma 值找最近的 t
        # sigma 可能是 scalar 或 tensor
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma, device=sigma_at_t.device, dtype=sigma_at_t.dtype)
        # 用 abs diff 找最近
        diffs = (sigma_at_t.unsqueeze(0) - sigma.reshape(-1, 1)).abs()  # (batch, T)
        t_idx = diffs.argmin(dim=1)                                     # (batch,)
        return t_idx

    @torch.no_grad()
    def _model_to_denoiser(self, model, x_edm, sigma, block_features, conn_weights,
                            mask, group_bias, B):
        """
        把你的 noise-predictor model 包成 EDM 的 denoiser D_theta。
        x_edm: EDM 座標下的 sample（x_edm = x_0 + sigma * n）
        sigma: (B,) 每個 batch sample 的 sigma
        return: D_theta(x_edm; sigma) = 估計的 clean x_0
        """
        device = x_edm.device
        # 換算到 DDPM 座標：x_ddpm = x_edm / sqrt(1 + sigma^2)
        scale = torch.sqrt(1.0 + sigma * sigma)                    # (B,)
        x_ddpm = x_edm / scale.view(B, 1, 1)
        # 找 t 值傳給 model
        t = self._sigma_to_t(sigma).to(device).long()
        # Model forward: 預測 noise
        eps_pred = model(x_ddpm, block_features, conn_weights, t, mask, group_bias)
        # D_theta = (x_ddpm - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
        # 用 sigma 表示：sqrt(alpha_bar) = 1/scale, sqrt(1-alpha_bar) = sigma/scale
        # → D = (x_ddpm - sigma/scale * eps) * scale = x_edm - sigma * scale * eps / scale
        #      Wait 讓我重推
        # x_ddpm = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * eps
        # x_0 = (x_ddpm - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
        #     = x_ddpm * scale - sigma * eps * scale / scale  (代入 sqrt(alpha_bar)=1/scale, sqrt(1-alpha_bar)=sigma/scale)
        #     = x_ddpm * scale - sigma * eps
        # 而 x_edm = x_ddpm * scale 所以：
        # D_theta = x_edm - sigma * eps
        D = x_edm - sigma.view(B, 1, 1) * eps_pred
        return D

    @torch.no_grad()
    def edm_sample_with_forces(
        self, model, shape, block_features, conn_weights,
        mask=None, num_steps=50,
        sigma_min=0.01, sigma_max=157.4, rho=7.0,
        # Hard constraints
        gt_state=None, fixed_mask=None, preplaced_mask=None,
        group_bias=None,
        mib_group=None,
        # Soft-as-force
        grouping_group=None,
        boundary_code=None,
        pin_targets=None,
        pin_weights=None,
        areas_norm=None,
        # Best-of-N
        n_renoise_ratio=0.7,               # 70% 步時 select best + re-noise
        select_metric_fn=None,
        # Post-repel
        post_repel_steps=30,
        # 力的強度（跟 DDIM 版一致）
        pin_force_strength=0.02,
        grouping_force_strength=0.015,
        boundary_nudge_strength=0.05,
        repulsion_strength=0.05,
        max_step_per_iter=0.05,
        # 窗口：step index 為單位（EDM 用「跑到第幾步」而非 t 值定義窗口）
        # 對應 DDIM 的 (t≥20 for mib/pin, t≥30 for grouping, t≤50 for repulsion, t≤20 for boundary)
        # 假設 100 步 DDIM：mib/pin 前 80%, grouping 前 70%, repulsion 後 50%, boundary 後 20%
        mib_clamp_until_ratio=0.8,
        pin_force_until_ratio=0.8,
        grouping_until_ratio=0.7,
        repulsion_from_ratio=0.5,          # step_idx/num_steps >= 0.5 才施
        boundary_from_ratio=0.8,           # step_idx/num_steps >= 0.8 才施
        clamp_bbox=None,
    ):
        """
        EDM-style Heun sampler with force guidance。
        """
        device = block_features.device
        B, N, _ = shape

        has_constraints = (gt_state is not None and
                           fixed_mask is not None and
                           preplaced_mask is not None)
        has_mib = mib_group is not None and (mib_group > 0).any()

        # ---- 建立 EDM 時間步 sigma_0 > sigma_1 > ... > sigma_{N-1} = 0 ----
        step_indices = torch.arange(num_steps, device=device, dtype=torch.float64)
        sigmas = (sigma_max ** (1 / rho) +
                  step_indices / (num_steps - 1) *
                  (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device, dtype=torch.float64)])
        sigmas = sigmas.float()                                    # (num_steps + 1,)

        # ---- 初始 sample：從 N(0, sigma_max^2 I) 抽 ----
        x = torch.randn(shape, device=device) * sigmas[0]

        # ---- inpainting noise（跟 DDIM 版一樣，for hard constraints）----
        if has_constraints:
            inpaint_noise = torch.randn(shape, device=device)

        # renoise 檢查點
        renoise_idx = int(num_steps * n_renoise_ratio) if select_metric_fn is not None else None
        renoise_done = False

        def _apply_all_side_effects(x, i, sigma_cur):
            """在 EDM step 結束後套用所有 hard/soft/force 機制。"""
            # 進度 ratio: 0 (start) -> 1 (end)
            ratio = i / max(num_steps - 1, 1)
            # 對應 DDIM 的 t：t≈T*(1-ratio)，即 ratio↑ → t↓

            # Hard inpainting（用 sigma 表示的等效噪聲）
            if has_constraints:
                # 在 EDM 座標下：gt_edm = gt_ddpm * sqrt(1+sigma^2)
                # 對 preplaced/fixed 位置，我們希望 x = gt + sigma * noise
                sigma_broad = sigma_cur.view(1, 1, 1) if sigma_cur.dim() == 0 else sigma_cur.view(-1, 1, 1)
                x_known = gt_state + sigma_broad * inpaint_noise
                pre_mask = preplaced_mask.unsqueeze(-1).float()
                x = x * (1 - pre_mask) + x_known * pre_mask
                fix_only = (fixed_mask & ~preplaced_mask).unsqueeze(-1).float()
                x_dim2 = x.clone(); x_dim2[:, :, 2] = x_known[:, :, 2]
                x = x * (1 - fix_only) + x_dim2 * fix_only

            # MIB clamp（前 80% 步）
            if has_mib and ratio < mib_clamp_until_ratio:
                x = self._project_mib_log_r(x, mib_group, fixed_mask, preplaced_mask)

            # Forces
            mask_f = mask.float() if mask is not None else torch.ones(B, N, device=device)
            deltas = []
            if ratio < pin_force_until_ratio and pin_targets is not None:
                d = self._force_pin(x, pin_targets, pin_weights, mask_f, pin_force_strength)
                if d is not None: deltas.append(d)
            if ratio < grouping_until_ratio:
                d = self._force_grouping(x, grouping_group, mask_f, grouping_force_strength)
                if d is not None: deltas.append(d)
            if ratio >= repulsion_from_ratio and areas_norm is not None:
                d = self._force_repulsion(x, areas_norm, mask_f, strength=repulsion_strength)
                if d is not None: deltas.append(d)
            if ratio >= boundary_from_ratio and areas_norm is not None:
                d = self._force_boundary_nudge(x, boundary_code, mask_f, areas_norm,
                                                boundary_nudge_strength)
                if d is not None: deltas.append(d)
            if deltas:
                x = self._apply_forces_clipped(x, deltas, preplaced_mask, fixed_mask,
                                                max_step=max_step_per_iter,
                                                clamp_bbox=clamp_bbox)
            return x

        # ============= 主 Heun 迴圈 =============
        i = 0
        while i < num_steps:
            sigma_cur = sigmas[i].expand(B)                        # (B,)
            sigma_next = sigmas[i + 1].expand(B)                   # (B,)

            # ---- Heun 第一次 eval (Euler 預估) ----
            D_cur = self._model_to_denoiser(model, x, sigma_cur, block_features,
                                            conn_weights, mask, group_bias, B)
            d_cur = (x - D_cur) / sigma_cur.view(B, 1, 1)          # dx/dt
            x_est = x + (sigma_next - sigma_cur).view(B, 1, 1) * d_cur

            # ---- Heun 第二次 eval (2 階校正)。sigma_next=0 時退化 Euler ----
            if sigmas[i + 1] > 0:
                D_next = self._model_to_denoiser(model, x_est, sigma_next, block_features,
                                                 conn_weights, mask, group_bias, B)
                d_next = (x_est - D_next) / sigma_next.view(B, 1, 1)
                x_new = x + (sigma_next - sigma_cur).view(B, 1, 1) * 0.5 * (d_cur + d_next)
            else:
                x_new = x_est

            x = x_new

            # ---- 每步 Heun 結束後統一套 side effects ----
            x = _apply_all_side_effects(x, i, sigma_cur)

            # Best-of-N + re-noise
            if (renoise_idx is not None and i == renoise_idx - 1 and not renoise_done
                and select_metric_fn is not None):
                scores = select_metric_fn(x)
                best_idx = int(scores.argmin().item())
                x_best = x[best_idx:best_idx+1].expand_as(x).clone()
                # Re-noise 到當前 sigma
                noise = torch.randn_like(x_best)
                x = x_best + noise * sigma_cur.view(B, 1, 1)
                renoise_done = True

            i += 1

        # ============= Post-repel 階段 =============
        if post_repel_steps > 0 and areas_norm is not None:
            mask_f = mask.float() if mask is not None else torch.ones(B, N, device=device)
            for _ in range(post_repel_steps):
                deltas = []
                d = self._force_repulsion(x, areas_norm, mask_f, strength=repulsion_strength)
                if d is not None: deltas.append(d)
                d = self._force_boundary_nudge(x, boundary_code, mask_f, areas_norm,
                                                boundary_nudge_strength)
                if d is not None: deltas.append(d)
                if deltas:
                    x = self._apply_forces_clipped(x, deltas, preplaced_mask, fixed_mask,
                                                    max_step=max_step_per_iter,
                                                    clamp_bbox=clamp_bbox)
                # Post-repel 階段仍維持 hard constraints
                if has_constraints:
                    pre_mask = preplaced_mask.unsqueeze(-1).float()
                    x = x * (1 - pre_mask) + gt_state * pre_mask
                    fix_only = (fixed_mask & ~preplaced_mask).unsqueeze(-1).float()
                    x_dim2 = x.clone(); x_dim2[:, :, 2] = gt_state[:, :, 2]
                    x = x * (1 - fix_only) + x_dim2 * fix_only
                if has_mib:
                    x = self._project_mib_log_r(x, mib_group, fixed_mask, preplaced_mask)

        return x