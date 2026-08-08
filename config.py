"""
Hyperparameters and configuration for Floorplan Diffusion Model (v3).

v2 -> v3 變動：
  - block_feat_dim: 9 -> 13
      原本 boundary 是 1 個被 clip 的值（資訊毀損），改成 4 個 bit（L/R/T/B），
      另外新增 2 個旗標：is_mib_member、is_cluster_member。
      新組成 (13 dims):
        [0]      area_norm
        [1,2]    is_fixed, is_preplaced            (hard-shape flags)
        [3,4]    is_mib_member, is_cluster_member  (soft group membership)
        [5..8]   bnd_left, bnd_right, bnd_top, bnd_bottom  (boundary bits)
        [9,10,11] pin_cx, pin_cy, pin_total_weight (p2b features)
        [12]     reserved (= 0，保留擴充用)
  - 新增 soft constraint loss 權重（lambda_*）
  - 新增 group attention bias 開關
"""
from dataclasses import dataclass


@dataclass
class Config:
    # -- Data --
    max_blocks: int = 120
    min_blocks: int = 21
    max_pins: int = 200
    block_feat_dim: int = 13      # v3: 見檔頭說明

    # -- Model --
    d_model: int = 256
    n_heads: int = 8
    encoder_layers: int = 6
    denoiser_layers: int = 8
    dropout: float = 0.1
    dim_feedforward: int = 1024

    # -- Diffusion --
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    ddim_steps: int = 50

    # -- Training --
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 300
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    ema_decay: float = 0.9999
    num_workers: int = 0
    # 固定資料 shuffle 順序 + 模型初始權重，讓「同一組設定跑兩次」或
    # 「只改一個變因（例如 weight_soft_loss_by_alpha_bar）比較兩次」時，
    # 差異不會被隨機初始化/shuffle 順序汙染。不保證 GPU 運算逐 bit
    # 重現（cudnn 非決定性演算法沒關），但足以讓 A/B 比較乾淨。
    seed: int = 42

    # -- Soft constraint loss 權重（v3 新增）--
    # 這些懲罰項加在 diffusion 的 MSE loss 之上，從預測的 x0 計算。
    # 建議從小值開始，避免壓過主要的去噪 loss。
    # v3.5: 調整權重以對應觀察到的違規分佈
    #   - cluster/boundary 違規率較高且模型較難學到 → 提到 0.3
    #   - mib 改在 inference 端用硬投影強制（見 diffusion.ddim_sample_constrained），
    #     訓練端權重可調低
    lambda_mib: float = 0.1
    lambda_cluster: float = 0.3
    lambda_boundary: float = 0.3
    # v4.0: overlap soft loss，作用於 x0_pred，懲罰 pair-wise bbox 重疊面積。
    # 跟其他 soft loss 同步 warmup，從 warmup 之後才啟用。
    lambda_overlap: float = 0.3
    # v5.29（實驗用，尚未訓練驗證，預設關閉）：packing density（bbox 面積）
    # soft loss。這個 session 針對「怎麼提升 packing density」在 legalize
    # 階段測試過九個機制（v5.22-v5.28）全部不採用，換個方向從訓練端下手
    # ——讓模型一開始預測出來的座標就更緊，不是每次都靠 legalize 事後
    # 補救。重用 overlap loss 已經算好的每個 block (x_left, x_right,
    # y_bot, y_top)，直接算整個預測佈局的 bounding box 面積，除以 block
    # 總面積當 loss（見 diffusion.py:
    # GaussianDiffusion._soft_constraint_loss 說明）——跟 lambda_overlap
    # 同時作用會形成正確的張力（面積 loss 拉緊、overlap loss 擋著不讓真的
    # 疊在一起），均衡點就是「盡量緊但不重疊」，正是 packing density 的
    # 定義本身。不新增模型參數、不改架構，舊 checkpoint 對 inference 完全
    # 相容。
    lambda_area: float = 0.0
    soft_loss_warmup_epochs: int = 10   # 前幾個 epoch 先只做去噪

    # v5.8（實驗用，尚未訓練驗證，預設關閉）：soft constraint loss
    # （mib/cluster/boundary/overlap）目前對所有隨機取樣到的 timestep t
    # 一視同仁地套用同樣力道，但這些 loss 全部是從
    # x0_pred = (x_t - sqrt(1-ᾱ_t)·noise_pred) / sqrt(ᾱ_t) 反推出來的——
    # t 越大（雜訊越多）ᾱ_t 越接近 0，這個反推在數值上越不穩定，x0_pred
    # 這時候基本上不可信（現在只有 clamp 防 NaN/inf，沒有依可信度調整這些
    # loss 的貢獻）。開啟這個旗標後，每個 batch 內每個樣本的 soft loss 會
    # 依自己那個樣本的 ᾱ_t 做加權平均（ᾱ_t 越接近 1 = t 越小 = 越可信 =
    # 權重越大），直接重用 training_loss 裡本來就會算的 ᾱ_t，不需要額外
    # 調參數（見 diffusion.py: GaussianDiffusion.training_loss /
    # _soft_constraint_loss 說明）。
    # 設 False 時完全等同這個旗標加入之前的訓練行為（mib/cluster/overlap
    # 三項數學上完全等價，boundary 項刻意維持原本的全 batch pooled 算法，
    # 不是逐樣本平均）——這樣可以在同一個訓練設定下只單獨測試「加不加 t
    # 權重」這一個變因，不會混進「boundary loss 怎麼 pooled」這第二個
    # 變因。這個功能還沒有實際訓練驗證過，先維持 False，等短跑（例如 30
    # epoch）A/B 有訊號再考慮改預設值。
    weight_soft_loss_by_alpha_bar: bool = False

    # v5.9（實驗用，尚未訓練驗證，預設關閉）：兩個從網路研究來的、跟現有
    # attention 架構相容（不換 backbone）的小改動，見 method.md 1.5 節。
    #   - use_qk_norm：對每個 head 的 Q/K 在算 attention logits 之前先各自
    #     做 RMSNorm（見 model.py: RMSNorm/ConnectivityBiasedAttention）。
    #     這個 attention 的 logits 是 QK^T/sqrt(d_k) 再疊加兩個 additive
    #     bias（b2b 連線、同組），沒有機制防止 logits 隨訓練變大，QK-norm
    #     是近期 LLM（Qwen3/Gemma3 等）常用的穩定手法。會改變參數量（新增
    #     RMSNorm 的可學習 scale），舊 checkpoint（v4/v5）不能直接載入這個
    #     設定，需要重新訓練。
    #   - use_min_snr_main_loss + min_snr_gamma：主要去噪 MSE 依 Min-SNR-
    #     gamma（Hang et al., ICCV 2023）加權，見 diffusion.py:
    #     GaussianDiffusion.training_loss 說明。gamma 用論文預設值，不當
    #     成要掃的超參數。純 loss 層面的改動，不影響模型參數/架構，可以跟
    #     use_qk_norm 獨立開關。
    use_qk_norm: bool = False
    use_min_snr_main_loss: bool = False
    min_snr_gamma: float = 5.0

    # v5.10（實驗用，尚未訓練驗證，預設關閉）：座標的 Fourier 位置編碼，
    # 參考"Chip Placement with Diffusion Models"（ICML 2025，同領域的晶片
    # 巨集擺放 diffusion 論文）——他們的 ablation 顯示拿掉這個編碼會讓
    # 合法擺放率下降（0.960→0.949），作者說能提升精確擺放小物件的能力。
    # 見 model.py: CoordFourierEmbedding/Denoiser 說明。會新增可學習參數
    # （coord_proj），舊 checkpoint 不能直接載入這個設定，需要重新訓練。
    use_coord_sincos: bool = False
    coord_n_freqs: int = 16

    # v5.18（不採用，見 CHANGELOG.md）：Self-Conditioning（Chen, Zhang &
    # Hinton, 2022, "Analog Bits"）。跑完整 300 epoch 驗證，val_loss 比 v4
    # 高（0.1425 vs 0.1026）、100 樣本 paired raw overlap 明顯變差
    # （+24%）、官方 evaluate 真實分數也比目前 production 差（+1.75%），
    # 三個獨立訊號方向一致，確認不採用。機制與 checkpoint 保留備用。
    use_self_cond: bool = False

    # v5.19（不採用，見 CHANGELOG.md）：幾何 D4 對稱資料增強。30 epoch
    # 短跑跟 100 樣本 paired 都給出這個 session 最強的正面訊號（raw
    # overlap -7%~-13%），但三次獨立官方 evaluate 平均後跟 v4 幾乎打平
    # （1.1266 vs 1.128，+0.14%~-0.12% 之間）——paired 測試的變異數縮減
    # 讓一個很小的真實效果顯得比獨立重跑時更一致，是這個 session 的方法論
    # 教訓。機制與 checkpoint 保留備用。
    use_geo_augment: bool = False

    # v5.20（實驗用，尚未訓練驗證，預設 "epsilon" = 關閉，跟改動前完全
    # 等價）：v-prediction 參數化（Salimans & Ho, 2022, "Progressive
    # Distillation for Fast Sampling of Diffusion Models"）。把訓練目標從
    # 預測 noise（epsilon）換成預測 v = sqrt(ᾱ_t)·eps − sqrt(1−ᾱ_t)·x0
    # （eps 和 x0 的混合），文獻上在少步數取樣（本專案 DDIM_STEPS=10 屬於
    # 激進跳步）下通常比 epsilon-prediction 更穩定。不改架構、不新增
    # 可學習參數（跟 epsilon-prediction 用同一個網路，只是輸出的意義不同、
    # 訓練目標換了），但輸出的意義不同，舊 checkpoint 不能跟新
    # prediction_type 混用。見 diffusion.py: GaussianDiffusion 的
    # `_recover_x0_and_eps` 說明。可選值："epsilon"（預設）｜ "v"。
    prediction_type: str = "epsilon"

    # -- Attention group bias（v3 新增）--
    use_group_attention_bias: bool = True

    # -- Inference / Post-processing --
    legalization_iters: int = 100
    legalization_step: float = 0.5

    # -- Logging --
    log_interval: int = 100
    save_interval: int = 50
    output_dir: str = "checkpoints"