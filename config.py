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
    soft_loss_warmup_epochs: int = 10   # 前幾個 epoch 先只做去噪

    # -- Attention group bias（v3 新增）--
    use_group_attention_bias: bool = True

    # -- Inference / Post-processing --
    legalization_iters: int = 100
    legalization_step: float = 0.5

    # -- Logging --
    log_interval: int = 100
    save_interval: int = 50
    output_dir: str = "checkpoints"