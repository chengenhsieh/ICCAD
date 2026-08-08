"""
Training Script v3.4 — performance optimizations

v3.3 -> v3.4 變動（純加速，不更動訓練邏輯）：
  - AMP (mixed precision)：用 torch.cuda.amp 加速 GPU 訓練 1.5-2x，
    並省一半 VRAM。可用 config.use_amp 一鍵關掉。
  - DataLoader 平行載入：自動偵測 OS，Linux 用 num_workers=2 + persistent_workers，
    Windows 仍用 num_workers=0（避免 multiprocessing 相容性問題）。
  - non_blocking transfers：搭配 pin_memory 才有效，CPU→GPU 傳輸與計算可重疊。
  - 減少 .item() 同步點：epoch 內累積成 tensor，最後才同步取數值。
  - tqdm 更新頻率降低：避免每 batch 都觸發進度條重繪。

  Dataset / Diffusion 的 Python 迴圈也已向量化（見 dataset.py 和 diffusion.py），
  本檔不再需要任何 batch 內的逐元素處理。

輸出位置（皆放在 config.output_dir 下）：
  - loss_curve_300epoch.png    四子圖訓練曲線（從 in-memory history 畫，無 csv）
  - model_*.pt                 checkpoints

訓練資訊只印到螢幕，不再寫 train.log / train_metrics.csv。
"""
import os
import sys
import time
import platform
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from model import FloorplanDiffusionModel
from diffusion import GaussianDiffusion
from dataset import create_dataloader
from utils import EMA


# ============================================================
# ============================================================
# Loss curve plotting
# ============================================================

def plot_loss_curve(history, save_path, title="Training curves"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    if not history.get("epoch"):
        return False

    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    mse = history["mse"]
    soft = history["soft"]
    mib = history["mib"]
    cluster = history["cluster"]
    boundary = history["boundary"]
    overlap = history.get("overlap", [0.0] * len(epochs))   # v4.0
    lr = history["lr"]

    # 找到第一個 soft_on=True 的 epoch（畫垂直分隔線標記）
    soft_start = None
    for i, on in enumerate(history.get("soft_on", [])):
        if on:
            soft_start = epochs[i]; break

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="train", color="C0", linewidth=1.5)
    ax.plot(epochs, val_loss, label="val", color="C3", linewidth=1.5)
    if soft_start is not None:
        ax.axvline(soft_start, color="gray", linestyle="--", alpha=0.5, label="soft loss on")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Total loss (train vs val)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, mse, color="C2", linewidth=1.5, label="train MSE")
    ax.plot(epochs, val_loss, color="C3", linewidth=1.0, alpha=0.6, label="val MSE (= val_loss)")
    if soft_start is not None:
        ax.axvline(soft_start, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.set_title("Denoising MSE")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if any(v is not None for v in mib):
        ax.plot(epochs, mib, label="MIB", color="C4", linewidth=1.5)
        ax.plot(epochs, cluster, label="cluster", color="C1", linewidth=1.5)
        ax.plot(epochs, boundary, label="boundary", color="C6", linewidth=1.5)
        ax.plot(epochs, overlap, label="overlap", color="C2", linewidth=1.5)   # v4.0
        ax.plot(epochs, soft, label="soft total (weighted)", color="black",
                linewidth=1.0, alpha=0.5, linestyle="--")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (unweighted per-component)")
    ax.set_title("Soft constraint loss components")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, lr, color="C5", linewidth=1.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("lr")
    ax.set_title("Learning rate schedule")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    return True


# ============================================================
# Helpers
# ============================================================

def _soft_weights_for_epoch(config, epoch):
    if epoch <= config.soft_loss_warmup_epochs:
        return None
    return {
        "lambda_mib": config.lambda_mib,
        "lambda_cluster": config.lambda_cluster,
        "lambda_boundary": config.lambda_boundary,
        "lambda_overlap": getattr(config, "lambda_overlap", 0.0),   # v4.0
        "lambda_area": getattr(config, "lambda_area", 0.0),         # v5.29
        # v5.8：見 config.py / diffusion.py 說明，預設 False
        "weight_soft_loss_by_alpha_bar": getattr(config, "weight_soft_loss_by_alpha_bar", False),
    }


def _auto_num_workers(config):
    """Linux 預設 2，Windows 強制 0（multiprocessing 相容性問題）。"""
    if platform.system() == "Windows":
        return 0
    # 尊重 config 顯式設定（若大於 0）
    if getattr(config, "num_workers", 0) > 0:
        return config.num_workers
    return 2


# ============================================================
# Training
# ============================================================

def train(config):
    # 固定 seed，放在任何會消耗隨機數的操作（模型初始化、DataLoader shuffle）
    # 之前——見 config.py: seed 說明，讓「只改一個變因」的比較乾淨。
    seed = getattr(config, "seed", 42)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = getattr(config, "use_amp", True) and device.type == "cuda"

    os.makedirs(config.output_dir, exist_ok=True)
    loss_png_path = os.path.join(config.output_dir, "loss_curve_300epoch_overlap_v11.png")

    # In-memory history（取代 csv），畫 loss curve 用
    history = {k: [] for k in [
        "epoch", "train_loss", "val_loss", "mse", "soft",
        "mib", "cluster", "boundary", "overlap", "lr", "elapsed_sec", "soft_on"
    ]}

    num_workers = _auto_num_workers(config)

    print("=" * 60)
    print("Training started")
    print("Seed: {}".format(seed))
    print("Device: {} | AMP: {} | OS: {} | num_workers: {}".format(
        device, use_amp, platform.system(), num_workers))
    print("Output dir: {}".format(os.path.abspath(config.output_dir)))
    print("Config: d_model={}, enc={}, dec={}, batch={}, epochs={}".format(
        config.d_model, config.encoder_layers, config.denoiser_layers,
        config.batch_size, config.epochs))
    print("Soft loss: lambda_mib={}, lambda_cluster={}, lambda_boundary={}, "
          "lambda_overlap={}, lambda_area={}, warmup={}, "
          "weight_soft_loss_by_alpha_bar={}".format(
        config.lambda_mib, config.lambda_cluster, config.lambda_boundary,
        getattr(config, "lambda_overlap", 0.0), getattr(config, "lambda_area", 0.0),
        config.soft_loss_warmup_epochs,
        getattr(config, "weight_soft_loss_by_alpha_bar", False)))
    print("v5.9: use_qk_norm={}, use_min_snr_main_loss={} (gamma={})".format(
        getattr(config, "use_qk_norm", False),
        getattr(config, "use_min_snr_main_loss", False),
        getattr(config, "min_snr_gamma", None)))
    print("v5.10: use_coord_sincos={} (n_freqs={})".format(
        getattr(config, "use_coord_sincos", False),
        getattr(config, "coord_n_freqs", None)))
    print("v5.18: use_self_cond={}".format(getattr(config, "use_self_cond", False)))
    print("v5.19: use_geo_augment={}".format(getattr(config, "use_geo_augment", False)))
    print("v5.20: prediction_type={}".format(getattr(config, "prediction_type", "epsilon")))

    # -- 載入資料 --
    print("Loading FloorSet datasets...")
    from lite_dataset import FloorplanDatasetLite
    train_official = FloorplanDatasetLite("../")
    subset_indices = list(range(200000))
    train_official = Subset(train_official, subset_indices)

    from litetestLoader import FloorplanDatasetLiteTest
    val_official = FloorplanDatasetLiteTest("../")

    train_loader = create_dataloader(
        train_official, batch_size=config.batch_size,
        max_blocks=config.max_blocks, shuffle=True,
        num_workers=num_workers, is_test=False,
        augment=getattr(config, "use_geo_augment", False),
    )
    val_loader = create_dataloader(
        val_official, batch_size=config.batch_size,
        max_blocks=config.max_blocks, shuffle=False,
        num_workers=num_workers, is_test=True,
    )
    # 注意：persistent_workers 和 prefetch_factor 已在 create_dataloader 內部
    # 根據 num_workers 自動設定（>0 時啟用），這裡不需重複處理。

    print("Train: {} samples, Val: {} samples".format(
        len(train_official), len(val_official)))

    # -- 模型 --
    model = FloorplanDiffusionModel(config).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameters: {:,}".format(param_count))

    diffusion = GaussianDiffusion(T=config.T, beta_start=config.beta_start,
                                  beta_end=config.beta_end)
    diffusion.to(device)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)
    ema = EMA(model, decay=config.ema_decay)

    # AMP scaler（CPU 訓練時自動 disabled）
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(config.output_dir, exist_ok=True)

    best_val_loss = float("inf")
    global_step = 0

    # v5.9：跟 soft_weights 不同，Min-SNR 不受 soft_loss_warmup_epochs
    # 影響（作用在主要去噪 loss，從 epoch 1 就該套用），所以在迴圈外算一次。
    min_snr_gamma = config.min_snr_gamma if getattr(config, "use_min_snr_main_loss", False) else None
    use_self_cond = getattr(config, "use_self_cond", False)
    prediction_type = getattr(config, "prediction_type", "epsilon")

    for epoch in range(1, config.epochs + 1):
        model.train()
        soft_w = _soft_weights_for_epoch(config, epoch)
        soft_on = soft_w is not None

        # 累積成 tensor，最後才 .item()，省掉每 batch 一次 GPU→CPU 同步
        loss_sum = torch.zeros((), device=device)
        mse_sum  = torch.zeros((), device=device)
        soft_sum = torch.zeros((), device=device)
        mib_sum  = torch.zeros((), device=device)
        clu_sum  = torch.zeros((), device=device)
        bnd_sum  = torch.zeros((), device=device)
        ovr_sum  = torch.zeros((), device=device)   # v4.0
        n_batches = 0
        t_start = time.time()

        pbar = tqdm(train_loader, desc="Epoch {}/{}{}".format(
            epoch, config.epochs, " [soft]" if soft_on else ""))
        for batch in pbar:
            # 注意：batch 內 tensor 由 dataloader 從 worker 傳過來，
            # 帶 pin_memory + non_blocking 可讓傳輸與計算重疊。
            # diffusion.training_loss 內部會自己 .to(device)。
            # 這裡用 AMP autocast 包住前向+loss。

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, info = diffusion.training_loss(model, batch, soft_weights=soft_w,
                                                      min_snr_gamma=min_snr_gamma,
                                                      use_self_cond=use_self_cond,
                                                      prediction_type=prediction_type)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            ema.update()

            # 累積 metric（不 .item()，留到 epoch 結束統一同步）
            loss_sum += loss.detach()
            mse_sum  += info["mse"]
            if soft_on:
                soft_sum += info["soft"]
                mib_sum  += info["mib"]
                clu_sum  += info["cluster"]
                bnd_sum  += info["boundary"]
                ovr_sum  += info["overlap"]   # v4.0
            n_batches += 1
            global_step += 1

            # tqdm 預設會顯示「it/s」速度，已經夠用。
            # 不再每 N batch 同步 loss.item()——這會強制 CPU/GPU sync 打斷 AMP pipeline。
            # 完整的 loss 數值會在 epoch 結束時統一 .item() 印出（見下方）。

        scheduler.step()
        nb = max(n_batches, 1)
        # 統一同步：epoch 結束才 .item()
        epoch_loss = (loss_sum / nb).item()
        epoch_mse  = (mse_sum / nb).item()
        epoch_soft = (soft_sum / nb).item()
        epoch_mib  = (mib_sum / nb).item()
        epoch_cluster = (clu_sum / nb).item()
        epoch_boundary = (bnd_sum / nb).item()
        epoch_overlap = (ovr_sum / nb).item()   # v4.0
        elapsed = time.time() - t_start

        val_loss = validate(model, diffusion, val_loader, device, use_amp, use_self_cond,
                            prediction_type)
        cur_lr = scheduler.get_last_lr()[0]

        print(
            "Epoch {:3d} | train={:.4f} mse={:.4f} soft={:.4f} "
            "(mib={:.4f} clu={:.4f} bnd={:.4f} ovr={:.4f}) "
            "| val={:.4f} | lr={:.2e} | {:.0f}s{}".format(
                epoch, epoch_loss, epoch_mse, epoch_soft,
                epoch_mib, epoch_cluster, epoch_boundary, epoch_overlap,
                val_loss, cur_lr, elapsed,
                " [soft]" if soft_on else ""))

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # In-memory history（取代 csv），畫 loss curve 用
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["mse"].append(epoch_mse)
        history["soft"].append(epoch_soft)
        history["mib"].append(epoch_mib)
        history["cluster"].append(epoch_cluster)
        history["boundary"].append(epoch_boundary)
        history["overlap"].append(epoch_overlap)   # v4.0
        history["lr"].append(cur_lr)
        history["elapsed_sec"].append(elapsed)
        history["soft_on"].append(soft_on)

        if is_best or epoch % config.save_interval == 0 or epoch == config.epochs:
            ema.apply_shadow()
            tag = "best" if is_best else "epoch{}".format(epoch)
            path = os.path.join(config.output_dir, "model_{}_overlap_v11.pt".format(tag))
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
            }, path)
            print("  -> Saved {}".format(path))
            ema.restore()

            ok = plot_loss_curve(history, loss_png_path,
                                 title="Training curves (epoch {}/{})".format(
                                     epoch, config.epochs))
            if ok:
                print("  -> Updated {}".format(loss_png_path))

    print("Done! Best val loss: {:.4f}".format(best_val_loss))
    print("=" * 60)


@torch.no_grad()
def validate(model, diffusion, val_loader, device, use_amp=False, use_self_cond=False,
             prediction_type="epsilon"):
    """validation 只看去噪 MSE（不加 soft loss）。"""
    model.eval()
    total_loss = torch.zeros((), device=device)
    n = 0
    for batch in val_loader:
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, info = diffusion.training_loss(model, batch, soft_weights=None,
                                                  use_self_cond=use_self_cond,
                                                  prediction_type=prediction_type)
        total_loss += info["mse"]
        n += 1
    model.train()
    return (total_loss / max(n, 1)).item()


if __name__ == "__main__":
    # v5.29（packing density soft loss，`lambda_area`）決定不繼續投入，
    # 使用者改為專注在推論端方向——沒有實際跑過訓練，機制保留在
    # config.py／diffusion.py／train.py（`lambda_area` 預設 0.0，關閉），
    # 之後如果要重啟這個實驗，接線都還在。目前沒有排定中的訓練實驗，
    # 用 Config() 預設值。
    config = Config()
    train(config)