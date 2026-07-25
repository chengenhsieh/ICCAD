"""
Inference Pipeline v3 — 用沒看過的 training 資料 + optimal 對比

v2 -> v3 變動：
  1. main() 改用 index >= 50000 的 training 資料（模型訓練時只用前 50000 筆）
  2. block_features 改為 13 維（含正確的 boundary bits / group flags）
  3. 傳 group_bias 給 model
  4. 印出 optimal 的 area / b2b / p2b HPWL（從 metrics 取）與 inference 對比
  5. 存 optimal vs inference 並排對照圖
  6. 計算並印出 soft constraint violations
"""
import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from model import FloorplanDiffusionModel
from diffusion import GaussianDiffusion
from utils import (
    state_to_xywh,
    compute_hpwl_vectorized,
    compute_p2b_hpwl,
    total_overlap,
    count_overlaps,
    compute_soft_violations,
    legalize_lff,
    hard_zero_overlap,
)


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = FloorplanDiffusionModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Loaded model from {} (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    return model, config


def _decode_boundary_bits(code):
    code = int(code)
    return (1.0 if code & 1 else 0.0, 1.0 if code & 2 else 0.0,
            1.0 if code & 4 else 0.0, 1.0 if code & 8 else 0.0)


@torch.no_grad()
def generate_floorplan(
    model, config, areas, W_int,
    canvas_w=100.0, canvas_h=100.0,
    x_offset=0.0, y_offset=0.0,   # v3.1: canvas 左下角的絕對座標（用於對齊 pin bbox）
    n_samples=1, ddim_steps=50, device="cpu",
    constraints=None,        # (k, 5) raw constraint values（未 clip）
    p2b_edges=None,
    pins_pos=None,
    gt_w=None, gt_h=None,
    gt_x=None, gt_y=None,
    sampler="ddim",          # v4.2: "ddim" | "edm"
    edm_steps=50,            # v4.2: EDM 步數（EDM 每步 2 次 forward，50 步 = 100 forward）
    use_amp=False,           # v4.3: model forward 用 fp16 autocast（實驗用，預設關閉）
    post_repel_steps=30,     # v4.3: diffusion 結束後純物理 repel 步數（實驗用，legalize 已有壓縮）
    scale_t_windows=False,   # v4.3: force-guidance 的 t 窗口是否照 ddim_steps 等比例縮放（A/B 驗證無效，預設關閉）
    pin_force_strength=0.02,
    # v5.0: 100 樣本 quasi-paired 掃描（同一個 sample idx 固定 torch seed，
    # 讓不同力設定至少共用同一組初始噪聲，逼近 legalize 端實驗用的 paired
    # 設計）後改為預設開啟。三個力個別掃描都指向「原本寫死的強度偏強、蓋過
    # 模型自己學到的訊號」：grouping_force 從 0.015 加到 0.030 讓 V_grouping
    # 單調改善（359→355，再加到 0.050 又惡化，甜蜜點在 0.030）；
    # repulsion_strength 從 0.05 減半到 0.025 則是 area/hpwl/V_relative 同時
    # 變好（0.1092→0.1046）；boundary_nudge_strength 加倍到 0.10 讓
    # V_boundary 反而變差（116→122），減半到 0.025 才是對的方向。三個各自
    # 最佳值合在一起測（不是簡單相加，見下方數字），area_gap/hpwl_gap 完全
    # 持平，V_relative 從 0.1092 降到 0.1032（主要來自 V_grouping
    # 359→339），換算官方 cost 公式淨效益約 -1.26%，比任何單一個力的效果都
    # 好。之後在這組最佳值附近（固定另外兩個力）再細掃一輪：grouping_force
    # 在 0.030 兩側都變差，維持原值；boundary_nudge 在 0.025~0.0375 之間打平
    # 、維持原值；repulsion_strength 卻在 0.025 兩側（0.0125、0.0375）都更好
    # ——換兩組不同的 random seed 各自跑 100 樣本獨立確認，0.0375 都比 0.025
    # 好（cost 公式估計 -0.2%~-0.3%），改善來源在兩次測試中不太一樣
    # （一次主要是 V_grouping、一次主要是 V_boundary），效應本身不大、但
    # 方向穩定，改為新預設。
    grouping_force_strength=0.030,
    boundary_nudge_strength=0.025,
    repulsion_strength=0.0375,
):
    """
    Args:
        canvas_w, canvas_h: 目標 canvas 的長寬（要符合 floorplan 的實際形狀）
        x_offset, y_offset: canvas 左下角的絕對座標。歸一化座標 [0,1] 對應到
                            [x_offset, x_offset + canvas_w]、[y_offset, y_offset + canvas_h]。
                            設為 0 = 沿用舊行為（左下角在原點）。
    """
    k = len(areas)
    N = config.max_blocks
    canvas_area = canvas_w * canvas_h

    # -- Block features (13 dims) --
    block_features = np.zeros((N, 13), dtype=np.float32)
    block_features[:k, 0] = areas / canvas_area

    mib_group = np.zeros(N, dtype=np.int64)
    cluster_group = np.zeros(N, dtype=np.int64)
    boundary_code = np.zeros(N, dtype=np.int64)

    if constraints is not None:
        cons = np.array(constraints, dtype=np.float64)
        cons = np.where(cons < 0, 0, cons)
        for i in range(k):
            block_features[i, 1] = 1.0 if cons[i, 0] > 0.5 else 0.0   # fixed
            block_features[i, 2] = 1.0 if cons[i, 1] > 0.5 else 0.0   # preplaced
            mib_group[i]     = int(cons[i, 2])
            cluster_group[i] = int(cons[i, 3])
            boundary_code[i] = int(cons[i, 4])
            block_features[i, 3] = 1.0 if mib_group[i] > 0 else 0.0
            block_features[i, 4] = 1.0 if cluster_group[i] > 0 else 0.0
            l, r, t, b = _decode_boundary_bits(boundary_code[i])
            block_features[i, 5:9] = [l, r, t, b]

    # Pin features
    if p2b_edges is not None and pins_pos is not None:
        pins = np.array(pins_pos, dtype=np.float32)
        pin_cx = np.zeros(N, dtype=np.float32)
        pin_cy = np.zeros(N, dtype=np.float32)
        pin_tw = np.zeros(N, dtype=np.float32)
        for p_idx, b_idx, w in p2b_edges:
            p_idx, b_idx = int(p_idx), int(b_idx)
            if p_idx < 0 or b_idx < 0 or b_idx >= k or p_idx >= len(pins):
                continue
            pin_cx[b_idx] += w * (pins[p_idx, 0] - x_offset) / canvas_w
            pin_cy[b_idx] += w * (pins[p_idx, 1] - y_offset) / canvas_h
            pin_tw[b_idx] += w
        valid = pin_tw > 0
        pin_cx[valid] /= pin_tw[valid]
        pin_cy[valid] /= pin_tw[valid]
        max_tw = pin_tw.max()
        if max_tw > 0:
            pin_tw /= max_tw
        block_features[:, 9] = pin_cx
        block_features[:, 10] = pin_cy
        block_features[:, 11] = pin_tw

    conn_pad = np.zeros((N, N), dtype=np.float32)
    conn_pad[:k, :k] = W_int

    # group bias matrix（同 MIB / cluster 組）
    group_bias = np.zeros((N, N), dtype=np.float32)
    for i in range(k):
        for j in range(i + 1, k):
            same_mib = (mib_group[i] > 0 and mib_group[i] == mib_group[j])
            same_clu = (cluster_group[i] > 0 and cluster_group[i] == cluster_group[j])
            if same_mib or same_clu:
                group_bias[i, j] = group_bias[j, i] = 1.0

    mask_np = np.zeros(N, dtype=bool)
    mask_np[:k] = True

    # -- 約束 masks（hard: fixed / preplaced）--
    fixed_mask_np = np.zeros(N, dtype=bool)
    preplaced_mask_np = np.zeros(N, dtype=bool)
    gt_state_np = np.zeros((N, 3), dtype=np.float32)

    has_constraints = False
    if constraints is not None:
        cons = np.array(constraints[:k], dtype=np.float64)
        cons = np.where(cons < 0, 0, cons)
        for i in range(k):
            is_preplaced = cons[i, 1] > 0.5
            is_fixed = cons[i, 0] > 0.5
            if is_preplaced and gt_x is not None and gt_y is not None and \
               gt_w is not None and gt_h is not None:
                preplaced_mask_np[i] = True
                fixed_mask_np[i] = True
                gt_state_np[i, 0] = (gt_x[i] - x_offset) / canvas_w
                gt_state_np[i, 1] = (gt_y[i] - y_offset) / canvas_h
                gt_state_np[i, 2] = np.log(gt_w[i] / max(gt_h[i], 1e-8))
                has_constraints = True
            elif is_fixed and gt_w is not None and gt_h is not None:
                fixed_mask_np[i] = True
                gt_state_np[i, 2] = np.log(gt_w[i] / max(gt_h[i], 1e-8))
                has_constraints = True

    # -- Batch 化 --
    feats_t = torch.tensor(block_features).unsqueeze(0).expand(n_samples, -1, -1).to(device)
    conn_t = torch.tensor(conn_pad).unsqueeze(0).expand(n_samples, -1, -1).to(device)
    gb_t = torch.tensor(group_bias).unsqueeze(0).expand(n_samples, -1, -1).to(device)
    mask_t = torch.tensor(mask_np).unsqueeze(0).expand(n_samples, -1).to(device)

    if has_constraints:
        gt_state_t = torch.tensor(gt_state_np).unsqueeze(0).expand(n_samples, -1, -1).to(device)
        fixed_t = torch.tensor(fixed_mask_np).unsqueeze(0).expand(n_samples, -1).to(device)
        preplaced_t = torch.tensor(preplaced_mask_np).unsqueeze(0).expand(n_samples, -1).to(device)
    else:
        gt_state_t = fixed_t = preplaced_t = None

    diffusion = GaussianDiffusion(T=config.T, beta_start=config.beta_start,
                                  beta_end=config.beta_end)
    diffusion.to(device)

    shape = (n_samples, N, 3)

    # v3.8: 準備 force-guided sampler 需要的 batched tensors
    # 全部從 (k,) 或 (N,) 擴展成 (n_samples, N)
    mib_t = torch.tensor(mib_group, dtype=torch.long).unsqueeze(0).expand(n_samples, -1).to(device)
    grp_t = torch.tensor(cluster_group, dtype=torch.long).unsqueeze(0).expand(n_samples, -1).to(device)
    bnd_t = torch.tensor(boundary_code, dtype=torch.long).unsqueeze(0).expand(n_samples, -1).to(device)

    # areas_norm: (B, N) — block area / canvas area
    canvas_area_total = float(canvas_w * canvas_h)
    areas_np_pad = np.zeros(N, dtype=np.float32)
    areas_np_pad[:k] = areas / canvas_area_total
    areas_t = torch.tensor(areas_np_pad).unsqueeze(0).expand(n_samples, -1).to(device)

    # Pin Force：每個 block 的目標 pin 中心（normalized [0,1] coord）
    pin_targets_np = np.zeros((N, 2), dtype=np.float32)
    pin_weights_np = np.zeros(N, dtype=np.float32)
    if p2b_edges is not None and pins_pos is not None and len(p2b_edges) > 0:
        pins_np = np.asarray(pins_pos, dtype=np.float32)
        for p_idx, b_idx, w_e in p2b_edges:
            p_idx = int(p_idx); b_idx = int(b_idx); w_e = float(w_e)
            if 0 <= p_idx < len(pins_np) and 0 <= b_idx < k:
                px = (pins_np[p_idx, 0] - x_offset) / canvas_w
                py = (pins_np[p_idx, 1] - y_offset) / canvas_h
                pin_targets_np[b_idx, 0] += w_e * px
                pin_targets_np[b_idx, 1] += w_e * py
                pin_weights_np[b_idx] += w_e
        valid = pin_weights_np > 0
        pin_targets_np[valid, 0] /= pin_weights_np[valid]
        pin_targets_np[valid, 1] /= pin_weights_np[valid]
    pin_targets_t = torch.tensor(pin_targets_np).unsqueeze(0).expand(n_samples, -1, -1).to(device)
    pin_weights_t = torch.tensor(pin_weights_np).unsqueeze(0).expand(n_samples, -1).to(device)

    # Best-of-N 的 select metric：用 overlap 為主（state 階段算不出 HPWL，但能算重疊）
    def _select_metric(x_state):
        """
        對 (B, N, 3) state 估「不好」分數，低 = 好。
        近似評分：基於 areas_norm 和 log_r 推 w_norm, h_norm，
        然後算 normalized canvas 內的總重疊面積。
        """
        Bx, Nx, _ = x_state.shape
        r = torch.exp(x_state[:, :, 2]).clamp(min=0.1, max=10.0)
        w = torch.sqrt(areas_t * r)
        h = torch.sqrt(areas_t / r)
        cx, cy = x_state[:, :, 0], x_state[:, :, 1]
        cx_i, cx_j = cx[:, :, None], cx[:, None, :]
        cy_i, cy_j = cy[:, :, None], cy[:, None, :]
        w_i, w_j = w[:, :, None], w[:, None, :]
        h_i, h_j = h[:, :, None], h[:, None, :]
        ovx = torch.minimum(cx_i + w_i / 2, cx_j + w_j / 2) - \
              torch.maximum(cx_i - w_i / 2, cx_j - w_j / 2)
        ovy = torch.minimum(cy_i + h_i / 2, cy_j + h_j / 2) - \
              torch.maximum(cy_i - h_i / 2, cy_j - h_j / 2)
        m_pair = mask_t[:, :, None] * mask_t[:, None, :]
        eye = torch.eye(Nx, device=device).unsqueeze(0)
        m_pair = m_pair * (1 - eye)
        overlap = (ovx.clamp(min=0) * ovy.clamp(min=0)) * m_pair
        return overlap.sum(dim=(1, 2))    # (B,)

    # v3.9: clamp_bbox = pin bbox 在 normalized 座標下的範圍。
    # slack=1.10 的 canvas 設計下，pin bbox 大約落在 [~0.045, ~0.955] 之間。
    # 把它當邊界，套力後 clamp block 中心，避免被推出 pin 範圍。
    if pins_pos is not None and len(pins_pos) >= 2:
        pins_arr = np.asarray(pins_pos, dtype=np.float32)
        px_min, px_max = float(pins_arr[:, 0].min()), float(pins_arr[:, 0].max())
        py_min, py_max = float(pins_arr[:, 1].min()), float(pins_arr[:, 1].max())
        clamp_bbox_norm = (
            (px_min - x_offset) / canvas_w,
            (py_min - y_offset) / canvas_h,
            (px_max - x_offset) / canvas_w,
            (py_max - y_offset) / canvas_h,
        )
    else:
        clamp_bbox_norm = None

    # v4.2: 二選一 sampler
    if sampler == "edm":
        # 修正：v5.0 調過的力強度只有 threading 進 DDIM 分支，這裡漏了——
        # edm_sample_with_forces 有自己的一份舊、v5.0 調參前的預設值
        # （grouping=0.015/boundary=0.05/repulsion=0.05），不補上就是拿
        # 「DDIM 用新參數 vs EDM 用舊參數」在比，對 EDM 不公平。
        generated = diffusion.edm_sample_with_forces(
            model, shape, feats_t, conn_t, mask_t,
            num_steps=edm_steps,
            gt_state=gt_state_t, fixed_mask=fixed_t, preplaced_mask=preplaced_t,
            group_bias=gb_t,
            mib_group=mib_t,
            grouping_group=grp_t,
            boundary_code=bnd_t,
            pin_targets=pin_targets_t,
            pin_weights=pin_weights_t,
            areas_norm=areas_t,
            n_renoise_ratio=0.7,
            select_metric_fn=_select_metric,
            post_repel_steps=post_repel_steps,
            clamp_bbox=clamp_bbox_norm,
            pin_force_strength=pin_force_strength,
            grouping_force_strength=grouping_force_strength,
            boundary_nudge_strength=boundary_nudge_strength,
            repulsion_strength=repulsion_strength,
        )
    else:  # sampler == "ddim"
        # v4.3 實驗：曾嘗試把 mib_clamp_until_t / pin_force_until_t /
        # grouping_until_t / repulsion_from_t / boundary_from_t（寫死在 t
        # 空間的窗口）照 ddim_steps 縮小的比例等比例放大，理論上能讓這些
        # 「後段」機制在稀疏排程下摸到跟原本調參時差不多數量的 step。100
        # 樣本 A/B 結果：raw/legalized 兩邊指標幾乎沒有差異（在雜訊範圍
        # 內），推測是 post_repel_steps（不受 ddim_steps 影響、獨立於 loop
        # 外）跟 legalize 自己的 boundary 處理已經覆蓋掉這些機制的邊際貢獻。
        # 驗證後維持不縮放（scale_t_windows 預設 False），避免留下沒有實際
        # 效果的複雜度。
        _t_scale = (100.0 / max(ddim_steps, 1)) if scale_t_windows else 1.0
        generated = diffusion.ddim_sample_with_forces(
            model, shape, feats_t, conn_t, mask_t,
            ddim_steps=ddim_steps, eta=0.0,
            gt_state=gt_state_t, fixed_mask=fixed_t, preplaced_mask=preplaced_t,
            group_bias=gb_t,
            mib_group=mib_t,
            grouping_group=grp_t,
            boundary_code=bnd_t,
            pin_targets=pin_targets_t,
            pin_weights=pin_weights_t,
            areas_norm=areas_t,
            n_renoise_steps=int(ddim_steps * 0.7),    # 70% 處 best-of-N
            select_metric_fn=_select_metric,
            post_repel_steps=post_repel_steps,          # v3.9: 50 → 30
            clamp_bbox=clamp_bbox_norm,
            use_amp=use_amp,
            mib_clamp_until_t=int(20 * _t_scale),
            pin_force_until_t=int(20 * _t_scale),
            grouping_until_t=int(30 * _t_scale),
            repulsion_from_t=int(50 * _t_scale),
            boundary_from_t=int(20 * _t_scale),
            pin_force_strength=pin_force_strength,
            grouping_force_strength=grouping_force_strength,
            boundary_nudge_strength=boundary_nudge_strength,
            repulsion_strength=repulsion_strength,
        )

    preplaced_indices = [i for i in range(k) if preplaced_mask_np[i]]

    results = []
    for s in range(n_samples):
        state = generated[s, :k].cpu().numpy()
        x, y, w, h = state_to_xywh(state, areas, canvas_w, canvas_h)
        # v3.1: state_to_xywh 出來的是「以原點為左下角」的座標，
        # 加上 offset 對齊到 pin bbox（或其他外部指定的）左下角。
        x = x + x_offset
        y = y + y_offset

        # 強制 fixed-shape blocks 的精確尺寸
        for i in range(k):
            if fixed_mask_np[i] and gt_w is not None and gt_h is not None:
                w[i] = gt_w[i]; h[i] = gt_h[i]
        # 強制 preplaced blocks 的精確位置
        for i in preplaced_indices:
            if gt_x is not None and gt_y is not None:
                x[i] = gt_x[i]; y[i] = gt_y[i]

        # v3.2: 不做 legalize。這個 best 是要傳給下游 legalization 步驟的，
        # 在這裡先 legalize 等於做兩次、會破壞 diffusion 學到的相對位置。
        # 所有指標都從 diffusion 原始輸出計算。

        b2b_hpwl = compute_hpwl_vectorized(x, y, w, h, W_int)
        p2b_hpwl = 0.0
        if p2b_edges is not None and pins_pos is not None:
            p2b_hpwl = compute_p2b_hpwl(x, y, w, h, p2b_edges, pins_pos)

        overlap = total_overlap(x, y, w, h)
        n_overlaps_count = count_overlaps(x, y, w, h)
        bbox_w = np.max(x + w) - np.min(x)
        bbox_h = np.max(y + h) - np.min(y)
        bbox_area = bbox_w * bbox_h

        soft = compute_soft_violations(x, y, w, h,
                                       mib_group[:k], cluster_group[:k], boundary_code[:k])

        results.append({
            "x": x, "y": y, "w": w, "h": h,
            "b2b_hpwl": b2b_hpwl, "p2b_hpwl": p2b_hpwl,
            "total_hpwl": b2b_hpwl + p2b_hpwl,
            "overlap": overlap, "n_overlaps": n_overlaps_count,
            "bbox_area": bbox_area, "sample_idx": s,
            "soft": soft,
        })

    # v3.2 排序鍵：(total_overlap, V_relative, total_hpwl, bbox_area)
    #
    # 為什麼這樣排：
    # - 目標是讓下游 legalize 好做，所以「總重疊面積越小越好」
    #   （比 n_overlaps 更細：能區分擦邊 vs 嚴重重疊，後者 legalize 會把好的相對位置推壞）
    # - V_relative 是 soft constraint 違規率（[0,1]），legalize 通常不會改善這個，
    #   所以這裡盡量挑低的
    # - 同樣品質下挑 HPWL 較短的（線長）
    # - 最後比 bbox_area（緊湊度）
    # 不用 optimal 當基準，因為實際比賽/部署時拿不到。
    results.sort(key=lambda r: (
        r["overlap"],
        r["soft"]["V_relative"],
        r["total_hpwl"],
        r["bbox_area"],
    ))
    return results[0], results


def _count_official_violations(x, y, w, h):
    """照官方 check_overlap 的逐 pair、逐軸判定數違規數（純 python，給小規模驗證用）。"""
    n_off = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            ox = min(x[i] + w[i], x[j] + w[j]) - max(x[i], x[j])
            oy = min(y[i] + h[i], y[j] + h[j]) - max(y[i], y[j])
            if ox > 1e-6 and oy > 1e-6:
                n_off += 1
    return n_off


def _guarantee_zero_overlap(x, y, w, h, preplaced_idx_list):
    """
    保證回傳結果通過官方 check_overlap（逐 pair、逐軸判定）。三層防線：
      1. hard_zero_overlap（迭代收斂 + eject fallback）
      2. 驗證 + 加大 margin 重試最多 5 次（極少數大座標量級下 margin 可能被
         float32 精度吃掉，或 eject fallback 邊界情況）
      3. 絕對保底：仍未過關就強制把違規的可動一方搬到全域範圍外，保證
         100% 過關（除非兩個違規 block 都是 preplaced，那是資料本身的衝突）
    """
    preplaced_idx_list = list(preplaced_idx_list)
    x, y = hard_zero_overlap(x, y, w, h, preplaced_indices=preplaced_idx_list)

    for retry in range(5):
        if _count_official_violations(x, y, w, h) == 0:
            break
        x, y = hard_zero_overlap(x, y, w, h, preplaced_indices=preplaced_idx_list,
                                  margin=1e-3 * (10 ** retry))

    if _count_official_violations(x, y, w, h) != 0:
        frozen_set = set(preplaced_idx_list)
        n_final = len(x)
        eject_x = float((x + w).max()) + 1.0
        eject_y = float(y.min())
        n_unresolvable = 0
        for i in range(n_final):
            for j in range(i + 1, n_final):
                ox = min(x[i] + w[i], x[j] + w[j]) - max(x[i], x[j])
                oy = min(y[i] + h[i], y[j] + h[j]) - max(y[i], y[j])
                if ox > 1e-6 and oy > 1e-6:
                    target = j if j not in frozen_set else (i if i not in frozen_set else None)
                    if target is None:
                        n_unresolvable += 1
                        continue
                    x[target] = eject_x
                    y[target] = eject_y
                    eject_y += h[target] + 1.0
        if n_unresolvable > 0:
            print("ERROR: {} overlap pair(s) between two preplaced blocks — "
                  "cannot resolve by repositioning (data-level conflict).".format(n_unresolvable))
        else:
            print("WARNING: hard_zero_overlap needed the final forced-eject safety net "
                  "(5 retries were not enough). Result is still guaranteed overlap-free.")

    return x, y


def legalize_result(
    best, areas, W_int, p2b_edges, pins_pos,
    preplaced_mask, fixed_mask,
    mib_group, cluster_group, boundary_code,
    outline_bbox=None,
    allow_reshape=True,
    reinsert_sweeps=3,
    reinsert_grid_density=12,
    tie_break_modes=None,
    use_gravity=False,
    gravity_iters=40,
    # v4.7: 100 樣本 paired A/B（同一組 diffusion 輸出餵給不同 legalize 設定，
    # 排除取樣雜訊）驗證後改為預設開啟。compact_merge_cluster_groups 直接針對
    # 每個 cluster group 的連通性做剛體貼合，並用 hpwl_slack_ratio 控制「最多
    # 願意用多少額外 HPWL 換一個 grouping 違規」——paired 測試下 area_gap
    # 100/100 樣本零變化、V_grouping 從未在任何樣本變差（29/100 樣本變好），
    # 平均 hpwl_gap 代價僅 +0.16%（換算官方 cost 公式約 -1.3%，比 slack=0 的
    # 嚴格零代價版本更好）。見 compact_merge_cluster_groups docstring 與呼叫處
    # 說明（utils.py）。
    use_cluster_merge=True,
    hpwl_slack_ratio=5.0,
    use_snap_boundary=False,   # 實驗用：見 compact_snap_boundary 呼叫處說明（utils.py）
    boundary_hpwl_slack_ratio=0.0,   # 實驗用：見 compact_snap_boundary docstring
    use_pair_reinsert=False,  # 實驗用：見 compact_pair_reinsert 呼叫處說明（utils.py）
    pair_reinsert_sweeps=2,
    pair_reinsert_grid_density=8,
    pair_reinsert_hpwl_slack_ratio=0.0,
    use_reinsert_reshape=False,  # 實驗用：見 compact_reinsert_reshape 呼叫處說明（utils.py）
    reinsert_reshape_sweeps=2,
    reinsert_reshape_grid_density=8,
    # v5.2（實驗、最終停用）：一度因為修正 warmup 量測誤差後看起來有淨改善
    # 而短暫改為預設開啟，後來發現 compact_gradient_finetune 的 loss 對
    # 「整體平移」完全不變，導致 Adam 在這個零梯度方向上隨機漂移、把 block
    # 帶出 outline_bbox 之外——連當初的展示案例（-4.54%）事後查也是越界的
    # 無效解。補上錨定項＋outline 硬 gate 修好之後，同一批樣本重測變成
    # 0/30 有真正改善，時間成本卻還在，因此改回預設關閉（詳見 utils.py
    # legalize_lff 呼叫處與 compact_gradient_finetune docstring 的完整說明）。
    use_gradient_finetune=False,
    gradient_finetune_steps=400,
    gradient_finetune_lr=0.5,
    gradient_finetune_patience=30,
    gradient_finetune_hpwl_slack_ratio=0.0,
    use_second_merge_pass=True,
    weight_dist=1.0,
    weight_boundary=3.0,
    weight_cluster=1.0,
    weight_b2b=0.5,
    weight_p2b=0.15,
    weight_shape=3.0,
    use_cluster_adjacency=False,
    cluster_adjacency_bonus=5.0,
    verbose=True,
):
    """
    對 generate_floorplan() 選出的 raw best 做 legalize 後處理，保證零 overlap、
    保證 block 落在 outline_bbox（例如 pin bbox）裡頭。回傳跟 generate_floorplan
    結果同 schema 的 dict，可直接沿用 evaluate_and_report / build_solution_entry。

    v4：改用 legalize_lff（見 utils.py 的說明）——LFF 風格、決定性單趟的自由
    矩形（MAXRECTS）排布，不再是 legalize_v2 的「anchor 網格搜尋 + 一堆事後
    補丁清重疊/壓緊密度」。overlap-free、preplaced/fixed/面積 hard constraint、
    outline 包含都是演算法結構上直接保證。

    tie_break_modes: 實驗用。None（預設）= 跟以前完全一樣，只用 'area_desc'
    跑一次。傳入 list（例如 ['area_desc','area_asc','flexibility']）時，會
    對同一個 raw best 各自完整跑一次 legalize_lff（含 compact_reinsert 等
    後處理），取 bbox_area 最小的當最終結果——因為 LFF 是決定性單趟貪婪排布
    ，同一組 block 用不同的放置順序，最終 bounding box 可能不同，這裡等於
    是 legalize 版的 best-of-N。注意這是真的跑 N 次完整 legalize（不像
    diffusion 的候選是同一個 GPU batch），時間成本跟 len(tie_break_modes)
    成正比，不是免費的。
    """
    modes = tie_break_modes if tie_break_modes else ["area_desc"]

    def _run_once(mode):
        xa, ya, wa, ha = legalize_lff(
            x_init=best["x"], y_init=best["y"],
            w_init=best["w"], h_init=best["h"],
            areas=areas,
            preplaced_mask=preplaced_mask,
            fixed_mask=fixed_mask,
            mib_group=mib_group,
            cluster_group=cluster_group,
            boundary_code=boundary_code,
            outline_bbox=outline_bbox,
            W_int=W_int,
            p2b_edges=p2b_edges,
            pins_pos=pins_pos,
            allow_reshape=allow_reshape,
            reinsert_sweeps=reinsert_sweeps,
            reinsert_grid_density=reinsert_grid_density,
            tie_break_mode=mode,
            use_gravity=use_gravity,
            gravity_iters=gravity_iters,
            use_cluster_merge=use_cluster_merge,
            hpwl_slack_ratio=hpwl_slack_ratio,
            use_snap_boundary=use_snap_boundary,
            boundary_hpwl_slack_ratio=boundary_hpwl_slack_ratio,
            use_pair_reinsert=use_pair_reinsert,
            pair_reinsert_sweeps=pair_reinsert_sweeps,
            pair_reinsert_grid_density=pair_reinsert_grid_density,
            pair_reinsert_hpwl_slack_ratio=pair_reinsert_hpwl_slack_ratio,
            use_reinsert_reshape=use_reinsert_reshape,
            reinsert_reshape_sweeps=reinsert_reshape_sweeps,
            reinsert_reshape_grid_density=reinsert_reshape_grid_density,
            use_gradient_finetune=use_gradient_finetune,
            gradient_finetune_steps=gradient_finetune_steps,
            gradient_finetune_lr=gradient_finetune_lr,
            gradient_finetune_patience=gradient_finetune_patience,
            gradient_finetune_hpwl_slack_ratio=gradient_finetune_hpwl_slack_ratio,
            use_second_merge_pass=use_second_merge_pass,
            weight_dist=weight_dist,
            weight_boundary=weight_boundary,
            weight_cluster=weight_cluster,
            weight_b2b=weight_b2b,
            weight_p2b=weight_p2b,
            weight_shape=weight_shape,
            use_cluster_adjacency=use_cluster_adjacency,
            cluster_adjacency_bonus=cluster_adjacency_bonus,
            verbose=(mode == modes[0]) and verbose,
        )
        # legalize_lff 內部已經呼叫過 hard_zero_overlap 做防禦性驗證，這裡再走
        # 一次完整的 _guarantee_zero_overlap（含重試 + 絕對保底 eject）當雙重
        # 保險，確保「hard constraints 100% 不違背」這個要求無論如何都成立。
        pp_idx = [i for i in range(len(xa)) if preplaced_mask[i]]
        xa, ya = _guarantee_zero_overlap(xa, ya, wa, ha, pp_idx)
        bbox_area_a = (np.max(xa + wa) - np.min(xa)) * (np.max(ya + ha) - np.min(ya))
        return xa, ya, wa, ha, bbox_area_a

    candidates = [_run_once(m) for m in modes]
    x, y, w, h, _ = min(candidates, key=lambda c: c[4])
    if len(modes) > 1:
        print("legalize_result: tie_break candidates bbox_area = {} -> picked {:.1f}".format(
            ["{:.1f}".format(c[4]) for c in candidates], min(c[4] for c in candidates)))

    b2b_hpwl = compute_hpwl_vectorized(x, y, w, h, W_int)
    p2b_hpwl = 0.0
    if p2b_edges is not None and pins_pos is not None:
        p2b_hpwl = compute_p2b_hpwl(x, y, w, h, p2b_edges, pins_pos)

    overlap = total_overlap(x, y, w, h)
    n_overlaps_count = count_overlaps(x, y, w, h)
    bbox_w = np.max(x + w) - np.min(x)
    bbox_h = np.max(y + h) - np.min(y)
    bbox_area = bbox_w * bbox_h

    soft = compute_soft_violations(x, y, w, h, mib_group, cluster_group, boundary_code)

    return {
        "x": x, "y": y, "w": w, "h": h,
        "b2b_hpwl": b2b_hpwl, "p2b_hpwl": p2b_hpwl,
        "total_hpwl": b2b_hpwl + p2b_hpwl,
        "overlap": overlap, "n_overlaps": n_overlaps_count,
        "bbox_area": bbox_area,
        "soft": soft,
    }


def evaluate_and_report(result, W_int, areas, constraints=None, optimal=None, stage=""):
    k = len(areas)
    print("=" * 64)
    print("Floorplan Evaluation Report" + (" — {}".format(stage) if stage else ""))
    print("=" * 64)
    print("  Blocks:        {}".format(k))
    print("  --- Inference ---")
    print("  B2B HPWL:      {:.2f}".format(result['b2b_hpwl']))
    print("  P2B HPWL:      {:.2f}".format(result['p2b_hpwl']))
    print("  Total HPWL:    {:.2f}".format(result['total_hpwl']))
    print("  Bbox area:     {:.2f}".format(result['bbox_area']))
    print("  Total overlap: {:.4f}".format(result['overlap']))
    print("  Overlap pairs: {}".format(result['n_overlaps']))

    if optimal is not None:
        print("  --- Optimal (GT) ---")
        print("  B2B HPWL:      {:.2f}".format(optimal.get('b2b_hpwl', float('nan'))))
        print("  P2B HPWL:      {:.2f}".format(optimal.get('p2b_hpwl', float('nan'))))
        print("  Total HPWL:    {:.2f}".format(optimal.get('total_hpwl', float('nan'))))
        print("  Bbox area:     {:.2f}".format(optimal.get('bbox_area', float('nan'))))
        print("  --- Gap (inference vs optimal) ---")
        if optimal.get('total_hpwl', 0) > 0:
            hpwl_gap = (result['total_hpwl'] - optimal['total_hpwl']) / optimal['total_hpwl']
            print("  HPWL gap:      {:+.2%}".format(hpwl_gap))
        if optimal.get('bbox_area', 0) > 0:
            area_gap = (result['bbox_area'] - optimal['bbox_area']) / optimal['bbox_area']
            print("  Area gap:      {:+.2%}".format(area_gap))

    actual_areas = result["w"] * result["h"]
    area_errors = np.abs(actual_areas - areas) / areas
    print("  --- Hard constraint check ---")
    print("  Max area err:  {:.6f}".format(area_errors.max()))

    s = result["soft"]
    print("  --- Soft violations ---")
    print("  V_relative:    {:.4f}  (boundary={}, grouping={}, mib={}, N_soft={})".format(
        s["V_relative"], s["V_boundary"], s["V_grouping"], s["V_mib"], s["N_soft"]))

    feasible = (result["n_overlaps"] == 0) and (area_errors.max() <= 0.01)
    print("  Feasible:      {}".format("YES" if feasible else "NO"))
    print("=" * 64)


def build_solution_entry(
    sample_idx, areas, area_tolerance,
    result_x, result_y, result_w, result_h,
    canvas_bbox,                   # (x_min, y_min, x_max, y_max)
    canvas_source,                 # 字串，例如 "pin_bbox"
    pins_pos, b2b_edges, p2b_edges,
    mib_group, cluster_group, boundary_code,
    fixed_mask_per_block,          # (k,) bool
    preplaced_mask_per_block,      # (k,) bool
    optimal_metrics=None,          # v3.5: dict {area, b2b_hpwl, p2b_hpwl} 或 None
    actual_metrics=None,           # v3.7: dict {area, b2b_hpwl, p2b_hpwl} - inference 端算好的
):
    """
    依照官方範本格式 + p2b 擴充，建立單一 sample 的 solution dict。

    schema（在原範本之上加 p2b_nets）：
      test_id, block_count, positions [[x,y,w,h]], area_target [], area_tolerance,
      canvas {x_min,y_min,x_max,y_max,width,height,source},
      preplaced [bool], fixed_shape [bool], boundary [int bitmask],
      mib [group_id], grouping [group_id],
      pins [[x, y]],
      nets [[block_i, block_j, weight]],       ← b2b (block-to-block)
      p2b_nets [[pin_idx, block_idx, weight]]  ← p2b (pin-to-block, 範本擴充)

    主要差異於範本原版：
      - p2b_nets 補上（範本只有 b2b，下游 cost 計算 HPWL_ext 需要 p2b）
      - 多 sample 共享同一個 JSON 容器
      - 每個 per-block constraint 都是 list，不是 group→ids 巢狀字典
    """
    k = len(areas)
    x_min, y_min, x_max, y_max = canvas_bbox

    # ---- positions: [[x, y, w, h], ...] ----
    positions = []
    for i in range(k):
        positions.append([
            float(result_x[i]), float(result_y[i]),
            float(result_w[i]), float(result_h[i]),
        ])

    # ---- area_target ----
    area_target_list = [float(a) for a in areas]

    # ---- canvas ----
    canvas_dict = {
        "x_min": float(x_min),
        "y_min": float(y_min),
        "x_max": float(x_max),
        "y_max": float(y_max),
        "width": float(x_max - x_min),
        "height": float(y_max - y_min),
        "source": str(canvas_source),
    }

    # ---- per-block constraint lists ----
    preplaced_list = [bool(preplaced_mask_per_block[i]) for i in range(k)]
    fixed_shape_list = [bool(fixed_mask_per_block[i]) for i in range(k)]
    boundary_list = [int(boundary_code[i]) for i in range(k)]
    mib_list = [int(mib_group[i]) for i in range(k)]
    grouping_list = [int(cluster_group[i]) for i in range(k)]

    # ---- pins: [[x, y], ...] ----
    pins_list = []
    if pins_pos is not None:
        for (px, py) in pins_pos:
            pins_list.append([float(px), float(py)])

    # ---- nets: [[i, j, weight], ...]  b2b only ----
    # 對於重複 edge (i,j) 出現多次的情況，採用 last-write-wins 語意，
    # 與 inference.py 構造 W_int 的方式一致（W_int[i,j] = w; W_int[j,i] = w 是覆寫），
    # 否則 HTML 顯示的 HPWL 會跟 terminal 用 W_int 算出來的對不起來。
    b2b_dict = {}   # (i_min, i_max) -> weight；後寫覆蓋前寫
    if b2b_edges is not None:
        for edge in b2b_edges:
            i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
            if i < 0 or j < 0 or i >= k or j >= k or i == j:
                continue
            key = (min(i, j), max(i, j))
            b2b_dict[key] = w
    nets_list = [[i, j, w] for (i, j), w in b2b_dict.items()]

    # ---- p2b_nets: [[pin_idx, block_idx, weight], ...]  (擴充欄位) ----
    p2b_nets_list = []
    if p2b_edges is not None:
        n_pins_total = len(pins_list)
        for edge in p2b_edges:
            p_idx, b_idx, w = int(edge[0]), int(edge[1]), float(edge[2])
            if p_idx < 0 or b_idx < 0 or b_idx >= k:
                continue
            if n_pins_total > 0 and p_idx >= n_pins_total:
                continue
            p2b_nets_list.append([p_idx, b_idx, w])

    entry = {
        "test_id": int(sample_idx),
        "block_count": int(k),
        "positions": positions,
        "area_target": area_target_list,
        "area_tolerance": float(area_tolerance),
        "canvas": canvas_dict,
        "preplaced": preplaced_list,
        "fixed_shape": fixed_shape_list,
        "boundary": boundary_list,
        "mib": mib_list,
        "grouping": grouping_list,
        "pins": pins_list,
        "nets": nets_list,
        "p2b_nets": p2b_nets_list,
    }

    # v3.5: optimal_metrics 為 viewer 計算 area/HPWL gap 用的對照基準。
    # 非標準擴充欄位，下游 legalize 可以忽略。validation 沒提供 metrics 就不寫。
    if optimal_metrics is not None:
        entry["optimal_metrics"] = {
            k_: float(v_) for k_, v_ in optimal_metrics.items()
        }

    # v3.7: actual_metrics = inference 端直接用 utils 算出的 area/HPWL，
    # 避免 HTML 端重算公式不一致。HTML viewer 直接讀這個顯示，
    # 不再自己跑 HPWL/area 公式。下游 legalize 也可選擇忽略。
    if actual_metrics is not None:
        entry["actual_metrics"] = {
            k_: float(v_) for k_, v_ in actual_metrics.items()
        }

    return entry


def write_solutions_json(save_path, solutions, submission_tag="diffusion"):
    """
    把多個 solution entry 包成範本格式的容器 JSON 並寫檔。
    """
    from datetime import datetime
    payload = {
        "submission": submission_tag,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "coordinate_frame": "raw_unshifted",
        "schema": {
            "canvas": "pin-defined bbox {x_min,y_min,x_max,y_max,width,height}; "
                      "source=pin_bbox (NOT the true die outline — known frame mismatch)",
            "positions": "[x_ll, y_ll, w, h] per block, raw units",
            "hard_constraints": "overlap-free; preplaced[i]=locked x,y,w,h; "
                                "fixed_shape[i]=locked w,h; "
                                "|area-area_target|<=area_tolerance",
            "soft_constraints": "boundary[i] code L=1,R=2,T=4,B=8 (>=1 flagged edge on layout bbox); "
                                "mib[i] MIB group id (0=none, 1..Q; same id => identical w,h required); "
                                "grouping[i] group id (0=none, 1..P; same id => blocks must ABUT into "
                                "one connected component).",
            "nets": "b2b netlist [block_i, block_j, weight] for HPWL (inter-module).",
            "p2b_nets": "p2b netlist [pin_idx, block_idx, weight] for HPWL (external; pin index into pins[]).",
            "optimal_metrics": "Optional {area, b2b_hpwl, p2b_hpwl} from validation GT, for viewer gap display. Non-standard; downstream may ignore.",
            "actual_metrics": "Optional {area, b2b_hpwl, p2b_hpwl} computed at inference time using utils functions; viewer reads these directly. Non-standard; downstream may ignore.",
        },
        "solutions": solutions,
    }
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("Saved {} solution(s) to {}".format(len(solutions), save_path))
    return payload   # 給 HTML 寫檔重用，不必再讀回 JSON


def write_solutions_html(save_path, payload, template_path=None):
    """
    把 solutions payload 嵌進 viewer HTML template，產出 self-contained HTML。
    雙擊即可在瀏覽器看到視覺化結果。

    template_path: viewer 模板路徑。預設找 inference.py 同目錄下的
                   layout_viewer_template.html。找不到就跳過 HTML 輸出。
    """
    if template_path is None:
        template_path = os.path.join(os.path.dirname(__file__),
                                     "layout_viewer_template.html")
    if not os.path.exists(template_path):
        print("WARNING: viewer template not found at {}, skip HTML output".format(
            template_path))
        return False

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # 把資料當成 JSON 字串塞進 <script> 標籤。
    # 注意：JSON 字串可能含 "</script>"（極少見但有可能），用 escape 規避。
    data_json = json.dumps(payload, ensure_ascii=False)
    data_json = data_json.replace("</", "<\\/")   # 安全處理

    inject = "<script>window.PRELOADED_DATA = " + data_json + ";</script>\n"

    # 必須注入到 viewer 主 script 「之前」，PRELOADED_DATA 才會在它執行時存在。
    # template 有兩個 <script> 標籤（一個在 <head>、一個在 <body>），
    # 我們插在第一個 <script> 之前最保險。
    script_idx = template.find("<script")
    if script_idx >= 0:
        html = template[:script_idx] + inject + template[script_idx:]
    elif "</body>" in template:
        html = template.replace("</body>", inject + "</body>")
    else:
        html = template + inject   # fallback

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("Saved self-contained HTML to {}".format(save_path))
    return True


def _build_entry_from_result(
    sample_idx, result, areas, canvas_bbox, canvas_source,
    pins_pos, b2b_conn, p2b_edges,
    mib_group_arr, cluster_group_arr, boundary_code_arr,
    fixed_mask_pb, preplaced_mask_pb, opt_for_json,
):
    """把一個 result dict（raw 或 legalized）包成 build_solution_entry 需要的格式。"""
    act_for_json = {
        "area": float(result["bbox_area"]),
        "b2b_hpwl": float(result["b2b_hpwl"]),
        "p2b_hpwl": float(result["p2b_hpwl"]),
    }
    return build_solution_entry(
        sample_idx=sample_idx,
        areas=areas,
        area_tolerance=0.01,
        result_x=result["x"], result_y=result["y"],
        result_w=result["w"], result_h=result["h"],
        canvas_bbox=canvas_bbox,
        canvas_source=canvas_source,
        pins_pos=pins_pos,
        b2b_edges=b2b_conn,
        p2b_edges=p2b_edges,
        mib_group=mib_group_arr,
        cluster_group=cluster_group_arr,
        boundary_code=boundary_code_arr,
        fixed_mask_per_block=fixed_mask_pb,
        preplaced_mask_per_block=preplaced_mask_pb,
        optimal_metrics=opt_for_json,
        actual_metrics=act_for_json,
    )


def run_one_sample(sample_idx, official, model, config, device,
                   n_samples=6, ddim_steps=100,
                   sampler="ddim", edm_steps=50, use_amp=False, post_repel_steps=30,
                   scale_t_windows=False, reinsert_sweeps=3, reinsert_grid_density=12,
                   pin_force_strength=0.02, grouping_force_strength=0.030,
                   boundary_nudge_strength=0.025, repulsion_strength=0.0375,
                   tie_break_modes=None, use_gravity=False, gravity_iters=40,
                   use_cluster_merge=True, hpwl_slack_ratio=5.0, use_snap_boundary=False,
                   boundary_hpwl_slack_ratio=0.0,
                   use_pair_reinsert=False, pair_reinsert_sweeps=2, pair_reinsert_grid_density=8,
                   pair_reinsert_hpwl_slack_ratio=0.0,
                   use_reinsert_reshape=False, reinsert_reshape_sweeps=2, reinsert_reshape_grid_density=8,
                   use_gradient_finetune=False, gradient_finetune_steps=400, gradient_finetune_lr=0.5,
                   gradient_finetune_patience=30, gradient_finetune_hpwl_slack_ratio=0.0,
                   use_second_merge_pass=True,
                   weight_dist=1.0, weight_boundary=3.0, weight_cluster=1.0,
                   weight_b2b=0.5, weight_p2b=0.15, weight_shape=3.0,
                   use_cluster_adjacency=False, cluster_adjacency_bonus=5.0,
                   extra_checkpoints=None):
    """
    跑單一 validation sample：
      1. 解析 inputs / GT / constraints
      2. diffusion 推論
      3. 評估 + 印報告 + 對比圖 + JSON dump

    extra_checkpoints（實驗用，v5.7）：`[(model2, config2), ...]` 的 list。
    給了的話，除了主要的 (model, config) 之外，對每個額外 checkpoint 也用
    同一組 areas/W_int/canvas/constraints 跑一次 `generate_floorplan`
    （各自的 `all_results` 候選池），全部候選（主要 + 額外）合併後用同一套
    排序鍵（total_overlap, V_relative, total_hpwl, bbox_area）重新選最好的
    一個，取代原本只從單一模型的 n_samples 個候選裡選。目的是測試不同
    checkpoint 之間是否互補——不需要重新訓練，純粹增加 diffusion 端候選池
    的多樣性，時間成本跟 checkpoint 數量成正比（額外的 forward 計算，
    legalize 仍然只跑一次，因為只有最終選出的單一 best 會送進 legalize）。

    validation 的 label 結構與 training 不同：
      labels[0] = polygons (k, n_verts, 2)   ← 從這裡推 bbox
      labels[2] = metrics (可能存在)
    """
    t_start = time.perf_counter()
    sample = official[sample_idx]
    inputs = sample['input']
    labels = sample['label']

    area_target = inputs[0]
    b2b_conn = inputs[1]
    p2b_conn = inputs[2]
    pins_pos_t = inputs[3]
    constraints_t = inputs[4]

    k = int((area_target != -1).sum().item())
    areas = area_target[:k].numpy().astype(np.float32)

    # GT (w, h, x, y)：validation 走 polygon → bbox 路徑
    polygons = labels[0]
    gt_w = np.zeros(k, dtype=np.float32)
    gt_h = np.zeros(k, dtype=np.float32)
    gt_x = np.zeros(k, dtype=np.float32)
    gt_y = np.zeros(k, dtype=np.float32)
    for i in range(k):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            gt_x[i] = float(x_min)
            gt_y[i] = float(y_min)
            gt_w[i] = float(x_max - x_min)
            gt_h[i] = float(y_max - y_min)

    # metrics（validation 可能也有，可能沒有；有就用、沒就 fallback）
    metrics = labels[2] if len(labels) > 2 else None

    # 建 W_int（k x k）
    W_int = np.zeros((k, k), dtype=np.float32)
    if b2b_conn is not None and len(b2b_conn) > 0:
        for edge in b2b_conn:
            i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
            if 0 <= i < k and 0 <= j < k:
                W_int[i, j] = w; W_int[j, i] = w

    # p2b edges + pins
    pins_pos = pins_pos_t.numpy().astype(np.float32) if pins_pos_t is not None else None
    p2b_edges = []
    if p2b_conn is not None and len(p2b_conn) > 0:
        for edge in p2b_conn:
            p, b, w = int(edge[0]), int(edge[1]), float(edge[2])
            if p >= 0 and 0 <= b < k:
                p2b_edges.append((p, b, w))

    constraints = constraints_t[:k].numpy() if constraints_t is not None else None

    # -- Canvas: 用 pin bbox 抓形狀 + area sum 校正大小 --
    total_area = float(areas.sum())
    if pins_pos is not None and len(pins_pos) >= 2:
        px_min, px_max = float(pins_pos[:, 0].min()), float(pins_pos[:, 0].max())
        py_min, py_max = float(pins_pos[:, 1].min()), float(pins_pos[:, 1].max())
        pin_w = max(px_max - px_min, 1e-6)
        pin_h = max(py_max - py_min, 1e-6)
        aspect = pin_w / pin_h
        slack = 1.10
        canvas_w = float(np.sqrt(total_area * aspect) * slack)
        canvas_h = float(np.sqrt(total_area / aspect) * slack)
        cx = (px_min + px_max) / 2.0
        cy = (py_min + py_max) / 2.0
        x_offset = cx - canvas_w / 2.0
        y_offset = cy - canvas_h / 2.0
        print("Pin bbox: ({:.1f}, {:.1f}) ~ ({:.1f}, {:.1f}), aspect={:.3f}".format(
            px_min, py_min, px_max, py_max, aspect))
        print("Canvas: {:.1f} x {:.1f}, offset=({:.1f}, {:.1f})".format(
            canvas_w, canvas_h, x_offset, y_offset))
    else:
        canvas_w = canvas_h = float(np.sqrt(total_area))
        x_offset = y_offset = 0.0
        print("No pin info, fallback to square canvas: {:.1f}".format(canvas_w))

    # 找出 fixed / preplaced index + 拆出 MIB / cluster / boundary
    fixed_idx, preplaced_idx = [], []
    mib_group_arr = np.zeros(k, dtype=np.int64)
    cluster_group_arr = np.zeros(k, dtype=np.int64)
    boundary_code_arr = np.zeros(k, dtype=np.int64)
    if constraints is not None:
        cons = np.where(constraints < 0, 0, constraints)
        for i in range(k):
            if cons[i, 1] > 0.5:
                preplaced_idx.append(i)
            elif cons[i, 0] > 0.5:
                fixed_idx.append(i)
            mib_group_arr[i]     = int(cons[i, 2])
            cluster_group_arr[i] = int(cons[i, 3])
            boundary_code_arr[i] = int(cons[i, 4])

    # 把 fixed_idx / preplaced_idx 轉成 per-block bool array。
    # legalize_v2 和 build_solution_entry 都要用，這裡先算好、兩處共用。
    fixed_mask_pb = np.zeros(k, dtype=bool)
    preplaced_mask_pb = np.zeros(k, dtype=bool)
    for i in fixed_idx:
        fixed_mask_pb[i] = True
    for i in preplaced_idx:
        preplaced_mask_pb[i] = True
        fixed_mask_pb[i] = True   # preplaced 也算 fixed_shape

    print("Generating floorplan (k={})...".format(k))
    t_diff_start = time.perf_counter()
    best, all_results = generate_floorplan(
        model, config, areas, W_int,
        canvas_w=canvas_w, canvas_h=canvas_h,
        x_offset=x_offset, y_offset=y_offset,
        n_samples=n_samples, ddim_steps=ddim_steps, device=device,
        constraints=constraints,
        p2b_edges=p2b_edges, pins_pos=pins_pos,
        gt_w=gt_w, gt_h=gt_h, gt_x=gt_x, gt_y=gt_y,
        sampler=sampler, edm_steps=edm_steps, use_amp=use_amp,
        post_repel_steps=post_repel_steps, scale_t_windows=scale_t_windows,
        pin_force_strength=pin_force_strength,
        grouping_force_strength=grouping_force_strength,
        boundary_nudge_strength=boundary_nudge_strength,
        repulsion_strength=repulsion_strength,
    )

    # v5.7（實驗用）：把額外 checkpoint 的候選池併進來，一起重新選最好的
    if extra_checkpoints:
        pooled = list(all_results)
        for extra_model, extra_config in extra_checkpoints:
            _, extra_results = generate_floorplan(
                extra_model, extra_config, areas, W_int,
                canvas_w=canvas_w, canvas_h=canvas_h,
                x_offset=x_offset, y_offset=y_offset,
                n_samples=n_samples, ddim_steps=ddim_steps, device=device,
                constraints=constraints,
                p2b_edges=p2b_edges, pins_pos=pins_pos,
                gt_w=gt_w, gt_h=gt_h, gt_x=gt_x, gt_y=gt_y,
                sampler=sampler, edm_steps=edm_steps, use_amp=use_amp,
                post_repel_steps=post_repel_steps, scale_t_windows=scale_t_windows,
                pin_force_strength=pin_force_strength,
                grouping_force_strength=grouping_force_strength,
                boundary_nudge_strength=boundary_nudge_strength,
                repulsion_strength=repulsion_strength,
            )
            pooled.extend(extra_results)
        pooled.sort(key=lambda r: (
            r["overlap"],
            r["soft"]["V_relative"],
            r["total_hpwl"],
            r["bbox_area"],
        ))
        all_results = pooled
        best = pooled[0]

    t_diffusion = time.perf_counter() - t_diff_start

    # -- optimal 指標 --
    # v3.6: 永遠從 GT 的 (x, y, w, h) 自己算 HPWL，不再依賴 labels[2] 的 metrics
    # tensor。理由：(a) validation 的 label 結構不一定跟 training 相同；
    # (b) 即使有，metrics 可能用「真實 polygon」算，而 inference 結果是用 bbox，
    # 自己算才是同公式對比、最公平。
    opt_bbox_w = np.max(gt_x + gt_w) - np.min(gt_x)
    opt_bbox_h = np.max(gt_y + gt_h) - np.min(gt_y)
    opt_b2b_hpwl = compute_hpwl_vectorized(gt_x, gt_y, gt_w, gt_h, W_int)
    opt_p2b_hpwl = 0.0
    if p2b_edges and pins_pos is not None:
        opt_p2b_hpwl = compute_p2b_hpwl(gt_x, gt_y, gt_w, gt_h, p2b_edges, pins_pos)
    optimal = {
        "x": gt_x, "y": gt_y, "w": gt_w, "h": gt_h,
        "bbox_area": float(opt_bbox_w * opt_bbox_h),
        "b2b_hpwl": float(opt_b2b_hpwl),
        "p2b_hpwl": float(opt_p2b_hpwl),
        "total_hpwl": float(opt_b2b_hpwl + opt_p2b_hpwl),
    }
    if metrics is not None:
        m = metrics.numpy().astype(np.float64) if hasattr(metrics, "numpy") else np.array(metrics)
        if len(m) >= 1:
            optimal["area_metric"] = float(m[0])

    # ====================================================
    # Raw（diffusion 原始輸出，未 legalize）的評估
    # ====================================================
    print("\n>>> RAW (diffusion output, no legalize)")
    evaluate_and_report(best, W_int, areas, constraints, optimal=optimal, stage="RAW")

    # canvas 用 pin bbox（範本 source = "pin_bbox"）——legalize_lff 也用這個
    # 當 outline_bbox，讓 block 落在 pin 圍出的方框裡頭。
    if pins_pos is not None and len(pins_pos) >= 2:
        bb_x_min = float(pins_pos[:, 0].min())
        bb_y_min = float(pins_pos[:, 1].min())
        bb_x_max = float(pins_pos[:, 0].max())
        bb_y_max = float(pins_pos[:, 1].max())
        canvas_source = "pin_bbox"
    else:
        # 沒 pin 就 fallback 用 GT bbox
        bb_x_min = float(np.min(gt_x))
        bb_y_min = float(np.min(gt_y))
        bb_x_max = float(np.max(gt_x + gt_w))
        bb_y_max = float(np.max(gt_y + gt_h))
        canvas_source = "gt_bbox_fallback"
    canvas_bbox = (bb_x_min, bb_y_min, bb_x_max, bb_y_max)

    # ====================================================
    # Legalize：保證零 overlap，同時盡量保留 soft constraint / HPWL 品質，
    # 並讓 block 落在 pin bbox 裡頭
    # ====================================================
    t_legal_start = time.perf_counter()
    legalized = legalize_result(
        best, areas, W_int, p2b_edges, pins_pos,
        preplaced_mask=preplaced_mask_pb,
        fixed_mask=fixed_mask_pb,
        mib_group=mib_group_arr,
        cluster_group=cluster_group_arr,
        boundary_code=boundary_code_arr,
        outline_bbox=canvas_bbox,
        reinsert_sweeps=reinsert_sweeps,
        reinsert_grid_density=reinsert_grid_density,
        tie_break_modes=tie_break_modes,
        use_gravity=use_gravity,
        gravity_iters=gravity_iters,
        use_cluster_merge=use_cluster_merge,
        hpwl_slack_ratio=hpwl_slack_ratio,
        use_snap_boundary=use_snap_boundary,
        boundary_hpwl_slack_ratio=boundary_hpwl_slack_ratio,
        use_pair_reinsert=use_pair_reinsert,
        pair_reinsert_sweeps=pair_reinsert_sweeps,
        pair_reinsert_grid_density=pair_reinsert_grid_density,
        pair_reinsert_hpwl_slack_ratio=pair_reinsert_hpwl_slack_ratio,
        use_reinsert_reshape=use_reinsert_reshape,
        reinsert_reshape_sweeps=reinsert_reshape_sweeps,
        reinsert_reshape_grid_density=reinsert_reshape_grid_density,
        use_gradient_finetune=use_gradient_finetune,
        gradient_finetune_steps=gradient_finetune_steps,
        gradient_finetune_lr=gradient_finetune_lr,
        gradient_finetune_patience=gradient_finetune_patience,
        gradient_finetune_hpwl_slack_ratio=gradient_finetune_hpwl_slack_ratio,
        use_second_merge_pass=use_second_merge_pass,
        weight_dist=weight_dist,
        weight_boundary=weight_boundary,
        weight_cluster=weight_cluster,
        weight_b2b=weight_b2b,
        weight_p2b=weight_p2b,
        weight_shape=weight_shape,
        use_cluster_adjacency=use_cluster_adjacency,
        cluster_adjacency_bonus=cluster_adjacency_bonus,
    )
    t_legalize = time.perf_counter() - t_legal_start

    print("\n>>> LEGALIZED (post-processed, zero overlap guaranteed)")
    evaluate_and_report(legalized, W_int, areas, constraints, optimal=optimal, stage="LEGALIZED")

    # 視覺化由 HTML viewer 處理（在 main 統一輸出）。這裡不再生 PNG。

    # ====================================================
    # Build solution entries（raw + legalized，給最後統一寫 JSON 用）
    # ====================================================

    # v3.5: 把 validation optimal 的關鍵 metric 抽成 dict 給 JSON 用
    # area 用 GT bbox（跟 evaluate_and_report 一致）；HPWL 從 metrics tensor 拿
    opt_for_json = None
    if optimal is not None:
        opt_for_json = {"area": optimal.get("bbox_area", 0.0)}
        if "b2b_hpwl" in optimal:
            opt_for_json["b2b_hpwl"] = optimal["b2b_hpwl"]
        if "p2b_hpwl" in optimal:
            opt_for_json["p2b_hpwl"] = optimal["p2b_hpwl"]

    raw_entry = _build_entry_from_result(
        sample_idx, best, areas, canvas_bbox, canvas_source,
        pins_pos, b2b_conn, p2b_edges,
        mib_group_arr, cluster_group_arr, boundary_code_arr,
        fixed_mask_pb, preplaced_mask_pb, opt_for_json,
    )
    legalized_entry = _build_entry_from_result(
        sample_idx, legalized, areas, canvas_bbox, canvas_source,
        pins_pos, b2b_conn, p2b_edges,
        mib_group_arr, cluster_group_arr, boundary_code_arr,
        fixed_mask_pb, preplaced_mask_pb, opt_for_json,
    )

    print("\nAll candidates (raw, sorted by overlap_area, V_rel, hpwl, bbox_area):")
    for r in all_results:
        print("  Sample {}: overlap={:.2f} ({} pairs), V_rel={:.3f}, HPWL={:.1f}, bbox={:.1f}".format(
            r['sample_idx'], r['overlap'], r['n_overlaps'],
            r['soft']['V_relative'], r['total_hpwl'], r['bbox_area']))

    elapsed_total = time.perf_counter() - t_start
    print("\nTiming: diffusion={:.2f}s, legalize={:.2f}s, total={:.2f}s".format(
        t_diffusion, t_legalize, elapsed_total))

    return {
        "raw_entry": raw_entry,
        "legalized_entry": legalized_entry,
        "raw": best,
        "legalized": legalized,
        "optimal": optimal,
        "block_count": k,
        "timing": {
            "diffusion": t_diffusion,
            "legalize": t_legalize,
            "total": elapsed_total,
        },
    }


def main():
    """用 validation dataset 跑推理，每個 sample 輸出對照圖；最後合併寫一個 JSON。"""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.dirname(__file__))

    # v4.2: 切換 sampler。"ddim" 走原本的 DDIM sampler、"edm" 走 Heun 2 階 + EDM 時間步
    # 兩個 sampler NFE 一致（DDIM 100 步 vs EDM 50 步，都是 100 次 model forward）
    SAMPLER = "ddim"        # <-- 切換這裡：改成 "edm" 就跑 EDM
    EDM_STEPS = 50
    # v4.3: 100 → 30 步。11-sample A/B 掃過 100/80/60/50/40/30，發現 legalize
    # 對 raw 品質很魯棒（不管 raw V_relative 是 0.35 還是 0.79，legalize 後都
    # 收斂到 ~0.11-0.14），最終 area_gap/hpwl_gap/V_relative 在 30 步時仍在
    # 雜訊範圍內、沒有隨步數下降而變差，diffusion 時間卻線性省了 ~66%。
    DDIM_STEPS = 30
    # v4.4: 6 → 14。best-of-N 候選是同一個 batch 一起跑，100 樣本 A/B 顯示
    # diffusion 時間幾乎不變（candidate 數在 GPU 上幾乎是平行、非序列成本），
    # 但 HPWL_gap（16.3%→15.4%）、V_relative（0.114→0.111）都有小幅下降。
    N_SAMPLES = 14

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model("checkpoints/model_epoch300_overlap_v4.pt", device)

    # -- 載入 validation dataset --
    # validation 共 100 個樣本（k 從 21 到 120 各一個）
    from litetestLoader import FloorplanDatasetLiteTest
    official = FloorplanDatasetLiteTest("../")
    print("Validation dataset size: {}".format(len(official)))
    print("Sampler: {}".format(SAMPLER))

    # 想跑哪幾個 sample 就改這個 list（涵蓋小、中、大 k 比較好）
    sample_indices = [i for i in range(100)]

    solutions_raw = []
    solutions_legalized = []
    timing_records = []
    overlaps_raw = []
    overlaps_legal = []
    rels = []
    areas = []
    hpwls = []
    times = []
    for idx in sample_indices:
        if idx < 0 or idx >= len(official):
            print("Skip invalid index {}".format(idx))
            continue
        print("\n" + "#" * 64)
        print("# Sample {} / {}".format(idx, len(official) - 1))
        print("#" * 64)
        out = run_one_sample(
            idx, official, model, config, device,
            sampler=SAMPLER, edm_steps=EDM_STEPS, ddim_steps=DDIM_STEPS,
            n_samples=N_SAMPLES,
        )
        if out is None:
            continue
        solutions_raw.append(out["raw_entry"])
        solutions_legalized.append(out["legalized_entry"])
        timing_records.append({
            "test_id": idx,
            "block_count": out["block_count"],
            "diffusion_time": out["timing"]["diffusion"],
            "legalize_time": out["timing"]["legalize"],
            "total_time": out["timing"]["total"],
        })
        overlaps_raw.append(out["raw"]["overlap"])
        overlaps_legal.append(out["legalized"]["overlap"])
        rels.append(out["legalized"]["soft"]["V_relative"])
        areas.append(out["legalized"]["bbox_area"] / out["optimal"]["bbox_area"] - 1.0)
        hpwls.append(out["legalized"]["total_hpwl"] / out["optimal"]["total_hpwl"] - 1.0)
        times.append(out["timing"]["total"])

    block_counts = [r["block_count"] for r in timing_records]

    plt.subplot(5, 1, 1)
    plt.plot(block_counts, overlaps_raw, label="raw")
    plt.plot(block_counts, overlaps_legal, label="legalized")
    plt.xlabel('n')
    plt.ylabel('Overlap')
    plt.legend()
    plt.grid()

    plt.subplot(5, 1, 2)
    plt.plot(block_counts, rels)
    plt.xlabel('n')
    plt.ylabel('V_relative (legalized)')
    plt.grid()

    plt.subplot(5, 1, 3)
    plt.plot(block_counts, areas)
    plt.xlabel('n')
    plt.ylabel('Area_gap (legalized)')
    plt.grid()

    plt.subplot(5, 1, 4)
    plt.plot(block_counts, hpwls)
    plt.xlabel('n')
    plt.ylabel('HPWL_gap (legalized)')
    plt.grid()

    plt.subplot(5, 1, 5)
    plt.plot(block_counts, times)
    plt.xlabel('n')
    plt.ylabel('Time (s)')
    plt.grid()

    plt.tight_layout()
    plt.savefig("../json/summary_plot_{}.png".format(SAMPLER), dpi=150)
    print("Saved summary plot to ../json/summary_plot_{}.png".format(SAMPLER))

    # ---- Per-case timing summary ----
    print("\n" + "=" * 64)
    print("Per-case timing summary")
    print("=" * 64)
    print("{:<8}{:<6}{:>14}{:>14}{:>12}".format(
        "test_id", "k", "diffusion(s)", "legalize(s)", "total(s)"))
    for rec in timing_records:
        print("{:<8}{:<6}{:>14.3f}{:>14.3f}{:>12.3f}".format(
            rec["test_id"], rec["block_count"],
            rec["diffusion_time"], rec["legalize_time"], rec["total_time"]))
    if timing_records:
        avg_diff = sum(r["diffusion_time"] for r in timing_records) / len(timing_records)
        avg_leg = sum(r["legalize_time"] for r in timing_records) / len(timing_records)
        avg_tot = sum(r["total_time"] for r in timing_records) / len(timing_records)
        print("-" * 54)
        print("{:<14}{:>14.3f}{:>14.3f}{:>12.3f}".format("average", avg_diff, avg_leg, avg_tot))

    print('\nAverage Overlap (raw):          {:.4f}'.format(sum(overlaps_raw) / len(overlaps_raw)))
    print('Average Overlap (legalized):    {:.4f}'.format(sum(overlaps_legal) / len(overlaps_legal)))
    print('Average V_relative (legalized): {:.4f}'.format(sum(rels) / len(rels)))
    print('Average Area_gap (legalized):   {:.4f}'.format(sum(areas) / len(areas)))
    print('Average HPWL_gap (legalized):   {:.4f}'.format(sum(hpwls) / len(hpwls)))
    print('Average Time:                   {:.4f}'.format(sum(times) / len(times)))

    # ---- 寫出三份結果：raw JSON、legalized JSON、timing summary ----
    os.makedirs("../json", exist_ok=True)
    out_json_raw = "../json/quick_eval_solutions_{}_raw.json".format(SAMPLER)
    out_json_legal = "../json/quick_eval_solutions_{}_legalized.json".format(SAMPLER)
    write_solutions_json(out_json_raw, solutions_raw, submission_tag="diffusion_privateval_raw")
    write_solutions_json(out_json_legal, solutions_legalized, submission_tag="diffusion_privateval_legalized")

    timing_path = "../json/timing_summary_{}.json".format(SAMPLER)
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_records, f, indent=2)
    print("Saved timing summary to {}".format(timing_path))


if __name__ == "__main__":
    main()