#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Optimizer

Diffusion + LFF-legalize pipeline (see README.md for the full write-up).

Stage 1 (diffusion, diffusion.py / inference.py:generate_floorplan):
  Force-guided DDIM sampler generates a raw layout. Runs a batch of
  best-of-N candidates together (candidates share one GPU batch, so extra
  candidates cost almost nothing), applies pin/grouping/repulsion/boundary
  forces during sampling, then a short physics-only "post-repel" phase.

Stage 2 (legalize, utils.py:legalize_lff via inference.py:legalize_result):
  Deterministic, single-pass, Less-Flexibility-First-style placement using
  MAXRECTS free-rectangle bin packing with weighted-median (L1-optimal)
  positioning per block. Guarantees, by construction:
    - zero overlap between blocks
    - preplaced blocks keep their exact position and shape
    - fixed-shape blocks keep their exact shape
    - all blocks stay within 1% of their target area
    - all blocks stay within the pin-derived bounding box
  Followed by compaction passes (compact_merge_clusters, compact_reinsert,
  compact_positions) that reduce wasted space without ever being able to
  reintroduce a hard-constraint violation.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

from iccad2026_evaluate import FloorplanOptimizer

# iccad2026_evaluate.py inserts FloorSet/ (its parent dir) at sys.path[0] as
# a side effect of import, and resolves its own `from utils import
# unpad_tensor, ...` against FloorSet/utils.py. That caches sys.modules
# ["utils"] = FloorSet/utils.py. inference.py needs a *different* module
# also named "utils" (this directory's utils.py, with legalize_lff etc.) -
# without clearing the cache and re-asserting this directory's priority,
# inference.py's own `from utils import ...` would silently resolve against
# the wrong file and fail with ImportError. Safe to do only *after*
# iccad2026_evaluate has finished importing: it already bound whatever it
# needed from FloorSet/utils.py into its own namespace by this point.
sys.modules.pop("utils", None)
sys.path.insert(0, str(_THIS_DIR))

from inference import load_model, generate_floorplan, legalize_result


class MyOptimizer(FloorplanOptimizer):
    """Diffusion-generate + LFF-legalize floorplanning optimizer."""

    # Config validated on the 100-sample validation set -- see method.md
    # sections 2.2 (v4.7 legalize-side cluster merge) and 2.1 (v5.0
    # diffusion-side force-strength tuning). ~2.5s/sample avg, 0/100
    # infeasible; see the evaluate run for area/hpwl/V_rel.
    CHECKPOINT_NAME = "model_epoch300_overlap_v4.pt"
    # v5.18（不採用，見 CHANGELOG.md）：Self-Conditioning。100 樣本 paired
    # 測試接近打平甚至略負（cost-proxy +0.28%，raw overlap +24% 明顯變差），
    # 訓練時 val_loss 也比 v4 高（0.1425 vs 0.1026），兩個獨立訊號方向
    # 一致。官方 evaluate 確認：real score 1.1477，比目前 production 的
    # 1.128 差 +1.75%，avg runtime 也較高（1.66s vs 1.46s）。維持關閉。
    USE_SELF_COND = False
    # v5.15（採用）：v4 原本平均 RuntimeFactor^0.3 約 0.837，離公式下限 0.7
    # 還有 16% 空間沒被利用到——DDIM_STEPS=30 從沒被系統性掃過。34 樣本
    # 分層抽樣（真實 median runtime 換算）發現從 30 降到 4 real cost 持續
    # 變好或打平（品質幾乎不變，甚至 V_relative 常常更低），steps=2 才
    # 崩潰（hpwl_gap 0.17→0.61）。選 10（離懸崖 2 倍以上安全邊際）用完整
    # 100 樣本官方 evaluate 確認兩次：real score（換算真實 median
    # runtime）1.1945 / 1.1614，平均 1.178，比 v4 原本的 1.2322 好
    # -4.4%，兩次都個別優於 baseline、0/100 infeasible，avg runtime
    # 2.485s→1.84s。見 CHANGELOG.md v5.15。
    DDIM_STEPS = 10
    N_SAMPLES = 14
    # v5.16（採用）：同一套真實 median runtime 換算方法論套用到
    # POST_REPEL_STEPS。34 樣本分層抽樣：30→20→15→10 real cost 持續變好
    # （1.117→1.091，-2.4%），但 `0`（完全關閉）V_relative 從 ~0.10 跳到
    # 0.19、real cost 反彈到 1.294——post-repel 不是純冗餘，對
    # boundary/overlap 有 legalize 自己補不回來的清理效果。完整 100 樣本
    # 官方 evaluate 確認兩次（在 DDIM_STEPS=10 之上疊加）：real score
    # 1.1807 / 1.1577，平均 1.1692，比純 DDIM_STEPS=10 的 1.178 再進步
    # 約 -0.75%，0/100 infeasible。見 CHANGELOG.md v5.16。
    POST_REPEL_STEPS = 10
    # v5.17（採用）：legalize 的 compact_reinsert 搜尋強度
    # （reinsert_sweeps/reinsert_grid_density，原預設 3/12）。34 樣本
    # 分層抽樣：真實資料上第一輪就幾乎收斂，area_gap/hpwl_gap 在所有測試
    # 組合下完全不變，grid_density 降到 4 以下 real cost 打平（不再有額外
    # 好處也沒有壞處）。選 sweeps=1/grid_density=4，完整 100 樣本官方
    # evaluate 確認兩次（在 v5.15+v5.16 之上疊加）：real score 1.1381 /
    # 1.1174，平均 1.128，比 v5.16 的 1.1692 再進步約 -3.5%，0/100
    # infeasible。見 CHANGELOG.md v5.17。
    REINSERT_SWEEPS = 1
    REINSERT_GRID_DENSITY = 4
    # v5.0: 100-sample quasi-paired sweep found the hardcoded force-guidance
    # strengths in diffusion.py were too strong, overriding the model's own
    # learned signal. grouping_force_strength 0.015->0.030 (sweet spot; both
    # weaker and 0.050 are worse), repulsion_strength 0.05->0.025, and
    # boundary_nudge_strength 0.05->0.025 (0.10 made V_boundary worse).
    # Combined (not just additive): area/hpwl unchanged, V_relative
    # 0.1092->0.1032 (V_grouping 359->339), ~-1.26% on the contest cost
    # formula vs. the old hardcoded defaults. A follow-up fine sweep around
    # this point found grouping_force_strength=0.030 and
    # boundary_nudge_strength=0.025 were already the local optimum, but
    # repulsion_strength=0.0375 beat 0.025 on both sides of two independent
    # 100-sample re-runs (different random seeds each time; ~-0.2%~-0.3% on
    # the cost formula, small but directionally consistent) -- adopted.
    GROUPING_FORCE_STRENGTH = 0.030
    # v5.21（不採用，見 CHANGELOG.md）：重新檢視 boundary_nudge_strength
    # 是否因為 DDIM_STEPS 30→10（v5.15）而不再是最佳值。34 樣本分層抽樣
    # 顯示 0.075 real cost 較好（V_relative 0.1092→0.1048，約 -1%），但
    # 完整 100 樣本官方 evaluate 兩次（1.1395 / 1.1573，平均 1.1484）
    # 明顯比 v4 baseline（~1.11-1.128）差，34 樣本篩選再次高估了效果，
    # 維持原值 0.025。grouping_force_strength 同時重掃過，目前 0.030 已是
    # 最佳（調高讓 V_relative 變差），無變動。
    BOUNDARY_NUDGE_STRENGTH = 0.025
    REPULSION_STRENGTH = 0.0375
    # v5.12（不採用，見 CHANGELOG.md）：純推論端，force 強度乘上
    # alpha_bar_t**power，讓每個力剛進窗口時弱、隨 x0_pred 信心平滑增強到滿
    # 強度。100 樣本 paired 測試 power=1.0 四項指標同向變好、但幅度小
    # （cost-proxy -0.55%，53/100）；兩次獨立官方 evaluate 平均反而比 v4
    # baseline 略差（+1.0%），沒有通過確認，維持關閉（0.0 = 跟改動前完全
    # 等價）。
    FORCE_CONFIDENCE_POWER = 0.0
    # v5.14（不採用，見 CHANGELOG.md）：RePaint 式 harmonization resampling。
    # 100 樣本 paired 測試 steps=2：area/hpwl/V_relative/raw overlap 四項
    # 全部同向變好（raw overlap -37%），cost-proxy -1.77%——是這個 session
    # 目前 paired 訊號最乾淨的一次。兩次官方 evaluate（中性 RuntimeFactor）
    # 平均 1.5159，看起來甚至略優於 v4 baseline 的 1.5248。但 v4 本來就比
    # alpha-test median runtime 快很多（99/100 樣本比 median 快），換算
    # 真實 median runtime 後 v4 real score 只有 1.2322；repaint 平均 runtime
    # 從 2.485s 拉到 3.505s（+41%），把這個速度優勢吃掉一大半，real score
    # 變成 1.3319（**+8.1%，明顯變差**）——中性 evaluate 完全看不出這個
    # 代價，必須換算真實 median runtime 才會現形。維持關閉（1 = 跟改動前
    # 完全等價）。
    REPAINT_RESAMPLE_STEPS = 1
    # v5.31（不採用，見 CHANGELOG.md）：post-repel 階段加入 grouping force。
    POST_REPEL_GROUPING = False
    # v5.34（不採用，見 CHANGELOG.md）：compact_merge_cluster_groups
    # 擴大候選搬移搜尋範圍，疊加 v5.33 的 cost-aware 閘門。
    USE_EXPANDED_SEARCH = False
    EXPANDED_SEARCH_MAX_PAIRS = 20
    USE_COST_AWARE_GATE = False

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_path = str(Path(__file__).parent / "checkpoints" / self.CHECKPOINT_NAME)
        self.model, self.config = load_model(checkpoint_path, self.device)

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None
    ) -> List[Tuple[float, float, float, float]]:
        k = block_count
        areas = area_targets[:k].numpy().astype(np.float32)

        # -- constraints: (fixed, preplaced, mib, cluster, boundary) --
        # generate_floorplan() does its own clipping internally and wants the
        # raw (unclipped, -1-for-missing) tensor; everything else here needs
        # the clipped version to build masks/group ids.
        constraints_raw = constraints[:k].numpy()
        cons = np.where(constraints_raw < 0, 0, constraints_raw)
        preplaced_mask = cons[:, 1] > 0.5
        fixed_mask = (cons[:, 0] > 0.5) | preplaced_mask
        mib_group = cons[:, 2].astype(np.int64)
        cluster_group = cons[:, 3].astype(np.int64)
        boundary_code = cons[:, 4].astype(np.int64)

        # -- target_positions: (x, y, w, h), -1 = free. Fixed-shape blocks
        #    have (w, h) set; preplaced blocks have all four set. This is
        #    the "ground truth" the diffusion model inpaints toward for
        #    hard-constrained blocks during sampling.
        gt_x = np.zeros(k, dtype=np.float32)
        gt_y = np.zeros(k, dtype=np.float32)
        gt_w = np.zeros(k, dtype=np.float32)
        gt_h = np.zeros(k, dtype=np.float32)
        if target_positions is not None:
            tp = target_positions[:k].numpy().astype(np.float32)
            gt_w[fixed_mask] = tp[fixed_mask, 2]
            gt_h[fixed_mask] = tp[fixed_mask, 3]
            gt_x[preplaced_mask] = tp[preplaced_mask, 0]
            gt_y[preplaced_mask] = tp[preplaced_mask, 1]

        # -- b2b connectivity -> dense weight matrix --
        W_int = np.zeros((k, k), dtype=np.float32)
        if b2b_connectivity is not None and len(b2b_connectivity) > 0:
            for edge in b2b_connectivity:
                i, j, wgt = int(edge[0]), int(edge[1]), float(edge[2])
                if 0 <= i < k and 0 <= j < k:
                    W_int[i, j] = wgt
                    W_int[j, i] = wgt

        # -- pins + p2b edges --
        pins_np = None
        if pins_pos is not None and len(pins_pos) > 0:
            pins_np = pins_pos.numpy().astype(np.float32)
        p2b_edges = []
        if p2b_connectivity is not None and len(p2b_connectivity) > 0:
            for edge in p2b_connectivity:
                p_idx, b_idx, wgt = int(edge[0]), int(edge[1]), float(edge[2])
                if p_idx >= 0 and 0 <= b_idx < k:
                    p2b_edges.append((p_idx, b_idx, wgt))

        # -- canvas: shape from pin bbox, size from total block area --
        total_area = float(areas.sum())
        if pins_np is not None and len(pins_np) >= 2:
            px_min, px_max = float(pins_np[:, 0].min()), float(pins_np[:, 0].max())
            py_min, py_max = float(pins_np[:, 1].min()), float(pins_np[:, 1].max())
            aspect = max(px_max - px_min, 1e-6) / max(py_max - py_min, 1e-6)
            slack = 1.10
            canvas_w = float(np.sqrt(total_area * aspect) * slack)
            canvas_h = float(np.sqrt(total_area / aspect) * slack)
            x_offset = (px_min + px_max) / 2.0 - canvas_w / 2.0
            y_offset = (py_min + py_max) / 2.0 - canvas_h / 2.0
            outline_bbox = (px_min, py_min, px_max, py_max)
        else:
            canvas_w = canvas_h = float(np.sqrt(total_area))
            x_offset = y_offset = 0.0
            outline_bbox = None

        best, _ = generate_floorplan(
            self.model, self.config, areas, W_int,
            canvas_w=canvas_w, canvas_h=canvas_h,
            x_offset=x_offset, y_offset=y_offset,
            n_samples=self.N_SAMPLES, ddim_steps=self.DDIM_STEPS, device=self.device,
            constraints=constraints_raw,
            p2b_edges=p2b_edges, pins_pos=pins_np,
            gt_w=gt_w, gt_h=gt_h, gt_x=gt_x, gt_y=gt_y,
            sampler="ddim", post_repel_steps=self.POST_REPEL_STEPS,
            grouping_force_strength=self.GROUPING_FORCE_STRENGTH,
            boundary_nudge_strength=self.BOUNDARY_NUDGE_STRENGTH,
            repulsion_strength=self.REPULSION_STRENGTH,
            force_confidence_power=self.FORCE_CONFIDENCE_POWER,
            repaint_resample_steps=self.REPAINT_RESAMPLE_STEPS,
            use_self_cond=self.USE_SELF_COND,
            post_repel_grouping=self.POST_REPEL_GROUPING,
        )

        legalized = legalize_result(
            best, areas, W_int, p2b_edges, pins_np,
            preplaced_mask=preplaced_mask,
            fixed_mask=fixed_mask,
            mib_group=mib_group,
            cluster_group=cluster_group,
            boundary_code=boundary_code,
            outline_bbox=outline_bbox,
            use_second_merge_pass=True,
            use_cluster_merge=True,
            hpwl_slack_ratio=5.0,
            use_expanded_search=self.USE_EXPANDED_SEARCH,
            expanded_search_max_pairs=self.EXPANDED_SEARCH_MAX_PAIRS,
            use_cost_aware_gate=self.USE_COST_AWARE_GATE,
            reinsert_sweeps=self.REINSERT_SWEEPS,
            reinsert_grid_density=self.REINSERT_GRID_DENSITY,
            verbose=self.verbose,
        )

        x, y, w, h = legalized["x"], legalized["y"], legalized["w"], legalized["h"]
        return [(float(x[i]), float(y[i]), float(w[i]), float(h[i])) for i in range(k)]
