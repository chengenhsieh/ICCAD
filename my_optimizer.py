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
    DDIM_STEPS = 30
    N_SAMPLES = 14
    POST_REPEL_STEPS = 30
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
    BOUNDARY_NUDGE_STRENGTH = 0.025
    REPULSION_STRENGTH = 0.0375

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
            verbose=self.verbose,
        )

        x, y, w, h = legalized["x"], legalized["y"], legalized["w"], legalized["h"]
        return [(float(x[i]), float(y[i]), float(w[i]), float(h[i])) for i in range(k)]
