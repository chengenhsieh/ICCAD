"""Energy & constraint utilities.

Mirrors cells 27, 29, 31, 33, 35 of the original notebook with ONE behavioural
change: `compute_e_rel` now uses a `bbox_baseline` ratio so that E_rel = 1.0
exactly on the ground-truth solution (v0.16 bug fix).

Old (v0.15) formula:
    energy = lam_hpwl * (hpwl / hpwl_baseline)
           + lam_over * overlap
           + lam_bbox * (bbox / 4)
    E_rel  = exp(-kappa * (energy - lam_hpwl))

Problem: on the GT, hpwl/hpwl_baseline = 1 and overlap = 0, but bbox/4 is a
positive number tied to the GT's bbox area (typically smaller than the canvas
under v0.13's pin-defined normalization, so well below 1). Subtracting only
`lam_hpwl` leaves a leftover `lam_bbox * bbox_gt/4` term, so E_rel on the GT
sits at exp(-kappa * lam_bbox * bbox_gt/4), which with default constants is
deep in (0, 1). The model never sees E_rel near 1 during training, so feeding
1.0 at inference is out-of-distribution and degenerates the layout.

New (v0.16) formula:
    energy = lam_hpwl * (hpwl / hpwl_baseline)
           + lam_over * overlap
           + lam_bbox * (bbox / bbox_baseline)
    E_rel  = exp(-kappa * (energy - lam_hpwl - lam_bbox))

Both ratios equal 1 on the GT, overlap = 0, so the inner expression is exactly
0 and E_rel = 1. For any layout worse than the GT (longer wires or larger
bbox), the ratios exceed 1 and E_rel drops below 1.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    KAPPA, LAMBDA_HPWL, LAMBDA_OVERLAP, LAMBDA_BBOX_EREL,
    OVERLAP_SAMPLE_K, OVERLAP_LINEAR_MIX,
)
from .canvas import _build_padded_groups


# --------------------------------------------------------------------------- #
# HPWL                                                                         #
# --------------------------------------------------------------------------- #
def compute_hpwl(pos, edge_index, edge_weight, batch_idx,
                 tau=0.01, differentiable=False):
    src, dst = edge_index
    dx = pos[src, 0] - pos[dst, 0]
    dy = pos[src, 1] - pos[dst, 1]
    if differentiable:
        abs_dx = tau * torch.logaddexp(dx / tau, -dx / tau)
        abs_dy = tau * torch.logaddexp(dy / tau, -dy / tau)
    else:
        abs_dx = dx.abs()
        abs_dy = dy.abs()
    return (edge_weight * (abs_dx + abs_dy)).sum()


# --------------------------------------------------------------------------- #
# Overlap loss                                                                 #
# --------------------------------------------------------------------------- #
def _pair_overlap_terms(g_cx, g_cy, g_w, g_h,
                        margin=0.0, quartic=False, normalise=False):
    """Pairwise overlap-area terms with margin / normalisation. Returns [n, n].

    Kept as a single-graph utility for callers (notably the vectorized batched
    core below uses a [G, Mmax, Mmax] equivalent and does not call this).
    """
    hw, hh = g_w / 2, g_h / 2
    dx = torch.min(g_cx.unsqueeze(1) + hw.unsqueeze(1),
                   g_cx.unsqueeze(0) + hw.unsqueeze(0)) \
       - torch.max(g_cx.unsqueeze(1) - hw.unsqueeze(1),
                   g_cx.unsqueeze(0) - hw.unsqueeze(0))
    dy = torch.min(g_cy.unsqueeze(1) + hh.unsqueeze(1),
                   g_cy.unsqueeze(0) + hh.unsqueeze(0)) \
       - torch.max(g_cy.unsqueeze(1) - hh.unsqueeze(1),
                   g_cy.unsqueeze(0) - hh.unsqueeze(0))
    rdx = F.relu(dx + margin)
    rdy = F.relu(dy + margin)
    if quartic:
        overlap = (rdx ** 2) * (rdy ** 2)
    else:
        overlap = rdx * rdy
    if normalise:
        area = g_w * g_h
        denom = torch.minimum(area.unsqueeze(0), area.unsqueeze(1)) + 1e-8
        overlap = overlap / denom
    return overlap


def _compute_overlap_vectorized(cx, cy, w, h, batch_idx, block_mask,
                                 preplaced_mask=None, preplaced_weight=1.0,
                                 margin=0.0, quartic=False, normalise=False,
                                 max_weight=0.0):
    """Vectorized per-graph mean + alpha*max overlap loss.

    Builds one [G, Mmax, Mmax] pairwise-overlap tensor across the whole batch
    instead of looping `for g in batch_idx.unique()`. This eliminates the
    per-graph CPU<->GPU sync (the Python iterator over `unique()` would force
    a sync for every graph just to extract the scalar id) and replaces O(G)
    kernel launches with O(1). For training batches with G in the tens or
    hundreds the speedup is substantial.

    `block_mask` is a [N] bool selecting which nodes participate. Nodes outside
    the mask are excluded from BOTH sides of every pair (so e.g. pins do not
    appear as obstacles).

    `preplaced_mask` (optional, bool [N]) drives the asymmetric multiplier:
    pairs where exactly one endpoint is preplaced have their overlap scaled by
    `preplaced_weight`. This biases the model toward avoiding obstacles more
    aggressively than just avoiding other movable blocks. preplaced-preplaced
    pairs keep the default weight 1.0 (irrelevant in practice since both rows
    are clamped to GT before this is called, so no gradient passes through).
    """
    device = cx.device
    grp = _build_padded_groups(batch_idx, block_mask)
    if grp is None:
        return torch.tensor(0.0, device=device)
    valid_idx, vb_local, local_pos, G, Mmax, gidx_pad, pad_mask = grp

    cx_pad = cx.new_zeros((G, Mmax))
    cy_pad = cy.new_zeros((G, Mmax))
    w_pad  = w.new_zeros((G, Mmax))
    h_pad  = h.new_zeros((G, Mmax))
    cx_pad[vb_local, local_pos] = cx[valid_idx]
    cy_pad[vb_local, local_pos] = cy[valid_idx]
    w_pad[vb_local, local_pos]  = w[valid_idx]
    h_pad[vb_local, local_pos]  = h[valid_idx]

    hw_pad = w_pad / 2
    hh_pad = h_pad / 2

    # Pairwise axis-aligned intersection sizes, shape [G, Mmax, Mmax].
    dx = (torch.minimum(cx_pad.unsqueeze(2) + hw_pad.unsqueeze(2),
                        cx_pad.unsqueeze(1) + hw_pad.unsqueeze(1))
        - torch.maximum(cx_pad.unsqueeze(2) - hw_pad.unsqueeze(2),
                        cx_pad.unsqueeze(1) - hw_pad.unsqueeze(1)))
    dy = (torch.minimum(cy_pad.unsqueeze(2) + hh_pad.unsqueeze(2),
                        cy_pad.unsqueeze(1) + hh_pad.unsqueeze(1))
        - torch.maximum(cy_pad.unsqueeze(2) - hh_pad.unsqueeze(2),
                        cy_pad.unsqueeze(1) - hh_pad.unsqueeze(1)))
    rdx = F.relu(dx + margin)
    rdy = F.relu(dy + margin)
    if quartic:
        # v0.17: blend quartic with bilinear. Bilinear gives non-vanishing
        # gradient for thin overlaps where the quartic ≈ 0.
        ov = ((1.0 - OVERLAP_LINEAR_MIX) * (rdx ** 2) * (rdy ** 2)
              + OVERLAP_LINEAR_MIX * rdx * rdy)
    else:
        ov = rdx * rdy

    if normalise:
        area_pad = w_pad * h_pad
        denom = torch.minimum(area_pad.unsqueeze(2), area_pad.unsqueeze(1)) + 1e-8
        ov = ov / denom

    # Asymmetric preplaced multiplier. Compute pair weights only when needed.
    if preplaced_mask is not None and preplaced_weight != 1.0:
        pp_pad = torch.zeros((G, Mmax), dtype=torch.bool, device=device)
        pp_pad[vb_local, local_pos] = preplaced_mask[valid_idx]
        # XOR: exactly one endpoint is preplaced -> movable-vs-preplaced pair.
        is_asymmetric = pp_pad.unsqueeze(2) ^ pp_pad.unsqueeze(1)
        pair_w = 1.0 + (preplaced_weight - 1.0) * is_asymmetric.float()
        ov = ov * pair_w

    # Build the per-graph upper-triangular validity mask. We restrict to
    # i < j to match the old triu-based behaviour (each unordered pair once).
    triu_mask = torch.triu(
        torch.ones(Mmax, Mmax, dtype=torch.bool, device=device), diagonal=1
    ).unsqueeze(0)                                              # [1, Mmax, Mmax]
    valid_pair = pad_mask.unsqueeze(2) & pad_mask.unsqueeze(1)  # [G, Mmax, Mmax]
    valid_pair = valid_pair & triu_mask

    # Per-graph mean over valid pairs (graphs with <2 valid nodes give 0).
    n_pairs_per_g = valid_pair.sum(dim=(1, 2)).to(ov.dtype)     # [G]
    ov_masked = ov * valid_pair.to(ov.dtype)
    sum_per_g = ov_masked.sum(dim=(1, 2))                       # [G]
    has_pairs = n_pairs_per_g > 0                               # [G] bool
    mean_per_g = sum_per_g / n_pairs_per_g.clamp(min=1.0)       # safe div

    if max_weight > 0:
        # Set invalid pairs to -inf so they cannot win the max. Graphs with no
        # valid pairs then have amax = -inf — replace with 0 to be safe.
        neg_inf = torch.full_like(ov, float('-inf'))
        ov_for_max = torch.where(valid_pair, ov, neg_inf)
        max_per_g = ov_for_max.amax(dim=(1, 2))                 # [G]
        max_per_g = torch.where(has_pairs, max_per_g,
                                 torch.zeros_like(max_per_g))
        loss_per_g = mean_per_g + max_weight * max_per_g
    else:
        loss_per_g = mean_per_g

    # Average over graphs that actually had >=2 valid nodes (matches the old
    # `total / max(n_graphs, 1)` semantics).
    loss_per_g = torch.where(has_pairs, loss_per_g,
                              torch.zeros_like(loss_per_g))
    n_active = has_pairs.sum().clamp(min=1).to(loss_per_g.dtype)
    return loss_per_g.sum() / n_active


def _derive_block_dims(x0_hat, raw_area, node_s_x, node_s_y):
    """Shared (cx, cy, w, h) reconstruction from state + raw_area + scales."""
    cx = x0_hat[:, 0]
    cy = x0_hat[:, 1]
    # Defensive clamp: at high-t the x0_hat estimate can be extreme; cap log-ar
    # so exp() cannot overflow to inf. ar in [exp(-10), exp(10)] covers any
    # realistic aspect ratio.
    ar = torch.exp(x0_hat[:, 2].clamp(min=-10.0, max=10.0))
    raw_area = raw_area.squeeze(-1) if raw_area.dim() > 1 else raw_area
    # The +1e-12 MUST be inside each sqrt so the argument is never exactly 0.
    # For pin rows raw_area=0, so sqrt(raw_area/...) would be sqrt(0), whose
    # backward is grad/(2*sqrt(0)) = 0/0 = NaN. This contaminates the full
    # gradient when differentiating w.r.t. unclamped x0_hat (inference guidance).
    w_raw = torch.sqrt(raw_area * ar + 1e-12)
    h_raw = torch.sqrt(raw_area / (ar + 1e-12) + 1e-12)
    return cx, cy, w_raw * node_s_x, h_raw * node_s_y


def compute_overlap_loss(x0_hat, raw_area, batch_idx,
                         node_s_x, node_s_y,
                         quartic=False, margin=0.0, normalise=False,
                         max_weight=0.0, block_mask=None,
                         preplaced_mask=None, preplaced_weight=1.0):
    """Per-graph mean + alpha*max pairwise overlap penalty.

    Thin wrapper over the vectorized core. Width/height are reconstructed from
    raw_area + per-node scales (rectangular-canvas safe).

    `preplaced_mask` + `preplaced_weight` enable the asymmetric multiplier
    that biases the model toward avoiding preplaced blocks specifically (see
    the core helper docstring for the semantics).
    """
    cx, cy, w, h = _derive_block_dims(x0_hat, raw_area, node_s_x, node_s_y)
    if block_mask is None:
        block_mask = torch.ones_like(batch_idx, dtype=torch.bool)
    return _compute_overlap_vectorized(
        cx, cy, w, h, batch_idx, block_mask,
        preplaced_mask=preplaced_mask, preplaced_weight=preplaced_weight,
        margin=margin, quartic=quartic, normalise=normalise,
        max_weight=max_weight,
    )


def compute_overlap_loss_sampled(x0_hat, raw_area, batch_idx,
                                  node_s_x, node_s_y,
                                  sample_k=OVERLAP_SAMPLE_K, quartic=False,
                                  margin=0.0, normalise=False, max_weight=0.0,
                                  block_mask=None,
                                  preplaced_mask=None, preplaced_weight=1.0):
    """Sampled variant: restrict to a random subset of `sample_k` graphs.

    Uses the same vectorized core as compute_overlap_loss; the sampling is
    folded into the `block_mask` so the core never sees the un-sampled graphs.
    """
    unique = batch_idx.unique()
    if len(unique) <= sample_k:
        sampled_graphs = unique
    else:
        perm = torch.randperm(len(unique), device=batch_idx.device)
        sampled_graphs = unique[perm[:sample_k]]

    # Per-node mask "this node's graph is in the sample"
    sampled_node_mask = torch.isin(batch_idx, sampled_graphs)
    if block_mask is None:
        effective_mask = sampled_node_mask
    else:
        effective_mask = block_mask & sampled_node_mask

    cx, cy, w, h = _derive_block_dims(x0_hat, raw_area, node_s_x, node_s_y)
    return _compute_overlap_vectorized(
        cx, cy, w, h, batch_idx, effective_mask,
        preplaced_mask=preplaced_mask, preplaced_weight=preplaced_weight,
        margin=margin, quartic=quartic, normalise=normalise,
        max_weight=max_weight,
    )


# --------------------------------------------------------------------------- #
# Boundary loss (experiment b)                                                 #
# --------------------------------------------------------------------------- #
def compute_boundary_loss(x0_hat, boundary, raw_area, batch_idx,
                          node_s_x, node_s_y, block_mask):
    """Soft per-block boundary penalty (the contest 'boundary' soft constraint).

    A flagged block's relevant edge should coincide with the layout's bounding-
    box edge (measured against OUR own solution's bbox -> self-referential, so
    'satisfied' = the flagged block is one of the layout's extreme blocks).
    `boundary` is [N,4] = [L,R,T,B] bits. Returns the mean gap over all flagged
    (block, edge) pairs; gaps are >=0 by construction (distance to the per-graph
    extreme), so 0 = every flagged block is flush with its edge.

    Vectorized per-graph via _build_padded_groups (no Python loop / CPU sync).
    """
    device = x0_hat.device
    grp = _build_padded_groups(batch_idx, block_mask)
    if grp is None:
        return torch.tensor(0.0, device=device)
    valid_idx, vb_local, local_pos, G, Mmax, gidx_pad, pad_mask = grp
    cx, cy, w, h = _derive_block_dims(x0_hat, raw_area, node_s_x, node_s_y)
    left = cx - w / 2; right = cx + w / 2
    bot  = cy - h / 2; top   = cy + h / 2

    def _pad(vals, fill):
        p = vals.new_full((G, Mmax), fill)
        p[vb_local, local_pos] = vals[valid_idx]
        return p

    x_min = _pad(left,  float('inf')).min(dim=1).values    # [G]
    x_max = _pad(right, float('-inf')).max(dim=1).values
    y_min = _pad(bot,   float('inf')).min(dim=1).values
    y_max = _pad(top,   float('-inf')).max(dim=1).values

    gl = (left[valid_idx]  - x_min[vb_local]).clamp(min=0.0)   # L: left edge -> x_min
    gr = (x_max[vb_local]  - right[valid_idx]).clamp(min=0.0)  # R
    gt = (y_max[vb_local]  - top[valid_idx]).clamp(min=0.0)    # T
    gb = (bot[valid_idx]   - y_min[vb_local]).clamp(min=0.0)   # B
    gaps = torch.stack([gl, gr, gt, gb], dim=1)               # [nv,4]
    bits = boundary[valid_idx]                                # [nv,4]
    den = bits.sum().clamp(min=1.0)
    return (bits * gaps).sum() / den


# --------------------------------------------------------------------------- #
# Bbox area + composite energy                                                 #
# --------------------------------------------------------------------------- #
def compute_bbox_area_per_graph(x_state, raw_area, batch_idx,
                                node_s_x, node_s_y, block_mask=None):
    """Per-graph bounding-box area in normalized space.

    v0.13: rectangular-canvas safe.
    """
    cx = x_state[:, 0]
    cy = x_state[:, 1]
    ar = torch.exp(x_state[:, 2])
    raw_area = raw_area.squeeze(-1) if raw_area.dim() > 1 else raw_area
    w_raw = torch.sqrt(raw_area * ar + 1e-12)
    h_raw = torch.sqrt(raw_area / (ar + 1e-12))
    w = w_raw * node_s_x
    h = h_raw * node_s_y
    hw, hh = w / 2, h / 2
    x_lo = cx - hw; x_hi = cx + hw
    y_lo = cy - hh; y_hi = cy + hh

    if block_mask is None:
        block_mask = torch.ones_like(batch_idx, dtype=torch.bool)

    out = []
    for g in batch_idx.unique():
        m = (batch_idx == g) & block_mask
        if not m.any():
            out.append(torch.tensor(0.0, device=x_state.device))
            continue
        bbox_w = x_hi[m].max() - x_lo[m].min()
        bbox_h = y_hi[m].max() - y_lo[m].min()
        out.append(bbox_w * bbox_h)
    return torch.stack(out) if out else torch.zeros(0, device=x_state.device)


def compute_e_rel(pos, edge_index, edge_weight, batch_idx,
                  raw_area, x_state, hpwl_baseline, bbox_baseline,
                  node_s_x, node_s_y,
                  block_mask=None, kappa=KAPPA,
                  lam_hpwl=LAMBDA_HPWL, lam_over=LAMBDA_OVERLAP,
                  lam_bbox=LAMBDA_BBOX_EREL):
    """Composite energy -> relative-quality score in [0, 1].

    v0.16 bug fix: both `hpwl_baseline` and `bbox_baseline` are now used as
    ratio denominators, and the offset subtracted before the exp is
    `lam_hpwl + lam_bbox`. The fixed point of this formula on the GT solution
    is exactly E_rel = 1 (since hpwl/hpwl_baseline = 1, bbox/bbox_baseline = 1,
    and overlap = 0). Worse layouts produce hpwl or bbox ratios > 1, so the
    inner expression becomes positive and E_rel drops below 1.

    The previous formula divided bbox by a constant (4) instead of by the
    GT bbox, so E_rel on the GT sat at exp(-kappa*lam_bbox*bbox_gt/4) which
    is well below 1 — conditioning the model at inference with `c_emb = 1.0`
    then asked for a state the model had never been trained on.
    """
    hpwl    = compute_hpwl(pos, edge_index, edge_weight, batch_idx)
    overlap = compute_overlap_loss(x_state, raw_area, batch_idx,
                                    node_s_x, node_s_y,
                                    block_mask=block_mask)
    bbox    = compute_bbox_area_per_graph(x_state, raw_area, batch_idx,
                                           node_s_x, node_s_y,
                                           block_mask=block_mask).sum()

    hpwl_bl_safe = hpwl_baseline.clamp(min=1e-6)
    bbox_bl_safe = bbox_baseline.clamp(min=1e-6)

    energy = (lam_hpwl * (hpwl / hpwl_bl_safe)
              + lam_over * overlap
              + lam_bbox * (bbox / bbox_bl_safe))
    # Subtract the GT-fixed-point energy so E_rel(GT) = 1 exactly.
    return torch.exp(-kappa * (energy - lam_hpwl - lam_bbox)).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# Layout metrics (numpy, used for scoring)                                     #
# --------------------------------------------------------------------------- #
def compute_layout_metrics(coords_np, raw_area, edge_index, edge_weight,
                              pin_positions_np=None):
    """Bounding-box area + HPWL + total pairwise overlap area.

    coords_np : [N_blocks, 4] = [x_ctr, y_ctr, w, h] in raw units.
    pin_positions_np : optional [N_pins, 2] in the SAME raw frame.
        When provided, p2b edges are evaluated against these coordinates
        (HPWL_total = HPWL_int + HPWL_ext). Otherwise p2b edges are dropped.
    """
    x_c, y_c = coords_np[:, 0], coords_np[:, 1]
    widths   = coords_np[:, 2]
    heights  = coords_np[:, 3]
    n_blocks = coords_np.shape[0]
    x_min = (x_c - widths / 2).min()
    x_max = (x_c + widths / 2).max()
    y_min = (y_c - heights / 2).min()
    y_max = (y_c + heights / 2).max()
    bbox_area = float((x_max - x_min) * (y_max - y_min))

    if pin_positions_np is not None and len(pin_positions_np) > 0:
        pos_full = np.concatenate([coords_np[:, :2], pin_positions_np], axis=0)
        pos_t = torch.tensor(pos_full, dtype=torch.float32)
        ei = edge_index.cpu()
        ew = edge_weight.cpu()
    else:
        pos_t = torch.tensor(coords_np[:, :2], dtype=torch.float32)
        ei_cpu = edge_index.cpu()
        ew_cpu = edge_weight.cpu()
        if ei_cpu.numel() > 0:
            keep = (ei_cpu[0] < n_blocks) & (ei_cpu[1] < n_blocks)
            ei = ei_cpu[:, keep]
            ew = ew_cpu[keep]
        else:
            ei = ei_cpu
            ew = ew_cpu

    bt = torch.zeros(pos_t.shape[0], dtype=torch.long)
    hpwl_val = compute_hpwl(pos_t, ei, ew, bt).item()

    hw = widths / 2
    hh = heights / 2
    dx = (np.minimum(x_c[:, None] + hw[:, None], x_c[None, :] + hw[None, :])
        - np.maximum(x_c[:, None] - hw[:, None], x_c[None, :] - hw[None, :]))
    dy = (np.minimum(y_c[:, None] + hh[:, None], y_c[None, :] + hh[None, :])
        - np.maximum(y_c[:, None] - hh[:, None], y_c[None, :] - hh[None, :]))
    pair_overlap = np.maximum(dx, 0.0) * np.maximum(dy, 0.0)
    triu_mask = np.triu(np.ones_like(pair_overlap, dtype=bool), k=1)
    ov = float(pair_overlap[triu_mask].sum())
    return {'bbox_area': bbox_area, 'hpwl': hpwl_val, 'overlap': ov}


def compute_boundary_violation(coords_np, boundary_np, tol_frac=0.01):
    """Soft-constraint boundary satisfaction on a finished layout.

    A flagged block satisfies the constraint if >=1 of its flagged edges lies on
    the matching edge of the SELF-REFERENTIAL block bounding box (contest metric:
    L->x_min, R->x_max, T->y_max, B->y_min), within tol_frac of the bbox extent.

    coords_np   : [N, 4] = [x_ctr, y_ctr, w, h] in raw units.
    boundary_np : [N, 4] = [L, R, T, B] bits (0/1).
    Returns (violation_frac, n_flagged): fraction of flagged blocks NOT satisfied
    (0.0 when no block is flagged). This is V_boundary feeding the exp(2*V) penalty.
    """
    coords_np = np.asarray(coords_np)
    bnd = np.asarray(boundary_np) > 0.5
    flagged = bnd.any(axis=1)
    n_flagged = int(flagged.sum())
    if n_flagged == 0:
        return 0.0, 0
    x_c, y_c, w, h = coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]
    left = x_c - w / 2; right = x_c + w / 2
    bott = y_c - h / 2; topp = y_c + h / 2
    x_min, x_max = left.min(), right.max()
    y_min, y_max = bott.min(), topp.max()
    tol_x = tol_frac * max(x_max - x_min, 1e-9)
    tol_y = tol_frac * max(y_max - y_min, 1e-9)
    sat_L = (left - x_min) <= tol_x
    sat_R = (x_max - right) <= tol_x
    sat_T = (y_max - topp) <= tol_y
    sat_B = (bott - y_min) <= tol_y
    satisfied = ((bnd[:, 0] & sat_L) | (bnd[:, 1] & sat_R)
                 | (bnd[:, 2] & sat_T) | (bnd[:, 3] & sat_B))
    viol = float((flagged & ~satisfied).sum()) / n_flagged
    return viol, n_flagged


def compute_mib_violation(coords_np, mib_np, tol_frac=0.01):
    """Soft-constraint MIB satisfaction on a finished layout.

    MIB groups (same id in mib_np, id>0) require IDENTICAL (w,h). The contest
    violation is V_mib = sum_q (s_q - 1) where s_q = number of DISTINCT shapes in
    group q. Two blocks are the same shape if their w AND h agree within tol_frac.
    Handles arbitrarily many groups (test set has Q=1, but the spec allows more).

    coords_np : [N, 4] = [x_ctr, y_ctr, w, h] in raw units.
    mib_np    : [N] integer MIB group ids (0 = unconstrained).
    Returns (V_mib, n_multi_groups): raw violation count + number of groups with
    >=2 members (0.0/0 when none).
    """
    coords_np = np.asarray(coords_np)
    mib = np.asarray(mib_np)
    w = coords_np[:, 2]
    h = coords_np[:, 3]
    ids = np.unique(mib[mib > 0])
    V = 0.0
    n_groups = 0
    for g in ids:
        idx = np.where(mib == g)[0]
        if len(idx) < 2:
            continue
        n_groups += 1
        ww = w[idx]; hh = h[idx]
        used = np.zeros(len(idx), dtype=bool)
        s = 0
        for i in range(len(idx)):
            if used[i]:
                continue
            s += 1
            used[i] = True
            for j in range(i + 1, len(idx)):
                if used[j]:
                    continue
                if (abs(ww[i] - ww[j]) <= tol_frac * max(ww[i], 1e-9) and
                        abs(hh[i] - hh[j]) <= tol_frac * max(hh[i], 1e-9)):
                    used[j] = True
        V += (s - 1)
    return float(V), n_groups


def blocks_connected(box_a, box_b, touch_tol=0.0):
    """Do two lower-left boxes (x, y, w, h) share a border the SCORER counts as connecting?

    The scorer merges a group's rectangles with shapely `unary_union` and counts the
    resulting polygons. GEOS merges two boxes into ONE polygon only when they share a
    border segment of POSITIVE LENGTH: one axis must touch-or-overlap (gap <= 0) while the
    other STRICTLY overlaps (gap < 0). Two consequences we rely on, both verified against
    GEOS directly:
      * a CORNER-only contact stays a MultiPolygon => still a violation;
      * there is NO tolerance -- a 1e-15 gap disconnects.
    So abutment means a shared edge, not proximity. `touch_tol` > 0 loosens the gap<=0 side
    into a proximity proxy; it is NOT the scorer's rule and exists only for diagnostics.
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    gap_x = max(ax, bx) - min(ax + aw, bx + bw)
    gap_y = max(ay, by) - min(ay + ah, by + bh)
    return ((gap_x <= touch_tol and gap_y < 0.0) or
            (gap_y <= touch_tol and gap_x < 0.0))


def grouping_components(pos_ll, idx, touch_tol=0.0):
    """Number of connected components among the boxes `idx` of a lower-left [N,4] layout,
    under the scorer's shared-edge rule (see blocks_connected)."""
    m = len(idx)
    parent = list(range(m))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(m):
        for b in range(a + 1, m):
            if blocks_connected(pos_ll[idx[a]], pos_ll[idx[b]], touch_tol):
                parent[find(a)] = find(b)
    return len({find(a) for a in range(m)})


def compute_grouping_violation(coords_np, grouping_np, touch_frac=0.0):
    """Soft-constraint grouping satisfaction (connectivity) on a finished layout.

    Grouping groups (same id, id>0) must ABUT into ONE connected component. The contest
    violation is V_grouping = sum_p (c_p - 1) over groups.

    touch_frac=0 (default) is the EXACT scorer rule: members must share a border segment of
    positive length. It used to default to 0.02, i.e. "within 2% of the layout extent counts
    as abutting" -- a proximity proxy for a pre-legalization layout that essentially never
    touches exactly. That ruler read ~3-4 violations/sample where the scorer reads ~14-16, so
    every GrpViol number before 2026-07-14 is on the loose scale and is NOT comparable to
    V_grp from tools/compute_vrel.py. Pass touch_frac>0 only for that legacy diagnostic.

    coords_np    : [N, 4] = [x_ctr, y_ctr, w, h] in raw units.
    grouping_np  : [N] integer group ids (0 = unconstrained).
    Returns (V_grouping, n_multi_groups).
    """
    coords_np = np.asarray(coords_np, dtype=np.float64)
    grp = np.asarray(grouping_np)
    x = coords_np[:, 0]; y = coords_np[:, 1]
    w = coords_np[:, 2]; h = coords_np[:, 3]
    pos_ll = np.stack([x - w / 2, y - h / 2, w, h], axis=1)
    ext = max((x + w / 2).max() - (x - w / 2).min(),
              (y + h / 2).max() - (y - h / 2).min(), 1e-9)
    tol = touch_frac * ext
    V = 0.0
    n_groups = 0
    for g in np.unique(grp[grp > 0]):
        idx = np.where(grp == g)[0]
        if len(idx) < 2:
            continue
        n_groups += 1
        V += grouping_components(pos_ll, idx, touch_tol=tol) - 1
    return float(V), n_groups


# --------------------------------------------------------------------------- #
# Canvas / pin helpers                                                         #
# --------------------------------------------------------------------------- #
def get_canvas_params(graph):
    """Return (s_x, s_y, offset_x, offset_y) as Python floats."""
    s_x      = float(graph.canvas_s_x.item())
    s_y      = float(graph.canvas_s_y.item())
    offset_x = float(graph.canvas_offset_x.item())
    offset_y = float(graph.canvas_offset_y.item())
    return s_x, s_y, offset_x, offset_y


def extract_pin_positions_raw(graph):
    """Recover pin positions in raw coords."""
    if not hasattr(graph, 'is_pin') or not graph.is_pin.any():
        return None
    s_x, s_y, offset_x, offset_y = get_canvas_params(graph)
    pin_state = graph.x_gt[graph.is_pin].cpu().numpy()
    # Inverse: x_raw = x_norm / s_x + offset_x
    x_raw = pin_state[:, 0] / s_x + offset_x
    y_raw = pin_state[:, 1] / s_y + offset_y
    return np.stack([x_raw, y_raw], axis=1)
