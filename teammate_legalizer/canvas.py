"""Canvas normalization, size reconstruction, and spatial-edge construction.

Mirrors cell 22 of the original notebook one-to-one.
"""

import torch

from .config import (
    USE_PIN_DEFINED_CANVAS, CANVAS_FRAC,
)


# --------------------------------------------------------------------------- #
# Canvas normalization                                                         #
# --------------------------------------------------------------------------- #
def compute_canvas_norm(raw_area, pins_pos=None):
    """Compute aspect-preserving canvas normalization parameters.

    v0.13 (aspect-preserving variant): pick an ISOTROPIC scale `s` so the
    canvas is mapped to constant area (= 4 * CANVAS_FRAC^2) while preserving
    its aspect ratio. With s_x = s_y = s:
        - raw aspect == normalized aspect (no stretching of block log_ar)
        - normalized canvas area is constant across samples
        - normalized canvas SHAPE matches the design's true canvas shape

    Returns
    -------
    s_x, s_y     : equal scale factors (aspect-preserving)
    offset_x, offset_y : raw-frame canvas centroid
    half_w, half_h : NORMALIZED half-extents of the canvas
        (their product equals CANVAS_FRAC^2, so canvas area = 4*CANVAS_FRAC^2)
    mode         : 'pin' or 'area'

    Forward transform: x_norm = (x_raw - offset_x) * s_x.

    Pin-defined branch fires when >=2 pins span both axes (non-zero range in
    BOTH x and y). Otherwise we fall back to constant-total-area (where the
    raw canvas's shape is unknown, so we map it to a square with area = 4).
    """
    total_area = float(raw_area.sum().clamp(min=1e-8))
    s_lin_area = (1.0 / total_area) ** 0.5

    use_pin = False
    if USE_PIN_DEFINED_CANVAS and pins_pos is not None and pins_pos.numel() > 0:
        valid = (pins_pos[:, 0] != -1) & (pins_pos[:, 1] != -1)
        pp = pins_pos[valid] if valid.any() else pins_pos
        if pp.shape[0] >= 2:
            x_lo, _ = pp[:, 0].min(dim=0); x_hi, _ = pp[:, 0].max(dim=0)
            y_lo, _ = pp[:, 1].min(dim=0); y_hi, _ = pp[:, 1].max(dim=0)
            half_w_raw = float((x_hi - x_lo) / 2.0)
            half_h_raw = float((y_hi - y_lo) / 2.0)
            if half_w_raw > 1e-6 and half_h_raw > 1e-6:
                # Aspect-preserving: single scale s such that the normalized
                # canvas area is 4 * CANVAS_FRAC^2 (matching the v0.12 area=4
                # when CANVAS_FRAC=1).
                #   (2*half_w_raw*s) * (2*half_h_raw*s) = 4 * CANVAS_FRAC^2
                #   => s = CANVAS_FRAC / sqrt(half_w_raw * half_h_raw)
                s = CANVAS_FRAC / (half_w_raw * half_h_raw) ** 0.5
                s_x = s
                s_y = s
                offset_x = float((x_lo + x_hi) / 2.0)
                offset_y = float((y_lo + y_hi) / 2.0)
                half_w = half_w_raw * s
                half_h = half_h_raw * s
                use_pin = True

    if not use_pin:
        # Constant-total-area fallback: assume a square raw canvas of area
        # 1/s_lin_area^2 = total_area. Map to [-1, +1]^2 (area 4).
        s_x = s_lin_area
        s_y = s_lin_area
        half_raw = 1.0 / s_lin_area / 2.0
        offset_x = half_raw
        offset_y = half_raw
        half_w = 1.0
        half_h = 1.0

    mode = 'pin' if use_pin else 'area'
    return s_x, s_y, offset_x, offset_y, half_w, half_h, mode


# --------------------------------------------------------------------------- #
# Edge gaps + size reconstruction                                              #
# --------------------------------------------------------------------------- #
def compute_edge_gaps(edge_index, pos, sizes, squash=True):
    """Signed gap (per axis) for every edge, optionally tanh-squashed."""
    if edge_index.numel() == 0:
        return torch.zeros(0, 2, device=pos.device, dtype=pos.dtype)
    src, dst = edge_index
    dpos = (pos[src] - pos[dst]).abs()
    dsize = (sizes[src] + sizes[dst]) / 2
    gap = dpos - dsize
    if squash:
        gap = torch.tanh(gap)
    return gap


def reconstruct_sizes_rect(x_state, raw_area, node_s_x, node_s_y):
    """Derive normalized (w, h) for rectangular canvases.

    w_norm = sqrt(raw_area * ar) * s_x
    h_norm = sqrt(raw_area / ar) * s_y

    All inputs are [N]-shaped (raw_area, node_s_x, node_s_y) or [N, 3] (x_state).
    Pin rows have raw_area = 0 so their reconstructed sizes are 0 — invisible
    to overlap, bbox, and edge-distance K-NN.
    """
    ar = torch.exp(x_state[:, 2])
    raw_area = raw_area.squeeze(-1) if raw_area.dim() > 1 else raw_area
    w_raw = torch.sqrt(raw_area * ar + 1e-12)
    h_raw = torch.sqrt(raw_area / (ar + 1e-12))
    w_norm = w_raw * node_s_x
    h_norm = h_raw * node_s_y
    return torch.stack([w_norm, h_norm], dim=-1)


def reconstruct_sizes(x_state, batch_data_or_graph):
    """Convenience wrapper: looks up raw_area, node_s_x, node_s_y from the
    Data/Batch object and returns [N, 2] normalized (w, h)."""
    return reconstruct_sizes_rect(
        x_state, batch_data_or_graph.raw_area,
        batch_data_or_graph.node_s_x, batch_data_or_graph.node_s_y)


# --------------------------------------------------------------------------- #
# Spatial K-NN edges (vectorized over graphs in a batch)                       #
# --------------------------------------------------------------------------- #
_PAD_GROUP_CACHE = {'key': None, 'refs': None, 'val': None}


def _build_padded_groups(batch_idx, valid_mask):
    """Group valid nodes by graph into dense padded tensors (no Python loop).

    Returns a tuple of index structures used by both spatial-KNN builders, or
    ``None`` if there are fewer than 2 valid nodes in every graph.

    MEMOIZED (size-1) because the result depends only on the batch layout and the
    valid mask, and the sampler rebuilds edges every Heun step with both held fixed --
    ~1580 identical recomputations per case, each paying a `nonzero` plus two `int(...)`
    device->host syncs. Keying on object identity is sound because the entry keeps a
    STRONG reference to both tensors, so no other live object can reuse their ids.
    Callers must therefore not mutate either tensor in place; the sampler and the
    training augmenter both build them fresh instead.
    """
    cache = _PAD_GROUP_CACHE
    key = (id(batch_idx), id(valid_mask))
    if cache['key'] == key:
        return cache['val']
    val = _build_padded_groups_uncached(batch_idx, valid_mask)
    cache['key'] = key
    cache['refs'] = (batch_idx, valid_mask)   # pins the ids the key depends on
    cache['val'] = val
    return val


def _build_padded_groups_uncached(batch_idx, valid_mask):
    device = batch_idx.device
    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
    if valid_idx.numel() == 0:
        return None
    vb = batch_idx[valid_idx]
    # Dense, contiguous graph ids in case some graphs have 0 valid nodes.
    _, vb_local = torch.unique(vb, sorted=True, return_inverse=True)
    G = int(vb_local.max()) + 1
    counts = torch.bincount(vb_local, minlength=G)
    Mmax = int(counts.max())
    if Mmax < 2:
        return None
    offsets = torch.zeros(G, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(counts, 0)[:-1]
    local_pos = torch.arange(valid_idx.numel(), device=device) - offsets[vb_local]
    gidx_pad = valid_idx.new_full((G, Mmax), -1)
    gidx_pad[vb_local, local_pos] = valid_idx
    pad_mask = gidx_pad >= 0
    return valid_idx, vb_local, local_pos, G, Mmax, gidx_pad, pad_mask


def _knn_edges_from_scores(scores, gidx_pad, pad_mask, k):
    """Given a per-graph [G,Mmax,Mmax] score matrix (lower = closer), select
    each real row's k nearest real columns and map back to global ids.
    """
    G, Mmax, _ = scores.shape
    kk = min(k, Mmax - 1)
    topv, topi = scores.topk(kk, dim=2, largest=False)
    device = scores.device
    src_local = torch.arange(Mmax, device=device).view(1, Mmax, 1).expand(G, Mmax, kk)
    g_ar = torch.arange(G, device=device).view(G, 1, 1).expand(G, Mmax, kk)
    valid_edge = pad_mask.unsqueeze(2) & torch.isfinite(topv)
    g_sel = g_ar[valid_edge]
    r_sel = src_local[valid_edge]
    c_sel = topi[valid_edge]
    src = gidx_pad[g_sel, r_sel]
    dst = gidx_pad[g_sel, c_sel]
    return torch.stack([src, dst], dim=0).long()


def build_spatial_edges_centroid(pos, batch_idx, k, valid_mask=None):
    """K-nearest by squared centroid distance, within each graph."""
    if k <= 0:
        return (torch.zeros(2, 0, dtype=torch.long, device=pos.device),
                torch.zeros(0, dtype=torch.long, device=pos.device))
    if valid_mask is None:
        valid_mask = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
    grp = _build_padded_groups(batch_idx, valid_mask)
    if grp is None:
        return (torch.zeros(2, 0, dtype=torch.long, device=pos.device),
                torch.zeros(0, dtype=torch.long, device=pos.device))
    valid_idx, vb_local, local_pos, G, Mmax, gidx_pad, pad_mask = grp
    D = pos.shape[1]
    pos_pad = pos.new_zeros((G, Mmax, D))
    pos_pad[vb_local, local_pos] = pos[valid_idx]
    diff = pos_pad.unsqueeze(2) - pos_pad.unsqueeze(1)
    d2 = (diff * diff).sum(dim=-1)
    d2 = d2.masked_fill(~pad_mask.unsqueeze(1), float('inf'))
    eye = torch.eye(Mmax, dtype=torch.bool, device=pos.device).unsqueeze(0)
    d2 = d2.masked_fill(eye, float('inf'))
    edge_index = _knn_edges_from_scores(d2, gidx_pad, pad_mask, k)
    return edge_index, torch.zeros(0)


def build_spatial_edges_edge_dist(pos, sizes, batch_idx, k, valid_mask=None):
    """K-nearest by signed Chebyshev gap, within each graph."""
    if k <= 0:
        return (torch.zeros(2, 0, dtype=torch.long, device=pos.device),
                torch.zeros(0, dtype=torch.long, device=pos.device))
    if valid_mask is None:
        valid_mask = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
    grp = _build_padded_groups(batch_idx, valid_mask)
    if grp is None:
        return (torch.zeros(2, 0, dtype=torch.long, device=pos.device),
                torch.zeros(0, dtype=torch.long, device=pos.device))
    valid_idx, vb_local, local_pos, G, Mmax, gidx_pad, pad_mask = grp
    D = pos.shape[1]
    pos_pad = pos.new_zeros((G, Mmax, D))
    siz_pad = sizes.new_zeros((G, Mmax, D))
    pos_pad[vb_local, local_pos] = pos[valid_idx]
    siz_pad[vb_local, local_pos] = sizes[valid_idx]
    dpos = (pos_pad.unsqueeze(2) - pos_pad.unsqueeze(1)).abs()
    dsize = (siz_pad.unsqueeze(2) + siz_pad.unsqueeze(1)) / 2
    cheb = (dpos - dsize).max(dim=-1).values
    cheb = cheb.masked_fill(~pad_mask.unsqueeze(1), float('inf'))
    eye = torch.eye(Mmax, dtype=torch.bool, device=pos.device).unsqueeze(0)
    cheb = cheb.masked_fill(eye, float('inf'))
    edge_index = _knn_edges_from_scores(cheb, gidx_pad, pad_mask, k)
    return edge_index, torch.zeros(0)
