"""ICCAD 2026 v10 scorer arithmetic shared by local tools.

The authoritative implementation is ``contest/iccad2026_evaluate.py``. Keep these small,
dependency-free helpers centralized so reporting and post-processing acceptance rules do not
silently drift from the official denominator or size weighting.
"""

import math

import numpy as np


ALPHA = 0.5
BETA = 2.0
GAMMA = 0.3
M_PENALTY = 10.0


def v10_soft_denominator(boundary, mib, grouping):
    """Return v10 ``N_soft``: boundary + grouping + MIB maximum violations."""
    boundary = np.asarray(boundary, dtype=int)
    mib = np.asarray(mib, dtype=int)
    grouping = np.asarray(grouping, dtype=int)
    n_soft = int((boundary != 0).sum())
    for ids in (mib, grouping):
        for group_id in np.unique(ids[ids > 0]):
            n_soft += max(0, int((ids == group_id).sum()) - 1)
    return n_soft


def v10_size_weights(block_counts):
    """Return normalized v10 aggregate weights proportional to ``exp(n_blocks / 12)``."""
    counts = np.asarray(block_counts, dtype=float)
    if counts.size == 0:
        return counts
    weights = np.exp((counts - counts.max()) / 12.0)
    return weights / weights.sum()


def v10_violation_counts(pos_ll, boundary, mib, grouping, boundary_tol=1e-6):
    """Return scorer-exact soft-constraint counts from a lower-left layout."""
    pos = np.asarray(pos_ll, dtype=float)
    boundary = np.asarray(boundary, dtype=int)
    mib = np.asarray(mib, dtype=int)
    grouping = np.asarray(grouping, dtype=int)
    x, y, w, h = pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3]

    x0, y0 = x.min(), y.min()
    x1, y1 = (x + w).max(), (y + h).max()
    v_boundary = 0
    for i, code in enumerate(boundary):
        if code == 0:
            continue
        touches = {
            1: abs(x[i] - x0) < boundary_tol,
            2: abs(x[i] + w[i] - x1) < boundary_tol,
            4: abs(y[i] + h[i] - y1) < boundary_tol,
            8: abs(y[i] - y0) < boundary_tol,
        }
        if not all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
            v_boundary += 1

    v_mib = 0
    for group_id in np.unique(mib[mib > 0]):
        idx = np.where(mib == group_id)[0]
        shapes = {(round(float(w[i]), 4), round(float(h[i]), 4)) for i in idx}
        v_mib += len(shapes) - 1

    def connected(i, j):
        gx = max(x[i], x[j]) - min(x[i] + w[i], x[j] + w[j])
        gy = max(y[i], y[j]) - min(y[i] + h[i], y[j] + h[j])
        return (gx <= 0.0 and gy < 0.0) or (gy <= 0.0 and gx < 0.0)

    v_grouping = 0
    for group_id in np.unique(grouping[grouping > 0]):
        idx = [int(i) for i in np.where(grouping == group_id)[0]]
        if len(idx) < 2:
            continue
        parent = {i: i for i in idx}

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for offset, i in enumerate(idx):
            for j in idx[offset + 1:]:
                if connected(i, j):
                    ri, rj = root(i), root(j)
                    if ri != rj:
                        parent[rj] = ri
        v_grouping += len({root(i) for i in idx}) - 1

    return {
        'boundary': int(v_boundary),
        'mib': int(v_mib),
        'grouping': int(v_grouping),
        'total': int(v_boundary + v_mib + v_grouping),
    }


def v10_total_hpwl(pos_ll, b2b_conn, p2b_conn=None, pins_pos=None):
    """Scorer-exact total HPWL: b2b + p2b, weighted Manhattan CENTER distance.

    A verbatim port of `calculate_hpwl_b2b` / `calculate_hpwl_p2b` in
    `contest/iccad2026_evaluate.py`: iterate every connectivity ROW once, skip `-1` padding,
    skip out-of-range indices, accumulate `weight * (|dx| + |dy|)` between block centers (or
    between a pin's absolute position and a block center).

    Do NOT compute this from the model's `edge_index_full`. That graph is built by
    `data._safe_canonicalise_edges`, which (a) emits each unordered pair in BOTH directions
    and (b) collapses duplicate rows for the same pair to their MAX weight. (a) is a harmless
    factor of 2, but (b) DISCARDS weight mass -- 37% of it on the official n=120 case, where
    7056 rows collapse to 4675 pairs -- and the loss varies per instance (5% of rows are
    duplicates at n=50, 34% at n=120). So it is not a constant factor and does not cancel out
    of `HPWL_gap`. Using the graph here made our reported totals differ from the official
    scorer's; see docs/experiments/RUNTIME_REPRICING_SCREEN.md.
    """
    pos = np.asarray(pos_ll, dtype=float)
    n = len(pos)
    cx = pos[:, 0] + pos[:, 2] / 2.0
    cy = pos[:, 1] + pos[:, 3] / 2.0

    total = 0.0
    b = np.asarray(b2b_conn, dtype=float)
    if b.ndim == 3:
        b = b[0]
    if b.size:
        b = b[b[:, 0] != -1]
        b = b[(b[:, 0] < n) & (b[:, 1] < n)]
        if b.size:
            i = b[:, 0].astype(int)
            j = b[:, 1].astype(int)
            total += float((b[:, 2] * (np.abs(cx[i] - cx[j]) + np.abs(cy[i] - cy[j]))).sum())

    if p2b_conn is not None and pins_pos is not None:
        p = np.asarray(p2b_conn, dtype=float)
        pins = np.asarray(pins_pos, dtype=float)
        if p.ndim == 3:
            p = p[0]
        if pins.ndim == 3:
            pins = pins[0]
        if p.size and pins.size:
            p = p[p[:, 0] != -1]
            p = p[(p[:, 1] < n) & (p[:, 0] < len(pins))]
            if p.size:
                pi = p[:, 0].astype(int)
                bi = p[:, 1].astype(int)
                total += float((p[:, 2] * (np.abs(pins[pi, 0] - cx[bi]) +
                                           np.abs(pins[pi, 1] - cy[bi]))).sum())
    return total


def v10_cost(hpwl_gap, area_gap, violations_relative, runtime_factor=1.0,
             feasible=True):
    """Compute official v10 per-instance cost (neutral runtime when omitted)."""
    if not feasible:
        return M_PENALTY
    quality = 1.0 + ALPHA * (max(0.0, hpwl_gap) + max(0.0, area_gap))
    violation = math.exp(BETA * violations_relative)
    runtime = max(0.7, max(0.01, runtime_factor) ** GAMMA)
    return min(quality * violation * runtime, M_PENALTY - 1e-6)


def v10_total_score(costs, block_counts):
    """Compute the v10 size-weighted average of per-instance costs."""
    costs = np.asarray(costs, dtype=float)
    if costs.size == 0:
        return 0.0
    if len(costs) != len(block_counts):
        raise ValueError('costs and block_counts must have the same length')
    return float(np.dot(costs, v10_size_weights(block_counts)))
