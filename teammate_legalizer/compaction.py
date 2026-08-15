"""Joint shape+position constraint-graph compaction (v1.5 post-process, "M9").

Given a near-legal placement (e.g. the v1.4 diffusion output), this shrinks the layout
outline by choosing block **aspect ratios and positions jointly** — the lever the whole
fixed-shape method family (v1.0–v1.4) never touched. It is TOFU's legalization formulation
(Kai et al., "TOFU: A Two-Step Floorplan Refinement Framework for Whitespace Reduction",
their Eq. 2): build horizontal/vertical precedence graphs from the current relative
positions, then solve ONE convex program over (x, y, w, h):

    minimize  W + H
    s.t.  x_i + w_i <= x_j          for i->j in G_h   (no slack => overlap-free)
          y_i + h_i <= y_j          for i->j in G_v
          x + w <= W,  y + h <= H,  x,y >= 0
          h_i >= a_i / w_i          (convex; binds at the min-outline optimum)
          1/AR_MAX <= w_i/h_i <= AR_MAX     (soft blocks; the dataset uses AR in [1/3,3])
          fixed-shape / preplaced (w,h) fixed; preplaced (x,y) pinned
          MIB group q: members share ONE (w,h) var pair  => V_mib = 0 by construction

The precedence graph is rebuilt from the new solution and re-solved a few times (TOFU's
outer loop). Two things that MUST be done or the result is invalid (both learned the hard
way, see docs/experiments/AREA_UTIL_SCREEN.md):
  1. The area relaxation h >= a/w is LOOSE off the critical path — the solver inflates free
     blocks (w*h >> a) to fake whitespace-filling, breaking the 1% area HARD constraint.
     Fix: rescale free blocks back to EXACT area (centered shrink, never adds overlap).
  2. Preplaced blocks need x,y pinned, not just w,h, or the solve relocates them (invalid).

Infeasibility: the precedence graph is acyclic, but when PREPLACED blocks are pinned at fixed
positions, a heavily-scrambled input can yield a relative ordering that puts a movable block on
the wrong side of an immovable anchor (forced both left-of and right-of a pinned block) => the
conic solve is infeasible. We retry from a position-only longest-path pre-clean; if it still
can't legalize, the ORIGINAL placement is returned unchanged (monotone — never worse than the
input; the downstream legalizer finishes it). On official-100 this hits 1 sample.

Official-100 (post-process on the v1.4 placement): utilization 0.751 -> 0.866 (+15%), outline
ratio vs GT 1.297 -> 1.125, overlap 15.7 -> 1.8, HPWL -4%, MibViol 0, GrpViol 3.55 -> 2.35.

Requires cvxpy + a conic solver (CLARABEL). coords convention: [N,4] = [cx,cy,w,h].

`legalize_sample()` is the ONE entry point that turns a raw placement into a submittable
one (compaction + contact gap + preplaced-frame alignment + optional scorer-exact grouping
snap + boundary pull). Both `infer.py` (in-process,
the real submission path) and `tools/compact_joint.py` (the offline JSON post-process)
call it, so there is a single definition of "what we ship".

(v1.5.4: this module moved from `tools/compact_joint.py` into the package. It is production
code -- infer.py depends on it -- and importing it out of tools/ from an entrypoint was
wrong. tools/compact_joint.py is now just the CLI shim.)

(v1.5.6: `snap_groups` plus `lam_grp=0.2` makes grouping abut under the scorer's exact
positive-border/zero-gap predicate; `pull_boundary` lands near-miss flagged blocks on their
self-referential bbox wall. Both are feasibility-gated and default on through `infer.py`.)
"""
from collections import Counter, defaultdict

import numpy as np

from .config import (
    CLUSTER_MAX_SIZE, CLUSTER_MIN_DENSITY, CLUSTER_MIN_OWED, CLUSTER_MIN_SIZE,
    CLUSTER_BATCH, CLUSTER_MAX_WASTE, CLUSTER_SUB_LIGHT,
    PRICED_HPWL, RESOLVE, SEP_NET_Q, SNAP_CROSS_FRAC,
    SNAP_MAX_AREA_GROWTH, SOLVE_RESEED_LAM,
    SOLVE_RESEED_ROUNDS,
)
from .scoring import ALPHA, BETA, v10_soft_denominator, v10_total_hpwl

try:
    import cvxpy as cp
    _HAVE_CVXPY = True
except Exception:
    _HAVE_CVXPY = False

AR_MAX = 3.0
_EPS = 1e-6

# Tie-break for precedence pairs that are separated on BOTH axes -- 67% of all edges on raw
# sampler output (measured over 43k pairs). Both axis choices preserve non-overlap for such a
# pair, so min-displacement is not a criterion here; the choice only decides which longest-path
# chain the edge lands on, i.e. W or H. 'smaller' is the PRE-v1.5.11 rule (inherited from the
# overlapping-pair min-penetration branch) and stacks a far-apart-in-x, near-in-y pair
# VERTICALLY. See docs/experiments/AREA_SELECTION_SCREEN.md, E6.
PREC_SEP_RULES = ('smaller', 'larger')
PREC_SEP_RULE = 'larger'      # v1.5.11; 'smaller' was the pre-v1.5.11 default
GAP_FRAC = 1e-3   # per-block shrink to keep packed contacts feasible under the exact scorer


# --------------------------------------------------------------------------- #
# position-only longest-path compaction (feasible fallback / pre-clean)        #
# --------------------------------------------------------------------------- #
def _compact_axis(cx, cy, w, h, immov, axis):
    c = cx if axis == 0 else cy
    s = w if axis == 0 else h
    oc = cy if axis == 0 else cx
    osz = h if axis == 0 else w
    lo = c - s / 2.0
    olo = oc - osz / 2.0; ohi = oc + osz / 2.0
    wall = lo.min()
    order = np.argsort(lo, kind='stable')
    new_lo = lo.copy(); placed = []
    for i in order:
        if immov[i]:
            new_lo[i] = lo[i]; placed.append(i); continue
        cand = wall
        for j in placed:
            if ohi[i] > olo[j] + 1e-9 and ohi[j] > olo[i] + 1e-9:
                cand = max(cand, new_lo[j] + s[j])
        new_lo[i] = cand; placed.append(i)
    return new_lo + s / 2.0


def _bbox(cx, cy, w, h):
    return ((cx + w / 2).max() - (cx - w / 2).min()) * ((cy + h / 2).max() - (cy - h / 2).min())


def compact_positions(coords, is_pp, iters=12):
    """Longest-path x/y compaction (shapes frozen). Preplaced pinned. Monotone in bbox."""
    cx = coords[:, 0].astype(float).copy(); cy = coords[:, 1].astype(float).copy()
    w = coords[:, 2].astype(float); h = coords[:, 3].astype(float)
    prev = _bbox(cx, cy, w, h)
    for _ in range(iters):
        cx = _compact_axis(cx, cy, w, h, is_pp, 0)
        cy = _compact_axis(cx, cy, w, h, is_pp, 1)
        cur = _bbox(cx, cy, w, h)
        if cur > prev - 1e-6:
            break
        prev = cur
    out = coords.astype(float).copy(); out[:, 0] = cx; out[:, 1] = cy
    return out


# --------------------------------------------------------------------------- #
# joint shape+position solve                                                   #
def _escape_blocked(m, p, axis, is_pp, xlo, xhi, ylo, yhi, cx, cy, tol=1e-3):
    """Would separating movable `m` from PINNED `p` along `axis` push m into another pinned
    block? True when a second pinned block abuts p on the side m must escape to (a zero-gap
    seam) and spans m on the cross-axis — i.e. the slot m is being sent to has no room.
    axis 0 = horizontal (escape left/right), 1 = vertical (escape up/down)."""
    N = len(cx)
    for k in range(N):
        if k == p or not is_pp[k]:
            continue
        if axis == 1:                                   # escaping in y: need x-overlap with m
            if min(xhi[k], xhi[m]) - max(xlo[k], xlo[m]) <= 0:
                continue
            if cy[m] > cy[p]:                           # m goes UP: is k stacked on top of p?
                if abs(ylo[k] - yhi[p]) < tol:
                    return True
            else:                                       # m goes DOWN: is k under p?
                if abs(ylo[p] - yhi[k]) < tol:
                    return True
        else:                                           # escaping in x: need y-overlap with m
            if min(yhi[k], yhi[m]) - max(ylo[k], ylo[m]) <= 0:
                continue
            if cx[m] > cx[p]:                           # m goes RIGHT: is k right of p?
                if abs(xlo[k] - xhi[p]) < tol:
                    return True
            else:                                       # m goes LEFT: is k left of p?
                if abs(xlo[p] - xhi[k]) < tol:
                    return True
    return False


# --------------------------------------------------------------------------- #
def net_sep_pairs(b2b_conn, n, quantile=0.90):
    """H3. The set of heavy block-to-block pairs the wirelength tie-break applies to.

    Scored the way the evaluator scores: every raw connectivity ROW is summed, so duplicate
    rows for one pair add up instead of collapsing to their max (SCORER_FIDELITY_BUGS.md B1).
    Returns the pairs at or above the given weight quantile of THIS case's pair distribution.

    Why a quantile rather than all pairs: H13 measured that 79% of our b2b excess sits in the
    heaviest weight decile -- 37.6% of the netlist's weight mass but only 14.5% of GT's
    wirelength, because GT abuts those pairs and we place them 1.6x farther apart. Applying the
    rule to every pair would spend the DAG's one free degree of freedom on nets that are not
    where the money is, and the tie-break is also v1.5.11's util lever.
    """
    quantile = min(max(float(quantile), 0.0), 1.0)
    b = np.asarray(b2b_conn, float)
    if b.ndim == 3:
        b = b[0]
    if not b.size:
        return frozenset()
    b = b[b[:, 0] != -1]
    b = b[(b[:, 0] < n) & (b[:, 1] < n) & (b[:, 0] != b[:, 1])]
    if not b.size:
        return frozenset()
    acc = {}
    for i, j, wt in b[:, :3]:
        key = (int(min(i, j)), int(max(i, j)))
        acc[key] = acc.get(key, 0.0) + float(wt)
    if not acc:
        return frozenset()
    thr = float(np.quantile(np.fromiter(acc.values(), float), quantile))
    return frozenset(k for k, v in acc.items() if v >= thr)


def _precedence(cx, cy, w, h, is_pp=None, sep_rule=None, net_pairs=None, net_invert=False):
    """Relative-position graph. Overlapping pairs are separated along the MIN-PENETRATION
    axis (cheapest displacement).

    PINNED-AWARE REPAIR (is_pp given): that greedy rule is wrong when the cheap axis is
    blocked by immovable geometry. On official n=95, two preplaced blocks abut EXACTLY
    (zero gap) and a movable straddles the seam; min-penetration sends it vertically into
    that zero-height slot -> it must be both above one anchor and below the other => the
    conic solve is INFEASIBLE, even though there is free space to the side. So for a
    movable-vs-preplaced pair whose cheap axis has no room to escape into, we separate
    along the OTHER axis instead. Fixes n=95 (infeasible -> optimal).

    SEPARATED-PAIR TIE-BREAK (`sep_rule`, default `PREC_SEP_RULE`): both axes are legal when
    the pair already clears on both, so the rule only picks which chain grows. See
    PREC_SEP_RULE.

    H3 (`net_pairs`): for a separated pair that is also a HEAVY netlist pair, override that
    tie-break with the choice that keeps the two blocks CLOSEST. A precedence edge on one axis
    forces the pair apart on that axis by their combined extent and leaves the other axis free
    to align, so the smallest centre distance the solve can still reach is (w_i+w_j)/2 under a
    horizontal edge and (h_i+h_j)/2 under a vertical one. Picking the smaller of the two is the
    whole rule: free, no extra solve, and it spends the same degree of freedom v1.5.11 spent on
    util."""
    N = len(cx)
    rule = PREC_SEP_RULE if sep_rule is None else sep_rule
    if rule not in PREC_SEP_RULES:
        raise ValueError(f'sep_rule must be one of {PREC_SEP_RULES}, got {rule!r}')
    wider = rule == 'larger'
    net_pairs = net_pairs or ()
    xlo, xhi = cx - w / 2, cx + w / 2
    ylo, yhi = cy - h / 2, cy + h / 2
    Gh, Gv = [], []
    for i in range(N):
        for j in range(i + 1, N):
            ox = min(xhi[i], xhi[j]) - max(xlo[i], xlo[j])
            oy = min(yhi[i], yhi[j]) - max(ylo[i], ylo[j])
            if ox <= 0 and oy <= 0:
                if (i, j) in net_pairs:
                    # H3: keep the heavy pair close. `net_invert` is the DIRECTION CONTROL --
                    # the same override pointed the wrong way, so that "the rule loses" can be
                    # told apart from "the tie-break is not a wirelength lever at all".
                    horiz = (w[i] + w[j]) <= (h[i] + h[j])
                    if net_invert:
                        horiz = not horiz
                else:
                    horiz = (-ox >= -oy) if wider else (-ox <= -oy)
            else:
                horiz = (oy >= ox)
            if (is_pp is not None and ox > 0 and oy > 0 and bool(is_pp[i]) != bool(is_pp[j])):
                m, p = (i, j) if not is_pp[i] else (j, i)
                axis = 0 if horiz else 1
                if _escape_blocked(m, p, axis, is_pp, xlo, xhi, ylo, yhi, cx, cy):
                    horiz = not horiz          # cheap axis is walled in -> go the other way
            if horiz:
                Gh.append((i, j) if cx[i] <= cx[j] else (j, i))
            else:
                Gv.append((i, j) if cy[i] <= cy[j] else (j, i))
    return Gh, Gv


def _rect_gap(cx, cy, w, h, i, j):
    gx = abs(cx[i] - cx[j]) - (w[i] + w[j]) / 2.0
    gy = abs(cy[i] - cy[j]) - (h[i] + h[j]) / 2.0
    return max(gx, 0.0), max(gy, 0.0)


def group_tree_edges(cx, cy, w, h, grouping):
    """Per grouping-group, a minimum spanning tree over its members (edge weight = the
    rectangles' gap distance in the CURRENT placement). Abutting every tree edge makes the
    group ONE connected component => V_grouping = 0 for it, and a tree is the CHEAPEST way
    to get there: |G|-1 abutments instead of all |G|*(|G|-1)/2 pairs, chosen between the
    members that are already closest, so the solve barely has to move anything.
    Returns [(i, j), ...] over all groups."""
    grouping = np.asarray(grouping, int)
    edges = []
    for g in np.unique(grouping[grouping > 0]):
        idx = np.where(grouping == g)[0]
        if len(idx) < 2:
            continue
        inside = {int(idx[0])}
        rest = set(int(v) for v in idx[1:])
        while rest:
            best = None
            for a in inside:
                for b in rest:
                    gx, gy = _rect_gap(cx, cy, w, h, a, b)
                    d = np.hypot(gx, gy)
                    if best is None or d < best[0]:
                        best = (d, a, b)
            _, a, b = best
            edges.append((a, b))
            inside.add(b); rest.discard(b)
    return edges


def group_noninterpose(Gh, Gv, cx, cy, w, h, grouping, is_pp=None):
    """B2: delete the FORCED gaps between grouping-group members. Returns (Gh, Gv, n_flips).

    A group pair (a, b) can only abut on the chain that carries it: a horizontal edge
    `x_a + w_a <= x_b` permits contact, a vertical one does not. But the precedence graph is
    COMPLETE over pairs, so a foreign block k with `a -> k -> b` on that same chain forces
    `x_b - (x_a + w_a) >= w_k`: the abutment is not merely obstructed, it is *infeasible in the
    program*, and no post-solve slide can repair it. Measured on the shipped artifact, a hard
    overlap at the destination is 64% of all owed merges -- the single largest blocking class.

    G3 answers this by forcing `gap == 0`, which is expensive and breaks solves. The weaker
    statement is enough: forbid the INTERPOSITION, not the gap. Moving one of the two edges
    `(a,k)` / `(k,b)` to the other chain leaves the solve completely free over the gap and
    merely stops it stacking k inside the corridor; `snap_groups` then closes what is left.
    Ordering costs no bbox, which docs/experiments/VIOLATION_REDUCTION_PLAN.md §3 argues is the only
    currency that has ever paid here.

    Safety: every pair keeps exactly one separation constraint, so the packing stays
    overlap-free; new edges are oriented by the cross-axis coordinate, so both chains remain
    subgraphs of a total order and stay acyclic. Two pairs are never flipped -- another group's
    tree edge (it would trade one abutment for another) and a preplaced/preplaced pair (their
    (x,y) are hard, so a forced re-ordering can only make the program infeasible).
    """
    grouping = np.asarray(grouping, int) if grouping is not None else None
    if grouping is None or not (grouping > 0).any():
        return Gh, Gv, 0
    n = len(cx)
    is_pp = np.zeros(n, bool) if is_pp is None else np.asarray(is_pp, bool)
    xlo, xhi = cx - w / 2, cx + w / 2
    ylo, yhi = cy - h / 2, cy + h / 2
    H = {(int(i), int(j)) for i, j in Gh}
    V = {(int(i), int(j)) for i, j in Gv}
    tree = [(int(a), int(b)) for a, b in group_tree_edges(cx, cy, w, h, grouping)]
    protected = {frozenset(e) for e in tree}
    flips = 0
    for a, b in tree:
        horiz = (a, b) in H or (b, a) in H
        chain, other = (H, V) if horiz else (V, H)
        u, v = (a, b) if (a, b) in chain else (b, a)
        for k in range(n):
            if k == u or k == v or (u, k) not in chain or (k, v) not in chain:
                continue
            best = None
            for i, j in ((u, k), (k, v)):
                if frozenset((i, j)) in protected or (is_pp[i] and is_pp[j]):
                    continue
                # cost of re-separating this pair on the OTHER axis = its penetration there
                pen = (min(yhi[i], yhi[j]) - max(ylo[i], ylo[j]) if horiz else
                       min(xhi[i], xhi[j]) - max(xlo[i], xlo[j]))
                if best is None or pen < best[0]:
                    best = (pen, i, j)
            if best is None:
                continue                      # both edges are untouchable: leave k where it is
            _, i, j = best
            chain.discard((i, j))
            cc = cy if horiz else cx          # orient on the axis the pair moves to
            other.add((i, j) if cc[i] <= cc[j] else (j, i))
            flips += 1
    return sorted(H), sorted(V), flips


def boundary_source_edges(Gh, Gv, cx, cy, w, h, boundary, is_pp=None, grouping=None):
    """E7: make a boundary-flagged block a SOURCE of the chain it is flagged on.

    The free version of P4. A block flagged LEFT can only be flush against the bbox's left wall
    if nothing is ordered to its left -- and with a complete precedence graph, anything ordered
    to its left holds it off the wall by that block's whole width, which `pull_boundary` can
    never undo (it slides one block at a time and the wall does not move). So for each flagged
    block, move every edge that puts a foreign block on its flagged side to the OTHER chain.

    Unlike P4 this adds no equality, introduces no X0/Y0, and costs no extra solve: it does not
    force flushness, it only stops the DAG forbidding it. Whether that is enough is the
    measurement -- the wall is set by whichever block ends up furthest out, and this only
    removes the *ordering* reason for a flagged block not to be that one.

    Same safety argument as `group_noninterpose`: one constraint per pair, new edges oriented by
    the cross-axis coordinate (acyclic), and preplaced/preplaced pairs plus grouping tree edges
    are never touched. Returns (Gh, Gv, n_flips).
    """
    boundary = np.zeros(len(cx), int) if boundary is None else np.asarray(boundary, int)
    if not (boundary != 0).any():
        return Gh, Gv, 0
    n = len(cx)
    is_pp = np.zeros(n, bool) if is_pp is None else np.asarray(is_pp, bool)
    H = {(int(i), int(j)) for i, j in Gh}
    V = {(int(i), int(j)) for i, j in Gv}
    protected = set()
    if grouping is not None and (np.asarray(grouping, int) > 0).any():
        protected = {frozenset((int(a), int(b)))
                     for a, b in group_tree_edges(cx, cy, w, h, grouping)}
    # (bit, chain, "the flagged block must come FIRST on this chain")
    rules = ((1, True, True), (2, True, False), (8, False, True), (4, False, False))
    flips = 0
    for i in range(n):
        code = int(boundary[i])
        if code == 0:
            continue
        for bit, horiz, first in rules:
            if not (code & bit):
                continue
            chain, other = (H, V) if horiz else (V, H)
            for j in range(n):
                if j == i:
                    continue
                bad = (j, i) if first else (i, j)      # j sits on i's flagged side
                if bad not in chain:
                    continue
                if frozenset(bad) in protected or (is_pp[i] and is_pp[j]):
                    continue
                chain.discard(bad)
                cc = cy if horiz else cx
                other.add((i, j) if cc[i] <= cc[j] else (j, i))
                flips += 1
    return sorted(H), sorted(V), flips


def group_sep_axis(Gh, Gv, cx, cy, w, h, grouping, is_pp=None):
    """B3: for a same-group pair, separate on the axis they can actually ABUT on.

    `_precedence` chooses each pair's axis knowing nothing about grouping. A pair carried on the
    horizontal chain can only share a VERTICAL border, which needs their y-spans to overlap; if
    the pair is separated in y by more than it is in x, that chain is the wrong one and the
    abutment has to travel further than it needs to. This re-routes group tree edges onto the
    axis whose CROSS-axis deficit is smaller.

    Note the shipped `PREC_SEP_RULE='larger'` (v1.5.11) already does this for pairs separated on
    BOTH axes -- it picks the axis with the larger separation, whose cross-axis gap is therefore
    the smaller one. So this can only reach pairs that OVERLAP on an axis, where the rule is
    min-penetration instead. Expect it to be small; it is measured, not assumed.
    Returns (Gh, Gv, n_flips).
    """
    grouping = np.asarray(grouping, int) if grouping is not None else None
    if grouping is None or not (grouping > 0).any():
        return Gh, Gv, 0
    n = len(cx)
    is_pp = np.zeros(n, bool) if is_pp is None else np.asarray(is_pp, bool)
    xlo, xhi = cx - w / 2, cx + w / 2
    ylo, yhi = cy - h / 2, cy + h / 2
    H = {(int(i), int(j)) for i, j in Gh}
    V = {(int(i), int(j)) for i, j in Gv}
    flips = 0
    for a, b in group_tree_edges(cx, cy, w, h, grouping):
        a, b = int(a), int(b)
        if is_pp[a] and is_pp[b]:
            continue
        ox = min(xhi[a], xhi[b]) - max(xlo[a], xlo[b])
        oy = min(yhi[a], yhi[b]) - max(ylo[a], ylo[b])
        want_h = oy >= ox              # y-spans overlap more => abut left-to-right
        cur = (a, b) if (a, b) in H else ((b, a) if (b, a) in H else None)
        if (cur is not None) == want_h:
            continue                   # already on the abutment axis
        chain, other = (H, V) if cur is not None else (V, H)
        pair = cur if cur is not None else ((a, b) if (a, b) in V else (b, a))
        chain.discard(pair)
        cc = cy if cur is not None else cx
        other.add((a, b) if cc[a] <= cc[b] else (b, a))
        flips += 1
    return sorted(H), sorted(V), flips


DAG_RULES = ('b3', 'b2', 'e7')      # applied in THIS order, whatever order they are named in


def apply_dag_rules(Gh, Gv, cx, cy, w, h, grouping, boundary, is_pp, rules):
    """Run the constraint-aware precedence edits named in `rules` (a subset of DAG_RULES).

    Order is fixed and not the caller's choice, because the three rules edit the same graph:
    B3 decides WHICH chain carries a group pair, so it has to run before B2 clears that chain;
    E7 runs last so its "nothing on the flagged side" property holds of the graph handed to the
    solve (it can re-introduce an interposition B2 removed, which is why the combination is
    screened rather than assumed).
    """
    rules = tuple(rules) if not isinstance(rules, str) else (rules,)
    bad = set(rules) - set(DAG_RULES)
    if bad:
        raise ValueError(f'dag rules must be a subset of {DAG_RULES}, got {sorted(bad)}')
    n_flips = 0
    for name in DAG_RULES:
        if name not in rules:
            continue
        fn = {'b3': group_sep_axis, 'b2': group_noninterpose}.get(name)
        if fn is not None:
            Gh, Gv, f = fn(Gh, Gv, cx, cy, w, h, grouping, is_pp=is_pp)
        else:
            Gh, Gv, f = boundary_source_edges(Gh, Gv, cx, cy, w, h, boundary,
                                              is_pp=is_pp, grouping=grouping)
        n_flips += f
    return Gh, Gv, n_flips


def _solve_once(cx, cy, w0, h0, area, free, fixed, is_pp, mib, Gh, Gv,
                grp_edges=(), lam_grp=0.0, ov_frac=0.25,
                net_edges=(), lam_net=0.0, soft_pin=0.0,
                grp_hard=(), grp_hard_ov=0.25, bnd_hard=None, part=None,
                hpwl_rows=None, bbox_cap=None, bnd_soft=None, lam_bnd=1.0,
                bnd_add=None, lam_bnd_add=0.0, true_bbox=False):
    # `part` (B8): an int partition id per block, minimising SUM of per-partition perimeters
    # against one shared outline. When the caller supplies no cross-partition precedence edge the
    # program SEPARATES -- the partitions share no variable and no constraint -- so the solution
    # is identical to solving each one on its own. It exists because cvxpy's cost is per PROBLEM,
    # not per row: a 7-block cluster sub-solve measured 17.8 ms against CLARABEL's own sub-
    # millisecond, i.e. essentially all canonicalisation. Batching a case's clusters into one
    # program pays that overhead once. Not compatible with `bnd_hard`, which needs one bbox.
    N = len(cx)
    llx0 = cx - w0 / 2.0; lly0 = cy - h0 / 2.0
    x = cp.Variable(N, nonneg=True); y = cp.Variable(N, nonneg=True)
    w = cp.Variable(N, nonneg=True); h = cp.Variable(N, nonneg=True)
    if part is None:
        W = cp.Variable(nonneg=True); H = cp.Variable(nonneg=True)
    else:
        if bnd_hard is not None:
            raise ValueError('part and bnd_hard are mutually exclusive (bnd_hard needs one bbox)')
        part = np.asarray(part, int)
        W = cp.Variable(int(part.max()) + 1, nonneg=True)
        H = cp.Variable(int(part.max()) + 1, nonneg=True)
    # EVERY block of constraints below is built as ONE vectorized row block. cvxpy spends ~80%
    # of this function in canonicalization, not in CLARABEL (measured: get_problem_data 25.6s of
    # a 28.4s solve call), and canonicalization cost tracks the number of EXPRESSION TREES, not
    # the number of scalar rows. The per-block Python loop this replaced built ~7N of them --
    # including N separate inv_pos SOC atoms -- for an identical cone program.
    cons = []
    mib_rep = {}
    for i in range(N):
        g = int(mib[i])
        if g > 0:
            mib_rep.setdefault(g, i)
    fx = np.flatnonzero(fixed)
    pp = np.flatnonzero(is_pp)
    fr = np.flatnonzero(~fixed)
    if len(fx):
        cons += [w[fx] == w0[fx], h[fx] == h0[fx]]
    if len(pp) and soft_pin <= 0:
        cons += [x[pp] == llx0[pp], y[pp] == lly0[pp]]
    if len(fr):
        cons += [cp.multiply(area[fr], cp.inv_pos(w[fr])) <= h[fr]]
        cons += [w[fr] <= AR_MAX * h[fr], h[fr] <= AR_MAX * w[fr]]
        cons += [w[fr] <= 3.0 * w0[fr], h[fr] <= 3.0 * h0[fr]]
    mem = np.array([i for i in range(N)
                    if int(mib[i]) > 0 and mib_rep[int(mib[i])] != i], dtype=int)
    if len(mem):
        rep = np.array([mib_rep[int(mib[i])] for i in mem], dtype=int)
        cons += [w[mem] == w[rep], h[mem] == h[rep]]
    if Gh:                                 # one vectorized row block, not len(Gh) scalar rows:
        ih, jh = np.asarray(Gh, int).T     # cvxpy canonicalization is the cost, not the count
        cons += [x[ih] + w[ih] <= x[jh]]
    if Gv:
        iv, jv = np.asarray(Gv, int).T
        cons += [y[iv] + h[iv] <= y[jv]]
    cons += ([x + w <= W, y + h <= H] if part is None else
             [x + w <= W[part], y + h <= H[part]])

    # P4 -- BOUNDARY FLAGS AS HARD CONSTRAINTS (v1.5.13, `bnd_hard`).
    #
    # The scorer's boundary predicate is flushness against the solution's OWN bounding box, so
    # it is linear in exactly these variables. `pull_boundary` can only slide a flagged block
    # OUT to a wall that already exists; it cannot move the wall, which is why 43% of the
    # residual (the preplaced flags) is unreachable to it, and why the post-solve retraction
    # experiment came out net -1510 flags -- a hill-climb can only TRANSLATE the blocks holding
    # the wall out, which un-flushes them. A re-solve can RE-PACK them.
    #
    # This needs the true lower corner as a variable. Without `bnd_hard` the shipped program
    # leaves it implicit (x >= 0 and minimising W = max(x+w) drives min(x) to 0), so X0/Y0 are
    # introduced ONLY here and the default path stays bit-for-bit identical.
    #
    # ARM D -- THE TRUE BBOX (`true_bbox`). The shipped objective is `W + H` with `x, y >= 0`,
    # i.e. the outline measured FROM THE ORIGIN, and it relies on "minimising W = max(x+w) drives
    # min(x) to 0" to make that the same thing as the outline itself. That identity fails exactly
    # when the packing cannot slide -- preplaced (x,y) are pinned by an equality in the SOLVE
    # frame, so `min(x)` is bounded below by whatever the anchors and their precedence chains
    # allow, and every unit of that offset is charged as if it were outline. Introducing the true
    # lower corner and minimising `W + H - X0 - Y0` charges the extent the CONTEST measures.
    # `bnd_hard` (P4) has always done this as a side effect of needing X0/Y0, which is one
    # reason it worked; here it is separable and measurable on its own.
    need_lo = bool(true_bbox) or (bnd_hard is not None and len(bnd_hard)) or (
        lam_bnd_add > 0 and bnd_add is not None and (np.asarray(bnd_add, int) != 0).any())
    X0 = Y0 = None
    if need_lo:
        if part is not None:
            raise ValueError('part needs one bbox per partition; X0/Y0 is a single outline')
        X0 = cp.Variable(nonneg=True); Y0 = cp.Variable(nonneg=True)
        cons += [x >= X0, y >= Y0]
    # Subtracted from W+H below. `bnd_hard` keeps its historical behaviour (it always charged the
    # true bbox), so P4 stays bit-for-bit; arm B alone does NOT, so the boundary term can be
    # attributed separately from the objective change it would otherwise smuggle in.
    obj_lo = (X0 + Y0) if (X0 is not None and
                           (true_bbox or (bnd_hard is not None and len(bnd_hard)))) else None
    if bnd_hard is not None and len(bnd_hard):
        code = np.asarray(bnd_hard, int)
        for bit, expr in ((1, lambda i: x[i] == X0), (2, lambda i: x[i] + w[i] == W),
                          (8, lambda i: y[i] == Y0), (4, lambda i: y[i] + h[i] == H)):
            sel = np.flatnonzero(code & bit)
            if len(sel):
                cons += [expr(sel)]

    # ARM B -- THE BOUNDARY DISTANCE ADDED TO THE PRIMARY OBJECTIVE (`bnd_add`/`lam_bnd_add`).
    #
    # H8 (`--resolve bnd@0.0`) proved the mechanism: an LP optimum sits at a VERTEX, so pricing
    # the distance of every flagged edge to its wall lands blocks flush, and it reached what no
    # post-solve move can -- the 2026-08-01 census finds 107 of the 119 residual flags are
    # PREPLACED anchors whose satisfaction needs a median of SIX movable blocks to leave the
    # overhang simultaneously. H8 pays for it with a second solve per candidate, which is a whole
    # extra wave (Phase 2: candidates cost ceil(K/W) waves) and +3.07% on the real score.
    #
    # The same term is convex in the SAME variables as the ordinary program, so it can simply be
    # added to `W + H` and cost no solve at all. This is not v1.5.6's rejected `lam_bnd`, which
    # is gone from the tree: that was measured under the pre-v1.5.11 tie-break against a
    # one-at-a-time pull, and its verdict ("adds nothing on top of the pull") was about the
    # 300-flag MOVABLE residual the pull already owned.
    #
    # SCALE. `frozen_resolve` normalises this term by `mean(w0+h0) * n_flags`, which is free
    # there because the boundary term is the WHOLE objective and the argmin is invariant to it
    # (Phase 3 H5-b: rescaling still moved the result, because the LP is degenerate). Here the
    # scale sets a real exchange rate against the perimeter, so the term is normalised to
    # LENGTH -- mean distance per flagged edge -- and `lam_bnd_add` reads as "one unit of mean
    # flag distance is worth this many units of perimeter".
    bnd_add_term = None
    if lam_bnd_add > 0 and bnd_add is not None and X0 is not None:
        code_a = np.asarray(bnd_add, int)
        n_flag = int((code_a != 0).sum())
        if n_flag:
            dist_a = []
            for bit, ex in ((1, lambda s: x[s] - X0), (2, lambda s: W - (x[s] + w[s])),
                            (8, lambda s: y[s] - Y0), (4, lambda s: H - (y[s] + h[s]))):
                sel = np.flatnonzero(code_a & bit)
                if len(sel):
                    dist_a.append(cp.sum(ex(sel)))
            if dist_a:
                bnd_add_term = (float(lam_bnd_add) / n_flag) * cp.sum(cp.hstack(dist_a))

    # G3 -- GROUPING ABUTMENT AS A HARD CONSTRAINT (v1.5.12, `grp_hard`).
    #
    # `lam_grp` above pays for abutment in the OBJECTIVE, which is why it only ever gets close:
    # W+H is worth more than a group contact wherever the two disagree, and "close" scores as
    # broken under the scorer's zero-tolerance union. The post-solve `snap_groups` then repairs
    # what it can -- but the autopsy (docs/experiments/VIOLATION_TRACKS.md, G0a) found 88% of its declines
    # are a foreign block already sitting in the destination, and no post-solve translation can
    # evict a block the solve chose to put there. This is the same abutment expressed while the
    # bbox is still being minimised, so the packing forms AROUND the contact instead of having
    # to be undone afterwards.
    #
    # Per spanning-tree edge, in the axis the precedence graph separates the pair on:
    #   equality        the separating gap is exactly 0        (1 row)
    #   cross overlap   a POSITIVE-LENGTH shared border        (2 non-trivial rows)
    # min(a,b) - max(c,d) >= eps expands to four linear rows, two of which (h[i] >= eps and
    # h[j] >= eps) are implied by the area/AR constraints, so only two are emitted.
    #
    # This CAN make the program infeasible -- it is a hard constraint grafted onto a
    # sampler-derived DAG, which is exactly what sank `bnd_aware` (86/100). The caller must
    # treat a None return as "drop the tightening and keep the ordinary solve"; see _grp_hard_pass.
    if len(grp_hard):
        hh_i, hh_j, hv_i, hv_j = [], [], [], []
        axis = {}
        for (i, j) in Gh:
            axis[(min(i, j), max(i, j))] = (True, i, j)
        for (i, j) in Gv:
            axis[(min(i, j), max(i, j))] = (False, i, j)
        for (a, b) in grp_hard:
            ent = axis.get((min(a, b), max(a, b)))
            if ent is None:
                continue
            is_h, i, j = ent
            (hh_i if is_h else hv_i).append(i)
            (hh_j if is_h else hv_j).append(j)
        if hh_i:
            i, j = np.array(hh_i, int), np.array(hh_j, int)
            eps = grp_hard_ov * np.minimum(h0[i], h0[j])
            cons += [x[j] == x[i] + w[i],
                     y[i] + h[i] - y[j] >= eps, y[j] + h[j] - y[i] >= eps]
        if hv_i:
            i, j = np.array(hv_i, int), np.array(hv_j, int)
            eps = grp_hard_ov * np.minimum(w0[i], w0[j])
            cons += [y[j] == y[i] + h[i],
                     x[i] + w[i] - x[j] >= eps, x[j] + w[j] - x[i] >= eps]

    # GROUPING ABUTMENT (v1.5.5). The solve minimizes W+H only, so it has no reason to keep a
    # group's members in contact: off the critical path a block can sit anywhere in its slack,
    # and the group drifts into several components => V_grouping. Pull each spanning-tree edge
    # of each group to a SHARED BORDER, in the objective only (never a constraint) so this
    # cannot make an already draw-fragile solve infeasible.
    #
    # The pair is separated by exactly one of Gh/Gv (the precedence graph orders every pair),
    # so "abut" = drive that axis's gap to 0 AND keep a positive-length overlap on the other
    # axis -- a corner contact does NOT connect two blocks under the scorer's shapely union.
    # gap is linear; cp.pos(eps - overlap) is convex (overlap is a concave min-minus-max), so
    # the problem stays a DCP conic program.
    # OBJECTIVE. `W + H` (perimeter), TOFU's Eq. 2. The contest charges `Area_gap` on `W * H`,
    # which is not convex; an iteratively reweighted linearization of it was screened (E4) and
    # came out NULL (+0.30% / -0.40%), because the packed outline's aspect is already 0.903
    # against GT's 0.888 -- there is nothing for an area-shaped objective to correct.
    # See docs/experiments/AREA_SELECTION_SCREEN.md.
    obj = ((W + H if obj_lo is None else (W + H - obj_lo)) if part is None
           else cp.sum(W) + cp.sum(H))
    if bnd_add_term is not None:
        obj = obj + bnd_add_term

    # H5/H6 -- WIRELENGTH AS THE OBJECTIVE, on a FROZEN topology.
    #
    # Every wirelength arm before this one acted on the graph (H3) or on a post-solve move's
    # price (J1/H10) and moved `hpwl_gap` by ~0, for one reason: the program minimises `W + H`,
    # so an ordering edge only BOUNDS a pair's separation and the freed room goes to whatever
    # shortens the outline. Put the scorer's own term in the objective instead. With Gh/Gv held
    # fixed the feasible region is a polyhedron and weighted Manhattan distance between centres
    # is convex, so this returns the EXACTLY HPWL-optimal placement inside the shipped topology
    # -- not an approximation and not a hill-climb.
    #
    # `bbox_cap` is H6's epsilon constraint. Without it the wirelength objective has no reason
    # to keep the outline small and pays `area_gap` for `hpwl_gap` at the ~100x unfavourable
    # rate AGENTS.md records. Capping W+H bounds the emitted perimeter conservatively (the true
    # bbox is max(x+w)-min(x) <= W), though at a FIXED perimeter a squarer outline has more
    # area -- so the caller must still price the result, which the candidate acceptance test in
    # `legalize_sample` does exactly.
    #
    # b2b ONLY, deliberately. b2b is translation-invariant, so it is well defined in the solve
    # frame, which floats: the preplaced alignment happens later in `finish()`. p2b is measured
    # against ABSOLUTE pin coordinates and would be meaningless here -- and H14 measured p2b at
    # GT parity (-0.0000 of `hpwl_gap`), so there is nothing to gain there anyway. Any p2b
    # damage is caught downstream, because the acceptance test prices the FULL v10 HPWL.
    if hpwl_rows is not None or bnd_soft is not None:
        if part is not None:
            raise ValueError('a frozen-topology objective and part are mutually exclusive')
        if bnd_add_term is not None:
            # this branch REPLACES the objective; silently dropping arm B's term would make a
            # combined arm measure the wrong thing
            raise ValueError('bnd_add and a frozen-topology objective are mutually exclusive')
        terms = []
        if hpwl_rows is not None:
            hi, hj, hwt = hpwl_rows
            if not len(hi):
                return None
            cxv, cyv = x + w / 2.0, y + h / 2.0
            terms.append(cp.sum(cp.multiply(
                hwt / max(float(hwt.sum()), 1e-9),
                cp.abs(cxv[hi] - cxv[hj]) + cp.abs(cyv[hi] - cyv[hj]))))
        perim = W + H
        if bnd_soft is not None:
            # The boundary predicate is flushness against the solution's OWN bounding box, so
            # the DISTANCE to a required wall is linear in exactly these variables. `bnd_hard`
            # (P4) states the same thing as an equality and can make the program infeasible on a
            # sampler-derived DAG; priced instead, it never can. It needs the true lower corner
            # as a variable -- the shipped program leaves it implicit, because minimising W+H
            # already drives min(x) to 0, but here the perimeter is a CAP and not the objective.
            code = np.asarray(bnd_soft, int)
            X0 = cp.Variable(nonneg=True); Y0 = cp.Variable(nonneg=True)
            cons += [x >= X0, y >= Y0]
            perim = W + H - X0 - Y0
            dist = []
            for bit, ex in ((1, lambda s: x[s] - X0), (2, lambda s: W - (x[s] + w[s])),
                            (8, lambda s: y[s] - Y0), (4, lambda s: H - (y[s] + h[s]))):
                sel = np.flatnonzero(code & bit)
                if len(sel):
                    dist.append(cp.sum(ex(sel)))
            if not dist:
                return None
            scale = max(float(np.mean(w0) + np.mean(h0)) * max(len(np.flatnonzero(code)), 1), 1e-9)
            terms.append(lam_bnd * cp.sum(cp.hstack(dist)) / scale)
        obj = terms[0] if len(terms) == 1 else cp.sum(cp.hstack(terms))
        if bbox_cap is not None:
            cons += [perim <= float(bbox_cap)]
    # SOFT PIN (reseed only). The preplaced equality above is what makes an inconsistent
    # precedence graph infeasible, so the reseed pass prices it instead of enforcing it: the
    # solve then always succeeds and its layout tells us which side of each anchor a movable
    # block actually wants to sit on. Never used for a layout we emit -- the caller re-solves
    # with the pins HARD on the reseeded ordering, because preplaced (x,y) is a hard constraint.
    if len(pp) and soft_pin > 0:
        obj = obj + soft_pin * cp.sum(cp.abs(x[pp] - llx0[pp]) + cp.abs(y[pp] - lly0[pp]))
    if lam_grp > 0 and len(grp_edges):
        horiz = {}
        for (i, j) in Gh:
            horiz[(min(i, j), max(i, j))] = (True, i, j)
        for (i, j) in Gv:
            horiz[(min(i, j), max(i, j))] = (False, i, j)
        # Bucketed by separating axis and built as two vectorized blocks, for the same reason
        # the per-block constraints are: one hstack of 2*|grp_edges| scalar atoms costs far
        # more to canonicalize than two elementwise ones. cp.minimum/maximum/pos are all
        # elementwise, so the penalty value is unchanged.
        eh_i, eh_j, ev_i, ev_j = [], [], [], []
        for (a, b) in grp_edges:
            ent = horiz.get((min(a, b), max(a, b)))
            if ent is None:
                continue
            is_h, i, j = ent
            (eh_i if is_h else ev_i).append(i)
            (eh_j if is_h else ev_j).append(j)
        pen = []
        if eh_i:
            i, j = np.array(eh_i, int), np.array(eh_j, int)
            ov = cp.minimum(y[i] + h[i], y[j] + h[j]) - cp.maximum(y[i], y[j])
            pen += [cp.sum(x[j] - (x[i] + w[i])),
                    cp.sum(cp.pos(ov_frac * np.minimum(h0[i], h0[j]) - ov))]
        if ev_i:
            i, j = np.array(ev_i, int), np.array(ev_j, int)
            ov = cp.minimum(x[i] + w[i], x[j] + w[j]) - cp.maximum(x[i], x[j])
            pen += [cp.sum(y[j] - (y[i] + h[i])),
                    cp.sum(cp.pos(ov_frac * np.minimum(w0[i], w0[j]) - ov))]
        if pen:
            obj = obj + lam_grp * cp.sum(cp.hstack(pen))

    # NETLIST ATTRACTION (experimental, default off). Mean UNWEIGHTED block-to-block Manhattan
    # center distance. This is deliberately NOT the scorer's weighted b2b+p2b HPWL: the exact
    # weighted term was screened on test/scorer-aligned-netlist and was HARMFUL, because this
    # term earns its keep as a topology regularizer (it keeps connected blocks in the same
    # neighbourhood, which stabilizes the precedence graph the next iteration infers) rather
    # than as a wirelength approximation. Normalized by edge count so one lambda transfers
    # across instance sizes. Convex (abs of affine) => the problem stays DCP.
    if lam_net > 0 and len(net_edges):
        edges = np.asarray(net_edges, dtype=int)
        edges = edges[edges[:, 0] != edges[:, 1], :2]
        if len(edges):
            center_x = x + w / 2
            center_y = y + h / 2
            src, dst = edges[:, 0], edges[:, 1]
            net_pen = (cp.abs(center_x[src] - center_x[dst]) +
                       cp.abs(center_y[src] - center_y[dst]))
            obj = obj + lam_net * cp.sum(net_pen) / len(edges)

    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        return None
    if x.value is None or prob.status not in ('optimal', 'optimal_inaccurate'):
        return None
    nx, ny = np.asarray(x.value), np.asarray(y.value)
    nw, nh = np.clip(np.asarray(w.value), 1e-6, None), np.clip(np.asarray(h.value), 1e-6, None)
    cxc, cyc = nx + nw / 2, ny + nh / 2
    for i in range(N):                     # rescale free blocks to EXACT area (centered)
        if not fixed[i]:
            s = np.sqrt(area[i] / (nw[i] * nh[i])); nw[i] *= s; nh[i] *= s
    return cxc, cyc, nw, nh, float(prob.value)


def _iterate(cx, cy, w, h, area, free, fixed, is_pp, mib, iters,
             grouping=None, boundary=None, lam_grp=0.0, ov_frac=0.25,
             net_edges=(), lam_net=0.0, sep_rule=None, grp_dag=False, net_pairs=None,
             net_invert=False, lam_bnd_add=0.0, true_bbox=False):
    best = np.stack([cx, cy, w, h], 1); best_obj = _bbox(cx, cy, w, h); solved = False
    best_solver = np.inf
    accepted = False          # has any SOLVED layout been taken yet?
    # Arm B makes bbox no longer the objective for exactly the same reason lam_grp does, so it
    # has to score by the solver's own penalized objective too -- otherwise the bbox test below
    # rejects every round in which the solve correctly bought flushness with perimeter.
    bnd_add = (np.asarray(boundary, int) if (lam_bnd_add > 0 and boundary is not None
                                             and (np.asarray(boundary, int) != 0).any()) else None)
    penalized = lam_grp > 0 or lam_net > 0 or bnd_add is not None or bool(true_bbox)
    for _ in range(iters):
        Gh, Gv = _precedence(cx, cy, w, h, is_pp=is_pp, sep_rule=sep_rule,
                             net_pairs=net_pairs, net_invert=net_invert)
        grp_edges = (group_tree_edges(cx, cy, w, h, grouping)
                     if (lam_grp > 0 and grouping is not None) else ())
        # B2/B6: bias THIS round's re-inferred graph, so the non-interposition rule costs no
        # extra solve at all -- `_iterate` re-infers the precedence graph every round anyway.
        Gh2, Gv2 = Gh, Gv
        if grp_dag:
            Gh2, Gv2, _nf = apply_dag_rules(Gh, Gv, cx, cy, w, h, grouping, boundary, is_pp,
                                            ('b2',) if grp_dag is True else grp_dag)
        out = _solve_once(cx, cy, w, h, area, free, fixed, is_pp, mib, Gh2, Gv2,
                          grp_edges=grp_edges, lam_grp=lam_grp, ov_frac=ov_frac,
                          net_edges=net_edges, lam_net=lam_net,
                          bnd_add=bnd_add, lam_bnd_add=lam_bnd_add, true_bbox=true_bbox)
        if out is None and (Gh2 is not Gh or Gv2 is not Gv):
            # the re-ordering made this round infeasible: keep the ordinary graph. One wasted
            # solve on the failing round, never a lost case.
            out = _solve_once(cx, cy, w, h, area, free, fixed, is_pp, mib, Gh, Gv,
                              grp_edges=grp_edges, lam_grp=lam_grp, ov_frac=ov_frac,
                              net_edges=net_edges, lam_net=lam_net,
                              bnd_add=bnd_add, lam_bnd_add=lam_bnd_add, true_bbox=true_bbox)
        if out is None:
            break
        solved = True
        cx, cy, w, h, solver_obj = out
        if penalized:
            # With a grouping penalty the solve is deliberately trading bbox for a soft
            # term, so bbox is NO LONGER the objective and the bbox test below would reject a good
            # solve (and, on the first iteration, hand back the raw OVERLAPPING input as "best").
            # Score by the solver's own penalized objective instead. Both lambdas 0 keeps the
            # original bbox-only path bit-for-bit.
            if solver_obj < best_solver - 1e-9:
                best_solver = solver_obj
                best_obj = _bbox(cx, cy, w, h)
                best = np.stack([cx, cy, w, h], 1)
            else:
                break
            continue
        obj = _bbox(cx, cy, w, h)
        # `best_obj` starts as the bbox of the raw OVERLAPPING input, which is SMALLER than any
        # legal packing of it, so a plain `obj < best_obj` test rejects the first solve and
        # returns the illegal input while reporting solved=True. The penalized branch above was
        # fixed for this in v1.5.6; the bbox-only branch was left alone and had the same defect.
        # Take the first solved layout unconditionally, then require improvement as before.
        if not accepted or obj < best_obj - 1e-9:
            accepted = True
            best_obj = obj; best = np.stack([cx, cy, w, h], 1)
        else:
            break
    return best, best_obj, solved


def bnd_hard_conflicts(pos_ll, boundary, is_pp):
    """E1d: preplaced flags that make MUTUALLY CONTRADICTORY demands on one wall.

    Two pinned blocks both flagged LEFT at different x are the only boundary demands that are
    unsatisfiable as an equality set, and they are detectable in O(N) BEFORE any solve -- there
    are 8 of them across the whole official set. The drop-and-retry tier ladder discovers the
    same fact by failing a conic solve and trying a smaller tier, which is where most of P4's
    +47% goes. Returns the set of block indices to exclude."""
    p = np.asarray(pos_ll, float)
    x, y, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    bnd = np.asarray(boundary, int)
    is_pp = np.asarray(is_pp, bool)
    demand = defaultdict(list)
    for i in np.flatnonzero(is_pp & (bnd != 0)):
        for bit, key, val in ((1, 'L', x[i]), (2, 'R', x[i] + w[i]),
                              (8, 'B', y[i]), (4, 'T', y[i] + h[i])):
            if int(bnd[i]) & bit:
                demand[key].append((round(float(val), 6), int(i)))
    drop = set()
    for key, vals in demand.items():
        if len({v for v, _ in vals}) < 2:
            continue
        # keep the majority coordinate (the wall the most flags agree on), drop the rest
        counts = Counter(v for v, _ in vals)
        keep_v = counts.most_common(1)[0][0]
        drop.update(i for v, i in vals if v != keep_v)
    return drop


def _grp_hard_pass(best, cx, cy, w, h, area, free, fixed, is_pp, mib, grouping,
                   ov, tiers, boundary=None, precheck=False, **kw):
    """One extra M9 solve with soft-constraint predicates promoted to HARD constraints.

    Runs ONCE, on the layout `_iterate` converged to, and only ever ADDS a candidate -- the
    caller keeps the ordinary solve unless this one is both solved and cheaper. That shape is
    deliberate: `bnd_aware` proved that a hard global constraint grafted onto a sampler-derived
    DAG craters feasibility, so this can never be the only solve, and a failure costs one wasted
    solve rather than a case.

    Two families of predicate, both linear in the solve's own variables:

      G3, grouping -- each group's spanning-tree edge pinned to a shared border.
        'all'    every group;
        'free'   only groups with no preplaced member -- a pinned member cannot move to meet
                 the contact, so those are where the equality most often contradicts the pins;
        'small'  additionally only groups of 2, whose tree is a single edge.

      P4, boundary -- each flagged block pinned flush to the wall it is flagged to.
        'bnd-all'  every flagged block, INCLUDING preplaced. That is the point: a preplaced
                   flag makes the wall follow the anchor, and the solve then re-packs whatever
                   was overhanging it. `pull_boundary` cannot do this -- it can only translate
                   the overhanging blocks, which un-flushes them (measured net -1510 flags).
        'bnd-free' flagged non-preplaced blocks only;
        'bnd-mov'  flagged free blocks only.

      Combined: 'both-all' / 'both-free'.

    Feasibility gating is DROP-AND-RETRY over `tiers` in the order given, cheapest conflict
    first. Returns (coords, obj) or None. At most len(tiers) solves.
    """
    Gh, Gv = _precedence(best[:, 0], best[:, 1], best[:, 2], best[:, 3],
                         is_pp=is_pp, sep_rule=kw.get('sep_rule'),
                         net_pairs=kw.get('net_pairs'),
                         net_invert=kw.get('net_invert', False))
    grouping = np.asarray(grouping, int)
    n = len(best)
    bnd = np.zeros(n, int) if boundary is None else np.asarray(boundary, int)
    per_group = {}
    for (a, b) in group_tree_edges(best[:, 0], best[:, 1], best[:, 2], best[:, 3], grouping):
        per_group.setdefault(int(grouping[a]), []).append((a, b))
    sizes = {g: int((grouping == g).sum()) for g in per_group}
    pinned = {g: bool(is_pp[grouping == g].any()) for g in per_group}
    movable = ~(np.asarray(is_pp, bool) | np.asarray(fixed, bool))

    def grp_for(tier):
        if tier not in ('all', 'free', 'small', 'both-all', 'both-free'):
            return []
        want_free = tier in ('free', 'small', 'both-free')
        return [e for g, es in per_group.items() for e in es
                if (not want_free or not pinned[g]) and (tier != 'small' or sizes[g] == 2)]

    def bnd_for(tier):
        if tier in ('bnd-all', 'both-all'):
            keep = np.ones(n, bool)
        elif tier in ('bnd-free', 'both-free'):
            keep = ~np.asarray(is_pp, bool)
        elif tier == 'bnd-mov':
            keep = movable
        elif tier == 'bnd-pp':
            # E1c: ONLY the preplaced flags -- 60% of the residual and the only share
            # `pull_boundary` structurally cannot reach (it moves the block, not the wall).
            # The smallest constraint set that still buys the unreachable class.
            keep = np.asarray(is_pp, bool)
        else:
            return None
        c = np.where(keep, bnd, 0)
        # a block flagged to BOTH opposite walls cannot be flush on both -- drop that axis
        c = np.where((c & 3) == 3, c & ~3, c)
        c = np.where((c & 12) == 12, c & ~12, c)
        return c if (c != 0).any() else None

    # E1d: drop the provably contradictory preplaced flags BEFORE the first solve rather than
    # discovering them by failing one. `best` is the converged layout, i.e. the frame the walls
    # will actually be measured against.
    drop = bnd_hard_conflicts(np.stack([best[:, 0] - best[:, 2] / 2,
                                        best[:, 1] - best[:, 3] / 2,
                                        best[:, 2], best[:, 3]], 1),
                              bnd, is_pp) if precheck else set()
    if drop:
        bnd = bnd.copy()
        bnd[list(drop)] = 0

    for tier in tiers:
        edges, codes = grp_for(tier), bnd_for(tier)
        if not edges and codes is None:
            continue
        out = _solve_once(cx, cy, w, h, area, free, fixed, is_pp, mib, Gh, Gv,
                          grp_edges=(), lam_grp=0.0, ov_frac=kw.get('ov_frac', 0.25),
                          net_edges=kw.get('net_edges', ()), lam_net=kw.get('lam_net', 0.0),
                          grp_hard=edges, grp_hard_ov=ov, bnd_hard=codes,
                          # the G3/P4 tiers are compared against the ordinary solve on the
                          # finished layouts, so they have to be solved under the SAME objective
                          # -- a tier minimising the origin-based outline against a main solve
                          # minimising the true one is not a comparison of the constraint.
                          true_bbox=kw.get('true_bbox', False))
        if out is not None:
            ncx, ncy, nw, nh, obj = out
            return np.stack([ncx, ncy, nw, nh], 1), _bbox(ncx, ncy, nw, nh)
    return None


def _soft_pin_seed(cur, cx, cy, w, h, area, free, fixed, is_pp, mib, lam, **kw):
    """One reseed pass: solve with the preplaced (x,y) equality PRICED instead of enforced,
    then hand back that layout with preplaced snapped exactly onto their pins.

    Returns [N,4] = [cx,cy,w,h] to use as the next `_iterate` start, or None if even the
    penalised solve fails. The returned layout is never emitted -- its only job is to induce a
    precedence graph that is consistent with where the anchors actually are, which is what the
    raw draw's graph is not. Preplaced are restored exactly (and fixed-shape sizes with them)
    so `_precedence` sees the true anchor geometry rather than the solver's drifted copy.
    """
    Gh, Gv = _precedence(cur[:, 0], cur[:, 1], cur[:, 2], cur[:, 3], is_pp=is_pp,
                         sep_rule=kw.get('sep_rule'), net_pairs=kw.get('net_pairs'),
                         net_invert=kw.get('net_invert', False))
    grp_edges = (group_tree_edges(cur[:, 0], cur[:, 1], cur[:, 2], cur[:, 3], kw.get('grouping'))
                 if (kw.get('lam_grp', 0.0) > 0 and kw.get('grouping') is not None) else ())
    out = _solve_once(cx, cy, w, h, area, free, fixed, is_pp, mib, Gh, Gv,
                      grp_edges=grp_edges, lam_grp=kw.get('lam_grp', 0.0),
                      ov_frac=kw.get('ov_frac', 0.25), net_edges=kw.get('net_edges', ()),
                      lam_net=kw.get('lam_net', 0.0), soft_pin=float(lam))
    if out is None:
        return None
    ncx, ncy, nw, nh, _ = out
    ncx = np.asarray(ncx).copy(); ncy = np.asarray(ncy).copy()
    nw = np.asarray(nw).copy(); nh = np.asarray(nh).copy()
    ncx[is_pp] = cx[is_pp]; ncy[is_pp] = cy[is_pp]
    nw[fixed] = w[fixed]; nh[fixed] = h[fixed]
    return np.stack([ncx, ncy, nw, nh], 1)


def align_preplaced(pos_ll, target_ll, is_pp):
    """Rigidly translate a lower-left [N,4] solution so preplaced blocks land on their
    given absolute lower-left target positions. The scorer checks preplaced (x,y) against
    the input EXACTLY, so a shift-to-nonneg frame makes every preplaced-bearing sample
    dimension-INFEASIBLE; this restores the frame. Translation-invariant for all scored
    quantities. No-op when there are no preplaced blocks."""
    pos_ll = np.asarray(pos_ll, float).copy()
    m = np.asarray(is_pp, bool)
    if not m.any():
        return pos_ll
    tgt = np.asarray(target_ll, float)
    off = (tgt[m, :2] - pos_ll[m, :2]).mean(0)
    pos_ll[:, 0] += off[0]; pos_ll[:, 1] += off[1]
    return pos_ll


def _mib_uniform(mask, mib):
    """Make a gap mask uniform per MIB group: a group is gapped only if EVERY member is.

    The conic solve gives an MIB group ONE (w,h) variable pair (see _solve_once), so V_mib = 0
    at solve exit -- measured 0 on every case whose solve succeeded. The gap step is what
    breaks it, because its exclusions are per BLOCK: open_gaps skips immutable blocks, and
    aware_gaps additionally skips grouped/boundary-flagged ones, so a group straddling that
    line emits two shapes 0.1% apart and the scorer reads a violation. Deciding per GROUP
    instead means the shrink can never split one. Costs the gap on those blocks (they stay at
    the exactly-touching solve geometry); measured 0/100 feasibility lost, V_mib 79 -> 14.
    """
    if mib is None:
        return mask
    mib = np.asarray(mib, int)
    m = np.asarray(mask, bool).copy()
    for g in np.unique(mib[mib > 0]):
        idx = np.where(mib == g)[0]
        if not m[idx].all():
            m[idx] = False
    return m


def open_gaps(coords, free, gap_frac=GAP_FRAC, mib=None):
    """Shrink each FREE block by (1-gap_frac) about its center so every packed contact
    gains a supra-threshold gap. WITHOUT this, the joint solve packs blocks to TOUCH
    (x_i+w_i == x_j); the CLARABEL residual + coordinate rounding then push touching pairs
    a hair past the scorer's exact overlap threshold (Area(b_i∩b_j)>0 on BOTH axes > 1e-6),
    so nearly every touching contact reads as an overlap VIOLATION -> infeasible (Cost=M).
    A gap_frac~1e-3 costs ~2*gap_frac area (inside the 1% soft-block tolerance) and leaves
    the bbox unchanged (util is flat), but takes official-100 feasibility from 13/100 to
    99/100 under the exact scorer. Fixed-shape / preplaced blocks are left EXACT (their
    (w,h)/(x,y) are hard-constrained and cannot be shrunk). coords [N,4]=[cx,cy,w,h].

    Pass `mib` to keep the shrink uniform inside each MIB group (see _mib_uniform); without
    it a group holding an immutable member emits two shapes and scores a V_mib violation."""
    if gap_frac <= 0:
        return coords
    out = np.asarray(coords, float).copy()
    f = _mib_uniform(np.asarray(free, bool), mib)
    out[f, 2] *= (1.0 - gap_frac)   # width
    out[f, 3] *= (1.0 - gap_frac)   # height (centers unchanged -> gap opens symmetrically)
    return out


GAP_REPAIR_ROUNDS = 3      # each round shrinks an offender by another (1-gap_frac) per side


def _overlap_pairs(r):
    """Index pairs violating the scorer's hard overlap rule (>1e-6 on BOTH axes)."""
    x, y, w, h = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    ox = np.minimum(x[:, None] + w[:, None], x[None, :] + w[None, :]) - \
        np.maximum(x[:, None], x[None, :])
    oy = np.minimum(y[:, None] + h[:, None], y[None, :] + h[None, :]) - \
        np.maximum(y[:, None], y[None, :])
    bad = np.triu((ox > 1e-6) & (oy > 1e-6), 1)
    return list(zip(*np.where(bad)))


def targeted_gaps(pos_ll, free, gap_frac=GAP_FRAC, mib=None, prec=None,
                  rounds=GAP_REPAIR_ROUNDS):
    """Open a gap on ONLY the blocks that actually overlap. Returns (pos, n_shrunk).

    `aware_gaps` keeps every grouped / boundary-flagged contact at exact touching, which is
    what makes its layout score well -- and occasionally one of those contacts rounds a hair
    past the scorer's 1e-6 overlap threshold. `legalize_sample` answered that by discarding the
    whole layout and re-gapping EVERY free block (`open_gaps`), which breaks every abutment and
    every flush edge in the case.

    Measured over 192 solved cases: the aware layout is infeasible on 29% of them, and when it
    is, the offence is a MEDIAN OF ONE overlapping pair out of ~2775 (max 3), always overlap and
    never the area tolerance. The cost of answering that with a wholesale re-gap, over 500 pool
    cases: V_grouping 6.74 per case on the plain path against 4.36 on aware, V_boundary 3.09
    against 2.33. Same defect class as the MIB split -- an all-or-nothing decision triggered by
    a numerically tiny local problem.

    So: find the offending pairs, shrink only their FREE members (about their centers, the same
    edit `open_gaps` makes, so area stays inside the 1% tolerance), and re-test. Blocks in no
    offending pair keep their exact contact. A few rounds, because clearing one pair can expose
    another. Returns the input unchanged with n_shrunk = 0 when nothing overlaps or when no
    offender is free to shrink -- the caller then falls back to `plain` as before.
    """
    pos = np.array(pos_ll, float)
    f = np.asarray(free, bool)
    prec = PREC_DEFAULT if prec is None else prec   # defined below this block
    n_shrunk = 0
    for _ in range(max(1, int(rounds))):
        pairs = _overlap_pairs(np.round(pos, prec))
        if not pairs:
            break
        m = np.zeros(len(pos), bool)
        for i, j in pairs:
            m[i] = m[j] = True
        m &= f                                   # immutable (w,h) may not be shrunk
        m = _mib_uniform(m, mib)                 # never split an MIB group's shared shape
        if not m.any():
            break                                # every offender is hard: nothing to try
        pos[m, 0] += pos[m, 2] * gap_frac / 2    # shrink about the center, as open_gaps does
        pos[m, 1] += pos[m, 3] * gap_frac / 2
        pos[m, 2] *= (1.0 - gap_frac)
        pos[m, 3] *= (1.0 - gap_frac)
        n_shrunk += int(m.sum())
    return pos, n_shrunk


def _precedence_ll(x, y, w, h):
    """Precedence graphs (H/V) from lower-left boxes; touching contacts assigned by the
    smaller-penetration axis (same convention as _precedence)."""
    N = len(x); xhi, yhi = x + w, y + h; Gh, Gv = [], []
    for i in range(N):
        for j in range(i + 1, N):
            ox = min(xhi[i], xhi[j]) - max(x[i], x[j])
            oy = min(yhi[i], yhi[j]) - max(y[i], y[j])
            horiz = (-ox <= -oy) if (ox <= 0 and oy <= 0) else (oy >= ox)
            if horiz:
                Gh.append((i, j) if x[i] <= x[j] else (j, i))
            else:
                Gv.append((i, j) if y[i] <= y[j] else (j, i))
    return Gh, Gv


def _longest_path(lo, size, edges, pinned):
    """Left/bottom-pack along a DAG so each edge (i->j) is EXACT-touching (lo[j]=lo[i]+size[i]).
    Pinned nodes keep their coordinate. Returns None on a cycle."""
    from collections import deque
    N = len(lo); adj = [[] for _ in range(N)]; ind = [0] * N
    for (i, j) in edges:
        adj[i].append(j); ind[j] += 1
    lo = lo.copy(); d = ind[:]; q = deque([k for k in range(N) if d[k] == 0]); order = []
    while q:
        u = q.popleft(); order.append(u)
        for v in adj[u]:
            d[v] -= 1
            if d[v] == 0:
                q.append(v)
    if len(order) < N:
        return None
    for u in order:
        for v in adj[u]:
            need = lo[u] + size[u]
            if not pinned[v] and lo[v] < need:
                lo[v] = need
    return lo


def aware_gaps(coords, free, is_pp, grouping=None, boundary=None, gap_frac=GAP_FRAC,
               mib=None):
    """Boundary/grouping-AWARE feasibility finish (recovers the low soft-violation V_rel
    that plain open_gaps sacrifices). Two steps on the solved layout:
      1. SNAP contacts to EXACT touching (longest-path left/bottom pack from the precedence
         graph). Exact touching has 0 penetration => feasible under the scorer, while
         grouped blocks stay ABUTTING (grouping satisfied) and flagged blocks stay FLUSH
         (boundary satisfied). Preplaced pinned.
      2. Open a gap ONLY on free blocks that are NOT grouped and NOT boundary-flagged, so
         those contacts clear the exact threshold without breaking any soft constraint.
    Emit at >=7 decimals so the exact-touching contacts don't round back into overlaps.
    coords [N,4]=[cx,cy,w,h]. Falls back to open_gaps on a precedence cycle.

    Pass `mib` to keep step 2 uniform inside each MIB group (see _mib_uniform). This path
    splits a group two ways -- a grouped or flagged member is held at full size while its
    unconstrained sibling shrinks -- so it needs the rule even when no member is immutable."""
    coords = np.asarray(coords, float)
    x = coords[:, 0] - coords[:, 2] / 2; y = coords[:, 1] - coords[:, 3] / 2
    w = coords[:, 2].copy(); h = coords[:, 3].copy()
    f = np.asarray(free, bool); pp = np.asarray(is_pp, bool)
    Gh, Gv = _precedence_ll(x, y, w, h)
    nx = _longest_path(x, w, Gh, pp); ny = _longest_path(y, h, Gv, pp)
    if nx is None or ny is None:
        return open_gaps(coords, free, gap_frac, mib=mib)   # cycle: plain gap
    grp = np.zeros(len(x)) if grouping is None else np.asarray(grouping)
    bnd = np.zeros(len(x)) if boundary is None else np.asarray(boundary)
    m = _mib_uniform(f & (grp == 0) & (bnd == 0), mib)   # gap only unconstrained free blocks
    nx = np.asarray(nx); ny = np.asarray(ny)
    nx[m] += w[m] * gap_frac / 2; ny[m] += h[m] * gap_frac / 2
    w[m] *= (1 - gap_frac); h[m] *= (1 - gap_frac)
    return np.stack([nx + w / 2, ny + h / 2, w, h], 1)


# --------------------------------------------------------------------------- #
# exact-contact snap (the feasibility/connectivity razor's edge)               #
# --------------------------------------------------------------------------- #
# The scorer applies TWO rules to the same contact, with DIFFERENT tolerances:
#   * hard overlap  : violation iff overlap_x > 1e-6 AND overlap_y > 1e-6   (a float tolerance)
#   * grouping union: shapely merges two boxes only on a SHARED BORDER of positive length --
#                     ZERO tolerance (a 1e-15 gap splits the group; a CORNER touch does not
#                     connect either).
# So a group contact must land in the window  gap in [-1e-6, 0]  -- touching, or overlapping by
# less than the hard threshold. `SNAP_OVERLAP` is how far into that window we aim.
#
# Why not simply set x_j = x_i + w_i (gap exactly 0)? Two reasons, and BOTH were measured:
#   * we move a whole component by a delta, so the block lands at x_j + ((x_i + w_i) - x_j),
#     which in floating point is NOT x_i + w_i (the subtraction cancels, leaving ~1 ulp); and
#   * we round the emitted coordinates, and the scorer then re-adds double(round(x_i)) +
#     double(round(w_i)), whose last bits do not reproduce double(round(x_j)).
# Either way the residual has a RANDOM SIGN, so half the "exact" contacts fall on the open side
# and disconnect. Measured: exact-0 targeting leaves V_grp 511-540 where a margin leaves 450
# (and 444 vs 367 with the abutment term) -- a ~17% penalty paid entirely to float noise.
#
# So aim a hair INSIDE the window. 1e-9 is ~1e3x the rounding noise at prec=12 (the contact
# never re-opens) and 1e3x UNDER the hard threshold (it never scores as an overlap). It gives
# exactly the same V_grp as a 1e-7 margin (450 / 370) with a 100x smaller overlap, so it is the
# strictly safer of the two. This is the tolerance the scorer grants precisely because exact
# float touching is unachievable -- but note the pipeline DOES depend on it: a scorer that
# rejected any overlap > 0 would fail these contacts. Keep prec >= 12 (see PREC_DEFAULT).
SNAP_OVERLAP = 1e-9
# SNAP_CROSS_FRAC (config) -- target shared-border length, as a fraction of the smaller
# block. v1.5.13 dropped it from 0.25 to 1e-6, the scorer's own predicate; see config.py.
PREC_DEFAULT = 12          # emission decimals; must stay >> SNAP_OVERLAP's rounding noise


def _hard_overlaps(pos):
    """Number of pairs violating the scorer's hard overlap check (threshold 1e-6)."""
    x, y, w, h = pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3]
    ox = np.minimum(x[:, None] + w[:, None], x[None, :] + w[None, :]) - \
        np.maximum(x[:, None], x[None, :])
    oy = np.minimum(y[:, None] + h[:, None], y[None, :] + h[None, :]) - \
        np.maximum(y[:, None], y[None, :])
    bad = (ox > 1e-6) & (oy > 1e-6)
    return int(np.triu(bad, 1).sum())


def _v_boundary(pos, boundary, eps=1e-6):
    """The scorer's V_boundary on a lower-left [N,4] layout (self-referential bbox edges)."""
    x, y, w, h = pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3]
    x0, y0, x1, y1 = x.min(), y.min(), (x + w).max(), (y + h).max()
    v = 0
    for i in range(len(pos)):
        code = int(boundary[i])
        if code == 0:
            continue
        t = {1: abs(x[i] - x0) < eps, 2: abs(x[i] + w[i] - x1) < eps,
             4: abs(y[i] + h[i] - y1) < eps, 8: abs(y[i] - y0) < eps}
        if not all(t[b] for b in (1, 2, 4, 8) if code & b):
            v += 1
    return v


def _v_grouping(pos, grouping):
    from .energy import grouping_components
    v = 0
    for g in np.unique(grouping[grouping > 0]):
        idx = np.where(grouping == g)[0]
        if len(idx) > 1:
            v += grouping_components(pos, idx) - 1
    return v


def _v_mib(pos, mib):
    """The scorer's V_mib on a lower-left [N,4] layout. The scorer compares dimensions
    ROUNDED TO 4 DECIMALS (iccad2026_evaluate.py, "distinct_shapes"), so a sub-1e-4 spread
    inside a group is NOT a violation -- do not "repair" one."""
    v = 0
    for g in np.unique(mib[mib > 0]):
        idx = np.where(mib == g)[0]
        v += len({(round(float(pos[i, 2]), 4), round(float(pos[i, 3]), 4)) for i in idx}) - 1
    return v


def _abut_deltas(bi, bj, snap_overlap, cross_frac):
    """Translations of box bj that give it a shared border with bi: one candidate per axis.
    Each closes the separation-axis gap to `-snap_overlap` (i.e. a hair INTO the box) and, if
    needed, shifts along the cross axis to buy a positive-length shared border."""
    xi, yi, wi, hi = bi
    xj, yj, wj, hj = bj
    out = []
    for axis in (0, 1):
        if axis == 0:
            near = (xi + wi / 2) <= (xj + wj / 2)
            dx = ((xi + wi) - xj) if near else ((xi - wj) - xj)
            dx += -snap_overlap if near else snap_overlap
            ov = min(yi + hi, yj + hj) - max(yi, yj)          # cross-axis (y) shared length
            want = cross_frac * min(hi, hj)
            dy = 0.0
            if ov < want:
                need = min(want, min(hi, hj)) - ov
                dy = -need if (yj + hj / 2) >= (yi + hi / 2) else need
            out.append((dx, dy))
        else:
            near = (yi + hi / 2) <= (yj + hj / 2)
            dy = ((yi + hi) - yj) if near else ((yi - hj) - yj)
            dy += -snap_overlap if near else snap_overlap
            ov = min(xi + wi, xj + wj) - max(xi, xj)
            want = cross_frac * min(wi, wj)
            dx = 0.0
            if ov < want:
                need = min(want, min(wi, wj)) - ov
                dx = -need if (xj + wj / 2) >= (xi + wi / 2) else need
            out.append((dx, dy))
    return out


def _obstructors(base_r, trial_r, moved):
    """Blocks OUTSIDE `moved` that the trial newly overlaps (the scorer's 1e-6 rule).

    The obstruction census for the snap autopsy: when a merge is declined for overlap, this
    is who was in the corridor. Returns a list of block indices."""
    others = np.setdiff1d(np.arange(len(base_r)), np.asarray(moved, int))
    if not len(others):
        return []
    m = np.asarray(moved, int)

    def ov(p):
        ax = np.minimum(p[m, 0, None] + p[m, 2, None], p[None, others, 0] + p[None, others, 2]) \
            - np.maximum(p[m, 0, None], p[None, others, 0])
        ay = np.minimum(p[m, 1, None] + p[m, 3, None], p[None, others, 1] + p[None, others, 3]) \
            - np.maximum(p[m, 1, None], p[None, others, 1])
        return (ax > 1e-6) & (ay > 1e-6)

    new = ov(trial_r) & ~ov(base_r)
    return [int(others[j]) for j in np.where(new.any(axis=0))[0]]


GRP_HARD_TIERS = ('all', 'free', 'small', 'bnd-all', 'bnd-free', 'bnd-mov', 'bnd-pp',
                  'both-all', 'both-free')   # drop-and-retry order, cheapest conflict first
GRP_HARD_OV = 0.25         # required shared-border length, as a fraction of the smaller block

CLEAR_MOVES = ('push', 'step', 'reshape', 'split', 'thin')
CLEAR_DEPTH = 3            # push-chain: how many rings of obstructors may be carried along
CLEAR_EVICT = 4            # step: how many obstructors may be evicted from one corridor


def _overlaps_any(p, i, exclude=()):
    """Does block i violate the scorer's hard overlap rule against anything else?"""
    ox = np.minimum(p[i, 0] + p[i, 2], p[:, 0] + p[:, 2]) - np.maximum(p[i, 0], p[:, 0])
    oy = np.minimum(p[i, 1] + p[i, 3], p[:, 1] + p[:, 3]) - np.maximum(p[i, 1], p[:, 1])
    bad = (ox > 1e-6) & (oy > 1e-6)
    bad[i] = False
    for k in exclude:
        bad[k] = False
    return bool(bad.any())


def _clear_split(pos, moved, other, dx, dy, movable, prec, base_ov):
    """A3. Move BOTH components toward each other, half the gap each.

    The rigid move walks one whole component across the corridor and is declined when its
    destination is occupied. Halving the trip halves the swept region on each side, so an
    obstruction that only fouls the far end of the corridor no longer fouls anything. Needs
    both components movable -- an anchored one contributes nothing and the caller falls back to
    the whole-gap move."""
    if any(not movable[i] for i in other):
        return None
    trial = pos.copy()
    trial[moved, 0] += dx / 2.0
    trial[moved, 1] += dy / 2.0
    trial[other, 0] -= dx / 2.0
    trial[other, 1] -= dy / 2.0
    return None if _hard_overlaps(np.round(trial, prec)) > base_ov else trial


def _clear_thin(pos, moved, dx, dy, reshapeable, prec, base_ov, ar_max=AR_MAX):
    """A6. Reshape the group MEMBER instead of the obstructor -- the dual of `_clear_reshape`.

    The member leading the move gives up extent along the direction of travel and takes it back
    on the cross axis, area-exact, holding its trailing edge fixed. That both shortens the slide
    (the leading edge starts further forward) and narrows what it sweeps. Free in area by
    construction, bounded by the AR limit, and forbidden for MIB members whose (w,h) is shared."""
    axis = 0 if abs(dx) >= abs(dy) else 1
    lead = None
    for i in moved:
        if not reshapeable[i]:
            continue
        if lead is None or (pos[i, axis] > pos[lead, axis]) == (dx if axis == 0 else dy) > 0:
            lead = i
    if lead is None:
        return None
    trial = pos.copy()
    w0, h0 = pos[lead, 2], pos[lead, 3]
    shrink = min(0.5 * abs(dx if axis == 0 else dy), 0.25 * (w0 if axis == 0 else h0))
    if shrink <= 0:
        return None
    if axis == 0:
        nw = w0 - shrink; nh = (w0 * h0) / nw
        if nh > ar_max * nw:
            return None
        trial[lead, 2], trial[lead, 3] = nw, nh
        if dx > 0:                       # travelling right: hold the LEFT edge
            trial[lead, 1] -= (nh - h0) / 2.0
        else:
            trial[lead, 0] += (w0 - nw); trial[lead, 1] -= (nh - h0) / 2.0
    else:
        nh = h0 - shrink; nw = (w0 * h0) / nh
        if nw > ar_max * nh:
            return None
        trial[lead, 2], trial[lead, 3] = nw, nh
        if dy > 0:
            trial[lead, 0] -= (nw - w0) / 2.0
        else:
            trial[lead, 1] += (h0 - nh); trial[lead, 0] -= (nw - w0) / 2.0
    trial[moved, 0] += dx
    trial[moved, 1] += dy
    return None if _hard_overlaps(np.round(trial, prec)) > base_ov else trial


def _clear_push(pos, moved, dx, dy, movable, prec, base_ov, depth=CLEAR_DEPTH):
    """M-a. Translate the component AND the blocks it runs into by the same delta.

    The corridor is cleared by carrying its occupants along, one ring at a time: whoever the
    enlarged set newly overlaps is added to it and moves too. Blocks inside the set keep their
    relative geometry, so no contact the set already had can break. Aborts on a PREPLACED
    obstructor (its (x,y) is hard input; a fixed-SHAPE block may still be translated) or if the
    chain has not closed within `depth` rings -- a deep chain is a wholesale rearrangement, not
    a corridor clear."""
    base_r = np.round(pos, prec)
    cur = list(moved)
    for _ in range(depth):
        trial = pos.copy()
        trial[cur, 0] += dx
        trial[cur, 1] += dy
        if _hard_overlaps(np.round(trial, prec)) <= base_ov:
            return trial
        obs = [o for o in _obstructors(base_r, np.round(trial, prec), cur) if o not in cur]
        if not obs or any(not movable[o] for o in obs):
            return None
        cur = cur + obs
    return None


def _clear_step(pos, moved, dx, dy, movable, prec, base_ov, max_evict=CLEAR_EVICT):
    """M-b. Evict the movable obstructors sideways, out of the corridor.

    The full eviction would relocate each obstructor into an inventoried void; this is the
    cheap version of the same idea -- an obstructor is offered the four minimal translations
    that clear it out of the destination box, and takes the first that lands somewhere empty.
    That is a 'nearest void' search restricted to the four axis-aligned voids touching the
    corridor, which is where the room usually is. Obstructors are evicted one at a time,
    innermost first, each seeing the ones already moved; up to `max_evict`, because the median
    corridor holds three blocks and a one-block rule leaves most of them on the table."""
    base_r = np.round(pos, prec)
    trial = pos.copy()
    trial[moved, 0] += dx
    trial[moved, 1] += dy
    obs = _obstructors(base_r, np.round(trial, prec), moved)
    if not obs or len(obs) > max_evict or any(not movable[o] for o in obs):
        return None
    mx0 = trial[moved, 0].min(); mx1 = (trial[moved, 0] + trial[moved, 2]).max()
    my0 = trial[moved, 1].min(); my1 = (trial[moved, 1] + trial[moved, 3]).max()
    # deepest first: an obstructor buried in the corridor has the fewest ways out, so let it
    # pick before the shallow ones spend the surrounding room
    obs.sort(key=lambda o: -min(mx1 - pos[o, 0], pos[o, 0] + pos[o, 2] - mx0,
                                my1 - pos[o, 1], pos[o, 1] + pos[o, 3] - my0))
    for o in obs:
        ox0, oy0, ow, oh = trial[o]
        placed = False
        for (sx, sy) in ((mx1 - ox0, 0.0), (mx0 - (ox0 + ow), 0.0),
                         (0.0, my1 - oy0), (0.0, my0 - (oy0 + oh))):
            t = trial.copy()
            t[o, 0] += sx + np.sign(sx) * SNAP_OVERLAP
            t[o, 1] += sy + np.sign(sy) * SNAP_OVERLAP
            if not _overlaps_any(np.round(t, prec), o):
                trial = t; placed = True
                break
        if not placed:
            return None
    return trial if _hard_overlaps(np.round(trial, prec)) <= base_ov else None


def _clear_reshape(pos, moved, dx, dy, reshapeable, prec, base_ov, ar_max=AR_MAX):
    """M-c. Make the single obstructor thinner, area-exact, so the corridor opens.

    Nobody is displaced: the obstructor gives up width (or height) along the direction of
    travel and takes it back on the cross axis, holding the edge AWAY from the incoming
    component fixed. Restricted to blocks that are free and outside any MIB group -- the same
    rule `pull_boundary`'s reshape follows, because an MIB member's (w,h) is shared."""
    base_r = np.round(pos, prec)
    trial = pos.copy()
    trial[moved, 0] += dx
    trial[moved, 1] += dy
    obs = _obstructors(base_r, np.round(trial, prec), moved)
    if len(obs) != 1 or not reshapeable[obs[0]]:
        return None
    o = obs[0]
    ox0, oy0, ow, oh = pos[o]
    area = ow * oh
    mx0 = trial[moved, 0].min(); mx1 = (trial[moved, 0] + trial[moved, 2]).max()
    my0 = trial[moved, 1].min(); my1 = (trial[moved, 1] + trial[moved, 3]).max()
    for axis in (0, 1):
        if axis == 0:
            keep_hi = (ox0 + ow / 2) >= (mx0 + mx1) / 2      # obstructor sits to the RIGHT
            nw = (ox0 + ow) - mx1 if keep_hi else mx0 - ox0
            if nw <= 1e-9:
                continue
            nh = area / nw
            if max(nw / nh, nh / nw) > ar_max + 1e-9:
                continue
            t = trial.copy()
            t[o, 0] = (ox0 + ow - nw) if keep_hi else ox0
            t[o, 1] = oy0 + (oh - nh) / 2                    # grow about the cross-axis center
            t[o, 2] = nw; t[o, 3] = nh
        else:
            keep_hi = (oy0 + oh / 2) >= (my0 + my1) / 2
            nh = (oy0 + oh) - my1 if keep_hi else my0 - oy0
            if nh <= 1e-9:
                continue
            nw = area / nh
            if max(nw / nh, nh / nw) > ar_max + 1e-9:
                continue
            t = trial.copy()
            t[o, 1] = (oy0 + oh - nh) if keep_hi else oy0
            t[o, 0] = ox0 + (ow - nw) / 2
            t[o, 2] = nw; t[o, 3] = nh
        if _hard_overlaps(np.round(t, prec)) <= base_ov:
            return t
    return None


def snap_groups(pos_ll, is_pp, grouping, boundary, prec=PREC_DEFAULT, passes=4,
                snap_overlap=SNAP_OVERLAP, cross_frac=SNAP_CROSS_FRAC, cost_aware=True,
                mib=None, max_area_growth=None, log=None, clear=(), is_fs=None,
                hpwl_nets=None):
    """Pull the disconnected pieces of each grouping-group into contact. Returns (pos, n_moves).

    The M9 solve minimizes W+H, so off the critical path it has no reason to keep a group's
    members together and they drift apart inside their slack; and even a "touching" contact
    carries the solver's residual (~1e-8), which the scorer's zero-tolerance union reads as a
    gap. This is the repair: a hill-climb that rigidly translates one whole COMPONENT of a
    group (rigid => it never breaks the contacts that component already has) so its nearest
    member abuts the other component, and keeps the move only if it strictly improves
    (V_grouping + V_boundary) without introducing a hard overlap. Never moves a component that
    contains a preplaced block (their (x,y) is a hard constraint). Monotone by construction.

    Everything is scored on the coordinates ROUNDED to `prec` -- the numbers we actually
    submit -- because the whole question is decided in the last decimals.

    `log`: pass a list to record one dict per ATTEMPTED merge (accept, or the reason it was
    declined: 'anchored' / 'overlap' / 'cap' / 'cost'), plus an obstruction census for the
    overlap declines. Diagnostic only -- it never changes a decision. See tools/archived_experiments/snap_autopsy.py.

    `clear`: obstruction-clearing finishers (G2), tried only where the plain rigid translation
    is declined for a hard overlap -- which the autopsy measured at 88% of all declines, with
    the cap and the priced test together under 1.5%. Each entry of CLEAR_MOVES is a different
    answer to "who gives way": 'push' carries the obstructors along, 'step' evicts a single one
    sideways, 'reshape' makes it thinner area-exactly. They run through the SAME priced
    acceptance and hard-overlap gates as the rigid move, so the worst case is no accepted moves.
    `is_fs` is only needed by 'reshape' (an MIB member's shape is shared and may not change).
    """
    pos = np.array(pos_ll, float)
    grouping = np.asarray(grouping, int)
    boundary = np.asarray(boundary, int)
    is_pp = np.asarray(is_pp, bool)
    from .energy import grouping_components

    def bbox_area(r):
        return (((r[:, 0] + r[:, 2]).max() - r[:, 0].min()) *
                ((r[:, 1] + r[:, 3]).max() - r[:, 1].min()))

    mib = np.zeros(len(pos), int) if mib is None else np.asarray(mib, int)
    if max_area_growth is not None and max_area_growth < 0:
        raise ValueError('max_area_growth must be non-negative or None')
    bad = set(clear) - set(CLEAR_MOVES)
    if bad:
        raise ValueError(f'clear must be a subset of {CLEAR_MOVES}, got {sorted(bad)}')
    is_fs_a = np.zeros(len(pos), bool) if is_fs is None else np.asarray(is_fs, bool)
    movable = ~(is_pp | is_fs_a)          # a fixed SHAPE may still be translated
    movable_pos = ~is_pp
    reshapeable = movable & (mib == 0)
    n_soft = max(v10_soft_denominator(boundary, mib, grouping), 1)
    area0 = bbox_area(np.round(pos, prec))

    hpwl0 = (max(v10_total_hpwl(np.round(pos, prec), *hpwl_nets), 1e-9)
             if hpwl_nets is not None and hpwl_nets[0] is not None else None)

    def score(p):
        """Lower is better. cost_aware => the contest's own trade: a move that abuts a group
        but inflates the outline is only worth it if exp(2*V_rel) falls by more than the area
        term rises. Cost = (1 + 0.5*area_gap) * exp(2*V/N_soft), self-referential (area_gap is
        measured against the layout we started from), which is all we can see at legalize time.
        Without this the hill-climb takes ANY violation-reducing move and pays for it in bbox.

    `hpwl_nets` (J1/H10) turns the self-referential price into the CONTEST'S OWN price. The
    scorer sums `area_gap` and `hpwl_gap` inside one ALPHA, but this hill-climb weighed only
    area, so a move that abutted a group by dragging a heavily-netted block across the layout
    looked free. Same defect class as B8-a, which found it in the candidate-acceptance test and
    turned 34-better/10-worse into 34/0. Both gaps are measured against the layout the climb
    started from, which is all that is visible at legalize time.
        """
        r = np.round(p, prec)
        v = _v_grouping(r, grouping) + _v_boundary(r, boundary)
        if not cost_aware:
            return (v, bbox_area(r))
        gap = bbox_area(r) / max(area0, 1e-9) - 1.0
        if hpwl0 is not None:
            gap += v10_total_hpwl(r, *hpwl_nets) / hpwl0 - 1.0
        return ((1.0 + ALPHA * gap) * np.exp(BETA * v / n_soft), 0.0)

    cur = score(pos)
    base_ov = _hard_overlaps(np.round(pos, prec))
    n_moves = 0
    for _ in range(passes):
        improved = False
        for g in np.unique(grouping[grouping > 0]):
            idx = np.where(grouping == g)[0]
            if len(idx) < 2:
                continue
            r = np.round(pos, prec)
            if grouping_components(r, idx) == 1:
                continue
            # components of THIS group, as index sets
            lbl = {}
            for a in idx:
                lbl[a] = a
            for a in idx:
                for b in idx:
                    if a < b and blocks_connected_np(r[a], r[b]):
                        ra, rb = lbl[a], lbl[b]
                        if ra != rb:
                            for k in idx:
                                if lbl[k] == rb:
                                    lbl[k] = ra
            comps = {}
            for a in idx:
                comps.setdefault(lbl[a], []).append(a)
            keys = list(comps)
            done = False
            for ki in range(len(keys)):
                for kj in range(len(keys)):
                    if ki == kj or done:
                        continue
                    A, B = comps[keys[ki]], comps[keys[kj]]
                    if any(is_pp[b] for b in B):
                        if log is not None:
                            log.append({'group': int(g), 'n_A': len(A), 'n_B': len(B),
                                        'axis': -1, 'a': -1, 'b': -1, 'dx': 0.0, 'dy': 0.0,
                                        'slide': 0.0, 'decision': 'anchored',
                                        'growth': 0.0, 'obstructors': []})
                        continue                      # B is anchored: it cannot move
                    # nearest member pair between the two components
                    best = min(((np.hypot(*_rect_gap_ll(pos[a], pos[b])), a, b)
                                for a in A for b in B), key=lambda t: t[0])
                    _, a, b = best
                    for axis, (dx, dy) in enumerate(
                            _abut_deltas(pos[a], pos[b], snap_overlap, cross_frac)):
                        trial = pos.copy()
                        trial[B, 0] += dx
                        trial[B, 1] += dy            # rigid: the component keeps its contacts
                        trial_r = np.round(trial, prec)
                        rec = None
                        if log is not None:
                            rec = {'group': int(g), 'n_A': len(A), 'n_B': len(B), 'axis': axis,
                                   'a': int(a), 'b': int(b), 'dx': float(dx), 'dy': float(dy),
                                   'slide': float(np.hypot(dx, dy)), 'decision': 'accept',
                                   'growth': float(bbox_area(trial_r) / max(area0, 1e-9) - 1.0),
                                   'obstructors': []}
                            log.append(rec)
                        cands = [('rigid', trial)]
                        if _hard_overlaps(trial_r) > base_ov:
                            if rec is not None:
                                rec['decision'] = 'overlap'
                                rec['obstructors'] = _obstructors(np.round(pos, prec),
                                                                  trial_r, B)
                            # G2: the corridor is occupied -- try clearing it instead of
                            # giving up. Each move is just another trial layout through the
                            # same gates below.
                            cands = []
                            for mv in clear:
                                if mv == 'push':
                                    t = _clear_push(pos, B, dx, dy, movable_pos, prec, base_ov)
                                elif mv == 'step':
                                    t = _clear_step(pos, B, dx, dy, movable_pos, prec, base_ov)
                                elif mv == 'split':
                                    t = _clear_split(pos, B, A, dx, dy, movable_pos, prec,
                                                     base_ov)
                                elif mv == 'thin':
                                    t = _clear_thin(pos, B, dx, dy, reshapeable, prec, base_ov)
                                else:
                                    t = _clear_reshape(pos, B, dx, dy, reshapeable, prec,
                                                       base_ov)
                                if t is not None:
                                    cands.append((mv, t))
                            if not cands:
                                continue
                        for mv, trial in cands:
                            trial_r = np.round(trial, prec)
                            outcome, s = 'accept', None
                            if _hard_overlaps(trial_r) > base_ov:
                                outcome = 'overlap'
                            elif (max_area_growth is not None and
                                  bbox_area(trial_r) > area0 * (1.0 + max_area_growth) + 1e-9):
                                outcome = 'cap'
                            else:
                                s = score(trial)
                                if s >= cur:
                                    outcome = 'cost'
                            if rec is not None and mv != 'rigid':
                                rec.setdefault('clear', []).append((mv, outcome))
                            if outcome != 'accept':
                                if rec is not None and mv == 'rigid':
                                    rec['decision'] = outcome
                                continue
                            pos = trial; cur = s
                            n_moves += 1; improved = True; done = True
                            if rec is not None:
                                rec['decision'] = 'accept' if mv == 'rigid' else f'clear:{mv}'
                            break
                        if done:
                            break
        if not improved:
            break
    return pos, n_moves


def snap_mib_shapes(pos_ll, is_pp, is_fs, mib, grouping=None, boundary=None,
                    prec=PREC_DEFAULT, cost_aware=True):
    """Undo the gap step's shape split inside each MIB group. Returns (pos, n_snapped).

    The conic program gives every member of an MIB group ONE (w,h) variable pair, so the
    solve leaves V_mib = 0. The gap finish then breaks it: open_gaps/aware_gaps shrink each
    FREE block by (1-gap_frac) and leave preplaced/fixed-shape blocks EXACT, so a group with
    an immutable member emits two shapes 0.1% apart -- hard-feasible (0.998x area, inside the
    1% tolerance) and silently worth a soft point. Same defect class as the grouping ruler in
    v1.5.5: a zero-tolerance predicate broken by a numerically tiny post-solve edit.

    So this is the INVERSE of that edit, not a fresh placement decision:

    * The anchor is the group's un-gapped shape -- the immutable member's exact dimensions
      when one exists, else the componentwise max (the max IS the pre-gap shape, since the
      gap only ever shrinks). Never the min: that would shrink members the gap step
      deliberately skipped, e.g. an aware_gaps block held at full size because it is
      grouped or boundary-flagged.
    * Members are resized ABOUT THEIR CENTER, because that is how the gap shrank them
      (open_gaps: "centers unchanged"; aware_gaps: nx += w*gap_frac/2). A resize that holds
      the lower-left instead displaces the box by delta/2 into its right/upper neighbour and
      needlessly fails the overlap check: on results/solve_repair_pool that repairs 32 of the
      71 violating cases where center-restoring repairs 64. Center-restoring returns the
      block to the box the solve already proved overlap-free.
    * Immutable rows are never written. A group whose immutable members disagree by more than
      the scorer's 1e-4 dimension tolerance has no legal common shape and is skipped, so the
      repair can never edit a hard-constrained dimension.

    Gated per GROUP, like snap_groups gates per component: a group that cannot be restored
    without a hard overlap must not cost the other groups their repair. A group is snapped
    only when it actually reduces the scorer's V_mib (a sub-1e-4 spread is already 0
    violations -- "repairing" it moves edges by ~1e-7 and can break a zero-tolerance grouping
    contact for nothing), when it adds no hard overlap, and, with `cost_aware`, when the
    priced soft cost does not get worse. All three soft terms share one denominator, so the
    MIB gain and any grouping/boundary loss are directly comparable.
    """
    pos = np.array(pos_ll, float)
    mib = np.asarray(mib, int)
    n = len(pos)
    grp = np.zeros(n, int) if grouping is None else np.asarray(grouping, int)
    bnd = np.zeros(n, int) if boundary is None else np.asarray(boundary, int)
    immutable = np.asarray(is_pp, bool) | np.asarray(is_fs, bool)

    n_soft = max(v10_soft_denominator(bnd, mib, grp), 1)
    area0 = _bbox_area_ll(np.round(pos, prec))

    def score(p):
        r = np.round(p, prec)
        v = _v_boundary(r, bnd) + _v_grouping(r, grp) + _v_mib(r, mib)
        if not cost_aware:
            return (v, _bbox_area_ll(r))
        return ((1.0 + 0.5 * (_bbox_area_ll(r) / max(area0, 1e-9) - 1.0)) *
                np.exp(2.0 * v / n_soft), 0.0)

    cur = score(pos)
    base_ov = _hard_overlaps(np.round(pos, prec))
    n_snapped = 0
    for g in np.unique(mib[mib > 0]):
        idx = np.where(mib == g)[0]
        if len(idx) < 2:
            continue
        r = np.round(pos, prec)
        if len({(round(float(r[i, 2]), 4), round(float(r[i, 3]), 4)) for i in idx}) < 2:
            continue                                  # already one shape to the scorer
        fixed = idx[immutable[idx]]
        if len(fixed):
            wh = pos[fixed[0], 2:4].copy()
            if np.abs(pos[fixed, 2:4] - wh).max() > 1e-4:
                continue                              # no shape can satisfy every hard member
        else:
            wh = pos[idx, 2:4].max(axis=0)
        movable = idx[~immutable[idx]]
        trial = pos.copy()
        trial[movable, :2] += (trial[movable, 2:4] - wh) / 2.0   # resize about the center
        trial[movable, 2:4] = wh
        if _hard_overlaps(np.round(trial, prec)) > base_ov:
            continue
        s = score(trial)
        if s < cur:
            pos = trial; cur = s
            n_snapped += 1
    return pos, n_snapped


def pull_boundary(pos_ll, is_pp, is_fs, boundary, grouping=None, mib=None, prec=PREC_DEFAULT,
                  passes=4, cost_aware=True, reshape=False, log=None, batch=False,
                  hpwl_nets=None):
    """Pull each violating boundary-flagged block onto its self-referential bbox wall
    (v1.5.6). Returns (pos, n_moves).

    Same shape as snap_groups: a feasibility-gated hill-climb. The M9 solve does not touch
    boundary at all, and the gap/snap finishes leave flagged blocks a hair off their wall
    (~0.01-0.1% of span -- a near-miss, see tools/archived_experiments/bnd_diag.py). For each flagged block that
    misses an edge, try the SINGLE move that lands its edge exactly on the wall (L->x_min,
    R->x_max, B->y_min, T->y_max; a corner does both axes at once), and keep it only if it
    strictly improves the priced soft cost AND adds no hard overlap. So an interior flagged
    block whose slide-path is blocked is simply left alone; resolving it needs a global
    arrangement change. Preplaced never move (pinned); a block
    flagged on BOTH opposite edges (L+R / T+B) cannot be satisfied by a move and is skipped
    on that axis. Everything scored on the coords ROUNDED to `prec` -- the boundary check has
    a 1e-6 tolerance, so no snap margin is needed (unlike grouping's zero-tolerance union).

    Unlike grouping, the boundary check is SELF-REFERENTIAL: moving one block can shift the
    bbox and (un)satisfy others, so the bbox is recomputed every evaluation and the passes
    iterate to a fixed point.

    reshape=True additionally lets a FREE block that cannot slide GROW toward its wall
    (area-exact: widen+shorten for L/R, taller+narrower for T/B) within the AR limit.

    batch=True (E3) adds one priced move PER WALL that translates EVERY movable block flagged
    to that wall onto it at once. The per-block move above is monotone in the priced score, so
    it can never take a step that needs two blocks to move together -- and because the bbox is
    self-referential, that is a real class: the block holding the wall out is often itself
    flagged to it, so neither moves alone but both move together. Same feasibility and priced
    gates as the single move, so the worst case is no accepted moves.
    """
    pos = np.array(pos_ll, float)
    boundary = np.asarray(boundary, int)
    is_pp = np.asarray(is_pp, bool); is_fs = np.asarray(is_fs, bool)
    grp = np.zeros(len(pos), int) if grouping is None else np.asarray(grouping, int)
    mib = np.zeros(len(pos), int) if mib is None else np.asarray(mib, int)
    # reshape changes a block's (w,h); an MIB-group member sharing its shape may NOT be reshaped
    # (it would break V_mib), so restrict reshape to free blocks outside any MIB group.
    reshapeable = (~(is_pp | is_fs)) & (mib == 0)

    n_soft = max(v10_soft_denominator(boundary, mib, grp), 1)
    area0 = _bbox_area_ll(np.round(pos, prec))

    hpwl0 = (max(v10_total_hpwl(np.round(pos, prec), *hpwl_nets), 1e-9)
             if hpwl_nets is not None and hpwl_nets[0] is not None else None)

    def score(p):
        """Lower is better, same priced trade the grouping snap uses.

    `hpwl_nets` (J1/H10) turns the self-referential price into the CONTEST'S OWN price. The
    scorer sums `area_gap` and `hpwl_gap` inside one ALPHA, but this hill-climb weighed only
    area, so a move that abutted a group by dragging a heavily-netted block across the layout
    looked free. Same defect class as B8-a, which found it in the candidate-acceptance test and
    turned 34-better/10-worse into 34/0. Both gaps are measured against the layout the climb
    started from, which is all that is visible at legalize time.
        """
        r = np.round(p, prec)
        v = _v_boundary(r, boundary) + _v_grouping(r, grp)
        if not cost_aware:
            return (v, _bbox_area_ll(r))
        gap = _bbox_area_ll(r) / max(area0, 1e-9) - 1.0
        if hpwl0 is not None:
            gap += v10_total_hpwl(r, *hpwl_nets) / hpwl0 - 1.0
        return ((1.0 + ALPHA * gap) * np.exp(BETA * v / n_soft), 0.0)

    cur = score(pos)
    base_ov = _hard_overlaps(np.round(pos, prec))
    n_moves = 0
    for _ in range(passes):
        improved = False
        r = np.round(pos, prec)
        x, y, w, h = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        x0, y0, x1, y1 = x.min(), y.min(), (x + w).max(), (y + h).max()
        for i in range(len(pos)):
            code = int(boundary[i])
            if code == 0 or is_pp[i]:
                continue
            xi, yi, wi, hi = pos[i]
            dx = dy = 0.0
            if (code & 1) and not (code & 2): dx = x0 - xi
            elif (code & 2) and not (code & 1): dx = (x1 - wi) - xi
            if (code & 8) and not (code & 4): dy = y0 - yi
            elif (code & 4) and not (code & 8): dy = (y1 - hi) - yi
            cands = []
            if dx != 0.0 or dy != 0.0:
                t = pos.copy(); t[i, 0] += dx; t[i, 1] += dy
                cands.append(t)
            if reshape and reshapeable[i]:
                cands.append(_reshape_to_wall(pos, i, code, x0, y0, x1, y1))
            for k, trial in enumerate(cands):
                if trial is None:
                    continue
                rec = None
                if log is not None:
                    span = (x1 - x0) if (dx != 0.0) else (y1 - y0)
                    rec = {'block': i, 'code': code, 'kind': 'slide' if k == 0 else 'reshape',
                           'pinned': bool(is_pp[i]), 'fixed_shape': bool(is_fs[i]),
                           'dist': float(np.hypot(dx, dy)) / max(span, 1e-9),
                           'decision': 'accept', 'obstructors': []}
                    log.append(rec)
                if _hard_overlaps(np.round(trial, prec)) > base_ov:
                    if rec is not None:
                        rec['decision'] = 'overlap'
                        rec['obstructors'] = _obstructors(np.round(pos, prec),
                                                          np.round(trial, prec), [i])
                    continue
                s = score(trial)
                if s < cur:
                    pos = trial; cur = s; n_moves += 1; improved = True
                    break
                if rec is not None:
                    rec['decision'] = 'cost'
            if batch:
                r = np.round(pos, prec)
                x, y, w, h = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
                x0, y0, x1, y1 = x.min(), y.min(), (x + w).max(), (y + h).max()
                for bit, opp, axis, tgt in ((1, 2, 0, lambda i: x0),
                                            (2, 1, 0, lambda i: x1 - pos[i, 2]),
                                            (8, 4, 1, lambda i: y0),
                                            (4, 8, 1, lambda i: y1 - pos[i, 3])):
                    sel = [i for i in range(len(pos))
                           if (int(boundary[i]) & bit) and not (int(boundary[i]) & opp)
                           and not is_pp[i] and abs(pos[i, axis] - tgt(i)) > 1e-12]
                    if len(sel) < 2:
                        continue
                    trial = pos.copy()
                    for i in sel:
                        trial[i, axis] = tgt(i)
                    if _hard_overlaps(np.round(trial, prec)) > base_ov:
                        continue
                    s = score(trial)
                    if s < cur:
                        pos = trial; cur = s; n_moves += 1; improved = True
        if not improved:
            break
    return pos, n_moves


def _reshape_to_wall(pos, i, code, x0, y0, x1, y1, ar_max=AR_MAX):
    """Area-exact reshape of free block i so a flagged edge reaches its wall (grow toward it).
    Returns a trial [N,4] or None if the target shape breaks the AR limit."""
    xi, yi, wi, hi = pos[i]
    area = wi * hi
    nw, nh = wi, hi
    if (code & 2) and not (code & 1):                 # R: widen so xi+nw = x1
        nw = x1 - xi
    elif (code & 1) and not (code & 2):               # L: block already left-shifted elsewhere
        return None
    if (code & 4) and not (code & 8):                 # T: heighten so yi+nh = y1
        nh = y1 - yi
    if nw <= 0 or nh <= 0:
        return None
    if (code & 2) and not (code & 4):                 # width was set -> height follows area
        nh = area / nw
    elif (code & 4) and not (code & 2):
        nw = area / nh
    else:
        return None                                   # corner reshape: skip (ambiguous)
    if max(nw / nh, nh / nw) > ar_max + 1e-9:
        return None
    t = pos.copy(); t[i, 2] = nw; t[i, 3] = nh        # lower-left fixed -> grows up/right
    return t


def _bbox_area_ll(p):
    return (((p[:, 0] + p[:, 2]).max() - p[:, 0].min()) *
            ((p[:, 1] + p[:, 3]).max() - p[:, 1].min()))


def _rect_gap_ll(bi, bj):
    gx = max(bi[0], bj[0]) - min(bi[0] + bi[2], bj[0] + bj[2])
    gy = max(bi[1], bj[1]) - min(bi[1] + bi[3], bj[1] + bj[3])
    return max(gx, 0.0), max(gy, 0.0)


def blocks_connected_np(bi, bj):
    gx = max(bi[0], bj[0]) - min(bi[0] + bi[2], bj[0] + bj[2])
    gy = max(bi[1], bj[1]) - min(bi[1] + bi[3], bj[1] + bj[3])
    return (gx <= 0.0 and gy < 0.0) or (gy <= 0.0 and gx < 0.0)


def hpwl_b2b_rows(b2b_conn, n):
    """The scorer's b2b rows as vectorized `(i, j, weight)` arrays for the solve objective.

    A row-for-row match with `scoring.v10_total_hpwl`: every raw contest row is kept and SUMMED.
    Never `edge_index_full`, which max-collapses duplicates and discards ~19.8% of the weight
    mass (SCORER_FIDELITY_BUGS B1). Self-pairs are dropped -- their distance is identically 0,
    so they contribute nothing to the objective and only cost canonicalization.
    """
    b = np.asarray(b2b_conn, dtype=float)
    if b.ndim == 3:
        b = b[0]
    if not b.size:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, float)
    b = b[b[:, 0] != -1]
    b = b[(b[:, 0] < n) & (b[:, 1] < n)] if b.size else b
    if not b.size:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, float)
    i, j, wt = b[:, 0].astype(int), b[:, 1].astype(int), b[:, 2].astype(float)
    keep = i != j
    return i[keep], j[keep], wt[keep]


def frozen_resolve(solved_c, is_pp, is_fs, area, mib, b2b_conn, eps=0.0, kind='hpwl',
                   sep_rule=None, grp_dag=False, grouping=None, boundary=None,
                   area_slack=0.0, lam_bnd=1.0):
    """H5/H6/H8. Re-place a SOLVED layout under a new objective at FIXED topology.

    `solved_c` is `[N,4] = [cx,cy,w,h]` straight out of `compact_solve` -- the layout whose
    precedence graph we are freezing. The graph is re-inferred from it with the SAME rules the
    solve used, so the ordering handed to the program is exactly the one that produced this
    packing; only the positions and shapes inside it are re-optimised.

    `eps` is H6: allow the outline's perimeter to grow by this fraction. `eps=0` is H5 proper --
    strictly no bbox growth, so the wirelength gain has to come out of slack that was already
    there. Returns centre coords like `_solve_once`, or None if the program did not solve.
    """
    cx, cy, w, h = solved_c[:, 0], solved_c[:, 1], solved_c[:, 2], solved_c[:, 3]
    n = len(cx)
    rows = None
    if kind in ('hpwl', 'both'):
        rows = hpwl_b2b_rows(b2b_conn, n)
        if not len(rows[0]):
            return None
    bnd_soft = None
    if kind in ('bnd', 'both'):
        bnd_soft = np.zeros(n, int) if boundary is None else np.asarray(boundary, int)
        if not (bnd_soft != 0).any():
            return None
    fixed = np.asarray(is_pp, bool) | np.asarray(is_fs, bool)
    free = ~fixed
    mib = np.zeros(n, int) if mib is None else np.asarray(mib, int)
    Gh, Gv = _precedence(cx, cy, w, h, is_pp=is_pp, sep_rule=sep_rule)
    if grp_dag:
        Gh, Gv, _ = apply_dag_rules(Gh, Gv, cx, cy, w, h, grouping, boundary, is_pp,
                                    ('b2',) if grp_dag is True else grp_dag)
    tgt_area = np.asarray(area, float) * (1.0 - float(area_slack))
    cap = ((cx + w / 2).max() - (cx - w / 2).min()) + ((cy + h / 2).max() - (cy - h / 2).min())
    return _solve_once(cx, cy, w, h, tgt_area, free, fixed, is_pp, mib, Gh, Gv,
                       hpwl_rows=rows, bnd_soft=bnd_soft, lam_bnd=lam_bnd,
                       bbox_cap=cap * (1.0 + float(eps)))


def compact_solve(coords, is_pp, is_fs, mib=None, boundary=None, iters=3,
                  grouping=None, lam_grp=0.0, ov_frac=0.25,
                  net_edges=(), lam_net=0.0, area_slack=0.0,
                  reseed_rounds=0, reseed_lam=SOLVE_RESEED_LAM,
                  sep_rule=None, grp_hard=(), grp_hard_ov=GRP_HARD_OV, alt=None,
                  grp_dag=False, grp_hard_min_bnd=0, grp_hard_precheck=False,
                  net_pairs=None, net_invert=False, lam_bnd_add=0.0, true_bbox=False):
    """The conic solve ONLY (no feasibility finish): returns (coords, solved). Split out so a
    caller can share one (expensive) solve across several finishes — e.g. the 'hybrid' handoff
    needs both the plain and the aware gap-finish of the SAME solved layout.

    lam_grp > 0 adds the grouping-abutment penalty (objective only) over a spanning tree of
    each group; needs `grouping`.

    reseed_rounds > 0 enables the soft-pin reseed after both ordinary attempts fail; it only
    ever runs on a placement that has NO solve at all, so it cannot change a solved case.

    grp_hard (G3) names the drop-and-retry tiers for the hard-abutment pass; when it solves,
    that SECOND layout is appended to the list `alt` and the return value is unchanged, so the
    caller decides between them on the finished, priced layouts rather than on the raw solve
    (whose exact-0 contacts carry a random-signed solver residual and cannot be scored)."""
    return _compact_core(coords, is_pp, is_fs, mib, boundary, iters,
                         grouping=grouping, lam_grp=lam_grp, ov_frac=ov_frac,
                         net_edges=net_edges, lam_net=lam_net, area_slack=area_slack,
                         reseed_rounds=reseed_rounds, reseed_lam=reseed_lam,
                         sep_rule=sep_rule, grp_hard=grp_hard, grp_hard_ov=grp_hard_ov,
                         alt=alt, grp_dag=grp_dag, grp_hard_min_bnd=grp_hard_min_bnd,
                         grp_hard_precheck=grp_hard_precheck, net_pairs=net_pairs,
                         net_invert=net_invert, lam_bnd_add=lam_bnd_add,
                         true_bbox=true_bbox)


def compact_joint(coords, is_pp, is_fs, mib=None, boundary=None, iters=3,
                  gap_frac=GAP_FRAC, gap_mode='plain', grouping=None):
    """coords [N,4]=[cx,cy,w,h] -> reshaped+compacted coords (area-exact within the 1%
    tolerance, overlap-free under the EXACT scorer when feasible; original placement returned
    unchanged if the solve can't legalize). is_pp/is_fs are bool masks; mib is an int
    group-id array (0=none); boundary is the L/R/T/B code array.

    gap_mode: 'plain' (open_gaps: gap every free block; simplest, max feasibility, but breaks
    abutment/flush so V_rel rises) or 'aware' (aware_gaps: snap contacts exact + gap only
    unconstrained free blocks; keeps grouping/boundary => low V_rel). 'aware' needs `grouping`
    (int group-ids) and `boundary` (L/R/T/B codes)."""
    best, solved = _compact_core(coords, is_pp, is_fs, mib, boundary, iters)
    if not solved:
        return best
    free = ~(np.asarray(is_pp, bool) | np.asarray(is_fs, bool))
    if gap_mode == 'aware':
        return aware_gaps(best, free, is_pp, grouping=grouping, boundary=boundary,
                          gap_frac=gap_frac)
    if gap_mode == 'none':
        return best
    return open_gaps(best, free, gap_frac)


def _compact_core(coords, is_pp, is_fs, mib=None, boundary=None, iters=3,
                  grouping=None, lam_grp=0.0, ov_frac=0.25,
                  net_edges=(), lam_net=0.0, area_slack=0.0,
                  reseed_rounds=0, reseed_lam=SOLVE_RESEED_LAM,
                  sep_rule=None, grp_hard=(), grp_hard_ov=GRP_HARD_OV, alt=None,
                  grp_dag=False, grp_hard_min_bnd=0, grp_hard_precheck=False,
                  net_pairs=None, net_invert=False, lam_bnd_add=0.0, true_bbox=False):
    if not _HAVE_CVXPY:
        raise RuntimeError('compact_joint needs cvxpy + CLARABEL (pip install cvxpy clarabel)')
    coords = np.asarray(coords, float)
    w, h = coords[:, 2].copy(), coords[:, 3].copy()
    sx = (coords[:, 0] - w / 2).min(); sy = (coords[:, 1] - h / 2).min()
    cx = coords[:, 0] - sx; cy = coords[:, 1] - sy
    area = (w * h).copy()
    fixed = (is_pp | is_fs); free = ~fixed
    # AREA SLACK (experimental, default off). The scorer allows soft blocks a TWO-SIDED 1%
    # relative area error (check_area_tolerance), but we have always targeted exact area and
    # so have spent none of it. Shrinking every free block by `area_slack` shrinks the packed
    # outline by roughly the same fraction (=> Area_gap) and hands the snap/pull hill-climbs
    # room to accept moves they currently reject. Only free blocks: fixed-shape and preplaced
    # dimensions are HARD and exact. Keep well under 0.01 -- the emitted coords are rounded and
    # the scorer recomputes w*h, so the realized error must stay inside the tolerance.
    if area_slack:
        area[free] *= (1.0 - float(area_slack))
    mib = np.zeros(len(cx)) if mib is None else np.asarray(mib)
    kw = dict(grouping=grouping, boundary=boundary, lam_grp=lam_grp, ov_frac=ov_frac,
              net_edges=net_edges, lam_net=lam_net, sep_rule=sep_rule, grp_dag=grp_dag,
              net_pairs=net_pairs, net_invert=net_invert, lam_bnd_add=lam_bnd_add,
              true_bbox=true_bbox)

    def run(start):
        return _iterate(start[:, 0], start[:, 1], start[:, 2], start[:, 3],
                        area, free, fixed, is_pp, mib, iters, **kw)

    start = np.stack([cx, cy, w, h], 1)
    best, best_obj, solved = run(start)
    if not solved:
        # Reached only when the raw draw's precedence graph admits NO solve, so `best` is the
        # raw OVERLAPPING input and `best_obj` its bbox. A legal packing is 1.02-1.37x that
        # outline (measured over 31 failing draws), so the old `obj2 < best_obj` guard compared
        # a legal layout against an illegal, overlap-compressed one and threw the solve away
        # 6 times out of 6 -- this retry was effectively dead code. Any solve beats no solve.
        pre = compact_positions(np.stack([cx, cy, w, h], 1), is_pp, iters=12)
        best2, obj2, solved2 = run(pre)
        if solved2:
            best = best2; solved = True
    if not solved and reseed_rounds > 0 and is_pp.any():
        cur = start
        for _ in range(int(reseed_rounds)):
            cur = _soft_pin_seed(cur, cx, cy, w, h, area, free, fixed, is_pp, mib,
                                 lam=reseed_lam, **kw)
            if cur is None:
                break
            best3, obj3, solved3 = run(cur)
            if solved3:
                best = best3; solved = True
                break
    _has_grp = grouping is not None and (np.asarray(grouping, int) > 0).any()
    _has_bnd = boundary is not None and (np.asarray(boundary, int) != 0).any()
    # E1b: CONDITIONAL ESCALATION. The extra solve is worth its wall clock only on cases that
    # still carry boundary violations after the ordinary solve; measured, most do not carry
    # enough to repay it. Counted on the solved layout (pre-finish) because that is what is
    # available here, and the finishes only ever reduce the count.
    if solved and grp_hard and grp_hard_min_bnd > 0 and _has_bnd:
        _ll = np.stack([best[:, 0] - best[:, 2] / 2, best[:, 1] - best[:, 3] / 2,
                        best[:, 2], best[:, 3]], 1)
        if _v_boundary(_ll, np.asarray(boundary, int)) < int(grp_hard_min_bnd):
            grp_hard = ()
    if solved and grp_hard and alt is not None and (_has_grp or _has_bnd):
        got = _grp_hard_pass(best, cx, cy, w, h, area, free, fixed, is_pp, mib, grouping,
                             grp_hard_ov, tuple(grp_hard),  boundary=boundary,
                             precheck=grp_hard_precheck,
                             **{k: v for k, v in kw.items()
                                if k not in ('grouping', 'boundary')})
        if got is not None:
            g_best = got[0].copy(); g_best[:, 0] += sx; g_best[:, 1] += sy
            alt.append(g_best)
    best = best.copy(); best[:, 0] += sx; best[:, 1] += sy   # back to input frame
    return best, solved


def legalize_worker(task):
    """`legalize_sample(coords, **kw)` for one `(coords, kw)` pair, in a worker process.

    Lives in the package rather than in `infer.py` so it is picklable by REFERENCE under every
    multiprocessing start method -- a function defined in an entrypoint loaded as anything other
    than `__main__` cannot be sent to a child. Pure numpy + cvxpy: it must never touch CUDA,
    because the pool is forked from a process that later initializes the GPU.
    """
    coords, kw = task
    if not kw.get('return_solved'):
        return legalize_sample(coords, **kw)
    # F1. `state` is mutated in place, which a child process cannot hand back, so unpack it into
    # the return value here -- the ONE place both the serial and the pooled path go through.
    kw = {k: v for k, v in kw.items() if k != 'return_solved'}
    st = {}
    return legalize_sample(coords, state=st, **kw) + (st.get('solved_c'),)


def is_feasible(pos_ll, area_target, is_pp, is_fs, target_ll, tol_area=0.01, tol_dim=1e-4):
    """The contest scorer's three HARD checks, on a lower-left [N,4] solution:
      overlap == 0 (a pair violates iff BOTH axes penetrate > 1e-6; touching is legal),
      soft-block |w*h - area| / area <= 1%,  fixed/preplaced (w,h) [and preplaced (x,y)] exact.
    Needs only OUR positions + the GIVEN inputs (area targets, preplaced/fixed poses) — no
    ground truth — so it is usable at submission time to pick between handoff variants."""
    p = np.asarray(pos_ll, float); n = len(p)
    x, y, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    for i in range(n):
        for j in range(i + 1, n):
            ox = min(x[i] + w[i], x[j] + w[j]) - max(x[i], x[j])
            oy = min(y[i] + h[i], y[j] + h[j]) - max(y[i], y[j])
            if ox > 1e-6 and oy > 1e-6:
                return False
    tgt = None if target_ll is None else np.asarray(target_ll, float)
    for i in range(n):
        fx, pp = bool(is_fs[i]), bool(is_pp[i])
        if fx or pp:
            if tgt is not None:
                if abs(w[i] - tgt[i, 2]) > tol_dim or abs(h[i] - tgt[i, 3]) > tol_dim:
                    return False
                if pp and (abs(x[i] - tgt[i, 0]) > tol_dim or abs(y[i] - tgt[i, 1]) > tol_dim):
                    return False
            continue                                   # exact-dim blocks skip the area check
        a = float(area_target[i])
        if a > 0 and abs(w[i] * h[i] - a) / a > tol_area:
            return False
    return True


# --------------------------------------------------------------------------- #
# B8 -- TWO-LEVEL CLUSTER SOLVE                                                #
# --------------------------------------------------------------------------- #
# Every grouping lever this project has shipped repairs an abutment AFTER the packing exists:
# `snap_groups` slides a component, `lam_grp` prices the gap, B2 stops a foreign block being
# ORDERED into the corridor. All three are corrections to a packing that was built without
# knowing the group was one object. `GROUP_SUPERMODULE_SCREEN.md` tried the obvious alternative
# -- pack the group compactly, then reinsert it rigidly -- and accepted ZERO placements out of
# 15 paired cases, because a solved M9 layout is dense and has no outline-preserving slot. Its
# conclusion named the fix precisely: the supermodule has to be in the global topology from the
# start.
#
# So: solve each owed group as its own small sub-floorplan (the SAME production legalizer, on a
# k-block instance), hand the resulting rectangle to the global solve as ONE fixed-shape node,
# and expand it in place afterwards. Three properties follow that no post-solve move can buy:
#   * the group is ONE component by construction -- its V_grouping is 0 before the global solve
#     starts, not repaired afterwards;
#   * no foreign block can be interposed, because the members are not separate nodes in the
#     global precedence graph at all (B2 removes interpositions one at a time; this makes them
#     unrepresentable); and
#   * the space it costs is its own bbox, and the measured density of a group's CURRENT footprint
#     on the shipped v1.5.13 artifact is a median of 0.502 -- a cluster packed tighter than that
#     costs the global packing nothing.
#
# Two hard-constraint restrictions define eligibility:
#   * no PREPLACED member. A cluster node has one free position; a pinned member does not.
#   * a MIB group straddling the cluster boundary would need its shared (w,h) equality to hold
#     across two separate solves. Handled by FREEZING that shape at the value the ordinary solve
#     already chose (fixed-shape on both sides), which is why this arm runs after the base solve
#     rather than instead of it. Measured on the v1.5.13 artifact, that lifts eligibility from
#     52.1% of groups (43.3% of owed merges) to 80.5% (74.3%).


def _cluster_subsolve(idx, base_c, is_fs, mib, boundary, area_target, prec, sub_kw,
                      min_density=CLUSTER_MIN_DENSITY, log=None, pre=None):
    """Solve ONE grouping group as its own floorplan. Returns (offsets_ll, W, H, code) or None.

    The sub-instance is a genuine contest instance of k blocks with no preplaced, one grouping
    group, and the members' own boundary flags -- so it goes through `legalize_sample` itself
    rather than a private packer. That is deliberate: the cluster's interior must satisfy the
    same exact overlap/area/abutment predicates as anything we emit, and reusing the shipping
    path is the only way to be sure it does. Fixed-shape members and MIB-frozen members enter as
    `is_fs` with their (w,h) in `target_ll`, so `is_feasible` checks them exactly.

    The member's boundary flags are passed through and pulled against the CLUSTER's own bbox: a
    member flush to the cluster's left wall is flush to the global left wall exactly when the
    cluster is, which is what makes the transferred `code` below meaningful.
    """
    k = len(idx)
    sc = np.asarray(base_c, float)[idx].copy()
    frozen = np.asarray(is_fs, bool)[idx] | (np.asarray(mib, int)[idx] > 0)
    tgt = np.full((k, 4), -1.0)
    tgt[frozen, 2] = sc[frozen, 2]
    tgt[frozen, 3] = sc[frozen, 3]
    area = np.asarray(area_target, float)[idx]
    # `pre` is a layout already produced by the BATCHED pack: the conic solve is done, so this
    # call runs only the finishes -- through the ordinary entrypoint, not a copy of it.
    pos, _tag = legalize_sample(sc if pre is None else np.asarray(pre, float),
                                is_pp=np.zeros(k, bool), is_fs=frozen,
                                area_target=area, target_ll=tgt,
                                mib=np.zeros(k, int), grouping=np.ones(k, int),
                                boundary=np.asarray(boundary, int)[idx],
                                prec=prec, presolved=pre is not None, **sub_kw)
    r = np.round(pos, prec)
    # A cluster that is not itself one component buys nothing and costs rigidity; and one that
    # is not feasible cannot be embedded at all.
    if _v_grouping(r, np.ones(k, int)) != 0:
        return None
    if not is_feasible(r, area, np.zeros(k, bool), frozen, tgt):
        return None
    x0, y0 = float(r[:, 0].min()), float(r[:, 1].min())
    W = float((r[:, 0] + r[:, 2]).max()) - x0
    H = float((r[:, 1] + r[:, 3]).max()) - y0
    if W <= 0 or H <= 0:
        return None
    # DENSITY GATE. A cluster is a rigid hole punched in the global packing, so what it really
    # costs is its own whitespace: in a layout at utilization u, area A of blocks already pays
    # A/u of outline, so a cluster of density >= u is FREE and one below u is a net loss of
    # (W*H - A/u). v1.5.13 packs at u = 0.950, which is why this gate is demanding and why the
    # ungated arm bought a 32% V_grouping cut for only -0.69%.
    dens = float((r[:, 2] * r[:, 3]).sum()) / (W * H)
    if log is not None:
        log.append({'k': k, 'W': W, 'H': H, 'density': dens})
    if dens < min_density:
        return None
    off = r.copy(); off[:, 0] -= x0; off[:, 1] -= y0
    # Transfer a member's boundary flag to the cluster ONLY if the member sits on the
    # corresponding cluster edge -- otherwise the cluster being flush says nothing about it.
    # Bits: 1=L, 2=R, 4=T, 8=B (see _v_boundary / bnd_hard).
    eps = 1e-9 * max(W, H)
    code = 0
    bnd = np.asarray(boundary, int)[idx]
    for m in range(k):
        c = int(bnd[m])
        if c & 1 and off[m, 0] <= eps:
            code |= 1
        if c & 2 and off[m, 0] + off[m, 2] >= W - eps:
            code |= 2
        if c & 8 and off[m, 1] <= eps:
            code |= 8
        if c & 4 and off[m, 1] + off[m, 3] >= H - eps:
            code |= 4
    return off, W, H, code


def _cluster_batch_pack(specs, iters, lam_grp, ov_frac, sep_rule, grp_dag):
    """Solve EVERY cluster of one case in a single conic program (B8, `cluster_batch`).

    A 7-block sub-instance measured 17.8 ms against CLARABEL's own sub-millisecond time: the cost
    of a sub-solve is cvxpy CANONICALISATION, which is paid per PROBLEM, not per row. The clusters
    of a case are independent -- no shared block, no shared constraint -- so stacking them into one
    program with a per-cluster outline (`_solve_once(part=...)`) and the SUM of their perimeters as
    the objective is not an approximation: the program separates, and each partition's optimum is
    exactly what it would be alone. It just pays the fixed overhead once instead of once per group.

    `specs` is a list of per-cluster dicts (coords, area, fixed, is_fs, mib, boundary, grouping).
    Returns a list of per-cluster [k,4] (cx,cy,w,h) layouts, or None if the batched solve failed.
    Each partition is kept at ITS OWN best round, mirroring `_iterate`'s per-instance rule: the
    partitions do not interact, so a group that stops improving must not stop the others.
    """
    off, part, cur = [], [], []
    for s in specs:
        off.append(len(part))
        part += [len(off) - 1] * len(s['coords'])
        cur.append(np.asarray(s['coords'], float).copy())
    part = np.asarray(part, int)
    area = np.concatenate([s['area'] for s in specs])
    fixed = np.concatenate([s['fixed'] for s in specs])
    is_pp = np.zeros(len(part), bool)
    mib = np.zeros(len(part), int)
    best = [c.copy() for c in cur]
    best_obj = [np.inf] * len(specs)
    for _round in range(max(1, int(iters))):
        Gh, Gv, grp_edges = [], [], []
        for gi, s in enumerate(specs):
            c = cur[gi]
            gh, gv = _precedence(c[:, 0], c[:, 1], c[:, 2], c[:, 3], sep_rule=sep_rule)
            if grp_dag:
                gh, gv, _ = apply_dag_rules(gh, gv, c[:, 0], c[:, 1], c[:, 2], c[:, 3],
                                            s['grouping'], s['boundary'], np.zeros(len(c), bool),
                                            ('b2',) if grp_dag is True else grp_dag)
            o = off[gi]
            Gh += [(a + o, b + o) for a, b in gh]
            Gv += [(a + o, b + o) for a, b in gv]
            if lam_grp > 0:
                grp_edges += [(a + o, b + o) for a, b in
                              group_tree_edges(c[:, 0], c[:, 1], c[:, 2], c[:, 3], s['grouping'])]
        allc = np.concatenate(cur, 0)
        out = _solve_once(allc[:, 0], allc[:, 1], allc[:, 2], allc[:, 3], area, ~fixed, fixed,
                          is_pp, mib, Gh, Gv, grp_edges=tuple(grp_edges), lam_grp=lam_grp,
                          ov_frac=ov_frac, part=part)
        if out is None:
            break
        cxc, cyc, nw, nh, _obj = out
        for gi in range(len(specs)):
            sl = slice(off[gi], off[gi] + len(cur[gi]))
            lay = np.stack([cxc[sl], cyc[sl], nw[sl], nh[sl]], 1)
            o = _bbox(lay[:, 0], lay[:, 1], lay[:, 2], lay[:, 3])
            if o < best_obj[gi] - 1e-9:
                best_obj[gi] = o; best[gi] = lay
            cur[gi] = lay
    return None if all(np.isinf(o) for o in best_obj) else best


def cluster_solve(base_c, targets, is_pp, is_fs, area_target, mib, grouping, boundary,
                  prec=PREC_DEFAULT, iters=3, sub_kw=None,
                  min_size=CLUSTER_MIN_SIZE, max_size=CLUSTER_MAX_SIZE,
                  min_density=CLUSTER_MIN_DENSITY, log=None, batch=False, sub_iters=2,
                  max_waste=None, **solve_kw):
    """B8. Returns a full [N,4] (cx,cy,w,h) candidate layout, or None if it did not form.

    `base_c` is the ordinary solve's layout: it seeds both levels and it is where the frozen MIB
    shapes come from. `targets` is the list of group ids to cluster (the caller picks the ones
    that are still owed after the ordinary finish).
    """
    base_c = np.asarray(base_c, float)
    N = len(base_c)
    grouping = np.asarray(grouping, int); mib = np.asarray(mib, int)
    is_pp = np.asarray(is_pp, bool); is_fs = np.asarray(is_fs, bool)
    boundary = np.asarray(boundary, int); area_target = np.asarray(area_target, float)
    sub_kw = dict(sub_kw or {})

    elig = [np.flatnonzero(grouping == int(g)) for g in targets]
    elig = [i for i in elig if min_size <= len(i) <= max_size and not is_pp[i].any()]

    pre = [None] * len(elig)
    if batch and len(elig) > 1:
        specs = [dict(coords=base_c[i],
                      area=(base_c[i][:, 2] * base_c[i][:, 3]),
                      fixed=(is_fs[i] | (mib[i] > 0)),
                      grouping=np.ones(len(i), int),
                      boundary=boundary[i]) for i in elig]
        packed = _cluster_batch_pack(specs, sub_iters,
                                     solve_kw.get('lam_grp', 0.0), solve_kw.get('ov_frac', 0.25),
                                     solve_kw.get('sep_rule'), solve_kw.get('grp_dag', False))
        if packed is not None:
            pre = packed

    clusters = []
    for i, pr in zip(elig, pre):
        got = _cluster_subsolve(i, base_c, is_fs, mib, boundary, area_target, prec, sub_kw,
                                min_density=min_density, log=log, pre=pr)
        if got is not None:
            clusters.append((i, *got))
    if not clusters:
        return None

    # WHITESPACE PRE-GATE. The sub-solves cost 34% of this arm and the reduced global solve the
    # other 61%, and the candidate is ACCEPTED on only 45% of the cases it runs on -- so the
    # cheap half can be used to decide whether to buy the expensive half. A cluster is a rigid
    # hole, so what it charges the packing is its own whitespace; if that already exceeds a share
    # of the whitespace the base layout HAS, the reduced solve has to grow the bbox and the priced
    # test will decline it. Measured per-case, before any second solve is spent.
    if max_waste is not None:
        waste = sum(W * H - float((base_c[i][:, 2] * base_c[i][:, 3]).sum())
                    for i, _off, W, H, _c in clusters)
        bb = _bbox(base_c[:, 0], base_c[:, 1], base_c[:, 2], base_c[:, 3])
        slack = max(bb - float((base_c[:, 2] * base_c[:, 3]).sum()), 1e-9)
        if waste > max_waste * slack:
            return None

    member_of = {}
    for ci, (idx, *_rest) in enumerate(clusters):
        for i in idx:
            member_of[int(i)] = ci
    keep = [i for i in range(N) if i not in member_of]
    # A MIB group with a member inside a cluster has its shape decided by the sub-solve, so every
    # OTHER member of that group must be held at the same shape in the reduced solve. Freezing
    # both sides at the base solve's value is what keeps V_mib == 0 across the cluster boundary.
    touched = {int(mib[i]) for i in member_of if mib[i] > 0}

    rc, r_pp, r_fs, r_mib, r_grp, r_bnd = [], [], [], [], [], []
    for i in keep:
        rc.append(base_c[i])
        r_pp.append(bool(is_pp[i]))
        frz = bool(is_fs[i]) or (mib[i] > 0 and int(mib[i]) in touched)
        r_fs.append(frz)
        r_mib.append(0 if (mib[i] == 0 or int(mib[i]) in touched) else int(mib[i]))
        r_grp.append(int(grouping[i]))
        r_bnd.append(int(boundary[i]))
    for idx, off, W, H, code in clusters:
        sub = base_c[idx]
        rc.append([float(sub[:, 0].mean()), float(sub[:, 1].mean()), W, H])
        r_pp.append(False); r_fs.append(True)      # the cluster rectangle is rigid
        r_mib.append(0); r_grp.append(0)           # already one component; shapes already fixed
        r_bnd.append(int(code))

    net_edges = solve_kw.pop('net_edges', ())
    if len(net_edges):
        pos_of = {int(i): j for j, i in enumerate(keep)}
        pos_of.update({int(i): len(keep) + member_of[int(i)] for i in member_of})
        e = np.asarray(net_edges)
        remap = np.array([[pos_of[int(a)], pos_of[int(b)], *rest]
                          for a, b, *rest in e.tolist()], float)
        net_edges = remap[remap[:, 0] != remap[:, 1]] if len(remap) else ()

    out, solved = _compact_core(np.asarray(rc, float), np.asarray(r_pp, bool),
                                np.asarray(r_fs, bool), mib=np.asarray(r_mib, int),
                                boundary=np.asarray(r_bnd, int), iters=iters,
                                grouping=np.asarray(r_grp, int), net_edges=net_edges,
                                **solve_kw)
    if not solved:
        return None

    res = np.empty((N, 4), float)
    for j, i in enumerate(keep):
        res[i] = out[j]
    for c, (idx, off, W, H, _code) in enumerate(clusters):
        row = out[len(keep) + c]
        x0 = row[0] - row[2] / 2.0; y0 = row[1] - row[3] / 2.0
        for m, i in enumerate(idx):
            res[i] = (x0 + off[m, 0] + off[m, 2] / 2.0, y0 + off[m, 1] + off[m, 3] / 2.0,
                      off[m, 2], off[m, 3])
    return res


def owed_groups(pos_ll, grouping):
    """Group ids that are still more than one component in `pos_ll` -- what B8 should target."""
    grouping = np.asarray(grouping, int)
    from .energy import grouping_components
    out = []
    for g in np.unique(grouping[grouping > 0]):
        idx = np.where(grouping == g)[0]
        if len(idx) > 1 and grouping_components(pos_ll, idx) > 1:
            out.append(int(g))
    return out


# --------------------------------------------------------------------------- #
# the single "what we ship" entry point                                        #
# --------------------------------------------------------------------------- #
LEGALIZE_MODES = ('none', 'plain', 'aware', 'hybrid')


def legalize_sample(coords, is_pp, is_fs, area_target, target_ll,
                    mib=None, grouping=None, boundary=None,
                    mode='hybrid', gap_frac=GAP_FRAC, iters=3, prec=PREC_DEFAULT,
                    lam_grp=0.0, snap=False, snap_overlap=SNAP_OVERLAP,
                    snap_cross_frac=SNAP_CROSS_FRAC,
                    snap_cost_aware=True, snap_max_area_growth=SNAP_MAX_AREA_GROWTH,
                    pull=False, pull_reshape=False, net_edges=(), lam_net=0.0,
                    area_slack=0.0, reseed_rounds=SOLVE_RESEED_ROUNDS,
                    reseed_lam=SOLVE_RESEED_LAM, sep_rule=None, snap_mib=False,
                    snap_clear=(), grp_hard=(), grp_hard_ov=GRP_HARD_OV,
                    gap_repair=False, gap_repair_rounds=GAP_REPAIR_ROUNDS,
                    finish_rounds=1, grp_dag=False, grp_hard_min_bnd=0,
                    grp_hard_precheck=False, pull_batch=False, cluster=False,
                    cluster_min_size=CLUSTER_MIN_SIZE, cluster_max_size=CLUSTER_MAX_SIZE,
                    cluster_min_density=CLUSTER_MIN_DENSITY,
                    cluster_min_owed=CLUSTER_MIN_OWED, cluster_iters=None,
                    cluster_sub_light=CLUSTER_SUB_LIGHT, cluster_sub_iters=2,
                    cluster_batch=CLUSTER_BATCH, cluster_max_waste=CLUSTER_MAX_WASTE,
                    cluster_log=None, hpwl_nets=None, sep_net=SEP_NET_Q,
                    priced_hpwl=PRICED_HPWL, presolved=False, state=None,
                    resolve=RESOLVE, lam_bnd=1.0, lam_bnd_add=0.0, true_bbox=False):
    """Raw placement -> submittable lower-left [N,4]=[x,y,w,h]. Returns (pos_ll, tag).

    coords     [N,4] = [cx,cy,w,h], the sampler's output (any frame).
    target_ll  [N,4] = [x,y,w,h] per block, the CONTEST INPUT `target_pos` (-1 = free;
               preplaced carry x,y,w,h; fixed-shape carry w,h). NOT ground truth -- the
               scorer hands this to optimizer.solve(). Needed for two things:
                 * preplaced-frame ALIGNMENT (a rigid translation landing preplaced on
                   their fixed input (x,y)) -- a HARD constraint, applied in EVERY mode,
                   including 'none'. Skipping it is what made v1.4/v1.5 0/100 feasible.
                 * the 'hybrid' feasibility test.

    mode: 'none'   = raw placement, frame-aligned only. NOT overlap-free -- the handoff
                     for a downstream legalizer that would rather start from our raw output.
          'plain'  = compact + gap EVERY free block (max feasibility; breaks abutment so
                     boundary/grouping V_rel rises).
          'aware'  = compact + snap contacts exact, gap only unconstrained blocks (keeps
                     grouping/boundary flushness => low V_rel, but only 91/100 feasible).
          'hybrid' = (default, shipped) aware where it is feasible under the scorer's own
                     hard checks, else plain. 100/100 feasible at the low V_rel.

    `prec` matters: the caller must round to the SAME precision it submits at, because
    'hybrid' tests feasibility on the ROUNDED coords (a pair that exactly touches can round
    into a 1e-9 overlap and the scorer's threshold is 1e-6). See release/v1.5.1/.
    """
    if mode not in LEGALIZE_MODES:
        raise ValueError(f'mode must be one of {LEGALIZE_MODES}, got {mode!r}')
    coords = np.asarray(coords, float)
    n = len(coords)
    is_pp = np.asarray(is_pp, bool); is_fs = np.asarray(is_fs, bool)
    free = ~(is_pp | is_fs)
    mib = np.zeros(n, int) if mib is None else np.asarray(mib, int)
    grouping = np.zeros(n, int) if grouping is None else np.asarray(grouping, int)
    boundary = np.zeros(n, int) if boundary is None else np.asarray(boundary, int)
    tgt = None if target_ll is None else np.asarray(target_ll, float)
    # H3. Heavy netlist pairs get a wirelength-aware precedence tie-break (see `_precedence`).
    # Built once per case here rather than inside `_iterate`, which re-infers the graph every
    # round: the pair set depends only on the netlist, not on the placement.
    # J1/H10: hand the contest netlist to the post-solve acceptance tests, not just to the
    # candidate comparison. `None` keeps the area-only behaviour bit-for-bit. `priced_hpwl` is
    # either a bool (all stages) or a collection naming the stages, so the two hill-climbs can
    # be attributed separately -- they trade in opposite directions and the combined arm hides
    # that.
    _pn = (hpwl_nets if (priced_hpwl and hpwl_nets is not None
                         and hpwl_nets[0] is not None) else None)
    _stages = (('snap', 'pull') if priced_hpwl is True
               else () if not priced_hpwl else tuple(priced_hpwl))
    snap_nets = _pn if 'snap' in _stages else None
    pull_nets = _pn if 'pull' in _stages else None
    sep_net_pairs = None
    if sep_net and hpwl_nets is not None and hpwl_nets[0] is not None:
        sep_net_pairs = net_sep_pairs(hpwl_nets[0], n, 1.0 - abs(float(sep_net)))

    def finish(c):
        c = np.asarray(c, float)
        ll = np.stack([c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2, c[:, 2], c[:, 3]], 1)
        if is_pp.any() and tgt is not None:
            return align_preplaced(ll, tgt, is_pp)   # preplaced (x,y) is a HARD constraint
        return ll                     # no preplaced: KEEP the frame (pins anchor HPWL)

    if mode == 'none':
        return finish(coords), 'raw (no legalization)'

    # MIB exactness (v1.5.12, `snap_mib`) is TWO mechanisms on different paths, because the
    # solve already holds a group to one (w,h) and only later steps break it:
    #   * PREVENT -- make the gap step's shrink uniform per group (`gap_mib`), which is where
    #     every violation on the solved path is manufactured; and
    #   * REPAIR  -- snap_mib_shapes below, the gated backstop for the paths prevention cannot
    #     reach (a failed solve ships the raw sampler's shapes, which were never MIB-equal).
    # Measured paired on results/solve_repair_pool: V_mib 79 control -> 14 prevent-only ->
    # 15 repair-only -> 11 both, where the 11 are exactly the 3 solve-failure cases.
    gap_mib = mib if snap_mib else None

    def soft_finishes(ll, tag):
        """Three feasibility-gated soft-constraint finishes, each a no-op when its flag is
        off: restore each MIB group's shared shape (v1.5.12), then snap groups into contact
        (v1.5.5), then pull boundary-flagged blocks onto their wall (v1.5.6). Each is gated
        on is_feasible and prices the OTHER soft terms, so they compose without fighting.

        The MIB restore runs FIRST because it is the only one that changes a (w,h): the
        grouping and boundary hill-climbs must see the shapes that will actually be emitted.
        Measured on results/solve_repair_pool: running it last instead costs V_grp 593 -> 623
        and V_bnd 319 -> 343, turning a -1.00% end-to-end win into +0.44%.

        `finish_rounds` > 1 (P7) alternates the grouping and boundary hill-climbs to a joint
        fixed point instead of running each once. They interact through the bounding box --
        a snap that grows it moves every wall, and a pull that moves a wall changes which
        snaps are affordable -- so one pass each leaves whichever runs second with the last
        word. Each is individually monotone under the same priced score, so iterating cannot
        make things worse; it just costs passes.
        """
        if snap_mib and (mib > 0).any():
            out, nsn = snap_mib_shapes(ll, is_pp, is_fs, mib, grouping=grouping,
                                       boundary=boundary, prec=prec)
            if nsn and is_feasible(np.round(out, prec), area_target, is_pp, is_fs, tgt):
                ll = out; tag = tag + '+mib'
        for _round in range(max(1, int(finish_rounds))):
            before = ll
            if snap and (grouping > 0).any():
                out, nmv = snap_groups(ll, is_pp, grouping, boundary, prec=prec,
                                       snap_overlap=snap_overlap, cross_frac=snap_cross_frac,
                                       cost_aware=snap_cost_aware,
                                       mib=mib, max_area_growth=snap_max_area_growth,
                                       clear=tuple(snap_clear), is_fs=is_fs,
                                       hpwl_nets=snap_nets)
                if nmv and is_feasible(np.round(out, prec), area_target, is_pp, is_fs, tgt):
                    ll = out
                    if _round == 0:
                        tag = tag + '+snap'
            if pull and (boundary != 0).any():
                out, nmv = pull_boundary(ll, is_pp, is_fs, boundary, grouping=grouping, mib=mib,
                                         prec=prec, reshape=pull_reshape, batch=pull_batch,
                                         hpwl_nets=snap_nets)
                if nmv and is_feasible(np.round(out, prec), area_target, is_pp, is_fs, tgt):
                    ll = out
                    if _round == 0:
                        tag = tag + '+pull'
            if ll is before:
                break                      # fixed point: neither hill-climb moved anything
        return ll, tag

    def gap_finish(c, tag_prefix):
        """The gap step + soft finishes for one solved layout, exactly as shipped."""
        if mode == 'plain':
            return soft_finishes(finish(open_gaps(c, free, gap_frac, mib=gap_mib)),
                                 tag_prefix + 'plain')
        aw = finish(aware_gaps(c, free, is_pp, grouping=grouping, boundary=boundary,
                               gap_frac=gap_frac, mib=gap_mib))
        if mode == 'aware' or is_feasible(np.round(aw, prec), area_target, is_pp, is_fs, tgt):
            return soft_finishes(aw, tag_prefix + 'aware')
        # The aware layout is infeasible -- but measured, that is a median of ONE overlapping
        # pair in ~2775. Gap just those blocks before giving up the whole layout; only if that
        # still fails do we fall back to re-gapping everything (see targeted_gaps).
        if gap_repair:
            tg, n_tg = targeted_gaps(aw, free, gap_frac, mib=gap_mib, prec=prec,
                                     rounds=gap_repair_rounds)
            if n_tg and is_feasible(np.round(tg, prec), area_target, is_pp, is_fs, tgt):
                return soft_finishes(tg, tag_prefix + 'aware+targeted')
        return soft_finishes(finish(open_gaps(c, free, gap_frac, mib=gap_mib)),
                             tag_prefix + 'plain')

    alt = [] if (grp_hard or cluster or resolve) else None
    if presolved:
        # `coords` IS the solved layout: skip the conic solve and run only what comes after it.
        # Two callers: the B8 batched cluster pack, which solves a whole case's clusters in one
        # program and then needs each one finished through the ORDINARY path rather than a copy
        # of it; and F1 (`--finish-winner`), which re-enters here with the SELECTED candidate's
        # cached solve to run the expensive stages once instead of once per candidate.
        #
        # This falls through rather than returning, so the cluster block and the candidate
        # acceptance test below are reachable from a cached solve. It is bit-identical for the
        # B8 caller, whose `sub_kw` sets neither `grp_hard` nor `cluster`: `alt` is then None and
        # the function returns `base_ll` exactly as the old early return did.
        #
        # `grp_hard` is NOT reachable this way -- its drop-and-retry ladder lives inside
        # `compact_solve`, which is precisely the call being skipped. F1 therefore never moves
        # `grp_hard` behind the selection; see `--finish-winner`.
        solved_c, solved = np.asarray(coords, float), True
    else:
        solved_c, solved = compact_solve(coords, is_pp, is_fs, mib=mib, boundary=boundary,
                                         iters=iters, grouping=grouping, lam_grp=lam_grp,
                                         net_edges=net_edges, lam_net=lam_net,
                                         area_slack=area_slack, reseed_rounds=reseed_rounds,
                                         reseed_lam=reseed_lam, sep_rule=sep_rule,
                                         grp_hard=tuple(grp_hard), grp_hard_ov=grp_hard_ov,
                                         alt=alt, grp_dag=grp_dag,
                                         grp_hard_min_bnd=grp_hard_min_bnd,
                                         grp_hard_precheck=grp_hard_precheck,
                                         net_pairs=sep_net_pairs,
                                         net_invert=float(sep_net) < 0,
                                         lam_bnd_add=lam_bnd_add, true_bbox=true_bbox)
    if state is not None:
        # F1. Hand the caller the SOLVED centre coords so the selected candidate can be
        # re-finished without paying for its conic solve a second time. `None` on a failed
        # solve, where there is nothing to resume from and the winner keeps its cheap layout.
        state['solved_c'] = np.array(solved_c, float) if solved else None
    if not solved:
        # The conic solve is infeasible for THIS placement: preplaced are pinned, so a scrambled
        # relative ordering can force a movable block both left-of and right-of an anchor.
        # `solved_c` is then the ORIGINAL placement (monotone: never worse than the input).
        raw = finish(solved_c)
        if is_feasible(np.round(raw, prec), area_target, is_pp, is_fs, tgt):
            return raw, 'raw (solve infeasible, already feasible)'
        # Otherwise the raw placement still overlaps => Cost = M. Try POSITION-ONLY longest-path
        # compaction: it keeps every (w,h) EXACTLY and only slides blocks, which is a weaker (and
        # often satisfiable) ordering problem than the joint shape+position solve. It gives up
        # the aspect-ratio/area win on this sample -- a fair trade against infeasibility.
        # Rescues ~2/3 of failed solves; the caller re-draws if even this fails.
        pos_only = finish(open_gaps(compact_positions(coords, is_pp, iters=12), free, gap_frac,
                                    mib=gap_mib))
        if is_feasible(np.round(pos_only, prec), area_target, is_pp, is_fs, tgt):
            return soft_finishes(pos_only, 'position-only (solve infeasible)')
        return raw, 'fallback (solve infeasible)'
    base_ll, base_tag = gap_finish(solved_c, '')
    n_g3 = 0 if alt is None else len(alt)

    # H5/H6. One extra solve on the SAME topology, with the scorer's b2b wirelength as the
    # objective and the outline capped. It enters as an ordinary candidate, so the priced
    # acceptance test below decides -- which is the guard that matters here, because a
    # wirelength objective under a PERIMETER cap can still buy `hpwl_gap` with `area_gap`, and
    # AGENTS.md prices that trade at ~100x unfavourable at the per-move margin.
    if resolve:
        _nets = hpwl_nets[0] if hpwl_nets is not None else None
        for _spec in resolve:
            _kind, _, _e = str(_spec).partition('@')
            if _kind in ('hpwl', 'both') and _nets is None:
                continue
            _hs = frozen_resolve(np.asarray(solved_c, float), is_pp, is_fs, area_target, mib,
                                 _nets, eps=float(_e or 0.0), kind=_kind, sep_rule=sep_rule,
                                 grp_dag=grp_dag, grouping=grouping, boundary=boundary,
                                 area_slack=area_slack, lam_bnd=lam_bnd)
            if _hs is not None:
                alt.append(np.stack(_hs[:4], 1))
    n_hp = 0 if alt is None else len(alt) - n_g3

    # B8. Targeted at the groups the ordinary finish LEFT owed, on the layout that finish
    # produced, so a group the shipped path already satisfies never pays for the extra solve.
    if cluster and (grouping > 0).any() and not is_pp.all():
        _bll = np.round(base_ll, prec)
        tg = (owed_groups(_bll, grouping)
              if _v_grouping(_bll, grouping) >= int(cluster_min_owed) else [])
        if tg:
            # `cluster_iters` / `cluster_sub_light` are the runtime knobs. The reduced global
            # solve is seeded from an ALREADY-SOLVED layout, unlike the base solve which starts
            # from the raw draw, so it does not need the same outer-loop budget; and the
            # sub-instances are k <= 12 blocks with no preplaced, where the expensive finishes
            # have almost nothing to find.
            sub = dict(mode=mode, gap_frac=gap_frac, iters=iters, lam_grp=lam_grp,
                       snap=snap, snap_overlap=snap_overlap,
                       snap_cross_frac=snap_cross_frac, snap_cost_aware=snap_cost_aware,
                       snap_max_area_growth=snap_max_area_growth, pull=pull,
                       pull_reshape=pull_reshape, pull_batch=pull_batch,
                       gap_repair=gap_repair, gap_repair_rounds=gap_repair_rounds,
                       finish_rounds=finish_rounds, grp_dag=grp_dag,
                       sep_rule=sep_rule, reseed_rounds=0)
            if cluster_sub_light:
                sub.update(iters=int(cluster_sub_iters), finish_rounds=1,
                           gap_repair=False, pull_reshape=False)
            cl = cluster_solve(
                solved_c, tg, is_pp, is_fs, area_target, mib, grouping, boundary,
                prec=prec, iters=iters if cluster_iters is None else int(cluster_iters),
                sub_kw=sub,
                min_size=cluster_min_size, max_size=cluster_max_size,
                min_density=cluster_min_density, log=cluster_log,
                batch=cluster_batch, sub_iters=int(cluster_sub_iters),
                max_waste=cluster_max_waste,
                lam_grp=lam_grp, sep_rule=sep_rule, grp_dag=grp_dag,
                net_edges=net_edges, lam_net=lam_net, area_slack=area_slack,
                reseed_rounds=reseed_rounds, reseed_lam=reseed_lam)
            if cl is not None:
                alt.append(cl)

    if not alt:
        return base_ll, base_tag

    # CANDIDATE ACCEPTANCE. A candidate solve satisfies strictly more constraints than the
    # ordinary one, so its bbox can only be larger -- comparing the two at the SOLVE is therefore
    # rigged, and comparing V_grouping there is meaningless because an exact-0 contact carries a
    # random-signed solver residual (see SNAP_OVERLAP). Both layouts are finished in full and
    # compared on the numbers we would actually submit, under the contest's own trade:
    # `(1 + ALPHA*(area_gap + hpwl_gap)) * exp(BETA*V_rel)`, self-referential against the ordinary
    # solve because both gaps are measured against a GT we do not have. An infeasible candidate is
    # discarded.
    #
    # `hpwl_nets` (B8). The contest weights `hpwl_gap` and `area_gap` EQUALLY -- they are summed
    # inside one ALPHA -- but this test priced only area, because until B8 no candidate arm moved
    # blocks far enough for wirelength to matter. A cluster does: measured on the 44 cases where
    # the ungated cluster candidate was accepted, it raised `hpwl_gap` 0.174 -> 0.271 while
    # raising `area_gap` 0.081 -> 0.172, so the test was understating the cost of accepting it by
    # about half. Pass the contest's own b2b/p2b/pins arrays to price both. Omitting them keeps
    # the area-only behaviour bit-for-bit.
    base_r = np.round(base_ll, prec)
    base_bbox = max(_bbox_area_ll(base_r), 1e-9)
    base_hpwl = None
    if hpwl_nets is not None and hpwl_nets[0] is not None and len(hpwl_nets[0]):
        base_hpwl = max(v10_total_hpwl(base_r, *hpwl_nets), 1e-9)
    else:
        hpwl_nets = None

    def priced(p):
        r = np.round(p, prec)
        v = _v_boundary(r, boundary) + _v_grouping(r, grouping) + _v_mib(r, mib)
        n_soft = max(v10_soft_denominator(boundary, mib, grouping), 1)
        gap = _bbox_area_ll(r) / base_bbox - 1.0
        if base_hpwl is not None:
            gap += v10_total_hpwl(r, *hpwl_nets) / base_hpwl - 1.0
        return (1.0 + ALPHA * gap) * np.exp(BETA * v / n_soft)

    best_ll, best_tag, best_p = base_ll, base_tag, priced(base_ll)
    for k, c in enumerate(alt):
        ll, tag = gap_finish(c, 'grp-hard ' if k < n_g3
                             else 'resolve ' if k < n_g3 + n_hp else 'cluster ')
        if not is_feasible(np.round(ll, prec), area_target, is_pp, is_fs, tgt):
            continue
        p = priced(ll)
        if p < best_p:
            best_ll, best_tag, best_p = ll, tag, p
    return best_ll, best_tag
