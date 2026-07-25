"""
Utility Functions v3 — 新增 soft violation 計算 + optimal/inference 並排對照圖

新增：
  compute_mib_violations, compute_cluster_violations, compute_boundary_violations
  compute_soft_violations  -> 回傳 V_relative 與各分量（照官方 README 公式）
  plot_comparison          -> 左 optimal、右 inference，存成一張圖
保留 v2 全部函式。
"""
import torch
import numpy as np


# ============================================================
# 座標轉換
# ============================================================

def state_to_xywh(state, areas, canvas_w=1.0, canvas_h=1.0):
    x_norm, y_norm, log_r = state[:, 0], state[:, 1], state[:, 2]
    log_r = np.clip(log_r, -3.0, 3.0)
    r = np.exp(log_r)
    w = np.sqrt(areas * r)
    h = np.sqrt(areas / r)
    x = x_norm * canvas_w
    y = y_norm * canvas_h
    return x, y, w, h


# ============================================================
# HPWL
# ============================================================

def compute_hpwl_vectorized(x, y, w, h, W_int):
    x, y, w, h = [np.array(a, dtype=np.float64) for a in [x, y, w, h]]
    W_int = np.array(W_int, dtype=np.float64)
    k = len(x)
    xr = x + w
    yt = y + h
    max_xr = np.maximum(xr[:, None], xr[None, :])
    min_xl = np.minimum(x[:, None], x[None, :])
    max_yt = np.maximum(yt[:, None], yt[None, :])
    min_yb = np.minimum(y[:, None], y[None, :])
    span_x = max_xr - min_xl
    span_y = max_yt - min_yb
    mask_upper = np.triu(np.ones((k, k), dtype=bool), k=1)
    return float(np.sum(W_int * (span_x + span_y) * mask_upper))


def compute_p2b_hpwl(x, y, w, h, p2b_edges, pins_pos):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    w = np.array(w, dtype=np.float64)
    h = np.array(h, dtype=np.float64)
    pins = np.array(pins_pos, dtype=np.float64)
    cx = x + w / 2.0
    cy = y + h / 2.0
    hpwl = 0.0
    for p_idx, b_idx, weight in p2b_edges:
        p_idx, b_idx = int(p_idx), int(b_idx)
        if p_idx < 0 or b_idx < 0 or p_idx >= len(pins) or b_idx >= len(x):
            continue
        dist = abs(cx[b_idx] - pins[p_idx, 0]) + abs(cy[b_idx] - pins[p_idx, 1])
        hpwl += weight * dist
    return hpwl


# ============================================================
# Overlap
# ============================================================

def compute_overlap_matrix(x, y, w, h):
    k = len(x)
    x, y, w, h = [np.array(a, dtype=np.float64) for a in [x, y, w, h]]
    xr = x + w
    yt = y + h
    overlap_w = np.maximum(0, np.minimum(xr[:, None], xr[None, :]) -
                              np.maximum(x[:, None], x[None, :]))
    overlap_h = np.maximum(0, np.minimum(yt[:, None], yt[None, :]) -
                              np.maximum(y[:, None], y[None, :]))
    overlap_area = overlap_w * overlap_h
    np.fill_diagonal(overlap_area, 0)
    return overlap_area


def total_overlap(x, y, w, h):
    return float(np.sum(compute_overlap_matrix(x, y, w, h)) / 2)


def count_overlaps(x, y, w, h, tol=1e-3):
    """
    共邊（touch）和浮點誤差不算重疊。重疊面積 > tol 才算。
    tol=1e-3：對任何實際的 floorplan 都是極小值（block 量級通常數十到數千）。
    """
    return int(np.sum(compute_overlap_matrix(x, y, w, h) > tol) // 2)


# ============================================================
# Soft constraint violations（照官方 README 公式）
# ============================================================

def _blocks_share_edge(x, y, w, h, i, j, tol=1e-3):
    """兩個 block 是否共邊（abut）：在一軸上區間重疊、另一軸上邊界相接。"""
    xi0, xi1, yi0, yi1 = x[i], x[i] + w[i], y[i], y[i] + h[i]
    xj0, xj1, yj0, yj1 = x[j], x[j] + w[j], y[j], y[j] + h[j]
    # 垂直共邊（左右相接）：x 邊界相接 且 y 區間有重疊
    x_touch = (abs(xi1 - xj0) < tol or abs(xj1 - xi0) < tol)
    y_overlap = min(yi1, yj1) - max(yi0, yj0) > tol
    if x_touch and y_overlap:
        return True
    # 水平共邊（上下相接）：y 邊界相接 且 x 區間有重疊
    y_touch = (abs(yi1 - yj0) < tol or abs(yj1 - yi0) < tol)
    x_overlap = min(xi1, xj1) - max(xi0, xj0) > tol
    if y_touch and x_overlap:
        return True
    return False


def _connected_components(members, adj):
    """members: list of block idx; adj: function(i,j)->bool。回傳連通分量數。"""
    if len(members) == 0:
        return 0
    seen = set()
    comp = 0
    for start in members:
        if start in seen:
            continue
        comp += 1
        stack = [start]
        seen.add(start)
        while stack:
            cur = stack.pop()
            for other in members:
                if other not in seen and adj(cur, other):
                    seen.add(other)
                    stack.append(other)
    return comp


def compute_cluster_violations(x, y, w, h, cluster_group):
    """V_grouping = sum_p (cp - 1)。cluster_group: (k,) 組別 ID（0=無）。"""
    x, y, w, h = [np.array(a, dtype=np.float64) for a in [x, y, w, h]]
    cluster_group = np.array(cluster_group)
    v = 0
    for gid in np.unique(cluster_group):
        if gid == 0:
            continue
        members = [i for i in range(len(x)) if cluster_group[i] == gid]
        if len(members) < 2:
            continue
        cp = _connected_components(
            members, lambda a, b: _blocks_share_edge(x, y, w, h, a, b))
        v += (cp - 1)
    return v


def compute_mib_violations(w, h, mib_group, tol=1e-3):
    """V_mib = sum_q (sq - 1)，sq = 組內 distinct (w,h) 數。"""
    w, h = np.array(w, dtype=np.float64), np.array(h, dtype=np.float64)
    mib_group = np.array(mib_group)
    v = 0
    for gid in np.unique(mib_group):
        if gid == 0:
            continue
        members = [i for i in range(len(w)) if mib_group[i] == gid]
        if len(members) < 2:
            continue
        shapes = []
        for i in members:
            key = (round(w[i] / tol), round(h[i] / tol))
            if key not in shapes:
                shapes.append(key)
        v += (len(shapes) - 1)
    return v


def compute_boundary_violations(x, y, w, h, boundary_code, tol=1e-2):
    """
    V_boundary = sum_b 1b（block 沒貼到指定邊/角 = 1）。
    邊界以「整個 floorplan 的 bounding box」為準。
    bit0=left, bit1=right, bit2=top, bit3=bottom。
    座標系原點左下、y 向上：top=y 最大、bottom=y 最小。
    """
    x, y, w, h = [np.array(a, dtype=np.float64) for a in [x, y, w, h]]
    boundary_code = np.array(boundary_code, dtype=np.int64)
    if len(x) == 0:
        return 0
    xmin = x.min()
    xmax = (x + w).max()
    ymin = y.min()
    ymax = (y + h).max()
    rel = tol * max(xmax - xmin, ymax - ymin, 1e-6)
    v = 0
    for i in range(len(x)):
        code = int(boundary_code[i])
        if code == 0:
            continue
        ok = True
        if code & 1:  # left
            ok = ok and (abs(x[i] - xmin) <= rel)
        if code & 2:  # right
            ok = ok and (abs((x[i] + w[i]) - xmax) <= rel)
        if code & 4:  # top
            ok = ok and (abs((y[i] + h[i]) - ymax) <= rel)
        if code & 8:  # bottom
            ok = ok and (abs(y[i] - ymin) <= rel)
        if not ok:
            v += 1
    return v


def compute_soft_violations(x, y, w, h, mib_group, cluster_group, boundary_code):
    """
    回傳 dict: V_relative 與各分量，照官方公式：
      V_rel = (V_boundary + V_grouping + V_mib) / N_soft
      N_soft = |B_boundary| + sum_p(|G_p|-1) + sum_q(|M_q|-1)
    """
    v_clu = compute_cluster_violations(x, y, w, h, cluster_group)
    v_mib = compute_mib_violations(w, h, mib_group)
    v_bnd = compute_boundary_violations(x, y, w, h, boundary_code)

    boundary_code = np.array(boundary_code, dtype=np.int64)
    mib_group = np.array(mib_group)
    cluster_group = np.array(cluster_group)

    n_boundary = int(np.sum(boundary_code > 0))
    n_grouping = 0
    for gid in np.unique(cluster_group):
        if gid == 0:
            continue
        sz = int(np.sum(cluster_group == gid))
        if sz >= 2:
            n_grouping += (sz - 1)
    n_mib = 0
    for gid in np.unique(mib_group):
        if gid == 0:
            continue
        sz = int(np.sum(mib_group == gid))
        if sz >= 2:
            n_mib += (sz - 1)

    n_soft = n_boundary + n_grouping + n_mib
    v_total = v_bnd + v_clu + v_mib
    v_rel = (v_total / n_soft) if n_soft > 0 else 0.0
    return {
        "V_relative": v_rel,
        "V_boundary": v_bnd,
        "V_grouping": v_clu,
        "V_mib": v_mib,
        "N_soft": n_soft,
    }


# ============================================================
# Legalization
# ============================================================

def legalize(x, y, w, h, max_iters=100, step_ratio=0.5,
             canvas_w=None, canvas_h=None, preplaced_indices=None,
             x_min=0.0, y_min=0.0):
    """
    v3.1: 新增 x_min/y_min，讓 clip 範圍變成 [x_min, x_min + canvas_w - w]，
    支援 canvas 不從原點開始的情形（例如對齊 pin bbox）。
    """
    x = x.copy().astype(np.float64)
    y = y.copy().astype(np.float64)
    w = np.array(w, dtype=np.float64)
    h = np.array(h, dtype=np.float64)
    k = len(x)

    frozen = set()
    if preplaced_indices is not None:
        frozen = set(preplaced_indices)

    for iteration in range(max_iters):
        overlap_mat = compute_overlap_matrix(x, y, w, h)
        if np.sum(overlap_mat) < 1e-10:
            break

        pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                if overlap_mat[i, j] > 0:
                    pairs.append((i, j))
        if not pairs:
            break

        dx = np.zeros(k)
        dy = np.zeros(k)

        for i, j in pairs:
            overlap_x = min(x[i] + w[i], x[j] + w[j]) - max(x[i], x[j])
            overlap_y = min(y[i] + h[i], y[j] + h[j]) - max(y[i], y[j])
            center_i_x = x[i] + w[i] / 2
            center_j_x = x[j] + w[j] / 2
            center_i_y = y[i] + h[i] / 2
            center_j_y = y[j] + h[j] / 2
            i_frozen = i in frozen
            j_frozen = j in frozen

            if overlap_x < overlap_y:
                push = overlap_x * step_ratio
                if i_frozen and j_frozen:
                    continue
                elif i_frozen:
                    if center_i_x < center_j_x: dx[j] += push
                    else: dx[j] -= push
                elif j_frozen:
                    if center_i_x < center_j_x: dx[i] -= push
                    else: dx[i] += push
                else:
                    half_push = push / 2
                    if center_i_x < center_j_x:
                        dx[i] -= half_push; dx[j] += half_push
                    else:
                        dx[i] += half_push; dx[j] -= half_push
            else:
                push = overlap_y * step_ratio
                if i_frozen and j_frozen:
                    continue
                elif i_frozen:
                    if center_i_y < center_j_y: dy[j] += push
                    else: dy[j] -= push
                elif j_frozen:
                    if center_i_y < center_j_y: dy[i] -= push
                    else: dy[i] += push
                else:
                    half_push = push / 2
                    if center_i_y < center_j_y:
                        dy[i] -= half_push; dy[j] += half_push
                    else:
                        dy[i] += half_push; dy[j] -= half_push

        for idx in frozen:
            dx[idx] = 0; dy[idx] = 0
        x += dx; y += dy

        if canvas_w is not None:
            for idx in range(k):
                if idx not in frozen:
                    x[idx] = np.clip(x[idx], x_min, x_min + canvas_w - w[idx])
        if canvas_h is not None:
            for idx in range(k):
                if idx not in frozen:
                    y[idx] = np.clip(y[idx], y_min, y_min + canvas_h - h[idx])

    return x.astype(np.float32), y.astype(np.float32)


def _resolve_overlaps_iterative(x, y, w, h, frozen, tol=1e-6, margin=1e-4, max_iters=5000):
    """
    純平移、worst-pair-first、向量化收斂迴圈（不含 eject fallback，見
    hard_zero_overlap 的說明）。x, y 會被原地修改並回傳。

    frozen: set of int，這些 index 的位置永遠不會被移動。

    收斂條件跟官方 check_overlap 的逐 pair、逐軸判定完全一致：
        violation ⟺ overlap_x > tol AND overlap_y > tol
    每次迭代只處理「當前面積最大的違規 pair」、立刻套用位移再重新掃描——
    比起把所有違規 pair 同時批次套用位移，這樣在密集重疊時才不會來回震盪、
    能保證單調收斂到 0（除非兩個違規 block 都在 frozen 裡，那種情況本質上
    無法透過移動位置解決，會提前跳出）。
    """
    k = len(x)
    if k < 2:
        return x, y
    iu = np.triu_indices(k, k=1)   # 上三角 pair 索引，(i,j) i<j，重用避免每次重建

    def _worst_violation():
        xr = x + w; yt = y + h
        ox_mat = np.minimum(xr[:, None], xr[None, :]) - np.maximum(x[:, None], x[None, :])
        oy_mat = np.minimum(yt[:, None], yt[None, :]) - np.maximum(y[:, None], y[None, :])
        ox_u, oy_u = ox_mat[iu], oy_mat[iu]
        viol = (ox_u > tol) & (oy_u > tol)
        if not viol.any():
            return None
        area_u = np.where(viol, ox_u * oy_u, -np.inf)
        t = int(np.argmax(area_u))
        i, j = int(iu[0][t]), int(iu[1][t])
        return i, j, float(ox_u[t]), float(oy_u[t])

    for _ in range(max_iters):
        worst = _worst_violation()
        if worst is None:
            break

        i, j, ox, oy = worst
        i_frozen = i in frozen
        j_frozen = j in frozen
        if i_frozen and j_frozen:
            break   # 兩個都鎖死，無法透過移動解決，避免無窮迴圈

        ci_x, cj_x = x[i] + w[i] / 2, x[j] + w[j] / 2
        ci_y, cj_y = y[i] + h[i] / 2, y[j] + h[j] / 2
        if ox < oy:
            push = ox + margin
            if i_frozen:
                x[j] += push if cj_x >= ci_x else -push
            elif j_frozen:
                x[i] += push if ci_x >= cj_x else -push
            else:
                half = push / 2
                if ci_x <= cj_x:
                    x[i] -= half; x[j] += half
                else:
                    x[i] += half; x[j] -= half
        else:
            push = oy + margin
            if i_frozen:
                y[j] += push if cj_y >= ci_y else -push
            elif j_frozen:
                y[i] += push if ci_y >= cj_y else -push
            else:
                half = push / 2
                if ci_y <= cj_y:
                    y[i] -= half; y[j] += half
                else:
                    y[i] += half; y[j] -= half

    return x, y


def _would_overlap_any(nx, ny, nw, nh, exclude_i, x, y, w, h, tol=1e-9):
    """向量化檢查候選矩形 (nx,ny,nw,nh) 是否跟除了 exclude_i 以外的任何 block 重疊。"""
    ox = np.minimum(nx + nw, x + w) - np.maximum(nx, x)
    oy = np.minimum(ny + nh, y + h) - np.maximum(ny, y)
    viol = (ox > tol) & (oy > tol)
    viol[exclude_i] = False
    return bool(viol.any())


def compact_gravity(x, y, w, h, preplaced_mask=None, boundary_code=None,
                    iters=150, shrink_ratio=0.15, line_search_steps=8,
                    min_step_ratio=1e-4):
    """
    把 anchor-guided 搜尋已經合法的 layout 進一步壓緊、減少不必要的空隙。

    動機：legalize_v2 的候選評分主要看「離 diffusion anchor 多近」，沒有主動
    把 block 往其他已放置的 block 拉近的力；後面的 compact_positions 也只做
    軸對齊的「往左下推到貼齊」，如果附近沒有東西可貼，空隙就留在那裡出不去。

    做法：一次只動一個 block，嘗試把它的中心往（面積加權）全域重心拉近一步；
    如果這一步會造成新的重疊，就用二分法縮小步長重試，最小步長都還是會撞人
    就這輪跳過這個 block（維持原位，下一輪重心更新後可能又有機會）。所有
    block 依序處理、重複很多輪，直到一整輪下來沒有任何 block 移動（收斂）。

    這個「移動前先檢查安全、不安全就退讓」的設計保證每一步都不會讓任何一對
    block 從不重疊變成重疊，也保證 bbox 只會縮小或持平，不會像「整批同時
    收縮、事後再一起解決衝突」那樣，在密集佈局下因為連鎖推擠而不穩定、
    實測甚至可能讓 bbox 不減反增（面積暴增好幾倍）。

    preplaced_mask: 完全不動（當固定的障礙物）。
    boundary_code:  有 LEFT(1)/RIGHT(2) 鎖定的 block 不參與 x 方向收縮、
                    有 TOP(4)/BOTTOM(8) 鎖定的不參與 y 方向收縮，避免破壞
                    已經達成的 boundary soft constraint。

    只動 x, y（不碰 w, h），面積 hard constraint 不受影響；因為每一步都先
    驗證過安全才套用，回傳結果本來就是零重疊（不需要呼叫端再清一次），但
    為了跟 pipeline 其他階段的保證方式一致，呼叫端仍可以放心再跑一次
    hard_zero_overlap 當雙重保險。
    """
    x = np.array(x, dtype=np.float64).copy()
    y = np.array(y, dtype=np.float64).copy()
    w = np.asarray(w, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    k = len(x)
    if k < 2:
        return x, y

    if preplaced_mask is None:
        preplaced_mask = np.zeros(k, dtype=bool)
    else:
        preplaced_mask = np.asarray(preplaced_mask, dtype=bool)
    if boundary_code is None:
        boundary_code = np.zeros(k, dtype=np.int64)
    else:
        boundary_code = np.asarray(boundary_code, dtype=np.int64)

    lock_x = ((boundary_code & 1) > 0) | ((boundary_code & 2) > 0)
    lock_y = ((boundary_code & 4) > 0) | ((boundary_code & 8) > 0)
    movable_x = ~preplaced_mask & ~lock_x
    movable_y = ~preplaced_mask & ~lock_y
    movable_idx = [i for i in range(k) if not preplaced_mask[i]]

    for _ in range(iters):
        moved_any = False
        for i in movable_idx:
            area_w = w * h
            total_area = float(area_w.sum())
            if total_area <= 0:
                break
            cx_all = x + w / 2.0
            cy_all = y + h / 2.0
            gx = float((cx_all * area_w).sum() / total_area)
            gy = float((cy_all * area_w).sum() / total_area)

            ci_x, ci_y = cx_all[i], cy_all[i]
            target_cx = ci_x + (gx - ci_x) * shrink_ratio if movable_x[i] else ci_x
            target_cy = ci_y + (gy - ci_y) * shrink_ratio if movable_y[i] else ci_y
            if target_cx == ci_x and target_cy == ci_y:
                continue

            step = 1.0   # 沿著 (ci -> target) 這段路的比例，1.0 = 走完整步
            while step >= min_step_ratio:
                nx = (ci_x + (target_cx - ci_x) * step) - w[i] / 2.0
                ny = (ci_y + (target_cy - ci_y) * step) - h[i] / 2.0
                if not _would_overlap_any(nx, ny, w[i], h[i], i, x, y, w, h):
                    x[i], y[i] = nx, ny
                    moved_any = True
                    break
                step *= 0.5

        if not moved_any:
            break

    return x, y


def compact_reinsert(x, y, w, h, preplaced_mask=None, boundary_code=None,
                     sweeps=6, grid_density=20, gap_weight=0.6):
    """
    Remove-and-reinsert 局部搜尋：依序把每個 block 暫時「移除」，在目前（其餘
    block）佔用範圍內搜尋一個位置，讓「重新插入後的總 bbox 面積」最小，比
    compact_gravity 的小步微調更有力——可以直接把 block 搬到 layout 另一側
    真正空著的縫隙，不受限於「一步一步安全移動」的小碎步限制。

    候選 cost = 重新插入後的總 bbox 面積 + gap_weight * avg_side * 離最近其他
    block 的距離。單純只看 bbox 面積有個盲點：如果 layout 裡有一大片「內部
    空洞」（例如主要群聚跟幾個貼邊的 block 中間空一塊），把某個 block 搬進
    那個洞裡通常「不會讓 bbox 變大」——bbox 早就被貼邊的 block 撐到那麼大了
    ——但也「不會讓 bbox 變小」，純看 bbox 面積時這一步的 cost 完全打平、
    等於沒有誘因把 block 往洞裡塞，內部空洞就一直留在那裡出不去。加上「離
    最近鄰居的距離」這個次要項後，bbox 打平的候選之間會依緊密度分高下，
    直接把 block 往內部空洞拉，才能真正填滿那種視覺上一大塊留白的情況。

    跟目前實際位置的 cost（=cur_cost，同樣公式）比較，只有嚴格更小才會採用
    ——這代表每一步都保證不會讓「bbox 面積 + 緊密度」這個組合指標變差，整個
    函式本身就是單調不變差的，不需要額外的「記住最佳快照」保護。

    preplaced_mask: 完全不動。
    boundary_code:  有 LEFT/RIGHT 鎖定的 block 只搜尋 y（x 固定在原值）、
                    有 TOP/BOTTOM 鎖定的只搜尋 x，避免破壞 boundary soft
                    constraint。

    grid_density：搜尋網格在目前 bbox 的每個維度上大約切幾格；只動 x, y。
    """
    x = np.array(x, dtype=np.float64).copy()
    y = np.array(y, dtype=np.float64).copy()
    w = np.asarray(w, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    k = len(x)
    if k < 2:
        return x, y

    if preplaced_mask is None:
        preplaced_mask = np.zeros(k, dtype=bool)
    else:
        preplaced_mask = np.asarray(preplaced_mask, dtype=bool)
    if boundary_code is None:
        boundary_code = np.zeros(k, dtype=np.int64)
    else:
        boundary_code = np.asarray(boundary_code, dtype=np.int64)
    lock_x = ((boundary_code & 1) > 0) | ((boundary_code & 2) > 0)
    lock_y = ((boundary_code & 4) > 0) | ((boundary_code & 8) > 0)
    movable_idx = [i for i in range(k) if not preplaced_mask[i]]
    others_mask_base = np.ones(k, dtype=bool)
    n_grid = max(int(grid_density), 3)
    avg_side = float(np.mean(np.sqrt(w * h))) if k else 1.0

    for _sweep in range(sweeps):
        moved_any = False
        for i in movable_idx:
            others = others_mask_base.copy()
            others[i] = False
            if not others.any():
                continue
            xo, yo, wo, ho = x[others], y[others], w[others], h[others]
            ox_min = float(xo.min()); ox_max = float((xo + wo).max())
            oy_min = float(yo.min()); oy_max = float((yo + ho).max())

            def _nearest_gap(xc, yc):
                gx = np.maximum(0.0, np.maximum(xo - (xc + w[i]), xc - (xo + wo)))
                gy = np.maximum(0.0, np.maximum(yo - (yc + h[i]), yc - (yo + ho)))
                return float(np.sqrt(gx * gx + gy * gy).min())

            cur_x, cur_y = float(x[i]), float(y[i])
            cur_cost = (((max(ox_max, cur_x + w[i]) - min(ox_min, cur_x)) *
                         (max(oy_max, cur_y + h[i]) - min(oy_min, cur_y)))
                        + gap_weight * avg_side * _nearest_gap(cur_x, cur_y))

            if lock_x[i]:
                x_candidates = np.array([cur_x])
            else:
                x_candidates = np.linspace(ox_min - w[i], ox_max, n_grid)
            if lock_y[i]:
                y_candidates = np.array([cur_y])
            else:
                y_candidates = np.linspace(oy_min - h[i], oy_max, n_grid)

            best_cost = cur_cost
            best_pos = (cur_x, cur_y)
            for xc in x_candidates:
                xc = float(xc)
                ox_arr = np.minimum(xc + w[i], xo + wo) - np.maximum(xc, xo)
                x_touches = ox_arr > 1e-9
                for yc in y_candidates:
                    yc = float(yc)
                    if xc == cur_x and yc == cur_y:
                        continue
                    oy_arr = np.minimum(yc + h[i], yo + ho) - np.maximum(yc, yo)
                    if np.any(x_touches & (oy_arr > 1e-9)):
                        continue   # 跟某個 block 重疊，跳過
                    bbox_cost = ((max(ox_max, xc + w[i]) - min(ox_min, xc)) *
                                 (max(oy_max, yc + h[i]) - min(oy_min, yc)))
                    cost = bbox_cost + gap_weight * avg_side * _nearest_gap(xc, yc)
                    if cost < best_cost - 1e-9:
                        best_cost = cost
                        best_pos = (xc, yc)

            if best_pos != (cur_x, cur_y):
                x[i], y[i] = best_pos
                moved_any = True

        if not moved_any:
            break

    return x, y


def compact_reinsert_reshape(x, y, w, h, preplaced_mask=None, fixed_mask=None,
                             mib_group=None, boundary_code=None,
                             sweeps=3, grid_density=12, gap_weight=0.6,
                             n_shape_variants=5, max_aspect_ratio=8.0,
                             verbose=False):
    """
    v5.1（實驗用）：`compact_reinsert` 的 remove-and-reinsert 局部搜尋只動
    (x, y)，形狀在 `legalize_lff` 最初的 LFF 貪婪排布階段選定後就凍結——
    但那個當下的「最佳長寬比」是在其他 block 還沒放完、佈局還沒定型時選的，
    等整體佈局收斂後，換一種長寬比可能更貼合當下實際留下的空間形狀。這裡
    把 `_aspect_variants`（跟 `legalize_v2` 用的同一套：面積不變、在 log
    空間取樣幾種長寬比）接進 `compact_reinsert` 的搜尋迴圈，拔出來重插時
    候選不再只是「不同位置、同一個形狀」，而是「不同位置 × 幾種長寬比」
    的組合，一起比較 bbox + 緊密度成本，嚴格更小才採用。

    可以重新選長寬比的 block 限定為：非 preplaced、非 fixed-shape、沒有
    MIB 約束（`mib_group == 0`；同一個 MIB group 的 block 必須共用相同
    形狀，牽一髮動全身，這版先不處理）、沒有 boundary 約束
    （`boundary_code == 0`；boundary 鎖定的是「該邊界座標」不是「該座標」，
    改形狀後要連帶調整鎖定軸的位置才不會鬆脫，這版先跳過、留給只搬不改形
    的其他 pass 處理）。不滿足這些條件的 block 仍然可以被搬動位置（跟
    `compact_reinsert` 一樣），只是形狀固定。

    面積透過 `_aspect_variants` 的構造方式精確守恆（`w = sqrt(area*r)`、
    `h = sqrt(area/r)`），不會違反面積 hard constraint。

    刻意放在 pipeline 最尾端呼叫（見 `legalize_lff` 呼叫處說明），理由跟
    `compact_pair_reinsert` 相同：避免被後續其他貪婪 pass 撤銷。
    """
    x = np.array(x, dtype=np.float64).copy()
    y = np.array(y, dtype=np.float64).copy()
    w = np.array(w, dtype=np.float64).copy()
    h = np.array(h, dtype=np.float64).copy()
    k = len(x)
    if k < 2:
        return x, y, w, h
    if preplaced_mask is None:
        preplaced_mask = np.zeros(k, dtype=bool)
    else:
        preplaced_mask = np.asarray(preplaced_mask, dtype=bool)
    if fixed_mask is None:
        fixed_mask = np.zeros(k, dtype=bool)
    else:
        fixed_mask = np.asarray(fixed_mask, dtype=bool)
    if mib_group is None:
        mib_group = np.zeros(k, dtype=np.int64)
    else:
        mib_group = np.asarray(mib_group, dtype=np.int64)
    if boundary_code is None:
        boundary_code = np.zeros(k, dtype=np.int64)
    else:
        boundary_code = np.asarray(boundary_code, dtype=np.int64)

    reshapable = (~preplaced_mask) & (~fixed_mask) & (mib_group == 0) & (boundary_code == 0)
    movable_idx = [i for i in range(k) if not preplaced_mask[i]]
    n_grid = max(int(grid_density), 3)
    avg_side = float(np.mean(np.sqrt(w * h))) if k else 1.0

    for _sweep in range(sweeps):
        moved_any = False
        for i in movable_idx:
            others = np.ones(k, dtype=bool)
            others[i] = False
            if not others.any():
                continue
            xo, yo, wo, ho = x[others], y[others], w[others], h[others]
            ox_min = float(xo.min()); ox_max = float((xo + wo).max())
            oy_min = float(yo.min()); oy_max = float((yo + ho).max())

            def _nearest_gap(xc, yc, wi, hi):
                gx = np.maximum(0.0, np.maximum(xo - (xc + wi), xc - (xo + wo)))
                gy = np.maximum(0.0, np.maximum(yo - (yc + hi), yc - (yo + ho)))
                return float(np.sqrt(gx * gx + gy * gy).min())

            cur_x, cur_y, cur_w, cur_h = float(x[i]), float(y[i]), float(w[i]), float(h[i])
            cur_cost = (((max(ox_max, cur_x + cur_w) - min(ox_min, cur_x)) *
                         (max(oy_max, cur_y + cur_h) - min(oy_min, cur_y)))
                        + gap_weight * avg_side * _nearest_gap(cur_x, cur_y, cur_w, cur_h))

            if reshapable[i]:
                shape_candidates = _aspect_variants(cur_w, cur_h, n=n_shape_variants,
                                                    max_ratio=max_aspect_ratio)
            else:
                shape_candidates = [(cur_w, cur_h)]

            lock_x = bool(int(boundary_code[i]) & 3)
            lock_y = bool(int(boundary_code[i]) & 12)

            best_cost = cur_cost
            best_pick = (cur_x, cur_y, cur_w, cur_h)
            for wi, hi in shape_candidates:
                x_candidates = [cur_x] if lock_x else np.linspace(ox_min - wi, ox_max, n_grid)
                y_candidates = [cur_y] if lock_y else np.linspace(oy_min - hi, oy_max, n_grid)
                for xc in x_candidates:
                    xc = float(xc)
                    ox_arr = np.minimum(xc + wi, xo + wo) - np.maximum(xc, xo)
                    x_touches = ox_arr > 1e-9
                    for yc in y_candidates:
                        yc = float(yc)
                        if xc == cur_x and yc == cur_y and wi == cur_w and hi == cur_h:
                            continue
                        oy_arr = np.minimum(yc + hi, yo + ho) - np.maximum(yc, yo)
                        if np.any(x_touches & (oy_arr > 1e-9)):
                            continue   # 跟某個 block 重疊，跳過
                        bbox_cost = ((max(ox_max, xc + wi) - min(ox_min, xc)) *
                                     (max(oy_max, yc + hi) - min(oy_min, yc)))
                        cost = bbox_cost + gap_weight * avg_side * _nearest_gap(xc, yc, wi, hi)
                        if cost < best_cost - 1e-9:
                            best_cost = cost
                            best_pick = (xc, yc, wi, hi)

            if best_pick != (cur_x, cur_y, cur_w, cur_h):
                x[i], y[i], w[i], h[i] = best_pick
                moved_any = True
                if verbose and (w[i] != cur_w or h[i] != cur_h):
                    print("compact_reinsert_reshape: sweep={} block={} reshaped "
                          "({:.2f},{:.2f})->({:.2f},{:.2f})".format(
                              _sweep, i, cur_w, cur_h, w[i], h[i]))

        if not moved_any:
            break

    return x, y, w, h


def compact_gradient_finetune(x, y, w, h, preplaced_mask=None, boundary_code=None,
                              cluster_group=None, W_int=None, outline_bbox=None,
                              n_steps=400, lr=0.5, patience=30, min_delta=1e-3,
                              weight_overlap=50.0, weight_area=1.0,
                              weight_boundary=2.0, weight_cluster=1.0, weight_hpwl=0.3,
                              weight_anchor=0.1, weight_containment=20.0,
                              hpwl_slack_ratio=0.0, boundary_violation_slack=0,
                              verbose=False):
    """
    v5.2（實驗用）：跟前面所有 `compact_*` pass 不同範式的一個嘗試——之前
    每個 pass 都是「一次挪一、兩個 block」的離散區域搜尋，`compact_pair_
    reinsert`／`compact_reinsert_reshape` 兩次證明這條路在真實資料上已經
    觸頂（`compact_reinsert` 的位置搜尋早就把殘留縫隙搜刮乾淨，尾端再加新
    pass 找不到東西）。這裡改用 DREAMPlace／ePlace／RePlAce 這系列類比
    (analytical) global placement 的核心概念：把「所有」非 preplaced
    block 的 (x, y) 一次性當成連續可微分變數，用 PyTorch autograd + Adam
    對一個平滑 loss 做梯度下降——不是一次一兩個 block 的離散移動，而是全部
    block 同時、連續地互相讓位，理論上能找到離散單/雙 block 搜尋永遠碰
    不到的整體更緊密排布。

    Loss = weight_overlap·重疊懲罰（可微分：pairwise 的 clamp(min,max) 面積）
         + weight_area·bbox 面積（可微分：直接用 x.max()/x.min() 之類，
           梯度會流到當下取到極值的那個 block）
         + weight_boundary·邊界懲罰（每個 boundary block 到目前 bbox 對應
           邊的平方距離，概念上跟 compute_boundary_violations 一致，但用
           連續距離取代離散判定式）
         + weight_cluster·分組懲罰（每個 cluster group 內成員中心到組重心
           的平方距離）
         + weight_hpwl·HPWL（可微分：跟 compute_hpwl_vectorized 同一套
           span-based 公式）
         + weight_anchor·錨定懲罰（movable block 平均座標到「優化開始前」
           平均座標的平方距離；理由見下方「已知問題」段落）
         + weight_containment·outline 圍堵懲罰（若提供 outline_bbox：每個
           block 超出 outline 邊界的距離平方，只在真的超出時才非零）

    已知問題與修法（v5.2 踩過、v5.5 補強）：overlap／area／boundary（相對
    自己當下 bbox 邊）／cluster（相對組內重心）／HPWL 這五項全部是**平移
    不變**的（把所有座標同時加一個常數 c，loss 完全不變）——這代表整體
    平移方向在梯度上是一片平坦（真正的梯度和沿這個方向剛好是 0），理論上
    不該漂移。但 Adam 對每個參數獨立做二階動量正規化，不保留「原始梯度和
    為 0」這個性質，實測幾百步下來會在這個方向隨機漂移、把整層 block 帶出
    outline 之外（投影階段的 hard_zero_overlap／compact_positions 都只
    保證彼此不重疊，不保證回到原本的 outline 座標系）。v5.2 只加了
    weight_anchor（把 movable block 的平均座標拉回優化開始前的位置）堵住
    這個平移方向，但這只是「不知道 outline 在哪裡、純粹不讓它漂移」的
    土法煉鋼，跟真正的 hard constraint（outline_bbox）本身沒有直接關係。
    v5.5 補上 weight_containment：對每個 block 直接算「超出 outline 邊界
    的距離」平方懲罰（只要沒超出就是 0，不影響 outline 內部的正常優化），
    是這個約束真正對應的可微分寫法——讓優化器一邊找更緊密的排布、一邊自己
    知道 outline 在哪裡，而不是先自由漂移再事後被 gate 擋下來。
    weight_anchor 保留（權重更小）當這個方向上的第二層保險，避免 outline
    有餘裕（block 離邊界還有一段距離）時 containment loss 是 0、又退化回
    v5.2 的無梯度漂移。

    梯度下降過程中**不保證**任何 hard/soft constraint（中途可能暫時比
    legalize 剛出來時還亂）——這是刻意的，把它當一個「提案產生器」：跑完
    之後一定會先過 `hard_zero_overlap` 把重疊修正投影回合法解，再用
    `compact_positions` 把投影後可能留下的縫隙壓一輪，最後才跟套用前的
    狀態比較，只有**同時滿足**以下全部條件才採用（否則整個操作等於沒發生，
    退回原本已經保證合法的解）：
      1. bbox 面積嚴格變小；
      2. boundary 違規數不超過 `baseline + boundary_violation_slack`（v5.6，
         見下方「boundary gate 加 slack」段落，預設 slack=0 等同原本的
         「不變差」）；
      3. cluster 違規數不變差；
      4. 總 HPWL 增加量不超過 `hpwl_slack_ratio * avg_side`（若提供
         W_int；閘門邏輯跟其他 v4.7+ 機制一致）；
      5. 若提供 `outline_bbox`：投影後所有 block 仍完整落在其內（見上方
         「已知問題」段落——weight_anchor／weight_containment 只是把違規
         機率壓低，不是硬保證，這一條才是真正的硬 gate，浮點數上不論
         違規多寡都會被擋下來）。

    boundary gate 加 slack（v5.6，實驗用）：v5.5 修好平移漂移／outline
    containment 之後，30 樣本真實資料重測仍是 0/30 有改善，追一個具體
    案例（idx=16，原本用來當展示案例的 -4.54%）發現真正卡住它的不是
    outline，而是這一條「boundary 違規數不變差」——這是離散 all-or-
    nothing 判定（`compute_boundary_violations` 算的是「有沒有真的貼到
    邊」，不是連續距離），跟 loss 裡用連續距離當代理的 `boundary_loss`
    對不齊：bbox 縮小的過程中，被縮小的那條邊本身在移動，貼邊的判定
    可能因此翻面，即使 area 改善很大，只要有 1 個 block 從「貼邊」變成
    「差一點」，離散 gate 就會整個否決掉。跟 `hpwl_slack_ratio` 一樣的
    精神，開放 `boundary_violation_slack`（違規數整數容忍度）讓這種
    「用一點點 boundary 違規换 area 改善」的交易有機會通過——**預設值
    仍是 0（跟 v5.5 之前完全一樣的嚴格行為）**，只有明確傳入正整數才會
    放寬，需要在真實資料上驗證淨效益（V_relative 變差是否被 area_gap
    改善划算）才能決定要不要真的當預設值用。

    只優化 (x, y)，不碰形狀（MIB group 的形狀一致性因此不受影響，不需要
    額外處理）；preplaced block 的座標維持不動（不當可微分變數）。

    `n_steps` 是上限，不是每次都跑滿：real data 上多數樣本在幾十步內就已
    經收斂（loss 不再下降）或根本沒有改善空間，跑滿 400 步是純浪費時間
    （100 樣本量測過 legalize 時間可以到 +1.5~3.6 秒／樣本，換算 contest
    cost 公式的 runtime 懲罰後很可能不划算）。用 `patience`：連續
    `patience` 步 loss 都沒有下降超過 `min_delta`，就提前停止。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    k = len(x)
    if k < 2:
        return x.copy(), y.copy()
    if preplaced_mask is None:
        preplaced_mask = np.zeros(k, dtype=bool)
    else:
        preplaced_mask = np.asarray(preplaced_mask, dtype=bool)
    if boundary_code is None:
        boundary_code = np.zeros(k, dtype=np.int64)
    else:
        boundary_code = np.asarray(boundary_code, dtype=np.int64)
    has_cluster = cluster_group is not None
    if has_cluster:
        cluster_group = np.asarray(cluster_group, dtype=np.int64)
    else:
        cluster_group = np.zeros(k, dtype=np.int64)
    check_hpwl = W_int is not None
    avg_side = float(np.mean(np.sqrt(w * h))) if k else 1.0

    baseline_bbox = (float((x + w).max()) - float(x.min())) * (float((y + h).max()) - float(y.min()))
    baseline_v = compute_boundary_violations(x, y, w, h, boundary_code)
    baseline_cluster_v = compute_cluster_violations(x, y, w, h, cluster_group) if has_cluster else 0
    baseline_hpwl = compute_hpwl_vectorized(x, y, w, h, W_int) if check_hpwl else 0.0
    hpwl_slack = max(0.0, hpwl_slack_ratio) * avg_side

    movable = ~preplaced_mask
    m_idx = np.nonzero(movable)[0]
    if len(m_idx) == 0:
        return x.copy(), y.copy()
    orig_mean_x = float(x[m_idx].mean())
    orig_mean_y = float(y[m_idx].mean())

    w_t = torch.tensor(w, dtype=torch.float64)
    h_t = torch.tensor(h, dtype=torch.float64)
    x_full = torch.tensor(x, dtype=torch.float64)
    y_full = torch.tensor(y, dtype=torch.float64)
    x_param = torch.nn.Parameter(x_full[m_idx].clone())
    y_param = torch.nn.Parameter(y_full[m_idx].clone())

    left_mask = torch.tensor((boundary_code & 1) > 0)
    right_mask = torch.tensor((boundary_code & 2) > 0)
    top_mask = torch.tensor((boundary_code & 4) > 0)
    bottom_mask = torch.tensor((boundary_code & 8) > 0)

    has_outline = outline_bbox is not None
    if has_outline:
        c_oxmin, c_oymin, c_oxmax, c_oymax = (float(v) for v in outline_bbox)

    cluster_ids = sorted(set(int(g) for g in cluster_group if g > 0)) if has_cluster else []
    cluster_masks = [torch.tensor(cluster_group == gid) for gid in cluster_ids]

    # HPWL 用稀疏邊表（只列出 W_int 非零的 pair），而不是每步都算一次完整
    # k×k 矩陣——真實資料的 B2B 連線通常很稀疏（一個樣本常常只有幾十條邊，
    # 相對 k^2/2 可能的 pair 數少很多），稀疏化直接把這項的每步成本從
    # O(k^2) 降到 O(edges)，是目前 profiling 下來單步最貴的項目之一。
    if check_hpwl:
        ei, ej = np.nonzero(np.triu(W_int, k=1))
        edge_w = W_int[ei, ej]
        edge_i_t = torch.tensor(ei, dtype=torch.long)
        edge_j_t = torch.tensor(ej, dtype=torch.long)
        edge_w_t = torch.tensor(edge_w, dtype=torch.float64)
    else:
        edge_i_t = edge_j_t = edge_w_t = None
    # 重疊懲罰是對稱的（pair (i,j) 跟 (j,i) 算出來的重疊面積一樣），先算好
    # 只算上三角（i<j）需要的索引，避免每一步都建構、相乘整個 k×k 矩陣再乘
    # 布林遮罩去掉下三角——直接用索引取值，計算量直接砍半，且不必配置那兩個
    # k×k 的中間矩陣。
    triu_i, triu_j = np.triu_indices(k, k=1)
    triu_i_t = torch.tensor(triu_i, dtype=torch.long)
    triu_j_t = torch.tensor(triu_j, dtype=torch.long)

    opt = torch.optim.Adam([x_param, y_param], lr=lr)

    best_loss = float("inf")
    no_improve = 0
    n_used = 0
    for step in range(n_steps):
        n_used = step + 1
        opt.zero_grad()
        xt = x_full.clone(); yt = y_full.clone()
        xt[m_idx] = x_param
        yt[m_idx] = y_param

        xr = xt + w_t; ytop = yt + h_t
        xi_p, xj_p = xt[triu_i_t], xt[triu_j_t]
        yi_p, yj_p = yt[triu_i_t], yt[triu_j_t]
        xri_p, xrj_p = xr[triu_i_t], xr[triu_j_t]
        ytopi_p, ytopj_p = ytop[triu_i_t], ytop[triu_j_t]
        ox = torch.clamp(torch.minimum(xri_p, xrj_p) - torch.maximum(xi_p, xj_p), min=0.0)
        oy = torch.clamp(torch.minimum(ytopi_p, ytopj_p) - torch.maximum(yi_p, yj_p), min=0.0)
        overlap_loss = (ox * oy).sum()

        xmin = xt.min(); xmax = xr.max(); ymin = yt.min(); ymax = ytop.max()
        area_loss = (xmax - xmin) * (ymax - ymin)

        boundary_loss = torch.tensor(0.0, dtype=torch.float64)
        if left_mask.any():
            boundary_loss = boundary_loss + ((xt[left_mask] - xmin) ** 2).sum()
        if right_mask.any():
            boundary_loss = boundary_loss + ((xr[right_mask] - xmax) ** 2).sum()
        if top_mask.any():
            boundary_loss = boundary_loss + ((ytop[top_mask] - ymax) ** 2).sum()
        if bottom_mask.any():
            boundary_loss = boundary_loss + ((yt[bottom_mask] - ymin) ** 2).sum()

        cluster_loss = torch.tensor(0.0, dtype=torch.float64)
        cx = xt + w_t / 2.0; cy = yt + h_t / 2.0
        for cm in cluster_masks:
            if cm.sum() < 2:
                continue
            gcx = cx[cm].mean(); gcy = cy[cm].mean()
            cluster_loss = cluster_loss + (((cx[cm] - gcx) ** 2 + (cy[cm] - gcy) ** 2)).sum()

        hpwl_loss = torch.tensor(0.0, dtype=torch.float64)
        if edge_i_t is not None and len(edge_i_t) > 0:
            xi, xj = xt[edge_i_t], xt[edge_j_t]
            yi, yj = yt[edge_i_t], yt[edge_j_t]
            xri, xrj = xr[edge_i_t], xr[edge_j_t]
            ytopi, ytopj = ytop[edge_i_t], ytop[edge_j_t]
            span_x = torch.maximum(xri, xrj) - torch.minimum(xi, xj)
            span_y = torch.maximum(ytopi, ytopj) - torch.minimum(yi, yj)
            hpwl_loss = (edge_w_t * (span_x + span_y)).sum()

        anchor_loss = (x_param.mean() - orig_mean_x) ** 2 + (y_param.mean() - orig_mean_y) ** 2

        containment_loss = torch.tensor(0.0, dtype=torch.float64)
        if has_outline:
            left_over = torch.clamp(c_oxmin - xt[m_idx], min=0.0)
            right_over = torch.clamp(xr[m_idx] - c_oxmax, min=0.0)
            bottom_over = torch.clamp(c_oymin - yt[m_idx], min=0.0)
            top_over = torch.clamp(ytop[m_idx] - c_oymax, min=0.0)
            containment_loss = (left_over ** 2 + right_over ** 2
                                + bottom_over ** 2 + top_over ** 2).sum()

        loss = (weight_overlap * overlap_loss + weight_area * area_loss
                + weight_boundary * boundary_loss + weight_cluster * cluster_loss
                + weight_hpwl * hpwl_loss + weight_anchor * anchor_loss
                + weight_containment * containment_loss)
        loss.backward()
        opt.step()

        loss_val = float(loss)
        if loss_val < best_loss - min_delta:
            best_loss = loss_val
            no_improve = 0
        else:
            no_improve += 1

        if verbose and step % 100 == 0:
            print("compact_gradient_finetune: step={:3d} loss={:.2f} overlap={:.4f} "
                  "area={:.1f}".format(step, float(loss), float(overlap_loss), float(area_loss)))

        if no_improve >= patience:
            break

    if verbose:
        print("compact_gradient_finetune: stopped after {} / {} steps".format(n_used, n_steps))

    x_prop = x_full.clone(); y_prop = y_full.clone()
    x_prop[m_idx] = x_param.detach()
    y_prop[m_idx] = y_param.detach()
    x_prop = x_prop.numpy(); y_prop = y_prop.numpy()

    preplaced_idx_list = [i for i in range(k) if preplaced_mask[i]]
    x_proj, y_proj = hard_zero_overlap(x_prop, y_prop, w, h, preplaced_indices=preplaced_idx_list)
    canvas_bbox = tuple(float(v) for v in outline_bbox) if outline_bbox is not None else None
    x_proj, y_proj = compact_positions(x_proj, y_proj, w, h, preplaced_mask=preplaced_mask,
                                       boundary_code=boundary_code, canvas_bbox=canvas_bbox)

    new_bbox = (float((x_proj + w).max()) - float(x_proj.min())) * \
               (float((y_proj + h).max()) - float(y_proj.min()))
    new_v = compute_boundary_violations(x_proj, y_proj, w, h, boundary_code)
    new_cluster_v = compute_cluster_violations(x_proj, y_proj, w, h, cluster_group) if has_cluster else 0
    new_hpwl = compute_hpwl_vectorized(x_proj, y_proj, w, h, W_int) if check_hpwl else 0.0

    within_outline = True
    if outline_bbox is not None:
        oxmin, oymin, oxmax, oymax = canvas_bbox
        within_outline = (float(x_proj.min()) >= oxmin - 1e-6
                          and float(y_proj.min()) >= oymin - 1e-6
                          and float((x_proj + w).max()) <= oxmax + 1e-6
                          and float((y_proj + h).max()) <= oymax + 1e-6)

    accept = (new_bbox < baseline_bbox - 1e-6
              and within_outline
              and new_v <= baseline_v + max(0, int(boundary_violation_slack))
              and new_cluster_v <= baseline_cluster_v
              and (not check_hpwl or new_hpwl <= baseline_hpwl + hpwl_slack + 1e-6))

    if verbose:
        print("compact_gradient_finetune: bbox {:.1f}->{:.1f}  V_bnd {}->{}  "
              "V_cluster {}->{}  hpwl {:.2f}->{:.2f}  within_outline={}  accept={}".format(
                  baseline_bbox, new_bbox, baseline_v, new_v,
                  baseline_cluster_v, new_cluster_v, baseline_hpwl, new_hpwl, within_outline, accept))

    if accept:
        return x_proj, y_proj
    return x.copy(), y.copy()


def _reinsert_best_position(idx, x, y, w, h, others_mask, boundary_code, avg_side,
                            grid_density, gap_weight):
    """
    compact_reinsert 內層搜尋的獨立版本：假設 block idx 已經從 layout「拔出來」
    （others_mask 標出還留在場上的 block），在 others_mask 目前佔用範圍內找
    一個新位置，讓「重新插入後的 bbox 面積 + gap_weight * avg_side * 離最近
    鄰居距離」最小。回傳 (best_x, best_y, best_cost)；others_mask 全空時回傳
    block 原本的位置（沒有東西可以參考，維持原地）。
    """
    if not others_mask.any():
        return float(x[idx]), float(y[idx]), 0.0
    xo, yo, wo, ho = x[others_mask], y[others_mask], w[others_mask], h[others_mask]
    ox_min = float(xo.min()); ox_max = float((xo + wo).max())
    oy_min = float(yo.min()); oy_max = float((yo + ho).max())
    wi, hi = float(w[idx]), float(h[idx])

    def _nearest_gap(xc, yc):
        gx = np.maximum(0.0, np.maximum(xo - (xc + wi), xc - (xo + wo)))
        gy = np.maximum(0.0, np.maximum(yo - (yc + hi), yc - (yo + ho)))
        return float(np.sqrt(gx * gx + gy * gy).min())

    n_grid = max(int(grid_density), 3)
    code = int(boundary_code[idx])
    lock_x = bool(code & 3)
    lock_y = bool(code & 12)
    x_candidates = [float(x[idx])] if lock_x else np.linspace(ox_min - wi, ox_max, n_grid)
    y_candidates = [float(y[idx])] if lock_y else np.linspace(oy_min - hi, oy_max, n_grid)

    best_cost, best_pos = None, (float(x[idx]), float(y[idx]))
    for xc in x_candidates:
        xc = float(xc)
        ox_arr = np.minimum(xc + wi, xo + wo) - np.maximum(xc, xo)
        x_touches = ox_arr > 1e-9
        for yc in y_candidates:
            yc = float(yc)
            oy_arr = np.minimum(yc + hi, yo + ho) - np.maximum(yc, yo)
            if np.any(x_touches & (oy_arr > 1e-9)):
                continue
            bbox_cost = ((max(ox_max, xc + wi) - min(ox_min, xc)) *
                         (max(oy_max, yc + hi) - min(oy_min, yc)))
            cost = bbox_cost + gap_weight * avg_side * _nearest_gap(xc, yc)
            if best_cost is None or cost < best_cost - 1e-9:
                best_cost, best_pos = cost, (xc, yc)
    if best_cost is None:
        return None, None, None
    return best_pos[0], best_pos[1], best_cost


def compact_pair_reinsert(x, y, w, h, preplaced_mask=None, boundary_code=None,
                          cluster_group=None, touch_tol_ratio=0.02, sweeps=3,
                          grid_density=12, gap_weight=0.6,
                          W_int=None, p2b_edges=None, pins_pos=None,
                          hpwl_slack_ratio=0.0, verbose=False):
    """
    v4.9（實驗用）：`compact_reinsert` 一次只拔一個 block 出來重插——如果
    「兩個互相靠近的 block 都要挪位置、才能一起讓 bbox 縮小」，單一 block
    搜尋永遠看不到這個組合（各自單獨嘗試時，任何一個先移動都可能暫時讓
    bbox 打平甚至變差，被 compact_reinsert「嚴格變好才採用」的規則擋下）。
    這是 detailed placement 文獻裡標準的 local search 手法——一次移動一小群
    互相鄰近的 block（這裡取最小的 pair），比單一 block 的搜尋更有機會跳出
    「兩邊互卡」的局部最佳解。

    只對 touching graph 上彼此鄰接的 pair 出手（把搜尋範圍限制在真正有機會
    互相影響的鄰居，避免 O(k^2) 全枚舉）：把兩個 block 都拔出來，用跟
    compact_reinsert 相同的網格搜尋依序找新位置（兩種順序都試，取較好的一
    組），只有在滿足以下全部條件時才採用：
      1. 最終全域 bbox 面積嚴格小於套用前（這是這個 pass 的核心目標，跟其他
         compact_* pass「不變差就好」不同，這裡要求真的變好）。
      2. boundary / cluster 違規數不變差（若提供 cluster_group）。
      3. 總 HPWL 增加量不超過 hpwl_slack_ratio * avg_side（若提供 W_int/
         p2b_edges/pins_pos；閘門邏輯跟 compact_merge_cluster_groups 一致）。
    """
    x = np.array(x, dtype=np.float64).copy()
    y = np.array(y, dtype=np.float64).copy()
    w = np.asarray(w, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    k = len(x)
    if k < 2:
        return x, y
    if preplaced_mask is None:
        preplaced_mask = np.zeros(k, dtype=bool)
    else:
        preplaced_mask = np.asarray(preplaced_mask, dtype=bool)
    if boundary_code is None:
        boundary_code = np.zeros(k, dtype=np.int64)
    else:
        boundary_code = np.asarray(boundary_code, dtype=np.int64)
    has_cluster = cluster_group is not None
    if has_cluster:
        cluster_group = np.asarray(cluster_group, dtype=np.int64)

    avg_side = float(np.mean(np.sqrt(w * h))) if k else 1.0
    touch_tol = max(touch_tol_ratio * avg_side, 1e-6)
    check_hpwl = W_int is not None or (p2b_edges and pins_pos is not None)
    hpwl_slack = max(0.0, hpwl_slack_ratio) * avg_side

    def _total_hpwl(xx, yy):
        total = 0.0
        if W_int is not None:
            total += compute_hpwl_vectorized(xx, yy, w, h, W_int)
        if p2b_edges and pins_pos is not None:
            total += compute_p2b_hpwl(xx, yy, w, h, p2b_edges, pins_pos)
        return total

    def _bbox_area(xx, yy):
        return (float((xx + w).max()) - float(xx.min())) * (float((yy + h).max()) - float(yy.min()))

    for _sweep in range(sweeps):
        xr = x + w; yt = y + h
        gx = np.maximum(0.0, np.maximum(x[:, None] - xr[None, :], x[None, :] - xr[:, None]))
        gy = np.maximum(0.0, np.maximum(y[:, None] - yt[None, :], y[None, :] - yt[:, None]))
        adj = np.sqrt(gx * gx + gy * gy) <= touch_tol
        np.fill_diagonal(adj, False)

        pairs = [(i, j) for i in range(k) for j in np.nonzero(adj[i])[0] if j > i]
        moved_any = False

        for i, j in pairs:
            if preplaced_mask[i] or preplaced_mask[j]:
                continue
            baseline_bbox = _bbox_area(x, y)
            baseline_v = compute_boundary_violations(x, y, w, h, boundary_code)
            baseline_cluster_v = compute_cluster_violations(x, y, w, h, cluster_group) if has_cluster else 0
            baseline_hpwl = _total_hpwl(x, y) if check_hpwl else 0.0

            best_result = None
            for first, second in ((i, j), (j, i)):
                others_first = np.ones(k, dtype=bool)
                others_first[i] = False; others_first[j] = False
                fx, fy, _ = _reinsert_best_position(first, x, y, w, h, others_first,
                                                    boundary_code, avg_side, grid_density, gap_weight)
                if fx is None:
                    continue
                xt = x.copy(); yt2 = y.copy()
                xt[first] = fx; yt2[first] = fy

                others_second = np.ones(k, dtype=bool)
                others_second[second] = False
                sx, sy, _ = _reinsert_best_position(second, xt, yt2, w, h, others_second,
                                                    boundary_code, avg_side, grid_density, gap_weight)
                if sx is None:
                    continue
                xt[second] = sx; yt2[second] = sy

                new_bbox = _bbox_area(xt, yt2)
                if new_bbox >= baseline_bbox - 1e-6:
                    continue
                if compute_boundary_violations(xt, yt2, w, h, boundary_code) > baseline_v:
                    continue
                if has_cluster and compute_cluster_violations(xt, yt2, w, h, cluster_group) > baseline_cluster_v:
                    continue
                if check_hpwl and _total_hpwl(xt, yt2) > baseline_hpwl + hpwl_slack + 1e-6:
                    continue
                if best_result is None or new_bbox < best_result[0] - 1e-9:
                    best_result = (new_bbox, xt, yt2)

            if best_result is not None:
                _, xt, yt2 = best_result
                x[:] = xt; y[:] = yt2
                moved_any = True
                if verbose:
                    print("compact_pair_reinsert: sweep={} pair=({},{}) bbox {:.1f} -> {:.1f}".format(
                        _sweep, i, j, baseline_bbox, best_result[0]))

        if not moved_any:
            break

    return x, y


def compact_merge_clusters(x, y, w, h, preplaced_mask=None, boundary_code=None,
                           touch_tol_ratio=0.02, rounds=20, verbose=False):
    """
    找出目前 layout 裡「彼此貼合」的連通元件（touching graph 的連通分量），
    把主要群（面積最大的那群）以外的每個衛星群整體平移、往主要群靠近，
    直到快貼上為止。

    動機：compact_reinsert / compact_gravity 都是一次只動一個 block、只在
    「對這個 block 自己有利」時才移動——如果一大群 block 已經彼此緊貼形成
    一個緊密團塊，團塊內任何一個 block 想往外移都會立刻撞到緊貼的鄰居，
    這種「一次一個」的搜尋完全沒辦法讓整團一起挪動。當團塊之外還有幾個
    因為 boundary 約束被釘在遠處的 block（衛星群）時，中間就會留下一大塊
    「誰都沒有誘因去填」的空白——這正是視覺上「主要群聚跟幾個貼邊 block
    中間空一大塊」的成因。這裡直接把衛星群當剛體一起平移過去，從根本解決。

    preplaced_mask: 若某群包含 preplaced block，整群不動（有固定錨點）。
    boundary_code:  用「移動後重算官方 boundary 判定式、違規數不能超過移動前」
                    當安全閘門（見 _try_axis），而非單純看有沒有沾到鎖定 bit。
                    v4.5：若「兩個以上」衛星群共享同一條邊界鎖（例如都是
                    RIGHT-locked），會先把它們當一個剛體家族一起平移，因為
                    單獨移動任一個都會讓它自己脫離那條邊、被閘門擋下——整族
                    一起移動才能在不違反任何成員鎖定的前提下縮小跟 main 的
                    距離（見下方「家族協同移動」區塊）。

    只動 x, y（不碰 w, h）；每次平移都用二分法找最大安全距離（不會跟任何
    「非本群」的 block 重疊），保證不會把不重疊的狀態變成重疊。
    """
    x = np.array(x, dtype=np.float64).copy()
    y = np.array(y, dtype=np.float64).copy()
    w = np.asarray(w, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    k = len(x)
    if k < 2:
        return x, y
    if preplaced_mask is None:
        preplaced_mask = np.zeros(k, dtype=bool)
    else:
        preplaced_mask = np.asarray(preplaced_mask, dtype=bool)
    if boundary_code is None:
        boundary_code = np.zeros(k, dtype=np.int64)
    else:
        boundary_code = np.asarray(boundary_code, dtype=np.int64)
    avg_side = float(np.mean(np.sqrt(w * h))) if k else 1.0
    touch_tol = max(touch_tol_ratio * avg_side, 1e-6)
    areas_arr = w * h

    for _round in range(rounds):
        # ---- 建 touching graph（gap <= touch_tol 視為同群）、找連通分量 ----
        xr = x + w; yt = y + h
        gx = np.maximum(0.0, np.maximum(x[:, None] - xr[None, :], x[None, :] - xr[:, None]))
        gy = np.maximum(0.0, np.maximum(y[:, None] - yt[None, :], y[None, :] - yt[:, None]))
        gap_mat = np.sqrt(gx * gx + gy * gy)
        adj = gap_mat <= touch_tol
        np.fill_diagonal(adj, False)

        comp_id = -np.ones(k, dtype=int)
        n_comp = 0
        for s in range(k):
            if comp_id[s] != -1:
                continue
            stack = [s]
            comp_id[s] = n_comp
            while stack:
                u = stack.pop()
                for v in np.nonzero(adj[u])[0]:
                    if comp_id[v] == -1:
                        comp_id[v] = n_comp
                        stack.append(v)
            n_comp += 1

        if n_comp <= 1:
            break   # 只有一群，沒有衛星群要合併

        comp_area = np.array([areas_arr[comp_id == c].sum() for c in range(n_comp)])
        main_c = int(np.argmax(comp_area))
        main_mask0 = comp_id == main_c

        moved_any = False

        # ---- 家族協同移動：多個衛星群共享同一條邊界鎖時，整族剛體一起平移 ----
        # 下面的「單一衛星 vs main」邏輯沒辦法處理「兩個以上各自獨立的衛星群，
        # 剛好都被鎖在同一條邊」的情況：任何一個衛星群單獨試著往內移動，都會
        # 被「移動後這個衛星自己不再貼著那條邊」的驗證擋下來——因為邊界本身
        # 是由「目前這條邊上最極端的 block」自我定義，若同一條邊還有其他衛星
        # 群守著，這個衛星群移開後自己就不再是最極端的，等於違反自己的鎖定。
        # 解法：把「共享同一條邊界鎖」的所有衛星群當成一個剛體家族，用同一個
        # 位移量一起移動——家族內部相對位置不變，所以移動後家族仍然共同定義
        # 同一條（跟著縮小的）邊界，不會違反任何一個成員的鎖定。
        family_handled_axis = {}
        for bit, axis in ((1, 'x'), (2, 'x'), (4, 'y'), (8, 'y')):
            fam = [c for c in range(n_comp) if c != main_c
                   and not preplaced_mask[comp_id == c].any()
                   and bool((boundary_code[comp_id == c] & bit).any())]
            if len(fam) < 2:
                continue
            if bool((boundary_code[main_mask0] & bit).any()):
                continue   # main 也鎖在這條邊，這軸沒有可壓縮的空間
            fam_mask = np.isin(comp_id, fam)
            main_mask = main_mask0
            others_mask = ~fam_mask & ~main_mask
            main_immovable = bool(preplaced_mask[main_mask].any())

            main_area_sum = areas_arr[main_mask].sum()
            main_cx = float(((x[main_mask] + w[main_mask] / 2) * areas_arr[main_mask]).sum() / main_area_sum)
            main_cy = float(((y[main_mask] + h[main_mask] / 2) * areas_arr[main_mask]).sum() / main_area_sum)
            fam_area_sum = areas_arr[fam_mask].sum()
            fam_cx = float(((x[fam_mask] + w[fam_mask] / 2) * areas_arr[fam_mask]).sum() / fam_area_sum)
            fam_cy = float(((y[fam_mask] + h[fam_mask] / 2) * areas_arr[fam_mask]).sum() / fam_area_sum)
            needed = (main_cx - fam_cx) if axis == 'x' else (main_cy - fam_cy)
            for c in fam:
                family_handled_axis.setdefault(c, set()).add(axis)
            if abs(needed) < 1e-9:
                continue

            def _apply_fam(dx_s, dy_s, dx_m, dy_m, t):
                xt = x.copy(); yt = y.copy()
                xt[fam_mask] = x[fam_mask] + dx_s * t
                yt[fam_mask] = y[fam_mask] + dy_s * t
                xt[main_mask] = x[main_mask] + dx_m * t
                yt[main_mask] = y[main_mask] + dy_m * t
                return xt, yt

            def _overlap_bad_fam(xt, yt):
                for grp_mask, other_mask in ((fam_mask, main_mask), (fam_mask, others_mask),
                                              (main_mask, others_mask)):
                    if not other_mask.any() or not grp_mask.any():
                        continue
                    gx_, gy_, gw_, gh_ = xt[grp_mask], yt[grp_mask], w[grp_mask], h[grp_mask]
                    ox_, oy_, ow_, oh_ = xt[other_mask], yt[other_mask], w[other_mask], h[other_mask]
                    ovx = np.minimum(gx_[:, None] + gw_[:, None], ox_[None, :] + ow_[None, :]) - \
                          np.maximum(gx_[:, None], ox_[None, :])
                    ovy = np.minimum(gy_[:, None] + gh_[:, None], oy_[None, :] + oh_[None, :]) - \
                          np.maximum(gy_[:, None], oy_[None, :])
                    if bool(((ovx > 1e-9) & (ovy > 1e-9)).any()):
                        return True
                return False

            baseline_v_fam = compute_boundary_violations(x, y, w, h, boundary_code)

            def _feasible_fam(dx_s, dy_s, dx_m, dy_m, t):
                xt, yt = _apply_fam(dx_s, dy_s, dx_m, dy_m, t)
                if _overlap_bad_fam(xt, yt):
                    return False
                return compute_boundary_violations(xt, yt, w, h, boundary_code) <= baseline_v_fam

            def _max_t_fam(dx_s, dy_s, dx_m, dy_m):
                if abs(dx_s) + abs(dy_s) + abs(dx_m) + abs(dy_m) < 1e-9:
                    return 0.0
                if _feasible_fam(dx_s, dy_s, dx_m, dy_m, 1.0):
                    return 1.0
                if not _feasible_fam(dx_s, dy_s, dx_m, dy_m, 0.0):
                    return 0.0
                lo, hi = 0.0, 1.0
                for _ in range(30):
                    mid = (lo + hi) / 2.0
                    if _feasible_fam(dx_s, dy_s, dx_m, dy_m, mid):
                        lo = mid
                    else:
                        hi = mid
                return lo

            dx_s, dy_s = (needed, 0.0) if axis == 'x' else (0.0, needed)
            t_fam = _max_t_fam(dx_s, dy_s, 0.0, 0.0)
            if t_fam > 1e-6:
                xt, yt = _apply_fam(dx_s, dy_s, 0.0, 0.0, t_fam)
                x[:] = xt; y[:] = yt
                moved_any = True
                if verbose:
                    print("compact_merge_clusters: round={} family bit={} ({} sats) "
                          "moved t={:.3f}".format(_round, bit, len(fam), t_fam))
            if t_fam < 0.999 and not main_immovable:
                remaining = needed * (1.0 - t_fam)
                if abs(remaining) > 1e-9:
                    dx_m, dy_m = (-remaining, 0.0) if axis == 'x' else (0.0, -remaining)
                    t_main = _max_t_fam(0.0, 0.0, dx_m, dy_m)
                    if t_main > 1e-6:
                        xt, yt = _apply_fam(0.0, 0.0, dx_m, dy_m, t_main)
                        x[:] = xt; y[:] = yt
                        moved_any = True

        for c in range(n_comp):
            if c == main_c:
                continue
            comp_mask = comp_id == c
            if preplaced_mask[comp_mask].any():
                continue

            main_mask = comp_id == main_c
            main_immovable = bool(preplaced_mask[main_mask].any())
            others_mask = ~comp_mask & ~main_mask

            main_area_sum = areas_arr[main_mask].sum()
            main_cx = float(((x[main_mask] + w[main_mask] / 2) * areas_arr[main_mask]).sum() / main_area_sum)
            main_cy = float(((y[main_mask] + h[main_mask] / 2) * areas_arr[main_mask]).sum() / main_area_sum)
            comp_area_sum = areas_arr[comp_mask].sum()
            comp_cx = float(((x[comp_mask] + w[comp_mask] / 2) * areas_arr[comp_mask]).sum() / comp_area_sum)
            comp_cy = float(((y[comp_mask] + h[comp_mask] / 2) * areas_arr[comp_mask]).sum() / comp_area_sum)
            needed_dx = main_cx - comp_cx
            needed_dy = main_cy - comp_cy

            def _apply(dx_s, dy_s, dx_m, dy_m, t):
                xt = x.copy(); yt = y.copy()
                xt[comp_mask] = x[comp_mask] + dx_s * t
                yt[comp_mask] = y[comp_mask] + dy_s * t
                xt[main_mask] = x[main_mask] + dx_m * t
                yt[main_mask] = y[main_mask] + dy_m * t
                return xt, yt

            def _overlap_bad(xt, yt):
                for grp_mask, other_mask in ((comp_mask, main_mask), (comp_mask, others_mask),
                                              (main_mask, others_mask)):
                    if not other_mask.any() or not grp_mask.any():
                        continue
                    gx, gy, gw, gh = xt[grp_mask], yt[grp_mask], w[grp_mask], h[grp_mask]
                    ox, oy, ow, oh = xt[other_mask], yt[other_mask], w[other_mask], h[other_mask]
                    ovx = np.minimum(gx[:, None] + gw[:, None], ox[None, :] + ow[None, :]) - \
                          np.maximum(gx[:, None], ox[None, :])
                    ovy = np.minimum(gy[:, None] + gh[:, None], oy[None, :] + oh[None, :]) - \
                          np.maximum(gy[:, None], oy[None, :])
                    if bool(((ovx > 1e-9) & (ovy > 1e-9)).any()):
                        return True
                return False

            def _feasible(dx_s, dy_s, dx_m, dy_m, t, baseline_v):
                xt, yt = _apply(dx_s, dy_s, dx_m, dy_m, t)
                if _overlap_bad(xt, yt):
                    return False
                return compute_boundary_violations(xt, yt, w, h, boundary_code) <= baseline_v

            def _max_t(dx_s, dy_s, dx_m, dy_m, baseline_v):
                if abs(dx_s) + abs(dy_s) + abs(dx_m) + abs(dy_m) < 1e-9:
                    return 0.0
                if _feasible(dx_s, dy_s, dx_m, dy_m, 1.0, baseline_v):
                    return 1.0
                if not _feasible(dx_s, dy_s, dx_m, dy_m, 0.0, baseline_v):
                    return 0.0
                lo, hi = 0.0, 1.0
                for _ in range(30):
                    mid = (lo + hi) / 2.0
                    if _feasible(dx_s, dy_s, dx_m, dy_m, mid, baseline_v):
                        lo = mid
                    else:
                        hi = mid
                return lo

            def _try_axis(axis, needed):
                # 用「移動後用官方 boundary 判定式（含 1% 容忍帶）重算，違規數
                # 不能超過移動前」當安全閘門，取代舊版「只要沾到 LEFT/RIGHT/
                # TOP/BOTTOM 任一 bit 就整軸鎖死不動」的過度保守判斷。因為
                # bbox 邊界本身是「目前最極端的 block」自我定義的，貼邊群整體
                # 往內移動時邊界通常會跟著收縮，並不會真的破壞自己的貼邊關
                # 係——只有在移動方向會讓某個「其他」被鎖住的 block 變成新的
                # 極值時才會真的破壞，這種情況就會被 violation 閘門擋下來。
                # 衛星群先自己盡量靠過去；剩下的距離（衛星群卡住的部分），若
                # 主要群不是 preplaced，再讓主要群補上剩餘距離。
                baseline_v = compute_boundary_violations(x, y, w, h, boundary_code)
                dx_s, dy_s = (needed, 0.0) if axis == 'x' else (0.0, needed)
                t_sat = _max_t(dx_s, dy_s, 0.0, 0.0, baseline_v)
                moved = False
                if t_sat > 1e-6:
                    xt, yt = _apply(dx_s, dy_s, 0.0, 0.0, t_sat)
                    x[:] = xt; y[:] = yt
                    moved = True

                if t_sat < 0.999 and not main_immovable:
                    remaining = needed * (1.0 - t_sat)
                    if abs(remaining) > 1e-9:
                        dx_m, dy_m = (-remaining, 0.0) if axis == 'x' else (0.0, -remaining)
                        baseline_v2 = compute_boundary_violations(x, y, w, h, boundary_code)
                        t_main = _max_t(0.0, 0.0, dx_m, dy_m, baseline_v2)
                        if t_main > 1e-6:
                            xt, yt = _apply(0.0, 0.0, dx_m, dy_m, t_main)
                            x[:] = xt; y[:] = yt
                            moved = True
                return moved

            handled = family_handled_axis.get(c, set())
            moved_x = False if 'x' in handled else _try_axis('x', needed_dx)
            moved_y = False if 'y' in handled else _try_axis('y', needed_dy)
            if moved_x or moved_y:
                moved_any = True
            elif verbose:
                print("compact_merge_clusters: round={} comp {} blocks stuck "
                      "(needed=({:.2f},{:.2f}), both sides boundary-locked on "
                      "the relevant axis)".format(_round, int(comp_mask.sum()),
                                                   needed_dx, needed_dy))

        if not moved_any:
            break

    if verbose:
        xr = x + w; yt = y + h
        gx = np.maximum(0.0, np.maximum(x[:, None] - xr[None, :], x[None, :] - xr[:, None]))
        gy = np.maximum(0.0, np.maximum(y[:, None] - yt[None, :], y[None, :] - yt[:, None]))
        adj = np.sqrt(gx * gx + gy * gy) <= touch_tol
        np.fill_diagonal(adj, False)
        comp_id2 = -np.ones(k, dtype=int)
        n2 = 0
        for s in range(k):
            if comp_id2[s] != -1:
                continue
            stack = [s]; comp_id2[s] = n2
            while stack:
                u = stack.pop()
                for v in np.nonzero(adj[u])[0]:
                    if comp_id2[v] == -1:
                        comp_id2[v] = n2; stack.append(v)
            n2 += 1
        print("compact_merge_clusters done: {} connected component(s) remaining "
              "(1 = fully merged)".format(n2))

    return x, y


def compact_snap_boundary(x, y, w, h, preplaced_mask=None, boundary_code=None,
                          cluster_group=None, touch_tol_ratio=0.02, rounds=10,
                          W_int=None, p2b_edges=None, pins_pos=None,
                          boundary_hpwl_slack_ratio=0.0, verbose=False):
    """
    針對目前還沒真正貼到邊的 boundary block，把它所在的剛體貼合分量整體
    平移，推向 layout 目前的真實邊界（xmin/xmax/ymin/ymax——boundary 的定義
    見 compute_boundary_violations，是「目前所有 block 的 bbox」，不是
    outline/canvas），直到貼上邊界或被別的 block 擋住為止。

    動機：`compact_positions` 只往 -x/-y 方向壓縮，這剛好跟 LEFT(1)/
    BOTTOM(8) 鎖定的方向一致，所以這兩種 lock 通常會被自然帶到邊界；但
    RIGHT(2)/TOP(4) 鎖定需要的是「往 +x/+y 推」，而 pipeline 裡沒有任何
    通用機制做這件事——這些 block 只有在 legalize_lff 最初的 LFF 放置階段
    被「往 outline 邊界拉」的加權中位數軟性訊號影響過一次（會跟 weight_dist
    /weight_cluster/weight_b2b 等競爭，不保證真的落在邊界上），放置完之後
    的每個壓縮 pass 都只會「保護」它們的位置不被 -x/-y 壓縮誤傷，沒有人會
    再把它們往外推。這是 V_boundary 違規裡結構性、方向不對稱的主因之一。

    每次移動都：
    1. 用二分法找最大安全平移距離——目標本身就是「現有的邊界值」，所以移動
       不會讓 bbox 變大；若中途被其他 block 擋住，則移到安全上限為止（不保
       證這輪就真的貼上，但下一輪 touching graph 更新後可能又有新空間）。
    2. 移動的是這個 block 所在的剛體貼合分量（沿用跟 compact_merge_clusters
       一樣的鬆散 touching graph），保留跟其他 block 既有的貼合關係，不拆散
       既有的 cluster/HPWL 接觸。
    3. 安全閘門：移動後總 boundary 違規數不能超過移動前（跟 compact_merge_
       clusters 用同一個判定式；因為目標本身就是現有邊界值，這個移動不可能
       讓任何「其他」被鎖定的 block 失去其極值地位）；若提供 cluster_group，
       移動後總 cluster 違規數也不能變差；若提供 W_int/p2b_edges/pins_pos，
       移動前後總 HPWL 增加量不能超過 boundary_hpwl_slack_ratio * avg_side
       （用獨立參數，不跟 compact_merge_cluster_groups 的 hpwl_slack_ratio
       共用，方便個別調參；閘門邏輯跟該函式一致，見其 docstring）。
    """
    x = np.array(x, dtype=np.float64).copy()
    y = np.array(y, dtype=np.float64).copy()
    w = np.asarray(w, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    k = len(x)
    if k < 2:
        return x, y
    if preplaced_mask is None:
        preplaced_mask = np.zeros(k, dtype=bool)
    else:
        preplaced_mask = np.asarray(preplaced_mask, dtype=bool)
    if boundary_code is None:
        boundary_code = np.zeros(k, dtype=np.int64)
    else:
        boundary_code = np.asarray(boundary_code, dtype=np.int64)
    if not (boundary_code > 0).any():
        return x, y
    has_cluster = cluster_group is not None
    if has_cluster:
        cluster_group = np.asarray(cluster_group, dtype=np.int64)

    avg_side = float(np.mean(np.sqrt(w * h))) if k else 1.0
    touch_tol = max(touch_tol_ratio * avg_side, 1e-6)

    check_hpwl = W_int is not None or (p2b_edges and pins_pos is not None)
    hpwl_slack = max(0.0, boundary_hpwl_slack_ratio) * avg_side

    def _total_hpwl(xx, yy):
        total = 0.0
        if W_int is not None:
            total += compute_hpwl_vectorized(xx, yy, w, h, W_int)
        if p2b_edges and pins_pos is not None:
            total += compute_p2b_hpwl(xx, yy, w, h, p2b_edges, pins_pos)
        return total

    # bit -> (axis, sign, edge_fn(x,y,w,h,i) -> current edge coord of block i)
    _BITS = (
        (1, 'x', -1),   # LEFT: 往 -x 貼齊 xmin
        (2, 'x', +1),   # RIGHT: 往 +x 貼齊 xmax
        (4, 'y', +1),   # TOP: 往 +y 貼齊 ymax
        (8, 'y', -1),   # BOTTOM: 往 -y 貼齊 ymin
    )

    for _round in range(rounds):
        xr = x + w; yt = y + h
        gx = np.maximum(0.0, np.maximum(x[:, None] - xr[None, :], x[None, :] - xr[:, None]))
        gy = np.maximum(0.0, np.maximum(y[:, None] - yt[None, :], y[None, :] - yt[:, None]))
        adj = np.sqrt(gx * gx + gy * gy) <= touch_tol
        np.fill_diagonal(adj, False)

        # 連通分量刻意排除 preplaced block（當圖上的洞，不可通過、不可移動）
        # 才計算——若照「鬆散 touching graph 的連通分量裡只要有一個 preplaced
        # block 就整個跳過」的舊邏輯，實測發現經過前面幾輪壓縮後，整個 layout
        # 常常已經收斂成一個涵蓋幾乎所有 block 的巨大連通分量，導致這個函式
        # 在真實資料上幾乎永遠是 no-op（因為 preplaced block 幾乎必然跟其他
        # 所有東西連在一起）。只把「經由非 preplaced block 互相貼合」能連到
        # i 的那些 block 一起搬，preplaced block 本身與只能透過它才連得到的
        # 部分則保持原地不動（在 others_mask 裡當固定的重疊檢查對象）。
        movable = ~preplaced_mask
        adj_mv = adj & movable[:, None] & movable[None, :]
        comp_id_mv = -np.ones(k, dtype=int)
        n_comp_mv = 0
        for s in range(k):
            if not movable[s] or comp_id_mv[s] != -1:
                continue
            stack = [s]; comp_id_mv[s] = n_comp_mv
            while stack:
                u = stack.pop()
                for v in np.nonzero(adj_mv[u])[0]:
                    if comp_id_mv[v] == -1:
                        comp_id_mv[v] = n_comp_mv; stack.append(v)
            n_comp_mv += 1

        xmin, xmax = float(x.min()), float(xr.max())
        ymin, ymax = float(y.min()), float(yt.max())
        rel_tol = 1e-2 * max(xmax - xmin, ymax - ymin, 1e-6)

        moved_any = False
        handled_comp_axis = set()   # (comp_id_mv, axis) 這輪已經處理過，避免重工

        def _try_evict_boundary_obstacles(i, dx, dy, baseline_v, baseline_cluster_v, baseline_hpwl):
            """
            v4.8：i 自己單獨滑到目標位置時，找出擋在路上的 obstacle block（≤2
            個、都不是 preplaced 才處理，太多個代表問題更複雜、不硬解）。依序
            把每個 obstacle「請出去」，用跟 compact_reinsert 完全一樣的網格
            搜尋邏輯（bbox 面積最小、避開所有其他 block）幫它們在別處找新家，
            i 則直接佔用騰出來的位置。全部 obstacle 都成功找到新家、且整體
            通過 boundary/cluster/HPWL 三道閘門才會採用；任何一步失敗就整個
            放棄（不留下部分套用的中間狀態）。

            回傳 (used_mask, apply_fn)；失敗回傳 (None, None)。
            """
            if preplaced_mask[i]:
                return None, None
            target_x, target_y = float(x[i] + dx), float(y[i] + dy)

            others = np.ones(k, dtype=bool); others[i] = False
            ox_ = np.minimum(target_x + w[i], x[others] + w[others]) - np.maximum(target_x, x[others])
            oy_ = np.minimum(target_y + h[i], y[others] + h[others]) - np.maximum(target_y, y[others])
            blocked = (ox_ > 1e-9) & (oy_ > 1e-9)
            obstacle_idx = np.nonzero(others)[0][blocked]
            if len(obstacle_idx) == 0 or len(obstacle_idx) > 2:
                return None, None
            if preplaced_mask[obstacle_idx].any():
                return None, None

            xt = x.copy(); yt2 = y.copy()
            xt[i] = target_x; yt2[i] = target_y
            moved_obstacles = []
            for oj in obstacle_idx.tolist():
                oj = int(oj)
                others2 = np.ones(k, dtype=bool); others2[oj] = False
                xo2, yo2, wo2, ho2 = xt[others2], yt2[others2], w[others2], h[others2]
                ox_min = float(xo2.min()); ox_max = float((xo2 + wo2).max())
                oy_min = float(yo2.min()); oy_max = float((yo2 + ho2).max())
                lock_x_oj = bool(int(boundary_code[oj]) & 3)
                lock_y_oj = bool(int(boundary_code[oj]) & 12)
                cur_xj, cur_yj = float(xt[oj]), float(yt2[oj])
                x_cands = [cur_xj] if lock_x_oj else np.linspace(ox_min - w[oj], ox_max, 12)
                y_cands = [cur_yj] if lock_y_oj else np.linspace(oy_min - h[oj], oy_max, 12)

                best_pos, best_cost = None, None
                for xc in x_cands:
                    xc = float(xc)
                    ox_arr = np.minimum(xc + w[oj], xo2 + wo2) - np.maximum(xc, xo2)
                    x_touches = ox_arr > 1e-9
                    for yc in y_cands:
                        yc = float(yc)
                        oy_arr = np.minimum(yc + h[oj], yo2 + ho2) - np.maximum(yc, yo2)
                        if np.any(x_touches & (oy_arr > 1e-9)):
                            continue
                        bbox_cost = ((max(ox_max, xc + w[oj]) - min(ox_min, xc)) *
                                     (max(oy_max, yc + h[oj]) - min(oy_min, yc)))
                        if best_cost is None or bbox_cost < best_cost - 1e-9:
                            best_cost, best_pos = bbox_cost, (xc, yc)
                if best_pos is None:
                    return None, None
                xt[oj], yt2[oj] = best_pos
                moved_obstacles.append(oj)

            xr2 = xt + w; ytop2 = yt2 + h
            ov_x = np.minimum(xr2[:, None], xr2[None, :]) - np.maximum(xt[:, None], xt[None, :])
            ov_y = np.minimum(ytop2[:, None], ytop2[None, :]) - np.maximum(yt2[:, None], yt2[None, :])
            ov = (ov_x > 1e-9) & (ov_y > 1e-9)
            np.fill_diagonal(ov, False)
            if ov.any():
                return None, None
            # 跟其他兩層（整團剛體、solo）不同：eviction 這條路徑的目標位置是
            # 靠網格搜尋找出來的，不是「移到現有邊界值」這種數學上保證不會讓
            # bbox 變大的操作，需要額外顯式檢查，維持跟其他 compact_* pass
            # 一致的「證明上單調不變差」設計原則。
            baseline_bbox = (float((x + w).max()) - float(x.min())) * (float((y + h).max()) - float(y.min()))
            new_bbox = (float(xr2.max()) - float(xt.min())) * (float(ytop2.max()) - float(yt2.min()))
            if new_bbox > baseline_bbox + 1e-6:
                return None, None
            if compute_boundary_violations(xt, yt2, w, h, boundary_code) > baseline_v:
                return None, None
            if has_cluster and compute_cluster_violations(xt, yt2, w, h, cluster_group) > baseline_cluster_v:
                return None, None
            if check_hpwl and _total_hpwl(xt, yt2) > baseline_hpwl + hpwl_slack + 1e-6:
                return None, None

            used = np.zeros(k, dtype=bool)
            used[i] = True
            for oj in moved_obstacles:
                used[oj] = True
            return used, (lambda t, xt=xt, yt2=yt2: (xt, yt2))

        for i in range(k):
            code = int(boundary_code[i])
            if code == 0 or preplaced_mask[i]:
                continue
            c = int(comp_id_mv[i])

            for bit, axis, sign in _BITS:
                if not (code & bit):
                    continue
                if (c, axis) in handled_comp_axis:
                    continue
                if axis == 'x':
                    cur = (x[i] + w[i]) if sign > 0 else x[i]
                    target = xmax if sign > 0 else xmin
                else:
                    cur = (y[i] + h[i]) if sign > 0 else y[i]
                    target = ymax if sign > 0 else ymin
                if abs(cur - target) <= rel_tol:
                    continue   # 已經貼齊，不用動
                needed = target - cur
                if sign * needed <= 1e-9:
                    continue   # 方向不對（理論上不會發生，防禦性檢查）
                handled_comp_axis.add((c, axis))

                dx, dy = (needed, 0.0) if axis == 'x' else (0.0, needed)
                baseline_v = compute_boundary_violations(x, y, w, h, boundary_code)
                baseline_cluster_v = compute_cluster_violations(x, y, w, h, cluster_group) if has_cluster else 0
                baseline_hpwl = _total_hpwl(x, y) if check_hpwl else 0.0

                def _solve(comp_mask, dx=dx, dy=dy, baseline_v=baseline_v,
                           baseline_cluster_v=baseline_cluster_v, baseline_hpwl=baseline_hpwl):
                    others_mask = ~comp_mask

                    def _apply(t):
                        xt = x.copy(); yt2 = y.copy()
                        xt[comp_mask] = x[comp_mask] + dx * t
                        yt2[comp_mask] = y[comp_mask] + dy * t
                        return xt, yt2

                    def _overlap_bad(xt, yt2):
                        if not others_mask.any():
                            return False
                        gx_, gy_, gw_, gh_ = xt[comp_mask], yt2[comp_mask], w[comp_mask], h[comp_mask]
                        ox_, oy_, ow_, oh_ = xt[others_mask], yt2[others_mask], w[others_mask], h[others_mask]
                        ovx = np.minimum(gx_[:, None] + gw_[:, None], ox_[None, :] + ow_[None, :]) - \
                              np.maximum(gx_[:, None], ox_[None, :])
                        ovy = np.minimum(gy_[:, None] + gh_[:, None], oy_[None, :] + oh_[None, :]) - \
                              np.maximum(gy_[:, None], oy_[None, :])
                        return bool(((ovx > 1e-9) & (ovy > 1e-9)).any())

                    def _feasible(t):
                        xt, yt2 = _apply(t)
                        if _overlap_bad(xt, yt2):
                            return False
                        if compute_boundary_violations(xt, yt2, w, h, boundary_code) > baseline_v:
                            return False
                        if has_cluster and compute_cluster_violations(xt, yt2, w, h, cluster_group) > baseline_cluster_v:
                            return False
                        if check_hpwl and _total_hpwl(xt, yt2) > baseline_hpwl + hpwl_slack + 1e-6:
                            return False
                        return True

                    if _feasible(1.0):
                        return 1.0, _apply
                    if not _feasible(0.0):
                        return 0.0, _apply
                    lo, hi = 0.0, 1.0
                    for _ in range(30):
                        mid = (lo + hi) / 2.0
                        if _feasible(mid):
                            lo = mid
                        else:
                            hi = mid
                    return lo, _apply

                # 先試整個剛體貼合分量（保留 i 跟其他 block 既有的貼合關係）；
                # 如果分量太大、被某個「跟這個違規本身無關」的旁觀成員卡住
                # （i 自己其實有空間，但硬被一起搬的鄰居擋住），退而求其次只搬
                # i 自己（會拆散 i 原本的貼合，但仍然受 boundary/cluster/HPWL
                # 閘門保護，不會讓其他 soft constraint 或 HPWL 明顯變差）。
                comp_mask = comp_id_mv == c
                t_max, apply_fn = _solve(comp_mask)
                used_mask = comp_mask
                if t_max <= 1e-6 and int(comp_mask.sum()) > 1:
                    solo_mask = np.zeros(k, dtype=bool)
                    solo_mask[i] = True
                    t_max, apply_fn = _solve(solo_mask)
                    used_mask = solo_mask

                if t_max <= 1e-6:
                    # 連「只搬 i 自己」都卡死：多半是直線路徑上剛好卡著一、兩個
                    # 同樣合法佔位的 obstacle（見 compact_snap_boundary docstring
                    # 的 v4.8 討論）。單軸滑動解不開，改用「detailed placement」
                    # 風格的做法——把擋路的 obstacle 也一起請出去，用 grid 搜尋
                    # （跟 compact_reinsert 同一套邏輯）幫它們在別處找新家，i 直接
                    # 佔用騰出來的位置貼齊邊界。
                    ev_mask, ev_apply = _try_evict_boundary_obstacles(
                        i, dx, dy, baseline_v, baseline_cluster_v, baseline_hpwl)
                    if ev_mask is not None:
                        t_max = 1.0
                        apply_fn = ev_apply
                        used_mask = ev_mask

                if t_max > 1e-6:
                    xt, yt2 = apply_fn(t_max)
                    x[:] = xt; y[:] = yt2
                    moved_any = True
                    if verbose:
                        print("compact_snap_boundary: round={} comp={} bit={} "
                              "moved t={:.3f} ({} block(s))".format(
                                  _round, c, bit, t_max, int(used_mask.sum())))

        if not moved_any:
            break

    return x, y


def compact_merge_cluster_groups(x, y, w, h, preplaced_mask=None, boundary_code=None,
                                 cluster_group=None, touch_tol_ratio=0.02, rounds=10,
                                 W_int=None, p2b_edges=None, pins_pos=None,
                                 hpwl_slack_ratio=0.0,
                                 verbose=False):
    """
    針對每個 cluster group，檢查該組成員在目前 touching graph 裡是否已經是
    「一個」連通分量——這正是官方 V_grouping 這個指標本身在算的東西（同組
    是否只有一個連通分量，見 compute_cluster_violations；連通性只看 cluster
    group 成員彼此是否直接共邊，非成員 block 不能當橋接）。legalize_lff 目前
    對 cluster 成員只有「加權中位數往組重心拉」這個軟性訊號，重心接近不等於
    真的共邊貼合（`_blocks_share_edge` 要求一軸邊界相接、另一軸要有正長度
    重疊，純角對角不算）——如果同一組被分裂成多塊，單純把重心拉近，常常只
    是讓兩塊斜對角地更靠近，卻還是沒有真的貼上、V_grouping 完全沒有改善。

    跟 compact_merge_clusters 的差別有兩層：
    1. merge 的「目標」不是 bbox 面積最大的全域主要群，而是「同一個 cluster
       group 內、該組成員面積最多」的那個連通分量。
    2. 移動量不是重心差，而是找出「這個連通分量裡離 target 最近的組員 a」和
       「target 裡離它最近的組員 b」，算出 4 種「a 精確貼齊 b 的某一邊、且
       另一軸置中對齊」的候選剛體位移，逐一驗證整段位移是否安全（不重疊、
       boundary 違規不變差），選第一個可行的套用——保證套用後 a、b 是真的
       共邊，而不只是比較靠近。

    保底機制跟 compact_merge_clusters 一致：用「移動後用官方 boundary 判定式
    重算，違規數不能超過移動前」當安全閘門，保證不會把不重疊變成重疊、也不
    會讓已經滿足的 boundary soft constraint 變得更差。

    v4.7：加入第三道閘門——若提供 W_int/p2b_edges/pins_pos，移動前後的總
    HPWL（B2B + P2B）不能增加超過 `hpwl_slack_ratio * avg_side`（預設 0，
    即完全不能增加）才會套用。這是為了修正 v4.5 時期發現的問題：直接放大
    「往 cluster 靠攏」的力道（放置時貼靠偏好、或這個函式早期沒有 HPWL
    閘門的版本）在真實資料上都會讓 grouping 的改善以 HPWL 變差為代價，
    因為剛體移動常常會連帶拖走跟其他 block 有真實 B2B/P2B 接線的成員。

    `hpwl_slack_ratio=0`（嚴格不變差）用獨立取樣的 100 樣本 A/B 測試時，
    V_grouping 幾乎沒有改善（374→375），但後來改用「paired」設計重測
    （同一組 diffusion 輸出餵給不同 legalize 設定，排除掉 diffusion 取樣
    本身的隨機性這個干擾源）才發現訊號其實一直都在，只是先前的獨立取樣
    雜訊量級跟訊號差不多大，蓋掉了它：paired 測試下 slack=0 讓 V_grouping
    378→371（100 樣本中沒有任何一個變差、6 個變好），而放寬到
    `hpwl_slack_ratio=5.0`（5 個 `avg_side`，block 平均邊長，等於是
    「一個 block 身位」的自然尺度）讓改善幅度增為 378→349（沒有任何一個
    變差、24 個變好），area_gap 100 樣本中只有 1 個有極微小變化，平均
    hpwl_gap 代價僅 +0.16%（同一 paired 集合上量出來）。換算官方 cost
    公式（ALPHA=0.5, BETA=2.0）粗估，slack=5.0 的淨效益（約 -1.3%）比
    slack=0（約 -0.4%）更好，因此採用 `hpwl_slack_ratio=5.0` 當預設。

    這個閾值仍然是一個經驗性的折衷，不是從官方 cost 公式精確反推出來的——
    contest 的 HPWL_gap/Area_gap 需要 GT optimal 值才能算，而 legalize 在
    真實推論時拿不到 GT，所以無法在執行當下精確計算「這步移動對最終分數的
    淨影響」，只能用這個長度尺度的代理閾值去近似「值不值得」。
    """
    x = np.array(x, dtype=np.float64).copy()
    y = np.array(y, dtype=np.float64).copy()
    w = np.asarray(w, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    k = len(x)
    if k < 2:
        return x, y
    if preplaced_mask is None:
        preplaced_mask = np.zeros(k, dtype=bool)
    else:
        preplaced_mask = np.asarray(preplaced_mask, dtype=bool)
    if boundary_code is None:
        boundary_code = np.zeros(k, dtype=np.int64)
    else:
        boundary_code = np.asarray(boundary_code, dtype=np.int64)
    if cluster_group is None:
        return x, y
    cluster_group = np.asarray(cluster_group, dtype=np.int64)
    if not (cluster_group > 0).any():
        return x, y

    avg_side = float(np.mean(np.sqrt(w * h))) if k else 1.0
    touch_tol = max(touch_tol_ratio * avg_side, 1e-6)
    areas_arr = w * h
    group_ids = sorted(int(g) for g in set(cluster_group[cluster_group > 0].tolist()))

    check_hpwl = W_int is not None or (p2b_edges and pins_pos is not None)
    hpwl_slack = max(0.0, hpwl_slack_ratio) * avg_side

    def _total_hpwl(xx, yy):
        total = 0.0
        if W_int is not None:
            total += compute_hpwl_vectorized(xx, yy, w, h, W_int)
        if p2b_edges and pins_pos is not None:
            total += compute_p2b_hpwl(xx, yy, w, h, p2b_edges, pins_pos)
        return total

    for _round in range(rounds):
        xr = x + w; yt = y + h
        gx = np.maximum(0.0, np.maximum(x[:, None] - xr[None, :], x[None, :] - xr[:, None]))
        gy = np.maximum(0.0, np.maximum(y[:, None] - yt[None, :], y[None, :] - yt[:, None]))
        gap_mat = np.sqrt(gx * gx + gy * gy)
        adj = gap_mat <= touch_tol
        np.fill_diagonal(adj, False)

        comp_id = -np.ones(k, dtype=int)
        n_comp = 0
        for s in range(k):
            if comp_id[s] != -1:
                continue
            stack = [s]
            comp_id[s] = n_comp
            while stack:
                u = stack.pop()
                for v in np.nonzero(adj[u])[0]:
                    if comp_id[v] == -1:
                        comp_id[v] = n_comp
                        stack.append(v)
            n_comp += 1

        moved_any = False
        for gid in group_ids:
            members = np.nonzero(cluster_group == gid)[0].tolist()
            if len(members) < 2:
                continue

            # 用跟 compute_cluster_violations 完全一樣的連通性定義找子分量：
            # 只看組員彼此是否直接共邊（_blocks_share_edge），不透過非成員
            # block 搭橋、也不是「距離容忍」的鬆散定義——上面 comp_id 用的
            # 全域 touching graph 只要有一條路徑貼合（可以繞過非成員 block）
            # 就算同一分量，在緊密排布的 layout 裡幾乎所有 block 常常都連在
            # 一起，會誤判「已經滿足」，但官方指標並不吃這種間接連通。
            seen = set()
            subcomps = []
            for start in members:
                if start in seen:
                    continue
                stack = [start]; seen.add(start); cur = [start]
                while stack:
                    u = stack.pop()
                    for v in members:
                        if v not in seen and _blocks_share_edge(x, y, w, h, u, v):
                            seen.add(v); stack.append(v); cur.append(v)
                subcomps.append(cur)

            if len(subcomps) <= 1:
                continue   # 已經是一個連通分量，滿足 constraint

            subcomps.sort(key=lambda ms: -float(areas_arr[ms].sum()))
            t_members_idx = np.array(subcomps[0], dtype=int)

            for sat_members in subcomps[1:]:
                s_members_idx = np.array(sat_members, dtype=int)

                # 找「最近的一對 (衛星組員 a, target 組員 b)」，算出 4 種
                # 「a 精確貼齊 b 的某一邊、且另一軸置中對齊（保證有重疊
                # margin）」候選位移，逐一驗證是否安全，選第一個可行的整段
                # 套用——套用後 a、b 保證真的共邊，不是只有比較近。
                best_pair = None
                best_d2 = None
                for a in s_members_idx:
                    for b in t_members_idx:
                        gxg = max(0.0, x[a] - (x[b] + w[b]), x[b] - (x[a] + w[a]))
                        gyg = max(0.0, y[a] - (y[b] + h[b]), y[b] - (y[a] + h[a]))
                        d2 = gxg * gxg + gyg * gyg
                        if best_d2 is None or d2 < best_d2:
                            best_d2 = d2; best_pair = (int(a), int(b))
                a, b = best_pair
                if preplaced_mask[a]:
                    continue   # a 本身位置固定，跳過

                # comp_mask：要整塊一起移動的剛體範圍。正常情況下是「a 所在的
                # 全域 touching component」（把跟 a 已經貼合的其他 block 一起
                # 搬，避免拆散既有的合法貼合關係）；但如果 a、b 剛好已經在
                # 同一個全域 component 裡（例如透過非成員 block 間接相連，
                # 但兩個組員本身沒有直接共邊——這正是前面 comp_id 判斷會誤判
                # 的那種情況），這時不能整塊移動（等於要把一個剛體往它自己
                # 身上搬），改成只搬 a 自己（把它從目前位置「拔出來」平移到
                # 新位置，其餘 block 留在原地）。
                if comp_id[a] == comp_id[b] or preplaced_mask[comp_id == comp_id[a]].any():
                    comp_mask = np.zeros(k, dtype=bool)
                    comp_mask[a] = True
                else:
                    comp_mask = comp_id == comp_id[a]
                others_mask = ~comp_mask

                def _apply(dx, dy, t):
                    xt = x.copy(); yt2 = y.copy()
                    xt[comp_mask] = x[comp_mask] + dx * t
                    yt2[comp_mask] = y[comp_mask] + dy * t
                    return xt, yt2

                def _overlap_bad(xt, yt2):
                    if not others_mask.any():
                        return False
                    gx_, gy_, gw_, gh_ = xt[comp_mask], yt2[comp_mask], w[comp_mask], h[comp_mask]
                    ox_, oy_, ow_, oh_ = xt[others_mask], yt2[others_mask], w[others_mask], h[others_mask]
                    ovx = np.minimum(gx_[:, None] + gw_[:, None], ox_[None, :] + ow_[None, :]) - \
                          np.maximum(gx_[:, None], ox_[None, :])
                    ovy = np.minimum(gy_[:, None] + gh_[:, None], oy_[None, :] + oh_[None, :]) - \
                          np.maximum(gy_[:, None], oy_[None, :])
                    return bool(((ovx > 1e-9) & (ovy > 1e-9)).any())

                baseline_v = compute_boundary_violations(x, y, w, h, boundary_code)
                # comp_mask 在「整塊剛體」情況下可能連帶了其他 cluster group
                # 的成員（同一個物理貼合團塊，未必只服務這一個 group）——只顧
                # 這個 group 的貼合、不管總體，可能把 comp_mask 裡順帶的其他
                # group 成員拖離了「它們自己」的 target，V_grouping 總和不降
                # 反升。用「移動後全體 V_grouping 總和不能變差」再加一道閘門，
                # 而不是只看這個 group 自己的貼合有沒有成立。
                baseline_cluster_v = compute_cluster_violations(x, y, w, h, cluster_group)
                # 第三道閘門（v4.7）：總 HPWL 不能變差，見函式 docstring。
                baseline_hpwl = _total_hpwl(x, y) if check_hpwl else 0.0

                def _feasible_full(dx, dy):
                    xt, yt2 = _apply(dx, dy, 1.0)
                    if _overlap_bad(xt, yt2):
                        return False
                    if compute_boundary_violations(xt, yt2, w, h, boundary_code) > baseline_v:
                        return False
                    if compute_cluster_violations(xt, yt2, w, h, cluster_group) > baseline_cluster_v:
                        return False
                    if check_hpwl and _total_hpwl(xt, yt2) > baseline_hpwl + hpwl_slack + 1e-6:
                        return False
                    return True

                ax0, ay0, aw_, ah_ = float(x[a]), float(y[a]), float(w[a]), float(h[a])
                bx0, by0, bw_, bh_ = float(x[b]), float(y[b]), float(w[b]), float(h[b])
                bcx, bcy = bx0 + bw_ / 2.0, by0 + bh_ / 2.0
                candidates = [
                    ((bx0 + bw_) - ax0, bcy - (ay0 + ah_ / 2.0)),          # 貼 b 右邊
                    (bx0 - (ax0 + aw_), bcy - (ay0 + ah_ / 2.0)),          # 貼 b 左邊
                    (bcx - (ax0 + aw_ / 2.0), (by0 + bh_) - ay0),          # 貼 b 上邊
                    (bcx - (ax0 + aw_ / 2.0), by0 - (ay0 + ah_)),          # 貼 b 下邊
                ]
                # 依所需移動距離由小到大嘗試，優先選最省力（最少擾動其他佈局）
                # 的可行貼合方式。
                candidates.sort(key=lambda d: d[0] * d[0] + d[1] * d[1])

                moved = False
                for dx, dy in candidates:
                    if abs(dx) + abs(dy) < 1e-9:
                        continue
                    if _feasible_full(dx, dy):
                        xt, yt2 = _apply(dx, dy, 1.0)
                        x[:] = xt; y[:] = yt2
                        moved = True
                        break

                if moved:
                    moved_any = True
                    if verbose:
                        print("compact_merge_cluster_groups: round={} group={} block {} "
                              "abutted to block {} (moved {} block(s))".format(
                                  _round, gid, a, b, int(comp_mask.sum())))

        if not moved_any:
            break

    return x, y


def hard_zero_overlap(x, y, w, h, preplaced_indices=None, tol=1e-6, margin=1e-4, max_iters=5000):
    """
    保證輸出通過官方 check_overlap（iccad2026_evaluate.py）的逐 pair、逐軸判定：
        violation ⟺ overlap_x > tol AND overlap_y > tol      (官方 tol = 1e-6)

    跟 legalize() 的差別：legalize() 用「全體重疊面積總和 < 1e-10」當收斂條件，
    這不等價於官方判定——一個「薄片」pair（例如 overlap_x 很小但 overlap_y 很大）
    面積乘積可能已經 < 1e-10，但 overlap_x 本身仍然 > 1e-6，官方仍判定為違規。
    這裡直接照官方判定式本身收斂（見 _resolve_overlaps_iterative）。

    只動 x, y（不碰 w, h），soft-block 的面積 hard constraint 不受影響。
    """
    x = np.array(x, dtype=np.float64).copy()
    y = np.array(y, dtype=np.float64).copy()
    w = np.asarray(w, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    k = len(x)
    frozen = set(preplaced_indices or [])
    iu = np.triu_indices(k, k=1)   # 上三角 pair 索引，(i,j) i<j，eject fallback 會用

    x, y = _resolve_overlaps_iterative(x, y, w, h, frozen, tol=tol,
                                       margin=margin, max_iters=max_iters)

    # ---- 保底 fallback ----
    # 極少數密集重疊的樣本可能在 max_iters 內還沒完全收斂（隨機 diffusion 噪聲
    # 導致的重疊嚴重度不定，難度也不定）。為了保證 100% 不出現 hard constraint
    # 違規，掃一次剩餘違規，把「可動的一方」直接搬到目前全域範圍右側、逐一堆疊
    # 的位置——這個位置跟其他所有 block 都不可能重疊，犧牲那一個 block 的
    # HPWL/位置品質換取保證零 overlap（正常情況下不會觸發，只是保險絲）。
    remaining = []
    if k >= 2:
        xr = x + w; yt = y + h
        ox_mat = np.minimum(xr[:, None], xr[None, :]) - np.maximum(x[:, None], x[None, :])
        oy_mat = np.minimum(yt[:, None], yt[None, :]) - np.maximum(y[:, None], y[None, :])
        viol_u = (ox_mat[iu] > tol) & (oy_mat[iu] > tol)
        remaining = [(int(iu[0][t]), int(iu[1][t])) for t in np.nonzero(viol_u)[0]]
    if remaining:
        eject_x = float((x + w).max()) + margin * 10
        eject_y = float(y.min())
        for i, j in remaining:
            ox = min(x[i] + w[i], x[j] + w[j]) - max(x[i], x[j])
            oy = min(y[i] + h[i], y[j] + h[j]) - max(y[i], y[j])
            if not (ox > tol and oy > tol):
                continue   # 前面 eject 過程中已經順便解掉了
            target = j if j not in frozen else (i if i not in frozen else None)
            if target is None:
                continue   # 兩個都 preplaced：資料本身衝突，位置搬不動，無法解決
            x[target] = eject_x
            y[target] = eject_y
            eject_y += h[target] + margin * 10

    # ---- DEBUG（暫時）：eject 之後還有殘留就印出完整診斷資訊 ----
    if k >= 2:
        xr = x + w; yt = y + h
        ox_mat = np.minimum(xr[:, None], xr[None, :]) - np.maximum(x[:, None], x[None, :])
        oy_mat = np.minimum(yt[:, None], yt[None, :]) - np.maximum(y[:, None], y[None, :])
        viol_u = (ox_mat[iu] > tol) & (oy_mat[iu] > tol)
        still = [(int(iu[0][t]), int(iu[1][t])) for t in np.nonzero(viol_u)[0]]
        if still:
            print("!!! hard_zero_overlap DEBUG: {} residual violation(s) after eject, k={}, "
                  "frozen={}".format(len(still), k, sorted(frozen)))
            for i, j in still:
                print("    pair ({},{}) i_frozen={} j_frozen={} "
                      "ox={:.6e} oy={:.6e}".format(
                          i, j, i in frozen, j in frozen,
                          float(ox_mat[i, j]), float(oy_mat[i, j])))
                print("      block {}: x={:.6f} y={:.6f} w={:.6f} h={:.6f}".format(
                    i, x[i], y[i], w[i], h[i]))
                print("      block {}: x={:.6f} y={:.6f} w={:.6f} h={:.6f}".format(
                    j, x[j], y[j], w[j], h[j]))

    # 刻意不 downcast 成 float32：座標量級大時（大 bbox 的 sample），float32 的
    # 絕對精度（~1e-4 甚至更粗）可能吃掉這裡辛苦推開的 margin，讓官方用更高精度
    # 重新算 overlap 時「復活」一個原本已經解決掉的違規（實測發生過）。保持
    # float64 到呼叫端，只在最後寫 JSON 時才轉型，不會有精度損失風險。
    return x.astype(np.float64), y.astype(np.float64)


def compact_positions(x, y, w, h, preplaced_mask=None, boundary_code=None,
                      canvas_bbox=None, rounds=12):
    """
    往左下角壓縮、消除多餘空隙，純平移（不碰 w,h）。每個 block 只會被推到
    「剛好貼住」擋路的另一個 block 或邊界為止，絕不會穿過任何人，所以保證
    不會把不重疊的狀態變成重疊——可以安全地在 legalize pipeline 任何階段
    （包含 hard_zero_overlap 之後）呼叫，用來把它留下的空隙壓緊。

    preplaced_mask: (k,) bool，True 的不動。
    boundary_code:  (k,) int bitmask，RIGHT(2) 鎖定不做 -x 壓縮、
                    TOP(4) 鎖定不做 -y 壓縮（避免破壞貼邊位置）。
    canvas_bbox:    (xmin, ymin, xmax, ymax)，壓縮邊界；None 則用目前
                    所有 block 的 bbox 左下角。
    rounds:         最多幾輪 x/y 交替壓縮；一輪內沒有任何 block 移動就提前結束。
    """
    x = np.array(x, dtype=np.float64).copy()
    y = np.array(y, dtype=np.float64).copy()
    w = np.asarray(w, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    k = len(x)
    if preplaced_mask is None:
        preplaced_mask = np.zeros(k, dtype=bool)
    else:
        preplaced_mask = np.asarray(preplaced_mask, dtype=bool)
    if boundary_code is None:
        boundary_code = np.zeros(k, dtype=np.int64)
    else:
        boundary_code = np.asarray(boundary_code, dtype=np.int64)

    if canvas_bbox is None:
        xmin_canvas = float(x.min()) if k else 0.0
        ymin_canvas = float(y.min()) if k else 0.0
    else:
        xmin_canvas, ymin_canvas = float(canvas_bbox[0]), float(canvas_bbox[1])

    for _ in range(rounds):
        moved = False

        order_x = sorted(range(k), key=lambda i: x[i])
        for i in order_x:
            if preplaced_mask[i] or (int(boundary_code[i]) & 2):
                continue
            xi0, yi0, yi1 = x[i], y[i], y[i] + h[i]
            max_left = xmin_canvas
            for j in range(k):
                if j == i:
                    continue
                yj0, yj1 = y[j], y[j] + h[j]
                if min(yi1, yj1) - max(yi0, yj0) <= 1e-9:
                    continue   # y 不重疊，不擋路
                xj1 = x[j] + w[j]
                if xj1 <= xi0 + 1e-9 and xj1 > max_left:
                    max_left = xj1
            new_x = max(max_left, xmin_canvas)
            if new_x < x[i] - 1e-9:
                x[i] = new_x
                moved = True

        order_y = sorted(range(k), key=lambda i: y[i])
        for i in order_y:
            if preplaced_mask[i] or (int(boundary_code[i]) & 4):
                continue
            xi0, yi0, xi1 = x[i], y[i], x[i] + w[i]
            max_down = ymin_canvas
            for j in range(k):
                if j == i:
                    continue
                xj0, xj1 = x[j], x[j] + w[j]
                if min(xi1, xj1) - max(xi0, xj0) <= 1e-9:
                    continue
                yj1 = y[j] + h[j]
                if yj1 <= yi0 + 1e-9 and yj1 > max_down:
                    max_down = yj1
            new_y = max(max_down, ymin_canvas)
            if new_y < y[i] - 1e-9:
                y[i] = new_y
                moved = True

        if not moved:
            break

    return x, y


# ============================================================
# Legalization v3: LFF 風格的自由矩形（MAXRECTS）決定性單趟排布
# ============================================================
#
# 跟 legalize_v2（grid search + 一堆事後補丁）完全不同的路線，改參考：
#   - LFF (Less Flexibility First)：決定性、單趟、原生支援 fixed-outline，
#     不需要 SA 或任何 floorplan representation。
#     (Deterministic VLSI Block Placement Algorithm Using Less Flexibility
#      First Principle; On handling fixed-outline constraints using LFF)
#   - MAXRECTS / guillotine free-rectangle 管理（2D bin packing 文獻常見手法）：
#     維護「目前所有還沒被佔用的最大矩形」列表，每放一個 block 就把它佔用的
#     區域從對應的自由矩形切掉，保證任何時刻已放置的 block 之間絕不重疊
#     ——重疊在演算法結構上就不可能發生，不需要事後清零。
#   - 每個 block 在候選自由矩形內的最佳位置，用「加權中位數」直接解閉式解
#     （anchor 距離、boundary 目標、cluster 重心、b2b/p2b wirelength 都是
#     L1 距離，可以合併成一個加權中位數問題），取代 legalize_v2 的網格掃描
#     ——不用窮舉候選點，O(自由矩形數) 而不是 O(半徑^2)。
#
# 整體流程：
#   1. preplaced block 先從自由矩形挖掉（當固定障礙物）
#   2. 其餘 block 依「限制越多越先放」排序（LFF 精神）：boundary > cluster
#      > MIB > 一般，同一層用面積大的優先（越大越難之後才找到位置）
#   3. 每個 block：對每個候選形狀（原形狀優先，必要時才試其他 aspect），
#      在每個「裝得下」的自由矩形內用加權中位數求最佳位置，取全域最低 cost
#      的 (矩形, 位置, 形狀) 組合放下去，然後用 MAXRECTS 規則切掉該區域
#   4. MIB 群組事後盡量統一尺寸（不破壞已放置的不重疊狀態才套用）
#   5. 最終仍呼叫 hard_zero_overlap 當零成本的保底驗證（照理論上一開始就
#      保證零重疊，這一步只是防禦性雙重確認，不是靠它來清重疊）
# ============================================================

def _rects_intersect(a, b, tol=1e-9):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 - tol and bx0 < ax1 - tol and ay0 < by1 - tol and by0 < ay1 - tol


def _rect_contains(a, b, tol=1e-9):
    """a 是否完全包含 b。"""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 <= bx0 + tol and ay0 <= by0 + tol and ax1 >= bx1 - tol and ay1 >= by1 - tol


class _FreeRectPool:
    """
    維護目前所有「最大自由矩形」（MAXRECTS）。每放一個 block，就把它佔用的
    範圍從所有跟它相交的自由矩形切掉，切法是標準的 guillotine leftover 切法
    （對每個相交的自由矩形，把「自由矩形扣掉被佔用範圍」的 leftover 拆成
    上下左右最多 4 塊最大矩形），事後剔除被其他矩形完全包含的重複矩形。
    """

    def __init__(self, x0, y0, x1, y1):
        self.rects = [(x0, y0, x1, y1)]

    def candidates(self, w, h, tol=1e-7):
        return [r for r in self.rects if (r[2] - r[0]) >= w - tol and (r[3] - r[1]) >= h - tol]

    def occupy(self, x, y, w, h):
        px0, py0, px1, py1 = x, y, x + w, y + h
        placed = (px0, py0, px1, py1)
        new_rects = []
        for r in self.rects:
            if not _rects_intersect(r, placed):
                new_rects.append(r)
                continue
            fx0, fy0, fx1, fy1 = r
            if fx0 < px0:
                new_rects.append((fx0, fy0, px0, fy1))
            if px1 < fx1:
                new_rects.append((px1, fy0, fx1, fy1))
            if fy0 < py0:
                new_rects.append((fx0, fy0, fx1, py0))
            if py1 < fy1:
                new_rects.append((fx0, py1, fx1, fy1))
        # 剔除面積過小、以及被別的矩形完全包含的重複矩形
        new_rects = [r for r in new_rects if (r[2] - r[0]) > 1e-6 and (r[3] - r[1]) > 1e-6]
        pruned = []
        for i, r in enumerate(new_rects):
            if any(j != i and _rect_contains(r2, r) and not _rect_contains(r, r2)
                   for j, r2 in enumerate(new_rects)):
                continue
            if any(j < i and r2 == r for j, r2 in enumerate(new_rects)):
                continue
            pruned.append(r)
        self.rects = pruned


def _weighted_median_1d(targets, weights, lo, hi):
    """
    求 x* = argmin_x sum(weight_i * |x - target_i|)，再夾到 [lo, hi]。
    這是加權中位數問題的標準解法：依 target 排序，累加權重找到過半的那個
    target 就是無限制下的最佳解；L1 目標函數是凸的，所以「先求無限制最佳解
    再夾進可行區間」等價於真正有限制下的最佳解。
    """
    if not targets:
        return (lo + hi) / 2.0
    order = sorted(range(len(targets)), key=lambda i: targets[i])
    total = sum(weights[i] for i in order)
    if total <= 0:
        return min(max((lo + hi) / 2.0, lo), hi)
    half = total / 2.0
    cum = 0.0
    med = targets[order[-1]]
    for i in order:
        cum += weights[i]
        if cum >= half - 1e-9:
            med = targets[i]
            break
    return min(max(med, lo), hi)


def _l1_cost(pos, targets, weights):
    return sum(w_ * abs(pos - t_) for t_, w_ in zip(targets, weights))


def legalize_lff(
    x_init, y_init, w_init, h_init,
    areas,
    preplaced_mask,        # (k,) bool，位置+形狀都不可動
    fixed_mask,            # (k,) bool，形狀不可動（preplaced 也算）
    mib_group=None,
    cluster_group=None,
    boundary_code=None,
    outline_bbox=None,     # (xmin, ymin, xmax, ymax)，硬性範圍（例如 pin bbox）
    W_int=None,
    p2b_edges=None,
    pins_pos=None,
    weight_dist=1.0,
    weight_boundary=3.0,
    weight_cluster=1.0,
    weight_b2b=0.5,
    weight_p2b=0.15,
    weight_shape=3.0,
    use_cluster_adjacency=False,     # 實驗用：放置時額外嘗試「直接貼齊已放置同組成員」候選位置
    cluster_adjacency_bonus=5.0,     # 貼合候選的成本折扣（乘 weight_cluster * avg_side）
    allow_reshape=True,
    use_reinsert=True,
    reinsert_sweeps=3,
    reinsert_grid_density=12,
    use_pair_reinsert=False,   # 實驗用：見 compact_pair_reinsert 呼叫處說明
    pair_reinsert_sweeps=2,    # 獨立於 reinsert_sweeps，時間成本較高，預設保守一點
    pair_reinsert_grid_density=8,
    pair_reinsert_hpwl_slack_ratio=0.0,
    use_reinsert_reshape=False,   # 實驗用：見 compact_reinsert_reshape 呼叫處說明
    reinsert_reshape_sweeps=2,
    reinsert_reshape_grid_density=8,
    # v5.2（實驗、最終停用）：一度因為修正 warmup 量測誤差＋稀疏化後看起來
    # 有淨改善而短暫改為預設開啟，但後來發現 compact_gradient_finetune 的
    # loss 對「整體平移」完全不變（overlap/area/boundary-相對自己 bbox/
    # cluster/HPWL 五項都是平移不變量），導致 Adam 在這個零梯度方向上隨機
    # 漂移，把 block 帶出 outline_bbox 之外而不被發覺——連當初拿來當展示
    # 案例的 -4.54% 那筆，事後用正確的 outline 檢查一查也是越界的無效解。
    # 補上 weight_anchor 錨定項＋outline 硬 gate 修好這個 bug 之後，同一批
    # 30 樣本重新量測變成 0/30 有真正改善，額外時間成本卻還在（約 +0.43
    # 秒／樣本）——整個機制的「效益」原來大部分是這個 bug 的假象。因此改回
    # 預設關閉；compact_gradient_finetune 與其 outline-safety gate 邏輯保留
    # 在 utils.py 內備用，之後如果想再嘗試（例如換一種平移錨定方式、或用
    # 不同的優化器設定去找回真正的改善），程式碼架構已經在了。
    use_gradient_finetune=False,
    gradient_finetune_steps=400,
    gradient_finetune_lr=0.5,
    gradient_finetune_patience=30,
    gradient_finetune_hpwl_slack_ratio=0.0,
    tie_break_mode="area_desc",   # 'area_desc' | 'area_asc' | 'flexibility'（實驗用）
    # v4.4 實驗：100 樣本 A/B 顯示 compact_gravity 對 area_gap/hpwl_gap/
    # V_relative 都是輕微負面（且不會修到「多個獨立衛星群共享同一條被鎖死
    # 邊界」這種情況——見 legalize_lff 呼叫處的說明），維持預設關閉。
    use_gravity=False,
    gravity_iters=40,
    # v4.7: 100 樣本 paired A/B（同一組起始佈局餵給不同設定，排除 diffusion
    # 取樣雜訊這個干擾源）驗證後改為預設開啟，hpwl_slack_ratio=5.0（見
    # compact_merge_cluster_groups docstring 與呼叫處說明）：area_gap
    # 100/100 樣本零變化、V_grouping 從未在任何樣本變差、29/100 樣本變好，
    # 平均 hpwl_gap 代價僅 +0.16%。
    use_cluster_merge=True,
    hpwl_slack_ratio=5.0,
    use_snap_boundary=False,   # 實驗用：見 compact_snap_boundary 呼叫處說明
    boundary_hpwl_slack_ratio=0.0,   # 實驗用：見 compact_snap_boundary docstring
    # v4.6: 100 樣本 A/B 驗證後改為預設開啟。compact_reinsert 的局部搜尋常常
    # 開出新的「彼此貼合」機會，第一次 compact_merge_clusters（在 reinsert
    # 之前）看不到——補跑第二次讓 area_gap 24.6%→23.1%、V_relative 也同步
    # 改善（0.112→0.106），且幾乎沒有時間成本（compact_merge_clusters 本身
    # 在沒有東西可移動時第一輪就會立刻收斂）。見下方呼叫處說明。
    use_second_merge_pass=True,
    verbose=False,
):
    """
    LFF 風格、決定性單趟的 legalization：不用 SA、不用 legalize_v2，靠
    MAXRECTS 自由矩形管理保證重疊在結構上就不可能發生，靠加權中位數求每個
    候選矩形內的最佳位置（取代網格掃描），靠 outline_bbox 原生限制所有
    block 落在指定範圍內（例如 pin bbox）。

    回傳 (x, y, w, h)，皆為 (k,) float64，保證：
      - 兩兩不重疊
      - preplaced block 位置/形狀完全等於輸入
      - fixed block 形狀完全等於輸入
      - 其餘 block 面積跟輸入完全相等（reshape 只改長寬比）
      - 盡量都落在 outline_bbox 內（見下方 outline 不足時的處理）
    """
    k = len(x_init)
    x = np.array(x_init, dtype=np.float64).copy()
    y = np.array(y_init, dtype=np.float64).copy()
    w = np.array(w_init, dtype=np.float64).copy()
    h = np.array(h_init, dtype=np.float64).copy()
    areas = np.array(areas, dtype=np.float64)
    preplaced_mask = np.array(preplaced_mask, dtype=bool)
    fixed_mask = np.array(fixed_mask, dtype=bool)
    if mib_group is None: mib_group = np.zeros(k, dtype=np.int64)
    if cluster_group is None: cluster_group = np.zeros(k, dtype=np.int64)
    if boundary_code is None: boundary_code = np.zeros(k, dtype=np.int64)
    mib_group = np.array(mib_group, dtype=np.int64)
    cluster_group = np.array(cluster_group, dtype=np.int64)
    boundary_code = np.array(boundary_code, dtype=np.int64)

    # ---- outline：預設用 raw bbox；若給定 outline_bbox（如 pin bbox），
    # 「先原原本本試一次給定的 outline」，只有真的塞不下（有 block 落入
    # fallback）才逐步放大重試——盡量把 block 留在使用者要求的範圍內，不要
    # 還沒試就先入為主放大（實測 pin bbox 常常剛好只比總面積寬裕一點點，
    # 提前放大只會不必要地讓 block 跑到框外）。
    raw_xmin = float(x.min()); raw_xmax = float((x + w).max())
    raw_ymin = float(y.min()); raw_ymax = float((y + h).max())
    if outline_bbox is None:
        outline_bbox = (raw_xmin, raw_ymin, raw_xmax, raw_ymax)
    base_oxmin, base_oymin, base_oxmax, base_oymax = outline_bbox
    # outline 至少要涵蓋所有 preplaced block（否則連固定的都放不進去）
    pp_idx = np.nonzero(preplaced_mask)[0]
    if len(pp_idx) > 0:
        base_oxmin = min(base_oxmin, float(x[pp_idx].min()))
        base_oymin = min(base_oymin, float(y[pp_idx].min()))
        base_oxmax = max(base_oxmax, float((x[pp_idx] + w[pp_idx]).max()))
        base_oymax = max(base_oymax, float((y[pp_idx] + h[pp_idx]).max()))
    base_ccx = (base_oxmin + base_oxmax) / 2.0
    base_ccy = (base_oymin + base_oymax) / 2.0
    base_half_w = (base_oxmax - base_oxmin) / 2.0
    base_half_h = (base_oymax - base_oymin) / 2.0

    avg_side = float(np.mean(np.sqrt(areas))) if k else 1.0

    # ---- tie-break（同一優先權層內的排序次序）----
    # 'area_desc'（預設，原行為）：大 block 先放。
    # 'area_asc'：小 block 先放，把大 block 留到最後填剩餘空間。
    # 'flexibility'：真正的「Less Flexibility First」——先算每個 block 在
    # 初始（只扣掉 preplaced）自由矩形池裡有幾個候選矩形放得下，數量越少
    # （越難放）優先權越高，數量相同時退回 area_desc。用初始 outline 算一次
    # 當近似值，不隨後續實際排布動態更新（避免大幅增加複雜度）。
    if tie_break_mode == "flexibility":
        _flex_pool = _FreeRectPool(base_oxmin, base_oymin, base_oxmax, base_oymax)
        for i in range(k):
            if preplaced_mask[i]:
                _flex_pool.occupy(x[i], y[i], w[i], h[i])
        flex_count = np.array([len(_flex_pool.candidates(w[i], h[i])) for i in range(k)])

    def _tie_key(i):
        if tie_break_mode == "area_asc":
            return (areas[i], 0.0)
        if tie_break_mode == "flexibility":
            return (float(flex_count[i]), -areas[i])
        return (-areas[i], 0.0)   # area_desc（預設）

    def _priority(i):
        if boundary_code[i] > 0:
            bits = bin(int(boundary_code[i])).count("1")
            return (0, -bits) + _tie_key(i)
        if cluster_group[i] > 0:
            return (1, 0) + _tie_key(i)
        if mib_group[i] > 0:
            return (2, 0) + _tie_key(i)
        return (3, 0) + _tie_key(i)

    pending = [i for i in range(k) if not preplaced_mask[i]]
    pending.sort(key=_priority)

    b2b_adj = [[] for _ in range(k)]
    if W_int is not None:
        W_arr = np.asarray(W_int)
        for i in range(k):
            for j in range(k):
                if i != j and W_arr[i, j] > 0:
                    b2b_adj[i].append((j, float(W_arr[i, j])))
    p2b_adj = [[] for _ in range(k)]
    if p2b_edges is not None and pins_pos is not None:
        pins_arr = np.asarray(pins_pos, dtype=np.float64)
        for edge in p2b_edges:
            p_idx, b_idx, w_e = int(edge[0]), int(edge[1]), float(edge[2])
            if 0 <= p_idx < len(pins_arr) and 0 <= b_idx < k:
                p2b_adj[b_idx].append((pins_arr[p_idx, 0], pins_arr[p_idx, 1], w_e))

    def _attempt(oxmin, oymin, oxmax, oymax):
        """單趟嘗試在給定 outline 內排完所有 block；回傳 (x,y,w,h,n_fallback,n_reshape)。"""
        xa = x.copy(); ya = y.copy(); wa = w.copy(); ha = h.copy()
        pool = _FreeRectPool(oxmin, oymin, oxmax, oymax)
        placed_mask = np.zeros(k, dtype=bool)
        for i in range(k):
            if preplaced_mask[i]:
                pool.occupy(xa[i], ya[i], wa[i], ha[i])
                placed_mask[i] = True

        cluster_sum = {}

        def _update_cluster(i):
            gid = int(cluster_group[i])
            if gid <= 0:
                return
            cx, cy = xa[i] + wa[i] / 2.0, ya[i] + ha[i] / 2.0
            if gid in cluster_sum:
                s = cluster_sum[gid]; s[0] += cx; s[1] += cy; s[2] += 1
            else:
                cluster_sum[gid] = [cx, cy, 1]

        for i in range(k):
            if placed_mask[i]:
                _update_cluster(i)

        n_reshape = 0
        n_fallback = 0
        cur_oymax = oymax   # 極端 fallback 分支可能往上疊放，需要動態擴張

        for i in pending:
            orig_w, orig_h = wa[i], ha[i]
            orig_log_r = float(np.log(orig_w / max(orig_h, 1e-9)))
            anchor_cx = xa[i] + orig_w / 2.0
            anchor_cy = ya[i] + orig_h / 2.0

            if fixed_mask[i] or not allow_reshape:
                shape_candidates = [(orig_w, orig_h)]
            else:
                area_i = areas[i]
                ratios = [orig_w / max(orig_h, 1e-9), 0.5, 2.0, 0.25, 4.0]
                seen = set(); shape_candidates = []
                for r in ratios:
                    r = max(min(r, 12.0), 1.0 / 12.0)
                    key = round(r, 6)
                    if key in seen:
                        continue
                    seen.add(key)
                    shape_candidates.append(((area_i * r) ** 0.5, (area_i / r) ** 0.5))

            b2b_placed = [(j, w_e) for (j, w_e) in b2b_adj[i] if placed_mask[j]]
            p2b_list = p2b_adj[i]
            clu_gid = int(cluster_group[i])
            clu_target = None
            clu_members_placed = []
            if clu_gid > 0 and clu_gid in cluster_sum:
                s = cluster_sum[clu_gid]
                clu_target = (s[0] / s[2], s[1] / s[2])
                if use_cluster_adjacency:
                    clu_members_placed = [j for j in range(k)
                                          if placed_mask[j] and int(cluster_group[j]) == clu_gid]

            best = None
            for (w_use, h_use) in shape_candidates:
                cands = pool.candidates(w_use, h_use)
                if not cands:
                    continue
                shape_dev = abs(float(np.log(w_use / max(h_use, 1e-9))) - orig_log_r)
                shape_pen = weight_shape * avg_side * shape_dev

                targets_x = [anchor_cx]; weights_x = [weight_dist]
                targets_y = [anchor_cy]; weights_y = [weight_dist]
                if boundary_code[i] & 1:
                    targets_x.append(oxmin + w_use / 2.0); weights_x.append(weight_boundary)
                if boundary_code[i] & 2:
                    targets_x.append(oxmax - w_use / 2.0); weights_x.append(weight_boundary)
                if boundary_code[i] & 4:
                    targets_y.append(oymax - h_use / 2.0); weights_y.append(weight_boundary)
                if boundary_code[i] & 8:
                    targets_y.append(oymin + h_use / 2.0); weights_y.append(weight_boundary)
                if clu_target is not None:
                    targets_x.append(clu_target[0]); weights_x.append(weight_cluster)
                    targets_y.append(clu_target[1]); weights_y.append(weight_cluster)
                for (j, w_e) in b2b_placed:
                    targets_x.append(xa[j] + wa[j] / 2.0); weights_x.append(weight_b2b * w_e)
                    targets_y.append(ya[j] + ha[j] / 2.0); weights_y.append(weight_b2b * w_e)
                for (px, py, w_e) in p2b_list:
                    targets_x.append(px); weights_x.append(weight_p2b * w_e)
                    targets_y.append(py); weights_y.append(weight_p2b * w_e)

                for rect in cands:
                    rx0, ry0, rx1, ry1 = rect
                    cx_lo, cx_hi = rx0 + w_use / 2.0, rx1 - w_use / 2.0
                    cy_lo, cy_hi = ry0 + h_use / 2.0, ry1 - h_use / 2.0
                    if cx_hi < cx_lo or cy_hi < cy_lo:
                        continue
                    bx = _weighted_median_1d(targets_x, weights_x, cx_lo, cx_hi)
                    by = _weighted_median_1d(targets_y, weights_y, cy_lo, cy_hi)
                    cost = _l1_cost(bx, targets_x, weights_x) + _l1_cost(by, targets_y, weights_y) + shape_pen
                    if best is None or cost < best[0] - 1e-9:
                        best = (cost, bx - w_use / 2.0, by - h_use / 2.0, w_use, h_use)

                    # RulePlanner 風格（見 method.md 參考文獻）：加權中位數只會
                    # 把 block 拉到「離同組已放置成員最近」，幾乎不可能剛好落在
                    # 「真的共邊」的位置——而 V_grouping 只認共邊，不認距離。
                    # 這裡對每個已放置的同組成員，額外算出「精確貼齊它某一邊」
                    # 的候選位置（若落在目前候選矩形範圍內），給予貼合成本
                    # 折扣，讓「真的共邊」在候選比較中贏過「只是比較近」。
                    if clu_members_placed:
                        bonus = cluster_adjacency_bonus * weight_cluster * avg_side
                        for j in clu_members_placed:
                            jx0, jy0, jw_, jh_ = xa[j], ya[j], wa[j], ha[j]
                            jcx, jcy = jx0 + jw_ / 2.0, jy0 + jh_ / 2.0
                            for (abx, aby) in (
                                (jx0 + jw_ + w_use / 2.0, jcy),
                                (jx0 - w_use / 2.0, jcy),
                                (jcx, jy0 + jh_ + h_use / 2.0),
                                (jcx, jy0 - h_use / 2.0),
                            ):
                                if not (cx_lo - 1e-6 <= abx <= cx_hi + 1e-6 and
                                        cy_lo - 1e-6 <= aby <= cy_hi + 1e-6):
                                    continue
                                abx_c = min(max(abx, cx_lo), cx_hi)
                                aby_c = min(max(aby, cy_lo), cy_hi)
                                abut_cost = (_l1_cost(abx_c, targets_x, weights_x) +
                                            _l1_cost(aby_c, targets_y, weights_y) +
                                            shape_pen - bonus)
                                if best is None or abut_cost < best[0] - 1e-9:
                                    best = (abut_cost, abx_c - w_use / 2.0,
                                           aby_c - h_use / 2.0, w_use, h_use)

            if best is None:
                n_fallback += 1
                if pool.rects:
                    rect = max(pool.rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
                    rx0, ry0, rx1, ry1 = rect
                    rw, rh = rx1 - rx0, ry1 - ry0
                    area_i = areas[i]
                    if fixed_mask[i]:
                        w_use, h_use = orig_w, orig_h
                    else:
                        w_use = min(rw, max(area_i / max(rh, 1e-9), area_i ** 0.5 / 12.0))
                        w_use = max(min(w_use, rw), 1e-6)
                        h_use = area_i / w_use
                    if h_use > rh:
                        h_use = rh; w_use = area_i / max(h_use, 1e-9)
                    bx = min(max(anchor_cx - w_use / 2.0, rx0), rx1 - w_use)
                    by = min(max(anchor_cy - h_use / 2.0, ry0), ry1 - h_use)
                else:
                    bx = oxmin
                    by = float((ya[placed_mask] + ha[placed_mask]).max()) if placed_mask.any() else oymin
                    w_use, h_use = orig_w, orig_h
                    cur_oymax = max(cur_oymax, by + h_use)
                best = (0.0, bx, by, w_use, h_use)

            _, bx, by, w_use, h_use = best
            if abs(w_use - orig_w) > 1e-6 or abs(h_use - orig_h) > 1e-6:
                n_reshape += 1
            xa[i], ya[i], wa[i], ha[i] = bx, by, w_use, h_use
            placed_mask[i] = True
            pool.occupy(bx, by, w_use, h_use)
            _update_cluster(i)

        # ---- MIB 群組事後盡量統一尺寸（不破壞不重疊才套用；不行就跳過該組）----
        for gid in np.unique(mib_group):
            if gid == 0:
                continue
            members = [i for i in range(k) if mib_group[i] == gid]
            if len(members) < 2:
                continue
            anchors = [i for i in members if fixed_mask[i]]
            if anchors:
                tw, th = wa[anchors[0]], ha[anchors[0]]
            else:
                aspects = [wa[i] / max(ha[i], 1e-9) for i in members]
                med_r = float(np.median(aspects))
                med_area = float(np.median([areas[i] for i in members]))
                tw = float(np.sqrt(med_area * med_r))
                th = float(np.sqrt(med_area / med_r))

            ok_for_all = True
            new_xywh = []
            for i in members:
                if fixed_mask[i]:
                    if abs(wa[i] - tw) > 1e-3 or abs(ha[i] - th) > 1e-3:
                        ok_for_all = False; break
                    new_xywh.append((xa[i], ya[i], wa[i], ha[i]))
                    continue
                if abs(tw * th - areas[i]) / max(areas[i], 1e-9) > 0.009:
                    ok_for_all = False; break
                cx_i, cy_i = xa[i] + wa[i] / 2.0, ya[i] + ha[i] / 2.0
                nx, ny = cx_i - tw / 2.0, cy_i - th / 2.0
                if nx < oxmin or ny < oymin or nx + tw > oxmax or ny + th > cur_oymax:
                    ok_for_all = False; break
                others = [j for j in range(k) if j != i]
                ox_j = xa[others]; oy_j = ya[others]; ow_j = wa[others]; oh_j = ha[others]
                ovx = np.minimum(nx + tw, ox_j + ow_j) - np.maximum(nx, ox_j)
                ovy = np.minimum(ny + th, oy_j + oh_j) - np.maximum(ny, oy_j)
                if bool(((ovx > 1e-9) & (ovy > 1e-9)).any()):
                    ok_for_all = False; break
                new_xywh.append((nx, ny, tw, th))
            if ok_for_all:
                for i, (nx, ny, nw, nh) in zip(members, new_xywh):
                    xa[i], ya[i], wa[i], ha[i] = nx, ny, nw, nh

        return xa, ya, wa, ha, n_fallback, n_reshape

    # ---- 先原封不動試一次給定的 outline，塞不下（有 fallback）才逐步放大重試 ----
    max_retries = 20
    expand_factor = 1.05   # 細粒度放大，減少「其實只差一點點卻整段放大 15%」的過度溢出
    x_res = y_res = w_res = h_res = None
    n_fallback = n_reshape = 0
    oxmin = oymin = oxmax = oymax = 0.0
    for attempt in range(max_retries):
        growth = expand_factor ** attempt
        oxmin = base_ccx - base_half_w * growth
        oymin = base_ccy - base_half_h * growth
        oxmax = base_ccx + base_half_w * growth
        oymax = base_ccy + base_half_h * growth
        x_res, y_res, w_res, h_res, n_fallback, n_reshape = _attempt(oxmin, oymin, oxmax, oymax)
        if n_fallback == 0:
            break
        if verbose:
            next_growth = expand_factor ** (attempt + 1)
            print("legalize_lff: outline attempt {} (+{:.0%}) had {} fallback placement(s), "
                  "retrying with +{:.0%} outline".format(
                      attempt, growth - 1.0, n_fallback, next_growth - 1.0))

    if verbose:
        print("legalize_lff done: reshape={}, fallback={}, outline_attempts={}".format(
            n_reshape, n_fallback, attempt + 1))
    if n_fallback > 0:
        print("WARNING: legalize_lff could not fit {} block(s) within any tried outline size "
              "(up to {:.0%} of the original) — hard constraints (overlap-free, exact "
              "area/shape) are still guaranteed, but those blocks may sit outside the "
              "requested outline.".format(n_fallback, expand_factor ** (max_retries - 1)))

    # ---- 收縮階段：四邊各自往內縮，盡量壓掉剩餘留白 ----
    # 只在已經有一個成功結果（n_fallback==0）時才做，避免在還沒合法的狀態上
    # 浪費時間。策略是「猜測 + 驗證」而不是逐步二分搜尋：收縮幅度直接依照
    # 目前四個方向各自「outline 邊界 vs 實際佔用範圍」的留白比例決定（留白多
    # 的方向就大膽多收一點，留白少的方向少收一點），失敗就把步伐減半再試同一
    # 個基準，成功就用新的（更緊的）狀態當下一步的基準繼續收。這樣通常幾次
    # 嘗試內就能收斂到接近最緊的程度，不需要對每一邊做完整二分搜尋——刻意
    # 把總嘗試次數壓在很低的上限，讓大 k 的 case 也不會因為這步而變慢太多。
    if n_fallback == 0:
        max_shrink_attempts = 4
        shrink_step = 0.35   # 每次成功嘗試，往內收目前留白的 35%
        cur_x, cur_y, cur_w, cur_h = x_res, y_res, w_res, h_res
        cur_ox0, cur_oy0, cur_ox1, cur_oy1 = oxmin, oymin, oxmax, oymax
        step = shrink_step
        n_shrink_success = 0
        for _ in range(max_shrink_attempts):
            occ_xmin = float(cur_x.min()); occ_xmax = float((cur_x + cur_w).max())
            occ_ymin = float(cur_y.min()); occ_ymax = float((cur_y + cur_h).max())
            slack_left = max(occ_xmin - cur_ox0, 0.0)
            slack_right = max(cur_ox1 - occ_xmax, 0.0)
            slack_bottom = max(occ_ymin - cur_oy0, 0.0)
            slack_top = max(cur_oy1 - occ_ymax, 0.0)
            if slack_left + slack_right + slack_bottom + slack_top < 1e-6:
                break   # 四邊都已經貼到底，沒有留白可以收了（實測發現這是常見
                        # 情況：boundary-priority block 通常已經把外框撐到最緊，
                        # 剩下的留白幾乎都在內部，縮外框這招對這種情況沒有用）

            trial_ox0 = cur_ox0 + slack_left * step
            trial_oy0 = cur_oy0 + slack_bottom * step
            trial_ox1 = cur_ox1 - slack_right * step
            trial_oy1 = cur_oy1 - slack_top * step
            if trial_ox1 - trial_ox0 < 1e-6 or trial_oy1 - trial_oy0 < 1e-6:
                step *= 0.5
                continue

            xa, ya, wa, ha, n_fb, _ = _attempt(trial_ox0, trial_oy0, trial_ox1, trial_oy1)
            if n_fb == 0:
                cur_x, cur_y, cur_w, cur_h = xa, ya, wa, ha
                cur_ox0, cur_oy0, cur_ox1, cur_oy1 = trial_ox0, trial_oy0, trial_ox1, trial_oy1
                n_shrink_success += 1
                # 成功就維持同樣步伐、用新的（更緊的）基準繼續收
            else:
                step *= 0.5   # 失敗就縮小步伐，對同一個基準再試

        if n_shrink_success > 0:
            x_res, y_res, w_res, h_res = cur_x, cur_y, cur_w, cur_h
            if verbose:
                print("legalize_lff: shrink phase succeeded {} / {} attempts".format(
                    n_shrink_success, max_shrink_attempts))

    x, y, w, h = x_res, y_res, w_res, h_res

    # ---- 合併「彼此貼合但互不相連」的分離群聚 ----
    # LFF 用加權中位數決定每個 block 的位置，boundary 約束的 block 會被拉去
    # 貼指定邊，但「拉去貼邊」這件事跟「跟主要群聚保持接觸」是兩回事——如果
    # 這些貼邊 block 剛好落在主要群聚構不到的角落，中間就會留一塊沒人有
    # 誘因去填的空白（主要群聚裡的 block 一旦互相貼合，個別 block 沒有單獨
    # 移動的空間，沒辦法讓整團一起挪過去貼緊衛星群）。compact_merge_clusters
    # 抓出「彼此貼合」的連通分量，把非主要群的衛星群整體平移過去貼緊主要群
    # （每一軸都優先移動沒有 boundary 鎖定的那一方，兩邊都鎖定就跳過該軸），
    # 保證只會讓 bbox 縮小或持平，直接解決這種巨集尺度的留白。
    x, y = compact_merge_clusters(x, y, w, h, preplaced_mask=preplaced_mask,
                                  boundary_code=boundary_code, verbose=verbose)

    # ---- 每個 block 各自往（面積加權）全域重心靠攏 ----
    # compact_merge_clusters 只處理「已經彼此貼合」的連通分量之間的巨集移動；
    # 一個「沒有跟任何人貼合、四周有明顯空地」的孤立 block（例如角落單獨一塊，
    # 沒有被任何 boundary/cluster 約束釘住）不屬於任何衛星群，merge_clusters
    # 抓不到它。compact_gravity 用「每個 block 各自往全域重心走一小步、
    # 撞到人就二分法退讓」的方式，讓這種孤立 block 有機會往有人群的方向移動——
    # 跟 compact_reinsert 的網格搜尋是互補的：這裡是「往一個有意義的方向走」，
    # 便宜（不用枚举網格）但不保證找到「最佳」位置；compact_reinsert 之後
    # 會再對每個 block 做更精細的局部搜尋。
    if use_gravity:
        x, y = compact_gravity(x, y, w, h, preplaced_mask=preplaced_mask,
                               boundary_code=boundary_code, iters=gravity_iters)

    # ---- Remove-and-reinsert 局部搜尋：填內部縫隙、進一步壓縮 bbox ----
    # compact_merge_clusters 處理的是巨集尺度（整群衛星群搬移）；LFF 排布
    # 本身（MAXRECTS + 加權中位數單趟決定）在 block 之間仍會留下不少細碎
    # 縫隙——實測 packing density 平均只有 ~77.5%，而 GT optimal 隱含的
    # packing density 高達 ~97%，落差主要來自這種細碎的內部留白，不是巨集
    # 尺度的群聚分離。compact_reinsert 把每個 block 逐一「拔出來再插回」
    # 目前佔用範圍內成本最低的位置（bbox 面積 + 緊密度），能找到
    # compact_positions 那種單純滑動搆不到的位置，直接把內部縫隙填掉。
    if use_reinsert:
        x, y = compact_reinsert(x, y, w, h, preplaced_mask=preplaced_mask,
                                boundary_code=boundary_code, sweeps=reinsert_sweeps,
                                grid_density=reinsert_grid_density)

    # ---- 補一個「往鄰居貼齊」的軸對齊壓縮 ----
    # 縮外框那招只對「外框本身還有margin」的情況有用；實測發現 outline 通常
    # 已經被 boundary-priority block 撐到最緊（四邊 slack 幾乎是 0），真正的
    # 留白幾乎都在「block 之間」的內部縫隙——這是 MAXRECTS 排布本來就有的
    # 一點碎片化，縮外框動不了它，要用「往鄰居滑動貼齊」才能消掉。
    # compact_positions 是純幾何的沿 x/y 軸滑到貼齊，保證不會把不重疊變成
    # 重疊、不會讓 bbox 變大，成本也很低（不需要重新跑排布），適合當這裡的
    # 收尾動作。
    x, y = compact_positions(x, y, w, h, preplaced_mask=preplaced_mask,
                             boundary_code=boundary_code)

    # ---- compact_reinsert / compact_positions 之後再跑一次
    # compact_merge_clusters（預設開啟，見上方參數說明）----
    # compact_reinsert 的局部搜尋可能移動了原本卡住的衛星群成員、開出新的
    # 「彼此貼合」機會，第一次 compact_merge_clusters（在 reinsert 之前）
    # 看不到這些新機會。在 pipeline 尾端再補一次，成本低（多數情況下第一輪
    # 就會因為沒有東西可移動而立刻收斂）。
    if use_second_merge_pass:
        x, y = compact_merge_clusters(x, y, w, h, preplaced_mask=preplaced_mask,
                                      boundary_code=boundary_code, verbose=verbose)

    # ---- 把還沒真正貼到邊的 boundary block 推向真實邊界（v4.8，實驗用）----
    # compact_positions 只往 -x/-y 壓縮，剛好跟 LEFT/BOTTOM 鎖定的方向一致，
    # 但 RIGHT/TOP 鎖定需要往 +x/+y 推，pipeline 裡沒有對應機制——這些 block
    # 只在最初 LFF 放置時被一個會跟其他目標競爭的加權中位數軟性訊號拉過一次，
    # 之後所有壓縮 pass 都只保護它們的位置、不會再往外推。compact_snap_
    # boundary 直接找出目前沒貼齊的 boundary block、把它所在的剛體貼合分量
    # 整體推向 layout 目前的真實邊界，同樣用一個獨立的 boundary_hpwl_slack_ratio
    # 當代價閘門（見該函式 docstring；跟 compact_merge_cluster_groups 的
    # hpwl_slack_ratio 分開，方便個別調參）。放在跟 compact_merge_cluster_groups
    # 一樣的 pipeline 尾端位置，理由相同：避免被後續步驟撤銷。
    if use_snap_boundary:
        x, y = compact_snap_boundary(x, y, w, h, preplaced_mask=preplaced_mask,
                                     boundary_code=boundary_code,
                                     cluster_group=cluster_group,
                                     W_int=W_int, p2b_edges=p2b_edges,
                                     pins_pos=pins_pos,
                                     boundary_hpwl_slack_ratio=boundary_hpwl_slack_ratio,
                                     verbose=verbose)

    # ---- 針對 cluster group 本身的連通性做合併（v4.7，實驗用）----
    # compact_merge_clusters 合併的依據是「bbox 面積最大的全域主要群」，跟
    # cluster soft constraint（V_grouping，見 compute_cluster_violations：
    # 同組是否只有一個連通分量）不是同一件事——LFF 排布時 cluster 成員只有
    # 「加權中位數往組重心拉」這個軟性訊號，重心接近不代表真的貼在一起，
    # 若同一組被分裂成多個連通分量，compact_merge_clusters 不會特別去修，
    # 因為它只看全域 bbox 面積、不知道「這幾塊剛好同屬一個 cluster group」。
    # compact_merge_cluster_groups 直接針對每個 cluster group 檢查連通性，
    # 把分裂的子塊往組內面積最大的子塊合併，並額外要求移動後總 HPWL 不能
    # 變差（見該函式 docstring），只對「幾乎零 HPWL 代價」的 grouping 違規
    # 出手。刻意放在 pipeline 最尾端、compact_reinsert / compact_positions
    # 之後才跑：100 樣本測試發現放在 compact_reinsert 之前會被之後的逐一
    # 局部搜尋悄悄拆散剛建立好的貼合（compact_reinsert 的成本函式不知道
    # grouping 這個 boolean 鄰接需求），導致總 V_grouping 不降反升；放在
    # 最尾端、所有其他幾何調整都完成之後才做，就不會再被後續步驟撤銷。
    if use_cluster_merge:
        x, y = compact_merge_cluster_groups(x, y, w, h, preplaced_mask=preplaced_mask,
                                            boundary_code=boundary_code,
                                            cluster_group=cluster_group,
                                            W_int=W_int, p2b_edges=p2b_edges,
                                            pins_pos=pins_pos,
                                            hpwl_slack_ratio=hpwl_slack_ratio,
                                            verbose=verbose)

    # ---- 2-block 聯合 reinsert（v4.9，實驗用）----
    # compact_reinsert 一次只拔一個 block，看不到「兩個互相鄰近的 block 都
    # 要挪、才能一起讓 bbox 縮小」這種組合式改善——這是 detailed placement
    # 文獻裡標準的 local search 手法（一次移動一小群、而不是一個）。只對
    # touching graph 上彼此鄰接的 pair 出手，只有在讓全域 bbox 面積嚴格
    # 變小、且不讓 boundary/cluster/HPWL 變差時才採用（見 compact_pair_
    # reinsert docstring）。刻意放在 pipeline 最尾端、所有其他幾何調整都
    # 完成之後才做：早期實測放在 compact_reinsert 之後、compact_positions
    # 之前，雖然每一步本身都保證讓 bbox 嚴格變小，但改變了起始點後，後續
    # compact_positions / 兩次 compact_merge_clusters 等貪婪 pass 有時會走
    # 到不同、甚至更差的局部最佳解（同一批 seed 中出現過 +0.36% 的淨退步）
    # ——這正是先前 v4.6/v4.7/v4.8 都學到的教訓：任何「局部保證變好」的
    # pass，只要後面還有其他貪婪 pass 會重新處理同一批 block，就不能保證
    # 對「最終」結果也是單調不變差，必須放在真正的尾端才安全。
    if use_pair_reinsert:
        x, y = compact_pair_reinsert(x, y, w, h, preplaced_mask=preplaced_mask,
                                     boundary_code=boundary_code, cluster_group=cluster_group,
                                     sweeps=pair_reinsert_sweeps, grid_density=pair_reinsert_grid_density,
                                     W_int=W_int, p2b_edges=p2b_edges, pins_pos=pins_pos,
                                     hpwl_slack_ratio=pair_reinsert_hpwl_slack_ratio,
                                     verbose=verbose)

    # ---- Remove-and-reinsert 局部搜尋，同時重新考慮長寬比（v5.1，實驗用）----
    # legalize_lff 最初的 LFF 貪婪排布階段，每個 block 放置當下會試幾種長寬
    # 比（面積不變），但一旦放完，形狀從此凍結——後面所有壓縮 pass 都只動
    # (x, y)，不會重新考慮形狀。但「當下」最好的長寬比，等其他 block 陸續
    # 放進來、整體佈局改變後，未必還是最好的。compact_reinsert_reshape 把
    # `_aspect_variants` 接進跟 compact_reinsert 一樣的 remove-and-reinsert
    # 搜尋迴圈，候選變成「不同位置 × 幾種長寬比」的組合。只對沒有 MIB／
    # boundary 約束的 block 開放重新選長寬比（見該函式 docstring 說明限制
    # 原因），其餘 block 仍可搬位置、形狀不變。刻意放在 pipeline 最尾端，
    # 理由跟 compact_pair_reinsert 相同：避免被後續其他貪婪 pass 撤銷。
    if use_reinsert_reshape:
        x, y, w, h = compact_reinsert_reshape(x, y, w, h, preplaced_mask=preplaced_mask,
                                              fixed_mask=fixed_mask, mib_group=mib_group,
                                              boundary_code=boundary_code,
                                              sweeps=reinsert_reshape_sweeps,
                                              grid_density=reinsert_reshape_grid_density,
                                              verbose=verbose)

    # ---- 可微分全域微調（v5.2，實驗用）----
    # 前面所有 compact_* pass 都是離散、一次挪一兩個 block 的區域搜尋——
    # compact_pair_reinsert／compact_reinsert_reshape 兩次證明這條路在真實
    # 資料上已經觸頂。這裡改用類比（analytical）global placement 的核心
    # 概念（DREAMPlace/ePlace/RePlAce 這一系列）：把所有非 preplaced block
    # 的 (x, y) 一次性當連續變數，用 PyTorch autograd + Adam 對一個平滑
    # loss（重疊 + bbox 面積 + boundary + cluster + HPWL）做梯度下降，讓
    # 全部 block 同時、連續地互相讓位——不保證中途合法，跑完一定會投影回
    # `hard_zero_overlap` + `compact_positions` 保證的合法解，且只有
    # bbox 嚴格變小、boundary/cluster 違規不變差、HPWL 代價在
    # `gradient_finetune_hpwl_slack_ratio` 之內時才採用，否則整個操作
    # 視同沒發生（見 compact_gradient_finetune docstring）。
    if use_gradient_finetune:
        x, y = compact_gradient_finetune(x, y, w, h, preplaced_mask=preplaced_mask,
                                         boundary_code=boundary_code,
                                         cluster_group=cluster_group, W_int=W_int,
                                         outline_bbox=(oxmin, oymin, oxmax, oymax),
                                         n_steps=gradient_finetune_steps,
                                         lr=gradient_finetune_lr,
                                         patience=gradient_finetune_patience,
                                         hpwl_slack_ratio=gradient_finetune_hpwl_slack_ratio,
                                         verbose=verbose)

    # ---- 保底驗證：理論上此時已經零重疊，這裡只是零成本的防禦性再確認 ----
    preplaced_idx_list = [i for i in range(k) if preplaced_mask[i]]
    x, y = hard_zero_overlap(x, y, w, h, preplaced_indices=preplaced_idx_list)

    return x.astype(np.float64), y.astype(np.float64), w.astype(np.float64), h.astype(np.float64)


# ============================================================
# Legalization v2: anchor-guided packing + soft-aware
# ============================================================
#
# 設計：尊重 diffusion 給的座標當「目標位置」，按優先順序逐一找
# 「不重疊且離目標最近、soft cost 最低」的合法位置放下。
# 保證輸出零重疊。
#
# 可變動：
#   - 所有非 preplaced block 的 (x, y)
#   - 所有非 preplaced 且非 fixed 的 block 的 (w, h)，維持面積不變
#
# 流程：
#   1. preplaced block 釘住，當不可動障礙物
#   2. 排序剩餘 block（boundary 約束 > cluster 組 > 一般）
#   3. 對每個 block：先試原 (w,h)，往目標位置周圍的網格找；
#      若失敗則嘗試不同 aspect ratio（若可變形）；
#      仍失敗則擴大搜尋半徑
#   4. MIB 後處理：每個 MIB 組統一尺寸，若造成重疊則回退
#   5. Compaction：往左下推到貼齊或撞到別人為止
# ============================================================


def _build_spatial_grid(x, y, w, h, placed_mask, cell_size):
    """簡單 bucket grid：把已放置 block 索引存到對應 cell。"""
    grid = {}
    k = len(x)
    for i in range(k):
        if not placed_mask[i]:
            continue
        c0 = int(x[i] // cell_size)
        c1 = int((x[i] + w[i]) // cell_size)
        r0 = int(y[i] // cell_size)
        r1 = int((y[i] + h[i]) // cell_size)
        for cx in range(c0, c1 + 1):
            for cy in range(r0, r1 + 1):
                grid.setdefault((cx, cy), []).append(i)
    return grid


def _grid_overlap(x_cand, y_cand, w_cand, h_cand, grid, cell_size,
                  x, y, w, h, tol=1e-9):
    """檢查候選矩形跟 grid 中任何 block 是否重疊。回傳 True 表示有重疊。"""
    c0 = int(x_cand // cell_size)
    c1 = int((x_cand + w_cand) // cell_size)
    r0 = int(y_cand // cell_size)
    r1 = int((y_cand + h_cand) // cell_size)
    checked = set()
    for cx in range(c0, c1 + 1):
        for cy in range(r0, r1 + 1):
            for idx in grid.get((cx, cy), ()):
                if idx in checked:
                    continue
                checked.add(idx)
                # AABB overlap test
                ox = min(x_cand + w_cand, x[idx] + w[idx]) - max(x_cand, x[idx])
                oy = min(y_cand + h_cand, y[idx] + h[idx]) - max(y_cand, y[idx])
                if ox > tol and oy > tol:
                    return True
    return False


def _placement_cost(x_cand, y_cand, w_cand, h_cand,
                    target_x, target_y,
                    boundary_code_i, canvas_bbox,
                    cluster_id_i, cluster_centroids,
                    b2b_neighbors_i,         # list of (j, weight) for already-placed
                    p2b_pins_i,              # list of (px, py, weight)
                    x_arr, y_arr, w_arr, h_arr,
                    weights):
    """
    候選位置的成本（5 個 term）：
      (1) dist_to_anchor  : 離 diffusion 給的目標位置
      (2) boundary_pen    : 該貼邊的 block 沒貼到指定邊
      (3) cluster_pen     : 同 cluster 組已放 block 的距離
      (4) b2b_pen         : 跟已放好的 b2b 鄰居的加權 wirelength
      (5) p2b_pen         : 跟連到的 pin 的加權 wirelength
      (6) overflow_pen    : 越過 canvas 邊界的距離（強懲罰）
    """
    cx = x_cand + w_cand / 2
    cy = y_cand + h_cand / 2
    xmin_c, ymin_c, xmax_c, ymax_c = canvas_bbox

    # (1) 距離目標位置
    tcx = target_x + w_cand / 2
    tcy = target_y + h_cand / 2
    dist = abs(cx - tcx) + abs(cy - tcy)

    # (2) boundary 違規距離
    bnd_pen = 0.0
    if boundary_code_i > 0:
        if boundary_code_i & 1:
            bnd_pen += abs(x_cand - xmin_c)
        if boundary_code_i & 2:
            bnd_pen += abs((x_cand + w_cand) - xmax_c)
        if boundary_code_i & 4:
            bnd_pen += abs((y_cand + h_cand) - ymax_c)
        if boundary_code_i & 8:
            bnd_pen += abs(y_cand - ymin_c)

    # (3) 同 cluster 組重心距離
    clu_pen = 0.0
    if cluster_id_i > 0 and cluster_id_i in cluster_centroids:
        ccx, ccy = cluster_centroids[cluster_id_i]
        clu_pen = abs(cx - ccx) + abs(cy - ccy)

    # (4) b2b wirelength：跟已放好的鄰居的加權 Manhattan 距離
    # 這就是 HPWL 的 bounding-box 近似（兩兩之間用中心 Manhattan）
    b2b_pen = 0.0
    for j, w_edge in b2b_neighbors_i:
        cxj = x_arr[j] + w_arr[j] / 2
        cyj = y_arr[j] + h_arr[j] / 2
        b2b_pen += w_edge * (abs(cx - cxj) + abs(cy - cyj))

    # (5) p2b wirelength：到連到的 pin
    p2b_pen = 0.0
    for px, py, w_edge in p2b_pins_i:
        p2b_pen += w_edge * (abs(cx - px) + abs(cy - py))

    # (6) canvas overflow：block 任何部分跑出 canvas 的距離（強懲罰）
    over_pen = 0.0
    if x_cand < xmin_c:
        over_pen += xmin_c - x_cand
    if x_cand + w_cand > xmax_c:
        over_pen += (x_cand + w_cand) - xmax_c
    if y_cand < ymin_c:
        over_pen += ymin_c - y_cand
    if y_cand + h_cand > ymax_c:
        over_pen += (y_cand + h_cand) - ymax_c

    return (weights["dist"]     * dist
            + weights["boundary"] * bnd_pen
            + weights["cluster"]  * clu_pen
            + weights["b2b"]      * b2b_pen
            + weights["p2b"]      * p2b_pen
            + weights["overflow"] * over_pen)


def _aspect_variants(w0, h0, n=5, max_ratio=8.0):
    """產生幾組 (w, h) 候選，保持面積 = w0*h0 但改 aspect。"""
    area = w0 * h0
    # 原始 aspect 一定包含。其他在 log 空間均勻取樣。
    orig_r = w0 / max(h0, 1e-9)
    rs = [orig_r]
    # 加幾個變形（更扁、更高）
    factors = [0.5, 2.0, 0.25, 4.0]
    for f in factors:
        r = orig_r * f
        if 1.0 / max_ratio <= r <= max_ratio:
            rs.append(r)
    out = []
    for r in rs[:n]:
        w = (area * r) ** 0.5
        h = (area / r) ** 0.5
        out.append((w, h))
    return out


def legalize_v2(
    x_init, y_init, w_init, h_init,
    areas,
    preplaced_mask,        # (k,) bool, True = 位置和形狀都不可動
    fixed_mask,            # (k,) bool, True = 形狀不可動（preplaced 也算）
    mib_group=None,        # (k,) int, 0=無
    cluster_group=None,    # (k,) int, 0=無
    boundary_code=None,    # (k,) int, bitmask
    canvas_bbox=None,      # (xmin, ymin, xmax, ymax)，搜尋範圍 + 邊界懲罰用
    W_int=None,            # (k, k) b2b 連線權重矩陣
    p2b_edges=None,        # list of (pin_idx, block_idx, weight)
    pins_pos=None,         # (n_pins, 2) terminal 座標
    search_radius_steps=8, # 搜尋網格的「半徑」（cell 數）；reshape 開著時已有
                           # 5 種 shape candidate，半徑不用再加大補償，加大只會讓
                           # 候選數 (2r+1)^2 * 5 平方成長、拖慢速度
    weight_dist=1.0,
    weight_boundary=2.0,
    weight_cluster=1.0,
    weight_b2b=0.5,        # b2b wirelength 權重
    weight_p2b=0.1,        # p2b wirelength 權重（小，因 diffusion 已處理過）
    weight_overflow=50.0,  # 超出 canvas 的強懲罰（比 boundary/cluster 高一個量級）
    weight_shape=5.0,      # aspect 偏離懲罰（乘上 avg_side 換成距離單位）；
                           # 夠大讓「換形狀」只在明顯有好處或別無選擇時才會被選中，
                           # 避免 block 為了貼近 anchor 一點點距離就被拉成長條狀。
    weight_compact=1.5,    # 讓候選位置盡量不擴大「目前已放置 block 的 bbox」，
                           # 直接把緊密度目標放進主要的放置決策，而不是只能靠
                           # 事後壓縮補救——單靠事後壓縮效果有限（容易卡在區域
                           # 最佳解），從一開始放置時就考慮緊密度才是根本作法。
    allow_reshape=True,
    enable_quick_accept=True,  # 設 False 強制每個 block 都走完整搜尋
                               # （含 weight_compact 緊密度評分），追求品質時
                               # 關掉——quick-accept 只要 anchor 位置合法就直接
                               # 採用，完全不考慮緊不緊密，會讓很多 block 停在
                               # 「合法但有空隙」的位置，搆不到 compact 的加分。
    verbose=False,
):
    """
    Anchor-guided packing legalization。

    輸入皆為 (k,) 陣列。回傳 (x, y, w, h) 都是 (k,) float32，保證零重疊。

    搜尋優先在 canvas_bbox 內進行；只有當 canvas 內塞不下時才允許外擴。
    評分函數同時考慮 anchor 距離、boundary、cluster、b2b/p2b wirelength、
    以及越界懲罰，盡量同時兼顧 hard、soft、HPWL 三方面。
    """
    k = len(x_init)
    x = np.array(x_init, dtype=np.float64).copy()
    y = np.array(y_init, dtype=np.float64).copy()
    w = np.array(w_init, dtype=np.float64).copy()
    h = np.array(h_init, dtype=np.float64).copy()
    areas = np.array(areas, dtype=np.float64)
    preplaced_mask = np.array(preplaced_mask, dtype=bool)
    fixed_mask = np.array(fixed_mask, dtype=bool)
    if mib_group is None:     mib_group = np.zeros(k, dtype=np.int64)
    if cluster_group is None: cluster_group = np.zeros(k, dtype=np.int64)
    if boundary_code is None: boundary_code = np.zeros(k, dtype=np.int64)
    mib_group     = np.array(mib_group, dtype=np.int64)
    cluster_group = np.array(cluster_group, dtype=np.int64)
    boundary_code = np.array(boundary_code, dtype=np.int64)

    # 預設 canvas bbox = 所有 block 的初始 bbox
    if canvas_bbox is None:
        xmin0 = float(x.min()); ymin0 = float(y.min())
        xmax0 = float((x + w).max()); ymax0 = float((y + h).max())
        canvas_bbox = (xmin0, ymin0, xmax0, ymax0)

    # ---- 預處理 b2b / p2b 連線資訊（轉成 per-block 的鄰居列表）----
    # b2b_adj[i] = list of (j, weight)，只記 weight > 0 的
    b2b_adj = [[] for _ in range(k)]
    if W_int is not None:
        W_int_arr = np.asarray(W_int)
        for i in range(k):
            for j in range(k):
                if i != j and W_int_arr[i, j] > 0:
                    b2b_adj[i].append((j, float(W_int_arr[i, j])))

    # p2b_adj[i] = list of (px, py, weight)
    p2b_adj = [[] for _ in range(k)]
    if p2b_edges is not None and pins_pos is not None:
        pins_arr = np.asarray(pins_pos, dtype=np.float64)
        for edge in p2b_edges:
            p_idx, b_idx, w_edge = int(edge[0]), int(edge[1]), float(edge[2])
            if 0 <= p_idx < len(pins_arr) and 0 <= b_idx < k:
                p2b_adj[b_idx].append((pins_arr[p_idx, 0], pins_arr[p_idx, 1], w_edge))

    # cell size = 平均 block 邊長的 1/3（搜尋網格密度）
    avg_side = float(np.mean(np.sqrt(areas))) if len(areas) > 0 else 1.0
    cell_size = max(avg_side / 3.0, 1e-3)

    weights = {"dist":     weight_dist,
               "boundary": weight_boundary,
               "cluster":  weight_cluster,
               "b2b":      weight_b2b,
               "p2b":      weight_p2b,
               "overflow": weight_overflow,
               "shape":    weight_shape,
               "compact":  weight_compact}

    # ---- 排序：preplaced 不處理；其他按優先順序 ----
    # 優先序：boundary > cluster > 一般
    def priority(i):
        # 數字越小越優先（先放）
        if boundary_code[i] > 0:
            # 角落 > 邊
            bits = bin(int(boundary_code[i])).count("1")
            return (0, -bits)
        if cluster_group[i] > 0:
            return (1, 0)
        return (2, 0)

    placed_mask = preplaced_mask.copy()    # preplaced 視為已放
    pending = [i for i in range(k) if not preplaced_mask[i]]
    pending.sort(key=priority)

    # 保底 eject 游標：只有真的三個 phase 都找不到合法位置時才用（應該很罕見）。
    # 固定在「初始 layout 的最大 x」右側、按實際 block 高度堆疊，保證跟所有人
    # 都不重疊——修正舊版用固定 cell_size 間距堆疊、可能讓 eject 的 block 彼此
    # 重疊、也常常把 bbox 撐得比實際需要大很多的問題。
    xmin0, ymin0, xmax0, ymax0 = canvas_bbox
    eject_x = xmax0 + cell_size
    eject_y = ymin0

    # 已放好的 cluster 組重心（隨放置過程更新）
    cluster_centroids = {}   # cluster_id -> (cx_sum, cy_sum, count)

    def update_cluster_centroid(idx):
        gid = int(cluster_group[idx])
        if gid <= 0:
            return
        cx = x[idx] + w[idx] / 2
        cy = y[idx] + h[idx] / 2
        if gid in cluster_centroids:
            sx, sy, cnt = cluster_centroids[gid]
            cluster_centroids[gid] = (sx + cx, sy + cy, cnt + 1)
        else:
            cluster_centroids[gid] = (cx, cy, 1)

    def get_cluster_avg():
        return {gid: (sx / cnt, sy / cnt)
                for gid, (sx, sy, cnt) in cluster_centroids.items()}

    # preplaced 先進 cluster centroid
    for i in range(k):
        if preplaced_mask[i]:
            update_cluster_centroid(i)

    # ---- 主迴圈：逐一放置 ----
    n_reshape = 0
    n_extra_search = 0
    n_quick_accept = 0
    for i in pending:
        target_x = x[i]
        target_y = y[i]

        # ---- Quick-accept fast path ----
        # 如果這個 block 的 anchor 位置（diffusion/前一輪給的原始 x,y,w,h）
        # 已經跟目前已放置的 block 們不重疊、且完全落在 canvas_bbox 內，
        # 直接採用，不做網格搜尋。這對 HPWL / anchor 距離完全無損（cost=0），
        # 也省下絕大多數 block 的搜尋成本（force-guided sampler 通常已經把
        # 大部分 block 推到接近合法的位置）。
        # 只對「無 boundary、無 cluster」約束的 block 做（priority()==(2,0)），
        # 因為 boundary/cluster block 即使目前不重疊，仍可能需要透過搜尋
        # 移到更貼邊/更貼近同組的位置以降低 soft violation。
        is_plain = (boundary_code[i] == 0) and (cluster_group[i] == 0)
        if is_plain and enable_quick_accept:
            xmin_c0, ymin_c0, xmax_c0, ymax_c0 = canvas_bbox
            in_canvas0 = (x[i] >= xmin_c0 - 1e-9 and x[i] + w[i] <= xmax_c0 + 1e-9 and
                         y[i] >= ymin_c0 - 1e-9 and y[i] + h[i] <= ymax_c0 + 1e-9)
            if in_canvas0:
                quick_grid = _build_spatial_grid(x, y, w, h, placed_mask, cell_size)
                if not _grid_overlap(x[i], y[i], w[i], h[i], quick_grid, cell_size, x, y, w, h):
                    placed_mask[i] = True
                    update_cluster_centroid(i)
                    n_quick_accept += 1
                    continue

        # 可用尺寸候選：fixed 只有原尺寸；其他可調 aspect
        if fixed_mask[i] or not allow_reshape:
            shape_candidates = [(w[i], h[i])]
        else:
            shape_candidates = _aspect_variants(w[i], h[i])
        # reshape 的 aspect 偏離懲罰基準（log ratio），非原始形狀的候選會在下面
        # cost 計算時扣分，讓 reshape 只在真的有明顯好處或別無選擇時才會被選中。
        orig_log_r = float(np.log(w[i] / max(h[i], 1e-9)))

        best = None  # (cost, x_place, y_place, w_use, h_use)
        cluster_avg = get_cluster_avg()
        xmin_c, ymin_c, xmax_c, ymax_c = canvas_bbox

        # ---- compactness 參考：目前已放置 block 的 bbox ----
        # 候選位置若會把這個 bbox 撐大，依撐大的面積扣分——把「盡量不要讓
        # bbox 變大」直接做進放置決策，而不是等全部放完才靠事後壓縮補救。
        has_placed = bool(placed_mask.any())
        if has_placed:
            pm_x = x[placed_mask]; pm_y = y[placed_mask]
            pm_w = w[placed_mask]; pm_h = h[placed_mask]
            placed_xmin = float(pm_x.min()); placed_xmax = float((pm_x + pm_w).max())
            placed_ymin = float(pm_y.min()); placed_ymax = float((pm_y + pm_h).max())
            placed_area = (placed_xmax - placed_xmin) * (placed_ymax - placed_ymin)

        # 取出此 block 已放好的 b2b 鄰居（只看 placed_mask 為 True 的）
        b2b_placed = [(j, w_e) for (j, w_e) in b2b_adj[i] if placed_mask[j]]
        p2b_list = p2b_adj[i]

        # 搜尋策略：分三階段
        #   Phase 1 (preferred): 候選位置必須讓 block 完全在 canvas 內
        #   Phase 2 (fallback):  允許部分超出，但 overflow 懲罰會發力
        #   Phase 3 (last resort): 大幅擴大搜尋範圍
        # 三階段搜尋：
        #   Phase 0: 嚴格內含 + 小半徑 + 原 shape 優先
        #   Phase 1: 嚴格內含 + 大半徑 + 全部 shape 變體
        #   Phase 2: 最後手段，允許溢出 canvas（overflow 懲罰會發力把它拉回邊緣）
        for phase in range(3):
            if phase == 0:
                strict_canvas = True
                radius_mult = 1
            elif phase == 1:
                strict_canvas = True   # 仍嚴格內含，只是加大搜尋
                radius_mult = 3
            else:
                strict_canvas = False
                radius_mult = 4
                n_extra_search += 1

            grid = _build_spatial_grid(x, y, w, h, placed_mask, cell_size)
            radius = search_radius_steps * radius_mult

            for (w_use, h_use) in shape_candidates:
                # 計算合法的 x、y 範圍（讓 block 完全在 canvas 內）
                if strict_canvas:
                    x_lo = xmin_c
                    x_hi = xmax_c - w_use
                    y_lo = ymin_c
                    y_hi = ymax_c - h_use
                    if x_hi < x_lo or y_hi < y_lo:
                        # canvas 比 block 還小，這個 shape 無法嚴格內含，跳過
                        continue
                else:
                    # phase 2: 給 1 個 block 寬度的 margin 確保有候選可選；
                    # 強 overflow penalty 會讓演算法優先選擇 margin 小的位置
                    margin = max(w_use, h_use) * 1.0
                    x_lo = xmin_c - margin
                    x_hi = xmax_c - w_use + margin
                    y_lo = ymin_c - margin
                    y_hi = ymax_c - h_use + margin

                # 在 [x_lo, x_hi] × [y_lo, y_hi] 內以 target 為中心向外掃描
                for dx_step in range(-radius, radius + 1):
                    for dy_step in range(-radius, radius + 1):
                        x_cand = target_x + dx_step * cell_size
                        y_cand = target_y + dy_step * cell_size
                        # 範圍限制
                        if x_cand < x_lo or x_cand > x_hi: continue
                        if y_cand < y_lo or y_cand > y_hi: continue
                        # 重疊檢查
                        if _grid_overlap(x_cand, y_cand, w_use, h_use,
                                         grid, cell_size, x, y, w, h):
                            continue
                        cost = _placement_cost(
                            x_cand, y_cand, w_use, h_use,
                            target_x, target_y,
                            int(boundary_code[i]), canvas_bbox,
                            int(cluster_group[i]), cluster_avg,
                            b2b_placed, p2b_list,
                            x, y, w, h,
                            weights)
                        # 形狀偏離懲罰：換算成跟 anchor 距離同量級的單位（乘上
                        # avg_side），讓「換形狀」跟「多繞一點路但保持原形狀」
                        # 可以放在同一個 cost 尺度上公平比較。
                        shape_dev = abs(float(np.log(w_use / max(h_use, 1e-9))) - orig_log_r)
                        cost = cost + weights["shape"] * avg_side * shape_dev
                        # boundary 約束的 block 排除在 compact 懲罰之外：它們的
                        # 工作是「必須貼到 bbox 的某條邊」，這件事本質上一定會
                        # 撐大 bbox（總要有人定義那條邊），跟「盡量不要撐大
                        # bbox」直接衝突。實測發現兩者同時作用會讓 boundary
                        # violation 大幅惡化（compact 項把它們拉離該貼的邊，
                        # 換取小一點的 bbox 成長）——boundary 該不該貼邊優先權
                        # 更高，讓 weight_boundary 主導、不要被 compact 稀釋。
                        if has_placed and boundary_code[i] == 0:
                            new_xmin = min(placed_xmin, x_cand)
                            new_ymin = min(placed_ymin, y_cand)
                            new_xmax = max(placed_xmax, x_cand + w_use)
                            new_ymax = max(placed_ymax, y_cand + h_use)
                            growth = (new_xmax - new_xmin) * (new_ymax - new_ymin) - placed_area
                            # 除以 avg_side 把「面積」換算成跟 dist（長度）同量級的單位
                            cost = cost + weights["compact"] * (growth / avg_side)
                        if best is None or cost < best[0]:
                            best = (cost, x_cand, y_cand, w_use, h_use)

            if best is not None:
                break  # 該 phase 找到了就不用走下一階段

        # 兜底：所有 phase 都失敗，強制放在安全 eject 區（應該很罕見）。
        # 用原本的 shape（不 reshape），照實際高度堆疊，保證不跟任何人重疊。
        if best is None:
            print("WARNING: legalize_v2 found no legal position for block {} "
                  "within search radius, placing in safe eject stack".format(i))
            w_use, h_use = w[i], h[i]
            best = (1e18, eject_x, eject_y, w_use, h_use)
            eject_y += h_use + cell_size

        cost, x_place, y_place, w_use, h_use = best
        if (abs(w_use - w[i]) > 1e-6 or abs(h_use - h[i]) > 1e-6):
            n_reshape += 1
        x[i] = x_place; y[i] = y_place
        w[i] = w_use;   h[i] = h_use
        placed_mask[i] = True
        update_cluster_centroid(i)

    # ---- MIB 後處理：統一同組尺寸 ----
    if mib_group is not None:
        n_mib_fixed = 0
        n_mib_skipped = 0
        for gid in np.unique(mib_group):
            if gid == 0:
                continue
            members = [i for i in range(k) if mib_group[i] == gid]
            if len(members) < 2:
                continue
            # 不能動形狀的成員（fixed/preplaced）：取它們的形狀為基準
            anchors = [i for i in members if fixed_mask[i]]
            if anchors:
                tw = w[anchors[0]]; th = h[anchors[0]]
            else:
                # 取面積中位數的 aspect
                aspects = [w[i] / max(h[i], 1e-9) for i in members]
                med_r = float(np.median(aspects))
                # 用每個 member 自己的面積算（面積不能改）
                # 但 MIB 要求同 (w, h)，這只能在面積相同時才嚴格滿足
                # 這裡取面積中位數做為共同尺寸（會輕微違反個別面積，但 hard
                # 容差 1% 內通常還能撐住；若超出則跳過該組）
                med_area = float(np.median([areas[i] for i in members]))
                tw = float(np.sqrt(med_area * med_r))
                th = float(np.sqrt(med_area / med_r))

            # 嘗試把每個 member 換成 (tw, th)。檢查面積誤差和重疊。
            ok_for_all = True
            new_xywh = []
            for i in members:
                if fixed_mask[i]:
                    # 不能動：跳過，但若它本來尺寸跟 tw,th 不同，這組就無望了
                    if abs(w[i] - tw) > 1e-3 or abs(h[i] - th) > 1e-3:
                        ok_for_all = False
                        break
                    new_xywh.append((x[i], y[i], w[i], h[i]))
                    continue
                # 面積誤差檢查
                err = abs(tw * th - areas[i]) / max(areas[i], 1e-9)
                if err > 0.009:   # 留點安全邊際給 1% 上限
                    ok_for_all = False
                    break
                # 嘗試以 block 中心為錨換尺寸（不破壞中心位置）
                cx_i = x[i] + w[i] / 2
                cy_i = y[i] + h[i] / 2
                nx = cx_i - tw / 2
                ny = cy_i - th / 2
                # 把這個 member 暫時拿掉再檢查重疊
                placed_mask[i] = False
                grid = _build_spatial_grid(x, y, w, h, placed_mask, cell_size)
                if _grid_overlap(nx, ny, tw, th, grid, cell_size, x, y, w, h):
                    # 嘗試在小範圍微調位置
                    found = False
                    for r in range(1, 4):
                        for dx_s in range(-r, r + 1):
                            for dy_s in range(-r, r + 1):
                                cx_try = nx + dx_s * cell_size
                                cy_try = ny + dy_s * cell_size
                                if not _grid_overlap(cx_try, cy_try, tw, th,
                                                     grid, cell_size, x, y, w, h):
                                    nx, ny = cx_try, cy_try
                                    found = True; break
                            if found: break
                        if found: break
                    if not found:
                        placed_mask[i] = True
                        ok_for_all = False
                        break
                placed_mask[i] = True
                new_xywh.append((nx, ny, tw, th))

            if ok_for_all:
                for i, (nx, ny, nw, nh) in zip(members, new_xywh):
                    x[i], y[i], w[i], h[i] = nx, ny, nw, nh
                n_mib_fixed += 1
            else:
                n_mib_skipped += 1
        if verbose:
            print("  MIB groups unified: {}, skipped (hard conflict): {}".format(
                n_mib_fixed, n_mib_skipped))

    # ---- Compaction：往左下推到貼齊，縮小 bbox ----
    x, y = compact_positions(x, y, w, h, preplaced_mask=preplaced_mask,
                             boundary_code=boundary_code,
                             canvas_bbox=canvas_bbox, rounds=12)

    if verbose:
        print("Legalize done: quick_accept={}, reshape={}, extra_search={}".format(
            n_quick_accept, n_reshape, n_extra_search))

    # 保持 float64（不 downcast 成 float32）：這個輸出會再餵進 hard_zero_overlap
    # 做最後嚴格清零，過早降精度會讓後續計算的極小 margin 被 float32 精度吃掉
    # （見 hard_zero_overlap 的說明），在大座標量級的 sample 上曾實際造成
    # 官方 check_overlap 判定殘留違規。
    return (x.astype(np.float64), y.astype(np.float64),
            w.astype(np.float64), h.astype(np.float64))


# ============================================================
# 視覺化
# ============================================================

def _draw_single(ax, x, y, w, h, title, W_int=None,
                 fixed_indices=None, preplaced_indices=None,
                 p2b_edges=None, pins_pos=None,
                 mib_group=None, cluster_group=None, boundary_code=None,
                 xlim=None, ylim=None):
    """
    繪製單張 floorplan。

    視覺編碼：
      - fixed block:     藍色填色（hard constraint）
      - preplaced block: 紅色填色（hard constraint）
      - MIB 同組:        粗虛線邊框 + 左上角「M{ID}」標籤（soft）
      - cluster 同組:    粗點線邊框 + 右上角「C{ID}」標籤（soft）
      - boundary block:  在指定邊上畫粗紫色短線（soft）
      - b2b 連線:        紅色細線 alpha=0.4，zorder=3
      - p2b 連線:        藍色細線 alpha=0.4，zorder=3
      - pins:            綠色菱形

      xlim, ylim:  若給定則強制此範圍（用於並排對照時共用 canvas）
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    k = len(x)
    colors = plt.cm.Set3(np.linspace(0, 1, max(k, 1)))
    fixed_set = set(fixed_indices or [])
    preplaced_set = set(preplaced_indices or [])

    mib_group = np.array(mib_group) if mib_group is not None else np.zeros(k, dtype=int)
    cluster_group = np.array(cluster_group) if cluster_group is not None else np.zeros(k, dtype=int)
    boundary_code = np.array(boundary_code) if boundary_code is not None else np.zeros(k, dtype=int)

    # 為每個 MIB / cluster 組別配一個獨特的邊框顏色，讓「同組」一眼看出來
    def _group_palette(ids):
        uniq = [int(g) for g in np.unique(ids) if int(g) > 0]
        if not uniq:
            return {}
        pal = plt.cm.tab10(np.linspace(0, 1, max(len(uniq), 1)))
        return {gid: pal[i] for i, gid in enumerate(uniq)}

    mib_palette = _group_palette(mib_group[:k])
    cluster_palette = _group_palette(cluster_group[:k])

    # ---- 畫 block 本體 ----
    for i in range(k):
        if i in preplaced_set:
            fc = "red"; ec_base = "darkred"
        elif i in fixed_set:
            fc = "royalblue"; ec_base = "navy"
        else:
            fc = colors[i]; ec_base = "black"

        # 主框
        rect = patches.Rectangle((x[i], y[i]), w[i], h[i],
                                 linewidth=1, edgecolor=ec_base, facecolor=fc,
                                 alpha=0.7, zorder=1)
        ax.add_patch(rect)

        # ---- soft constraint 標記 ----
        mib_id = int(mib_group[i]) if i < len(mib_group) else 0
        clu_id = int(cluster_group[i]) if i < len(cluster_group) else 0

        # MIB: 粗虛線邊框（套用組別配色）
        if mib_id > 0:
            mib_color = mib_palette.get(mib_id, "purple")
            mib_rect = patches.Rectangle((x[i], y[i]), w[i], h[i],
                                         linewidth=2.0, edgecolor=mib_color,
                                         facecolor='none', linestyle='--',
                                         alpha=0.95, zorder=4)
            ax.add_patch(mib_rect)

        # cluster: 粗點線邊框（往內縮一點，避免和 MIB 重疊看不清）
        if clu_id > 0:
            clu_color = cluster_palette.get(clu_id, "darkorange")
            inset = min(w[i], h[i]) * 0.05
            clu_rect = patches.Rectangle((x[i] + inset, y[i] + inset),
                                         w[i] - 2 * inset, h[i] - 2 * inset,
                                         linewidth=2.0, edgecolor=clu_color,
                                         facecolor='none', linestyle=':',
                                         alpha=0.95, zorder=4)
            ax.add_patch(clu_rect)

        # block 編號（中央）
        cx, cy = x[i] + w[i] / 2, y[i] + h[i] / 2
        ax.text(cx, cy, str(i), ha="center", va="center", fontsize=6, zorder=5)

        # 角落小標籤：左上 MIB、右上 cluster
        if mib_id > 0:
            ax.text(x[i] + w[i] * 0.05, y[i] + h[i] * 0.95,
                    "M{}".format(mib_id), ha="left", va="top", fontsize=5,
                    color=mib_palette.get(mib_id, "purple"),
                    fontweight='bold', zorder=6)
        if clu_id > 0:
            ax.text(x[i] + w[i] * 0.95, y[i] + h[i] * 0.95,
                    "C{}".format(clu_id), ha="right", va="top", fontsize=5,
                    color=cluster_palette.get(clu_id, "darkorange"),
                    fontweight='bold', zorder=6)

    # ---- boundary 標記：在指定邊上畫粗紫色短線 ----
    # 用 floorplan 的 bounding box 當參考（boundary constraint 是相對於 bbox）
    if k > 0 and (boundary_code > 0).any():
        xmin_b = float(np.min(x))
        xmax_b = float(np.max(np.asarray(x) + np.asarray(w)))
        ymin_b = float(np.min(y))
        ymax_b = float(np.max(np.asarray(y) + np.asarray(h)))
        bnd_color = 'magenta'
        for i in range(k):
            c = int(boundary_code[i]) if i < len(boundary_code) else 0
            if c == 0:
                continue
            xi0, xi1 = x[i], x[i] + w[i]
            yi0, yi1 = y[i], y[i] + h[i]
            # 在指定邊上畫一條粗線（沿 block 對應邊的整段）
            if c & 1:  # LEFT  -> 畫在 block 左邊
                ax.plot([xi0, xi0], [yi0, yi1], color=bnd_color,
                        linewidth=3.5, alpha=0.9, zorder=7, solid_capstyle='round')
            if c & 2:  # RIGHT
                ax.plot([xi1, xi1], [yi0, yi1], color=bnd_color,
                        linewidth=3.5, alpha=0.9, zorder=7, solid_capstyle='round')
            if c & 4:  # TOP
                ax.plot([xi0, xi1], [yi1, yi1], color=bnd_color,
                        linewidth=3.5, alpha=0.9, zorder=7, solid_capstyle='round')
            if c & 8:  # BOTTOM
                ax.plot([xi0, xi1], [yi0, yi0], color=bnd_color,
                        linewidth=3.5, alpha=0.9, zorder=7, solid_capstyle='round')

    # ---- b2b 連線（紅）----
    if W_int is not None:
        W_int = np.asarray(W_int)
        for i in range(k):
            for j in range(i + 1, k):
                if W_int[i, j] > 0:
                    cx_i, cy_i = x[i] + w[i] / 2, y[i] + h[i] / 2
                    cx_j, cy_j = x[j] + w[j] / 2, y[j] + h[j] / 2
                    lw = min(W_int[i, j] / 5, 2.0)
                    ax.plot([cx_i, cx_j], [cy_i, cy_j], color='red',
                            alpha=0.4, linewidth=lw, zorder=3)

    # ---- p2b 連線（藍）+ pins（綠菱形）----
    if p2b_edges is not None and pins_pos is not None:
        pins = np.array(pins_pos)
        ax.scatter(pins[:, 0], pins[:, 1], marker='D', c='darkgreen',
                   s=35, zorder=8, edgecolors='black', linewidths=0.5)
        for p_idx, b_idx, weight in p2b_edges:
            p_idx, b_idx = int(p_idx), int(b_idx)
            if p_idx < 0 or b_idx < 0 or p_idx >= len(pins) or b_idx >= k:
                continue
            px, py = pins[p_idx, 0], pins[p_idx, 1]
            bx = x[b_idx] + w[b_idx] / 2
            by = y[b_idx] + h[b_idx] / 2
            ax.plot([px, bx], [py, by], color='blue',
                    alpha=0.4, linewidth=min(weight / 3, 2.0), zorder=3)

    # ---- 範圍 ----
    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim); ax.set_ylim(ylim)
    else:
        allx = np.concatenate([np.asarray(x), np.asarray(x) + np.asarray(w)]) if k else np.array([0, 1])
        ally = np.concatenate([np.asarray(y), np.asarray(y) + np.asarray(h)]) if k else np.array([0, 1])
        pad = max(allx.max() - allx.min(), ally.max() - ally.min(), 1) * 0.03
        ax.set_xlim(allx.min() - pad, allx.max() + pad)
        ax.set_ylim(ally.min() - pad, ally.max() + pad)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _compute_shared_limits(opt, gen, pins_pos=None, pad_ratio=0.03):
    """算共用的 x、y 範圍，涵蓋兩邊所有 block 與 pin。"""
    xs, ys = [], []
    for d in (opt, gen):
        xs.append(np.asarray(d["x"]))
        xs.append(np.asarray(d["x"]) + np.asarray(d["w"]))
        ys.append(np.asarray(d["y"]))
        ys.append(np.asarray(d["y"]) + np.asarray(d["h"]))
    if pins_pos is not None and len(pins_pos) > 0:
        pins = np.asarray(pins_pos)
        xs.append(pins[:, 0]); ys.append(pins[:, 1])
    allx = np.concatenate(xs); ally = np.concatenate(ys)
    span = max(allx.max() - allx.min(), ally.max() - ally.min(), 1)
    pad = span * pad_ratio
    return (allx.min() - pad, allx.max() + pad), (ally.min() - pad, ally.max() + pad)


def _add_constraint_legend(fig):
    """在圖底部加一個小圖例，說明 soft constraint 視覺編碼。"""
    import matplotlib.patches as patches
    import matplotlib.lines as mlines
    handles = [
        patches.Patch(facecolor='red', edgecolor='darkred', alpha=0.7, label='Preplaced (hard)'),
        patches.Patch(facecolor='royalblue', edgecolor='navy', alpha=0.7, label='Fixed-shape (hard)'),
        patches.Patch(facecolor='none', edgecolor='gray', linestyle='--', linewidth=2,
                      label='MIB group (dashed, colored by group)'),
        patches.Patch(facecolor='none', edgecolor='gray', linestyle=':', linewidth=2,
                      label='Cluster group (dotted, colored by group)'),
        mlines.Line2D([], [], color='magenta', linewidth=3, label='Boundary edge (soft)'),
        mlines.Line2D([], [], color='red', linewidth=1.5, alpha=0.6, label='B2B net'),
        mlines.Line2D([], [], color='blue', linewidth=1.5, alpha=0.6, label='P2B net'),
        mlines.Line2D([], [], color='darkgreen', marker='D', linestyle='None',
                      markersize=7, label='Terminal pin'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=True)


def plot_comparison(opt, gen, title="", save_path=None,
                    W_int=None, p2b_edges=None, pins_pos=None,
                    fixed_indices=None, preplaced_indices=None,
                    mib_group=None, cluster_group=None, boundary_code=None):
    """
    左：optimal（ground truth），右：inference 結果。共用 canvas，存成一張圖。

    opt, gen 各為 dict，含 x, y, w, h。
    mib_group / cluster_group / boundary_code 為 (k,) 陣列（共用，兩邊 block 順序一致）。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    xlim, ylim = _compute_shared_limits(opt, gen, pins_pos=pins_pos)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    _draw_single(axes[0], opt["x"], opt["y"], opt["w"], opt["h"],
                 "Optimal (GT)", W_int, fixed_indices, preplaced_indices,
                 p2b_edges, pins_pos,
                 mib_group=mib_group, cluster_group=cluster_group,
                 boundary_code=boundary_code,
                 xlim=xlim, ylim=ylim)
    _draw_single(axes[1], gen["x"], gen["y"], gen["w"], gen["h"],
                 "Inference", W_int, fixed_indices, preplaced_indices,
                 p2b_edges, pins_pos,
                 mib_group=mib_group, cluster_group=cluster_group,
                 boundary_code=boundary_code,
                 xlim=xlim, ylim=ylim)
    if title:
        fig.suptitle(title, fontsize=13)
    _add_constraint_legend(fig)
    plt.tight_layout(rect=(0, 0.04, 1, 1))   # 留底部空間給 legend
    if save_path:
        import os
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved comparison plot to {}".format(save_path))
    else:
        plt.show()
    plt.close()


def plot_floorplan(x, y, w, h, title="Floorplan", W_int=None, save_path=None,
                   fixed_indices=None, preplaced_indices=None,
                   p2b_edges=None, pins_pos=None,
                   mib_group=None, cluster_group=None, boundary_code=None):
    """單張繪圖（保留 v2 介面，新增 soft constraint 標記）。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    _draw_single(ax, x, y, w, h, title, W_int, fixed_indices, preplaced_indices,
                 p2b_edges, pins_pos,
                 mib_group=mib_group, cluster_group=cluster_group,
                 boundary_code=boundary_code)
    _add_constraint_legend(fig)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved plot to {}".format(save_path))
    else:
        plt.show()
    plt.close()


# ============================================================
# EMA
# ============================================================

class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}