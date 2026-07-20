"""
Dataset for FloorSet-Lite v3 — 正確解析 soft constraints

v2 的致命問題：對整個 constraints tensor 做 np.clip(0,1)，
把 MIB / cluster 的「組別 ID」和 boundary 的「方向 bitmask」全部壓成 1，
soft constraint 資訊完全毀損。v3 修正這一點。

官方 placement_constraints: (n_blocks, 5)
  col 0: fixed      — 0/1
  col 1: preplaced  — 0/1
  col 2: MIB        — 0=無，否則為組別 ID（同 ID 同尺寸）
  col 3: cluster    — 0=無，否則為組別 ID（同 ID 需相鄰成一塊）
  col 4: boundary   — 0=無，否則方向 bitmask:
                       LEFT=1, RIGHT=2, TOP=4, BOTTOM=8
                       角落為相加（TOP-LEFT=5, TOP-RIGHT=6,
                       BOTTOM-LEFT=9, BOTTOM-RIGHT=10）
                       bit0=left, bit1=right, bit2=top, bit3=bottom

新增輸出欄位：
    block_features:  (N, 13)  — 見下方組裝註解
    mib_group:       (N,)     — long, MIB 組別 ID（0=無）
    cluster_group:   (N,)     — long, cluster 組別 ID（0=無）
    boundary_code:   (N,)     — long, boundary bitmask（0=無）
    group_bias:      (N, N)   — float, 同 MIB 組或同 cluster 組 = 1，用於 attention bias
    fixed_mask / preplaced_mask / mask / state / conn_weights / num_blocks（同 v2）
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def _decode_boundary_bits(code):
    """把 boundary bitmask 拆成 (left, right, top, bottom) 4 個 0/1。"""
    code = int(code)
    left   = 1.0 if (code & 1) else 0.0
    right  = 1.0 if (code & 2) else 0.0
    top    = 1.0 if (code & 4) else 0.0
    bottom = 1.0 if (code & 8) else 0.0
    return left, right, top, bottom


class FloorplanDataset(Dataset):

    def __init__(self, official_dataset, max_blocks=120, is_test=False):
        self.official = official_dataset
        self.max_blocks = max_blocks
        self.is_test = is_test

    def __len__(self):
        return len(self.official)

    def __getitem__(self, idx):
        sample = self.official[idx]
        inputs = sample['input']
        labels = sample['label']

        area_target = inputs[0]   # (n_blocks,)
        b2b_conn = inputs[1]      # (n_edges, 3): [block_i, block_j, weight]
        p2b_conn = inputs[2]      # (n_edges, 3): [pin_idx, block_idx, weight]
        pins_pos = inputs[3]      # (n_pins, 2)
        constraints = inputs[4]   # (n_blocks, 5)

        k = int((area_target != -1).sum().item())
        N = self.max_blocks

        # -- Ground truth (x, y, w, h) --
        if self.is_test:
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
        else:
            fp_sol = labels[1]
            gt_w = fp_sol[:k, 0].numpy().astype(np.float32)
            gt_h = fp_sol[:k, 1].numpy().astype(np.float32)
            gt_x = fp_sol[:k, 2].numpy().astype(np.float32)
            gt_y = fp_sol[:k, 3].numpy().astype(np.float32)

        # -- Canvas & 歸一化 --
        if k == 0:
            canvas_w, canvas_h = 1.0, 1.0
            x_offset, y_offset = 0.0, 0.0
        else:
            x_offset = float(np.min(gt_x[:k]))
            y_offset = float(np.min(gt_y[:k]))
            canvas_w = float(np.max(gt_x[:k] + gt_w[:k]) - x_offset) + 1e-6
            canvas_h = float(np.max(gt_y[:k] + gt_h[:k]) - y_offset) + 1e-6

        x_norm = (gt_x - x_offset) / canvas_w
        y_norm = (gt_y - y_offset) / canvas_h
        log_r = np.log(gt_w / np.clip(gt_h, 1e-8, None))

        areas = area_target[:k].numpy().astype(np.float32)
        canvas_area = canvas_w * canvas_h
        areas_norm = areas / canvas_area

        # -- 解析 constraints（v3 核心：不再無腦 clip）--
        # 先取原始整數值
        mib_group_arr = np.zeros(N, dtype=np.int64)
        cluster_group_arr = np.zeros(N, dtype=np.int64)
        boundary_code_arr = np.zeros(N, dtype=np.int64)
        is_fixed = np.zeros(N, dtype=np.float32)
        is_preplaced = np.zeros(N, dtype=np.float32)
        bnd_bits = np.zeros((N, 4), dtype=np.float32)  # left,right,top,bottom

        if constraints is not None and k > 0:
            cons_np = constraints[:k].numpy()
            # padding 是 -1，視為「無約束」(0)
            cons_np = np.where(cons_np < 0, 0, cons_np)

            is_fixed[:k]     = (cons_np[:, 0] > 0.5).astype(np.float32)
            is_preplaced[:k] = (cons_np[:, 1] > 0.5).astype(np.float32)
            mib_group_arr[:k]     = cons_np[:, 2].astype(np.int64)   # 組別 ID（保留！）
            cluster_group_arr[:k] = cons_np[:, 3].astype(np.int64)   # 組別 ID（保留！）
            boundary_code_arr[:k] = cons_np[:, 4].astype(np.int64)   # bitmask（保留！）

            # 向量化 boundary bit 解碼：對 (N,) 的 bitmask 直接做 bitwise AND
            codes = boundary_code_arr   # (N,) int64
            bnd_bits[:, 0] = (codes & 1).astype(np.float32)   # LEFT
            bnd_bits[:, 1] = ((codes & 2) > 0).astype(np.float32)   # RIGHT
            bnd_bits[:, 2] = ((codes & 4) > 0).astype(np.float32)   # TOP
            bnd_bits[:, 3] = ((codes & 8) > 0).astype(np.float32)   # BOTTOM

        is_mib_member     = (mib_group_arr > 0).astype(np.float32)       # (N,)
        is_cluster_member = (cluster_group_arr > 0).astype(np.float32)   # (N,)

        # -- P2B: 計算每個 block 的 weighted pin centroid --
        pin_cx = np.zeros(N, dtype=np.float32)
        pin_cy = np.zeros(N, dtype=np.float32)
        pin_tw = np.zeros(N, dtype=np.float32)

        if p2b_conn is not None and pins_pos is not None and len(p2b_conn) > 0:
            pins_np = pins_pos.numpy().astype(np.float32)

            # 向量化：把 p2b_conn 整個轉成 numpy 一次處理
            p2b_np = p2b_conn.numpy() if hasattr(p2b_conn, "numpy") else np.asarray(p2b_conn)
            p_idx = p2b_np[:, 0].astype(np.int64)
            b_idx = p2b_np[:, 1].astype(np.int64)
            weights = p2b_np[:, 2].astype(np.float32)

            # 過濾無效 edge
            valid = (p_idx >= 0) & (b_idx >= 0) & (b_idx < k) & (p_idx < len(pins_np))
            p_idx = p_idx[valid]; b_idx = b_idx[valid]; weights = weights[valid]

            if len(p_idx) > 0:
                px_norm = (pins_np[p_idx, 0] - x_offset) / canvas_w
                py_norm = (pins_np[p_idx, 1] - y_offset) / canvas_h
                # np.add.at 做 scatter-add（處理重複 index）
                np.add.at(pin_cx, b_idx, weights * px_norm)
                np.add.at(pin_cy, b_idx, weights * py_norm)
                np.add.at(pin_tw, b_idx, weights)

            valid_pw = pin_tw > 0
            pin_cx[valid_pw] /= pin_tw[valid_pw]
            pin_cy[valid_pw] /= pin_tw[valid_pw]
            max_tw = pin_tw.max()
            if max_tw > 0:
                pin_tw = pin_tw / max_tw

        # -- 組合 block features: (N, 13) --
        # [0]=area, [1]=fixed, [2]=preplaced, [3]=mib_member, [4]=cluster_member,
        # [5..8]=boundary bits(L,R,T,B), [9]=pin_cx, [10]=pin_cy, [11]=pin_tw, [12]=reserved
        block_features = np.zeros((N, 13), dtype=np.float32)
        block_features[:k, 0] = areas_norm
        block_features[:, 1]  = is_fixed
        block_features[:, 2]  = is_preplaced
        block_features[:, 3]  = is_mib_member
        block_features[:, 4]  = is_cluster_member
        block_features[:, 5:9] = bnd_bits
        block_features[:, 9]  = pin_cx
        block_features[:, 10] = pin_cy
        block_features[:, 11] = pin_tw
        # [12] reserved = 0

        # -- B2B connectivity matrix --
        conn_matrix = np.zeros((N, N), dtype=np.float32)
        if b2b_conn is not None and len(b2b_conn) > 0:
            b2b_np = b2b_conn.numpy() if hasattr(b2b_conn, "numpy") else np.asarray(b2b_conn)
            i_idx = b2b_np[:, 0].astype(np.int64)
            j_idx = b2b_np[:, 1].astype(np.int64)
            wts = b2b_np[:, 2].astype(np.float32)
            valid = (i_idx >= 0) & (j_idx >= 0) & (i_idx < k) & (j_idx < k)
            i_idx = i_idx[valid]; j_idx = j_idx[valid]; wts = wts[valid]
            # 依出現順序寫入（保留原版 last-write-wins 語意，避免 fancy
            # indexing 對重複 (i,j) 處理的不確定性）。對重複 edge 來說，
            # 後出現的 weight 會覆蓋前面的。150 條 edge 級別的迴圈成本可忽略。
            for ii, jj, ww in zip(i_idx.tolist(), j_idx.tolist(), wts.tolist()):
                conn_matrix[ii, jj] = ww
                conn_matrix[jj, ii] = ww

        # -- Group bias matrix: 同 MIB 組 或 同 cluster 組 = 1 --
        # 向量化：把 group id 廣播比較
        # G[i,j] = (mib[i]>0 且 mib[i]==mib[j]) 或 (clu[i]>0 且 clu[i]==clu[j])
        group_bias = np.zeros((N, N), dtype=np.float32)
        if k > 0:
            mib_v = mib_group_arr[:k]              # (k,)
            clu_v = cluster_group_arr[:k]          # (k,)
            same_mib = (mib_v[:, None] == mib_v[None, :]) & (mib_v[:, None] > 0)  # (k,k)
            same_clu = (clu_v[:, None] == clu_v[None, :]) & (clu_v[:, None] > 0)
            mask = same_mib | same_clu
            # 對角線去掉（block 跟自己不算）
            np.fill_diagonal(mask, False)
            group_bias[:k, :k] = mask.astype(np.float32)

        # -- State (training target) --
        state_pad = np.zeros((N, 3), dtype=np.float32)
        state_pad[:k, 0] = x_norm
        state_pad[:k, 1] = y_norm
        state_pad[:k, 2] = log_r

        # -- Masks --
        mask = np.zeros(N, dtype=bool)
        mask[:k] = True

        fixed_mask = np.zeros(N, dtype=bool)
        preplaced_mask = np.zeros(N, dtype=bool)
        fixed_mask[:k] = (is_fixed[:k] > 0.5) | (is_preplaced[:k] > 0.5)
        preplaced_mask[:k] = (is_preplaced[:k] > 0.5)

        # v4.0: areas_norm 完整版 (N,) 給 overlap loss 用
        areas_norm_pad = np.zeros(N, dtype=np.float32)
        areas_norm_pad[:k] = areas_norm

        return {
            "state": torch.tensor(state_pad),                  # (N, 3)
            "block_features": torch.tensor(block_features),    # (N, 13)
            "conn_weights": torch.tensor(conn_matrix),         # (N, N)
            "group_bias": torch.tensor(group_bias),            # (N, N)
            "mask": torch.tensor(mask),                        # (N,)
            "fixed_mask": torch.tensor(fixed_mask),            # (N,)
            "preplaced_mask": torch.tensor(preplaced_mask),    # (N,)
            "mib_group": torch.tensor(mib_group_arr),          # (N,)
            "cluster_group": torch.tensor(cluster_group_arr),  # (N,)
            "boundary_code": torch.tensor(boundary_code_arr),  # (N,)
            "areas_norm": torch.tensor(areas_norm_pad),        # (N,) v4.0
            "num_blocks": k,
        }


def create_dataloader(official_dataset, batch_size=64, max_blocks=120,
                      shuffle=True, num_workers=0, is_test=False):
    dataset = FloorplanDataset(official_dataset, max_blocks=max_blocks, is_test=is_test)
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )
    # 只有 num_workers > 0 時才設這兩個（否則 PyTorch 會報錯）
    if num_workers > 0:
        kwargs["persistent_workers"] = True   # 避免每個 epoch 重啟 worker
        kwargs["prefetch_factor"] = 2          # 預載 2 個 batch
    return DataLoader(dataset, **kwargs)