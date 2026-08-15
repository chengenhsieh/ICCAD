"""
inference_v2.py -- 實驗用替代推論入口，legalize 步驟改用隊友團隊
（ICCAD2026-Problem-C/diffusion-floorplanner）的正式 legalizer
`teammate_legalizer.compaction.legalize_sample`，取代我們自己的
`utils.legalize_lff`。diffusion 端完全不變（直接重用 inference.py 的
`generate_floorplan`）。

背景：跟已經否決的 v5.39（`compact_joint_convex`）不是同一件事。v5.39
只 port 了「單次凸優化」這個最小核心，套用在我們自己已經跑完
`legalize_lff` 全部壓縮 pass 之後的成品上，因為目標函式對 grouping/
boundary soft violation 零感知，V_relative 大幅變差（166%~386%）。這裡
改用隊友實際「出貨」的完整版本：precedence 圖重建+求解跑 3 輪
（compact_iters）、grouping/boundary 懲罰項直接在凸優化目標裡
（lam_grp/lam_bnd_add）、求解完接一整串逐步驟都有 feasibility gate 的
收尾 pass（snap_groups/pull_boundary/snap_mib_shapes/gap_repair），而且
是直接吃原始 diffusion 輸出跑完整 legalize（不是疊加在我們自己已經
仔細做過 cluster/boundary 感知壓縮的成品上跟自己打架）。詳見
`teammate_legalizer/README.md` 與 `docs/plan` 討論。

驗證：100 樣本配對比較（production 等價設定：DDIM_STEPS=10/N_SAMPLES=14，
兩邊都套用 TOP_K_CANDIDATES=5 的 legalize-then-select）確認 real_cost
-20.87%（96/100 勝、0/100 infeasible），已經接進 my_optimizer.py 當
正式送測的 legalizer（見該檔案頂部的版本說明）。`legalize_top_k_candidates_v2`
是給 my_optimizer.py 用的正式接線函式；`run_one_sample_v2`／`__main__`
區塊維持給人工測試用。
"""
import sys
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, _THIS_DIR)

# 順序有陷阱（跟 my_optimizer.py 頂部同一個問題，這裡複製同一套修法）：
# `iccad2026_evaluate.py` import 時會把 FloorSet/（它的上一層目錄）插進
# sys.path[0]，而它自己的 `from cost import ...` 會連帶讓 `cost.py` 執行
# `from utils import (unpad_tensor, ...)`——這個 `utils` 解析到的是
# FloorSet/utils.py（一個跟這個 contest 目錄同名但完全不同的泛用模組）。
# 如果我們先 import `inference`/`utils`（把 sys.modules["utils"] 綁定成
# iccad2026contest/utils.py），之後才 import `iccad2026_evaluate`，
# `cost.py` 會撈到快取住的錯誤模組、找不到 `unpad_tensor` 而炸掉。這裡
# 只有 __main__ 區塊用得到 ContestEvaluator，但為了讓這個檔案不管誰先
# import 都正確，比照 my_optimizer.py：先讓 iccad2026_evaluate 把它要的
# 東西綁進自己的 namespace，再清掉快取、把這個目錄重新插到最前面。
from iccad2026_evaluate import ContestEvaluator  # noqa: E402 (import-order fixup, see above)

sys.modules.pop("utils", None)
sys.path.insert(0, _THIS_DIR)

import time

import numpy as np
import torch

from inference import (
    generate_floorplan,
    evaluate_and_report,
    _build_entry_from_result,
    _guarantee_zero_overlap,
    load_model,
    legalize_result,  # v6.1: solve-infeasible 保底用（見 legalize_result_v2）
)
from utils import (
    compute_hpwl_vectorized,
    compute_p2b_hpwl,
    total_overlap,
    count_overlaps,
    compute_soft_violations,
)
from teammate_legalizer.compaction import legalize_sample

# 「出貨」設定，抓自隊友 infer.py 的 argparse 預設值 / config.py（見
# plan 文件裡的完整對照表跟出處，不是猜的）。legalize_sample() 函式本身
# 的 bare default 大多是全部關閉（snap=False/pull=False/lam_grp=0.0/...），
# 跟他們實際出貨時 CLI 套用的值不同，這裡明確覆寫成出貨值。
DEFAULT_LEGALIZE_SAMPLE_KWARGS = dict(
    mode="hybrid",
    gap_frac=1e-3,
    iters=3,                  # infer.py 的 --compact-iters
    prec=12,
    lam_grp=0.2,
    snap=True,
    snap_mib=True,
    snap_clear=("step",),
    gap_repair=True,
    # v6.4（採用）：出貨值 4，我們自己的資料上篩過（5 個大樣本 ×
    # (iters, finish_rounds) 掃描，legalize_sample() 對同一批 raw
    # candidate 共用、排除 RNG confound）：4→2 在 4/5 樣本上 area_gap/
    # hpwl_gap 逐位元不變（唯一有波動的樣本 V_rel 只從 0.0476→0.0635 的
    # 小幅雜訊），legalize 時間有小幅節省；4→1 或同時調低 iters 則明顯
    # 傷品質（area_gap 最多從 +2.45% 惡化到 +21.30%，甚至讓一個原本
    # legalize_sample 解得出來的樣本變成觸發 legalize_lff fallback）。
    # 官方公式裡 area_gap/hpwl_gap 沒有下限、runtime 有 max(0.7,...) 下限
    # （這個專案的 rt_mult 已經有近半數樣本卡在 0.7），继续往下調用時間
    # 換品質大概率虧本，只有 4→2 這一格是乾淨的免費節省。
    finish_rounds=2,
    grp_hard=(),
    grp_dag=("b2", "e7"),
    pull_batch=True,
    cluster=False,
    pull=True,
    pull_reshape=False,
    area_slack=0.0,
    reseed_rounds=2,
    sep_rule=None,
    resolve=(),
    true_bbox=True,
    lam_bnd_add=4096.0,
    # alt_lam_bnd_add 不是 legalize_sample() 的參數（那是 infer.py 自己
    # 「候選混搭」機制用的 CLI 概念，不會轉呼叫進 legalize_sample），
    # 傳進去會直接 TypeError——第一次跑 smoke test 就是這樣炸的。
)


def legalize_result_v2(
    best, areas, W_int, p2b_edges, pins_pos,
    preplaced_mask, fixed_mask,
    mib_group, cluster_group, boundary_code,
    opt_target_pos,
    verbose=True,
    **legalize_kwargs
):
    """
    把我們自己的 `best`（generate_floorplan 回傳的 raw 候選，x,y 是左下角）
    轉成 `legalize_sample()` 要的參數、呼叫、再把結果轉回我們自己的
    result dict schema（跟 `legalize_result` 回傳的東西同一種形狀，方便
    直接沿用 `evaluate_and_report`/`_build_entry_from_result`）。

    opt_target_pos: (k,4) array/tensor，-1=free；preplaced 帶
    (x,y,w,h)，fixed-shape 帶 (-1,-1,w,h)——跟這個專案其他地方
    （run_one_sample/my_optimizer.py/各篩選腳本）已經在用的
    `opt_target_pos` 建構慣例逐位元相同，直接沿用即可。

    刻意用我們自己的 compute_hpwl_vectorized/compute_p2b_hpwl/
    compute_soft_violations/total_overlap/count_overlaps 重新算所有指標
    ——不信任 legalize_sample 內部算好的東西或它回傳的 tag 字串，確保
    跟這個專案其他地方的報告/評分口徑一致。
    """
    x, y, w, h = best["x"], best["y"], best["w"], best["h"]
    k = len(x)
    coords = np.stack([x + w / 2.0, y + h / 2.0, w, h], axis=1)

    is_pp = np.asarray(preplaced_mask, dtype=bool)
    is_fs = np.asarray(fixed_mask, dtype=bool)
    area_target = np.asarray(areas, dtype=np.float64)
    target_ll = np.asarray(opt_target_pos, dtype=np.float64)
    mib = np.asarray(mib_group, dtype=np.int64)
    grouping = np.asarray(cluster_group, dtype=np.int64)
    boundary = np.asarray(boundary_code, dtype=np.int64)

    kwargs = dict(DEFAULT_LEGALIZE_SAMPLE_KWARGS)
    kwargs.update(legalize_kwargs)

    t0 = time.perf_counter()
    try:
        pos_ll, tag = legalize_sample(
            coords, is_pp, is_fs, area_target, target_ll,
            mib=mib, grouping=grouping, boundary=boundary,
            **kwargs
        )
    except Exception as e:
        if verbose:
            print("legalize_result_v2: legalize_sample raised {!r}, "
                  "falling back to raw placement (frame-aligned only, "
                  "NOT overlap-free -- caller must not skip the "
                  "zero-overlap guarantee below).".format(e))
        pos_ll = np.stack([x, y, w, h], axis=1)
        tag = "exception-fallback"
    t_legalize = time.perf_counter() - t0

    x2 = pos_ll[:, 0].astype(np.float64)
    y2 = pos_ll[:, 1].astype(np.float64)
    w2 = pos_ll[:, 2].astype(np.float64)
    h2 = pos_ll[:, 3].astype(np.float64)

    # v6.1（採用，見 CHANGELOG.md）：legalize_sample() 自己的凸優化求解在
    # 某些高度受限的樣本上（大量 preplaced/cluster/boundary 疊加、又碰上
    # raw diffusion 輸出本身嚴重重疊）會判定 infeasible，退回「幾乎原封
    # 不動的輸入」——這個輸入常常還帶著幾十對重疊，遠超下面
    # `_guarantee_zero_overlap` 正常迭代式修復能處理的量級，逼得它的最後
    # 一道絕對保底（強制彈射）把幾十個 block 硬搬到遠處，佈局整個報廢。
    # 官方 evaluate 實測踩到過（test_id=94/98，real_cost 飆到 18-23），
    # 追查發現是 legalize_sample 對這兩個樣本 10/10 次重跑（5 候選×2
    # seed）全部回傳 tag='fallback (solve infeasible)'，raw overlap 本身
    # 就有 ~90 對。偵測到 legalize_sample 沒能把重疊清乾淨時，直接改用
    # 我們自己的 legalize_lff（保證用建構式方法從零排出合法佈局，不管
    # raw 座標多爛都不會失敗）處理這個候選，而不是讓它流到彈射保底。
    n_overlaps_v2 = count_overlaps(x2, y2, w2, h2)
    used_lff_fallback = n_overlaps_v2 > 0
    if used_lff_fallback:
        if verbose:
            print("legalize_result_v2: legalize_sample 沒清乾淨重疊"
                  "（{} 對，tag={!r}），改用 legalize_lff 處理這個候選"
                  .format(n_overlaps_v2, tag))
        fallback = legalize_result(
            best, areas, W_int, p2b_edges, pins_pos,
            preplaced_mask, fixed_mask, mib_group, cluster_group, boundary_code,
            outline_bbox=None, verbose=False,
        )
        x2, y2, w2, h2 = fallback["x"], fallback["y"], fallback["w"], fallback["h"]
        tag = "legalize_lff_fallback (v2 left {} overlaps, orig_tag={!r})".format(
            n_overlaps_v2, tag)

    # 防禦性保底：legalize_sample 自己的 prec=12 捨入 + feasibility 檢查
    # 跟我們自己 ContestEvaluator 的重疊判定不保證用同一套容差，這裡沿用
    # legalize_result() 的同一道保底（不應該真的觸發，只是雙重保險；上面
    # 的 legalize_lff fallback 分支也一樣需要這道保底，legalize_lff 自己
    # 雖然結構上保證零重疊，但走過這裡多一層確認不吃虧）。
    preplaced_idx_list = [i for i in range(k) if is_pp[i]]
    x2, y2 = _guarantee_zero_overlap(x2, y2, w2, h2, preplaced_idx_list)

    b2b_hpwl = compute_hpwl_vectorized(x2, y2, w2, h2, W_int)
    p2b_hpwl = 0.0
    if p2b_edges is not None and pins_pos is not None:
        p2b_hpwl = compute_p2b_hpwl(x2, y2, w2, h2, p2b_edges, pins_pos)
    overlap = total_overlap(x2, y2, w2, h2)
    n_overlaps = count_overlaps(x2, y2, w2, h2)
    bbox_area = float((np.max(x2 + w2) - np.min(x2)) * (np.max(y2 + h2) - np.min(y2)))
    soft = compute_soft_violations(x2, y2, w2, h2, mib, grouping, boundary)

    if verbose:
        print("legalize_result_v2: tag={!r} time={:.3f}s".format(tag, t_legalize))

    return {
        "x": x2, "y": y2, "w": w2, "h": h2,
        "b2b_hpwl": b2b_hpwl, "p2b_hpwl": p2b_hpwl,
        "total_hpwl": b2b_hpwl + p2b_hpwl,
        "overlap": overlap, "n_overlaps": n_overlaps,
        "bbox_area": bbox_area,
        "soft": soft,
        "used_lff_fallback": used_lff_fallback,
        "legalize_tag": tag,
        "legalize_time": t_legalize,
    }


def _warmup_worker(_=None):
    """v6.3：給 ProcessPoolExecutor 熱身用的 no-op。`ProcessPoolExecutor`
    建構子本身不會真的 spawn worker process——第一個 `.map()`/`.submit()`
    呼叫到的時候才會，而 worker 一啟動要先重跑一次這個模組（跟它遞移
    import 到的 torch/numpy/cvxpy 等）的 import，這筆一次性成本在 Windows
    spawn 語意下不小。官方 evaluate 量到過：第一個 test case 的 runtime
    比其他樣本高出好幾倍（tid=0 的 rt_mult 飆到 1.55，其他樣本大多落在
    0.7-0.9），全部都是這個冷啟動成本，跟樣本本身的難度無關。在
    `MyOptimizer.__init__` 建好 pool 之後立刻呼叫這個函式把每個 worker
    都熱身一次，把成本挪到 `__init__`（不計入任何 test case 的 runtime）
    裡吸收掉，而不是讓隨機哪個先跑到的 test case 揹這筆帳。"""
    return True


def _legalize_worker_v2(args):
    """v5.37b 同款：ProcessPoolExecutor 的 worker 進入點，必須是 module-level
    函式才能在 Windows（spawn）下被 pickle 送進子行程。legalize_sample
    全部是純 Python/numpy/cvxpy，不碰 torch/CUDA，丟到獨立行程跑安全。"""
    cand, fixed_args, legalize_kwargs = args
    (areas, W_int, p2b_edges, pins_pos, preplaced_mask, fixed_mask,
     mib_group, cluster_group, boundary_code, target_ll) = fixed_args
    kw = dict(legalize_kwargs)
    kw["verbose"] = False
    return legalize_result_v2(
        cand, areas, W_int, p2b_edges, pins_pos,
        preplaced_mask, fixed_mask, mib_group, cluster_group, boundary_code,
        target_ll, **kw
    )


def legalize_top_k_candidates_v2(candidates, areas, W_int, p2b_edges, pins_pos,
                                  preplaced_mask, fixed_mask, mib_group, cluster_group,
                                  boundary_code, target_ll,
                                  legalize_kwargs=None, n_workers=1, executor=None):
    """跟 inference.py 的 legalize_top_k_candidates 同一套原則跟同一套
    GT-free 選擇鍵（V_relative, total_hpwl, bbox_area）——只是每個候選
    改用 legalize_result_v2（隊友的 legalize_sample）legalize，不是
    legalize_lff。100 樣本 production 設定（DDIM_STEPS=10/N_SAMPLES=14/
    TOP_K_CANDIDATES=5，兩邊都套用這個 top-K 機制）配對驗證過：real_cost
    -20.87%，96/100 勝、0/100 infeasible（詳見 my_optimizer.py 頂部
    的 v6.0 說明）。

    n_workers/executor：跟 legalize_top_k_candidates 同一套平行化模式
    （ProcessPoolExecutor）——legalize_sample 也是純 Python/numpy/cvxpy，
    丟到獨立 process 平行跑一樣安全，沒有這個平行化正式比賽的 runtime
    代價會隨 K 幾乎線性疊加（同一個道理，見 legalize_top_k_candidates
    docstring）。
    """
    legalize_kwargs = legalize_kwargs or {}
    fixed_args = (areas, W_int, p2b_edges, pins_pos, preplaced_mask, fixed_mask,
                  mib_group, cluster_group, boundary_code, target_ll)

    if n_workers <= 1 or len(candidates) <= 1:
        best, best_key = None, None
        for cand in candidates:
            leg = legalize_result_v2(
                cand, areas, W_int, p2b_edges, pins_pos,
                preplaced_mask, fixed_mask, mib_group, cluster_group, boundary_code,
                target_ll, verbose=False, **legalize_kwargs)
            s = leg["soft"]
            key = (s["V_relative"], leg["total_hpwl"], leg["bbox_area"])
            if best_key is None or key < best_key:
                best_key, best = key, leg
        return best

    args_list = [(cand, fixed_args, legalize_kwargs) for cand in candidates]
    n_proc = min(n_workers, len(candidates))
    if executor is not None:
        results = list(executor.map(_legalize_worker_v2, args_list))
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_proc) as tmp_executor:
            results = list(tmp_executor.map(_legalize_worker_v2, args_list))

    best, best_key = None, None
    for leg in results:
        s = leg["soft"]
        key = (s["V_relative"], leg["total_hpwl"], leg["bbox_area"])
        if best_key is None or key < best_key:
            best_key, best = key, leg
    return best


def _build_opt_target_pos(k, constraints, target_pos_hint):
    """
    比照 run_one_sample/screening 腳本已經在用的建構方式：只有
    preplaced/fixed-shape 的 row 帶真實 target，其餘維持 -1（free）。
    `target_pos_hint` 是 GT 的 (x,y,w,h)（validation 才有；比賽正式輸入
    是 evaluator 自己給的 target_pos，這裡用 GT 當本地測試的替代來源）。
    """
    opt_target_pos = torch.full((k, 4), -1.0)
    if constraints is None or target_pos_hint is None:
        return opt_target_pos
    cons = constraints
    nc = cons.shape[1] if cons.ndim > 1 else 0
    for i in range(k):
        is_fixed = nc > 0 and cons[i, 0] > 0.5
        is_preplaced = nc > 1 and cons[i, 1] > 0.5
        if is_preplaced:
            opt_target_pos[i] = torch.tensor([float(v) for v in target_pos_hint[i]])
        elif is_fixed:
            opt_target_pos[i, 2] = float(target_pos_hint[i][2])
            opt_target_pos[i, 3] = float(target_pos_hint[i][3])
    return opt_target_pos


def run_one_sample_v2(sample_idx, official, model, config, device,
                       n_samples=6, ddim_steps=100,
                       legalize_kwargs=None,
                       **generate_kwargs):
    """
    比照 inference.py 的 run_one_sample，legalize 步驟改用
    `legalize_result_v2`（隊友的 legalize_sample）取代
    `legalize_top_k_candidates`/`legalize_result`（我們自己的
    legalize_lff）。回傳 schema 跟 run_one_sample 相同，方便直接沿用
    既有的比較/報告工具。
    """
    t_start = time.perf_counter()
    sample = official[sample_idx]
    inputs = sample["input"]
    labels = sample["label"]

    area_target, b2b_conn, p2b_conn, pins_pos_t, constraints_t = inputs
    k = int((area_target != -1).sum().item())
    areas = area_target[:k].numpy().astype(np.float32)

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
            gt_x[i] = float(x_min); gt_y[i] = float(y_min)
            gt_w[i] = float(x_max - x_min); gt_h[i] = float(y_max - y_min)

    W_int = np.zeros((k, k), dtype=np.float32)
    if b2b_conn is not None and len(b2b_conn) > 0:
        for edge in b2b_conn:
            i, j, wgt = int(edge[0]), int(edge[1]), float(edge[2])
            if 0 <= i < k and 0 <= j < k:
                W_int[i, j] = wgt; W_int[j, i] = wgt

    pins_pos = pins_pos_t.numpy().astype(np.float32) if pins_pos_t is not None else None
    p2b_edges = []
    if p2b_conn is not None and len(p2b_conn) > 0:
        for edge in p2b_conn:
            p, b, wgt = int(edge[0]), int(edge[1]), float(edge[2])
            if p >= 0 and 0 <= b < k:
                p2b_edges.append((p, b, wgt))

    constraints = constraints_t[:k].numpy() if constraints_t is not None else None

    total_area = float(areas.sum())
    if pins_pos is not None and len(pins_pos) >= 2:
        px_min, px_max = float(pins_pos[:, 0].min()), float(pins_pos[:, 0].max())
        py_min, py_max = float(pins_pos[:, 1].min()), float(pins_pos[:, 1].max())
        aspect = max(px_max - px_min, 1e-6) / max(py_max - py_min, 1e-6)
        slack = 1.10
        canvas_w = float(np.sqrt(total_area * aspect) * slack)
        canvas_h = float(np.sqrt(total_area / aspect) * slack)
        x_offset = (px_min + px_max) / 2.0 - canvas_w / 2.0
        y_offset = (py_min + py_max) / 2.0 - canvas_h / 2.0
        canvas_bbox = (px_min, py_min, px_max, py_max)
        canvas_source = "pin_bbox"
    else:
        canvas_w = canvas_h = float(np.sqrt(total_area))
        x_offset = y_offset = 0.0
        canvas_bbox = (float(gt_x.min()), float(gt_y.min()),
                       float((gt_x + gt_w).max()), float((gt_y + gt_h).max()))
        canvas_source = "gt_bbox_fallback"

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
            mib_group_arr[i] = int(cons[i, 2])
            cluster_group_arr[i] = int(cons[i, 3])
            boundary_code_arr[i] = int(cons[i, 4])

    fixed_mask_pb = np.zeros(k, dtype=bool)
    preplaced_mask_pb = np.zeros(k, dtype=bool)
    for i in fixed_idx:
        fixed_mask_pb[i] = True
    for i in preplaced_idx:
        preplaced_mask_pb[i] = True
        fixed_mask_pb[i] = True

    t_diff_start = time.perf_counter()
    best, all_results = generate_floorplan(
        model, config, areas, W_int,
        canvas_w=canvas_w, canvas_h=canvas_h,
        x_offset=x_offset, y_offset=y_offset,
        n_samples=n_samples, ddim_steps=ddim_steps, device=device,
        constraints=constraints,
        p2b_edges=p2b_edges, pins_pos=pins_pos,
        gt_w=gt_w, gt_h=gt_h, gt_x=gt_x, gt_y=gt_y,
        **generate_kwargs
    )
    t_diffusion = time.perf_counter() - t_diff_start

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

    print("\n>>> RAW (diffusion output, no legalize)")
    evaluate_and_report(best, W_int, areas, constraints, optimal=optimal, stage="RAW")

    gt_target = np.stack([gt_x, gt_y, gt_w, gt_h], axis=1)
    opt_target_pos = _build_opt_target_pos(k, constraints, gt_target)

    t_legal_start = time.perf_counter()
    # 注意：legalize_sample() 沒有 outline_bbox 這種硬性圍住約束（跟我們
    # 自己的 legalize_lff 不同）——它靠 preplaced (x,y) 對齊 target_ll
    # 當座標系錨點（align_preplaced），沒有 preplaced 時就直接沿用
    # diffusion 給的 pin-anchored frame，不額外約束絕對範圍。
    legalized = legalize_result_v2(
        best, areas, W_int, p2b_edges, pins_pos,
        preplaced_mask_pb, fixed_mask_pb,
        mib_group_arr, cluster_group_arr, boundary_code_arr,
        opt_target_pos,
        **(legalize_kwargs or {})
    )
    t_legalize = time.perf_counter() - t_legal_start

    print("\n>>> LEGALIZED_V2 (teammate's legalize_sample, tag={!r})".format(
        legalized.get("legalize_tag")))
    evaluate_and_report(legalized, W_int, areas, constraints, optimal=optimal, stage="LEGALIZED_V2")

    opt_for_json = {"area": optimal.get("bbox_area", 0.0),
                     "b2b_hpwl": optimal.get("b2b_hpwl", 0.0),
                     "p2b_hpwl": optimal.get("p2b_hpwl", 0.0)}
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
        "constraints_pb": {
            "preplaced_mask": preplaced_mask_pb,
            "fixed_mask": fixed_mask_pb,
            "mib_group": mib_group_arr,
            "cluster_group": cluster_group_arr,
            "boundary_code": boundary_code_arr,
        },
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str,
                     default="checkpoints/model_epoch300_overlap_v4.pt")
    ap.add_argument("--samples", type=int, nargs="*", default=[0, 30, 80],
                     help="sample indices to run (default: one small/medium/large pick)")
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--ddim-steps", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model(args.checkpoint, device)

    ev = ContestEvaluator(data_path="../", verbose=False)
    ev._load_dataset()

    for idx in args.samples:
        print("\n" + "#" * 70)
        print("# sample idx={}".format(idx))
        print("#" * 70)
        torch.manual_seed(idx)
        result = run_one_sample_v2(
            idx, ev.dataset, model, config, device,
            n_samples=args.n_samples, ddim_steps=args.ddim_steps,
        )
