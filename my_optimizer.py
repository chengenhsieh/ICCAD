#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Optimizer

Diffusion + LFF-legalize pipeline (see README.md for the full write-up).

Stage 1 (diffusion, diffusion.py / inference.py:generate_floorplan):
  Force-guided DDIM sampler generates a raw layout. Runs a batch of
  best-of-N candidates together (candidates share one GPU batch, so extra
  candidates cost almost nothing), applies pin/grouping/repulsion/boundary
  forces during sampling, then a short physics-only "post-repel" phase.

Stage 2 (legalize, teammate_legalizer/compaction.py:legalize_sample via
inference_v2.py:legalize_result_v2):
  v6.0（採用，見下方 MyOptimizer 類別 docstring 的完整說明）：改用隊友團隊
  （ICCAD2026-Problem-C/diffusion-floorplanner）的正式 legalizer，取代原本
  的 utils.py:legalize_lff。precedence 圖重建+求解跑 3 輪、grouping/
  boundary 懲罰項直接在凸優化目標裡、求解完接一整串逐步驟都有
  feasibility gate 的收尾 pass（snap_groups/pull_boundary/
  snap_mib_shapes/gap_repair）。保證零重疊、preplaced/fixed-shape
  不可變、面積在 1% 容差內——跟舊版 legalize_lff 同一組 hard constraint
  保證，只是換了一套演算法達成。舊版 legalize_lff 保留在 utils.py／
  inference.py 內未刪除，只是不再是這裡的預設路徑。
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
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

from inference import load_model, generate_floorplan
from inference_v2 import legalize_top_k_candidates_v2, _warmup_worker


class MyOptimizer(FloorplanOptimizer):
    """Diffusion-generate + legalize（隊友 legalize_sample）floorplanning optimizer."""

    # v6.0（採用）：legalize 步驟從 utils.py 自己的 legalize_lff 改成隊友
    # 團隊（ICCAD2026-Problem-C/diffusion-floorplanner）的正式 legalizer
    # `teammate_legalizer/compaction.py:legalize_sample`（vendor 進來的
    # 未修改原始碼，見 teammate_legalizer/README.md）。跟已經否決的 v5.39
    # （compact_joint_convex，只 port 了單次凸優化核心、V_relative 反而
    # 大幅變差）不是同一件事——這次用的是完整出貨版本：precedence 圖
    # 重建+求解跑 3 輪、grouping/boundary 懲罰項直接寫進凸優化目標、求解
    # 完接一整串逐步驟都有 feasibility gate 的收尾 pass（snap_groups/
    # pull_boundary/snap_mib_shapes/gap_repair）。
    #
    # 驗證（inference_v2.py + 100 樣本配對比較，production 等價設定：
    # DDIM_STEPS=10/N_SAMPLES=14，兩邊都套用 TOP_K_CANDIDATES=5 的
    # legalize-then-select，不是只比較單一候選）：real_cost 1.3143→
    # 1.0400（-20.87%），96/100 勝、4/100 負（負的樣本裡有 2 個是隊友
    # 文件自己記載的「約 1/100 凸求解 infeasible、退回較弱結果」的已知
    # 邊界案例），0/100 infeasible（兩邊都是）。平均 V_relative
    # 0.0836→0.0238。詳見 CHANGELOG.md。
    #
    # 舊版 legalize_lff／legalize_top_k_candidates（utils.py／
    # inference.py）保留未刪除，只是不再是這裡呼叫的路徑——如果之後想
    # 切回來或做 A/B，程式碼還在。
    #
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
    # v5.17（採用，v6.0 起未使用——legalize_lff 專屬參數，legalize_sample
    # 沒有對應概念。保留數值跟說明供之後若切回 legalize_lff 或做 A/B 用）：
    # legalize 的 compact_reinsert 搜尋強度（reinsert_sweeps/
    # reinsert_grid_density，原預設 3/12）。34 樣本分層抽樣：真實資料上
    # 第一輪就幾乎收斂，area_gap/hpwl_gap 在所有測試組合下完全不變，
    # grid_density 降到 4 以下 real cost 打平（不再有額外好處也沒有壞處）。
    # 選 sweeps=1/grid_density=4，完整 100 樣本官方 evaluate 確認兩次
    # （在 v5.15+v5.16 之上疊加）：real score 1.1381 / 1.1174，平均
    # 1.128，比 v5.16 的 1.1692 再進步約 -3.5%，0/100 infeasible。見
    # CHANGELOG.md v5.17。
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
    # v6.6（實驗用，預設關閉，見 diffusion.py: _force_wirelength
    # docstring／CHANGELOG.md）：現有四個力（pin/grouping/repulsion/
    # boundary）都跟 b2b 連線權重無關，wirelength 完全交給模型自己學到
    # 的訊號；`legalize_sample` 主目標只 minimize W+H，事後也補不回來
    # （見 v6.5，wirelength-aware legalize 沒有效果）。新增一個依連線
    # 權重把 block 拉向鄰居加權中心的力，inference-time guidance、不用
    # 重新訓練模型（跟 "Chip Placement with Diffusion Models"，
    # arXiv:2407.12282 的做法同一個精神）。**不採用**（見 CHANGELOG v6.6）：
    # 6 樣本篩選看到 area/hpwl 隨強度改善但 V_relative 抵銷掉大半好處
    # （淨效果 -0.9%~-1%），擴大到 20 樣本後訊號沒撐住（逐樣本勝敗接近
    # 50/50），且 strength=0.08 在 20 樣本裡新增 2 個 legalize_sample
    # fallback 觸發（idx=75 原本正常，加了這個力才觸發）、idx=90 出現
    # 單樣本嚴重暴走。0.0 = 跟改動前完全等價，維持關閉。
    WIRELENGTH_FORCE_STRENGTH = 0.0
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
    # v5.34（不採用，v6.0 起未使用——同上，legalize_lff 專屬參數）：
    # compact_merge_cluster_groups 擴大候選搬移搜尋範圍，疊加 v5.33 的
    # cost-aware 閘門。
    USE_EXPANDED_SEARCH = False
    EXPANDED_SEARCH_MAX_PAIRS = 20
    USE_COST_AWARE_GATE = False
    # v5.37（採用）：generate_floorplan() 原本只用 legalize 之前的 raw
    # 指標排序，只 legalize 排第一名的候選——但 legalize（尤其是
    # compact_merge_clusters / compact_merge_cluster_groups）常常大幅
    # 改變 overlap 和 bbox_area，raw 排序不保證跟 legalize 後的真實品質
    # 同序。改成把排名前 TOP_K_CANDIDATES 個候選都各自 legalize，legalize
    # 完之後才用真實的 (V_relative, total_hpwl, bbox_area) 選最終答案
    # （見 inference.py: legalize_top_k_candidates docstring，含跟隊友
    # repo diffusion-floorplanner 的比較）。100 樣本官方 evaluate 前的
    # 篩選：K=5 real cost 1.0519->1.0002（-4.9%），71/100 樣本變好、
    # 14/100 變差。
    #
    # 序列跑 K 次 legalize 在 real cost 上是淨負的（K=2 就已經輸給
    # K=1，runtime 代價完全蓋過品質好處）——`legalize_lff` 是純
    # Python/numpy、不碰 torch/CUDA，可以安全丟到獨立 process 平行跑
    # （LEGALIZE_WORKERS 個 worker，用 ProcessPoolExecutor），平行化後
    # wall-clock 時間不再跟 K 成正比，才讓這個機制轉成真正的淨改善。
    # LEGALIZE_WORKERS 用 min(TOP_K_CANDIDATES, cpu_count) 動態決定，
    # 不要寫死成開發機（12 核）量到的數字——正式比賽硬體是 48 核
    # ICELAKE，核心數比開發機多很多。
    TOP_K_CANDIDATES = 5
    LEGALIZE_WORKERS = min(TOP_K_CANDIDATES, os.cpu_count() or 1)
    # v5.38（不採用，見 CHANGELOG.md）：品質觸發的自適應 step 預算。參考
    # 隊友 repo（diffusion-floorplanner）「先試 16 步，legalize 失敗就升級
    # 到 64、再不行升到 100」的做法——但我們的 legalize_lff 結構上保證一定
    # 「成功」（不會真的失敗），沒有對應的「失敗」訊號可以觸發升級。改成
    # 用 legalize 後的真實 V_relative（不需要 GT，跟正式送測環境一致）
    # 當觸發訊號：第一次用正常的 DDIM_STEPS 跑完，V_relative 還是偏高
    # 就重跑一次用更多步數的 diffusion 取樣，兩次都 legalize 完之後用跟
    # legalize_top_k_candidates 同一套排序鍵 (V_relative, total_hpwl,
    # bbox_area) 選較好的一個。只有「難」的樣本才會付出第二次
    # diffusion+legalize 的成本，容易的樣本完全不受影響——這個設計本身
    # 就是為了閃開這個 session 一路以來讓好幾個機制陣亡的問題（品質變好
    # 但 runtime 代價蓋過去，見 v5.14/v5.31/v5.34 等）。
    #
    # V_REL_THRESHOLD=0.12：100 樣本官方 evaluate（v5.37 run1）量到的
    # V_relative 分布 mean=0.079/median=0.076/p75=0.102/p90=0.136，0.12
    # 大約卡在 p75-p90 之間，抓最難的 15-20% 左右觸發重試，不是隨便選的。
    #
    # 20 樣本配對篩選（同一批 test_id、每個樣本呼叫前都明確
    # torch.manual_seed(BASE+tid)，確保兩邊起點相同——第一次沒有這樣做，
    # 重試機制消耗的隨機數次數不固定，觸發過重試的樣本會讓後面所有樣本的
    # RNG 狀態跟對照組分岔，比較結果嚴重失真，見 CHANGELOG 說明）：
    # retry_steps=30 時 real cost 1.1504→1.2413（+7.9%，9 好/7 壞/4 平）；
    # retry_steps=15 時收斂到 1.2113（+5.3%，9 好/8 壞/3 平），V_relative
    # 反而是三者中最好的（0.0901 vs OFF 的 0.0954）——機制本身有效（挑出來
    # 的候選確實不比原本差），但重試的固定成本（整個 legalize top-K
    # pipeline 要再跑一次）太貴，runtime 只跟著步數減半小幅下降
    # （1.379s→1.277s，不是等比例），再往下調步數也很難真正打平。
    # **不採用**，保留機制當 opt-in（預設關閉），retry 步數留在測出來
    # 較好的 15，不是原本試的 30。
    USE_ADAPTIVE_STEPS = False
    ADAPTIVE_STEPS_V_REL_THRESHOLD = 0.12
    ADAPTIVE_STEPS_RETRY_DDIM_STEPS = 15

    # v6.2（實驗用，預設關閉）：跟 v5.38 同一種「品質不夠好就重跑
    # diffusion」的機制，但觸發訊號完全不同、動機也不一樣。v5.38 用
    # V_relative 當訊號，在 legalize_lff 年代被否決是因為 legalize_lff
    # 結構上保證一定「成功」，V_relative 偏高只是「品質普通」，訊號不夠
    # 強，重試的固定成本划不來。v6.1 換上 legalize_sample 之後多了一個
    # 更明確的訊號：`legalize_result_v2` 現在會回傳 `used_lff_fallback`
    # ——代表 legalize_sample 的凸優化求解對這批候選**全部**判定
    # infeasible（見 CHANGELOG v6.0/v6.1 的 tid=94/98 根因診斷：raw
    # diffusion 輸出嚴重重疊時求解才會失敗，而重疊嚴重度會隨 random draw
    # 大幅波動），退回較弱的 legalize_lff 結果，而不是「這批候選裡最好的
    # 也只是普通」。換一批新的 random draw（不同的 raw 佈局、通常重疊
    # 程度會不一樣）有機會讓 legalize_sample 這次真的解出來，賭的是「同一
    # 個難樣本、換個 seed 有機會避開這次的壞 draw」，跟 v5.38 賭「同一批
    # draw 再跑更多 diffusion step 品質會更好」是不同的機制。
    USE_LFF_FALLBACK_RETRY = False
    LFF_FALLBACK_RETRY_DDIM_STEPS = 15

    # v5.39（不採用，v6.0 起未使用——同上，legalize_lff 專屬參數）：
    # post-legalize 聯合形狀+位置凸優化壓縮，見 utils.py:
    # compact_joint_convex docstring 與 legalize_lff 呼叫處說明。
    USE_JOINT_COMPACTION = False
    JOINT_COMPACTION_AR_BOUND = 8.0
    JOINT_COMPACTION_AREA_TOL = 0.009
    JOINT_COMPACTION_SOLVER = "CLARABEL"

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_path = str(Path(__file__).parent / "checkpoints" / self.CHECKPOINT_NAME)
        self.model, self.config = load_model(checkpoint_path, self.device)
        # 常駐 process pool：跨所有 solve() 呼叫重複使用，攤提 process
        # 啟動 + 每個子行程重新 import torch 的一次性成本（Windows spawn
        # 語意下這個成本不小，每個 test case 都新建 pool 會嚴重低估
        # 平行化的真實效益，見 legalize_top_k_candidates docstring）。
        self._legalize_pool = (
            ProcessPoolExecutor(max_workers=self.LEGALIZE_WORKERS)
            if self.TOP_K_CANDIDATES > 1 else None
        )
        # v6.3（採用）：ProcessPoolExecutor 的建構子不會真的 spawn worker
        # process，第一個 `.map()` 呼叫到時才會，每個 worker 啟動要重跑
        # 一次這個模組（含 torch/numpy/cvxpy）的 import——這筆一次性成本
        # 官方 evaluate 量到過整整轉嫁到「不管哪個先跑到」的那個 test
        # case 頭上（tid=0 的 runtime 因此從其他樣本的 ~1-5s 飆到 8.5-9s，
        # rt_mult 因此比其他樣本高出近 2 倍）。這裡建完 pool 立刻用一個
        # no-op 熱身，把成本挪到 __init__（不計入任何 test case 的
        # runtime）裡吸收掉。
        if self._legalize_pool is not None:
            list(self._legalize_pool.map(_warmup_worker, range(self.LEGALIZE_WORKERS)))

    def __del__(self):
        pool = getattr(self, "_legalize_pool", None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

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

        # -- target_ll for legalize_sample()：跟上面 gt_x/gt_y/gt_w/gt_h
        #    同一份 target_positions、同一組 mask，只是打包成隊友
        #    legalize_sample() 要的 [N,4]=[x,y,w,h] 格式（-1=free；
        #    preplaced 帶完整 (x,y,w,h)；fixed-shape 只帶 (_,_,w,h)）——
        #    這正是 target_positions 本來的 schema，不需要另外重建。
        target_ll = np.full((k, 4), -1.0, dtype=np.float64)
        if target_positions is not None:
            target_ll[fixed_mask, 2] = tp[fixed_mask, 2]
            target_ll[fixed_mask, 3] = tp[fixed_mask, 3]
            target_ll[preplaced_mask, 0] = tp[preplaced_mask, 0]
            target_ll[preplaced_mask, 1] = tp[preplaced_mask, 1]

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
        else:
            canvas_w = canvas_h = float(np.sqrt(total_area))
            x_offset = y_offset = 0.0

        n_cand = max(1, self.TOP_K_CANDIDATES)

        def _attempt(ddim_steps):
            _, all_results = generate_floorplan(
                self.model, self.config, areas, W_int,
                canvas_w=canvas_w, canvas_h=canvas_h,
                x_offset=x_offset, y_offset=y_offset,
                n_samples=self.N_SAMPLES, ddim_steps=ddim_steps, device=self.device,
                constraints=constraints_raw,
                p2b_edges=p2b_edges, pins_pos=pins_np,
                gt_w=gt_w, gt_h=gt_h, gt_x=gt_x, gt_y=gt_y,
                sampler="ddim", post_repel_steps=self.POST_REPEL_STEPS,
                grouping_force_strength=self.GROUPING_FORCE_STRENGTH,
                boundary_nudge_strength=self.BOUNDARY_NUDGE_STRENGTH,
                repulsion_strength=self.REPULSION_STRENGTH,
                wirelength_force_strength=self.WIRELENGTH_FORCE_STRENGTH,
                force_confidence_power=self.FORCE_CONFIDENCE_POWER,
                repaint_resample_steps=self.REPAINT_RESAMPLE_STEPS,
                use_self_cond=self.USE_SELF_COND,
                post_repel_grouping=self.POST_REPEL_GROUPING,
            )
            return legalize_top_k_candidates_v2(
                all_results[:n_cand], areas, W_int, p2b_edges, pins_np,
                preplaced_mask, fixed_mask, mib_group, cluster_group, boundary_code,
                target_ll,
                n_workers=self.LEGALIZE_WORKERS, executor=self._legalize_pool,
            )

        legalized = _attempt(self.DDIM_STEPS)

        # v5.38：只有第一次的結果 legalize 後 V_relative 仍偏高（「難」樣本）
        # 才付出第二次 diffusion+legalize 的成本，重跑一次用更多步數，兩次
        # 都跑完才用同一套 GT-free 排序鍵挑較好的——不是無條件都跑兩次。
        if self.USE_ADAPTIVE_STEPS and legalized["soft"]["V_relative"] > self.ADAPTIVE_STEPS_V_REL_THRESHOLD:
            retry = _attempt(self.ADAPTIVE_STEPS_RETRY_DDIM_STEPS)
            key_first = (legalized["soft"]["V_relative"], legalized["total_hpwl"], legalized["bbox_area"])
            key_retry = (retry["soft"]["V_relative"], retry["total_hpwl"], retry["bbox_area"])
            if key_retry < key_first:
                legalized = retry

        # v6.2：只有選中的候選是靠 legalize_lff 保底（代表這批候選
        # legalize_sample 全部求解失敗）才重跑一次新的 diffusion draw，
        # 賭新的 raw 佈局重疊程度不同、這次能讓 legalize_sample 真的解
        # 出來。兩次都跑完才用同一套 GT-free 排序鍵選較好的一個——換到
        # 更差的話（新 draw 一樣求解失敗、甚至更爛）不會被採用。
        if self.USE_LFF_FALLBACK_RETRY and legalized.get("used_lff_fallback", False):
            retry = _attempt(self.LFF_FALLBACK_RETRY_DDIM_STEPS)
            key_first = (legalized["soft"]["V_relative"], legalized["total_hpwl"], legalized["bbox_area"])
            key_retry = (retry["soft"]["V_relative"], retry["total_hpwl"], retry["bbox_area"])
            if key_retry < key_first:
                legalized = retry

        x, y, w, h = legalized["x"], legalized["y"], legalized["w"], legalized["h"]
        return [(float(x[i]), float(y[i]), float(w[i]), float(h[i])) for i in range(k)]
