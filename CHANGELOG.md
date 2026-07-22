# CHANGELOG

記錄 diffusion + LFF-legalize pipeline（`diffusion.py` / `inference.py` /
`utils.py` / `my_optimizer.py`）逐版的修改內容與決策依據。詳細的實驗數字、
方法論說明見 `method.md`；這份文件著重「每一版改了什麼、為什麼、結果如何」。

---

## 早期版本（v3.1 – v4.3）—— pipeline 基礎架構

這幾版建立了目前 pipeline 的骨架，早於本輪詳細記錄的實驗週期，僅列重點：

- **v3.1 – v3.2**：canvas 座標系（用絕對座標對齊 pin bbox，而非原點固定）、
  raw best 候選的排序鍵改為 `(total_overlap, V_relative, total_hpwl,
  bbox_area)`。
- **v3.5 – v3.7**：`optimal_metrics` / `actual_metrics` 兩組 dict 分開，
  viewer 用官方 optimal 當對照基準、inference 端自己用 `utils.py` 算
  area/HPWL，兩邊用同一套公式才公平比較。
- **v3.8 – v3.9**：導入 GNN-style force-guided DDIM sampler（pin force、
  grouping force、repulsion force、boundary nudge，見下方 v5.0 說明）；
  repulsion 改用 layout bbox 邊界（非固定 [0,1] canvas 邊）；加入
  `clamp_bbox` 把套力後的 block 中心拉回 pin bbox 內；best-of-N 的
  re-noise 邏輯改成「短第二段」；`post_repel_steps` 50→30。
- **v4.0 – v4.1**：訓練端加入 overlap loss；修正 NaN 問題。**Legalize 從
  `legalize_v2`（anchor 網格搜尋 + 一堆事後補丁清重疊/壓緊密度）換成
  `legalize_lff`**（LFF 風格、決定性單趟的 MAXRECTS 自由矩形排布 +
  加權中位數定位）——overlap-free、preplaced/fixed/面積等 hard constraint
  都變成演算法結構上直接保證，不再需要事後修正。
- **v4.2**：新增 EDM（Heun 2 階）取樣器當 DDIM 的替代選項（實驗用，
  `sampler="ddim"|"edm"`）。
- **v4.3**：`ddim_steps` 100→30（11 樣本 A/B 掃過 100/80/60/50/40/30，
  確認 legalize 端能補回品質差距）；新增 `use_amp`（fp16 autocast，
  實驗用，預設關閉）、`post_repel_steps`、`scale_t_windows`
  （force-guidance 的 timestep 窗口是否隨 `ddim_steps` 等比例縮放，
  A/B 驗證無效，預設關閉）。

---

## v4.4 —— `compact_gravity`（不採用）

**改了什麼**：新增一個壓縮 pass，讓每個 block 各自往全域（面積加權）重心
走一小步，撞到人就二分法退讓。

**結果**：100 樣本 A/B 顯示 area_gap / hpwl_gap / V_relative 都輕微變差，
且不會處理「多個獨立衛星群共享同一條邊界鎖」這種情況。

**決定**：不採用，程式碼保留為 opt-in（`use_gravity=False`）。

---

## v4.5 —— `compact_merge_clusters` 家族協同移動（採用）

**改了什麼**：`compact_merge_clusters`（找出彼此貼合的連通分量、把非主要群
的衛星群整體剛體平移靠向主要群）原本用「只要沾到任一邊界約束就整軸鎖死」
的過度保守判斷，改成「移動後用官方 boundary 判定式重算違規數，不能超過
移動前」的驗證閘門。並新增**家族協同移動**：若兩個以上互不相鄰的衛星分量
共享同一條邊界鎖（例如都是 RIGHT-locked），單獨移動任一個都會被閘門擋下，
改成把它們當一個剛體家族、用同一個位移量一起移動。

**結果**：同一個樣本原本 14 個衛星分量卡死 0 個能合併，變成 11 個成功
合併；100 樣本 A/B：area gap 24.7% → 23.8%，legalize 最差情況時間反而
下降（5.10s → 3.58s），V_relative 小幅上升（0.109 → 0.113），0/100
infeasible 不變。

**決定**：採用。

---

## v4.6 —— 補跑第二次 `compact_merge_clusters`（採用）

**改了什麼**：在 `compact_reinsert` / `compact_positions` 之後，pipeline
尾端再補跑一次 `compact_merge_clusters`（`use_second_merge_pass`）。

**為什麼**：`compact_reinsert` 的局部搜尋常會移動原本卡住的衛星分量、
開出新的「彼此貼合」機會，第一次 merge（在 reinsert 之前）看不到這些新
機會。

**結果**：100 樣本 A/B：area gap 24.6% → 23.1%、V_relative 0.112 →
0.106，兩項同時改善，時間幾乎不變（第一輪多半立即收斂）。

**決定**：採用，改為預設開啟。

---

## v4.7 —— `compact_merge_cluster_groups` + HPWL 閘門（採用）

**改了什麼**：新增 `compact_merge_cluster_groups`，直接針對每個 cluster
group 檢查連通性（用跟官方 `compute_cluster_violations` 完全一致的
`_blocks_share_edge` 判定式），把分裂的子塊往組內面積最大的子塊做精確
貼合，並額外要求：(a) boundary/cluster 違規數不變差；(b) 移動前後總
HPWL 增加量不超過 `hpwl_slack_ratio * avg_side`（`avg_side` = block
平均邊長）。放在 pipeline 最尾端（第二次 `compact_merge_clusters` 之
後）——早期版本放在 `compact_reinsert` 之前，會被後續的逐一局部搜尋悄悄
拆散剛建立好的貼合。

**驗證方式**：獨立取樣的 100 樣本 A/B 一開始測 `hpwl_slack_ratio=0`
幾乎沒改善（374→375），改用 **paired** 設計（同一組 diffusion 輸出餵給
不同 legalize 設定，排除掉取樣雜訊）重測才發現訊號被雜訊蓋住：`slack=0`
讓 V_grouping 378→371（0 個樣本變差）；`slack=5.0` 效果更好，378→349
（同樣 0 個變差、24 個變好），area_gap 100 樣本中只有 1 個可忽略的變化，
hpwl_gap 平均代價僅 +0.16%。換算官方 cost 公式，`hpwl_slack_ratio=5.0`
淨效益約 −1.3%。

**決定**：採用，`hpwl_slack_ratio=5.0`，改為預設開啟。

（同批測試也掃過 `post_repel_steps`〔15/30/45/60，雜訊範圍內無單調趨勢〕、
`weight_cluster`〔1.0/2.0/3.0，調高會讓 hpwl_gap 明顯變差〕、放置時
「同 cluster group 貼靠候選位置」偏好〔合成測試看似有效，真實資料 area/
hpwl/V_relative 全面變差，根因是合成測試沒有 B2B/P2B 連線可以競爭〕，
三者皆不採用。）

---

## v4.8 —— `compact_snap_boundary`（不採用，保留 opt-in）

**改了什麼**：比照 grouping 的做法，新增 `compact_snap_boundary`：找出
還沒真正貼到邊的 boundary block，把它所在的剛體貼合分量（卡住時退而只搬
它自己，必要時再嘗試把擋路的 1-2 個 obstacle 一起請出去找新家）推向
layout 目前的真實邊界，用獨立的 `boundary_hpwl_slack_ratio` 當代價閘門。

**開發過程中的兩個 bug**：
1. 連通分量只要含任何一個 preplaced block 就整個跳過——但真實資料壓縮完
   常常整個 layout 收斂成一塊涵蓋幾乎所有 block 的連通分量，導致這個機制
   幾乎永遠是 no-op。修正：連通分量計算時把 preplaced block 視為圖上的
   洞（不可通過、不可移動），只把「經由非 preplaced block 互相貼合」的
   部分當剛體搬。
2. 加了「請走擋路的 obstacle」這層 fallback 後，忘記加「最終 bbox 不能
   變大」的顯式閘門（前兩層的移動目標本身就是現有邊界值，數學上保證不
   會變大，但這層是網格搜尋出來的新位置，沒有這個保證），導致 100 樣本
   測試中有 2-4 個樣本 area 些微變差，修正後補上這道閘門。

**結果**：修好 bug 後，對 30 樣本中 17 個真的有 boundary 違規的案例逐一
測試，只有極少數能被修好（多數情況下追一個具體案例會發現：卡住的原因是
另一個「同樣合法佔位」的 block 剛好擋在直線路徑上，這種情況需要真正的
2D 重新排位，不是單軸滑動能解的），淨效益遠小於 grouping 那次
（估計約 −0.2%），加上額外的複雜度（obstacle 請走邏輯），效益/複雜度
比不划算。

**決定**：不採用，程式碼保留為 opt-in 函式（`use_snap_boundary=False`）。

---

## v4.9 —— `compact_pair_reinsert`（不採用，保留 opt-in）

**改了什麼**：仿 VLSI detailed placement / large-neighborhood-search
文獻的做法，新增 2-block 聯合 remove-and-reinsert：`compact_reinsert`
一次只拔一個 block，看不到「兩個都要挪才能一起讓 bbox 縮小」的組合式
改善。只對 touching graph 上彼此鄰接的 pair 出手，兩種搬移順序都試，
只有讓全域 bbox 面積嚴格變小、且不讓 boundary/cluster/HPWL 變差時才
採用。

**踩到的坑**：一開始放在 `compact_reinsert` 之後、`compact_positions`
之前，雖然每一步本身都保證讓 bbox 嚴格變小，但改變起始點後，後續的
`compact_positions` / 兩次 `compact_merge_clusters` 等貪婪 pass 有時會
走到不同、甚至更差的局部最佳解（合成測試中出現過 +0.36% 的淨退步）。
移到 pipeline 最尾端（所有其他幾何調整都完成之後）解決。

**合成測試 vs. 真實資料的落差**：合成測試（無 boundary/cluster 約束）
100% 不變差、多組 seed 最多 −12.76% bbox，看起來很有效；但真實資料
100 樣本 paired 測試 **0/100 樣本有任何變化**，legalize 時間卻多了
46%。追蹤發現：選 pair 的依據（touching graph 鄰接）在真實資料上恰好
選錯了目標——真實 layout 在這個 pass 之前已經被兩次
`compact_merge_clusters` 和 `compact_merge_cluster_groups` 充分拉緊過，
彼此貼合的兩塊通常是有正當理由才貼在一起，拆開各自重插幾乎必定更差
（追蹤到的案例新 bbox 比原本大 15-20%），安全閘門正確擋下了每一次
嘗試。合成測試看似有效，是因為合成資料沒有 boundary/cluster 約束、
跳過了那兩層額外拉緊機制，留下的是真正可以拆開重排的貼合對。

**決定**：不採用（`use_pair_reinsert=False`）——機制本身正確、安全，
只是在真實資料上是零效益 + 高成本的 no-op。

---

## v5.0 —— Diffusion 採樣 force-guidance 強度調整（採用）

**背景**：v4.4–v4.9 全部都是 legalize 端（生成完之後修）的嘗試，v5.0
第一次改動 diffusion 生成本身。`ddim_sample_with_forces` 裡的幾個力
（pin force、grouping force、repulsion force、boundary nudge，v3.8–v3.9
就已經存在）從第一版開始強度就寫死，從未調過。

**驗證方式**：因為改動的是生成本身，無法像 legalize 端那樣共用同一個
raw 輸出做 paired 比較，改用「同一個 sample idx 固定 `torch.manual_
seed`，讓不同力設定至少從同一組初始噪聲出發」逼近 paired 設計。

**改了什麼／結果**（100 樣本掃描）：
- `grouping_force_strength`：0.015 → 0.030 讓 V_grouping 單調改善
  （364→359→355），但加到 0.050 又惡化（357）——甜蜜點在 2 倍，之後在
  0.030 兩側細掃（0.0225、0.040）確認就是甜蜜點。
- `repulsion_strength`：0.05 → 0.025（減半）讓 area_gap、hpwl_gap、
  V_relative 同時變好；加倍到 0.10 反而 area_gap 變差。後續在 0.025
  兩側細掃（0.0125、0.0375）發現兩側都比 0.025 本身更好，換兩組不同
  random seed 各自跑 100 樣本獨立確認，**`0.0375` 兩次都比 `0.025`
  好**（cost 公式估計約 −0.2%～−0.3%，改善來源兩次不太一致：一次主要
  是 V_grouping、一次主要是 V_boundary，效應本身不大但方向穩定）——
  最終定案 `repulsion_strength=0.0375`。
- `boundary_nudge_strength`：0.05 → 0.025（減半）小幅改善；加倍到 0.10
  讓 V_boundary 反而變差（116→122）。0.025～0.0375 之間細掃打平，
  維持 0.025。

三個力原本都偏強，蓋過了模型自己學到的訊號；三個最佳值合在一起測（不是
簡單相加）：area_gap／hpwl_gap 完全持平，V_relative 從 0.1092 降到
0.1032（主要來自 V_grouping 359→339），換算官方 cost 公式淨效益約
**−1.26%**，是這批推論參數實驗裡最大的一次改善。

**決定**：採用，最終預設值 `grouping_force_strength=0.030`、
`boundary_nudge_strength=0.025`、`repulsion_strength=0.0375`。

**Checkpoint 順帶確認**：同一輪也檢查過是否有更好的 checkpoint 可換
——`model_epoch300_overlap_v4.pt`（目前用的）val_loss=0.1026，是所有
架構相容的 checkpoint 裡最低的（v3=0.134、v2=0.120、overlap=0.133、
best_softconstraints=0.132），確認已經是最佳選擇，不用換。

---

## 目前的官方 evaluate 結果（100 樣本，`my_optimizer.py`）

| 版本 | Total Score | Avg Runtime |
|---|---|---|
| v4.6 | 1.9955 | 2.55s |
| v4.7（兩次獨立評估平均） | ~1.9956 | 2.45-2.51s |
| v5.0（`repulsion_strength=0.025`） | 1.9894 | 2.24s |
| v5.0（`repulsion_strength=0.0375`，目前預設） | 2.0128 | 2.46s（max 6.54s） |

（單次 evaluate 本身有約 ±2% 的雜訊，各版本之間的小幅波動不一定代表真實
差異——真正控制雜訊的是 quasi-paired/paired 100 樣本測試，數字見上方
各版本說明。）
