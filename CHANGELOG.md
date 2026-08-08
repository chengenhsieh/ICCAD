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

## v5.1 —— `compact_reinsert_reshape`（不採用，保留 opt-in）

**改了什麼**：`legalize_lff` 最初的 LFF 貪婪排布階段，每個 block 放置當下
會試幾種長寬比（面積不變，沿用 `legalize_v2` 用的 `_aspect_variants`），
但放完就凍結，後面所有壓縮 pass 都只動 (x, y)、不會重新考慮形狀。新增
`compact_reinsert_reshape`，把 `_aspect_variants` 接進跟 `compact_
reinsert` 一樣的 remove-and-reinsert 搜尋迴圈，候選變成「不同位置 × 幾種
長寬比」的組合。只對沒有 preplaced/fixed/MIB/boundary 約束的 block 開放
重選長寬比（MIB group 成員必須共用同一個形狀、boundary 鎖定的是座標不是
位置，這兩種先跳過不處理），放在 pipeline 最尾端。

**結果**：合成測試（無 boundary/cluster/MIB 約束）多組 seed 從不變差，
最多 −12.21% bbox，看起來很有效；但真實資料 25 樣本 quasi-paired 測試
**22/25 完全零變化**，另外 3 個的差異是浮點數雜訊等級（相對誤差
<1e-6%），實質上等於全部零效果。追一個具體案例（k=21，9/21 block 符合
可重塑條件、搜尋範圍正常）確認不是 bug，是結構性問題：`compact_
reinsert`（純位置搜尋）在 pipeline 較早階段已經把每個 block 能找到的
最佳位置窮舉過，等走到尾端的 reshape pass 時，殘留縫隙已經被壓縮到
「換形狀也擠不進去」的程度；合成資料因為沒有 boundary/cluster/MIB 這些
會觸發額外拉緊機制的約束，留下的縫隙比真實資料多，才會看到假的改善
空間——**跟 v4.9 `compact_pair_reinsert` 是同一個根因**，這是這批「往
pipeline 尾端加新的局部搜尋 pass」類型的嘗試第二次踩到同樣的陷阱。

**決定**：不採用（`use_reinsert_reshape=False`），程式碼保留為 opt-in
函式。

---

## v5.2 —— `compact_gradient_finetune`（不採用，保留 opt-in）

**改了什麼**：v4.4–v5.1 全部都是「離散、一次挪一兩個 block」的區域搜尋
（`compact_pair_reinsert`/`compact_reinsert_reshape` 兩次證明這條路在
真實資料上已經觸頂）。這次改用完全不同的範式——DREAMPlace／ePlace／
RePlAce 這系列類比（analytical）global placement：把所有非 preplaced
block 的 (x, y) 一次性當連續變數，PyTorch autograd + Adam 對一個平滑 loss
（overlap + bbox 面積 + boundary + cluster + HPWL，全部可微分）做聯合梯度
下降，讓所有 block 同時、連續地互相讓位。梯度下降過程不保證任何 hard/soft
constraint，跑完一定投影回 `hard_zero_overlap` + `compact_positions`
保證的合法解，只有 bbox 嚴格變小、boundary/cluster 違規不變差、HPWL 代價
在 slack 內時才採用。

**兩個先發現、先修正的方法論陷阱**：
1. **一次性 warmup 成本污染量測**：最初量測 legalize 時間從 ~0.2-0.4s
   暴增到 1.8-3.8s/樣本，看起來是災難性的開銷。追查發現這是 PyTorch/
   autograd「每個 process 第一次呼叫」才會付的一次性初始化成本（同一
   process 內第二次呼叫起降到 ~0.16s），實際生產環境（單一長跑 process
   跑完 100 樣本）只會付一次，修正量測方式後真實開銷只有 +0.36 秒/樣本。
2. **稠密計算浪費**：overlap／HPWL 原本每步都建構完整 k×k 矩陣，真實資料
   的 B2B 連線通常很稀疏；改成稀疏邊表（HPWL）+ 只算上三角索引
   （overlap），單步成本再降 ~15%。

**lr/patience 掃描**（負面結果）：lr 越高越容易讓 Adam 提早收斂到較差的
局部最優（掃過 1.0/1.5/2.0/3.0 都比 0.5 差，過大的步伐直接跳過真正的好
解）；patience 太低會殺死需要較多步數才找到的真正改善（掃過 12/20 都讓
已知能找到 -4.54% 的案例變成 0%，該案例需要約 130-145 步才能跳出一個
loss 平原）。找不到「免費」的降時間成本方法——最終以 `lr=0.5`、
`patience=30` 改為預設開啟，接受約 +0.36 秒/樣本（總時間 +16%）的成本
換 area_gap 改善。

**修另一個 bug 時意外發現的致命問題**：在跑合成正確性測試時，`test_lff.py`
的 `within_outline` 檢查對 2/12 個案例回報 `False`。追查發現：整個 loss
（overlap／area／boundary-相對自己當下 bbox 邊／cluster-相對組內重心／
HPWL）全部是**平移不變**的（把所有座標同時加一個常數，loss 完全不變）
——這個方向上真正的梯度和理論上是 0，但 Adam 對每個參數獨立做二階動量
正規化、不保留這個「梯度和為 0」的性質，實測幾百步下來會在這個零約束
方向上隨機漂移，把整層 block 帶出 `outline_bbox` 之外而不被投影階段的
`hard_zero_overlap`/`compact_positions`（兩者都只保證彼此不重疊，不保證
回到原本的座標系）發現。追蹤發現連當初拿來當展示案例的那筆 idx=16、
-4.54% 的改善，事後用正確的 outline 檢查一查，**同一條優化軌跡走出來的
候選解本身就是越界的無效解**——只是原本的程式碼從未檢查過這件事。

**修法**：(1) 加一個很小的 `weight_anchor` 平移錨定項，把 movable block
的平均座標拉回優化開始前的平均座標——這是平移方向上唯一有梯度的項，足以
消除漂移且不影響其餘四項的正常優化（梯度方向正交）；(2) 補上真正的硬
gate：投影後明確檢查所有 block 是否落在 `outline_bbox` 內，不符合就直接
拒絕、退回原本已保證合法的解（跟其他 v4.7+ 機制的安全閘門邏輯一致）。

**修好之後的結果**：同一批 30 樣本重新量測，`n_improved` 從 1/30（那筆
-4.54%）變成 **0/30**——原本以為的「淨效益」其實大部分是這個未被發現的
bug 造成的假象：一部分「改善」根本是越界的無效解，被 outline gate 正確
擋下，而真正合法、又比原本明顯更好的解，在剩下的 29 個案例裡從未出現過。
額外時間成本（約 +0.36~0.43 秒/樣本）仍然存在。

**決定**：不採用（`use_gradient_finetune=False`）。`compact_gradient_
finetune` 本體與新加的 `weight_anchor`／outline 硬 gate 保留在
`utils.py` 備用——修正過的正確性基礎已經在，日後如果想換一種平移錨定
方式或優化器設定去找回真正的改善空間，不需要從頭再踩一次這個 bug。

**後續追加（v5.5）——補上真正的 outline containment loss**：weight_anchor
只是「不知道 outline 在哪裡、純粹不讓它漂移」的土法煉鋼，補上
`weight_containment`：對每個 block 直接算「超出 outline_bbox 邊界的
距離」平方懲罰（只在真的超出時才非零），讓優化器一邊優化一邊自己知道
outline 在哪，而不是先自由漂移再事後被 gate 擋下來。30 樣本重測**仍是
0/30**，但追一個具體案例（idx=16）發現真正卡住的瓶頸換了——已經不是
outline，而是「boundary 違規數不變差」這條離散 gate：`compute_boundary_
violations` 判定的是「有沒有真的貼到邊」的 all-or-nothing 結果，跟 loss
裡用連續距離當代理的 `boundary_loss` 對不齊，bbox 縮小的過程中被縮小的
那條邊本身在移動，貼邊判定可能因此翻面，即使 area 改善很大，只要 1 個
block 從「貼邊」變成「差一點」，離散 gate 就整個否決。決定：仍不採用，
但這次是有結構性理由的負面結果（不是優化器選錯或漂移沒修好），`weight_
containment` 保留在 `utils.py`。

**再後續追加（v5.6）——boundary gate 加 slack**：既然瓶頸是 boundary
violation 硬 gate，比照 `hpwl_slack_ratio` 的精神加上
`boundary_violation_slack`（違規數整數容忍度，預設 0＝跟之前完全一樣
的嚴格行為），測 `slack=1`（多容忍 1 個違規）30 樣本 paired：只解鎖了
**1/30** 樣本（其餘 29 個結果不變，代表其他樣本的瓶頸根本不是這條
gate），而那唯一被解鎖的樣本換算 cost 公式（`quality * exp(BETA*V_rel)`
這部分）反而是 **+0.16%（變差）**——V_relative 進 cost 公式是指數項，
一點點違規增加換到的 area 改善划不來，這還沒算上真實的 +0.5 秒/樣本
runtime 成本。**決定**：不採用（`boundary_violation_slack` 預設維持
0）。三次獨立、針對三個不同假設瓶頸的修正嘗試（漂移／containment／
gate 寬容度）全部收斂到同一個結論：這個機制在目前的離散安全閘門架構下，
對這份真實資料沒有可及的淨效益——不是哪一次沒調對，是結構性的。程式碼
（`weight_anchor`／`weight_containment`／`boundary_violation_slack`）
全部保留在 `utils.py` 備用，預設值都退回原本嚴格、無副作用的行為。

---

## v5.3 —— EDM 取樣器品質 A/B（不採用）

**背景**：v4.2 加了 EDM（Heun 2 階）取樣器當 DDIM 的替代選項，但從那之後
只測過 DDIM 自己的步數（100→30），EDM 本身從未真的驗證過品質。

**先修一個公平性 bug**：`generate_floorplan` 的 EDM 分支呼叫
`edm_sample_with_forces` 時沒有把 v5.0 調過的力強度
（`grouping_force_strength`/`boundary_nudge_strength`/`repulsion_strength`）
傳進去，導致 EDM 分支一直在用 `edm_sample_with_forces` 自己內部寫死、
v5.0 調參前的舊值（0.015/0.05/0.05 vs. 新的 0.030/0.025/0.0375）。不修
這個就是拿「DDIM 用新參數 vs EDM 用舊參數」在比，對 EDM 不公平——已在
`inference.py` 補上傳遞。

**測法**：iso 前向計算量比較——DDIM 30 步（30 次 forward）vs. EDM 15 步
（Heun 2 階，每步 2 次 forward，約 30 次 forward），quasi-paired（同一個
sample idx 固定 `torch.manual_seed`）40 樣本。

**結果**：EDM 在每一項都更差——area_gap 23.50%→27.70%（+4.2pp）、
hpwl_gap 16.14%→21.72%（+5.6pp）、V_relative 0.1013→0.1251，40 樣本中
28 個 DDIM 的 area_gap 明顯更好、只有 7 個 EDM 更好。同計算量下 DDIM
全面勝出，沒有必要再花更多步數幫 EDM 追——即使追上，換算 runtime 增加後
的 cost 公式也未必划算。

**決定**：不採用，維持 `sampler="ddim"`。EDM 分支保留（連同這次修的力
強度傳遞 bug）當 opt-in 選項，`edm_sample_with_forces` 本身沒有問題，只是
在這個任務上目前不如 DDIM。

---

## v5.4 —— `n_samples` 掃描找真正上限（不採用，維持 14）

**背景**：method.md 記錄 `n_samples` 之前只調過一次（6→14），理由是
best-of-N 候選共用同一個 GPU batch、時間幾乎不變。這次掃更大的值
（8/14/20/30/40）找真正的邊際效益消失點。

**結果**（30 樣本 quasi-paired）：

| N | area_gap | hpwl_gap | V_relative | avg total time |
|---|---|---|---|---|
| 8 | 24.28% | 15.38% | 0.1111 | 1.56s |
| 14（目前預設） | 24.83% | 14.24% | 0.1083 | 1.59s |
| 20 | 24.29% | 13.54% | 0.1184 | 1.56s |
| 30 | 22.74% | 14.40% | 0.1148 | 1.87s |
| 40 | 22.77% | 14.37% | 0.1032 | 2.29s |

「幾乎免費」的說法只在 N≤20 成立（時間確實打平在 ~1.56-1.59s），但品質
在這個範圍內也沒有明顯改善、純粹是雜訊等級的上下波動。N>20 開始有真實
時間成本（N=30 +17%、N=40 +44%），品質也不是單調變好：area_gap 在
N≥30 確實下降，但 V_relative 反而在 N=20/30 變差、N=40 才又變好——因為
候選排序鍵優先看 total_overlap，N 變大有時只是找到「重疊更小但
V_relative 更差」的候選，不保證每一項都同步變好。換算 cost 公式，N=40
看起來有約 -2.4% 的淨效益，但只用 30 樣本雜訊偏大，且這個估計還沒扣掉
runtime 項（N=40 時間多 44%，換算 cost 公式的 runtime 懲罰後很可能把
這個淨效益吃掉大半甚至轉負）。

**決定**：不採用，維持 `n_samples=14`。訊號太弱、時間成本又是實打實的，
不值得為了雜訊等級的差異冒進。

---

## v5.7 —— 多 checkpoint ensemble（不採用）

**改了什麼**：`run_one_sample` 加上實驗用的 `extra_checkpoints` 參數
（`[(model2, config2), ...]`）。給了的話，除了主要模型外，對每個額外
checkpoint 用同一組 areas/W_int/canvas/constraints 也跑一次
`generate_floorplan`，把所有候選（主要 + 額外）用同一套排序鍵
（total_overlap, V_relative, total_hpwl, bbox_area）合併重選，取代原本
只從單一模型的 `n_samples` 個候選裡選——不用重新訓練，純粹用已有的另一個
checkpoint 增加候選池多樣性。legalize 仍然只跑一次（只送最終選出的單一
best 進去），時間成本只在 diffusion 端跟 checkpoint 數量成正比。

**測法**：`model_epoch300_overlap_v4.pt`（目前預設，val_loss=0.1026）+
`model_epoch300_overlap_v2.pt`（次佳，val_loss=0.120）兩個 checkpoint
合併候選池，vs. 純 v4，quasi-paired 30 樣本。

**結果**：area_gap 24.21%→23.46%（小幅改善）、hpwl_gap 13.43%→13.41%
（打平）、V_relative 0.1238→0.1262（小幅變差）——15/30 樣本 ensemble
真的選中 v2 的候選（不是沒作用，兩個 checkpoint 確實會生成不同結果），
但淨效果被 V_relative 的小幅退步抵銷：換算 cost 公式（`quality *
exp(BETA·V_rel)` 這部分）**+0.25%（變差）**，這還沒算 diffusion 時間
翻倍（1.21s→2.43s）的真實 runtime 成本。v4/v2 來自同一個訓練系列（只是
不同 checkpoint/超參數變體），彼此的候選分佈可能高度相關、不夠互補；
這個結果也跟 v5.4 的 `n_samples` 掃描結論一致——不管是同一個模型多取樣、
還是跨 checkpoint 多取樣，只要候選池本身的品質分佈夠集中，加大候選池
都很難再找到真正更好的解。

**決定**：不採用。`extra_checkpoints` 保留在 `run_one_sample` 當 opt-in
參數（預設 `None`，不影響任何既有呼叫）。

---

## v5.8 —— 訓練端 timestep 加權 soft constraint loss（不採用，維持 v4）

**背景**：v5.2-v5.7 都是推論端（生成完之後的後處理/取樣策略）的嘗試，
在真實資料上一致找不到淨效益，逼近「後處理已經榨乾」的天花板。這是這輪
第一次改動訓練本身。

**改了什麼**：`diffusion.py` 的 `training_loss` 裡，soft constraint loss
（mib/cluster/boundary/overlap）原本對所有隨機取樣到的 timestep t 一視同仁
套用同樣力道，但這些 loss 全部是從 `x0_pred = (x_t - sqrt(1-ᾱ_t)·noise_
pred) / sqrt(ᾱ_t)` 反推出來的——t 越大（雜訊越多）這個反推在數值上越不
穩定，x0_pred 這時候基本上不可信（原本只有 clamp 防 NaN/inf，沒有依可信
度調整這些 loss 的貢獻）。新增 `config.weight_soft_loss_by_alpha_bar`
（預設 `False`），開啟後把訓練 loop 裡本來就會算的 ᾱ_t 直接拿來當每個
sample 的權重，讓 t 小（雜訊少、x0_pred 可信）的樣本在 soft loss 裡佔比較
大的份量——不需要新增任何要另外調的超參數。`_group_variance_loss`／
`_soft_constraint_loss` 同步改成先攤成 per-sample 再視情況加權平均，並用
pure unit test（合成 tensor，不牽涉訓練/資料集）驗證過關閉這個旗標時跟
改動前的訓練行為數學上完全等價（mib/cluster/overlap 三項精確一致，
boundary 項刻意維持原本的全 batch pooled 算法不動）。

**30 epoch 短跑驗證**（`weighted` vs `unweighted`，固定 seed=42，只差這
一個變因）：`val_loss`（去噪 MSE）從 0.1993 降到 0.1894；更重要的是拿兩個
30-epoch checkpoint 直接跑真實資料 inference（30 樣本 quasi-paired）：
raw overlap（legalize 之前，最直接反映模型本身生成品質）990→822（明顯
且一致地變好），area_gap 27.46%→24.22%（21/30 樣本更好），但 V_relative
只小幅改善且逐樣本勝率接近五五波。訊號夠有希望，決定投入完整 300 epoch
正式訓練（`model_epoch300_overlap_v5.pt`）。

**300 epoch 正式訓練結果**：
- `val_loss` 單一最後 epoch 讀數（v4: 0.1026 vs v5: 0.1362）具誤導性——
  兩個模型的 val MSE 曲線在整個訓練過程都劇烈震盪（100 樣本 validation
  set 雜訊很大，在 0.05~0.30 之間來回），單一 epoch 的讀數不能代表真實
  差異；更穩定的 train MSE 趨勢兩者其實相近（v5 略低）。
- 真正的下游 inference 比較（100 樣本 quasi-paired，跟 v4 目前 production
  checkpoint 直接對比）：raw overlap 1004.3→848.8（**-15.5%，83/100 樣本
  更好**，這個信號在 300 epoch 依然穩固），hpwl_gap 15.70%→14.68%（小幅
  改善）；但 area_gap 23.42%→23.73%（**略變差**，只 42/100 樣本更好）、
  V_relative 0.1054→0.1056（打平，34/100 vs 35/100，逐樣本幾乎是銅板）、
  換算 cost 公式的綜合值只有 **-0.25%**、逐樣本勝率 51/100 vs 49/100
  ——統計上就是打平。也就是說：模型自己生成的原始 layout 確實一致變好，
  但這個優勢在通過 `legalize_lff` 的強力壓縮/去重疊後被大幅抹平，實際
  決定分數的下游指標（area_gap、V_relative）沒有可靠的淨改善。
- 額外拿 `my_optimizer.py`（沿用 v4 調過的其他預設參數，只換 checkpoint）
  跑了一次官方 `iccad2026_evaluate.py`：Total Score 2.0949（v4 先前記錄
  ~2.0128），單次數字看起來更好，但這個專案自己的經驗是單次 evaluate
  雜訊約 ±2%（過去版本比較都用兩次獨立評估平均來過濾這個雜訊），這次
  +4.1% 的落差比典型雜訊帶大，卻跟前面更嚴謹的 100 樣本 paired 比較
  （明確打平）方向不一致——只跑了一次，沒有重複驗證，不足以推翻 paired
  比較的結論。

**決定**：不採用，`my_optimizer.py`／`inference.py` 維持使用
`model_epoch300_overlap_v4.pt`。`weight_soft_loss_by_alpha_bar` 機制與
`model_epoch300_overlap_v5.pt` checkpoint 都保留——這是目前唯一一次讓
「模型自己生成的原始品質」有穩定、一致改善的訓練端嘗試（v5.2-v5.7 的
推論端修改連這一步都做不到），只是這個優勢目前沒能穿透 legalize 後處理
反映到分數上；如果之後 legalize 端有新的、對原始品質更敏感的改動，這個
checkpoint／機制值得重新拿出來配對測試。

---

## v5.9 —— QK-norm + Min-SNR main loss（QK-norm 單獨保留 opt-in，Min-SNR 不採用）

**背景**：v5.8 之後上網查了幾個「跟現有 attention 架構相容、不用換
backbone」的訓練端小改動，挑了兩個文獻上驗證過、彼此獨立、改動範圍小的
來試：

1. **QK-norm**（Dehghani et al. 2023；Qwen3/Gemma3 等近期 LLM 常用）：對
   每個 attention head 的 Q/K 在算 logits 之前先做 RMSNorm，避免 logits
   隨訓練無界增長。`model.py` 新增 `RMSNorm` 類別，經
   `ConnectivityBiasedAttention` → `TransformerBlock` →
   `BlockEncoder`/`Denoiser` → `FloorplanDiffusionModel` 一路傳遞
   `config.use_qk_norm`（預設 `False`）。會新增可學習參數，舊 checkpoint
   不能直接載入這個設定。
2. **Min-SNR-gamma 主要去噪 loss 加權**（Hang et al., ICCV 2023）：
   `diffusion.py: training_loss` 新增 `min_snr_gamma` 參數，依
   `w = min(SNR_t, gamma)/SNR_t` 加權，`config.use_min_snr_main_loss`
   控制開關（預設 `False`）。純 loss 層面改動，不影響模型參數。

兩者都先用合成 tensor 的 pure unit test 驗證過：關閉時跟改動前數學上
完全等價（不是近似），開啟時能正常運算、不產生 NaN/inf。

**效能量測**：合成 benchmark（production 尺寸：d_model=256, batch=64,
N=120）顯示 Min-SNR 幾乎不吃計算成本（1.00x），QK-norm 則有真實的
**+21% 計算開銷**——換成 PyTorch 內建的 fused `F.rms_norm` 結果一樣
（1.22x vs 1.22x），確認這不是「很多小 kernel launch」的可優化開銷，是
這個模型尺寸下 28 次額外正規化運算的真實計算成本。

**30 epoch 短跑，三輪拆解**（固定 seed，quasi-paired 30 樣本真實資料
inference 比較，聚焦 raw overlap 這個「legalize 之前、最直接反映模型
本身生成品質」的指標）：

| 組合 | raw overlap vs baseline | 結論 |
|---|---|---|
| QK-norm + Min-SNR 都開 | 990.0→1880.1（**+90%**，30/30 樣本一致變差） | 明確負面訊號 |
| 只開 Min-SNR | 990.0→1947.1（**+97%**，30/30 樣本一致變差） | 跟上面幾乎同樣嚴重，鎖定真兇 |
| 只開 QK-norm | 990.0→1006.4（+1.7%，噪音等級），area_gap 19/30 樣本明顯較好 | QK-norm 清白，訊號正面 |

Min-SNR 把訓練力氣從「t 小、雜訊少」的步驟移開、讓給「t 大、雜訊多」的
步驟去學全域結構——但「block 間會不會重疊」剛好是一個高度依賴低雜訊
階段精確定位的任務，被 Min-SNR 系統性犧牲掉，這個因果關係跟量測到的
現象吻合，不像巧合。

**QK-norm 完整 300 epoch 訓練**（`model_epoch300_overlap_v6.pt`，
`use_qk_norm=True`, `use_min_snr_main_loss=False`）：
- 100 樣本 paired inference（跟 v4 對比）：area_gap 23.42%→23.12%
  （44/100 較好）、V_relative 0.1049→0.1017（40/100 較好）、raw overlap
  1004.3→945.1（**-5.9%，67/100 較好**，接近 2:1）、hpwl_gap 小幅變差
  （15.72%→16.07%）。換算 cost 公式綜合值 **-0.61%**，55/100 樣本較好
  ——方向一致、有 100 樣本統計支撐，明顯比 v5.8 那次（-0.25%，51:49 幾乎
  是雜訊）更紮實。
- 拿 `my_optimizer.py` 跑兩次獨立官方 evaluate（同 v4.7 的作法，過濾單次
  ±2% 雜訊）：Total Score 1.9803 / 2.0235，平均 **2.0019**，跟 v4 的
  2.0128 相比只差約 -0.5%，落在雜訊範圍內、大致打平——但兩次的 **Avg
  Runtime 都明顯偏高**（2.67s / 2.87s，平均 2.77s，比 v4 的 2.46s 多
  **+12.6%**），兩次都一致偏高、不是單次雜訊，直接對應到前面量測到的
  QK-norm 單步 +21% 計算成本。本地 evaluate 用 `RuntimeFactor=1.0`
  （中性）沒有把這個時間差異真的算進分數，但正式比賽的 cost 公式會把
  runtime 算進去，換算過去會是真實的扣分項。

**決定**：QK-norm 不採用為預設（`my_optimizer.py`／`inference.py` 維持
`model_epoch300_overlap_v4.pt`）——品質面的改善是真的（配對比較跟兩次
evaluate 平均都支持），但代價（真實的 +21% 單步計算成本、+12.6% 完整
pipeline runtime）在換算完整 cost 公式後，不足以確定是淨正效益，整體是
一個溫和但方向不夠壓倒性的取捨，選擇維持現狀。Min-SNR 確認不採用
（`use_min_snr_main_loss=False`）——30/30 樣本一致的 raw overlap 惡化
是結構性的，不是雜訊。`use_qk_norm`／`use_min_snr_main_loss` 機制與
`model_epoch300_overlap_v6.pt` checkpoint 都保留在程式碼／`checkpoints/`
備用；如果之後想在「品質 vs. 時間」這個取捨上做別的選擇（例如比賽情境
本身對 runtime 沒那麼敏感），QK-norm 這個方向的正確性基礎已經在，不需要
重新走一次這輪的診斷過程。

---

## v5.10 —— 座標 Fourier 編碼 `CoordFourierEmbedding`（不採用）

**背景**：上網查文獻找「跟 attention 無關」的訓練端改動方向時，找到
*Chip Placement with Diffusion Models*（ICML 2025）——跟本專案同樣是用
diffusion 生成 2D 座標式佈局，其消融實驗顯示替座標加上 NeRF 風格的
multi-frequency sinusoidal 編碼（`sin(2^k·π·p)`／`cos(2^k·π·p)`,
k=0..n_freqs-1）對定位精度有幫助——跟座標回歸任務普遍已知的「座標本身
是低頻訊號、網路難以直接學到高頻定位細節」現象一致。

**實作**：`model.py` 新增 `CoordFourierEmbedding` 類別（`n_freqs=16`，
輸出 `4*n_freqs` 維），接一個 2 層 MLP（`coord_proj`）投影回
`d_model`，再以**相加**的方式併入 `Denoiser` 的 embedding 加總。刻意
沒有重用既有的 `SinusoidalEmbedding`（那是替 diffusion timestep 調過頻率
範圍的，跟 `[0,1]` 附近的正規化座標不是同一個頻率尺度）。`config.py` 新增
`use_coord_sincos`（預設 `False`）、`coord_n_freqs`（預設 16）。只加在
`Denoiser`，不加在 `BlockEncoder`（後者的座標本來就是要被生成的目標，
不是已知輸入）。單元測試驗證參數量差恰好等於新 MLP 大小，且模型輸出
對座標平移確實敏感（編碼有被正確接進計算圖，不是死代碼）。

**30 epoch 短跑**（quasi-paired 30 樣本）：raw overlap -5.1%（20/30 樣本
較好），area_gap 好壞參半（13:14，接近雜訊等級）——訊號不算壓倒性，但
raw overlap 這個「legalize 前最直接反映模型生成品質」的指標方向一致，
值得跑完整 300 epoch 驗證。

**完整 300 epoch 訓練**（`model_epoch300_overlap_v7.pt`）：

- 100 樣本 paired inference 對比 v4：area_gap 大致打平、V_relative 小幅
  領先、raw overlap 明顯領先（77/100 樣本較好，-5.7%），換算 cost 公式
  綜合值 **-0.53%**。單看這組數字，方向是正面的。
- 但這組 paired 比較是在 **v5.11 違規判定 bug 修好之前**跑的（見下一節）
  ——而 v5.11 修的東西直接影響 legalize pipeline 內部的安全閘門判斷邏輯，
  不只是回報數字，代表 v7 vs v4 的公平比較必須在**同一套（修好的）
  codebase** 下重跑才算數，不能沿用舊結果。
- 拿修好 v5.11 之後的 codebase，`my_optimizer.py` 分別指向 v7 跟 v4 各跑
  兩次獨立官方 evaluate（同慣例作法，過濾單次 ±2% 雜訊）：v7 平均
  **1.5482**，v4（同一套修好的 codebase 下重新評估）平均 **1.5248**——
  v7 其實略差，而且差距落在單次評估雜訊範圍內，不構成可靠的改善。

**決定**：`use_coord_sincos` **不採用**（`my_optimizer.py` 維持
`model_epoch300_overlap_v4.pt`）——30 epoch 短跑跟修 bug 前的 100 樣本
paired 比較都曾顯示正面訊號，但用公平（同一套 v5.11 修正後 codebase）的
官方 evaluate 重新檢驗後，訊號沒有存活下來。這是本 session 抓到「v5.11
修法本身會改變比較基準」這個陷阱之後，主動回頭補做公平驗證才發現的——
提醒之後任何牽涉 legalize 安全閘門邏輯改動前後的比較，都必須注意基準
是否一致。機制與 `model_epoch300_overlap_v7.pt` 保留備用。

---

## v5.34 —— `compact_merge_cluster_groups` 擴大候選搬移搜尋範圍（不採用，官方 evaluate 變異度過大）

**背景**：v5.33 證實 HPWL 閘門鬆緊不是真實資料上的瓶頸——真正限制是
`compact_merge_cluster_groups` 自己的候選搬移邏輯太窄。這次直接針對這個
根因：函式對每個分裂成多塊的 cluster group，找「面積最大的子分量」當
target，其餘子分量逐一嘗試合併，但有兩個具體限制：(1) 每個衛星子分量
只算「距離最近的一對 (衛星組員, target 組員)」，只試 4 種貼齊方向，這一
對被擋住就直接放棄，不會改試次近的配對；(2) 永遠只搬衛星、從不搬
target——姊妹函式 `compact_merge_clusters` 已經有「衛星卡住時改搬 main」
的 fallback，這裡完全沒有對應機制。

**實作**：新增 `use_expanded_search`（預設 `False`，跟改動前完全等價）、
`expanded_search_max_pairs`（預設 20，避免大 group 時配對數平方成長）。
`True` 時：候選配對依距離排序，最多嘗試 `expanded_search_max_pairs`
對（沿用完全相同的 4 方向候選 + 既有安全閘門邏輯，抽成共用的
`_try_abut` 內部函式）；所有「搬衛星」的配對都卡住時，追加一輪「搬
target」的對稱嘗試（target 所在剛體含 preplaced block 則跳過，比照
`compact_merge_clusters` 對 `main_immovable` 的既有處理）。兩個新方向
的每個候選解都還是要通過現有的不重疊／boundary／cluster／HPWL 閘門，
純粹是多給候選方向去試，不改變任何驗收標準。`inference.py`／
`legalize_lff` 一路傳遞（4 點接線）。

**驗證**：
- 逐位元反向相容：`use_expanded_search=False` 跟改動前完全一致。
- 60 組隨機合成 fuzz test：100% 安全（不重疊、boundary 違規恆為 0）；
  `True` 時 20/60 案例找到比 `False` 更多的改善，**0 個更差**。

**真實資料驗證**（疊加 v5.33 的 `use_cost_aware_gate=True`，因為兩者
要一起生效才能讓「搜尋更廣」跟「閘門更聰明」同時發揮作用）：
- 20 樣本：V_grouping 4.25→4.00，**4 個變好、0 個變差**、16 個打平——
  這個 session 整個 grouping 調查裡目標指標最乾淨的一次訊號。real cost
  1.0869→1.0860（5 好/4 壞/11 平），avg_legalize_time +8.7%。
- 100 樣本：V_grouping 3.59→3.43（**16 好/3 壞/81 平，約 5:1**），
  real cost 1.0799→1.0743（-0.52%，33 好/25 壞/42 平），
  avg_legalize_time 只增加 +3.5%（0.981s→1.015s）、0/100 infeasible。

**官方 evaluate 三次獨立跑**：

| | run1 | run2 | run3 | 平均 | 標準差 |
|---|---|---|---|---|---|
| 真實 median runtime 換算分數 | 1.1126 | 1.1338 | **1.1833** | 1.1432 | **0.0296** |

前兩次（平均 1.1232）看起來比 v4 baseline（1.128）好，但第三次出現
明顯離群值（1.1833，中性 Total Score 也同步偏高，不是單純 runtime
波動），把三次平均拉到 **1.1432——比 baseline 差約 +1.3%**，標準差
（0.0296）是 v4 baseline 自身跨跑變異度（v5.17 確認時約 0.0104）的
將近 3 倍。

**決定**：**不採用**（`use_expanded_search`／`USE_EXPANDED_SEARCH`
維持預設 `False`）。這是這個 session 目標指標（V_grouping）篩選訊號
最乾淨、最一致的一次（20/100 樣本均為壓倒性正比例、zero/少量負比例），
卻是官方 evaluate 變異度最大的一次——推測「搜尋更廣」雖然平均而言更常
找到修復 grouping 的機會，但也讓「這一輪 legalize 具體修好哪些、修好後
連帶影響後續貪婪 pass 怎麼收斂」的結果對 diffusion 取樣的隨機性更敏感，
在少數樣本上把 V_relative 以外的其他指標帶往更差的方向，篩選階段用的
平均值/勝負比看不出這種「偶爾大幅波動」的風險，只有官方 evaluate 的
獨立重跑才會現形——這是 v5.14（中性 evaluate 看不出 runtime 代價）之後
這個 session 第二次抓到「篩選指標本身有盲點」的案例，但這次的盲點是
「變異度」而不是某個特定被忽略的維度。機制保留備用
（`use_expanded_search`／`expanded_search_max_pairs`）。

---

## v5.33 —— `compact_merge_cluster_groups` 的 HPWL 閘門改成跟官方 cost 公式對齊（不採用，真實資料上很少觸發到差異）

**背景**：v5.31/v5.32 兩次嘗試從「推論端加 grouping force」修 V_grouping
都沒通過官方 evaluate 的雜訊考驗。這次改從 legalize 階段下手：
`compact_merge_cluster_groups`（v4.7，**production 預設開啟**的機制）
用來決定「值不值得為了修 V_grouping 多花 HPWL」的安全閘門，是一個
經驗性的絕對門檻（`hpwl_slack_ratio=5.0`，「HPWL 增加不能超過 5 個
block 平均邊長」），跟官方 cost 公式（HPWL/area 線性代價、V_relative
指數代價）完全脫鉤——函式自己的 docstring 也承認這是「經驗性折衷，不是
從公式精確反推」，理由是「legalize 在真實推論時拿不到 GT，沒辦法精確算
這步移動對最終分數的淨影響」。

**關鍵設計限制**：真正的 GT（`opt_hpwl`/`opt_area`）在實際比賽的推論
流程裡本來就拿不到——`my_optimizer.py` 被官方 evaluate 呼叫時看不到
GT，所以新閘門**不能**依賴 GT 當輸入，否則沒辦法真的部署到 production。

**設計**：官方 docstring 那句話只對了一半——我們真正要問的不是「這步
移動後 cost 的絕對值」，而是「cost 變好還是變差」，這只需要**符號**，
不需要絕對值。把移動前的目前總 HPWL（函式本來就會算的 `baseline_hpwl`）
當一致的代理分母，兩邊除以同一個東西，符號判斷不需要 GT 就穩健：

```
Δhpwl_frac = (hpwl_after - hpwl_before) / hpwl_before      # GT-free
Δv_rel     = (v_total_after - v_total_before) / N_soft     # 本來就 GT-free
接受 ⟺ (1 + α·Δhpwl_frac) · exp(β·Δv_rel) < 1
```

`N_soft`（= N_boundary + N_grouping + N_mib）只跟約束規格本身有關，不是
GT 答案。`compact_merge_cluster_groups` 新增 `mib_group`（只為了讓
`N_soft` 分母精確）、`use_cost_aware_gate`（預設 `False`，跟改動前完全
等價）、`cost_alpha=0.5`/`cost_beta=2.0`（對齊官方常數）。這個判斷式
只取代 HPWL 那道閘門，**不取代**「boundary/cluster 違規數不能比移動前
差」這兩道既有硬性安全底線。`inference.py`／`legalize_lff` 一路傳遞。

**驗證**：
- 逐位元反向相容：`use_cost_aware_gate=False`（不論有沒有提供 HPWL
  相關參數）跟改動前完全一致。
- 40 組合成 fuzz test：100% 不重疊、boundary 違規恆為 0、cluster
  違規（相對移動前）恆不變差；20/40 案例 `cost_aware` 找到比舊版
  `hpwl_slack_ratio=5.0` **更多**的改善，0 個更少——證實機制本身
  正確、至少不比舊版差。
- 刻意構造案例（5 人 group 被拆成 2 塊，一塊透過另一個真實 B2B net
  連到別的 block）：`hpwl_slack_ratio=5.0` 只部分修好（V_grouping
  4→3），`cost_aware` 完全修好（4→0）。

**真實資料驗證**：20 樣本篩選，效果遠比合成測試小得多——16/20 樣本
完全打平（兩種閘門從頭到尾沒有任何差異），只有 4 個樣本有變化。100
樣本確認：V_grouping **0 個變好、4 個變差、96 個打平**；avg_real_cost
1.0836→1.0843，幾乎打平（24 好/20 壞/56 平）。

**為什麼合成測試效果好、真實資料幾乎沒差**：這個函式的候選搬移邏輯本身
有限——每個 group 每輪只挑「最近的一對 (衛星組員, target 組員)」、只試
4 種貼齊方向，很多真實案例裡候選搬移根本就找不到（被其他 block 擋住、
只能單獨搬 a 自己、preplaced 卡死等），HPWL 閘門只在「候選搬移已經
幾何可行、唯一卡住的是 HPWL」這個相對窄的情境下才有機會發揮作用——這個
情境在真實資料上顯然不常見，96-100% 的樣本兩種閘門完全沒有差異可言。

**決定**：**不採用**（`use_cost_aware_gate` 維持預設 `False`）。機制
本身設計正確、驗證安全，但真實資料上很少觸發，不值得為了幾乎打平的
淨效果承擔任何風險或維護成本，也不需要再花成本跑官方 evaluate 確認。
這次的落差再次印證：這個 legalize 階段的候選搬移**搜尋範圍**本身
（而不是安全閘門的鬆緊）才是真正的限制——跟 v5.24/v5.27 探查
`compact_merge_clusters`／`compact_seqpair_grouped` 時得到的結論
一致，可能要往「搜尋更多候選搬移方式」而不是「閘門更聰明」的方向
才能真正突破。機制保留備用（`use_cost_aware_gate`）。

---

## v5.32 —— post-repel grouping force 強度獨立掃描（不採用，20 樣本訊號在 100 樣本反轉）

**背景**：v5.31 的 post-repel grouping force 沿用主迴圈已調校過的
`grouping_force_strength=0.030`，但那個值是在跟 pin/repulsion/boundary
force 互相競爭的情境下調出來的（v5.0）——post-repel 階段只剩
repulsion／boundary 兩個對手，同一個數值不一定是這個情境下的最佳點。
使用者選擇先試這個成本最低的方向：把 post-repel 的 grouping force
強度獨立出來（`post_repel_grouping_strength`，預設 0.030 = 跟改動前
行為相同），單獨掃描。

**實作**：`diffusion.py: ddim_sample_with_forces` 新增
`post_repel_grouping_strength` 參數，post-repel 迴圈裡的 `_force_
grouping` 呼叫改用這個值（不再固定沿用主迴圈的 `grouping_force_
strength`）。`inference.py` 一路傳遞。

**20 樣本掃描**（`{0.015, 0.030, 0.050, 0.075, 0.10, 0.15}`）：
`strength=0.05` 明顯突出——avg_real_cost 1.0959→1.0543（**-3.8%**），
是整張表裡唯一一個好壞樣本比對 off 有利的設定（12 好/8 壞；其餘強度
都是好壞打平或不利）。V_grouping 3.70→3.35 同步改善。

**100 樣本確認**：訊號完全反轉——avg_real_cost 1.0898→**1.1053
（+1.4%，變差）**，45 好/55 壞（不利）。這是這個 session 第二次遇到
「20-34 樣本篩選連方向都判斷錯，不只是誇大幅度」的情況（第一次是
v5.21 boundary_nudge_strength），再次印證小樣本篩選的雜訊量級在這個
問題上經常跟真實訊號同量級，方向本身都不可靠。

**決定**：**不採用**（`post_repel_grouping_strength` 維持預設 `0.030`，
`post_repel_grouping` 本身也維持 v5.31 的結論——關閉）。100 樣本已經
是決定性的負面結果，不需要再花成本跑官方 evaluate 確認。連同 v5.31，
「在 post-repel 階段加 grouping force」這整個方向（不論強度）目前看
起來都無法穩定地帶來真實改善——問題可能出在 post-repel 是純物理、不
知道 wirelength／boundary 這些其他目標，硬拉近同組 block 中心的副作用
（hpwl_gap、V_boundary 都觀察到系統性變差）恰好抵銷掉 V_grouping 的
改善，且這個抵銷關係本身就有相當大的樣本間變異，不會因為調整單一個
強度參數而變得穩定。機制保留備用（`post_repel_grouping_strength`）。

---

## v5.31 —— post-repel 階段加入 grouping force（不採用，3 次官方 evaluate 平均在雜訊範圍內）

**背景**：使用者要求比對 `quick_eval_solutions_ddim_legalized.json`（我們
的 legalized DDIM 輸出，100 筆真實測試案例）跟 `optimal_solutions.json`
（GT），找出目前最嚴重的問題並對症下藥。用兩份 JSON 已經算好的
`actual_metrics`/`optimal_metrics` 加上 `utils.py` 的官方對齊違規函式，
不用重新跑推論就能算出完整 100 樣本的診斷：

| | mean | 對 cost 的邊際貢獻（log 尺度） |
|---|---|---|
| area_gap | +22.9% | 0.179 |
| hpwl_gap | +16.4% |（併入上面）|
| V_relative | 0.108 | **0.216**（比面積+HPWL項還高）|

拆開 V_relative 三個分量，**grouping violation 出現在 97/100 樣本
（97%）**——比 boundary（65%）、mib（8%）都普遍得多，且對 cost 的邊際
貢獻比面積+HPWL項加起來還大，是目前最嚴重、最普遍的問題。

**找根因**：抓最嚴重的案例（test_id=70, V_grouping=11）逐 block 檢查，
同一個 grouping group 的成員常常相距 50-100+ 單位（畫布約 150×230）——
不是差一點點沒貼齊，是根本在完全不同的區域。這種規模的落差，
`compact_merge_cluster_groups`（legalize 階段既有機制）的安全閘門
（HPWL 代價門檻）會正確拒絕合併，問題出在取樣過程本身沒把同組 block
拉近，不是 legalize 沒做好。追查 `diffusion.py` 發現：現有的
`_force_grouping`（拉同組 block 中心靠近）只在主 DDIM 迴圈內套用，
post-repel 階段（純物理收尾迴圈）只有 repulsion／boundary nudge，沒有
grouping force；而且 v5.15 把 `DDIM_STEPS` 30→10 之後，grouping force
能作用的步數只剩三分之一，post-repel 沒有補上這個缺口。

**實作**：`diffusion.py: ddim_sample_with_forces` 新增 `post_repel_
grouping` 參數（預設 `False`，跟改動前完全等價），`True` 時在 post-repel
迴圈裡加一個 `_force_grouping`（純物理、無 model forward，成本極低），
用跟主迴圈相同的 `grouping_force_strength`。`inference.py` 一路傳遞。
單元測試：機制本身正確（同組 block 平均中心距離確實縮小，0.142→
0.123）；「逐位元重現」的檢查一開始失敗，追查發現是已知的 GPU
非決定性（`scatter_add_` atomic 運算，config.py 的 `seed` 說明本來就
寫明「不保證 GPU 運算逐 bit 重現」），差異量級 ~1e-7，改用寬鬆容忍度
確認後正常，不是真的 bug。

**真實資料驗證**：
- 20 樣本：V_grouping 4.60→3.65（-20.7%，10 好/3 壞/7 平），real cost
  1.0685→1.0435（-2.34%，12 好/8 壞）——這個 session 目前為止推論端
  實驗裡最強的正面訊號。
- 100 樣本：訊號方向一致但幅度收斂（符合這個 session 一貫的「篩選階段
  效果會被稀釋」現象）——V_grouping 3.69→3.35（47 好/26 壞/27 平，約
  2:1 有利），real cost 1.0827→1.0785（-0.39%，56 好/44 壞）。副作用：
  V_boundary 略微變差（1.18→1.36，把同組 block 拉近偶爾會把 boundary
  鎖定的 block 拉離它的邊）、hpwl_gap 也略微變差（+2%，post-repel 是
  純物理、不知道 wirelength）。

**官方 evaluate 三次獨立跑**（`POST_REPEL_GROUPING=True`，其餘沿用
production 設定）：

| | run1 | run2 | run3 | 平均 | 標準差 |
|---|---|---|---|---|---|
| 真實 median runtime 換算分數 | 1.1105 | 1.1433 | 1.1186 | **1.1241** | 0.0140 |

對照 v4 baseline（v5.17 確認值 1.128）：平均看起來微幅變好（-0.35%），
但三次跑之間的標準差（0.014，約 1.4%）比這個微小的平均差距還大，而且
**run2（1.1433）本身就高於 baseline**——不是「三次都一致優於 baseline」
（比照 v5.15/16/17 那種每次都在 baseline 之下的一致模式），是跨過
baseline 兩邊都有，落在雜訊範圍內，不構成可信賴的真實改善。

**決定**：**不採用**（`post_repel_grouping` 維持預設 `False`，
`my_optimizer.py` 的 `POST_REPEL_GROUPING` 改回 `False`）。這次的診斷
方法論（直接比對已存的官方 JSON、不用重跑推論就能找出「哪個問題最
嚴重」）跟根因分析（post-repel 缺少 grouping force）都是對的方向，
20/100 樣本篩選也確實看到目標指標（V_grouping）朝正確方向移動，但
放大到官方 evaluate 的獨立重跑後，淨效果被 hpwl/boundary 的輕微副作用
抵銷、被跑與跑之間本來就存在的隨機變異蓋過——這是這個 session 第三次
遇到「篩選階段訊號方向正確、但強度不足以撐過跨獨立跑的雜訊」的情況
（前兩次是 v5.19 幾何增強、v5.30 x0_pred 評分）。機制與診斷腳本
（比對兩份官方 JSON 找最嚴重問題的方法）保留備用。

---

## v5.29 —— `lambda_area` packing density 訓練端 soft loss（未實際執行，使用者改為專注推論端）

**背景**：v5.22-v5.28 九個 legalize 階段的機制全部不採用後，使用者同意
換一個槓桿：從訓練端加一項 packing density（bbox 面積）soft loss，讓
model 一開始預測出來的座標就更緊。

**實作**：重用 `_soft_constraint_loss`（v4.0）已經在算 overlap loss 時
算好的每個 block `x_left/x_right/y_bot/y_top`，直接算整個預測佈局的
bounding box 面積除以 block 總面積當 loss（`area_loss_ps`，數學上恆
`>= 0`，對應這個 session 全程在用的 `area_gap` 定義）。`config.py` 新增
`lambda_area`（預設 0.0）、`diffusion.py: _soft_constraint_loss`／
`training_loss` 接上計算與回傳、`train.py: _soft_weights_for_epoch`
接上傳遞。不改 `model.py`，沒有新增可學習參數，舊 checkpoint 對
inference 完全相容。單元測試驗證：`lambda_area=0.0` 時逐位元跟改動前
相同；`>0` 時梯度有限、`area_loss_ps` 數值 `>= 0`（驗證 masked min/max
padding 處理正確）。`train.py` 一度設定好 30 epoch 短跑（`lambda_area=
0.1`）等使用者手動執行。

**決定**：使用者決定不繼續投入這個方向，改為專注在推論端——**沒有實際
跑過任何訓練**，所以不算「測試後不採用」，是「實作完成、單元測試通過，
但沒有機會驗證真實效果就先擱置」。程式碼保留（`lambda_area` 預設
`0.0`，跟改動前完全等價），`train.py` 的 `__main__` 恢復成沒有排定中
實驗的中性狀態。之後如果要重啟這個方向，接線都還在，可以直接繼續走
「30 epoch 短跑 → 真實資料快速檢驗 → 完整 300 epoch → 官方 evaluate」
這條既定路徑。

---

## v5.30 —— best-of-N 評分改用 `x0_pred`（不採用，訊號在 100 樣本被稀釋到雜訊範圍）

**背景**：使用者要求上網查推論端還有什麼改善方向。查到 SMC/particle
resampling 類文獻（中途淘汰差的候選、保留好的），本來要提議實作，但動手
前先查了現有程式碼，發現這個核心想法（依分數 softmax 加權重抽樣，取代
硬性 argmin）**已經在 v5.13（`resample_temperature`）測試過、而且不
採用**——30 樣本 quasi-paired 掃過 4 組 temperature，全部變差樣本數 >
變好樣本數，診斷認為是「中途唯一能算的分數（overlap）沒辦法可靠預測
最終品質」。

使用者要求不要直接放棄或直接換方向，先深入分析 v5.13 失敗的細節、看能
不能對症下藥。追查 `select_metric_fn` 的實際呼叫方式，發現一個更基礎的
粗糙點：v5.13（以及 v5.13 之前的原始 best-of-N）餵給 `select_metric_fn`
的是這一步 DDIM 更新算出來的**下一個雜訊 state**（`x`），不是
`x0_pred`——model 對「乾淨最終佈局」的估計其實在 `_one_diffusion_step`
內部已經算好，只是沒有被傳出來給評分用。連現有唯一在用的 overlap 分數，
都是在一個還沒收斂的雜訊訊號上算出來的近似，這比「HPWL/V_relative 算
不出來」更基礎。

**實作**：`diffusion.py: ddim_sample_with_forces` 的 `_one_diffusion_
step` 改成同時回傳 `x0_pred`（純內部 closure，不影響外部介面）；新增
`score_from_x0_pred` 參數（預設 `False`，跟改動前完全等價），`True`
時 resample checkpoint 用 `select_metric_fn(x0_pred)` 取代
`select_metric_fn(x)`，評分公式本身完全不用改（`_select_metric` 的
overlap 計算不在乎輸入是雜訊還是乾淨估計，形狀/語意相同）。`inference.py:
generate_floorplan`／`run_one_sample` 一路傳遞。單元測試驗證：
`score_from_x0_pred=False` 時同一 seed 逐位元可重現；`True` 時
`select_metric_fn` 收到的 tensor 確實跟 `False` 時不同（證實 x0_pred
真的被傳進去，不是靜默退回 x）。

**驗證分兩階段、刻意把「評分品質」跟「軟性/硬性重抽樣」兩個變因分開測**
（避免像 v5.13 一樣把兩者混在一起看）：

30 樣本 quasi-paired（`score_from_x0_pred ∈ {False, True}` ×
`resample_temperature ∈ {None, 0.3, 0.6, 1.0, 2.0}`）：`x0pred_hard`
（只換評分輸入、維持硬性 argmin）比目前 production 行為
（`xnoisy_hard`）real cost 1.1010→1.0901、15 好/12 壞/3 平——支持「評分
品質本身是瓶頸」這個假說；`x0pred_t1.0`（評分+軟性重抽樣一起換）
1.1010→1.0564、22 好/8 壞，是整張表裡最強的訊號。

**100 樣本 paired 確認**（`x0pred_hard`、`x0pred_t1.0` 對照
`off`）：訊號明顯被稀釋——`x0pred_hard` real cost 1.0822→1.0802
（-0.18%），但好/壞樣本數反而**壞比好多**（46 好/52 壞）；
`x0pred_t1.0` 1.0822→1.0768（-0.5%），好/壞打成 53:47 幾乎五五波。
兩者都比這個 session 任何一次**採用**的改動在篩選階段的訊號弱上不只
一個量級（v5.15-17 篩選階段都是 -2%~-16% 且好壞比例明顯偏向好），也比
v5.19（**不採用**，paired -7%~-13% 卻在官方 evaluate 打平）篩選階段的
訊號還弱——依這個 session 已經建立的校準基準，這個強度、這個好壞比的
訊號如果拿去跑官方 evaluate，方向逆轉或打平的機率非常高，不值得再投入
2-3 次官方 evaluate 的成本去確認。

**決定**：**不採用**（`score_from_x0_pred` 維持預設 `False`）。這次的
根因分析是對的方向（x0_pred 確實比雜訊 x 更適合當評分輸入，30 樣本、
特別是隔離出「只換評分」這個變因時，訊號方向也確實一致轉正），但即使
用上這個更合理的評分依據，訊號強度在 100 樣本就已經被稀釋到接近雜訊
範圍——這進一步印證 v5.13 原本的診斷更深一層的含義：不只是「當下能算的
分數不準」，而是這個問題的最終品質（尤其 V_relative、HPWL）有太大一部分
是由 70% 之後的取樣步驟、加上整套 legalize 收尾管線共同決定的，**任何
中途訊號**（不論算得多準）能提供的資訊量本身就有限，這是這個「中途
評分 + best-of-N 選擇」整個機制家族的結構性天花板，不是換一個更好的
分數來源就能突破的。機制保留備用（`score_from_x0_pred`／回傳 `x0_pred`
的 `_one_diffusion_step` 改動）。

---

## v5.28 —— `sa_construct_layout`：重新設計 LFF 初始排布（不採用，合成環境就決定性輸給 LFF）

**背景**：v5.22-v5.27 一共八個「事後在 LFF 已經放好的結果上做局部/聯合
搜尋」的機制全部找不到真實改善。使用者同意投入更大規模的改動，方向選定
「重新設計 LFF 初始排布演算法」。動手前先明確提醒了一個風險：如果只是
把 `compact_seqpair_grouped` 從事後補丁改成主要機制，用同一種搜尋鄰域、
同一個起點，很可能只是換個入口走到同一個局部最佳解——v5.27 已經證實這個
鄰域在 LFF 結果上加碼 1000 次找不到任何改善。這次要真的不一樣，需要
（1）從零開始搜尋、不從 LFF 收斂的結果出發，（2）大幅加大迭代預算，
（3）加入「搬節點」以外的 move（交換節點順序、換長寬比），（4）
multi-start。

**實作**：`sa_construct_layout`——sequence-pair + 模擬退火，從零開始建構
一個保證不重疊、boundary 100% 合規的初始排布，取代 `legalize_lff` 原本
`_attempt()`（MAXRECTS + 加權中位數單趟貪婪放置）。`legalize_lff` 新增
`construction_mode="lff"|"sa"` 參數（預設 `"lff"`，跟改動前完全等價），
`"sa"` 時用這個函式取代初始建構、其餘既有的 compact_* 收尾 pass 原封不動
繼續套用。

**節點建構**沿用 `compact_seqpair_grouped`（v5.27）的抽象——cluster group
當剛體超節點，先用 `_pack_cluster_group_internal`（遞迴呼叫 `legalize_lff`
本身排出組內緊湊佈局）排好、凍結內部相對位置；preplaced block 當固定
節點；其餘個別 block（含 boundary 鎖定的）各自一個自由節點。

**Boundary 合規機制**（開發過程中修了三個逐層深入的正確性 bug，值得記錄）：
Sequence-pair 的 x 座標是「累加左邊所有人的貢獻」，這讓 LEFT/BOTTOM
（零 predecessor）和 RIGHT/TOP（累加所有 predecessor）兩類邊界合規條件
在拓樸上並不對稱：
1. 一開始誤以為 RIGHT 只要放在 Γ-（seq_n）最後面就好（比照 LEFT 放最
   前面），但這只保證「不會把別人推得更右」，不保證「自己被推到最右」
   ——實測驗證後發現須同時是 Γ+ 和 Γ- 的最後一個，才能讓「其他每一個
   節點」都真正被累加進去（TOP 對稱：Γ+ 最後、Γ- 最前）。
2. 同一條邊鎖了兩個以上「RIGHT/TOP 累加型」節點時，即使拓樸上彼此不再
   互推，寬/高不同的節點從同一個 baseline 起算，右/上邊緣仍然不會自動
   對齊（LEFT/BOTTOM 的「零 predecessor」型不受影響，因為那個基準
   0 本身不受寬高影響）——加了 `_align_boundary_edges` 事後在完整節點
   陣列上算出真正的全域極值，把每條邊上的自由節點統一位移過去。
3. 這個事後對齊函式起初只考慮「同條邊鎖定的那組節點自己的極值」，沒考慮
   preplaced 節點可能本來就比這組節點的自然基準更極端（例如 y 座標本來
   就是負的）——把 `_align_boundary_edges` 的極值計算範圍從「同組節點」
   改成「全部節點（含固定節點）」才修好。
   
   同一批修正過程也堵住一個 hard-constraint 洞：最尾端的保底 fallback
   （每條 SA chain 都找不到合法起點時）呼叫 `hard_zero_overlap` 清理
   殘留重疊，最初只凍結「多成員節點」，漏了「個別 preplaced block」，
   一度讓保底路徑可能真的搬動 preplaced block；修好後又補上第二層保底
   （連凍結非 preplaced 剛體節點都清不乾淨時，優先保證零重疊這個更基礎
   的硬指標，放寬到只凍結真正 preplaced 的 block）。

**驗證**：90 組隨機合成 fuzz test（含 preplaced／單一邊界鎖／corner
雙邊界鎖／cluster group／負座標範圍），用官方逐 pair 逐軸判定
（`ox>1e-6 AND oy>1e-6`，不是面積總和）而不是先前誤用的面積門檻：
100% 通過 hard constraint（不重疊、preplaced 位置/形狀不變、面積守恆）；
單一邊界鎖 100% 合規；corner（同時鎖兩條邊）刻意不強制拓樸（v1 已知
限制，交給重試機制碰運氣），對應樣本裡確實會有殘留違規，但不影響
overlap/preplaced 這些更基礎的 hard constraint。

**合成環境品質對照**（30 組隨機案例，k=15-57，`legalize_lff` 完整既有
compact_* 收尾管線不變，只切換 `construction_mode`）：

| | LFF（預設） | SA 建構 |
|---|---|---|
| 平均 bbox 面積比（sa/lff） | 1.000（基準） | **1.202**（+20.2%，更差） |
| sa 贏／輸／打平 | — | 4 / 26 / 0 |
| 平均耗時 | 0.26s | **5.61s**（約 22 倍） |

而且問題規模越大，SA 越吃虧：k=15-19 的小案例 sa/lff 比值常常 <1（甚至
到 0.71），k=45-57 的大案例則普遍 1.3-1.7——sequence-pair 拓樸空間隨
節點數成組合爆炸成長，固定的迭代預算能覆蓋的比例隨 k 變大而急遽下降，
LFF 的貪婪建構是多項式時間、不需要「運氣」就能拿到不錯的解，這個劣勢
結構上無法單純加大迭代數解決（除非成比例地加大到不切實際的量級，
而 legalize 的 runtime 又是這個 contest 評分公式明確在意的項目，見
v5.15-17）。

**決定**：**不採用**（`construction_mode` 維持預設 `"lff"`，`legalize_lff`
行為完全不變）。按照事先訂好的分階段驗證計畫，合成環境就已經是決定性
的負面結果（不只是「沒有優勢」，是持續、隨規模惡化的劣勢，外加無法接受
的 runtime 代價），不需要再花成本往下做真實資料測試。這次的結論比
v5.22-v5.27 更進一步：不只是「事後局部搜尋」這個方向走不通，連「換掉
LFF、改用通用組合最佳化從零建構」這個更大幅度的方向，也在真正開始比較
之前就先被合成環境擋下來了——`legalize_lff` 目前的貪婪 + 十幾輪專門
調校過的 `compact_*` 收尾管線，體現的是這個 session 從 v3 一路累積下來
的大量領域知識（哪些 block 該優先放、加權中位數怎麼平衡各種軟性目標），
這些知識沒辦法被一個通用、無先驗資訊的 SA 搜尋在合理時間內重新發現。
機制（`sa_construct_layout`／`_pack_cluster_group_internal`／
`construction_mode` 開關）保留備用，正確性驗證方法論（官方逐 pair 逐軸
判定、boundary 拓樸強制的完整推導）如果之後還有其他 sequence-pair 相關
實驗可以直接複用。

---

## v5.27 —— `compact_seqpair_grouped`：cluster group 當剛體超節點（不採用，八個機制收斂到同一個結論）

**背景**：v5.26 診斷出真實資料上的縫隙，很多時候是被 cluster 分組的 block
擋住——但 `compact_swap`/`compact_anneal`/`compact_seqpair`（含 v5.25
`relax_boundary`）/`compact_boundary_shelf` 全部都把 cluster 分組的 block
永遠排除在搜尋外，結構上就碰不到「boundary block 移動、連帶把擋路的
cluster group 也一起挪開」這種聯合式改善。使用者同意進行一次有明確範圍的
「重寫」，針對這個具體瓶頸。

**設計**：把每個 cluster group 當成 sequence-pair 拓樸裡的**一個剛體超
節點**（節點形狀 = group 目前的 bounding box，成員相對 group 錨點的偏移量
全程凍結，group 只整體平移、內部佈局完全不動），跟個別的 free/boundary
block 一起參與 `compact_seqpair`（v5.24）同一套 SA 搜尋。核心優勢：
sequence-pair 的 longest-path 解碼在拓樸改變時本來就會重新計算所有非固定
節點的位置，讓 group 變成一個節點，代表單一次被接受的搬遷就可能連帶讓它
自動避開其他節點——不需要另外寫「同時搬兩個東西」的邏輯，這是「一次一個
block」的局部搜尋在結構上做不到的。實作直接複用 `_seqpair_decode`（它本來
就不在乎一個節點是一個真實 block 還是合成的聚合體，完全不用改）。

正確性沿用 `compact_seqpair` 同一套「不信任解碼本身」的驗證模式：完整逐
block 陣列上做 `overlaps` 檢查、`relax_boundary=True` 時額外檢查
`compute_boundary_violations`；**不需要**呼叫 `compute_cluster_violations`
——因為 group 成員偏移量凍結、只整體平移，V_grouping 在數學上保證不變，
這點在 docstring 裡以證明而非假設的方式寫清楚。

**驗證**：
- 40 組隨機合成案例 fuzz test（含隨機 cluster 分組 + boundary code）：
  100% 不重疊、100% boundary 違規不變差、100% bbox 面積不變差，**新增的
  不變量**——每個 group 成員相對自己 group 錨點的偏移量前後逐位元相同
  ——100% 通過，證明「內部佈局凍結」這個保證在實作裡真的成立。
- 刻意構造的驗證案例（LEFT-locked boundary block 貼著 x=0、3 人 cluster
  group 隔著一段距離、2 個無關 free block）：純 `compact_seqpair`（cluster
  被排除）只能靠 free block 把面積從 703 壓到 444；`compact_seqpair_
  grouped` 額外把整個 cluster group 當剛體滑過去貼齊 boundary block，
  面積進一步壓到 **221**，不重疊、boundary 限制依然滿足、group 內部偏移量
  完全保留——證實機制本身正確運作，而且確實能找到純 `compact_seqpair`
  找不到的改善。

**真實資料驗證**（20 樣本，`seqpair_grouped_iters=1000`，
`relax_boundary=True`，在兩個候選 pipeline 位置各測一次——比照 v5.26 的
教訓，一個放在跟其餘 seqpair 家族一樣的中段位置，一個放在
`compact_boundary_shelf` 旁邊的尾端位置）：**area_gap 兩個位置都是
20/20 完全打平**（0 個變好、0 個變差），legalize 時間卻明顯增加（0.86s→
1.71-1.75s，將近兩倍）；V_relative 中段位置打平，尾端位置 1/20 略為變差
（0.031→0.047）。即使是驗證過在合成案例上確實比純 `compact_seqpair` 更
強力的機制，在真實資料上依然一次改善都找不到。

**決定**：**不採用**（`use_seqpair_grouped`／`use_seqpair_grouped_end`
維持預設 `False`）。這是這個 session 針對「怎麼提升 packing density」這個
問題測試的第八個機制（v5.22 swap、v5.23 anneal、v5.24 seqpair、v5.25
relax_boundary、v5.26 boundary_shelf ×2 個位置、v5.27 seqpair_grouped ×2
個位置），全部收斂到同一個結論：**每一個「事後在 legalize 階段做局部/聯合
搜尋」的機制，不論多強力、不論驗證過在合成案例上多有效，在真實資料上都是
零改善**。這已經不是單一機制不夠強的問題——`compact_seqpair_grouped`
明確補上了前七個機制都沒有的「聯合搬遷 cluster group」能力，依然找不到
任何真實改善，代表真實資料上的縫隙結構，跟這個 session 診斷出來、也在合成
案例上成功重現的「boundary block 被孤立的 cluster group 卡住」這個模式，
即使存在，也不是真實資料 area_gap 的主要成因——真正的落差更可能來自：
初始 diffusion 預測本身的品質、preplaced block 位置這種完全不能談判的
硬限制、或者 block 形狀（aspect ratio）選擇在最初 LFF 放置階段就已經
定型、後面任何幾何搬遷都補不回來。要再往下挖，方向必須離開「legalize 階段
的局部搜尋」這整個家族，改成從 diffusion 模型本身（例如訓練時直接把
packing density 當成一個 loss 項）或者從根本重新設計 LFF 的初始放置策略
下手——這兩者的風險與工作量都遠高於這個 session 目前為止試過的任何一個
機制，需要使用者另外決定要不要投入。機制與診斷方法論全部保留備用。

---

## v5.26 —— `compact_boundary_shelf`：視覺化診斷 + 同邊 boundary block 貼齊（不採用，找到比 v5.24 更精確的根因）

**背景**：使用者問「想盡力解決空間使用率問題，可以怎麼做」。v5.22-v5.25
已經證實三種局部搜尋機制（swap/anneal/seqpair/relax_boundary）在真實
資料上全部找不到改善，但那些實驗都是「先假設一種自由度、再測試」——這次
改成先診斷：把 legalized 解跟官方 GT optimal 解逐 block 畫出來比對
（`x,y,w,h` 都從 `run_one_sample` 回傳，新增 `constraints_pb` 欄位帶出
每個 block 的 preplaced/boundary/cluster 分類），對真實資料 20 樣本篩選
中 area_gap 最差的 4 個樣本（tid=75/40/15/30）視覺化。

**發現**：三個樣本都是同一個模式——GT 的 boundary block 沿著同一條鎖定邊
緊密相連、直接貼齊內部的 cluster/free block；我們的解在同一條邊上的
boundary block 之間有明顯縫隙，boundary 那排跟內部主體之間也空一大塊
沒人填。**先排除搜尋解析度問題**：把 `compact_reinsert` 的
`grid_density` 4→40、`sweeps` 1→8 重新跑同樣 4 個樣本，legalized 座標
逐 block 完全相同（不是差距縮小，是連一個 block 都沒有移動）——證實
`compact_reinsert` 的「單一 block cost 必須嚴格變小」貪婪準則結構上碰
不到這種「移動這個 block 本身不會讓 bbox 變小、純粹是幫另一個 block
讓路」的改善。追查 `compact_merge_clusters`（v4.5-v4.9，目前預設開啟）
發現它雖然設計動機正是「主要群聚跟衛星群中間空一塊」，但只對「已經彼此
貼合的連通分量」做整群剛體平移、且只沿鎖定軸方向處理——同一條邊上互不
相鄰的 boundary block（各自是獨立的連通分量）之間，沿著自由軸的縫隙
完全沒有任何現有機制處理過。

**實作**：`compact_boundary_shelf`（見 docstring）：對 LEFT/RIGHT/TOP/
BOTTOM 四個 bit 各自收集「有這個 bit、非 preplaced、非 cluster 分組」的
block（cluster 分組排除比照 `compact_swap`/`compact_anneal` 既有慣例），
依自由軸座標排序後由低到高逐一嘗試滑向前一個 block 的邊緣。安全性：
每步都用二分法找最大安全距離，顯式檢查跟其餘所有 block 不重疊、且用
官方對齊的 `compute_boundary_violations` 確認總違規數不超過移動前，
兩者任一不通過就退回不動。

**驗證**：
- 40 組隨機合成案例 fuzz test（含隨機 cluster 分組）：100% 不重疊、
  100% boundary 違規不變差、100% cluster block 完全不動、100% bbox
  面積不變差。
- 刻意設計的驗證案例（3 個 LEFT-locked block 沿 y 軸有縫隙、2 個內部
  free block）：縫隙完全消除（y 座標從 [0,15,40] 收斂到 [0,5,10]），
  面積 1344.0→980.0（-27%），不重疊、boundary 限制依然滿足——證實機制
  本身正確運作。

**真實資料驗證，第一次（放在第一次 compact_merge_clusters 之後、其餘
壓縮 pass 之前）**：20 樣本中 area_gap **0 個變好、1 個明顯變差**
（tid=75：+36.5%→+40.4%）、V_relative 同一樣本也變差；avg_real_cost
1.1149→1.1157，退步。追查發現**不是 compact_boundary_shelf 自己的移動
讓面積變大**（它有自己的安全閘門，數學上不可能）——而是它改變了後續
`compact_gravity`/`compact_reinsert`/`compact_positions`/第二次
`compact_merge_clusters` 這些貪婪 pass 的起點，讓它們收斂到不同、更差的
最終解。這正是 v4.9（`compact_pair_reinsert`）踩過的同一個坑：任何
「局部保證不變差」的 pass，只要放在後面還有其他貪婪 pass 會重新處理同一批
block 的位置之前，就不能保證對「最終」結果也是單調不變差。

**真實資料驗證，第二次（改放到 pipeline 最尾端，比照 v4.9/v5.1 的
慣例）**：同樣 20 樣本，area_gap **20/20 完全打平**（0 個變好、0 個變
差）、V_relative 1 個變好／1 個變差／18 個打平（大致抵銷）、
avg_real_cost 幾乎無變化（1.0940→1.0930，雜訊範圍內）。放對位置後不再
退步，但也完全沒有找到任何淨改善——**到 pipeline 尾端時，其他壓縮 pass
早就把可用的自由空間用掉了，沒有剩餘空間留給這個機制去關閉縫隙**。

**這次診斷比 v5.24 更精確地定位了根因**：視覺上看到的縫隙是真的、GT 也
證實它是可以被壓縮掉的，但真實資料上「看起來空著」的地方，幾乎都已經被
附近某個 block 占走大半（只是沒有精確貼齊）——單一 block 沿自由軸滑動、
撞到第一個障礙物就停的搜尋，能碰到的空間非常有限。要真正關掉這種縫隙，
需要的是「boundary block 滑動的同時，連帶把擋路的內部 block 也一起挪開」
這種真正的聯合/2D 重新排位，不是任何「一次一個 block、保證不變差」的
局部搜尋能達到的——這跟 v5.24 的結論一致（真正決定 packing 效率的是
constrained block 的整體佈局結構），但這次多了具體的視覺化證據跟兩個
不同 pipeline 位置的對照實驗，把「為什麼碰不到」講得更清楚。

**決定**：**不採用**（`use_boundary_shelf` 維持預設 `False`）。機制、
視覺化診斷腳本方法論保留備用——如果之後要挖「boundary block 聯合內部
obstacle 一起重新排位」這個方向，這裡的診斷工具跟安全驗證機制可以直接
複用。

---

## v5.25 —— `compact_seqpair` 的 `relax_boundary` 選項（不採用，驗證了 v5.24 根因的另一面）

**背景**：v5.24 找到真正的根因後，結尾提到往下挖的方向是「怎麼讓
constrained block 也排得更緊」。使用者接著具體問了其中一種：目前
boundary 鎖定的 block（貼著佈局邊界的那些）在 `compact_seqpair`／
`compact_swap`／`compact_anneal` 裡都被當「不自由」排除在搜尋外，
座標維持原樣不動——但它們同時又常常在佈局最外圍，是不是反而在撐大
整個 bbox？如果放寬讓它們也能被搬遷（但搬完之後仍然貼齊同一條邊界，
不違反 boundary 這個 hard constraint），會不會找到「整體更緊」的解？

**實作**：`compact_seqpair` 新增 `relax_boundary` 參數（預設 `False`，
跟改動前完全等價）。`False` 時「自由」的定義沿用 `compact_swap`／
`compact_anneal` 那套（非 preplaced、無 boundary 鎖定、無 cluster
分組）；`True` 時放寬成「非 preplaced、無 cluster 分組」——boundary
鎖定的 block 也加入搜尋、也會被 `_seqpair_decode` 的 longest-path
重新算座標，不再當常數處理。**正確性用兩層驗證**：除了原本就有的
`overlaps`（不重疊）顯式檢查，`relax_boundary=True` 時每個候選解
**額外**呼叫 `compute_boundary_violations`（v5.11 修正、跟官方判定
逐位元對齊的版本——貼邊定義是相對候選解自己的 bounding box，不是絕對
座標）確認所有 boundary 限制依然滿足；只要有任何一個 boundary block
沒貼齊，候選解直接丟棄。`legalize_lff` 新增對應的 `seqpair_relax_
boundary` 開關，`inference.py` 的 `legalize_result`／`run_one_sample`
比照 `use_seqpair` 同一套模式接上（4 個接線點）。

**驗證**：
- 25 組隨機合成案例 fuzz test（不同 k、隨機分配 1-2 個 boundary code）：
  `relax_boundary=True` 100% 不重疊、100% boundary 限制依然滿足、
  100% 面積不變差。
- 刻意設計的驗證案例（5 個自由 block 聚在原點附近，1 個 RIGHT 鎖定的
  boundary block 被孤立放在遠處 x=50）：`relax_boundary=False` 時
  bbox 面積 495.0（搬遷聚集 free block 後）→136.0，`relax_boundary=
  True` 進一步搬動那個孤立的 boundary block（x=50→x=8，貼著同一條
  右邊界，只是位置換了）把面積再壓到 117.0，全程不重疊、boundary
  限制依然滿足——證實這個機制在「boundary block 真的被孤立、拖累整體
  bbox」的情境下確實能運作、確實能找到改善。

**真實資料驗證**（20 樣本，`seqpair_iters=2000`，`relax_boundary=
False` vs `True`，都疊在 `use_seqpair=True` 之上跟基準 `off` 比較）：
**area_gap 三組完全一模一樣（20/20 打平，`relax_boundary` 沒有在任何
一個樣本額外找到改善）**，`legalize` 時間反而因為搜尋空間變大而增加
（2.35s→2.76s）。跟 v5.24 的 `compact_seqpair` 本身一樣，這一層額外的
搜尋在真實資料上一次改善都沒找到。

**這次的落差說明了什麼**：合成驗證案例證實了機制本身沒問題、確實能在
「boundary block 被孤立」的情境下工作，但真實資料上這個情境幾乎不
發生——推測是因為真實佈局裡的 boundary block 本來就不是孤立的：它們
要嘛已經很自然地貼著佈局其他部分（不像刻意構造的案例被硬放在遠處），
要嘛移動它會立刻在原地製造新的縫隙、抵銷掉貼齊後省下的空間。換句話說，
使用者一開始的假設（boundary block 卡在外圍、增加了 area 或造成
grouping 失敗）在合成案例上是成立的，但不是真實資料 79% packing
efficiency 落差的主要成因——這進一步印證 v5.24 的結論：真正決定 packing
效率的是 60-80% 有限制 block 的整體佈局結構（哪些 block 被分到哪個
cluster、boundary 分配本身怎麼決定），而不是「搜尋不夠力、找不到更好的
擺法」。

**決定**：**不採用**（`seqpair_relax_boundary` 維持預設 `False`，
且父開關 `use_seqpair` 本身也還是 `False`）。機制保留備用，`relax_
boundary` 的正確性驗證（fuzz test + 官方對齊的 boundary 檢查）以後
如果要再往這個方向挖（例如把整個 cluster group 當一個節點一起搬遷）
可以直接複用。

---

## v5.24 —— `compact_seqpair`：sequence-pair 表示法 + 模擬退火（不採用，找到真正的根因）

**背景**：v5.22/v5.23 證實 swap（互換兩個 block 位置）這個 move 類型
本身在真實資料上找不到任何改善——問題可能不在「牽動幾個 block」，而在
swap 沒有「單一 block 搬遷、連帶重新定義它跟一整排其他 block 相對順序」
的表達能力。Sequence-pair 表示法（Murata et al., 1996）正是為了這個能力
設計的：兩條 block 排列（Γ+、Γ-）決定一組 pairwise 拓樸關係，任何一組
排列都能用 longest-path 決定性地解出「給定這個拓樸下最緊的合法擺法」。
這是使用者原本就想試的「換掉 LFF」的一個折衷版本——不是重寫整個排布
邏輯，而是在 LFF+既有壓縮跑完之後，用 sequence-pair 的搜尋能力補一輪
更強力的後處理。

**實作**：`utils.py` 新增 `_seqpair_decode`（sequence-pair -> 座標的
longest-path 解碼器）與 `compact_seqpair`（局部搜尋主體）。跟
`compact_anneal` 用同一套退火接受準則跟排除規則（非 preplaced、無
boundary 鎖定、無 cluster 分組的「自由」block 才會被移動）；被排除的
block 直接把它們目前的座標當已知常數餵給 decode（不參與 longest-path
計算）。**正確性用兩層保護**：(1) decode 本身的 longest-path 公式在
數學上保證同一拓樸下的最緊合法擺法；(2) 但因為排除的 block 只是被
「當常數處理」，沒有嚴格證明對它們一定自洽，所以每次搜尋出候選解都會
額外呼叫 `total_overlap` 顯式驗證，不合法直接丟棄，不信任理論保證。

驗證得非常徹底（這是這個 session 目前寫過最複雜的一個函式）：
- decode 正確性：對一個已知合法佈局，反推它自己的 sequence pair 再解碼
  回去，確實得到不重疊、面積更小或相等的結果（實測 1345.8→562.6，
  找到原佈局自己都沒發現的內部縫隙）。
- 30 組隨機合成案例 fuzz test（不同 k、不同自由/固定比例）：100% 不重疊、
  100% 面積不變差、固定 block 100% 不動。
- edge case（k=0/1/2、全部 block 都固定）：正常運作、不 crash。
- 決定性：給定 seed 後可重現；不同 seed 給不同結果（隨機性確實在探索）。
- 合成測試效果非常好：15 個 block 的隨機網格佈局，1345.8→360.8
  （無限制）／441.0（3 個固定 block），比 compact_swap／compact_anneal
  在類似合成案例上的效果都好上一截。

**真實資料驗證**（20 樣本，`seqpair_iters ∈ {500, 2000}` vs 關閉）：
**area_gap 三組完全一模一樣（0/20 較好、0/20 較差、20/20 打平）**，
legalize 時間卻大幅增加（1.29s→2.77s，2000 次迭代時整體 runtime 增加
超過 70%）——即使是驗證過遠比 swap 更強力的搜尋機制，在真實資料上依然
一次改善都找不到。

**真正的根因**：檢查真實資料的「自由 block」比例才發現關鍵——**平均只
有 ~39% 的 block 完全沒有 boundary/cluster/preplaced 限制**，小案例
（k=21-36）甚至只有 19-26%。合成測試幾乎全部 block 都自由（12-15/15），
真實資料卻只有兩三成，這就是為什麼合成測試效果好、真實資料完全找不到
改善的根本原因：**這三個機制（swap/anneal/seqpair）能碰的解空間，
在真實資料上本來就只佔整個問題的一小塊**，真正決定 packing 效率的是
那 60-80% 有限制的 block，而它們的位置由專門的機制（`compact_merge_
clusters`／`compact_merge_cluster_groups`／boundary 相關邏輯）決定，
這些機制在 session 更早期（v4.5-v4.9）已經被大量調校過。

**決定**：**不採用**（`use_seqpair` 維持預設 `False`）。這次的調查（連同
v5.22、v5.23）把「LFF 造成的 packing 天花板」這個假說徹底釐清了：不是
「排布演算法本身太弱、需要換掉」，而是「絕大多數 block 的位置從一開始
就被 boundary/cluster/preplaced 這些 hard/soft constraint 決定了，跟
排布演算法選 LFF 還是別的沒有太大關係」——79% 這個數字，主要反映的是
「這個問題本身有多少 block 是真正自由可排的」，而不是某個特定排布演算法
的品質上限。要再往下挖，方向會是「怎麼讓 constrained block 也排得更緊」
（例如 cluster group 整體當一個 sequence-pair 節點搬遷），但那已經不是
「換掉 LFF」，而是要動既有的、已經調校過的 constraint-handling 機制，
風險層級不一樣。機制（`_seqpair_decode`／`compact_seqpair`）保留備用。

---

## v5.23 —— `compact_anneal`：swap + 模擬退火（不採用，證實問題不在搜尋策略）

**背景**：v5.22 的純貪婪 `compact_swap` 在真實資料上一次改善都找不到。
假說：貪婪法可能卡在「單步不划算、但連續幾步組合起來划算」的局部最優
（每步都要求嚴格變好，永遠踏不出第一步）——用模擬退火（允許暫時接受
讓 bbox 稍微變差的 swap）驗證這個假說，看能不能跨過這種局部最優的谷。

**實作**：`utils.py` 新增 `compact_anneal`——在跟 `compact_swap` 相同的
排除規則（非 preplaced、無 boundary 鎖定、無 cluster 分組）下，隨機挑
自由 block 配對嘗試互換，用退火機率接受準則（變好必接受、變差以
`exp(-Δarea/T)` 機率接受，溫度隨迭代線性冷卻，用目前 bbox 面積的比例
當溫度尺度），全程記錄看過的最佳解。刻意設計成「給定 seed 後完全決定性
可重現」（用獨立 `np.random.default_rng`，不動全域亂數狀態），維持跟
現有 quasi-paired 測試方法論相容——這是為了讓 SA 的隨機性只用於探索，
不影響既有的可重現驗證流程。單元測試驗證：合成隨機網格上找得到改善、
不會重疊、preplaced/boundary/cluster block 完全不動、給定 seed 決定性、
不同 seed 給不同結果（隨機性確實在運作）。

**驗證**（20 樣本真實資料，`anneal_iters ∈ {500, 2000, 5000}` vs 關閉）：
**area_gap 四組完全一模一樣（0.2165，5000 次迭代——10 倍於最初測的
500 次——結果依然分毫不差）**，legalize 時間卻隨迭代次數持續增加
（0.983s→1.146s，純額外成本，零效益）。

**決定**：**不採用**（`use_anneal` 維持預設 `False`）。這個結果比
v5.22 更明確地指出問題所在：不是「貪婪法卡在局部最優、需要跳脫」，而是
**swap 這個 move 類型本身，在這套 pipeline 跑完之後的佈局上，結構性地
沒有任何可改善的空間**——不管包裝成貪婪窮舉還是模擬退火，結果都一樣，
因為瓶頸不在搜尋策略，在「交換兩個 block 位置」這個動作能觸及的解
空間本身太窄。真正可能有效的下一步是「多個 block 協調搬遷」（例如硬塞
一個 block 進去、連帶擠開好幾個其他 block）——這是 swap（兩兩互換）跟
`compact_reinsert`（單一 block 搬進空位）結構上都做不到的能力，需要
完整的 sequence-pair（或類似）表示法才能表達，工程量遠大於這次的
swap+SA 延伸。機制保留備用。

---

## v5.22 —— `compact_swap`：pairwise swap 局部搜尋（不採用，跟 v4.9 同一種結論）

**背景**：使用者想把 packing density（block 面積 / bbox 面積）從目前的
~79% 拉高到 85-90%+。先確認了 GT optimal 本身的 packing density 高達
~96.9%，代表理論上有真實空間。原本提議兩個方向（outline 迭代收縮、
constraint-graph 式壓縮）追查後發現**都已經在現有程式碼裡實作並在跑**
（`legalize_lff` 內建的 outline 自適應收縮、`compact_positions` 本身就是
constraint-graph 壓縮的等價實作），且程式碼裡的註解明確記錄過「outline
通常已經被 boundary block 撐到最緊，真正的留白是 block 之間的內部碎片」
——於是改試一個真正還沒做過的 move 類型：pairwise swap（直接互換兩個
block 的位置），跟 `compact_reinsert`（只搬單一 block 到目前空著的位置）
不同，swap 能找到「兩個位置都被佔用、但換過來雙方都更省」的改善，這是
reinsert 結構上搆不到的。

**實作**：`utils.py` 新增 `compact_swap`——對所有「自由」block（非
preplaced、無 boundary 鎖定、無 cluster 分組）窮舉兩兩配對，決定性逐對
嘗試互換位置，兩邊都嚴格不重疊且讓 bbox 面積變小才採用（跟
`legalize_lff`「不用 SA」的一貫精神一致，不用隨機取樣，維持
paired 測試方法論相容）。放在 `compact_reinsert` 之後、`compact_positions`
之前。單元測試驗證：合成隨機網格佈局上真的能找到改善（1778→1540）、
不會產生重疊、preplaced/boundary/cluster block 完全不動、決定性（同輸入
同輸出）。

**驗證**：20 樣本真實資料快篩（`use_swap=False` vs `swap_sweeps=1` vs
`swap_sweeps=3`，其餘用目前 production 設定）：**area_gap 三組完全一模
一樣（0.2017，沒有任何差異）**，legalize 時間卻隨 sweep 數增加而變慢
（0.881s→0.891s→0.932s，純額外成本）。合成測試能找到改善，但那是刻意
構造、脫離現有 pipeline 脈絡的場景；在 `compact_reinsert`／
`compact_merge_clusters`／`compact_positions` 這整套既有機制跑完之後，
真實資料上找不到任何「換位置」能省面積的組合。

**決定**：**不採用**（`use_swap` 維持預設 `False`）。這跟 v4.9
`compact_pair_reinsert`（也是嘗試不同的 move 類型，也是 0/100 樣本有變化）
是同一種結論，進一步印證：**79% vs GT optimal 96.9% 的 packing density
落差，看起來是 LFF 決定性單趟排布在下決定當下就定型的拓樸限制，不是任何
形式的事後局部搜尋（不管搬一個 block 還是換兩個 block）能修補的**。要
真正縮小這個落差，很可能需要一個能探索不同拓樸的更強力排布搜尋（例如
sequence-pair 表示法上的輕量 local search/SA），而不是在現有 move 類型
家族裡再加一種——這是比目前 session 做過的任何調整都大的工程量，且有
真實 runtime 風險（多次迭代搜尋通常比單趟 LFF 慢很多）。機制保留備用。

---

## v5.21 —— 重新檢視 boundary_nudge_strength（不採用，34 樣本篩選再次高估效果）

**背景**：v5.0 調 `boundary_nudge_strength`（跟 `grouping_force_strength`／
`repulsion_strength`）時，`DDIM_STEPS=30`；v5.15 把它降到 10 之後，force
guidance 能作用的步數少了 2/3，當初調好的強度理論上可能不再是最佳值——
這是唯一一個「條件已經改變、舊結論可能過時」的推論端方向，動機上比其他
選項更充分。

**掃描結果**（34 樣本分層抽樣，真實 median runtime 換算，在目前 production
設定之上）：

- Phase 1（`boundary_nudge_strength` 單獨掃，`grouping_force_strength`
  維持 0.030）：0.025→0.075 real cost 持續變好（V_relative
  0.1092→0.1048，約 -4%），0.075→0.15 打平或略差。最佳點 0.075。
- Phase 2（`grouping_force_strength` 在 `boundary_nudge_strength=0.075`
  之上重掃）：目前的 0.030 仍是最佳，調高（0.05~0.15）反而讓
  V_relative 變差（0.1091→0.1186）——這個參數沒有額外空間，維持不變。

**完整 100 樣本官方 evaluate 確認**（`boundary_nudge_strength=0.075`，
兩次獨立跑）：real score（換算真實 median runtime）**1.1395 / 1.1573**，
平均 **1.1484**——明顯比 v4 baseline（近期落在 ~1.11-1.128 之間）**差**，
34 樣本篩選預測的 -1% 效益完全沒有撐住，方向反過來。

**決定**：**不採用**（`my_optimizer.py` 維持 `BOUNDARY_NUDGE_STRENGTH=
0.025`）。這是這個 session 第三次遇到「34/100 樣本篩選訊號看起來正面，
完整官方 evaluate 卻打平或惡化」的情況（前兩次是 v5.19 幾何增強、v5.20
v-prediction），但方向更負面——不只是打平，是真的變差。累積的方法論
教訓：**任何只在小樣本（34 或 100）篩選階段驗證過的改動，在正式採用前
都必須跑完整官方 evaluate（至少 2 次獨立跑）確認，樣本數越小，越可能
高估效果**，尤其是像 boundary_nudge_strength 這種同時牽動 area/hpwl/
V_relative 三個互相拉扯的指標的參數，小樣本下的淨效應估計特別不穩定。

---

## v5.20 —— 訓練端 v-prediction 參數化（不採用，但發現一個真實、被抵銷掉的效果）

**背景**：v-prediction（Salimans & Ho, 2022, "Progressive Distillation for
Fast Sampling of Diffusion Models"）把訓練目標從預測 noise（epsilon）換成
預測 v = sqrt(ᾱ_t)·eps − sqrt(1−ᾱ_t)·x0（eps 和 x0 的混合），文獻上在
少步數取樣下通常比 epsilon-prediction 更穩定——本專案 `DDIM_STEPS=10`
（v5.15 之後）屬於相當激進的跳步，理論上正好是 v-prediction 該有優勢的
場景。

**實作**：`diffusion.py` 新增 `GaussianDiffusion._recover_x0_and_eps`
統一介面，把 model 的原始輸出（依 `prediction_type` 是 epsilon 或 v）
轉成下游共用的 `(x0_pred, eps_pred)`——DDIM 更新公式、soft constraint
loss 的 x0 重建、self-conditioning 的估計都只吃這組介面，只有這一個
地方需要依 prediction_type 分支。不改架構、不新增可學習參數。**設計上
刻意跟 `use_self_cond` 不同**：`prediction_type` 不是可以自由選的推論端
行為開關，用錯了會讓整個 DDIM 更新公式解讀錯誤、生成結果變垃圾，所以
讓 `inference.py: generate_floorplan` 直接從 model 自己的 config 自動
帶入（`getattr(config, "prediction_type", "epsilon")`），呼叫方不需要
（也不應該）手動指定，避免重演 v5.18 測試時忘記手動傳 `use_self_cond=True`
的那種失誤模式。驗證：epsilon 路徑跟舊的 inline 公式逐位元完全相同；
v-prediction 的還原公式用真實的前向擴散關係驗證過（給定真實 x0/noise
算出 v-target，反推回去精確等於原本的 x0/noise）；soft loss／
self-conditioning／min-SNR 各種組合都測過無 NaN；真實 dataloader +
backward + optimizer step 也跑過。

**驗證（30 epoch 短跑）**：area_gap／hpwl_gap／V_relative 全部同向變好，
raw overlap -17.4%（28/30 樣本較好），cost-proxy -3.67%（20/30 較好）
——強度接近 v5.19 幾何增強當初的短跑結果。

**完整 300 epoch 訓練**（`model_epoch300_overlap_v10.pt`）：

- **100 樣本 paired** 對比 v4：area_gap／hpwl_gap 同向小幅變好，
  **V_relative 反而略差**（0.1070→0.1099），**raw overlap 大幅、極度
  一致地變好**（1601.3→1268.6，-20.8%，**94/100 樣本較好**，是這個
  session 目前 paired 測試最一致的一次，比 v5.19 的 82/100 還高）。但
  cost 公式的 `exp(2·V_relative)` 是指數項，V_relative 的小幅退步抵銷掉
  了 raw overlap 的巨幅改善，整體 cost-proxy 幾乎打平（**+0.40%**，
  48/100 較好、52/100 較差）。
- 三次獨立官方 evaluate（跟 v4 用同一套 production 推論設定）：real
  score（換算真實 median runtime）**1.1203 / 1.1009 / 1.1576**，平均
  **1.1263**——跟 v4 的 ~1.128 幾乎完全打平，方向跟 100 樣本 paired 的
  cost-proxy（+0.40%，接近打平）一致。

**決定**：**不採用**（`my_optimizer.py` 維持
`model_epoch300_overlap_v4.pt`）。但這次的發現**不是**單純的「訊號太小、
被雜訊蓋過」（跟 v5.19 不同）——raw overlap 94/100 樣本一致變好是一個
真實、量級很大的效果，只是被另一個真實但方向相反的效果（V_relative 略
變差）系統性抵銷掉了，兩者加總後淨值接近零。v-prediction 顯然改善了
模型本身的生成精度（避免重疊的能力），但同時讓 boundary/grouping 這類
soft constraint 的滿足度變差——機制上的原因還不清楚（可能是訓練目標的
數值尺度變化，隱含地改變了不同 loss 分量之間的相對梯度大小，類似
v5.9 QK-norm／v5.8 timestep 加權那類「改變梯度預算分配」的效應）。對
之後想繼續深入的人：如果能找到方法同時穩住 V_relative（例如把
v-prediction 跟某種 soft loss 加權方式結合），raw overlap 這塊真實的
改善空間就有機會被完整兌現成淨分數提升。機制與
`model_epoch300_overlap_v10.pt` checkpoint 保留備用。

---

## v5.19 —— 訓練端幾何 D4 對稱資料增強（不採用，方法論教訓：paired 測試會系統性高估效果）

**背景**：Floorplan 本身沒有全域方向偏好，旋轉/翻轉整個佈局後，所有幾何
約束（overlap、preplaced 相對位置、boundary 對齊、cluster 相鄰、MIB
同尺寸）仍然是同一個合法解——訓練時對每個樣本隨機套用 D4 群（4 旋轉 x
2 鏡射 = 8 個元素，含 identity）的其中一個，把有效訓練資料量乘 8 倍，
純訓練端技巧、不新增可學習參數、推論成本完全不變。

**實作**：`dataset.py` 新增 `_augment_bbox`／`_augment_point`／
`_augment_boundary_code`，在 `FloorplanDataset.__getitem__` 算 canvas
範圍**之前**套用到原始 (x,y,w,h)／pin 座標／boundary bitmask，讓後續
normalize 邏輯自動對新的 bounding box 重新算範圍。`config.py` 新增
`use_geo_augment`（預設 `False`）；`train.py` 只在 train_loader（非
`is_test`）套用。

實作過程抓到一個真的方向性 bug：boundary bit 的旋轉映射一開始寫反了
（憑直覺推導「LEFT 轉 90 度變 TOP」，但用真實座標代入驗證後發現實際上是
「LEFT 轉 90 度變 BOTTOM」——原因是我用了兩種不同方向的旋轉公式做交叉
推導，其中一個其實是順時針、不是逆時針）。寫了整合測試（構造貼在特定
邊界/角落的 block，套用變換後直接檢查座標是否落在程式碼宣稱的新邊界
上，6 種邊界/角落案例 x 8 種變換）才抓出來並修正。其餘驗證：面積守恆、
8 個變換兩兩互為反元素、拓樸/分組/面積在增強前後不變、真實 dataloader +
完整 training_loss（含 boundary soft loss）+ backward 全部正常。

**驗證（第一階段，訊號很強）**：

- 30 epoch 短跑（`test_geoaug` vs `test_unweighted` baseline）：
  area_gap／hpwl_gap／V_relative／raw overlap **四項全部同向變好**，
  raw overlap -12.8%（**28/30 樣本較好**，這個 session 目前 paired
  測試最一致的一次），cost-proxy **-5.56%**（24/30 較好）。
- 投入完整 300 epoch（`model_epoch300_overlap_v9.pt`）。val_loss
  0.1463，比 v4 的 0.1026 高——但跟 v5.18 self-conditioning 不同，這是
  資料增強的預期模式（訓練任務變難但泛化變好，類似影像分類增強常見的
  「train loss 上升、test 表現進步」現象），不當成負面訊號。
- **100 樣本 paired** 對比 v4：四項指標同樣全部同向變好（raw overlap
  -7.0%，**82/100 樣本較好**），cost-proxy **-1.17%**（61/100 較好）
  ——訊號雖然比 30 樣本時弱一些，但依然一致、依然強。

**驗證（第二階段，官方 evaluate 卻打平）**：

兩次獨立官方 evaluate（跟 v4 用同一套 production 推論設定）：real score
（換算真實 median runtime）**1.1286 / 1.1306**，平均 **1.1296**，跟 v4
的 **1.128** 相比只差 **+0.14%**——在單次評估 ±2% 的雜訊範圍內，等於
打平，完全沒有反映 paired 測試裡看到的一致優勢。

**決定**：**不採用**（`my_optimizer.py` 維持
`model_epoch300_overlap_v4.pt`）。**方法論教訓**：paired 測試（同一組
diffusion 輸出/固定 seed 餵給不同設定比較）雖然能有效濾掉「取樣本身的
隨機性」這個雜訊來源，但也可能因此系統性放大真實效果——當真實效果本身
很小時，paired 設計的變異數縮減會讓小訊號顯得比獨立重跑時更一致、更
可信，而官方 evaluate 的兩次獨立跑（各自完全重新取樣）才是更貼近實際
比賽情境的驗證方式。這是繼 v5.14（runtime 代價被中性 evaluate 隱藏）之
後，這個 session 第二次抓到「驗證方法論本身的選擇會影響結論」的案例，
但性質不同：v5.14 是中性 evaluate 低估了 runtime 代價，這次是 paired
測試低估了訊號的雜訊。機制與 `model_epoch300_overlap_v9.pt` checkpoint
保留備用；如果之後想確認訊號是否只是差一點沒達到統計顯著（例如再跑
第三、四次官方 evaluate 平均），基礎建設已經在，不需要重新走一次訓練
過程。

---

## v5.18 —— 訓練端 Self-Conditioning（不採用）

**背景**：查文獻找訓練端改善方向，找到 Self-Conditioning（Chen, Zhang &
Hinton, 2022, "Analog Bits: Generating Discrete Data using Diffusion
Models with Self-Conditioning"）——讓模型在訓練時看到「自己上一步的
x0_pred 估計」當額外輸入，訓練時以 50% 機率先做一次 no-grad forward（用
零向量當自我調節輸入）算出估計、detach 後餵回真正的 forward；推論時把
上一步真的算出的 x0_pred 接到下一步。概念上讓模型能「修正」估計而非每步
從頭生成。

**實作**：`model.py: Denoiser` 新增 `self_cond_proj`（2 層 MLP，
`config.use_self_cond=True` 時才建立，+66,816 個參數），以加法方式疊加在
`state_proj` 的輸出上——跟 v5.10 `coord_sincos` 同一種掛法，不改
`state_proj` 本身形狀（維持跟舊 checkpoint 相容）。`diffusion.py:
training_loss` 新增 `use_self_cond` 參數實作上述 50% 機率訓練方案；
`ddim_sample_with_forces` 新增同名參數，把 x0_pred 跨步傳遞，在
best-of-N re-noise checkpoint 重置（batch 身分跟雜訊量級都變了，沿用舊
self_cond 會誤導模型）。單元測試驗證：`use_self_cond=False`（訓練與
推論兩條路徑）逐位元跟改動前相同；`True` 時梯度正確流過新模組、無
NaN/inf；用真實 dataloader 跑了幾步訓練＋backward＋optimizer step 全部
正常。

使用者要求跳過 30 epoch 短跑，直接投入完整 300 epoch 訓練（
`model_epoch300_overlap_v8.pt`，QK-norm/Min-SNR/coord_sincos 全部維持
`False` 避免混淆變因）。訓練完成後 val_loss **0.1425**，比 v4 的
**0.1026** 明顯高——第一個警訊。

**驗證**：

- 100 樣本 paired inference 對比 v4（都用目前 production 推論設定
  `DDIM_STEPS=10`／`POST_REPEL_STEPS=10`／`REINSERT_SWEEPS=1`／
  `REINSERT_GRID_DENSITY=4`，v8 額外開 `use_self_cond=True`）：area_gap
  小幅較好（0.2326→0.2266），hpwl_gap／V_relative 小幅較差，**raw
  overlap 明顯較差**（1555.1→**1930.4，+24%**）——這是 legalize 前最
  直接反映模型生成品質的指標。cost-proxy **+0.28%**（55/100 較好、
  45/100 較差，接近打平），訊號強度遠不如 QK-norm（v5.9）或 repaint
  （v5.14）當初的 paired 結果。
- 訊號雖弱，使用者要求仍跑一次官方 evaluate 確認：real score（換算真實
  median runtime）**1.1477**，比目前 production 的 **1.128** 差
  **+1.75%**，avg runtime 也較高（1.66s vs 1.46s，+14%，`self_cond_proj`
  每步都要多算一次，不是免費的）。方向跟 paired 測試、跟 val_loss 三個
  獨立訊號完全一致。

**決定**：**不採用**（`my_optimizer.py` 維持 `model_epoch300_overlap_
v4.pt`，`USE_SELF_COND=False`）。三個獨立訊號（val_loss、100 樣本
paired、官方 evaluate）方向一致指向沒有真實幫助，不需要更多確認。推測
原因：self-conditioning 的訓練方案讓「沒有自我調節資訊」那 50% 的
sub-task 變得更難（模型要學會在資訊更少時也表現好），這可能稀釋了
300 epoch 裡花在核心去噪任務上的有效學習量，跟 v5.9 Min-SNR 被拒絕的
機制不同，但都是「改變了梯度預算的分配方式，副作用大於預期效益」的
同一類故事。機制與 `model_epoch300_overlap_v8.pt` checkpoint 保留備用。

---

## v5.17 —— legalize `reinsert_sweeps`/`reinsert_grid_density`（採用，在 v5.15+v5.16 之上疊加）

**背景**：延續同一套真實 median runtime 方法論，檢查 legalize
`compact_reinsert`（對每個 block 做「拔出來、找 bbox 面積+緊密度最小的
新位置」局部搜尋）的搜尋強度（`reinsert_sweeps` 掃幾輪、
`reinsert_grid_density` 候選網格多細，原預設 3/12）。舊記錄（method.md
§2.3）早就發現「搜尋強度加倍/三倍沒有幫助」，但從沒測過**降低**強度是否
安全。

**掃描結果**（34 樣本分層抽樣，`DDIM_STEPS=10`+`POST_REPEL_STEPS=10` 之上
疊加）：`area_gap`／`hpwl_gap` 在所有測試組合下**完全沒有變化**
（`compact_reinsert` 在真實資料上第一輪就幾乎收斂，多掃幾輪、網格切細都
是白費），`grid_density` 降到 4 以下 real cost 打平（不再有額外好處、
也沒有壞處，代表已經摸到 legalize 這部分計算量的地板）。最終選
`reinsert_sweeps=1, reinsert_grid_density=4`。

**完整 100 樣本官方 evaluate 確認**（兩次獨立跑）：

| | run1 | run2 | 平均 |
|---|---|---|---|
| Total Score（中性） | 1.5350 | 1.5182 | 1.5266 |
| Total Score（換算真實 median runtime） | 1.1381 | 1.1174 | **1.128** |
| avg runtime | 1.47s | 1.45s | 1.46s |
| n_infeasible | 0 | 0 | 0 |

比 v5.16 的 1.1692 再進步約 **-3.5%**，兩次評估都個別優於 baseline。

**決定**：**採用**。累計效果：v4 的真實分數從最初的 1.2322，經 v5.15→
v5.16→v5.17 依序疊加，降到 **1.128**（累計 **-8.5%**），全程沒有犧牲
任何品質（area/hpwl/V_relative 在雜訊範圍內、0/100 infeasible 從沒變過）
——純粹是重新檢視「哪些推論端計算是真的冗餘」找到的。

---

## v5.16 —— POST_REPEL_STEPS 30→10（採用，在 v5.15 之上疊加）

**背景**：v5.15 確認「真實 median runtime 換算」這套方法論能挖出真實分數
改善之後，把同一套流程套用到另一個從沒被系統性掃過、純推論端、跟
runtime 直接相關的參數：`post_repel_steps`（diffusion 結束後純物理迴圈，
只用 Direct Repulsion + Boundary Nudge，沒有 model forward）。

**掃描方法**：同 v5.15，34 樣本分層抽樣 + 真實 median runtime 換算，在
新採用的 `DDIM_STEPS=10` 之上疊加測試。

**結果**（real cost，越低越好；baseline `post_repel=30` = 1.117）：

| post_repel_steps | 30 | 20 | 15 | 10 | 5 | 0 |
|---|---|---|---|---|---|---|
| real cost | 1.117 | 1.107 | 1.102 | **1.091** | 1.107 | 1.294 |

`30→10` 持續變好（-2.4%），但**跟 DDIM_STEPS 不一樣**：完全關閉
（`post_repel_steps=0`）會讓 V_relative 從 ~0.10 跳到 **0.19**、real cost
反彈到 1.294——post-repel 不是純冗餘步數，對 boundary/overlap 有
legalize 自己補不回來的清理效果，不能無限縮減。

**完整 100 樣本官方 evaluate 確認**（`my_optimizer.py`，
`DDIM_STEPS=10` + `POST_REPEL_STEPS=10`，兩次獨立跑）：

| | run1 | run2 | 平均 |
|---|---|---|---|
| Total Score（中性 `RuntimeFactor=1.0`） | 1.5327 | 1.5166 | 1.5247 |
| Total Score（換算真實 median runtime） | 1.1807 | 1.1577 | **1.1692** |
| avg runtime | 1.77s | 1.70s | 1.74s |
| n_infeasible | 0 | 0 | 0 |

比純 `DDIM_STEPS=10`（真實分數 1.178）再進步約 **-0.75%**——比 34 樣本
篩選預測的 -2.4% 弱一些（100 樣本 vs. 34 樣本分層抽樣的正常雜訊差異），
但方向一致、兩次評估都在合理範圍內，確實是真實的額外改善。

**決定**：**採用**，`my_optimizer.py` 的 `POST_REPEL_STEPS` 從 30 改成
10。跟 v5.15 疊加後，v4 的真實分數從最初的 1.2322 降到 **1.1692**
（累計 -5.1%），純粹靠重新檢視「哪些推論端步數是真的冗餘」拿到，沒有
犧牲任何品質（0/100 infeasible 全程不變）。

---

## v5.15 —— DDIM_STEPS 30→10（採用，這個 session 目前最大的一次真實分數改善）

**背景**：v5.14 為了確認 repaint 不划算，第一次把 v4 的真實
（換算 alpha-test median runtime）分數拆開來看，順便算出一個之前沒注意到
的數字：v4 平均 `RuntimeFactor^0.3`（cost 公式的時間乘項）約 **0.837**，
離公式下限 **0.7** 還有 **16.4%** 空間沒被利用到——意思是還有明確、可
量化的空間可以靠純粹加速拿到分數，不需要犧牲品質。回頭檢查發現
`DDIM_STEPS=30` 是這整個 session 從沒被系統性掃過的核心超參數：它是
序列迴圈（每步等前一步），是 diffusion 時間的主要驅動者（不像
`N_SAMPLES`，候選共用同一個 GPU batch，v5.4 已驗證幾乎不影響時間）。

**過去為什麼沒試過**：這個 session 之前所有跟 runtime 有關的判斷，幾乎
都只看「中性 `RuntimeFactor=1.0`」的官方 evaluate 或不含 runtime 項的
cost-proxy 公式——v5.14 才第一次確認這個方法論會嚴重低估「本來就比賽場
快的 checkpoint」的真實 runtime 價值。`DDIM_STEPS` 從一開始就沿用官方
baseline 附近的預設值，沒有被重新檢視過。

**掃描方法**：34 樣本分層抽樣（`test_id` 每隔 3 個取一個，涵蓋 21~140
block 的完整規模範圍，而不是只挑小案例），對每個樣本量測**真實** wall
time（而非 cost-proxy 假設 `RuntimeFactor=1.0`），用
`C_Median Runtime per Testcase(Alpha).csv` 換算真實 `RuntimeFactor`、
代入完整官方 cost 公式。

**結果**（real cost，越低越好；baseline `steps=30` = 1.263）：

| DDIM_STEPS | 30 | 24 | 20 | 16 | 12 | 10 | 8 | 6 | 4 | 2 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real cost | 1.263 | 1.194 | 1.165 | 1.16 | 1.155 | 1.119 | **1.109** | 1.148 | 1.125 | 1.845 | 1.909 |

從 30 降到 4，real cost **持續變好或打平**（品質幾乎不受影響——
area_gap／hpwl_gap 在雜訊範圍內波動，V_relative 甚至常常比 30 步更低，
34 樣本全部 0 infeasible），最低點在 `steps=8` 附近（比 baseline 好約
12%）；`steps=2` 才突然崩潰（hpwl_gap 0.17→0.61、V_relative
0.11→0.29）。懸崖比原本預期的位置晚很多——DDIM 在這個任務上顯然有相當
大的步數冗餘。

**最終選擇 `steps=10`**（不選懸崖邊的 `steps=8`）：只用 34/100 樣本、
懸崖離 `steps=4` 這麼近，選一個離懸崖 2 倍以上安全邊際、但仍拿到大半效益
的值更穩健，避免不在樣本裡的邊緣案例（特別大或特別不規則的問題）踩到
沒被發現的局部懸崖。

**完整 100 樣本官方 evaluate 確認**（`my_optimizer.py`，兩次獨立跑，
`DDIM_STEPS=10`，其餘不變）：

| | run1 | run2 | 平均 |
|---|---|---|---|
| Total Score（中性 `RuntimeFactor=1.0`） | 1.5133 | 1.5221 | 1.5177 |
| Total Score（換算真實 median runtime） | 1.1945 | 1.1614 | **1.178** |
| avg area_gap | 0.2308 | 0.2274 | 0.2291 |
| avg hpwl_gap | 0.2803 | 0.2877 | 0.2840 |
| avg V_relative | 0.1094 | 0.1118 | 0.1106 |
| avg runtime | 1.87s | 1.81s | 1.84s |
| n_infeasible | 0 | 0 | 0 |

跟 v4（`DDIM_STEPS=30`）的真實分數 **1.2322** 相比，兩次獨立評估**各自**
都更好（1.1945、1.1614），平均 **-4.4%**——品質幾乎沒有差異（area/hpwl/
V_relative 三項都在雜訊範圍內），純粹靠 runtime 從 2.485s 降到 1.84s
（-26%）拿到的真實分數改善。

**決定**：**採用**，`my_optimizer.py` 的 `DDIM_STEPS` 從 30 改成 10。這是
這個 session 目前唯一一個「不用犧牲任何品質、純粹靠重新檢視方法論假設
（中性 runtime → 真實 median runtime）就找到」的真實分數改善，也印證了
v5.14 教訓的價值：runtime 相關的判斷一定要換算真實 median runtime，不能
只看中性 evaluate。

---

## v5.14 —— RePaint 式 harmonization resampling（不採用，關鍵教訓：必須用真實 median runtime 驗證）

**背景**：`ddim_sample_with_forces` 每一步都對 preplaced/fixed 的已知
區域做 hard inpainting（強制貼回已知加噪版本），但這個「貼回去」只在
當前這一步發生一次，已知區域跟自由生成區域之間的資訊只靠下一步的
attention 慢慢傳遞，容易在交界處留下不協調的痕跡。RePaint
（Lugmayr et al., CVPR 2022）針對這個已知的 DDPM inpainting 缺陷提出
resampling：同一個 t 做完一次 reverse step 後，先把結果加噪聲跳回同一個
t，再重跑一次同樣的 step，重複 `repaint_resample_steps` 次，讓已知/未知
區域有更多機會在同一雜訊量級上互相對齊。

**實作**：`diffusion.py: ddim_sample_with_forces` 新增
`repaint_resample_steps` 參數（預設 1 = 關閉，跟改動前完全等價），新增
`_repaint_jump_back` 輔助方法（用跟既有 re-noise 檢查點一致的 DDPM 邊際
分布公式，把 x 從 t_prev 往回加噪聲跳到 t_cur）。只有存在
preplaced/fixed 硬限制的樣本才會觸發，其餘樣本零成本。單元測試驗證
`steps=1` 逐位元不變、`steps>1` 確實改變輸出，且 preplaced block 的位置
在任何 `steps` 下都精確不變（hard inpainting 機制沒被破壞）。
`inference.py` 一路傳遞。純推論端，不需要重新訓練。

**驗證（第一階段，很有希望）**：

- 30 樣本 quasi-paired 掃 `steps ∈ {2, 3}`：cost-proxy 分別 -3.08% / -3.86%，
  raw overlap 大幅下降（424→266→208），是這個 session 目前**paired 測試
  訊號最強**的一次。
- 放大到 **100 樣本 paired** 確認 `steps=2`：area_gap／hpwl_gap／
  V_relative／raw overlap **四項全部同向變好**（raw overlap 992→627，
  -37%），cost-proxy -1.77%（60/100 較好）——訊號存活。
- 兩次獨立官方 evaluate（中性 `RuntimeFactor=1.0`）：1.5220 / 1.5098，
  平均 **1.5159**，比 v4 baseline（同一套 codebase 下的 1.5248）
  **看起來略好 -0.58%**——是這個 session 第一個「paired 測試 + 中性官方
  evaluate 都撐住」的正面結果。

**驗證（第二階段，用真實 median runtime 重算才發現真相）**：

user 直接追問「這樣加入 RuntimeFactor 考慮過後，整體成績是不是會下滑」，
促成用 `C_Median Runtime per Testcase(Alpha).csv` 重算真實分數（同
v5.11 章節最後用的方法）。結果：

| | 中性（`RuntimeFactor=1.0`） | 真實（換算 alpha-test median runtime） |
|---|---|---|
| v4 baseline | 1.499129 | **1.2322** |
| repaint steps=2（兩次平均） | 1.5159 | **1.3319**（**+8.1%，明顯變差**） |

v4 本來就比 alpha-test median 快很多（99/100 樣本比 median 快，avg
runtime 2.485s），這個速度優勢讓它在真實分數裡拿到比中性分數低 18% 的
大幅折扣。repaint 把 avg runtime 拉到 3.505s（+41%，只算受影響樣本的
邊際成本被拉低到全體平均），大幅吃掉這個折扣——中性 evaluate 因為兩邊
runtime 都用假設值 1.0，完全看不出這個代價，甚至讓 repaint 顯得「略勝」，
是會誤導判斷的假象。這跟 QK-norm（v5.9）當初被拒絕是同一個機制（真實
compute 成本 vs. 品質改善的取捨），只是這次沒有先做真實 runtime 換算，
一度差點做出錯誤的「採用」判斷。

**決定**：`repaint_resample_steps` **不採用**（`my_optimizer.py` 維持
`REPAINT_RESAMPLE_STEPS=1`）。機制保留備用。**方法論教訓**：任何會實質
增加 runtime 的改動，不能只看中性 `RuntimeFactor=1.0` 的官方 evaluate
結果來判斷——對一個本來就比賽場平均快很多的 checkpoint，中性分數會系統性
低估 runtime 增加的真實代價，必須換算 `C_Median Runtime per
Testcase(Alpha).csv` 的真實 median runtime 才能看到完整圖像。之後任何
runtime 相關的實驗都應該比照這個流程。

---

## v5.13 —— 推論端 best-of-N 加權重抽樣（不採用）

**背景**：現有 best-of-N 機制在 70% 那個 checkpoint，是把 N 個候選裡
overlap 分數最好的「唯一一個」複製到全部 N 個 batch slot、各自加噪聲重跑
剩下步驟，其餘 N-1 個候選直接丟棄——是 SMC/Feynman-Kac 類文獻裡
resampling 步驟的一個退化特例（全部權重收斂到單一 particle）。文獻認為
這種硬性 collapse 通常不如「依分數做加權重新抽樣、保留多個較好候選」
有效。

**實作**：`diffusion.py: ddim_sample_with_forces` 新增
`resample_temperature` 參數（預設 `None` = 關閉，跟改動前完全等價）。
給正浮點數時，分數先做 z-score 正規化，再用
`softmax(-normalized_scores / temperature)` 當機率、`torch.multinomial`
做加權重抽樣決定新的 N 個 batch slot 各自複製哪個候選，取代 `argmin`
硬選。`inference.py` 一路傳遞。單元測試驗證 `temperature=None` 逐位元
不變、給值後確實改變輸出。純推論端，不需要重新訓練。

**驗證**：30 樣本 quasi-paired 掃 `temperature ∈ {0.3, 0.6, 1.0, 2.0}`，
四組**全部都是「變差的樣本數 > 變好的樣本數」**（13:17、13:17、12:18、
10:20），沒有任何一組看起來值得往下做 100 樣本/官方 evaluate 確認，訊號
明顯比 v5.12、v5.14 都弱，決定不繼續深入。

推測原因：現有機制在 70% 對「贏家」軌跡做 N 次獨立加噪聲延伸，本來就有
多樣性來源；加權重抽樣改的是「用比較弱的候選當種子」，但排序分數（純
overlap 近似）在 70% 這個時間點本身就不夠準（HPWL、V_relative 這些真正
決定最終品質的指標，在 state 階段根本算不出來），讓「相信次好候選」的
價值打了折扣——文獻裡提到的「中途評分很難準確預測最終品質」這個已知
難題，在這裡看起來確實成立。

**決定**：`resample_temperature` **不採用**（維持 `None`）。機制保留
備用。

---

## v5.12 —— 推論端 force-guidance 信心加權排程（不採用）

**背景**：上網查「純推論端、不用重訓」的品質改善方向，找到 inference-time
scaling for diffusion models 這條文獻——guidance 強度隨去噪過程調整（早期
弱、晚期強，或反過來）通常比整段固定強度好。`ddim_sample_with_forces`
的四個 force（pin/grouping/repulsion/boundary）目前各自有 v5.0 調過的
on/off t 窗口，但窗口內是固定常數強度。production 只用 30 步 DDIM，直接
在窗口邊界做平滑淡入淡出可用的 step 數太少（部分窗口本身只覆蓋 1-2 個
離散 step），改成让整個窗口內的強度依 `alpha_bar_t`（denoising 信心，
隨 t 從 999→0 單調從接近 0 升到 1）連續縮放更有意義。

**實作**：`diffusion.py: ddim_sample_with_forces` 新增
`force_confidence_power` 參數（預設 0.0，`alpha_bar_t**0=1` 恆等於 1，
跟改動前完全等價），四個 force 各自的有效強度改成
`base_strength * alpha_bar_t**power`。窗口本身（何時開始/結束）與
`power=0` 時的滿強度數值完全沿用 v5.0 已驗證的預設，不變。單元測試驗證
`power=0.0` 兩次獨立跑（同 seed）逐位元相同，`power≠0.0` 確實改變輸出
（機制有接上，不是死代碼）。`inference.py: generate_floorplan`／
`run_one_sample` 一路傳遞這個參數。純推論端改動，不需要重新訓練，直接用
現有 `model_epoch300_overlap_v4.pt` 測試。

**驗證**：

- 30 樣本 quasi-paired 掃 `power ∈ {0.5, 1.0, 2.0}`：`power=1.0` 看起來
  最好，cost 公式 -2.42%（20/30 較好）。
- 放大到 **100 樣本 paired**（更可靠）重測 `power=1.0`：area_gap
  0.2258→0.2241、hpwl_gap 0.1633→0.1556、V_relative 0.1077→0.1065、raw
  overlap 1007.6→997.4——**四項指標全部同向變好**（不像大多數先前實驗
  是「一項變好、一項變差」的取捨），但幅度都不大，cost 公式僅
  **-0.55%**（53/100 較好、47/100 較差，接近打平）。
- 拿 `my_optimizer.py`（`FORCE_CONFIDENCE_POWER=1.0`）跑兩次獨立官方
  evaluate（同慣例作法）：**1.5189 / 1.5620**，平均 **1.5405**——跟
  v4 baseline（同一套修好的 v5.11 codebase 下重新評估的平均
  1.5248，見 v5.10 章節）相比，反而**略差 +1.0%**，沒有通過確認。

**決定**：`force_confidence_power` **不採用**（`my_optimizer.py` 維持
`FORCE_CONFIDENCE_POWER=0.0`）——100 樣本 paired 測試的訊號雖然乾淨（四項
指標同向），但幅度太小，扛不住官方 evaluate 本身的單次 ±2% 雜訊，兩次
獨立評估平均沒有支持「變好」的結論，反而略偏負面。機制（`diffusion.py`／
`inference.py` 裡的 `force_confidence_power` 參數）保留備用，之後如果想
在其他 power 值或搭配其他窗口設計上繼續試，不需要重新走一次接線過程。

---

## v5.11 —— 修正 `utils.py` 的 soft violation 判定，跟官方對齊（bug fix）

**發現**：`my_optimizer_results.json`（官方 `iccad2026_evaluate.py --evaluate`
的輸出）裡記錄的 `violations_relative`，跟我們自己用 `utils.py:
compute_soft_violations` 對同一組 positions 重算出來的 V_relative 對不
起來。100 樣本平均：官方 0.2518，我們算出來 ~0.10-0.13——**這代表這整個
session 用來做決策的 V_relative 數字，絕對值一直被系統性低估到官方的
40-50%**。

**根因逐項排查**（把 `iccad2026_evaluate.py` 的判定邏輯完整複製一份、
拿官方 100 樣本已記錄的真實 positions 反推驗證，`reproduced_official_
style` 對 100/100 樣本精確吻合官方數字，確認複製對了，才拿去跟
`utils.py` 現有的三個函式逐一比對）：

1. **`compute_boundary_violations`（主因，佔壓倒性多數差距）**：舊版用
   「layout 自己邊長的 1% 相對容忍度」（`tol=1e-2` 乘上邊長）判定是否
   貼邊，對一個 100 單位大小的佈局等於放寬到快 1 個完整單位——把「其實
   沒真的貼邊、只是離得比較近」的 block 大量誤判成合法。官方用的是
   `eps=1e-6`（近乎精確貼齊）。20 個案例交叉驗證：我們算出的 V_boundary
   常常只有官方的 1/4~1/8（例如某案例我們算 1、官方算 8）。
2. **`compute_cluster_violations`（次因，影響小很多）**：舊版用
   `_blocks_share_edge`（pairwise 邊緣距離 < tol 的近似判定）算連通分量，
   20 個案例裡有 3 個跟官方（Shapely 精確多邊形聯集）差 1。
3. **`compute_mib_violations`（未觀察到實際影響，但精度本來就對不齊）**：
   舊版四捨五入到 3 位小數（`tol=1e-3`），官方是 4 位小數。

**修法**：三個函式都改成跟官方逐位元對齊——`compute_boundary_violations`
容忍度改成絕對值 `tol=1e-6`；`compute_cluster_violations` 改用 Shapely
的 `unary_union` 判斷連通分量（跟官方完全一致的演算法，不是近似），
Shapely 不可用時退回舊的近似判定並印警告；`compute_mib_violations` 精度
改成 4 位小數。修完後拿 100 個官方已記錄的真實 positions 重算，
**100/100 樣本精確吻合**官方 `violations_relative`（差距 < 1e-9）。

**影響範圍與後續**：這三個函式不只是最終回報用的診斷指標，也被
`compact_pair_reinsert`／`compact_reinsert_reshape`／
`compact_merge_cluster_groups`／`compact_gradient_finetune` 等多個
legalize pass 拿來當「這個候選解有沒有讓違規變差」的安全閘門判斷依據
（`baseline_v`／`new_v` 比較）——舊版容忍度過鬆，代表這些安全閘門過去
可能比實際設計的意圖更寬鬆。修完後跑過一次合成正確性測試
（`test_lff.py`），所有 hard constraint（overlap/area/preplaced/fixed）
仍然全部通過，沒有引入新的錯誤或明顯效能劣化。**但這也代表本 session
之前每一輪 A/B／paired 比較裡「V_relative 变好/变差多少」這類絕對數字
的判斷，都是基於被低估的舊算法**——各版本之間「哪個相對比較好」的方向性
結論，因為 A/B 兩邊用的是同一套（有問題的）算法，大機率還是站得住腳；
但任何牽涉到 V_relative 絕對量級的判斷（尤其 cost 公式裡
`exp(BETA·V_relative)` 這個指數項，真實影響力比先前以為的更大）如果要
嚴謹重新確認，需要拿修正後的版本重跑。

| 版本 | Total Score | Avg Runtime |
|---|---|---|
| v4.6 | 1.9955 | 2.55s |
| v4.7（兩次獨立評估平均） | ~1.9956 | 2.45-2.51s |
| v5.0（`repulsion_strength=0.025`） | 1.9894 | 2.24s |
| v5.0（`repulsion_strength=0.0375`，目前預設） | 2.0128 | 2.46s（max 6.54s） |

（單次 evaluate 本身有約 ±2% 的雜訊，各版本之間的小幅波動不一定代表真實
差異——真正控制雜訊的是 quasi-paired/paired 100 樣本測試，數字見上方
各版本說明。以上都是 v5.11 修 bug **之前**跑的，`violations_relative`
被系統性低估，不能跟下面的修後數字直接比大小。）

**修 bug 之後重新評估 v4**（目前 production checkpoint，
`model_epoch300_overlap_v4.pt`，`my_optimizer.py --evaluate`，100 樣本）：

| 指標 | 數值 |
|---|---|
| Total Score（`RuntimeFactor=1.0` 中性） | **1.499129** |
| Total Score（換算 alpha-test 實際 median runtime，見下方） | **1.2322** |
| avg area_gap | 0.2266 |
| avg hpwl_gap | 0.2688 |
| avg V_relative（官方精確定義，修完全對齊） | 0.1090 |
| avg runtime | 2.485s（max 5.786s） |
| n_infeasible | 0 / 100 |

修完 bug 後 Total Score 從 ~2.0 降到 1.50（中性 runtime）——這不是「回報
數字變準了所以看起來變好」，而是真實效果：`compute_boundary_violations`
等函式也被 `compact_pair_reinsert`／`compact_gradient_finetune`／
`compact_merge_cluster_groups` 拿去當安全閘門用，修嚴之後這些 legalize
pass 真的會更積極拒絕會讓真實違規變差的候選解，直接改變了
`legalize_lff` 的實際輸出，不只是診斷報告的數字。可以看到代價：
hpwl_gap 比修 bug 前的估計（~15.7-16.1%）明顯變差（26.88%）——這是可以
理解的取捨，閘門變嚴之後，一部分「當年被誤判為安全、其實會讓真實違規
變差」的 HPWL 改善移動現在會被正確擋下來；而 cost 公式裡
`exp(2·V_relative)` 是指數項，V_relative 的真實改善（從真正意義上的
~0.25 降到 0.109）換算下來遠遠壓過 hpwl_gap 變差的代價，Total Score
淨效果是大幅改善。

**用真實 median runtime 換算的分數**（`C_Median Runtime per Testcase
(Alpha).csv`，alpha test 100 個 test case 的實際 median runtime）：
本專案 99/100 個 test case 都比 alpha-test median 快，換算
`RuntimeFactor = my_runtime / median_runtime` 後代入
`max(0.7, RuntimeFactor^0.3)`，Total Score 從中性版的 1.499129 進一步降到
**1.2322**——`RuntimeFactor` 帶來的加成比原本以為的更明顯，這也是「不
盲目追求最快、把預算留給品質」這個取捨在真實比賽情境下的量化依據。
