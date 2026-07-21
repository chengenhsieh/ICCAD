# ICCAD 2026 FloorSet — 解法說明（Training + Inference）

本文件說明整個 floorplanning 解法的設計，分成兩大階段：**訓練**一個 GNN 風格的
diffusion 生成模型、以及**推論**時「diffusion 生成 + 決定性 legalize」的兩段式
pipeline。目標拆解如下：

- **Hard constraints（絕不可違反）**：block 間不重疊、preplaced block 位置與形狀
  完全固定、fixed-shape block 形狀固定、soft block 面積誤差 ≤ 1%。
- **Soft constraints（盡量滿足）**：boundary（block 貼齊指定邊）、cluster（同組
  block 需相鄰成一塊）、MIB（Multi-Instance Block，同組 block 需同尺寸）。
- **品質指標**：與官方 optimal 解相比的 area gap、HPWL gap，以及推論總時間。

核心設計哲學：**diffusion 模型負責「盡量學到好的擺法」（soft constraint、HPWL、
面積效率），legalize 負責「用演算法結構保證 hard constraint 100% 不違反」**。
兩者分工明確——不會為了保 hard constraint 而讓 diffusion 模型背負它處理不好的
組合最佳化問題，也不會讓 legalize 承擔它不擅長的「學出好品味」的工作。

---

## Part 1 — Training

### 1.1 資料表示

每個 floorplan 樣本被編碼成固定 `N = max_blocks = 120` 個 slot（不足補 padding，
`mask` 標記有效 block）：

- **`state` (N, 3)** — diffusion 的訓練目標，每個 block 是
  `(x_norm, y_norm, log_r)`：中心點正規化座標（相對 canvas 的 bounding box）
  + 長寬比的 log（`log_r = log(w/h)`）。用 `log_r` 而非直接學 `(w, h)` 是因為
  面積本身已由 `area_targets` 給定（硬性數值），模型只需要學「形狀比例」與
  「位置」，這樣可以把「面積必須精確」這個 hard constraint 完全排除在生成任務
  之外，交給 legalize 階段用解析方式保證。
- **`block_features` (N, 13)** — 每個 block 的條件特徵：
  `[area_norm, is_fixed, is_preplaced, is_mib_member, is_cluster_member,
  bnd_left, bnd_right, bnd_top, bnd_bottom, pin_cx, pin_cy, pin_total_weight,
  reserved]`。boundary 特別編碼成 4 個獨立 bit 而非單一被 clip 的數值，避免
  「TOP-LEFT 角落」這種疊加語意被壓成單一值時資訊損毀。
- **`conn_weights` (N, N)** — B2B（block-to-block）連線權重矩陣，對稱。
- **`group_bias` (N, N)** — 同 MIB 組或同 cluster 組的 block 之間標 1，其餘 0，
  之後直接餵進 attention 當額外的 bias 訊號（見 1.2）。
- **`mib_group` / `cluster_group` / `boundary_code` (N,)** — 保留完整組別 ID /
  bitmask（不像早期版本粗暴 clip 成 0/1，那樣會讓「同組」這個關鍵資訊完全消失）。

### 1.2 模型架構（`model.py`）

`FloorplanDiffusionModel` = **BlockEncoder**（條件編碼器）+ **Denoiser**（去噪
網路），兩者都是多層 **Connectivity-Biased Transformer**：標準 multi-head
self-attention（Vaswani et al., *"Attention Is All You Need,"* NeurIPS 2017），
但在 attention logits 上疊加兩種可學習的 additive bias：

```
attn = QK^T / sqrt(d_k) + Bias(conn_weights) + Bias(group_bias)
```

也就是把「這兩個 block 之間有沒有 B2B 連線」「是不是同一個 MIB / cluster 組」
這種圖結構資訊，直接編碼進 attention 分數裡，而不是只靠位置編碼或訊息傳遞層。
這個做法在精神上與 **Graphormer**（Ying et al., *"Do Transformers Really
Perform Bad for Graph Representation?,"* NeurIPS 2021）把圖的邊/最短路徑資訊
編碼成 attention bias 的手法一致——用 Transformer 的全域 attention 取代傳統
GNN 的局部訊息傳遞，同時仍讓模型明確感知圖結構。

- **BlockEncoder**：吃 `block_features` + 連線/群組 bias，輸出每個 block 的
  condition embedding（不含時間資訊）。
- **Denoiser**：吃當前 timestep 的 noisy state、BlockEncoder 的 condition
  embedding、sinusoidal 時間嵌入（做法同標準 DDPM 的 timestep embedding），
  預測該步加入的噪聲 `noise_pred`。

Config：`d_model=256`、`n_heads=8`、encoder 6 層 / denoiser 8 層、
`dim_feedforward=1024`，總參數量在訓練時印出（見 `train.py` log）。

### 1.3 Diffusion 公式與訓練目標（`diffusion.py`）

標準 **DDPM**（Ho, Jain, Abbeel, *"Denoising Diffusion Probabilistic Models,"*
NeurIPS 2020）forward process，linear beta schedule（`T=1000`,
`beta: 1e-4 → 0.02`）：

```
q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) x_0, (1-ᾱ_t) I)
```

訓練時對每個 batch 隨機取 timestep `t ~ U(0, T)`，用 `q_sample` 加噪，模型預測
噪聲，loss 是標準的 masked MSE（denoising score matching 目標，同 Ho et al.
2020 的 simplified objective `L_simple`）：

```
mse = MSE(noise_pred * mask, noise * mask)
```

#### Soft constraint 輔助 loss（訓練端）

除了主要的去噪 MSE，還從**預測出的 x0**（用標準 DDPM 的
`x0_pred = (x_t - sqrt(1-ᾱ_t)·noise_pred) / sqrt(ᾱ_t)` 反推）上疊加 4 個可微
的輔助懲罰項，讓模型在去噪的同時就學到 soft constraint 的傾向，而不是只靠
inference 端的事後修正：

| Loss | 定義 | 直覺 |
|---|---|---|
| `mib_loss` | 同 MIB 組內 `log_r` 的組內變異數總和 | 同組應該同形狀 |
| `cluster_loss` | 同 cluster 組內 `x_norm`、`y_norm` 的組內變異數總和 | 同組中心應彼此靠近 |
| `boundary_loss` | 該貼邊的 block，對應座標到邊界（0 或 1）的 MSE | 貼邊 block 應該真的貼在邊上 |
| `overlap_loss` | 用 `x0_pred` 反推出的 `(w, h)`，向量化算所有 pair 的 bbox 重疊面積 | 直接對「重疊」這個 hard constraint 施加訓練期壓力，減輕 legalize 負擔 |

這幾項全部用 `scatter_add` 向量化跨整個 batch 一次算完（避免 Python 雙迴圈 +
GPU→CPU 同步）。權重 `lambda_mib=0.1, lambda_cluster=0.3, lambda_boundary=0.3,
lambda_overlap=0.3`，並有 **warmup**：前 `soft_loss_warmup_epochs=10` 個 epoch
只用純去噪 MSE，之後才加入 soft loss——讓模型先學會基本的去噪能力，再疊加額外
的結構性壓力，避免訓練初期梯度被幾個懲罰項主導導致不收斂。

這種「對擴散模型預測出的 x0 疊加任務相關的可微懲罰」的做法，概念上與
guidance-based diffusion sampling（Dhariwal & Nichol, *"Diffusion Models Beat
GANs on Image Synthesis,"* NeurIPS 2021 的 classifier guidance）師出同源——
差別在於這裡是**訓練期**把懲罰直接加進 loss 一起反傳，而不是**推論期**才用
額外訊號去 steer 採樣軌跡（後者是 Part 2 的 force-guided sampling 在做的事）。

### 1.4 訓練迴圈（`train.py`）

- **資料**：FloorSet-Prime 訓練集取前 200,000 筆（`../` 下 `LiteTensorData/`）
  訓練、100 筆官方 validation set（`LiteTensorDataTest/`）驗證。
- **Optimizer**：AdamW（`lr=1e-4`, `weight_decay=1e-5`），
  `CosineAnnealingLR`（`T_max=epochs`, `eta_min=1e-5`）逐 epoch 退火。
- **EMA**：對模型權重維護 decay=0.9999 的指數移動平均，存 checkpoint 時用
  EMA 權重（`ema.apply_shadow()`）而非即時權重——diffusion 模型公認對權重雜訊
  敏感，EMA 是穩定生成品質的標準作法。
- **AMP 混合精度**：GPU 上預設開啟（`torch.cuda.amp`），約 1.5–2× 訓練加速。
- **梯度裁剪**：`grad_clip=1.0`。
- `batch_size=64`，訓練 300 epoch，每 `save_interval=50` epoch 或刷新最佳
  val loss 時存 checkpoint（`model_epoch300_overlap_v4.pt` 即為本次提交使用
  的最終權重）。

---

## Part 2 — Inference

推論分兩段：**(A) Force-guided DDIM 生成**產生一個「盡量好」但不保證 hard
constraint 的初始佈局，**(B) 決定性 legalize**把它轉成一個 hard constraint
100% 合法、且盡量保留 soft constraint / HPWL 品質的最終解。

### 2.1 Stage A — Force-guided DDIM 採樣（`diffusion.py:ddim_sample_with_forces`）

採樣器基於 **DDIM**（Song, Meng, Ermon, *"Denoising Diffusion Implicit
Models,"* ICLR 2021）的非馬可夫、可跳步反向過程，而非原始 DDPM 的逐步採樣——
這讓我們能用遠少於 `T=1000` 的步數（目前設定 `ddim_steps=30`）產生高品質樣本。
每一步反向擴散之外，還疊加以下機制：

1. **Hard inpainting**：對 preplaced block，每一步都把該 block 的 state 直接
   替換成「加噪版的真實 GT 座標」（`sqrt(ᾱ)·gt_state + sqrt(1-ᾱ)·noise`）；
   對 fixed-shape（非 preplaced）block，只 inpaint 形狀維度（`log_r`）、位置
   仍自由。這保證這兩類 hard constraint 在生成過程中就被尊重，而不是生成完
   再硬修正（硬修正容易破壞周圍佈局的合理性）。
2. **MIB log-ratio 投影**：`t >= mib_clamp_until_t` 時，把同 MIB 組的 `log_r`
   投影成組內一致的值。
3. **Force 項**（累加 delta、clip 總位移後套用，避免單步跳太遠）：
   - **Pin force**：把有 p2b 連線的 block 往其加權 pin 中心拉。
   - **Grouping force**：把同 cluster 組的 block 互相拉近。
   - **Repulsion force**：對所有 block 施加基於面積的排斥力，減少 overlap。
   - **Boundary nudge**：把該貼邊的 block 往對應邊界推。
   
   這幾個力都是簡單的解析幾何力（非學習得來），只在特定 timestep 窗口內生效
   （例如 repulsion 只在採樣後段 `t <= repulsion_from_t` 生效，此時 x0
   已經比較穩定，力的作用不會被前段的高噪聲淹沒）。概念上這類「用額外訊號
   在採樣時 steer 生成軌跡」的手法與 diffusion guidance 文獻（見 1.3 引用的
   Dhariwal & Nichol 2021）同源，但這裡用的是任務特定的解析幾何力，而非
   學出來的 classifier gradient。
4. **Best-of-N + re-noise**：一次生成 `n_samples` 個候選（同一個 batch 一起
   跑，在 GPU 上幾乎是平行成本，不是序列疊加——100 樣本 A/B 顯示
   `n_samples: 6→14` 對 diffusion 時間幾乎沒有影響）。在 `t = renoise_idx`
   （目前設定為 `0.7 × ddim_steps`）時，用一個近似 overlap 分數選出當下最好的
   候選，把它複製到所有 batch slot、加入少量噪聲後繼續跑完剩下的步數——讓好
   候選有機會被進一步 refine，而非在半路就固定死。
5. **Post-repel**：DDIM 步驟結束後，再跑 `post_repel_steps=30` 步「純物理、
   無 model」的 repulsion + boundary nudge 迴圈。100 樣本 A/B 測試證實這一步
   仍然必要：拿掉後 V_relative（soft constraint 違規率）從 0.117 惡化到
   0.198（近乎翻倍），因為它給了 legalize 階段一個「起點已經比較有秩序」的
   佈局，legalize 自己的局部搜尋沒辦法完全補回這個落差。

### 2.2 Stage B — Legalize（`utils.py:legalize_lff`）

Diffusion 輸出仍可能有重疊、面積誤差、blockshape 偏移。Legalize 階段用
**決定性、單趟**的演算法把它轉成保證合法的最終解——「決定性」是刻意的設計
選擇：我們最初評估過用 B*-tree + Simulated Annealing（Chang, Chang, Wu, Wu,
*"B\*-Trees: A New Representation for Non-Slicing Floorplans,"* DAC 2000——
這也是官方範本 `optimizer_template.py` 提供的 baseline 做法）去搜尋合法解，
但 SA 的隨機搜尋在我們的時間預算下不夠快、也不夠穩定，因此改用一個結構上就
保證正確、只需單趟即可完成的演算法。

**排布順序（LFF, Less-Flexibility-First）**：先放「越不自由」的 block——
boundary 約束越多（貼越多邊）優先權越高，其次是 cluster 成員、MIB 成員，最後
才是完全自由的一般 block；同優先權內按面積大到小排序。這屬於「最受限變數優先」
這一類排布/排程啟發式的一般家族（在 constraint satisfaction 文獻中常見於
「fail-first」/「most-constrained-variable」啟發式的精神，例如 Haralick &
Elliott 對 CSP 搜尋順序的討論），套用在 floorplanning 的直覺是：越受限的
block 選擇越少，應該先搶到適合的空間，避免被後放的自由 block 佔走。

**空間管理（MAXRECTS）**：維護一組「目前可用的最大自由矩形」集合
（`_FreeRectPool`），每放入一個 block，就用 guillotine 分割規則把跟它相交的
自由矩形切成最多 4 個「剩餘」矩形，並剔除被其他矩形完全包含的重複區域。這是
Jylänki 在 *"A Thousand Ways to Pack the Bin — A Practical Approach to
Two-Dimensional Rectangle Bin Packing"*（2010）中系統整理、廣泛使用於
2D 矩形裝箱問題的自由矩形維護演算法家族。

**位置決定（加權中位數）**：對每個待放置 block，在每個候選自由矩形內，用
「加權中位數」獨立算出每一軸的最佳位置——因為 HPWL、boundary 目標距離、
cluster 中心距離、pin 距離全部都是 L1（曼哈頓）距離，而**加權和的 L1 距離
之無約束最小值，恰好是加權中位數**（這是一個標準的最佳化結果：weighted
median minimizes weighted sum of absolute deviations）。這讓我們不需要對每
個候選位置做網格搜尋，而是直接用封閉形式解出每個候選矩形內的最佳點，再取全
域最小成本的 (矩形, 位置, 形狀) 組合。

**Hard constraint 如何被保證**：
- **零重疊**：MAXRECTS 的自由矩形本身就保證放進去的區域不會跟已放置的 block
  重疊；結尾再跑一次向量化的 worst-pair-first 重疊消解 + 「保底 eject」
  （把極端無法安置的 block 移到當前佈局範圍外、保證不重疊）雙重防線。
- **Preplaced / fixed-shape 精確**：這兩類 block 完全不進入排布迴圈，位置
  / 形狀直接沿用輸入值。
- **面積誤差 ≤ 1%**：soft block 的形狀候選都是「面積固定、只變長寬比」的
  變體（原始比例 + 幾個常見比例如 0.5/2.0/0.25/4.0），因此天生就滿足面積
  容差，不需要事後修正。
- **落在 pin bounding box 內**：outline 預設用給定的 pin bbox；若初次嘗試
  有 block 放不下（fallback），才以 5% 為步階逐步放大 outline 重試（最多
  20 次），優先把 block 留在使用者要求的範圍內，不會一開始就過度放大。

**壓縮後處理（減少 area gap，不影響 hard constraint）**：

1. `compact_merge_clusters`——找出目前「彼此貼合」的連通分量，把非主要群的
   衛星群整體剛體平移，靠向主要群。原始版本用「只要沾到任一邊界約束就整軸
   鎖死」的過度保守判斷，導致大部分情況完全不會移動；後改為對每次候選平移
   做「移動後用官方 boundary 判定式重算違規數，不能超過移動前」的驗證閘門
   （因為 bbox 邊界本身是由目前最極端的 block 自我定義，貼邊群整體往內移動
   時邊界通常會跟著收縮，並不會真的破壞貼邊關係）——修正後同一個樣本原本
   14 個衛星分量卡死 0 個能合併，變成 11 個成功合併。

   **家族協同移動（v4.5）**：上述「單一衛星 vs main」的邏輯仍有一個結構性
   盲區——如果**兩個以上**各自獨立、互不相鄰的衛星分量剛好都被鎖在**同一條**
   邊界（例如都是 RIGHT-locked），任何一個單獨試著往內移動，都會被「移動後
   自己不再貼著那條邊」的驗證閘門擋下來（因為邊界是由「這條邊上最極端的
   block」自我定義，若同一條邊還有其他衛星群守著，這個衛星群移開後自己就不
   再是最極端的）。這正是視覺上「主要群聚旁邊有兩、三塊各自獨立、卻共同卡在
   同一條邊上」留白的成因。

   解法參考經典 VLSI 版圖壓縮（layout compaction）文獻的核心概念：把版圖的
   非重疊/邊界關係表示成有向**限制圖（constraint graph）**，用**最長路徑**
   求出滿足所有限制下最緊的擺法——Murata 等人在 *sequence-pair* 表示法
   （Murata et al., 1996）中用水平/垂直限制圖 + 最長路徑，從一組序對算出
   對應的最緊面積；Tang & Wong 的 FAST-SP（2001）進一步把 preplaced、range、
   **boundary** 這幾種限制直接編碼進同一套限制圖求解框架，是最早能處理
   boundary constraint 的 sequence-pair 方法。本專案沒有實作完整的限制圖 /
   mixed-constraint 求解器（那需要同時處理「不重疊」這種不等式限制和「貼邊」
   這種等式限制，屬於更一般的 mixed-constraint compaction 問題，見 Hsueh &
   Pederson 對這類問題的討論），而是採用一個範圍更窄、風險更低的對應做法：
   把「共享同一條邊界鎖」的所有衛星分量當成一個**剛體家族**，用同一個位移量
   一起平移——家族內部相對位置不變，所以移動後家族仍然共同定義同一條（跟著
   縮小的）邊界，不會違反任何一個成員的鎖定，等同於在這個受限情境下手動保證
   了「多個等式限制同時滿足」，而不需要一般化的限制圖求解器。

   這個修正用一個獨立、乾淨的單元測試驗證過機制本身正確：兩個互不相鄰、都
   鎖在同一條邊的衛星分量，在有真正空間可移動時會正確地一起平移、保持邊界
   對齊，bbox 寬度從 80 縮小到 35、且零重疊。100 樣本官方驗證集上，這是
   本次投入的多個壓縮嘗試裡**唯一**帶來平均 area gap 實質下降的改動（見
   2.3 節的完整比較表）。
2. `compact_reinsert`——對每個 block 做「拔出來、在目前佔用範圍內找 bbox 面
   積 + 緊密度最小的新位置、比較後決定要不要換」的局部搜尋，能找到單純滑動
   搆不到的位置。
3. `compact_positions`——純幾何的沿 x/y 軸滑到貼齊，成本最低的收尾動作。

以上三步都被設計成**證明上單調不變差**（不會把不重疊變成重疊、不會讓 bbox
變大）——因此可以安全疊加在保證正確的 legalize 結果之上，沒有任何 hard
constraint 被重新破壞的風險。

### 2.3 推論參數與其調參過程

最終生產參數（`inference.py: DDIM_STEPS / N_SAMPLES / post_repel_steps` 及
`utils.py: legalize_lff` 的 `reinsert_sweeps/reinsert_grid_density`）是透過
100 樣本 A/B 實驗逐一驗證後選定，過程刻意保留了「試了但沒有用」的負面結果，
而不是只留下有效的修改：

| 嘗試 | 結果 | 決定 |
|---|---|---|
| `ddim_steps: 100 → 30` | 總時間 -50%，品質指標在雜訊範圍內無明顯下降 | 採用 |
| `n_samples: 6 → 14` | best-of-N 候選共用一個 GPU batch，時間幾乎不變，HPWL/V_rel 小幅改善 | 採用 |
| Mixed precision（fp16 autocast）推論 | 小模型下 dtype 轉換開銷蓋過算力節省，反而慢 20% | 不採用 |
| 依 `ddim_steps` 等比例縮放 force-guidance 的 timestep 窗口 | 100 樣本測試前後幾乎沒有差異 | 不採用（已還原） |
| `compact_reinsert` 搜尋強度加倍/三倍 | area gap 沒有改善（甚至略差），時間成本卻明顯增加 | 維持原設定 |
| Legalize 用多種 tie-break 順序取最佳 bbox（legalize 版 best-of-N） | 單一輸入下不同順序确實能差到 ±10% bbox，但套用 `compact_reinsert` 後這個差異幾乎被磨平；時間成本卻是線性倍增 | 不採用（保留為可選功能） |
| `compact_gravity`（每個 block 各自往全域重心走一小步）當額外壓縮 pass | area gap、hpwl gap、V_relative 都輕微變差，且不會處理「多個獨立衛星群共享同一條邊界鎖」這種情況 | 不採用（保留為可選功能） |
| `compact_merge_clusters` 加入**家族協同移動**（多個衛星分量共享同一條邊界鎖時一起剛體平移）+ `rounds: 5 → 20` | 100 樣本 area gap 24.7% → **23.8%**（實質改善），legalize 最差情況時間反而下降（5.10s → 3.58s），V_relative 小幅上升（0.109 → 0.113），0/100 infeasible 不變 | **採用**（唯一一個帶來實質 area gap 改善的壓縮嘗試） |

這個過程反映的核心判斷：**diffusion 端「批次內免費」的候選數（`n_samples`）
值得投資；legalize 端單純「加碼同一種搜尋的強度」（不管是加大 `compact_reinsert`
的網格、多跑幾種排布順序、或加一個新的通用拉力 `compact_gravity`）在目前的
演算法結構下已經觸頂，邊際效益接近於零——但這不代表 legalize 端已經沒有
空間，`compact_merge_clusters` 的家族協同移動證明了：只要找到現有搜尋結構
「結構性漏掉」的一整類情況（此處是「多個獨立衛星群共享同一條邊界鎖」），
用一個針對性、範圍明確的機制去補，仍然能拿到實質、幾乎零成本的改善。要再
進一步壓低 area gap，比較有希望的方向是繼續找這類「結構性漏洞」，而不是
籠統加大現有搜尋的強度。

---

## 最終結果（100 樣本官方 validation set）

| 指標 | 數值 |
|---|---|
| Hard constraint 違規 | 0 / 100（zero overlap, exact preplaced/fixed shape, area ≤1% error, 皆保證滿足） |
| Area gap（vs. optimal） | ~23.7% |
| HPWL gap（vs. optimal） | ~16.1% |
| Soft constraint 違規率（V_relative） | ~0.113 |
| 平均單樣本總時間 | ~2.5s（diffusion ~1.3s + legalize ~1.2s） |

---

## 參考文獻

1. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic
   Models.* NeurIPS 2020. — DDPM forward process 與去噪訓練目標。
2. Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit
   Models.* ICLR 2021. — 推論用的 DDIM 非馬可夫、可跳步採樣器。
3. Dhariwal, P., & Nichol, A. (2021). *Diffusion Models Beat GANs on Image
   Synthesis.* NeurIPS 2021. — Classifier guidance；本專案訓練期輔助 loss
   與推論期 force-guided sampling 的設計精神所本。
4. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS 2017. —
   BlockEncoder / Denoiser 所用的 multi-head self-attention Transformer block。
5. Ying, C., et al. (2021). *Do Transformers Really Perform Bad for Graph
   Representation?* NeurIPS 2021. — 把圖結構（邊、群組關係）編碼成 attention
   additive bias 的手法（Graphormer），對應本專案的
   connectivity-biased / group-biased attention。
6. Chang, Y.-C., Chang, Y.-W., Wu, G.-M., & Wu, S.-W. (2000). *B\*-Trees: A
   New Representation for Non-Slicing Floorplans.* DAC 2000. — 官方 baseline
   範本（`optimizer_template.py`）採用的表示法；本專案評估後改用決定性
   legalize 取代其 B*-tree + SA 搜尋。
7. Jylänki, J. (2010). *A Thousand Ways to Pack the Bin — A Practical
   Approach to Two-Dimensional Rectangle Bin Packing.* — legalize 階段
   MAXRECTS 自由矩形維護演算法所本。
8. Less-Flexibility-First / most-constrained-variable-first：屬於
   constraint satisfaction 與排布/裝箱文獻中廣泛使用的啟發式家族（例如
   Haralick, R. M., & Elliott, G. L. (1980). *Increasing Tree Search
   Efficiency for Constraint Satisfaction Problems.* Artificial
   Intelligence, 14(3) 對「fail-first」搜尋順序的討論）——本專案
   `legalize_lff` 的排布優先權即屬此精神，沒有對應單一原始論文的精確引用。
9. Murata, H., Fujiyoshi, K., Nakatake, S., & Kajitani, Y. (1996). *VLSI
   Module Placement Based on Rectangle-Packing by the Sequence-Pair.* IEEE
   TCAD, 15(12). — Sequence-pair 表示法：用水平/垂直限制圖 + 最長路徑從一組
   序對算出對應的最緊面積，是本專案 `compact_merge_clusters` 家族協同移動
   概念上的參考起點。
10. Tang, X., & Wong, D. F. (2001). *FAST-SP: A Fast Algorithm for Block
    Placement Based on Sequence Pair.* ASPDAC 2001. — 把 preplaced、range、
    **boundary** 限制直接編碼進 sequence-pair 的限制圖求解框架，是最早能
    處理 boundary constraint 的 sequence-pair 方法；本專案的家族協同移動
    是這個概念的範圍受限、風險較低版本，而非完整的限制圖求解器實作。
11. Hsueh, M.-Y., & Pederson, D. O. (1979). *Computer-Aided Layout Compaction
    for VLSI.* Memorandum, UC Berkeley / 相關後續文獻對「mixed constraint」
    layout compaction 的討論——同時處理「不重疊」（不等式）與「貼邊對齊」
    （等式）兩種限制，是比純最長路徑壓縮更一般的問題，本專案並未實作完整版本
    （見上方 `compact_merge_clusters` 家族協同移動的說明）。
