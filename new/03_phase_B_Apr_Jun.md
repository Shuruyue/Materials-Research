# 階段 B：核心技術深入（4月中 – 6月底）

> **目標**：掌握 DFT 理論、等變 GNN、多任務學習；完成 Phase 2-3 實驗
> **每日投入**：8-10 小時
> **產出**：理論筆記 15+ 章、精讀論文 15+ 篇、Phase 2 等變 GNN 訓練完成

---

## 第 9 週（4/14 – 4/20）：密度泛函理論入門

### 核心教材：Sholl & Steckel Ch1-4

**週一 4/14**
- [ ] 上午：讀 Sholl & Steckel Ch1（What is DFT?）
  - 重點：多體薛丁格方程 → Born-Oppenheimer 近似
  - 理解：為何 DFT 可以將 3N 維問題 → 3 維密度問題
- [ ] 下午：安裝 e3nn（Phase 2 依賴）
  ```bash
  pip install e3nn
  python -c "import e3nn; print(e3nn.__version__)"
  ```
  - 跑 Phase 2 smoke test
  ```bash
  python scripts/20_train_equivariant.py --max-samples 1000 --epochs 5
  ```
- [ ] 晚間：整理 DFT 的核心概念筆記

**週二 4/15**
- [ ] 上午：讀 Sholl & Steckel Ch2（Hohenberg-Kohn Theorems）
  - 推導：第一定理（密度唯一決定外勢能）
  - 推導：第二定理（能量泛函極小值）
  - 理解：exchange-correlation 的物理意義
- [ ] 下午：讀 Sholl & Steckel Ch3（Kohn-Sham Equations）
  - 推導：自洽場（SCF）迭代
  - 理解：LDA, GGA 交換-關聯泛函
- [ ] 晚間：整理 DFT 公式的推導鏈
  ```
  H·Ψ = E·Ψ  →  B-O approx  →  HK theorem  →  KS equations
  3N+3N 維         3N 維          n(r) 3 維        {φᵢ(r)} 求解
  ```

**週三 4/16**
- [ ] 上午：讀 Sholl & Steckel Ch4（DFT Calculations: Setup）
  - 重點：基組（plane waves）、赝勢（pseudopotential）、k-point sampling
  - 理解：為何選用平面波基組（周期性邊界條件自然匹配）
- [ ] 下午：研讀 JARVIS-DFT 計算參數
  ```python
  # 研究 JARVIS 用的 DFT 設定
  # 泛函：OptB88vdW (van der Waals corrected GGA)
  # k-mesh, cutoff energy, convergence criteria
  ```
- [ ] 晚間：理解 vdW 修正對帶隙和彈性的影響

**週四 4/17**
- [ ] 上午：讀 Sholl & Steckel Ch5-6（Properties from DFT）
  - 重點：各物性的 DFT 計算方法
  | 物性 | DFT 方法 | 精度等級 |
  |------|----------|---------|
  | Ef | total energy difference | 高 |
  | Eg | KS eigenvalue gap | 中（系統低估） |
  | Cij | finite strain method | 高 |
  | ε | DFPT (linear response) | 高 |
- [ ] 下午：代碼 — 分析 JARVIS 數據的 DFT 精度
  ```python
  # 比較 JARVIS 的 optb88vdw_bandgap vs mbj_bandgap
  # MBJ 通常更準確，比較兩者的差異
  ```
- [ ] 晚間：整理 DFT 精度限制的筆記

**週五 4/18**
- [ ] 上午：讀 Sholl & Steckel Ch7-8（Advanced Topics）
  - 重點：DFPT（密度泛函微擾理論）= 計算介電和壓電的方法
  - 理解 phonon 計算的原理
- [ ] 下午：研讀 Martin Ch1-3（作為 DFT 的補充深入材料）
- [ ] 晚間：做 Sholl & Steckel 練習題

**週六 4/19 – 週日 4/20**
- [ ] 整理 DFT 完整筆記
- [ ] 列表：每個物性對應的 DFT 計算方法
- [ ] 休息

> **本週驗收**：
> - ✅ 能推導 Hohenberg-Kohn 定理
> - ✅ 能解釋 KS 方程的自洽場迭代
> - ✅ 理解 DFT 計算帶隙的系統誤差
> - ✅ e3nn 安裝成功，Phase 2 smoke test 通過

---

## 第 10 週（4/21 – 4/27）：等變性與球諧函數

### 核心教材：e3nn 教程 + Vanderbilt Ch1

**週一 4/21**
- [ ] 上午：群論基礎速成
  - 重點：SO(3) 旋轉群、O(3) 正交群、E(3) 歐幾里得群
  - 理解：等變性（equivariance）vs 不變性（invariance）
  ```
  不變性：f(Rx) = f(x)           → scalars (能量, 帶隙)
  等變性：f(Rx) = D(R)·f(x)     → vectors, tensors
  ```
- [ ] 下午：e3nn 教程（官方 Jupyter notebooks）
  ```python
  import e3nn.o3 as o3
  # 練習 1：建構不可約表示 (Irreps)
  irreps = o3.Irreps("8x0e + 4x1o + 2x2e")
  # 練習 2：球諧函數
  Y = o3.spherical_harmonics(2, vectors, normalize=True)
  ```
- [ ] 晚間：整理 irreps 的意義
  | L | 名稱 | 分量數 | 物理意義 |
  |---|------|--------|---------|
  | 0 | 標量 | 1 | 能量, 帶隙 |
  | 1 | 向量 | 3 | 力, 電偶極 |
  | 2 | 二階張量 | 5 | 應力, 介電 |
  | 3 | 三階張量 | 7 | 壓電 |
  | 4 | 四階張量 | 9 | 彈性張量 |

**週二 4/22**
- [ ] 上午：深入球諧函數
  - 推導：Y_l^m(θ, φ) 的物理意義
  - 可視化：L=0,1,2,3 的球諧函數形狀
  ```python
  # 用 matplotlib 3D 畫出 Y_l^m
  # 理解每個 L 代表的角動量
  ```
- [ ] 下午：e3nn 教程 — Tensor Product
  ```python
  from e3nn.o3 import FullyConnectedTensorProduct
  # 練習：理解 tensor product 的輸入輸出 irreps
  # 1o x 1o = 0e + 1o + 2e (vector × vector = scalar + vector + tensor)
  ```
- [ ] 晚間：推導 CG 係數（Clebsch-Gordan coefficients）

**週三 4/23**
- [ ] 上午：讀 NequIP 論文（精讀第二遍）
  - 重點：每一層的 irreps 變換
  - 畫出 NequIP 的完整架構圖（含 irreps 標註）
- [ ] 下午：對比 ATLAS 代碼和 NequIP 論文
  - 研讀 `atlas/models/equivariant.py` 每一行
  - 標註每行代碼對應論文的哪個公式
- [ ] 晚間：整理 NequIP vs MACE vs PaiNN 的比較表

**週四 4/24**
- [ ] 上午：讀 Vanderbilt Ch1（Adiabatic Evolution, Berry Phase）
  - 理解 Berry phase 的幾何意義
  - 這與等變性的數學基礎（fibre bundle）相關
- [ ] 下午：e3nn 進階 — Gate activation
  ```python
  from e3nn.nn import Gate
  # 練習：理解 gated equivariant activation
  # 標量部分用 sigmoid gate 控制高階 irreps
  ```
- [ ] 晚間：理解為何 ReLU 不能直接用在等變特徵上
  ```
  問題：ReLU(D(R)·x) ≠ D(R)·ReLU(x)
  解決：gated activation — 用標量 gate 控制
  ```

**週五 4/25**
- [ ] 上午：讀 Vanderbilt Ch2（Berry Phase in Crystals）
  - 理解電極化（polarization）的 Berry phase 理論
  - 連結到壓電張量的計算
- [ ] 下午：Phase 2 正式訓練 — formation_energy
  ```bash
  python scripts/20_train_equivariant.py \
      --property formation_energy \
      --epochs 300 \
      --max-samples 5000
  ```
- [ ] 晚間：監控訓練，整理本週筆記

**週六 4/26 – 週日 4/27**
- [ ] 整理等變 GNN 理論筆記
- [ ] 製作「等變操作 cheat sheet」
- [ ] 休息

> **本週驗收**：
> - ✅ 能解釋球諧函數和不可約表示
> - ✅ 能用 e3nn 進行 tensor product 操作
> - ✅ 理解 NequIP 架構的每一個模組
> - ✅ Phase 2 smoke test 完成

---

## 第 11 週（4/28 – 5/4）：等變 GNN 訓練

**週一 4/28 – 週三 4/30**
- [ ] Phase 2 全量訓練（formation_energy, band_gap）
  ```bash
  python scripts/20_train_equivariant.py --all-properties
  ```
- [ ] 同時閱讀：
  - 週一：Tensor Field Networks（Thomas et al., 2018）
  - 週二：SE(3)-Transformers（Fuchs et al., 2020）
  - 週三：Cormorant（Anderson et al., 2019）

**週四 5/1 – 週五 5/2**
- [ ] 分析 Phase 2 結果
  - CGCNN vs EquivariantGNN 對比表
  - 每個物性的 improvement ratio
- [ ] 如果結果不佳：調整超參數
  - Learning rate: 嘗試 3e-4, 1e-4
  - Hidden irreps: 嘗試 "64x0e + 32x1o + 16x2e"
  - Layers: 嘗試 4-5 層

**週六 5/3 – 週日 5/4**
- [ ] 整理 Phase 2 結果文檔
- [ ] 休息

---

## 第 12 週（5/5 – 5/11）：多任務學習理論與實驗

### 核心閱讀：Kendall, GradNorm, PCGrad 論文

**週一 5/5**
- [ ] 上午：精讀 Kendall et al., 2018（Uncertainty-weighted loss）
  - 推導：homoscedastic uncertainty → task weight
  - 理解：L = Σ (1/2σᵢ²) Lᵢ + log(σᵢ)
- [ ] 下午：研讀 `atlas/training/losses.py` — MultiTaskLoss 實作
  - 對比論文公式和代碼
- [ ] 晚間：整理 uncertainty weighting 的推導

**週二 5/6**
- [ ] 上午：精讀 GradNorm（Chen et al., 2018）
  - 理解：gradient norm balancing
  - 研讀 `atlas/training/physics_losses.py` — GradNormWeighter
- [ ] 下午：精讀 PCGrad（Yu et al., 2020）
  - 理解：conflicting gradient projection
  - 研讀 `atlas/training/physics_losses.py` — PCGrad
- [ ] 晚間：製作「多任務策略比較表」

**週三 5/7 – 週五 5/9**
- [ ] Phase 3 多任務訓練
  ```bash
  python scripts/21_train_multitask.py --epochs 300
  ```
- [ ] 消融實驗設計：
  - 實驗 1：fixed-weight vs uncertainty-weight
  - 實驗 2：有/無物理約束
  - 實驗 3：single-task vs multi-task
- [ ] 整理結果

**週六 5/10 – 週日 5/11**
- [ ] 分析多任務 vs 單任務結果
- [ ] 識別 positive/negative transfer
- [ ] 休息

---

## 第 13 週（5/12 – 5/18）：物理約束與 Born Stability

**週一 5/12**
- [ ] 上午：深入理解 Born stability criteria
  - 立方晶系：C₁₁ > 0, C₄₄ > 0, C₁₁ > C₁₂, C₁₁ + 2C₁₂ > 0
  - 一般晶系：彈性張量正定（所有特徵值 > 0）
- [ ] 下午：研讀 `atlas/training/physics_losses.py`
  - PhysicsConstraintLoss 的 5 項約束
  - VoigtReussBoundsLoss 的上下界推導
- [ ] 晚間：推導 Voigt 和 Reuss 平均的公式

**週二 5/13 – 週三 5/14**
- [ ] 消融實驗：有/無物理約束
  ```bash
  # 無約束
  python scripts/21_train_multitask.py --no-physics
  # 有約束
  python scripts/21_train_multitask.py --physics-weight 0.1
  ```
- [ ] 分析：物理約束是否減少非物理預測？

**週四 5/15** ← **里程碑 ③**
- [ ] 整理所有 Phase 2-3 結果
  > ✅ 等變 GNN MAE 優於 CGCNN
  > ✅ 多任務優於（或等於）單任務
  > ✅ 物理約束減少非物理預測

**週五 5/16 – 週日 5/18**
- [ ] 閱讀不確定性量化論文：
  - MC-Dropout（Gal & Ghahramani, 2016）
  - Deep Ensembles（Lakshminarayanan et al., 2017）
  - Evidential DL（Amini et al., 2020）
- [ ] 研讀 `atlas/models/uncertainty.py`
- [ ] 休息

---

## 第 14 週（5/19 – 5/25）：張量預測理論

**週一 5/19 – 週二 5/20**
- [ ] 上午：等變張量輸出的理論
  - 彈性張量 Cᵢⱼₖₗ 的 irreps 分解：L=0 ⊕ L=2 ⊕ L=4
  - 介電張量 εᵢⱼ 的 irreps 分解：L=0 ⊕ L=2
  - Wigner-D 矩陣重建
- [ ] 下午：研讀 `atlas/models/multi_task.py` — TensorHead
  - 理解 irreps → 物理張量的轉換
  - 理解對稱性約束如何強制 Cᵢⱼ = Cⱼᵢ

**週三 5/21 – 週四 5/22**
- [ ] 設計張量預測實驗
  - 數據：JARVIS elastic_tensor (~20K 樣本)
  - 評估：component-wise MAE, Frobenius error, symmetry violation
- [ ] 開始張量預測訓練

**週五 5/23 – 週日 5/25**
- [ ] 分析張量預測結果
- [ ] 讀 Vanderbilt Ch3（Polarization and Piezoelectricity）
- [ ] 休息

---

## 第 15-16 週（5/26 – 6/8）：進階論文閱讀 + 實驗完善

### 第 15 週
- [ ] 每天讀 1 篇論文（連讀 5 天）
  - 週一：M3GNet（Chen & Ong, 2022）
  - 週二：CHGNet（Deng et al., 2023）
  - 週三：MACE-MP-0（Batatia et al., 2023）
  - 週四：GNoME（Merchant et al., 2023）
  - 週五：Crystal-LLM / AlphaFold-like for materials

### 第 16 週
- [ ] 超參數調優（grid search 或 Optuna）
  ```python
  # 用 Optuna 做自動超參數搜索
  import optuna
  # 搜索空間：lr, hidden_dim, n_layers, radial_basis
  ```
- [ ] 最終模型選擇和 5-fold cross validation
- [ ] 統計顯著性檢驗（paired t-test）

**6/8** ← **里程碑 ④**
> - ✅ Phase 2-3 全部完成
> - ✅ 等變 GNN 在所有物性上的最終結果
> - ✅ 消融實驗完成
> - ✅ 開始論文寫作準備

---

## 第 17 週（6/9 – 6/15）：主動學習與不確定性

**週一 6/9 – 週三 6/11**
- [ ] 不確定性實驗
  - MC-Dropout (30 forward passes)
  - Deep Ensemble (5 models)
  - Evidential Regression
  - 比較三種方法的 calibration
- [ ] 研讀 `atlas/models/uncertainty.py`
- [ ] 繪製 reliability diagram

**週四 6/12 – 週五 6/13**
- [ ] 主動學習實驗
  ```bash
  python scripts/31_active_learning.py --strategy uncertainty
  python scripts/31_active_learning.py --strategy random
  ```
- [ ] 比較 uncertainty vs random 的 learning curve
- [ ] 分析：不確定性選擇的材料有何特徵？

**週六 6/14 – 週日 6/15**
- [ ] 整理 Phase 4-5-6 結果
- [ ] 休息

---

## 第 18 週（6/16 – 6/22）：可解釋性分析

**週一 6/16 – 週三 6/18**
- [ ] GNNExplainer 分析
  ```bash
  python scripts/30_explainability_analysis.py \
      --model-dir models/equivariant_formation_energy \
      --n-explain 200
  ```
- [ ] 對每個物性找出 top-10 重要元素
- [ ] 比較不同物性的重要元素是否重疊

**週四 6/19 – 週五 6/20**
- [ ] Latent space 分析
  - t-SNE / UMAP 可視化
  - 以不同物性著色
  - 分析聚類結構

**週六 6/21 – 週日 6/22**
- [ ] 多任務梯度對齊分析
  - 哪些物性之間 positive transfer？
  - 物理解釋：Eg ↔ ε（Penn model），K ↔ Ef（鍵強度）
- [ ] 整理分析結果

---

## 第 19-20 週（6/23 – 7/6）：Pareto 篩選 + 階段 B 收尾

### 第 19 週
- [ ] 多目標 Pareto 篩選
  - 目標：找「高 K + 高 ε + 低 Eg」的壓電材料
  - 從 76K 材料 → GNN 預測 → Pareto 前沿 → Top 50
- [ ] DFT 驗算 Top 10 候選（如果可能用 MACE 代替 DFT）

### 第 20 週
- [ ] 整理所有實驗結果
- [ ] 製作完整結果表格
- [ ] 階段 B 總結報告

> **階段 B 最終驗收**：
> - ✅ DFT 理論完成（Sholl & Steckel 8 章 + Martin 部分）
> - ✅ 等變 GNN 理論完成（e3nn + 球諧 + tensor product）
> - ✅ Phase 2-6 實驗全部完成
> - ✅ 精讀論文 ≥ 30 篇（累計）
> - ✅ 所有消融實驗完成
> - ✅ Pareto 篩選候選材料成功
