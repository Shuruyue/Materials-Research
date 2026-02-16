# 階段 A：基礎知識建構（2月中 – 4月底）

> **目標**：建立固態物理 + 機器學習 + 程式實作的堅實基礎
> **每日投入**：6-8 小時
> **產出**：讀書筆記 20+ 章、代碼練習 50+ 個、Phase 1 baseline 全部通過

---

## 第 1 週（2/17 – 2/23）：晶體結構基礎

### 核心教材：Kittel Ch1-2

**週一 2/17**
- [ ] 上午：讀 Kittel Ch1 前半（Crystal Structure, p.1-15）
  - 重點：14 種 Bravais 晶格、7 種晶系
  - 筆記：畫出所有 14 種 Bravais 晶格的 3D 示意圖
- [ ] 下午：代碼練習 — 用 pymatgen 建構各晶系結構
  ```python
  from pymatgen.core import Structure, Lattice
  # 練習：建構 FCC, BCC, HCP, diamond 結構
  ```
- [ ] 晚間：整理 Ch1 重點筆記（中英對照術語表）

**週二 2/18**
- [ ] 上午：讀 Kittel Ch1 後半（Fundamental Types of Lattices, p.15-30）
  - 重點：點群（point group）、空間群（space group）的概念
  - 練習：列出立方晶系的所有點群 (Oh, Td, O, Th, T)
- [ ] 下午：用 spglib / pymatgen 分析晶體對稱性
  ```python
  from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
  # 練習：分析 Si, NaCl, BaTiO3 的空間群
  ```
- [ ] 晚間：閱讀 JARVIS 數據庫文檔，了解數據結構

**週三 2/19**
- [ ] 上午：讀 Kittel Ch2（Reciprocal Lattice, p.31-55）
  - 重點：倒格子（reciprocal lattice）、布里淵區（Brillouin zone）
  - 推導：倒格矢量 b₁ = 2π(a₂×a₃)/(a₁·a₂×a₃)
- [ ] 下午：代碼 — 計算並 3D 可視化倒格子和第一布里淵區
  ```python
  # 用 matplotlib 3D 畫出 FCC 的倒格子（= BCC）
  # 畫出第一 Brillouin zone
  ```
- [ ] 晚間：讀 Kittel Ch2 習題（挑 5 題做）

**週四 2/20**
- [ ] 上午：複習 Ch1-2，整理完整筆記
  - 練習：默寫 14 種 Bravais 晶格的基矢量
  - 練習：推導 FCC ↔ BCC 倒格子對應關係
- [ ] 下午：ATLAS 代碼熟悉
  - 研讀 `atlas/data/crystal_dataset.py` 代碼
  - 理解 `CrystalGraphBuilder` 如何將結構轉換為圖
- [ ] 晚間：閱讀 CGCNN 論文（Xie & Grossman, 2018）第一遍

**週五 2/21**
- [ ] 上午：精讀 CGCNN 論文
  - 重點：原子特徵編碼、邊特徵（高斯距離）、卷積層設計
  - 筆記：畫出 CGCNN 完整架構圖
- [ ] 下午：研讀 `atlas/models/cgcnn.py` 代碼
  - 對比論文和代碼的每一個模組
  - 列出論文與實作的差異
- [ ] 晚間：整理本週學習成果，更新進度表

**週六 2/22**
- [ ] 上午：Kittel 習題 Ch1-2（全部做完）
- [ ] 下午：複習 PyTorch 基礎（tensor 操作、autograd、nn.Module）
- [ ] 晚間：休息或自由閱讀

**週日 2/23**
- [ ] 半天休息
- [ ] 下午：週回顧 — 列出本週疑問，規劃下週重點

> **本週驗收**：
> - ✅ 能解釋 Bravais 晶格和倒格子的關係
> - ✅ 能用 pymatgen 建構和分析晶體結構
> - ✅ 讀完 CGCNN 論文並理解架構

---

## 第 2 週（2/24 – 3/2）：晶體鍵結與力學

### 核心教材：Kittel Ch3-4

**週一 2/24**
- [ ] 上午：讀 Kittel Ch3（Crystal Binding, p.55-80）
  - 重點：離子鍵（Madelung 常數）、共價鍵、金屬鍵、凡德瓦力
  - 推導：NaCl 的 Madelung 能量
- [ ] 下午：用 JARVIS 數據計算材料的形成能分布
  ```python
  # 統計 formation_energy 的分布
  # 按鍵結類型（離子/共價/金屬）分類
  ```
- [ ] 晚間：整理鍵結能量與形成能的關係筆記

**週二 2/25**
- [ ] 上午：讀 Kittel Ch3 續 — 彈性常數
  - 重點：應力張量 σᵢⱼ、應變張量 εᵢⱼ、彈性張量 Cᵢⱼₖₗ
  - 理解：Voigt 記號 (6×6 matrix)
- [ ] 下午：代碼 — 從 JARVIS 載入彈性張量資料
  ```python
  # 下載 elastic_tensor 數據
  # 計算 Voigt 平均 K_V, G_V
  # 驗算與 bulk_modulus_kv 的一致性
  ```
- [ ] 晚間：推導 Born stability criteria（彈性張量正定）

**週三 2/26**
- [ ] 上午：讀 Kittel Ch4（Phonons I: Crystal Vibrations, p.85-115）
  - 重點：簡正模式、色散關係、聲學分支 vs 光學分支
  - 推導：一維單原子鏈和雙原子鏈的色散關係
- [ ] 下午：代碼 — 畫出一維晶格色散關係
  ```python
  # 模擬 1D monatomic chain: ω = 2√(C/M) |sin(ka/2)|
  # 模擬 1D diatomic chain: 兩個分支
  ```
- [ ] 晚間：理解 Debye 模型和 Debye 溫度

**週四 2/27**
- [ ] 上午：讀 Kittel Ch4 續 — 熱學性質
  - 重點：比熱容（Debye model, Einstein model）
  - 推導：低溫 C ∝ T³ (Debye T³ law)
- [ ] 下午：代碼 — 畫出 Debye 比熱曲線
  ```python
  # Debye model: C_V(T) = 9nk_B (T/ΘD)³ ∫₀^{ΘD/T} x⁴eˣ/(eˣ-1)² dx
  # 比較不同 ΘD 的曲線
  ```
- [ ] 晚間：閱讀 SchNet 論文（Schütt et al., 2018）

**週五 2/28**
- [ ] 上午：精讀 SchNet 論文
  - 重點：continuous filter convolution vs CGCNN 的比較
  - 筆記：SchNet 的 3 大創新點
- [ ] 下午：ATLAS Phase 1 代碼 — 重跑 formation_energy
  ```bash
  python scripts/11_train_cgcnn_full.py --property formation_energy
  ```
- [ ] 晚間：監控訓練進度，整理本週筆記

**週六 3/1**
- [ ] 上午：做 Kittel Ch3-4 習題
- [ ] 下午：自由時間 — 深入一個感興趣的主題
- [ ] 晚間：休息

**週日 3/2**
- [ ] 半天休息
- [ ] 下午：週回顧

> **本週驗收**：
> - ✅ 能解釋鍵結類型與形成能的關係
> - ✅ 能用 Voigt 記號表示彈性張量
> - ✅ 理解色散關係和 Debye 模型
> - ✅ 讀完 SchNet 論文

---

## 第 3 週（3/3 – 3/9）：電子能帶理論

### 核心教材：Kittel Ch5-7

**週一 3/3**
- [ ] 上午：讀 Kittel Ch5（Free Electron Fermi Gas, p.131-160）
  - 重點：費米球、費米能量、態密度 D(E)
  - 推導：3D 自由電子態密度 D(E) = V/(2π²) · (2m/ħ²)^(3/2) · √E
- [ ] 下午：代碼 — 計算和可視化費米球
  ```python
  # 畫出 3D 費米球
  # 計算不同金屬的費米能量和態密度
  ```
- [ ] 晚間：理解費米-狄拉克分布與溫度效應

**週二 3/4**
- [ ] 上午：讀 Kittel Ch6（Energy Bands, p.161-185）
  - 重點：近自由電子模型、能帶間隙的起源
  - 推導：布拉格繞射條件 → 能帶間隙
- [ ] 下午：代碼 — 畫出一維近自由電子能帶圖
  ```python
  # nearly free electron model: E(k) with gap at BZ boundary
  # 比較自由電子 vs 近自由電子
  ```
- [ ] 晚間：理解直接帶隙 vs 間接帶隙

**週三 3/5**
- [ ] 上午：讀 Kittel Ch7（Semiconductors, p.185-220）
  - 重點：半導體物理、摻雜、帶隙
  - 理解：帶隙 → 電學性質 → 應用的邏輯鏈
- [ ] 下午：分析 JARVIS 數據中的帶隙分布
  ```python
  # 統計 band_gap 分布
  # 畫出金屬 (Eg=0) vs 半導體 vs 絕緣體的比例
  # 帶隙與形成能的相關性
  ```
- [ ] 晚間：理解 DFT 計算帶隙的系統誤差（LDA/GGA 低估）

**週四 3/6**
- [ ] 上午：讀 Kittel Ch7 續 — p-n 接面
  - 重點：內建電場、空乏層、整流效果
- [ ] 下午：整理能帶理論完整筆記
  - 畫出完整的 DFT → bands → properties 流程圖
  - 列出帶隙、形成能、彈性的 DFT 計算方法
- [ ] 晚間：閱讀 ALIGNN 論文（Choudhary & DeCost, 2021）

**週五 3/7**
- [ ] 上午：精讀 ALIGNN 論文
  - 重點：line graph 架構、角度特徵、與 CGCNN 對比
  - 筆記：ALIGNN 為何比 CGCNN 準確？
- [ ] 下午：繼續監控 Phase 1 formation_energy 重跑結果
  - 如果完成，分析新舊結果對比
- [ ] 晚間：做 Kittel Ch5-7 練習題

**週六 3/8**
- [ ] 上午：習題
- [ ] 下午：自由學習
- [ ] 晚間：休息

**週日 3/9**
- [ ] 半天休息
- [ ] 下午：週回顧 — 整理前三週所有筆記

> **本週驗收**：
> - ✅ 能推導費米能量和態密度
> - ✅ 能解釋能帶間隙的物理起源
> - ✅ 理解帶隙在材料設計中的重要性
> - ✅ 讀完 ALIGNN 論文

---

## 第 4 週（3/10 – 3/16）：磁性與介電

### 核心教材：Kittel Ch8, Ch13

**週一 3/10**
- [ ] 上午：讀 Kittel Ch8（Semiconductor Crystals, p.185-215）
  - 重點：有效質量、載子遷移率
  - 理解：為何 GNN 預測帶隙有物理意義
- [ ] 下午：代碼 — 從 JARVIS 收集 5 個關鍵物性的統計特徵
  ```python
  # formation_energy, band_gap, bulk_modulus, shear_modulus, dielectric
  # 統計：分布、相關性矩陣、缺失值
  ```
- [ ] 晚間：製作物性關聯矩陣（scatter plot matrix）

**週二 3/11**
- [ ] 上午：讀 Kittel Ch13（Dielectrics and Ferroelectrics, p.375-405）
  - 重點：介電常數 ε、極化 P、壓電效應
  - 理解：ε 和 Eg 的反比關係（Penn model: ε ∝ 1/Eg²）
- [ ] 下午：代碼 — 驗證 JARVIS 中 ε 和 Eg 的反比關係
  ```python
  # 畫出 dielectric vs 1/Eg² 的散點圖
  # 計算 Pearson correlation
  ```
- [ ] 晚間：閱讀壓電張量的物理意義

**週三 3/12**
- [ ] 上午：理解張量物性的對稱性約束
  - 介電張量 εᵢⱼ：3×3 對稱，6 獨立分量
  - 彈性張量 Cᵢⱼₖₗ：Voigt 6×6 對稱，最多 21 獨立分量
  - 不同晶系的額外約束（例如：立方只有 3 個獨立的 Cᵢⱼ）
- [ ] 下午：代碼 — 驗證 JARVIS 彈性張量的對稱性
  ```python
  # 載入 elastic_tensor 數據
  # 檢查 Cij = Cji 是否成立
  # 分析不同空間群的獨立分量數
  ```
- [ ] 晚間：整理「對稱性約束張量」的筆記表格

**週四 3/13**
- [ ] 上午：Ashcroft & Mermin Ch22（Classical Theory of Crystal Lattice）
  - 補充更深入的彈性理論
  - 理解 Born stability criteria 的物理意義
- [ ] 下午：研讀 `atlas/training/losses.py`
  - 理解 BornStabilityLoss 的實作
  - 理解 MultiTaskLoss 的 uncertainty weighting
- [ ] 晚間：整理代碼和理論的對應筆記

**週五 3/14**
- [ ] 上午：Ashcroft & Mermin Ch23（Quantum Theory of Crystal Lattice）
  - 聲子的量子力學處理
  - 理解為何機器學習需要量子力學結果（DFT）
- [ ] 下午：檢查 Phase 1 重跑結果
  - 如果 formation_energy 修復成功 → 整理最終結果
  - 如果未達標 → 分析原因
- [ ] 晚間：Kittel Ch8 + Ch13 習題

**週六 3/15** ← **里程碑 ①**
- [ ] 上午：完成所有 Kittel 筆記的整理
- [ ] 下午：自我測試 — 不看書回答 20 個核心問題
- [ ] 晚間：里程碑檢查
  > ✅ 能解釋布里淵區、能帶理論、聲子、彈性張量、介電
  > ✅ 能用代碼重現 FCC 的布里淵區

**週日 3/16**
- [ ] 休息日

---

## 第 5 週（3/17 – 3/23）：深度學習基礎

### 核心教材：Goodfellow Ch1-5

**週一 3/17**
- [ ] 上午：讀 Goodfellow Ch1-2（Introduction + Linear Algebra Review）
  - 特徵分解、SVD、梯度、Jacobian
  - 複習：線性代數核心操作
- [ ] 下午：PyTorch 練習 — tensor 操作、broadcasting、autograd
  ```python
  # 練習：手動實現 linear regression
  # 練習：使用 autograd 計算梯度
  # 練習：比較手動 vs autograd 的梯度
  ```
- [ ] 晚間：整理線性代數在 ML 中的使用手冊

**週二 3/18**
- [ ] 上午：讀 Goodfellow Ch3-4（Probability + Numerical Computation）
  - 重點：MLE、MAP、正則化
  - 理解：overfitting = high variance, underfitting = high bias
- [ ] 下午：PyTorch 練習 — 完整的 MLP 訓練
  ```python
  # 用 nn.Module 建構 MLP
  # 在 MNIST 或 Boston housing 上訓練
  # 畫出 learning curve（train vs val loss）
  ```
- [ ] 晚間：理解 Adam 優化器的原理

**週三 3/19**
- [ ] 上午：讀 Goodfellow Ch5（Machine Learning Basics）
  - 重點：bias-variance tradeoff、regularization、cross-validation
  - 理解：為何 early stopping 有效
- [ ] 下午：練習 — regularization 實驗
  ```python
  # 比較：no regularization vs L2 vs dropout vs early stopping
  # 畫出各自的 learning curve
  ```
- [ ] 晚間：整理 ML 基礎的公式表

**週四 3/20**
- [ ] 上午：讀 Goodfellow Ch6-8（Deep Feedforward + Regularization + Optimization）
  - 重點：BatchNorm、Dropout、learning rate scheduling
  - 理解：為何深度網路需要特殊的初始化（Xavier, He）
- [ ] 下午：代碼 — 從頭實做 BatchNorm 和 Dropout
  ```python
  class MyBatchNorm(nn.Module):
      # 不用 nn.BatchNorm, 手動實現
  class MyDropout(nn.Module):
      # 不用 nn.Dropout, 手動實現
  ```
- [ ] 晚間：閱讀 AdamW 論文（Loshchilov 2019）

**週五 3/21**
- [ ] 上午：讀 Goodfellow Ch10（Sequence Modeling）— 了解即可
  - 理解 attention 的基本概念
  - 這對理解 GNN 中的注意力機制有幫助
- [ ] 下午：代碼 — 完整訓練管線
  ```python
  # 練習：建構完整的訓練管線
  # 包含：data loading, model, optimizer, scheduler, 
  #       early stopping, checkpointing, TensorBoard
  ```
- [ ] 晚間：整理深度學習筆記

**週六 3/22**
- [ ] 上午：做 Goodfellow 課後概念問題
- [ ] 下午：自由學習
- [ ] 晚間：休息

**週日 3/23**
- [ ] 半天休息
- [ ] 下午：週回顧

> **本週驗收**：
> - ✅ 能從零實做 MLP + 完整訓練管線
> - ✅ 理解 regularization 的理論和實作
> - ✅ 能解釋 bias-variance tradeoff

---

## 第 6 週（3/24 – 3/30）：圖神經網路

### 核心教材：Hamilton Ch1-7

**週一 3/24**
- [ ] 上午：讀 Hamilton Ch1-2（Introduction + Background）
  - 重點：圖的基本概念、鄰接矩陣、特徵矩陣
  - 理解：為何用圖表示晶體結構
- [ ] 下午：PyTorch Geometric 入門
  ```python
  from torch_geometric.data import Data
  # 練習：手動建構一個晶體圖
  # 練習：理解 edge_index, x, edge_attr, batch 的格式
  ```
- [ ] 晚間：閱讀 PyG 官方教程

**週二 3/25**
- [ ] 上午：讀 Hamilton Ch3-4（Neighborhood Aggregation + GNN Architecture）
  - 重點：message passing framework
  - 推導：GCN layer: h' = σ(D⁻¹/²AD⁻¹/²HW)
- [ ] 下午：從頭實做 GCN layer（不用 PyG 的 GCNConv）
  ```python
  class MyGCNConv(nn.Module):
      # 手動實現 message + aggregate + update
  ```
- [ ] 晚間：對比 GCN, GAT, GraphSAGE 的差異

**週三 3/26**
- [ ] 上午：讀 Hamilton Ch5-6（GNN Variants + Theory）
  - 重點：Weisfeiler-Leman test、GNN 表達力上界
  - 理解：為何需要邊特徵（edge features）
- [ ] 下午：用 PyG 實現 CGCNN 簡化版
  ```python
  # 從頭實做一個 3 層的 crystal GNN
  # 測試在 JARVIS 小樣本上的表現
  ```
- [ ] 晚間：閱讀 GIN 論文（Xu et al., 2019）

**週四 3/27**
- [ ] 上午：讀 Hamilton Ch7（Graph-level Representation）
  - 重點：pooling（global mean, global max, set2set）
  - 理解：如何從節點特徵 → 圖特徵
- [ ] 下午：代碼 — 比較不同 pooling 的效果
  ```python
  # global_mean_pool vs global_max_pool vs global_add_pool
  # 在簡化 CGCNN 上比較
  ```
- [ ] 晚間：研讀 ATLAS 中的 pooling 實作

**週五 3/28**
- [ ] 上午：複習 GNN 理論，整理完整筆記
  - 畫出：message → aggregate → update 的流程圖
  - 表格：各種 GNN 變體的比較
- [ ] 下午：代碼 — 完整的 GNN 訓練 pipeline
  ```python
  # 整合：CrystalPropertyDataset + CGCNN + training loop
  # 在小樣本上跑一個完整實驗
  ```
- [ ] 晚間：閱讀 MEGNet 論文（Chen et al., 2019）

**週六 3/29**
- [ ] 上午：做 GNN 相關練習
- [ ] 下午：自由學習
- [ ] 晚間：休息

**週日 3/30**
- [ ] 半天休息
- [ ] 下午：週回顧

> **本週驗收**：
> - ✅ 能解釋 message passing 框架
> - ✅ 能從頭實做 GCN layer
> - ✅ 理解 CGCNN, SchNet, ALIGNN, MEGNet 的異同

---

## 第 7-8 週（3/31 – 4/13）：Phase 1 收尾 + 論文閱讀 Sprint

### 第 7 週 (3/31 – 4/6)

**週一 3/31 – 週三 4/2**
- [ ] 重跑 Phase 1 所有 4 個物性
  ```bash
  python scripts/11_train_cgcnn_full.py --all-properties --patience 120
  ```
- [ ] 同時閱讀以下論文（每天 1 篇）：
  - 週一：DimeNet（Gasteiger et al., 2020）
  - 週二：PaiNN（Schütt et al., 2021）
  - 週三：NequIP（Batzner et al., 2022）

**週四 4/3 – 週五 4/4**
- [ ] 繼續論文閱讀：
  - 週四：MACE（Batatia et al., 2022）
  - 週五：e3nn 論文（Geiger & Smidt, 2022）

**週六 4/5 – 週日 4/6**
- [ ] 整理所有 GNN 論文的比較表格
- [ ] 製作「GNN 演進年表」timeline
- [ ] 休息

### 第 8 週 (4/7 – 4/13)

**週一 4/7** ← **里程碑 ②**
- [ ] Phase 1 最終結果整理
  > ✅ 4 個物性 MAE ≤ target
  > ✅ 結果表格 + 對比文獻

**週二 4/8 – 週五 4/11**
- [ ] 多任務學習論文閱讀
  - 週二：Kendall et al., 2018（Uncertainty weighting）
  - 週三：GradNorm（Chen et al., 2018）
  - 週四：PCGrad（Yu et al., 2020）
  - 週五：Sener & Koltun, 2018（MGDA）

**週六 4/12 – 週日 4/13**
- [ ] 整理多任務學習筆記
- [ ] 寫「多任務學習策略比較」文檔
- [ ] 階段 A 總結報告

> **階段 A 最終驗收**：
> - ✅ Kittel Ch1-8, 13 完成（筆記 + 習題）
> - ✅ Goodfellow Ch1-8, 10 完成
> - ✅ Hamilton Ch1-7 完成
> - ✅ 精讀論文 ≥ 15 篇
> - ✅ Phase 1 baseline 4/4 通過
> - ✅ 能獨立實做 GNN + 訓練管線
