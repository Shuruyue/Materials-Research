# 完整逐章閱讀指南

> 每本書的每一章：要讀什麼、為什麼讀、預期時間、完成標準

---

## 📖 Book 1：Kittel — Introduction to Solid State Physics (8th Edition)

### 閱讀策略
- **精讀** Ch1-8, 13（與論文直接相關）
- **略讀** Ch9-12, 14-22（了解即可）
- 每章附帶 2-3 道習題

| 章 | 標題 | 頁數 | 時間 | 為何需要 | 完成標準 |
|:--:|:-----|:----:|:----:|:---------|:---------|
| 1 | Crystal Structure | 30p | 2天 | 晶體圖表示的物理基礎 | 能畫出 14 種 Bravais 晶格，解釋空間群 |
| 2 | Reciprocal Lattice | 25p | 2天 | 倒格子 = k-space = DFT 的工作空間 | 能推導倒格矢量，畫出 BZ |
| 3 | Crystal Binding | 25p | 2天 | 理解形成能的物理意義 | 能解釋離子鍵/共價鍵/金屬鍵能量 |
| 4 | Phonons I | 30p | 2天 | 聲子 = 力學性質基礎 | 能推導 1D 色散關係 |
| 5 | Phonons II | 25p | 1天 | Debye 模型 → 熱學性質 | 能推導 T³ law |
| 6 | Free Electron Model | 30p | 2天 | 態密度 → 金屬電子性質 | 能推導 3D DOS |
| 7 | Energy Bands | 35p | 2天 | 能帶 → 帶隙 → 半導體分類 | 能用近自由電子模型解釋帶隙 |
| 8 | Semiconductor Crystals | 30p | 2天 | 半導體物理 → 帶隙預測的意義 | 能解釋效質量和載子濃度 |
| 13 | Dielectrics | 30p | 2天 | 介電張量預測的物理意義 | 能解釋 Penn model: ε ∝ 1/Eg² |

**總計**：~260 頁，~17 天

### 重點公式整理（每章結束時記錄）

**Ch2 重點公式**：
```
倒格矢量：bᵢ = 2π (aⱼ × aₖ) / (a₁ · a₂ × a₃)
Bragg 條件：2d sin θ = nλ  ↔  Δk = G（倒格矢量）
結構因子：S(G) = Σⱼ fⱼ exp(-i G · rⱼ)
```

**Ch4 重點公式**：
```
1D 色散：ω² = (4C/M) sin²(ka/2)
3D Debye：D(ω) = (3V/2π²)(ω²/v³)
比熱：Cv = 9nkB (T/ΘD)³ ∫₀^{ΘD/T} x⁴eˣ/(eˣ-1)² dx
```

**Ch7 重點公式**：
```
近自由電子：Eg ≈ 2|V_G|（帶隙 ≈ 2 倍 Fourier 分量）
有效質量：1/m* = (1/ħ²) d²E/dk²
```

---

## 📖 Book 2：Ashcroft & Mermin — Solid State Physics

### 閱讀策略
- **選讀**：只讀與 Kittel 互補的章節
- 重點在彈性理論和近代方法

| 章 | 標題 | 時間 | 為何需要 |
|:--:|:-----|:----:|:---------|
| 4 | Crystal Lattices | 1天 | 更嚴格的晶格理論（補充 Kittel Ch1-2） |
| 5 | Reciprocal Lattice | 1天 | 更深入的倒格子推導 |
| 8 | Electron Levels (nearly free) | 1天 | 更詳細的能帶計算 |
| 10 | Tight Binding | 2天 | **重要**：LCAO → CGCNN 的原子軌道觀點 |
| 22 | Classical Theory of Harmonic Crystal | 2天 | 彈性張量的完整推導 |
| 23 | Quantum Theory of Harmonic Crystal | 1天 | 量子化聲子 |

**總計**：~180 頁，~8 天

---

## 📖 Book 3：Goodfellow et al. — Deep Learning

### 閱讀策略
- **精讀** Ch1-8（理論基礎）
- **精讀** Ch10（序列模型 → 理解 attention）
- **跳過** Ch9, 11-20（生成模型等，與本論文無關）

| 章 | 標題 | 時間 | 為何需要 | 完成標準 |
|:--:|:-----|:----:|:---------|:---------|
| 1 | Introduction | 1h | 大局觀 | 能說出 ML 三大類 |
| 2 | Linear Algebra | 2h | 數學準備 | 能做 SVD 和特徵分解 |
| 3 | Probability | 2h | 貝氏推論基礎 | 能推導 MLE 和 MAP |
| 4 | Numerical Computation | 2h | 梯度計算 | 理解計算穩定性 |
| 5 | ML Basics | 4h | 核心概念 | 能解釋 bias-variance |
| 6 | Deep Feedforward | 4h | MLP 理論 | 能從零實做 MLP |
| 7 | Regularization | 3h | 過擬合處理 | 能比較各種正則化方法 |
| 8 | Optimization | 3h | SGD → Adam | 能解釋動量和自適應 LR |
| 10 | Sequence Modeling | 2h | Attention 基礎 | 理解 attention 概念 |

**總計**：~350 頁，~23 小時（3-4 天密集閱讀）

---

## 📖 Book 4：Hamilton — Graph Representation Learning

### 閱讀策略
- **精讀** Ch1-7（全部讀）
- 這是 GNN 理論的核心教材

| 章 | 標題 | 時間 | 完成標準 |
|:--:|:-----|:----:|:---------|
| 1 | Introduction | 2h | 理解圖學習的 3 大任務 |
| 2 | Background and Notation | 3h | 能寫出鄰接矩陣、度矩陣 |
| 3 | Neighborhood Aggregation | 4h | 能推導 GCN layer |
| 4 | Multi-relational Data and Knowledge Graphs | 3h | 理解邊類型（如：晶體中的化學鍵） |
| 5 | GNN Architectures | 4h | 能比較 GCN, GAT, GraphSAGE, GIN |
| 6 | Theoretical Foundations | 3h | WL test, 表達力上界 |
| 7 | Generating Graphs | 2h | 了解生成模型（for 材料逆設計） |

**總計**：~200 頁，~21 小時（3 天）

---

## 📖 Book 5：Sholl & Steckel — DFT: A Practical Introduction

### 閱讀策略
- **精讀** Ch1-6（核心 DFT 理論）
- **選讀** Ch7-8（進階主題）

| 章 | 標題 | 時間 | 完成標準 |
|:--:|:-----|:----:|:---------|
| 1 | What is DFT? | 2h | 能解釋 DFT 解決什麼問題 |
| 2 | Theoretical Background | 4h | 能推導 HK 定理 |
| 3 | Kohn-Sham Equations | 4h | 能寫出 KS 方程 + SCF 迭代 |
| 4 | DFT Calculations (Setup) | 3h | 理解 k-points, cutoff, convergence |
| 5 | DFT Calculations (Results) | 3h | 理解各物性的 DFT 計算方法 |
| 6 | Electronic Structure | 3h | DOS, band structure calculation |
| 7 | Advanced Topics | 2h | 了解 DFPT, GW, hybrids |
| 8 | Surfaces and Reactions | 1h | 略讀 |

**總計**：~220 頁，~22 小時（4 天）

---

## 📖 Book 6：Vanderbilt — Berry Phases in Electronic Structure Theory

### 閱讀策略
- **精讀** Ch1-3（與壓電/介電直接相關）
- Ch4+ 為高階主題，可選讀

| 章 | 標題 | 時間 | 完成標準 |
|:--:|:-----|:----:|:---------|
| 1 | Adiabatic Evolution | 4天 | 能解釋 Berry phase 的幾何意義 |
| 2 | Berry Phase in Crystalline Solids | 3天 | 理解電極化的 Berry phase 理論 |
| 3 | Electric Polarization | 3天 | 理解 Born effective charge, piezoelectricity |

**總計**：~150 頁，~10 天

---

## 📄 論文閱讀清單（帶閱讀順序和重要度）

### 第一梯隊：必讀（★★★）— 15 篇
| # | 論文 | 年份 | 主題 | 閱讀週 |
|:-:|:-----|:----:|:-----|:------:|
| 1 | CGCNN (Xie & Grossman) | 2018 | Crystal graph convolution | W1 |
| 2 | SchNet (Schütt et al.) | 2018 | Continuous filter convolution | W2 |
| 3 | ALIGNN (Choudhary & DeCost) | 2021 | Line graph + angular features | W3 |
| 4 | NequIP (Batzner et al.) | 2022 | E(3)-equivariant GNN | W7 |
| 5 | MACE (Batatia et al.) | 2022 | Multi-body equivariant | W7 |
| 6 | e3nn (Geiger & Smidt) | 2022 | Euclidean neural networks library | W10 |
| 7 | Kendall et al. | 2018 | Multi-task uncertainty weighting | W8 |
| 8 | GradNorm (Chen et al.) | 2018 | Gradient normalization | W8 |
| 9 | PCGrad (Yu et al.) | 2020 | Projecting conflicting gradients | W8 |
| 10 | MC-Dropout (Gal & Ghahramani) | 2016 | Bayesian approximation | W13 |
| 11 | Deep Ensembles (Lakshminarayanan) | 2017 | Ensemble uncertainty | W13 |
| 12 | Evidential DL (Amini et al.) | 2020 | Single-pass uncertainty | W13 |
| 13 | JARVIS (Choudhary et al.) | 2020 | JARVIS database | W2 |
| 14 | Materials Project (Jain et al.) | 2013 | Largest materials database | W2 |
| 15 | DimeNet (Gasteiger et al.) | 2020 | Directional message passing | W7 |

### 第二梯隊：重要（★★）— 10 篇
| # | 論文 | 年份 | 主題 | 閱讀週 |
|:-:|:-----|:----:|:-----|:------:|
| 16 | MEGNet (Chen et al.) | 2019 | Global state features | W3 |
| 17 | PaiNN (Schütt et al.) | 2021 | Equivariant message passing | W10 |
| 18 | TFN (Thomas et al.) | 2018 | Tensor field networks | W11 |
| 19 | SE(3)-Transformers (Fuchs) | 2020 | Equivariant transformers | W11 |
| 20 | Lookman et al. | 2019 | Active learning for materials | W17 |
| 21 | Sener & Koltun | 2018 | Multi-objective optimization | W12 |
| 22 | M3GNet (Chen & Ong) | 2022 | Universal potential | W15 |
| 23 | CHGNet (Deng et al.) | 2023 | Charge-informed GNN | W15 |
| 24 | GNoME (Merchant et al.) | 2023 | Scaling graph networks | W15 |
| 25 | AFLOW (Curtarolo et al.) | 2012 | Automated workflow | W2 |

### 第三梯隊：參考（★）— 10+ 篇
| # | 論文 | 主題 |
|:-:|:-----|:-----|
| 26 | Cormorant (Anderson 2019) | Covariant neural networks |
| 27 | GIN (Xu et al. 2019) | Graph Isomorphism Network |
| 28 | Set2Set (Vinyals et al. 2016) | Graph-level pooling |
| 29 | Tran et al. (2020) | Active learning for catalysis |
| 30 | Born & Huang (1954) | Dynamical Theory of Crystal Lattices |
| 31+ | 相關 review 文章和博士論文 | 按需閱讀 |

---

## 每篇論文的閱讀模板

讀完每篇論文後，填寫以下模板：

```markdown
## 論文：[標題] ([作者], [年份])

### 核心問題
- 這篇論文要解決什麼問題？

### 方法
- 提出了什麼新方法/架構？
- 與之前的方法有何不同？

### 關鍵創新（1-3 點）
1. ...
2. ...
3. ...

### 實驗結果
- 在哪些 benchmark 上測試？
- 比之前的 SOTA 好多少？

### 與我的論文的關係
- 我能從這篇論文借鏡什麼？
- 哪些技術可以直接使用？

### 局限性
- 論文有哪些不足？
- 我的方法如何改善？

### 重要公式（top 3）
```
