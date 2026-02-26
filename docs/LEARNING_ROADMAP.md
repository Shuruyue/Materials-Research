# ATLAS 學習路線圖 — 從零到研究級

Last updated: 2026-02-26

> 假設你大學都沒讀過，要徹底理解本專案涉及的所有知識，需要學什麼。
> 優先級：**P0**（不會就動不了）→ **P1**（第一個月要會）→ **P2**（前三個月補齊）→ **P3**（進階 / 可選）

---

## P0 — 不會就完全動不了（第 1–2 週必須有基礎）

### 1. Python 程式設計

| 項目 | 內容 |
|------|------|
| **你要會** | 變數、迴圈、函數、class、模組、pip、虛擬環境、讀寫檔案 |
| **課本** | [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/) — 免費線上 |
| **影片** | Corey Schafer YouTube Python 系列（英文、免費） |
| **補充** | Python 官方 Tutorial: https://docs.python.org/3/tutorial/ |

### 2. NumPy / Pandas 基礎

| 項目 | 內容 |
|------|------|
| **你要會** | ndarray 操作、indexing / slicing、broadcasting、DataFrame 讀寫/篩選/groupby |
| **課本** | Wes McKinney, *Python for Data Analysis*, 3rd ed. (O'Reilly) |
| **影片** | Keith Galli — NumPy & Pandas YouTube 教學 |
| **練習** | 用 Pandas 載入 JARVIS-DFT JSON 並做統計（你的 Step 1） |

### 3. Git 版本控制

| 項目 | 內容 |
|------|------|
| **你要會** | init、add、commit、branch、merge、push/pull、.gitignore |
| **課本** | Pro Git Book: https://git-scm.com/book/zh-tw/v2 （有中文） |
| **練習** | 把你的 ATLAS 專案做一次乾淨的 commit history 整理 |

---

## P1 — 第一個月要會（能跑通 pipeline）

### 4. 線性代數

| 項目 | 內容 |
|------|------|
| **你要會** | 向量、矩陣乘法、轉置、行列式、特徵值/特徵向量、SVD、正交基底 |
| **課本** | Gilbert Strang, *Introduction to Linear Algebra*, 6th ed. |
| **影片** | 3Blue1Brown — *Essence of Linear Algebra*（YouTube 免費，必看，視覺化極好） |
| **補充** | MIT OCW 18.06 Linear Algebra（Gilbert Strang 親授） |

### 5. 微積分 + 多變量微積分

| 項目 | 內容 |
|------|------|
| **你要會** | 導數、偏導數、鏈鎖律（chain rule，ML 反向傳播的核心）、梯度、積分基礎 |
| **課本** | James Stewart, *Calculus: Early Transcendentals*, 9th ed. |
| **影片** | 3Blue1Brown — *Essence of Calculus*（YouTube 免費） |
| **重點** | 理解梯度下降的數學意義（這是所有 ML 訓練的根基） |

### 6. 機率與統計

| 項目 | 內容 |
|------|------|
| **你要會** | 機率分佈（正態/均勻/伯努利）、期望值、方差、貝氏定理、最大似然估計 (MLE)、交叉驗證 |
| **課本** | Sheldon Ross, *A First Course in Probability*, 10th ed. |
| **影片** | StatQuest by Josh Starmer（YouTube，極好的直覺解釋） |
| **補充** | Khan Academy — Statistics and Probability |

### 7. PyTorch 深度學習框架

| 項目 | 內容 |
|------|------|
| **你要會** | Tensor 操作、autograd、nn.Module、training loop、DataLoader、GPU 搬運、儲存/載入模型 |
| **課本** | PyTorch 官方 Tutorials: https://pytorch.org/tutorials/ |
| **影片** | Daniel Bourke — *PyTorch for Deep Learning*（YouTube，完整免費課） |
| **練習** | 用 PyTorch 從零寫一個 MNIST 分類器（確保你懂 forward / backward / optimizer.step） |
| **補充** | Andrej Karpathy — *Neural Networks: Zero to Hero*（YouTube 系列，極推） |

### 8. 機器學習基礎

| 項目 | 內容 |
|------|------|
| **你要會** | 監督學習（回歸/分類）、損失函數、過擬合/欠擬合、正則化、train/val/test split、評估指標（MAE/RMSE/R²） |
| **課本** | Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 3rd ed. (O'Reilly) — 前 8 章 |
| **影片** | Andrew Ng — *Machine Learning Specialization*（Coursera，經典） |
| **補充** | scikit-learn 官方 User Guide: https://scikit-learn.org/stable/user_guide.html |

---

## P2 — 前三個月補齊（能做研究、寫論文）

### 9. 深度學習理論

| 項目 | 內容 |
|------|------|
| **你要會** | CNN、RNN（概念即可）、注意力機制、BatchNorm、Dropout、學習率排程、早停 |
| **課本** | Ian Goodfellow et al., *Deep Learning* (MIT Press) — 免費線上: https://www.deeplearningbook.org/ |
| **影片** | Stanford CS231n（CNN）、CS224n（NLP / attention） |
| **重點** | 重點不是全學，而是理解你程式碼裡用到的每個元件 |

### 10. 圖神經網路 (GNN) ⭐ 核心

| 項目 | 內容 |
|------|------|
| **你要會** | 圖的表示（node/edge/adjacency）、message passing、聚合、global pooling、PyTorch Geometric |
| **課本** | Hamilton, *Graph Representation Learning* (Morgan & Claypool) — 免費線上: https://www.cs.mcgill.ca/~wlh/grl_book/ |
| **影片** | Stanford CS224W — *Machine Learning with Graphs*（YouTube 免費，Jure Leskovec） |
| **論文** | Gilmer et al., "Neural Message Passing for Quantum Chemistry" (ICML 2017) — GNN for molecules 的開山之作 |
| **實作** | PyTorch Geometric 官方教學: https://pytorch-geometric.readthedocs.io/ |
| **補充** | Xie & Grossman, "Crystal Graph Convolutional Neural Networks" (PRL 2018) — 你專案裡 CGCNN 的原始論文 |

### 11. 材料科學基礎

| 項目 | 內容 |
|------|------|
| **你要會** | 晶體結構（Bravais 晶格、米勒指標）、能帶理論（band gap）、彈性模數、形成能、熱力學穩定性 (ehull) |
| **課本** | William D. Callister, *Materials Science and Engineering: An Introduction*, 10th ed. — 材料系經典教科書 |
| **補充** | Kittel, *Introduction to Solid State Physics*, 8th ed. — 固態物理（偏物理但解釋能帶/晶格很清楚） |
| **線上** | Materials Project 教學文件: https://docs.materialsproject.org/ |

### 12. DFT 概念（不用會算，要懂原理）

| 項目 | 內容 |
|------|------|
| **你要會** | Schrödinger 方程式（概念）、Born-Oppenheimer 近似、電子密度、Kohn-Sham 方程式、泛函近似（LDA/GGA/PBE）、什麼是 VASP/QE |
| **課本** | Sholl & Steckel, *Density Functional Theory: A Practical Introduction* (Wiley) — 最適合非物理/化學背景的入門 |
| **影片** | TMP Chem YouTube — DFT 入門系列 |
| **重點** | 你不需要會跑 DFT，但你必須理解你資料集的標籤是怎麼算出來的 |

### 13. pymatgen 套件

| 項目 | 內容 |
|------|------|
| **你要會** | Structure 物件、Site / Lattice / Species、鄰居搜尋、CIF 讀寫 |
| **文件** | https://pymatgen.org/ |
| **練習** | 從 JARVIS 資料取一個結構，用 pymatgen 算鄰居列表，跟你專案的 graph_builder 對照 |

### 14. 不確定性量化 (UQ)

| 項目 | 內容 |
|------|------|
| **你要會** | 認識兩種不確定性（aleatoric vs epistemic）、ensemble、MC Dropout、Evidential Deep Learning |
| **論文** | Amini et al., "Deep Evidential Regression" (NeurIPS 2020) — 你程式碼裡 EvidentialHead 的來源 |
| **論文** | Kendall & Gal, "What Uncertainties Do We Need in Bayesian DL?" (NeurIPS 2017) — MultiTaskLoss 的理論基礎 |
| **補充** | Gal, "Uncertainty in Deep Learning" PhD Thesis (Cambridge, 2016) — 免費線上 |

---

## P3 — 進階 / 可選（做到出色需要）

### 15. 等變神經網路 (Equivariant GNN)

| 項目 | 內容 |
|------|------|
| **你要會** | 群論基礎（SO(3)、旋轉群）、球諧函數、irreducible representations、tensor product |
| **課本** | Tinkham, *Group Theory and Quantum Mechanics* — 物理向 |
| **影片** | e3nn 官方教學: https://e3nn.org/ |
| **論文** | Batzner et al., "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials" (Nature Communications, 2022) — NequIP，你的 EquivariantGNN 參考 |
| **補充** | Geiger & Smidt, "e3nn: Euclidean Neural Networks" (2022) |

### 16. 主動學習 (Active Learning)

| 項目 | 內容 |
|------|------|
| **你要會** | pool-based vs stream-based、acquisition function（uncertainty、EI、UCB）、query strategy |
| **課本** | Settles, "Active Learning Literature Survey" (2009) — 經典綜述 |
| **論文** | Smith et al., "Less is more: Sampling chemical space with AL" (JCP, 2018) |
| **補充** | BoTorch 官方文件: https://botorch.org/ |

### 17. 貝氏最佳化 (Bayesian Optimization)

| 項目 | 內容 |
|------|------|
| **你要會** | Gaussian Process (GP)、Prior/Posterior、acquisition functions |
| **課本** | Rasmussen & Williams, *Gaussian Processes for ML* (MIT Press) — 免費線上 |
| **補充** | GPyTorch 文件: https://gpytorch.ai/ |

### 18. 多尺度模擬概念

| 項目 | 內容 |
|------|------|
| **概念** | 電子尺度 → 原子尺度 → 介觀 → 宏觀的銜接方式 |
| **課本** | Tadmor & Miller, *Modeling Materials* (Cambridge) |
| **補充** | LeSar, *Introduction to Computational Materials Science* |

### 19. 軟體工程實務（寫出可發表的 repo）

| 項目 | 內容 |
|------|------|
| **你要會** | 單元測試（pytest）、CI/CD（GitHub Actions）、文件（README/docstring）、linting（ruff） |
| **課本** | Percival & Gregory, *Test-Driven Development with Python* — 免費線上 |
| **補充** | https://docs.pytest.org/ |

### 20. 論文寫作

| 項目 | 內容 |
|------|------|
| **你要會** | LaTeX / Overleaf、文獻管理（Zotero）、圖表規範、摘要/結論的寫法 |
| **課本** | Booth et al., *The Craft of Research* (U Chicago Press) |
| **補充** | Simon Peyton Jones, "How to write a great research paper" (Microsoft Research, YouTube) |

---

## 建議學習路線（時間軸）

```
第 1–2 週    P0 全部 + P1 的 PyTorch 基礎
             → 目標：能跑通 ATLAS 的 CGCNN 訓練

第 3–4 週    P1 剩餘（線代、微積分、機率、ML 基礎）
             → 目標：理解 training loop 裡每一行在做什麼

第 2 個月    P2 的 GNN + 材料基礎 + DFT 概念
             → 目標：能解釋你的模型為什麼這樣設計

第 3 個月    P2 的 UQ + pymatgen + 開始寫實驗
             → 目標：Gate 3 通過

第 4–6 個月  P3 按需求挑讀 + 論文寫作
             → 目標：投稿準備完成
```

---

## 免費資源快速清單

| 資源 | 用途 | 連結 |
|------|------|------|
| 3Blue1Brown | 線代/微積分直覺 | youtube.com/@3blue1brown |
| StatQuest | 統計/ML 直覺 | youtube.com/@statquest |
| Andrej Karpathy | 神經網路從零 | youtube.com/@andrejkarpathy |
| Stanford CS224W | GNN 完整課程 | youtube.com (搜 CS224W) |
| PyTorch Tutorials | 框架入門 | pytorch.org/tutorials |
| PyG Tutorials | 圖神經網路實作 | pytorch-geometric.readthedocs.io |
| Deep Learning Book | DL 理論 | deeplearningbook.org |
| GRL Book | 圖表示學習 | cs.mcgill.ca/~wlh/grl_book |
| Materials Project Docs | 材料資料庫 | docs.materialsproject.org |
| Pro Git (中文) | Git | git-scm.com/book/zh-tw |
