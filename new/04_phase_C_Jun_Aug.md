# 階段 C：實驗執行與深度分析（7月 – 8月）

> **目標**：完成所有消融實驗、深度分析、製作論文圖表
> **每日投入**：10-12 小時（密集期）
> **產出**：完整結果表 × 8、高品質學術圖表 30+、分析報告

---

## 第 21 週（7/7 – 7/13）：系統性消融實驗

### 消融實驗矩陣

| ID | 實驗 | 變數 | 基準 | 預期時間 |
|:--:|:-----|:-----|:-----|:--------:|
| A1 | 架構比較 | CGCNN/SchNet/ALIGNN/E(3)-GNN | CGCNN | 4 天 |
| A2 | 多任務效果 | single-task vs multi-task | single | 2 天 |
| A3 | 任務配對 | Eg+Ef / Eg+K / Ef+K / 全部 | 全部 | 3 天 |
| A4 | 權重策略 | fixed/uncertainty/GradNorm/PCGrad | uncertainty | 3 天 |
| A5 | 物理約束 | 有/無 physics constraint | 無 | 1 天 |
| A6 | 標量vs張量 | K from scalar head vs K from Cij→K | scalar | 1 天 |
| A7 | 數據量 | 1K/5K/10K/20K/50K/全量 | 全量 | 2 天 |
| A8 | UQ + AL | 有/無 UQ 的 active learning | 隨機 | 2 天 |

**週一 7/7**
- [ ] 上午：規劃消融實驗的完整腳本
  - 每個實驗：明確的輸入、輸出、控制變數
  - 建立自動化腳本批量跑實驗
  ```python
  # ablation_runner.py
  # 自動跑 A1-A8 的所有組合
  # 結果寫入 JSON + CSV
  ```
- [ ] 下午：開始 A1 — 架構比較
  - CGCNN（已有結果）
  - EquivariantGNN（已有結果）
  - 整理成統一格式的表格
- [ ] 晚間：準備 SchNet 和 ALIGNN 的代碼（如有需要從論文 repo）

**週二 7/8**
- [ ] 全天：A1 — 繼續架構比較
  - 如果 SchNet/ALIGNN 需要額外跑，開始訓練
  - 可用文獻報告的數字作為 baseline

**週三 7/9**
- [ ] 上午：A2 — single-task vs multi-task
  - 整理前階段的單任務和多任務結果
  - 製作對比表格
- [ ] 下午：A3 — 任務配對分析
  - 設計 6 種兩兩配對和 4 種三合一組合
  - 開始跑對應的多任務訓練

**週四 7/10 – 週五 7/11**
- [ ] A4 — 權重策略比較
  ```python
  strategies = ["fixed", "uncertainty", "gradnorm", "pcgrad"]
  for s in strategies:
      # 用相同數據和模型架構
      # 只變更 task weighting 策略
      train_multitask(strategy=s, ...)
  ```
- [ ] 監控訓練進度
- [ ] 開始整理結果

**週六 7/12 – 週日 7/13**
- [ ] A5 — 物理約束消融
- [ ] 整理本週結果
- [ ] 休息

> **本週驗收**：
> - ✅ A1-A5 實驗完成（至少初步結果）
> - ✅ 統一格式的結果表格

---

## 第 22 週（7/14 – 7/20）：數據量 + UQ 消融

**週一 7/14 – 週三 7/16**
- [ ] A7 — 數據量消融（learning curve）
  ```python
  for n in [1000, 5000, 10000, 20000, 50000, None]:
      train_equivariant(max_samples=n, ...)
  ```
  - 在每個數據量上跑等變 GNN + CGCNN
  - 畫 learning curve（x=數據量, y=MAE）

**週四 7/17 – 週五 7/18**
- [ ] A8 — Active Learning 消融
  ```bash
  python scripts/31_active_learning.py --strategy uncertainty --initial 1000 --query-size 100
  python scripts/31_active_learning.py --strategy random --initial 1000 --query-size 100
  python scripts/31_active_learning.py --strategy expected_improvement --initial 1000
  ```
  - 比較三種 acquisition function 的 learning curve

**週六 7/19 – 週日 7/20**
- [ ] A6 — 標量 vs 張量預測的 derived scalar 比較
  - K（直接預測）vs K_Voigt（從 Cij 計算）
  - 測試哪個更準確
- [ ] 整理本週結果

---

## 第 23 週（7/21 – 7/27）：高品質學術圖表製作

### 必須的圖表清單

| # | 圖表 | 對應論文章節 |
|:-:|:-----|:------------|
| 1 | 模型架構圖（E(3)-GNN 全架構） | Ch4 Methodology |
| 2 | 各架構 MAE 對比 bar chart | Ch5.1 |
| 3 | multi-task vs single-task 表格 | Ch5.2 |
| 4 | 任務配對 heatmap | Ch5.3 |
| 5 | 權重策略比較 bar chart | Ch5.3 |
| 6 | 物理約束 before/after 散點圖 | Ch5.4 |
| 7 | 彈性張量 parity plot (predicted vs DFT) | Ch6.1 |
| 8 | Frobenius error 分布 | Ch6.1 |
| 9 | 晶系特定的 Cij 精度 | Ch6.4 |
| 10 | derived scalar (K, G from tensor) parity | Ch6.5 |
| 11 | GNNExplainer 元素重要性 heatmap | Ch7.1 |
| 12 | Latent space t-SNE colored by Eg | Ch7.2 |
| 13 | Latent space t-SNE colored by K | Ch7.2 |
| 14 | Property-property correlation matrix | Ch7.3 |
| 15 | 梯度對齊 cosine similarity heatmap | Ch7.4 |
| 16 | Active learning curve (3 strategies) | Ch8.1 |
| 17 | Pareto front scatter plot | Ch8.2 |
| 18 | Data scaling curve | Ch5 supplementary |
| 19 | Learning rate curve during training | Supplementary |
| 20 | Convergence comparison (CGCNN vs E3-GNN) | Supplementary |

**週一 7/21 – 週三 7/23**
- [ ] 學術圖表風格設定
  ```python
  import matplotlib.pyplot as plt
  plt.style.use('seaborn-v0_8-paper')
  plt.rcParams.update({
      'font.size': 12,
      'font.family': 'serif',
      'axes.labelsize': 14,
      'figure.dpi': 300,
      'savefig.bbox': 'tight',
  })
  ```
- [ ] 製作圖 1-10（結果類圖表）
  - 每張圖至少 300 DPI
  - 統一色彩方案
  - 包含 error bars（如有 cross-validation）

**週四 7/24 – 週五 7/25**
- [ ] 製作圖 11-17（分析類圖表）
  - GNNExplainer heatmap
  - Latent space 可視化
  - Pareto front

**週六 7/26 – 週日 7/27**
- [ ] 製作圖 18-20（補充材料）
- [ ] 圖表品質檢查
  - [ ] 字體大小是否在印刷後可讀？
  - [ ] 色彩是否色盲友善？
  - [ ] 圖例是否完整？
  - [ ] 軸標籤是否包含單位？
- [ ] 休息

---

## 第 24 週（7/28 – 8/3）：深度物理分析

### 核心問題

**週一 7/28 – 週二 7/29**
- [ ] 分析 1：正遷移與負遷移的物理解釋
  - 問題：為何 Eg ↔ ε 正遷移？（Penn model: ε ∝ 1/Eg²）
  - 問題：為何 K ↔ Ef 正遷移？（鍵強度的共同因素）
  - 問題：K ↔ Eg 是否負遷移？（金屬 vs 半導體的矛盾）
  - 撰寫物理解釋段落（500+ 字）

**週三 7/30 – 週四 7/31**
- [ ] 分析 2：latent space 的物理意義
  - 是否存在「金屬 cluster」和「半導體 cluster」？
  - latent space 中的距離是否反映結構相似性？
  - 能否在 latent space 找到「物性邊界」？
  - 撰寫分析段落

**週五 8/1** ← **里程碑 ⑥**
- [ ] 所有實驗結果最終整理
  > ✅ 消融實驗 A1-A8 完成
  > ✅ 高品質圖表 20+ 張
  > ✅ 物理分析段落 3000+ 字

**週六 8/2 – 週日 8/3**
- [ ] 製作實驗結果的電子表格（Excel 備份）
- [ ] 休息

---

## 第 25-26 週（8/4 – 8/17）：補充實驗 + 5-fold CV

### 第 25 週
- [ ] 5-fold cross validation
  ```python
  for fold in range(5):
      # split data with different random seed
      # train + evaluate
      # record per-fold results
  ```
- [ ] 計算 mean ± std
- [ ] paired t-test / Wilcoxon signed-rank test
  ```python
  from scipy.stats import ttest_rel, wilcoxon
  # 比較 E(3)-GNN vs CGCNN 的 5 fold results
  ```

### 第 26 週
- [ ] 補充不足的實驗
- [ ] 確認所有結果的可復現性
- [ ] 清理代碼、整理所有結果到統一目錄
- [ ] 撰寫 README 和復現指南

> **階段 C 最終驗收**：
> - ✅ 8 組消融實驗完成（A1-A8）
> - ✅ 5-fold cross validation 完成
> - ✅ 統計顯著性檢驗完成
> - ✅ 高品質圖表 30+ 張
> - ✅ 物理分析 3000+ 字
> - ✅ 所有結果可復現
