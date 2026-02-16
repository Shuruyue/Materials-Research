# 博士論文完整執行企劃書
## E(3)-等變多任務圖神經網路預測晶體材料物性

---

## 🎯 總目標

從零開始，到完成一篇可答辯的博士論文。涵蓋：
- **理論基礎**：固態物理 + 量子力學 + 密度泛函理論
- **方法論**：圖神經網路 + 等變網路 + 多任務學習
- **實驗**：模型開發 + 訓練 + 消融實驗
- **論文**：撰寫 + 修改 + 答辯準備

---

## 📅 四大階段總覽

| 階段 | 時間 | 主題 | 每日時數 | 產出 |
|:----:|:----:|:-----|:-------:|:-----|
| **A** | 2-4 月 | 基礎知識建構 | 6-8h | 讀書筆記 × 20+ 章, 代碼練習 |
| **B** | 4-6 月 | 核心技術深入 | 8-10h | 論文精讀 × 30+, Phase 2-3 實驗 |
| **C** | 6-8 月 | 實驗執行與分析 | 10-12h | 完整實驗結果, 圖表, 分析 |
| **D** | 8-12 月 | 論文撰寫與答辯 | 8-10h | 完整論文初稿 + 修改 + 投稿 |

---

## 📚 核心教材（按階段排列）

### 階段 A 教材
| # | 書名 | 作者 | 重點章節 | 預估時間 |
|:-:|:-----|:-----|:---------|:--------:|
| 1 | Introduction to Solid State Physics (8th) | Kittel | Ch1-8, 13 | 4 週 |
| 2 | Solid State Physics | Ashcroft & Mermin | Ch4-10, 22-23 | 3 週 |
| 3 | Deep Learning | Goodfellow et al. | Ch1-8, 10 | 2 週 |
| 4 | Graph Representation Learning | Hamilton | Ch1-7 | 1 週 |

### 階段 B 教材
| # | 書名 | 作者 | 重點章節 | 預估時間 |
|:-:|:-----|:-----|:---------|:--------:|
| 5 | DFT: A Practical Introduction | Sholl & Steckel | Ch1-8 | 2 週 |
| 6 | Electronic Structure (Basic Theory) | Martin | Ch1-7, 10-12 | 2 週 |
| 7 | Berry Phases in Electronic Structure Theory | Vanderbilt | Ch1-3 | 2 週 |
| 8 | Pattern Recognition and ML | Bishop | Ch1-5, 9 | 1 週 |

### 階段 B-C 必讀論文（30 篇）
| 類別 | 論文 | 優先度 |
|:-----|:-----|:------:|
| **GNN 基礎** | CGCNN (Xie 2018), SchNet (Schütt 2018), MEGNet (Chen 2019) | ⭐⭐⭐ |
| **高階 GNN** | ALIGNN (Choudhary 2021), DimeNet (Gasteiger 2020), PaiNN (Schütt 2021) | ⭐⭐⭐ |
| **等變網路** | NequIP (Batzner 2022), MACE (Batatia 2022), e3nn (Geiger 2022) | ⭐⭐⭐ |
| **理論** | Tensor Field Networks (Thomas 2018), SE(3)-Transformers (Fuchs 2020) | ⭐⭐ |
| **多任務** | Kendall (2018) uncertainty, GradNorm (Chen 2018), PCGrad (Yu 2020) | ⭐⭐⭐ |
| **不確定性** | MC-Dropout (Gal 2016), Evidential DL (Amini 2020), Ensemble (Lakshminarayanan 2017) | ⭐⭐ |
| **主動學習** | Lookman (2019), Tran (2020) AL for materials | ⭐⭐ |
| **張量預測** | Tensor properties GNN (various recent) | ⭐⭐ |
| **材料數據** | JARVIS (Choudhary 2020), Materials Project (Jain 2013), AFLOW (Curtarolo 2012) | ⭐⭐ |

---

## 📂 檔案結構

```
new/
├── 00_master_plan.md           ← 本檔案（總覽）
├── 01_phase1_fix.md            ← Phase 1 CGCNN 修復方案
├── 02_phase_A_Feb_Apr.md       ← 階段 A：基礎知識（每日細項）
├── 03_phase_B_Apr_Jun.md       ← 階段 B：核心技術（每日細項）
├── 04_phase_C_Jun_Aug.md       ← 階段 C：實驗執行（每日細項）
├── 05_phase_D_Aug_Dec.md       ← 階段 D：論文撰寫（每周細項）
├── 06_reading_list.md          ← 完整逐章閱讀指南
└── 07_weekly_checklist.md      ← 每周自我檢核表
```

---

## ⚙️ 每日時間分配建議

```
┌─────────────────────────────────────────────────────────┐
│  上午 (9:00-12:00) │ 理論學習：讀原文書 + 做筆記       │
│  午休 (12:00-13:30)│ 休息                               │
│  下午 (13:30-17:30)│ 技術實作：寫代碼 + 跑實驗          │
│  晚間 (19:00-21:00)│ 論文精讀 + 寫作 / 整理筆記         │
└─────────────────────────────────────────────────────────┘
```

---

## 🏆 里程碑檢查點

| 日期 | 里程碑 | 驗收標準 |
|:----:|:-------|:---------|
| 3/15 | Kittel Ch1-8 完成 | 能解釋布里淵區、能帶理論、聲子 |
| 4/01 | ML/DL 基礎完成 | 能從頭實做 MLP + CNN + GNN |
| 4/15 | Phase 1 baseline 全通 | 4 個物性 MAE ≤ target |
| 5/15 | 等變 GNN 理論完成 | 能解釋球諧函數、不可約表示、張量積 |
| 6/01 | Phase 2 equivariant 訓練完成 | 等變 GNN MAE 優於 CGCNN |
| 7/01 | Phase 3-4 多任務 + 張量完成 | 多任務優於單任務, 彈性張量可預測 |
| 8/01 | Phase 5-6 explainability + AL 完成 | latent space 分析 + 候選材料 |
| 9/01 | 論文初稿 Ch1-4 完成 | 導論 + 文獻 + 理論 + 方法 |
| 10/01 | 論文初稿 Ch5-8 完成 | 全部結果章節 |
| 11/01 | 論文修改完成 | 指導教授審核通過 |
| 12/01 | 答辯準備 | 投影片 + 模擬答辯 |

---

> [!IMPORTANT]
> 這份計畫的核心理念是「每天都有具體的、可量化的產出」。
> 不是「今天讀了一些東西」，而是「今天完成了 Kittel Ch3 的筆記，
> 且能用代碼重現 FCC 的第一布里淵區。」
