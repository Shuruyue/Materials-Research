# ATLAS Research Direction & PhD Thesis Planning

Last updated: 2026-02-26

---

## 〇、研究者現況 (Researcher Profile)

| 項目 | 現況 |
|------|------|
| **階段** | ⚠️ 正在決定方向中（尚未定題） |
| **學歷背景** | 材料系 + 資工系學士一般課程，未專門深入 ML / 材料模擬 / 科學計算 |
| **目標方向** | 材料 × 資工結合，優先走演算法路線（ML 加速材料預測/模擬） |
| **研究身份** | **方法開發者**（造工具的人），但需要實驗驗證 |
| **預計產出** | 6 個月內完成投稿前準備 |
| **發表目標** | 中偏上期刊（如 *Computational Materials Science* IF~6 或 *npj Computational Materials* IF~12） |
| **可委託實驗** | 元智大學材料系 + 貴儀中心（XRD/SEM/熱分析等基本都能做） |
| **指導教授** | 物理系，做第一性原理，支持方向但不懂 ML → **教授負責物理審查，你負責 ML** |
| **畢業後** | 偏業界 → 需有可展示的系統/產品 |

### 硬體資源

| 設備 | 規格 | 實測參考 |
|------|------|---------|
| 實驗室桌機 | RTX 3060 (12GB) + i5-13400 + 32 GB RAM | Phase 1 Pro 模型 ≈ 20 小時 |
| 個人筆電 | RTX 4060 Laptop + R7 7735HS + 32 GB RAM | — |
| HPC / 雲端 | **不可用**（申請流程難、費用高，預設排除） | — |
| 訓練時間上限 | **3 天**（超過則錯誤代價太高） | — |

> [!IMPORTANT]
> 所有模型選型、資料集規模、訓練時間規劃，都必須在 **單張 3060/4060 + 32 GB RAM + ≤3天訓練** 的範圍內可行。

### 技術能力現況

| 技能 | 程度 |
|------|------|
| Python | ✅ 可用 |
| PyTorch | ⚠️ 僅上課用過 |
| GNN | ❌ 第一次接觸 |
| UQ (不確定性量化) | ❌ 無 |
| API (FastAPI等) | ⚠️ 理解概念但沒寫過 |
| Docker | ✅ 有經驗 |
| 材料直覺 | ✅ 金屬 / 半導體 |
| ATLAS 模型訓練 | ✅ 有跑過 |

---

## 一、問卷分析結論

### 三個關鍵洞察

**1. 教授是你的物理審查官（善用他）**
他做第一性原理 = 他能判斷你的 ML 預測在物理上合不合理。你不需要他懂 ML，你需要他告訴你：「這個 band gap 預測值正常嗎？」「這個 formation energy 合理嗎？」

**2. 瓶頸是學習曲線，不是算力**
PyTorch 初學、GNN 零基礎、UQ 零基礎 → 第一個月必須全力學。但你已經跑過 ATLAS 模型，有動手能力，學習速度不會太慢。

**3. 「材料缺陷是個問題」→ 潛在的第二年方向**
大多數 ML for materials 還停留在完美晶體（bulk），defect 預測是學術缺口。但資料少、構型空間大，6 個月內不適合當主軸。**先做 bulk crystal 打基礎，defect 當延伸方向。**

### 基於回答的策略調整

| 項目 | 原建議 | 調整後 |
|------|--------|--------|
| **貢獻點** | 三選一 | → **選 A（UQ + OOD 可信域）**，因為教授能幫驗證物理合理性 |
| **MVP-Service** | API 部署 | → **6 個月內不做 API**，專注 pipeline + benchmark，API 放第二年 |
| **Plan B** | 無 | → **descriptor + RF baseline vs GNN 的比較研究**，仍可發表 |
| **Gate 2** | 第 6 週做 API | → 第 6 週改為：**UQ 模組加入 + 第一個 OOD 實驗完成** |

---

## 二、核心命題

> 「如何設計一個最省成本的**取樣—擬合—積分—校正閉環**，使其在無機材料性質預測問題上保持可控誤差，並以平台化形態（類 Materials Project）交付？」

### 一句話版本

我們用 ML 取代/縮減 DFT，從公開 DFT 標註資料訓練 surrogate 模型，提供快速且帶不確定性估計的材料性質預測，打造「MP-lite」可部署平台。

---

## 三、三層瓶頸分析

| 層級 | 瓶頸 | 本專案的對策 |
|------|------|-------------|
| **物理層** | DFT 泛函近似、強關聯、有限溫度、缺陷/界面 | 使用公開高精度標註資料（JARVIS-DFT ~76K），不自行跑 DFT |
| **數值層** | 時間步長、誤差累積、取樣代表性、稀有事件 | 不確定性量化 + 主動學習校正閉環 |
| **計算架構層** | 馮紐曼瓶頸、memory wall | 本機 3060/4060，優先做可在單 GPU 跑的中等模型 |

---

## 四、邏輯鍊（Logic Chain）

```
材料宏觀性質 = 微觀自由度的統計平均/長時演化
  → 單次 DFT 只提供局部斜率（微分），不足以直接給宏觀行為
  → 需要動力學積分 (MD) 把力轉成軌跡→統計量
  → DFT-per-step 計算量崩潰
  → 解耦「力的取得」與「時間積分」
     = 離線高精度取樣(提取) → 擬合替代勢能/力場 → 線上長時間積分
  → 取樣不涵蓋未來軌跡 → 外推爆炸
  → 引入主動學習閉環（不確定性觸發補點 → 更新模型 → 繼續積分）
```

---

## 五、提取—積分—校正工作流

### Step 0：定義目標觀測量 (Observable)
- 目標：材料性質預測（band gap、formation energy、bulk/shear modulus 等）
- 驗證指標：Matbench 標準 benchmark（MAE、RMSE、R²）

### Step 1：選定微觀真理來源 (Reference)
- 使用公開 DFT 標註資料集（JARVIS-DFT、Materials Project、OQMD）
- 不自行做 DFT 計算

### Step 2：提取策略 (Sampling/Extraction)
- 直接使用 Matbench 標準任務資料集（已預處理、可公平評估）
- 確保覆蓋多樣化構型（穩定/不穩定、金屬/半導體/絕緣體）

### Step 3：擬合替代函數 (Surrogate Model)
- 基線：RF/descriptor baseline 或簡單 GNN
- 主打：GNN（CGCNN/EquivariantGNN）+ 不確定性輸出
- 物理約束：能量守恆、對稱性
- **硬體限制**：模型必須在 3060 (12GB VRAM) 上 ≤3 天訓練完

### Step 4：部署與推論 (Service) — *第二年*
- API 查詢入口：輸入材料（id/結構）→ 回傳預測性質 + 信心
- 批次篩選入口：輸入候選清單 → 回傳排序

### Step 5：閉環校正 (Active Learning)
- 觸發條件：不確定性高 / OOD 偵測
- 用公開資料中尚未使用的標註當 oracle 模擬 DFT 補點

### Step 6：驗證（雙重）
- **計算驗證**：Matbench fold-level 指標 + OOD transfer test
- **物理驗證**：請教授審查預測值的物理合理性
- **實驗驗證**（加分）：選 1–3 個預測結果，請材料系做樣品/量測對照

---

## 六、博士論文 3 Aims 架構

| Aim | 內容 | 產出 |
|-----|------|------|
| **Aim 1** (方法地基) | 建立 GNN + UQ pipeline，含 OOD 偵測與可信域界定 | 可重現 benchmark pipeline + 方法論 paper |
| **Aim 2** (科學問題) | 金屬/半導體性質預測：band gap + formation energy + 模數 | 應用 paper + benchmark 表格 + 教授審查 |
| **Aim 3** (可遷移性) | 跨材料 transfer + defect 初探（第二年延伸） | 泛化評估 + 失效模式分析 |

> **確定的貢獻點**：**A — UQ + OOD 可信域**（模型知道自己什麼時候不可靠）

---

## 七、6 個月 MVP 規格（主軸：Inorganic Bulk Crystal）

### MVP-Data
- 只做 inorganic bulk crystal（先不碰 surface / polymer / defect）
- 直接用 Matbench 任務資料

### MVP-Model
- 一個 baseline（RF/descriptor 或簡單 GNN）
- 一個主打模型（GNN + UQ）
- 不確定性輸出（ensemble / dropout 起步）
- **在 3060 12GB VRAM 上 ≤3 天训练完**

### MVP-Validation
- Matbench benchmark 表格
- OOD 實驗（抽掉某類材料 → 看模型是否能偵測到不可靠）
- 教授物理審查

---

## 八、里程碑 Gate（修正版）

| Gate | 時間點 | 判準 |
|------|--------|------|
| **Gate 1** | 第 2 週 | Matbench 端到端 pipeline 跑通（資料→baseline→CV→指標→可重現 repo） |
| **Gate 2** | 第 6 週 | UQ 模組加入 + 第一個 OOD 實驗完成 |
| **Gate 3** | 第 3 個月 | ≥1–3 個 Matbench 任務完成 + OOD 分析 + 貢獻點明確 |

> **Gate 3 不過**：立刻縮小至 1 個性質 + 1 個模型 + 1 個評估。
> **Plan B**：descriptor + RF baseline vs GNN 的比較研究（仍有發表價值）。

---

## 九、學科標籤

### 材料端
- Materials Informatics
- Computational Materials Science
- Inorganic / Solid-state Materials（金屬 + 半導體）

### 資工端
- Scientific ML / AI for Science
- Graph ML (GNN for atomistic systems)
- Uncertainty Quantification
- MLOps / Data & Model Provenance

### 前沿交叉 (Proposal Keywords)
- Machine Learning Force Fields (MLFF)
- Active Learning for Atomistic Simulation
- Out-of-Distribution Detection for Materials
- Physics-informed ML

---

## 十、選題自問清單（已填答）

> 以下問題用於幫助收斂研究方向。

### A. 研究動機與定位

1. 你最終想成為「做方法的人」還是「用方法解材料問題的人」？
   - 回答：作方法的人 但是還是要做驗證
2. 你的博士論文讀者是材料領域的人、還是資工領域的人、還是兩者都要說服？
   - 回答：兩者都要
3. 你有沒有一個「如果這個做出來我會很興奮」的具體場景？
   - 回答：這挺不錯的 但是材料缺陷是個問題
4. 你願意在 6 個月後發表在哪類期刊/會議？
   - 回答：能的話我希望至少中偏上

### B. 材料領域選擇

5. 哪個材料系統你最有直覺？
   - 回答：金屬 半導體
6. 元智材料系可以幫你做什麼？
   - 回答：基本都行 有貴儀中心
7. 你需不需要「實驗驗證」？
   - 回答：需要 但是計算過程同樣需要可驗證性
8. 有沒有特定的應用場景？
   - 回答：目前還沒決定

### C. 資工 / 演算法能力

9. PyTorch 熟悉度？ — 僅僅上課用過
10. GNN？ — 第一次接觸
11. UQ？ — 無
12. API？ — 理解概念但沒寫過
13. Docker？ — 有

### D. 資料與計算

14. JARVIS-DFT？ — 無，有在學習中
15. ATLAS 模型訓練過？ — 有
16. 3060 訓練時間？ — Phase 1 Pro 約 20 小時
17. 訓練上限？ — 3 天

### E. 風險管理

18. Plan B？ — 無（待補：建議 descriptor + RF vs GNN 比較研究）
19. 負面結果？ — 可以，但必須非重複且有貢獻
20. 指導教授？ — 物理系，做第一性原理，支持方向但不懂 ML

### F. 長期願景

21. 業界 vs 學界？ — 偏業界
22. 平台服務誰？ — 畢業 + 貢獻，公開但核心留給研究組/專利
23. 3 年後？ — 沒怎麼想過
24. 合作？ — 沒有，前端交給 AI
25. 台灣市場？ — 沒有競爭力，但國外很多前沿實驗室都在做
