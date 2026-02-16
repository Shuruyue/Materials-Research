# 階段 D：論文撰寫與答辯準備（8月中 – 12月）

> **目標**：完成博士論文全文（9 章 + 附錄），通過答辯
> **每日投入**：8-10 小時
> **產出**：完整論文初稿 → 修改稿 → 最終版 + 投影片

---

## 寫作原則

1. **先框架後填充**：每章先寫 outline，再逐節填充
2. **每天至少 1000 字**（中文）或 **500 字**（英文）
3. **先粗後精**：初稿不追求完美，重點是把所有內容寫出來
4. **圖先於文**：每一節先確定要放哪些圖表，再圍繞圖表寫文字
5. **每週交一章給指導教授**

---

## 第 27-28 週（8/18 – 8/31）：Ch1 Introduction + Ch2 Literature Review

### 第 27 週

**週一 8/18**
- [ ] Ch1 Introduction — 架構 outline
  ```
  1.1 背景：Materials Genome Initiative → ML for materials
  1.2 挑戰：DFT 計算量大 → ML 加速
  1.3 問題定義：多物性同時預測 + 物理一致性
  1.4 本文貢獻（4 點）
  1.5 論文結構
  ```
- [ ] 開始寫 1.1（1000 字）

**週二 8/19**
- [ ] 寫 Ch1.2-1.3（2000 字）
  - 從第一原理到 ML 的範式轉移
  - 引用 MGI (White House 2011), Materials Project, JARVIS
- [ ] 晚間：列出 Ch1 需要的所有引用文獻

**週三 8/20**
- [ ] 寫 Ch1.4 本文貢獻
  ```
  貢獻 1：設計 E(3)-equivariant multi-task GNN
  貢獻 2：Physics-constrained loss (Born stability, Voigt-Reuss)
  貢獻 3：完整彈性張量 + 介電張量預測
  貢獻 4：基於 UQ 的主動學習材料篩選
  ```
- [ ] 寫 Ch1.5 論文結構（500 字）
- [ ] 完成 Ch1 初稿，自己通讀一遍

**週四 8/21**
- [ ] Ch2 Literature Review — 架構 outline
  ```
  2.1 GNN for materials: CGCNN → SchNet → ALIGNN → NequIP → MACE
  2.2 Multi-task learning: theory, methods, and applications in materials
  2.3 Equivariant neural networks: e3nn, tensor field networks
  2.4 Uncertainty quantification: ensemble, MC-dropout, evidential
  2.5 Active learning for materials discovery
  ```
- [ ] 開始寫 2.1（2000 字）

**週五 8/22**
- [ ] 寫 2.2 Multi-task learning（2000 字）
  - 硬參數共享 vs 軟參數共享
  - Kendall, GradNorm, PCGrad, MGDA
  - 在材料科學中的應用
- [ ] 晚間：整理引用文獻

**週六 8/23 – 週日 8/24**
- [ ] 寫 2.3 等變網路（2000 字）
  - 對稱性在物理中的角色
  - 從不變 (CGCNN) 到等變 (NequIP) 的演進
- [ ] 休息

### 第 28 週

**週一 8/25**
- [ ] 寫 2.4 不確定性量化（1500 字）
- [ ] 寫 2.5 主動學習（1500 字）
- [ ] 完成 Ch2 初稿

**週二 8/26**
- [ ] 自我審閱 Ch1-2
  - 檢查邏輯流程
  - 檢查引用完整性
  - 標記需要改進的段落

**週三 8/27 – 週五 8/29**
- [ ] 修改 Ch1-2
- [ ] 交給指導教授審閱

**週六 8/30 – 週日 8/31**
- [ ] 等待回饋，開始準備 Ch3
- [ ] 休息

---

## 第 29-30 週（9/1 – 9/14）：Ch3 Theory + Ch4 Methodology

### 第 29 週

**週一 9/1** ← **里程碑 ⑦ Ch1-2 初稿交出**

**週一 9/1 – 週三 9/3**
- [ ] Ch3 Theory 撰寫
  ```
  3.1 晶體圖表示 (crystal graph representation)
      - 節點：原子特徵向量
      - 邊：距離 + 向量
      - 周期性邊界條件
      
  3.2 E(3)-等變性與不可約表示
      - SO(3), O(3), E(3) 群的定義
      - 標量 (L=0), 向量 (L=1), 張量 (L=2, 4)
      - Wigner-D 矩陣
      - 等變性的數學定義：f(Rr) = D(R)f(r)
      
  3.3 球諧分解與張量性質
      - Cij → L=0 ⊕ L=2 ⊕ L=4
      - εij → L=0 ⊕ L=2
      - CG 係數
      
  3.4 多任務學習：任務相關性與遷移
      - 正遷移 vs 負遷移
      - Kendall uncertainty weighting 推導
      
  3.5 不確定性量化
      - Bayesian 框架
      - MC-Dropout ← variational inference
      - Deep Ensemble
      - Evidential regression
  ```
- [ ] 每節 1500-2000 字，含公式推導

**週四 9/4 – 週五 9/5**
- [ ] Ch4 Methodology 撰寫
  ```
  4.1 數據準備與分割
      - JARVIS-DFT 數據集描述
      - 各物性的統計分布
      - 80/10/10 分割策略
      
  4.2 E(3)-equivariant multi-task GNN 架構
      - 完整架構圖
      - Species embedding → interaction blocks → pooling → heads
      - Bessel radial basis
      - FullyConnectedTensorProduct
      
  4.3 Physics-constrained loss function
      - 正定性約束：K≥0, G≥0
      - 介電約束：ε≥1
      - Born stability
      - Voigt-Reuss bounds
      
  4.4 Adaptive task weighting
      - Uncertainty (Kendall)
      - GradNorm (Chen)
      - PCGrad (Yu)
      
  4.5 Training protocol
      - 優化器、學習率、scheduler
      - 梯度裁剪
      - Early stopping
      - 超參數搜索
  ```

**週六 9/6 – 週日 9/7**
- [ ] 修改 Ch3-4 初稿
- [ ] 整合指導教授對 Ch1-2 的回饋
- [ ] 休息

### 第 30 週

**週一 9/8 – 週三 9/10**
- [ ] 修改 Ch3-4
- [ ] 補充遺漏的公式推導
- [ ] 製作 Ch4 的模型架構圖（高品質矢量圖）

**週四 9/11 – 週五 9/12**
- [ ] 交 Ch3-4 給指導教授
- [ ] 開始準備 Ch5

**週六 9/13 – 週日 9/14**
- [ ] 等待回饋
- [ ] 休息

---

## 第 31-33 週（9/15 – 10/5）：Ch5-6 Results

### 第 31 週

**週一 9/15** ← **里程碑 ⑧ Ch3-4 初稿交出**

**週一 9/15 – 週五 9/19**
- [ ] Ch5 Results I: Scalar Property Prediction
  ```
  5.1 Baseline 驗證 (Eg, Ef)
      - CGCNN results 表格
      - 與文獻對比
      - 每個物性的 parity plot
      
  5.2 Multi-task vs single-task
      - 4×4 比較表格
      - improvement ratio
      
  5.3 Task transfer 分析
      - 任務配對消融矩陣
      - positive/negative transfer 識別
      - 梯度對齊分析
      - 物理解釋
      
  5.4 Physics constraint 效果
      - 有/無約束的比較
      - 非物理預測的比例（K<0, ε<1）
      - Born stability violation rate
  ```
- [ ] 每節 1000-1500 字 + 圖表
- [ ] 重點：每個聲明（claim）都要有數據支持

### 第 32 週

**週一 9/22 – 週五 9/26**
- [ ] Ch6 Results II: Tensor Property Prediction
  ```
  6.1 彈性張量 Cij 預測
      - Component-wise MAE 表格
      - Parity plot (predicted vs DFT)
      - Frobenius error 分布
      
  6.2 介電張量 εij 預測
      - 同上格式
      
  6.3 壓電張量 eijk 預測（如果有做）
      
  6.4 Symmetry-aware 評估
      - 不同晶系的預測精度
      - 對稱性違反程度
      
  6.5 Derived scalar 精度
      - K from Cij vs K direct prediction
      - 哪個更準？為什麼？
  ```

### 第 33 週

**週一 9/29 – 週五 10/3**
- [ ] 修改 Ch5-6
- [ ] 整合指導教授回饋
- [ ] 確保所有圖表品質

**10/1** ← **里程碑 ⑨ Ch5-6 初稿完成**

---

## 第 34-36 週（10/6 – 10/26）：Ch7-8 Analysis + Application

### 第 34 週

**週一 10/6 – 週五 10/10**
- [ ] Ch7 Interpretability and Materials Property Space
  ```
  7.1 GNNExplainer 分析
      - 每個物性的 top-10 重要元素
      - 不同物性的重要子圖是否重疊？
      - Case study：3 個典型材料
      
  7.2 Latent space 分析
      - t-SNE/UMAP 可視化
      - 聚類與材料家族的對應
      - latent space 中的物性邊界
      
  7.3 Property-property 在 latent space 的相關性
      - 哪些維度與哪些物性相關？
      - 主成分分析 (PCA)
      
  7.4 多任務遷移的物理解釋
      - Penn model: ε ∝ 1/Eg² → 正遷移
      - 鍵強度：K ↔ Ef → 正遷移
      - 金屬 vs 半導體：K ↔ Eg → 可能負遷移
  ```

### 第 35 週

**週一 10/13 – 週五 10/17**
- [ ] Ch8 Application: Multi-Objective Materials Discovery
  ```
  8.1 Uncertainty-driven active learning
      - 3 種策略的 learning curve 比較
      - uncertainty 策略的加速因子
      
  8.2 Multi-objective Pareto 篩選
      - 目標：高 K + 高 ε + 低 Eg 的壓電材料
      - Pareto front 可視化
      - Top 50 候選清單
      
  8.3 Case study：高介電壓電材料
      - 選 3-5 個最有前景的候選
      - 分析其結構特徵
      
  8.4 DFT 驗算（如有條件）
      - 或用 MACE 驗算
      - 命中率分析
  ```

### 第 36 週

**週一 10/20 – 週五 10/24**
- [ ] 修改 Ch7-8
- [ ] 整合所有回饋
- [ ] 交給指導教授

**週六 10/25 – 週日 10/26**
- [ ] 休息

---

## 第 37-38 週（10/27 – 11/9）：Ch9 Conclusion + 全文修改

### 第 37 週

**週一 10/27 – 週三 10/29**
- [ ] Ch9 Conclusion and Future Work
  ```
  9.1 主要發現總結（5 個核心結論）
  9.2 局限性與不足
  9.3 未來工作方向
      - 更大模型（MACE-style）
      - 更多物性（光學、磁性）
      - 生成模型（材料逆設計）
      - 實驗驗證
  ```

**週四 10/30 – 週五 10/31**
- [ ] 撰寫 Abstract（中文 + 英文各 500 字）
- [ ] 撰寫致謝

### 第 38 週

**週一 11/3 – 週五 11/7**
- [ ] 全文通讀修改（第一輪）
  - 檢查邏輯一致性
  - 檢查術語統一
  - 檢查公式編號
  - 檢查圖表引用
  - 檢查參考文獻格式

**週六 11/8 – 週日 11/9**
- [ ] 全文通讀修改（第二輪）
- [ ] 交給指導教授最終審閱

**11/1** ← **里程碑 ⑩ 論文修改完成**

---

## 第 39-42 週（11/10 – 12/7）：答辯準備

### 第 39 週（11/10 – 11/16）
- [ ] 製作答辯投影片框架（40-50 頁）
  ```
  1-5:   Title + Outline                    (5 min)
  6-15:  Background + Literature             (10 min)
  16-25: Methodology                         (10 min)
  26-35: Results (Scalar + Tensor)           (10 min)
  36-42: Analysis + Application              (7 min)
  43-45: Conclusion                          (3 min)
  46-50: Backup slides (for Q&A)             
  ```
  Total: 45 分鐘 presentation

### 第 40 週（11/17 – 11/23）
- [ ] 完成投影片設計
- [ ] 準備常見問題的回答
  ```
  Q1: 為什麼選等變 GNN 而不是 Transformer？
  Q2: 你的模型和 MACE 有什麼區別？
  Q3: 物理約束真的有用嗎？消融實驗？
  Q4: 數據量不足怎麼辦？
  Q5: 張量預測的精度足夠實際應用嗎？
  Q6: active learning 的 oracle 在實際場景是什麼？
  Q7: 為何某些物性之間有負遷移？
  Q8: 你的方法在其他數據集上能泛化嗎？
  ```

### 第 41 週（11/24 – 11/30）
- [ ] 模擬答辯（第一次）
  - 自己計時 practice
  - 錄影回看
  - 修改投影片
- [ ] 找同學做練習答辯
  - 請他們問困難問題
  - 練習即時回答

### 第 42 週（12/1 – 12/7）
- [ ] 模擬答辯（第二次）
- [ ] 最終投影片修改
- [ ] 確保所有技術 demo 可以順利運行

**12/1** ← **里程碑 ⑪ 答辯準備完成**

---

## 第 43-44 週（12/8 – 12/21）：緩衝 + 投稿準備

### 如果進度超前
- [ ] 將論文改寫為期刊論文投稿版（英文）
  - 目標期刊：npj Computational Materials / Chemistry of Materials / ACS Applied Materials
- [ ] 準備 supplementary information
- [ ] 撰寫 cover letter

### 如果進度落後
- [ ] 優先完成論文主體
- [ ] 精簡實驗範圍
- [ ] 確保核心結果（Phase 1-2）完整且可靠

---

> **階段 D 最終驗收**：
> - ✅ 論文全文完成（9 章 + 附錄）
> - ✅ 圖表品質達到出版水準
> - ✅ 投影片 45-50 頁
> - ✅ 模擬答辯 × 2
> - ✅ 指導教授審核通過
> - ✅ 期刊投稿準備（optional）
