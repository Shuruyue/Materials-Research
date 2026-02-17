# Phase 2: E(3)-Equivariant GNN 分析報告

## 1. 核心概念：什麼是 E(3)-Equivariance？

**E(3)** 指的是 **Euclidean Group in 3D**，包含：
1.  **平移 (Translation)**：晶體移動位置，性質不變。
2.  **旋轉 (Rotation)**：晶體旋轉，性質不變（標量）或跟著旋轉（矢量/張量）。
3.  **反演 (Inversion)**：中心對稱操作。

**Phase 2 模型** (類似 NequIP, MACE, e3nn) 的核心在於：它不只知道原子間的「距離」，還知道它們的相對「方向」。

---

## 2. 優缺點分析

### ✅ 優點 (Pros)

1.  **高數據效率 (Data Efficiency)**
    *   因為模型「天生」懂幾何對稱性，它不需要透過大量數據增強 (Data Augmentation) 來學習旋轉不變性。
    *   **實例**：在 100-1000 筆數據的小樣本下，Equivariant GNN 遠勝傳統 GNN。
2.  **物理一致性 (Physical Consistency)**
    *   預測力場 (Force field) 或張量 (Tensor) 時，保證旋轉後的輸出與輸入旋轉一致。
    *   這對於預測 **彈性模量 (Elastic Tensor)** 或 **壓電係數** 至關重要。
3.  **辨識同分異構體 (Handling Isomers)**
    *   傳統 GNN (如 CGCNN) 只看距離直方圖，有時無法區分兩種距離相同但角度不同的結構（雖然在晶體中較少見，但在分子中很常見）。
    *   Equivariant GNN 能分辨這些細微的幾何差異。
4.  **解決「多體效應」(Many-body effects)**
    *   透過張量積 (Tensor Product)，模型可以自然地學習 3-body, 4-body 甚至更高階的原子間相互作用。

### ❌ 缺點 (Cons) & 常見問題

1.  **計算成本極高 (Computational Cost)**
    *   **慢**：涉及大量的球諧函數 (Spherical Harmonics) 計算和 Clebsch-Gordan 係數的張量積。
    *   **Phase 1 vs Phase 2**：訓練時間可能差 10-100 倍（如你所見，2.5s vs 257s）。
2.  **內存消耗大 (Memory Usage)**
    *   高階特徵 (High-L features, e.g., L=2, L=3) 的維度指數增長，容易 OOM (Out of Memory)。
3.  **數值穩定性 (Numerical Stability)**
    *   深層網絡中，反覆進行張量積可能導致梯度消失或爆炸，需要精細的標準化 (Normalization) 和初始化。
4.  **超參數敏感 (Hyperparameter Sensitivity)**
    *   `max_ell` (最大角動量, L=1,2,3) 和 `irreps` (不可約表示) 的選擇對性能和速度影響巨大。需權衡精度與速度。

---

## 3. 模型比較 (Comparison)

| 模型 | 幾何特徵 | 旋轉處理 | 優勢 | 劣勢 | 典型應用 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CGCNN** (Phase 1) | 僅距離 (Distances) | 不變 (Invariant) | 快、簡單、魯棒 | 丟失角度信息，無法預測張量 | 快速篩選、標量預測 (Ef, Eg) |
| **SchNet** | 距離 + 濾波器 | 不變 (Invariant) | 比 CGCNN 稍準，連續濾波 | 仍無角度信息 | 分子動力學 (MD) 早期模型 |
| **MEGNet** | 距離 + 全局狀態 | 不變 + 增強 | 加入全局特徵 (State) | 依賴數據增強學習旋轉 | Materials Project 預設模型 |
| **ALIGNN** | 距離 + **鍵角** (Angles) | 不變 (Invariant) | 加入角度圖 (Line Graph)，精度極高 | 顯存大，只是 Invariant (非 Equivariant) | 目前 JARVIS 最強 Baseline |
| **Equivariant** (Phase 2) | **球諧函數** (Full Geometry) | **等變 (Equivariant)** | **數據效率最高**，物理意義最強 | **最慢**，訓練難度高 | **高精度勢函數、發現新材料** |

---

## 4. 為什麼我們需要 Phase 2？

雖然 Phase 1 (CGCNN) 很快，但在**材料發現**中，我們往往關注那些「反直覺」或「微妙」的結構。

*   **例子**：某些拓撲材料的性質取決於晶格的微小畸變 (Distortion)。
    *   **CGCNN** 可能覺得這兩個結構差不多（距離沒變多少）。
    *   **Equivariant GNN** 能敏銳捕捉到對稱性的破缺 (Symmetry Breaking)，從而準確預測性質。

## 5. 實戰建議

1.  **先用 Phase 1 掃描**：從 100,000 個候選材料中篩出 1,000 個。
2.  **再用 Phase 2 精煉**：對這 1,000 個進行高精度預測，找出最終 10 個實驗候選。
3.  **多任務學習 (Multi-Task)**：因為 Equivariant 特徵很強，讓它同時學 Ef, Eg, K, G 可以互相輔助（例如：學會了力的結構，對預測模量 K 很有幫助）。
