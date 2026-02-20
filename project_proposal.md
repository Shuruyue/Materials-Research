# 專案執行企劃書 (Project Execution Plan)

## 🎯 專案目標 Summary
建立一套 **「原子級材料與性質預測系統」**，從簡單的單任務學習 (Phase 1) 演進到高精度的多任務通用模型 (Phase 2)，最終實現針對特定性質的專精微調 (Phase 3)。

---

## 📅 執行流程與時間預估 (Timeline)

| 階段 (Phase) | 核心任務 (Mission) | 使用模型 | 使用腳本 (Script) | 預估時間 (Time) | 說明 (Note) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | **單一性質基準測試 (Baseline)** | CGCNN (Classic) | `12_train_cgcnn_pro.py` | **30~60 分鐘** | 快速驗證數據品質，確認 GPU 環境正常。 (已完成) |
| **Phase 2 (Std)** | **多任務模型開發 (Development)** | E3NN (Equivariant) | `21_train_multitask_std.py` | **1~2 小時** (100 Epochs) | 已達成 `En: 0.046`, `Gap: 0.17` 的驚人精度。用來調參最適合。 |
| **Phase 2 (Pro)** | **通用大模型訓練 (Foundation Model)** | E3NN (Large) | `22_train_multitask_pro.py` | **12~24 小時** (2000 Epochs) | **核心產出**。訓練一個能同時預測 9 種性質的「通才大腦」。 |
| **Phase 3** | **專精模型微調 (Specialist Fine-tuning)** | E3NN (Fine-tune) | `32_train_singletask_pro.py` | **20~30 分鐘** (Per Task) | 載入 Phase 2 的大腦，針對單一性質 (如 Formation Energy) 進行極致優化。 |
| **總計 (Total)** | **完整流程執行時間** | **All Models** | **Phase 1 -> 3** | **約 14 ~ 28 小時** | 包含資料處理與模型訓練 (視硬體與性質數量而定)。 |

---

## 🚀 詳細執行步驟 (Execution Steps)

### 1. Phase 1: 建立基準 (Baseline)
*   **目的**：確認「單一性質」最好能練到多準？作為後續比較的底線。
*   **指令**：
    ```bash
    python scripts/phase1_baseline/12_train_cgcnn_pro.py --property formation_energy
    ```
*   **產出**：`checkpoints/cgcnn_pro/best.pt`

### 2. Phase 2: 訓練通用大腦 (The Brain) ✨ **(目前階段)**
*   **目的**：訓練一個理解化學結構、能同時處理多種任務的 E3NN 模型。
*   **特點**：
    *   **資料處理**：第一次執行會花約 20 分鐘建立圖形 (`Building graphs`)，之後會自動讀取快取 (10秒)。
    *   **多任務學習**：同時學習 `Energy`, `Band Gap`, `Modulus` 等，讓模型學會更通用的原子表示法。
*   **指令 (標準版 - 快速驗證)**：
    ```bash
    python scripts/phase2_multitask/21_train_multitask_std.py
    ```
*   **指令 (專業版 - 生產環境)**：
    ```bash
    python scripts/phase2_multitask/22_train_multitask_pro.py --all-properties
    ```
*   **產出**：`models/multitask_pro_e3nn/best.pt` (這是我們最珍貴的資產)

### 3. Phase 3: 專精微調 (Specialist)
*   **目的**：利用 Phase 2 練好的「通用大腦」，針對某個特別難的性質 (例如: Formation Energy) 進行特訓。
*   **原理**：
    *   就像讓一個已經讀完大學 (Phase 2) 的學生，去攻讀特定領域的博士 (Phase 3)。
    *   可以使用更低的學習率 (`lr=1e-4`) 和更少的 Epochs。
*   **指令**：
    ```bash
    # 載入 Phase 2 的模型權重 (--finetune-from)
    python scripts/phase3_singletask/32_train_singletask_pro.py \
        --property formation_energy \
        --finetune-from models/multitask_pro_e3nn/run_2026xxxx/best.pt
    ```
*   **預期效果**：誤差 (MAE) 應該會比 Phase 2 更低，達到 SOTA 水準。

---

## 🛠️ 開發工具 (Dev Tools)
*   **檢查模型狀況**：
    ```bash
    python scripts/dev_tools/inspect_checkpoint.py
    ```
    (可以隨時查看訓練好的模型 `.pt` 檔裡面紀錄的 MAE 數據，不用重跑訓練)

---

## 📊 成功標準 (Success Metrics)
| 性質 | 目標 MAE | 目前 Phase 2 (Std) 成績 | 狀態 |
| :--- | :--- | :--- | :--- |
| **Formation Energy** | < 0.05 eV/atom | **0.0465** | 🌟 已達標 |
| **Band Gap** | < 0.30 eV | **0.1697** | 🚀 超越預期 |
| **Bulk Modulus** | < 20 GPa | **9.89** | 💪 穩健 |
| **Shear Modulus** | < 20 GPa | **8.68** | 💪 穩健 |

> **建議**：Phase 2 Std 的結果已經非常優秀，Phase 2 Pro (長訓練) 和 Phase 3 (微調) 有望挑戰世界紀錄！

---

## ⚡ 指令快查表 (Command Cheat Sheet)
方便您直接複製貼上：

**Phase 1 (Baseline)**
```bash
python scripts/phase1_baseline/12_train_cgcnn_pro.py --property formation_energy
```

**Phase 2 (The Brain)**
```bash
# 開發 (Dev) - 快速驗證 (100 Epochs)
python scripts/phase2_multitask/21_train_multitask_std.py

# 生產 (Pro) - 完整訓練 (2000 Epochs, SOTA)
python scripts/phase2_multitask/22_train_multitask_pro.py --all-properties
```

**Phase 3 (Specialist)**
```bash
# 專精特訓 (需指定 Phase 2 的 checkpoint 路徑)
python scripts/phase3_singletask/32_train_singletask_pro.py \
    --property formation_energy \
    --finetune-from models/multitask_pro_e3nn/run_LATEST/best.pt
```

**工具 (Tools)**
```bash
# 檢查模型訓練狀況
python scripts/dev_tools/inspect_checkpoint.py
```
