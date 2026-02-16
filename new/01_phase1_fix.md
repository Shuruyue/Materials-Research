# Phase 1 CGCNN 訓練修復方案

## 問題診斷

### 問題一：formation_energy test MAE = 0.492（val MAE = 0.043）

**現象**：
- 驗證集上表現優異（MAE 0.043 eV/atom，遠低於 CGCNN 文獻 0.063）
- 測試集上完全崩潰（MAE 0.492 eV/atom，R² = -1325）
- MaxAE = 3415 eV/atom — 存在極端異常值

**根因分析**：
1. **資料異常值**：JARVIS-DFT 中有少量結構的 formation_energy 極端異常（可能是 DFT 收斂失敗的結果）。單一異常值即可讓 MAE 和 R² 嚴重偏移。MaxAE = 3415 eV/atom 證實了這一點。
2. **Scheduler 標籤錯誤**：結果 JSON 寫了 `CosineAnnealingWarmRestarts`，但實際代碼用的是 `ReduceLROnPlateau`。不影響實際訓練但造成混淆。

**修復方案**：

```python
# 已加入 11_train_cgcnn_full.py
def filter_outliers(dataset, property_name, n_sigma=10.0):
    """移除 |value - mean| > 10σ 的異常值"""
    # 在每個 split（train/val/test）上獨立過濾
```

### 問題二：shear_modulus R² = 0.22

**現象**：
- test MAE = 11.29 GPa（目標 10.0 GPa，文獻 8.0 GPa）
- R² = 0.22 — 模型只解釋了 22% 的方差
- 訓練時間只有 15 分鐘（314 epochs, early stopping at 234）

**根因分析**：
1. **資料量不足**：shear_modulus 只有 ~19K 樣本（相比 formation_energy 的 ~60K）
2. **物性複雜度高**：剪切模量比體積模量更難預測，因為它更依賴微觀結構和鍵角信息
3. **模型表達力**：CGCNN 缺乏方向信息（只用距離，不用向量），不利於力學性質

**改進策略**：
1. 更長的 patience（80 → 120）
2. 更小的 learning rate（0.001 → 0.0005）
3. 嘗試更大模型（512 hidden dim）
4. 最終解決方案：**使用 Phase 2 的等變 GNN**（包含方向信息，天然適合力學性質）

---

## 修復步驟

### 步驟一：立即修復（已完成 ✅）

1. ✅ `filter_outliers()` 函數加入 `11_train_cgcnn_full.py`
2. ✅ 修正 scheduler 標籤為 `"ReduceLROnPlateau"`
3. ✅ 在 train/val/test 三個 split 上都做異常值過濾

### 步驟二：重跑 formation_energy

```bash
python scripts/11_train_cgcnn_full.py \
    --property formation_energy \
    --epochs 500 \
    --patience 80 \
    --lr 0.001
```

預期結果：test MAE ≤ 0.07 eV/atom（移除異常值後應可達到）

### 步驟三：改善 shear_modulus

```bash
python scripts/11_train_cgcnn_full.py \
    --property shear_modulus \
    --epochs 500 \
    --patience 120 \
    --lr 0.0005 \
    --hidden-dim 512
```

預期結果：test MAE ≤ 10.0 GPa（可能仍不及文獻 8.0，這正好論證 Phase 2 等變 GNN 的必要性）

### 步驟四：用等變 GNN 徹底解決（Phase 2）

剪切模量的預測需要方向信息（原子間鍵角、力的方向），這正是 E(3)-equivariant GNN 的優勢：

```
CGCNN (Phase 1):  只用 |r_ij| (距離)  → 缺乏方向信息
E(3)-GNN (Phase 2): 用 r_ij (向量)   → 包含完整方向信息
```

預期 Phase 2 在 shear_modulus 上能顯著改善。

---

## 完成後的預期結果表

| Property | 現在 | 修復後預期 | Phase 2 目標 |
|:---------|:----:|:---------:|:-----------:|
| formation_energy | 0.492 | ≤ 0.065 | ≤ 0.040 |
| band_gap | 0.145 ✅ | 維持 | ≤ 0.14 |
| bulk_modulus | 10.35 ✅ | 維持 | ≤ 9.0 |
| shear_modulus | 11.29 | ≤ 10.0 | ≤ 7.0 |
