
# Phase 2 — Multi-Task & Equivariant Learning

> **Goal**: Transition from single-property models (Phase 1) to a "Universal Material Brain" (Phase 2).

## 🌌 The Grand Strategy (The Matrix)

We are unlocking a total of **29+ Models** across 3 dimensions of difficulty.

| Tier | Single-Task (9 Properties) | Multi-Task (All-in-One) |
| :--- | :--- | :--- |
| **Lite** | 9 Fast Debug Models | 1 Fast Debug Model |
| **Std** | 9 Tuned Models | 1 Tuned Model |
| **Pro** | **9 SOTA Models** (High Precision) | **1 Universal Model** (Holy Grail) |

### 💎 The 9 Discoverable Properties
Based on `jarvis-dft` data, we are now targeting:
1.  **Formation Energy** (Stability)
2.  **Opt Band Gap** (Optical properties)
3.  **MBJ Band Gap** (Accurate electronics)
4.  **Bulk Modulus** (Compressibility)
5.  **Shear Modulus** (Hardness)
6.  **Dielectric Constant** (Capacitors/Chips)
7.  **Piezoelectric Coeff** (Sensors)
8.  **Spillage** (Magnetics)
9.  **Ehull** (Synthesizability)

---

## Directory Structure

### 🛠️ Step 0: The Foundation
| Script | Role |
| :--- | :--- |
| `process_data_phase2.py` | **Data Prep**. Pre-computes 3-body angles for ALL 29 models.<br>Updated to extract **all 9 properties**. |

### 🟢 Step 1: Multi-Task Baseline (CGCNN)
| Script | Role |
| :--- | :--- |
| `12_train_multitask_cgcnn.py` | **Main Baseline**.<br>Shared CGCNN encoder + 9 regression heads. |
| `13_train_multitask_cgcnn_phase1.py` | [LEGACY] Phase 1 attempt. Deprecated. |

### 🚀 Step 2: Next-Gen Equivariant (E3NN)
| Script | Role |
| :--- | :--- |
| `20_train_equivariant.py` | **Single-Task Laser**.<br>The "Pro" script for the 27 Single-Task models. |
| `21_train_multitask.py` | **The Universal Brain**.<br>The "Pro" script for the Multi-Task model. |


## Quick Start (Phase 2)

### 1. Establish Baseline (Fast)
```bash
python scripts/phase2_multitask/12_train_multitask_cgcnn.py --preset medium
```

### 2. Train Next-Gen Model (Slow & Precise)
```bash
# Train formation energy with E(3) symmetry
python scripts/phase2_multitask/20_train_equivariant.py --property formation_energy
```
