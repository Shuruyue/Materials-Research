
# Phase 2 — Multi-Task & Equivariant Learning

> **Goal**: Transition from single-property models (Phase 1) to a "Universal Material Brain" (Phase 2).

## The Grand Strategy (The Matrix)

We are unlocking a total of **29+ Models** across 3 dimensions of difficulty.

| Tier | Single-Task (9 Properties) | Multi-Task (All-in-One) |
| :--- | :--- | :--- |
| **Lite** | 9 Fast Debug Models | 1 Fast Debug Model |
| **Std** | 9 Tuned Models | 1 Tuned Model |
| **Pro** | **9 SOTA Models** (High Precision) | **1 Universal Model** (Holy Grail) |

### The 9 Discoverable Properties
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

### Step 0: The Foundation
| Script | Role |
| :--- | :--- |
| `process_data_phase2.py` | **Data Prep**. Pre-computes 3-body angles for ALL 29 models.<br>Updated to extract **all 9 properties**. |


### Step 1: Multi-Task Models (Main Track)
The primary "Universal Brain" models (E3NN) training on all properties.

| Script | Role |
| :--- | :--- |
| `20_train_multitask_lite.py` | **Lite (Debug)**. Fast check (2 epochs). |
| `21_train_multitask_std.py` | **Std (Dev)**. Balanced training. |
| `22_train_multitask_pro.py` | **Pro (SOTA)**. The "Holy Grail". `--all-properties` unlocked. |

### Step 2: Single-Task Specialists (Deep Dive)
High-precision models for specific properties.

| Script | Role |
| :--- | :--- |
| `32_train_singletask_pro.py` | **Single-Task Pro**. E.g., `python 32...py --property band_gap`. |

### Legacy / Baselines
| Script | Role |
| :--- | :--- |
| `12_train_multitask_cgcnn.py` | **Baseline (CGCNN)**. For comparison. |



## Quick Start (Phase 2)

Choose the appropriate tier for your task:

###  Multi-Task: The Universal Brain (Recommended)

```bash
# 1. LITE: Fast Debug (2 epochs, CPU friendly)
python scripts/phase2_multitask/20_train_multitask_lite.py

# 2. STD: Standard Dev (Resume + Outlier Check)
python scripts/phase2_multitask/21_train_multitask_std.py

# 3. PRO: Production SOTA (9 Properties)
# Add --all-properties to unlock full discovery mode
python scripts/phase2_multitask/22_train_multitask_pro.py --all-properties
```

### Single-Task: The Specialist (Deep Dive)

Train on a specific property. Replace `band_gap` with any of the **9 Supported Properties**:
`formation_energy`, `band_gap`, `band_gap_mbj`, `bulk_modulus`, `shear_modulus`, `dielectric`, `piezoelectric`, `spillage`, `ehull`

```bash
# Example: Train formation energy
python scripts/phase2_multitask/32_train_singletask_pro.py --property formation_energy

# Example: Train dielectric constant
python scripts/phase2_multitask/32_train_singletask_pro.py --property dielectric
```

##  Hardware & Estimated Performance

**Reference System:**
- **GPU:** NVIDIA GeForce RTX 3060
- **CPU:** 16 Logical Processors (10 Cores)
- **RAM:** Optimal for `pin_memory=True`

**Estimated Training Times (`num_workers=0`):**

| Tier | Script | Est. Time | Notes |
| :--- | :--- | :--- | :--- |
| **Lite** | `20_train_multitask_lite.py` | ~2-3 mins | Smoke Test |
| **Std** | `21_train_multitask_std.py` | ~1-2 hours | E3NN is compute-heavy |
| **Pro** | `22_train_multitask_pro.py` | ~4-8 hours | Deep E3NN (500 epochs) |
| **Single**| `32_train_singletask_pro.py`| ~2-4 hours | High Precision |

> **Note:** E3NN (Equivariant GNN) is mathematically complex. We use `num_workers=0` to ensure no deadlocks on Windows.
