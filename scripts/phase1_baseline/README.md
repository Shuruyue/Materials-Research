# Phase 1 — Baseline Training (CGCNN)

> Validating the data pipeline and establishing performance baselines with CGCNN.

## What is Phase 1?

Phase 1 focuses on training a standard Crystal Graph Convolutional Neural Network (CGCNN) on the JARVIS-DFT dataset. We use a **tiered training strategy** to balance development speed and final accuracy:

1.  **Lite (Debug)** — Fast validation of the entire pipeline.
2.  **Std (Dev)** — Standard development and hyperparameter tuning.
3.  **Pro (Production)** — High-precision training for final benchmarking.

## Delivery Mode (Recommended for Team Ops)

Use the unified launcher and operation manual:

```bash
python scripts/phase1_baseline/run_phase1.py --level std
```

- Full teammate guide: `scripts/phase1_baseline/OPERATION.md`
- Includes 5 hyperparameter levels (`smoke/lite/std/pro/max`) and handoff checklist.

## Quick Start

### 1. Prepare Data

```bash
# Download JARVIS-DFT data (~76,000 materials)
python scripts/phase1_baseline/download_data.py
```

### 2. Train Model

Choose the appropriate tier for your task:

```bash
# Lite: Debug run (1 min)
python scripts/phase1_baseline/train_cgcnn_lite.py

# Std: Development run (45 min)
python scripts/phase1_baseline/train_cgcnn_std.py

# Pro: Production run (>12 hr)
# Use this for final PhD-level results
python scripts/phase1_baseline/train_cgcnn_pro.py
```

## Directory Structure

```
scripts/phase1_baseline/
├── download_data.py         # Data downloader & stats
├── train_cgcnn_lite.py      # Lite tier (10 epochs, 1k samples)
├── train_cgcnn_std.py       # Std tier (300 epochs, full data)
├── train_cgcnn_pro.py       # Pro tier (2000 epochs, full optimizations)
└── README.md                   # This file
```

## Usage Details

All training scripts support the following common arguments:

- `--property`: Target property (default: `formation_energy`)
  - Supported: `formation_energy`, `band_gap`, `bulk_modulus`, `shear_modulus`
- `--device`: Compute device
  - `cuda` (default if available) or `cpu`
- `--resume`: Resume training from last checkpoint (Std/Pro only)

```bash
# Example: Resume interrupted Pro training
python scripts/phase1_baseline/train_cgcnn_pro.py --resume
```
##  Hardware & Estimated Performance

**Reference System:**
- **GPU:** NVIDIA GeForce RTX 3060
- **CPU:** 16 Logical Processors (10 Cores)
- **RAM:** Optimal for `pin_memory=True`

**Estimated Training Times (`num_workers=0`):**

| Tier | Script | Est. Time | Device |
| :--- | :--- | :--- | :--- |
| **Lite** | `train_cgcnn_lite.py` | ~1 min | GPU (Fast) |
| **Std** | `train_cgcnn_std.py` | ~15-20 mins | GPU (Stable) |
| **Pro** | `train_cgcnn_pro.py` | ~45-60 mins | GPU (Reliable) |


> **Note:** Actual times may vary based on system load. We use `num_workers=0` to ensure stability on Windows.

## 4. Inference & Analysis

Run the provided inference script to make predictions with your trained model:

```bash
# 1. Verification (Random Test Sample)
python scripts/phase1_baseline/inference_demo.py --test-random

# 2. Single File Prediction
python scripts/phase1_baseline/inference_demo.py --cif structure.cif

# 3. Batch Processing (Entire Folder)
python scripts/phase1_baseline/inference_demo.py --dir my_structures/ --output results.csv
```

Results are saved in `models/cgcnn_pro_formation_energy/`.
- `results.json`: Final test metrics
- `history.json`: Loss/MAE curves
- `best.pt`: PyTorch model weights

