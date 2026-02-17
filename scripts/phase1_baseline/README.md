# Phase 1 — Baseline Training (CGCNN)

> Validating the data pipeline and establishing performance baselines with CGCNN.

## What is Phase 1?

Phase 1 focuses on training a standard Crystal Graph Convolutional Neural Network (CGCNN) on the JARVIS-DFT dataset. We use a **tiered training strategy** to balance development speed and final accuracy:

1.  **Lite (Debug)** — Fast validation of the entire pipeline.
2.  **Std (Dev)** — Standard development and hyperparameter tuning.
3.  **Pro (Production)** — High-precision training for final benchmarking.

## Quick Start

### 1. Prepare Data

```bash
# Download JARVIS-DFT data (~76,000 materials)
python scripts/phase1_baseline/01_download_data.py
```

### 2. Train Model

Choose the appropriate tier for your task:

```bash
# 🟢 Lite: Debug run (1 min)
python scripts/phase1_baseline/10_train_cgcnn_lite.py

# 🟡 Std: Development run (45 min)
python scripts/phase1_baseline/11_train_cgcnn_std.py

# 🔴 Pro: Production run (>12 hr)
# Use this for final PhD-level results
python scripts/phase1_baseline/12_train_cgcnn_pro.py
```

## Directory Structure

```
scripts/phase1_baseline/
├── 01_download_data.py         # Data downloader & stats
├── 10_train_cgcnn_lite.py      # Lite tier (10 epochs, 1k samples)
├── 11_train_cgcnn_std.py       # Std tier (300 epochs, full data)
├── 12_train_cgcnn_pro.py       # Pro tier (2000 epochs, full optimizations)
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
python scripts/phase1_baseline/12_train_cgcnn_pro.py --resume
```
