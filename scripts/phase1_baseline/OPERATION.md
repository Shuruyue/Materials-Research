# Phase 1 Operation Manual (CGCNN Baseline)

This document is the teammate-facing runbook for Phase 1 training and inference.

## 1. What This Phase Does

- Objective: single-property regression on crystal materials (`formation_energy` by default).
- Core algorithm: `CGCNN`.
- Entry point (recommended): `scripts/phase1_baseline/run_phase1.py`.

## 2. Prerequisites

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Download/check data:

```bash
python scripts/phase1_baseline/download_data.py
python scripts/phase1_baseline/download_data.py --stats
```

## 3. Hyperparameter Levels (5 Levels)

| Level | Backend Script | Default Main Hyperparameters | Use Case |
| :--- | :--- | :--- | :--- |
| `smoke` | `train_cgcnn_lite.py` | `epochs=2`, `max_samples=256`, `batch_size=16` | Pipeline sanity check |
| `lite` | `train_cgcnn_lite.py` | `epochs=10`, `max_samples=1000`, `batch_size=32`, `lr=0.005`, `hidden_dim=64`, `n_conv=2` | Fast debug |
| `std` | `train_cgcnn_std.py` | `epochs=300`, `batch_size=64`, `lr=0.001`, `hidden_dim=128`, `n_conv=3` | Development |
| `pro` | `train_cgcnn_pro.py` | `epochs=2000`, `batch_size=64`, `lr=0.001`, `hidden_dim=512`, `n_conv=5` | Main training |
| `max` | `train_cgcnn_pro.py` | `epochs=3000`, `batch_size=48`, `lr=7e-4`, `hidden_dim=768`, `n_conv=6` | Highest precision |

## 4. Competition Profile (Independent Mode)

Competition mode is not one of the 5 levels. It is an independent profile focused on strong score/time ratio:

- backend: `train_cgcnn_std.py`
- defaults: `epochs=450`, `batch_size=48`, `lr=8e-4`, `patience=80`, `hidden_dim=192`, `n_conv=4`, `max_samples=40000`

```bash
python scripts/phase1_baseline/run_phase1.py --competition --property formation_energy
```

## 5. Standard Commands (Recommended)

```bash
# Default dev run
python scripts/phase1_baseline/run_phase1.py --level std

# Change target property
python scripts/phase1_baseline/run_phase1.py --level std --property band_gap

# Resume (supported in std/pro/max)
python scripts/phase1_baseline/run_phase1.py --level pro --resume

# Disable outlier filter (supported in std/pro/max)
python scripts/phase1_baseline/run_phase1.py --level pro --no-filter

# Competition mode (independent from levels)
python scripts/phase1_baseline/run_phase1.py --competition --property band_gap
```

Available common overrides:

```bash
python scripts/phase1_baseline/run_phase1.py --level pro --epochs 1200 --batch-size 64 --lr 0.0008
python scripts/phase1_baseline/run_phase1.py --level std --hidden-dim 192 --n-conv 4
python scripts/phase1_baseline/run_phase1.py --competition --epochs 520 --lr 0.0007
```

## 6. Direct Script Mode (If Needed)

```bash
python scripts/phase1_baseline/train_cgcnn_lite.py --property formation_energy
python scripts/phase1_baseline/train_cgcnn_std.py --property formation_energy --resume
python scripts/phase1_baseline/train_cgcnn_pro.py --property formation_energy --resume
```

## 7. Inference

```bash
# Random sample check
python scripts/phase1_baseline/inference_demo.py --test-random

# Single CIF
python scripts/phase1_baseline/inference_demo.py --cif data/your_structure.cif

# Batch folder
python scripts/phase1_baseline/inference_demo.py --dir data/structures --output predictions.csv
```

## 8. Expected Outputs

- Model folder: `models/cgcnn_*`
- Key artifacts:
  - `best.pt`
  - `checkpoint.pt` (std/pro)
  - `results.json`
  - `history.json`

## 9. Team Handoff Checklist

- Confirm command and run mode used (`level` or `competition`).
- Record property and all CLI overrides.
- Save final metrics from `results.json`.
- Keep model directory name and run timestamp in experiment notes.
