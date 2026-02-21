# Phase 3 Operation Manual (Potentials + Specialist Models)

Phase 3 has two training tracks:

- `mace`: interatomic potential learning.
- `equivariant`: single-task high-precision specialist model.

Recommended entry point: `scripts/phase3_potentials/run_phase3.py`.

## 1. What This Phase Does

- Potential track (`mace`):
  - Learns neural interatomic potential for energy (and optionally force/stress) prediction.
  - Supports downstream relaxation via `run_relaxation.py`.
- Specialist track (`equivariant`):
  - High-precision single-property model (`scripts/phase3_singletask/train_singletask_pro.py`).
  - Supports fine-tuning from Phase 2 checkpoints.

## 2. Hyperparameter Levels (5 Levels)

### MACE Track

| Level | Backend Script | Default Main Hyperparameters | Use Case |
| :--- | :--- | :--- | :--- |
| `smoke` | `train_mace.py` | `epochs=20`, `batch_size=16`, `lr=1e-3`, `r_max=4.5` | Sanity check |
| `lite` | `train_mace.py` | `epochs=100`, `batch_size=16`, `lr=5e-4`, `r_max=5.0` | Fast debug |
| `std` | `train_mace.py` | `epochs=300`, `batch_size=32`, `lr=3e-4`, `r_max=5.0` | Development |
| `pro` | `train_mace.py` | `epochs=600`, `batch_size=32`, `lr=2e-4`, `r_max=5.5` | Production |
| `max` | `train_mace.py` | `epochs=1000`, `batch_size=48`, `lr=1e-4`, `r_max=6.0` | Maximum precision |

### Equivariant Specialist Track

| Level | Backend Script | Default Main Hyperparameters | Use Case |
| :--- | :--- | :--- | :--- |
| `smoke` | `phase3_singletask/train_singletask_pro.py` | `epochs=50`, `batch_size=8`, `acc_steps=1`, `lr=5e-4`, `max_samples=2000` | Fast check |
| `lite` | `phase3_singletask/train_singletask_pro.py` | `epochs=300`, `batch_size=12`, `acc_steps=2`, `lr=3e-4`, `max_samples=12000` | Light tuning |
| `std` | `phase3_singletask/train_singletask_pro.py` | `epochs=800`, `batch_size=16`, `acc_steps=4`, `lr=2e-4` | Development |
| `pro` | `phase3_singletask/train_singletask_pro.py` | Script defaults (`epochs=1500`, etc.) | Main training |
| `max` | `phase3_singletask/train_singletask_pro.py` | `epochs=2200`, `acc_steps=6`, `lr=1.5e-4`, `--use-swa` | Highest precision |

## 3. Competition Profile (Independent Mode)

Competition mode is independent from levels and emphasizes score/time efficiency.

### MACE competition profile

- backend: `train_mace.py`
- defaults: `epochs=350`, `batch_size=32`, `lr=2.5e-4`, `r_max=5.5`

```bash
python scripts/phase3_potentials/run_phase3.py --algorithm mace --competition --prepare-mace-data
```

### Equivariant specialist competition profile

- backend: `phase3_singletask/train_singletask_pro.py`
- defaults: `epochs=900`, `batch_size=16`, `acc_steps=3`, `lr=2e-4`, `ema_decay=0.999`, `outlier_sigma=7.5`

```bash
python scripts/phase3_potentials/run_phase3.py --algorithm equivariant --competition --property band_gap
```

## 4. Standard Commands (Recommended)

```bash
# 3A) MACE with auto data preparation
python scripts/phase3_potentials/run_phase3.py --algorithm mace --level std --prepare-mace-data

# MACE custom element set and force/stress mode
python scripts/phase3_potentials/run_phase3.py --algorithm mace --level pro --prepare-mace-data --elements Bi Se Te --with-forces

# 3B) Specialist model (single property)
python scripts/phase3_potentials/run_phase3.py --algorithm equivariant --level pro --property band_gap

# Specialist fine-tuning from Phase 2
python scripts/phase3_potentials/run_phase3.py --algorithm equivariant --level pro --property formation_energy --finetune-from models/multitask_pro_e3nn/run_xxx/best.pt

# Competition mode
python scripts/phase3_potentials/run_phase3.py --algorithm equivariant --competition --property formation_energy
```

Common overrides:

```bash
python scripts/phase3_potentials/run_phase3.py --algorithm mace --level std --epochs 450 --lr 0.0002
python scripts/phase3_potentials/run_phase3.py --algorithm equivariant --level std --epochs 1000 --batch-size 16 --acc-steps 5
python scripts/phase3_potentials/run_phase3.py --algorithm mace --competition --epochs 420 --with-forces
```

## 5. Structure Relaxation (Post-Training)

```bash
python scripts/phase3_potentials/run_relaxation.py --structure data/raw/target.cif --model models/mace/best.model
```

Foundation model mode example:

```bash
python scripts/phase3_potentials/run_relaxation.py --structure data/raw/target.cif --model medium
```

## 6. Expected Outputs

- MACE outputs: `models/mace/`
- Specialist outputs: `models/specialist_<property>/`
- Optional analysis outputs: `analysis/` under model folders.

## 7. Team Handoff Checklist

- Record algorithm (`mace` / `equivariant`) and mode (`level` or `competition`).
- If MACE: record element set and whether `--with-forces` is enabled.
- If specialist: record target property and fine-tuning source checkpoint.
- Save final metrics JSON and model path.
