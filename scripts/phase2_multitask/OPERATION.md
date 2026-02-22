# Phase 2 Operation Manual (Multi-Task Learning)

This is the execution guide for teammates running multi-task material-property training.

## 1. What This Phase Does

- Objective: multi-property prediction with shared representation learning.
- Algorithms:
  - `e3nn`: equivariant GNN (main track).
  - `cgcnn`: CGCNN multi-task baseline (comparison track).
- Recommended entry point: `scripts/phase2_multitask/run_phase2.py`.

## 2. Data Preparation Notes

- Default training scripts (`train_multitask_*.py`) already build/cache datasets automatically via `CrystalPropertyDataset`.
- `process_data_phase2.py` is optional for bulk precompute workflows, but it is not a required step for standard training.

Optional command:

```bash
python scripts/phase2_multitask/process_data_phase2.py --workers 8
```

## 3. Hyperparameter Levels (5 Levels)

### E3NN Track

| Level | Backend Script | Default Main Hyperparameters | Use Case |
| :--- | :--- | :--- | :--- |
| `smoke` | `train_multitask_std.py` | `epochs=5`, `batch_size=8`, `lr=0.002` | End-to-end sanity test |
| `lite` | `train_multitask_lite.py` | Fixed tiny run (2 epochs, tiny subset) | Fast pipeline check |
| `std` | `train_multitask_std.py` | `epochs=100`, `batch_size=16`, `lr=0.001` | Dev workhorse |
| `pro` | `train_multitask_pro.py` | `epochs=500`, `batch_size=4`, `lr=5e-4` | Production |
| `max` | `train_multitask_pro.py` | `epochs=800`, `batch_size=4`, `lr=3e-4` | Highest precision |

### CGCNN Baseline Track

| Level | Backend Script | Default Main Hyperparameters | Use Case |
| :--- | :--- | :--- | :--- |
| `smoke` | `train_multitask_cgcnn.py` | `preset=small`, `epochs=5`, `max_samples=800` | Quick baseline check |
| `lite` | `train_multitask_cgcnn.py` | `preset=small`, `epochs=40`, `max_samples=3000` | Fast baseline |
| `std` | `train_multitask_cgcnn.py` | `preset=medium`, `epochs=200` | Standard baseline |
| `pro` | `train_multitask_cgcnn.py` | `preset=large`, `epochs=300` | Strong baseline |
| `max` | `train_multitask_cgcnn.py` | `preset=large`, `epochs=500`, `batch_size=160` | Maximum baseline |

## 4. Competition Profile (Independent Mode)

Competition mode is independent from the 5 levels and is tuned for balanced score vs runtime.

### E3NN competition profile

- backend: `train_multitask_pro.py`
- defaults: `epochs=260`, `batch_size=6`, `lr=4.5e-4`

```bash
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --competition
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --competition --all-properties
```

### CGCNN competition profile

- backend: `train_multitask_cgcnn.py`
- defaults: `preset=medium`, `epochs=280`, `batch_size=128`, `lr=9e-4`

```bash
python scripts/phase2_multitask/run_phase2.py --algorithm cgcnn --competition
```

## 5. Standard Commands (Recommended)

```bash
# E3NN standard development run
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level std

# E3NN production on all 9 properties
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level pro --all-properties

# Resume E3NN run
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level pro --resume

# CGCNN baseline run
python scripts/phase2_multitask/run_phase2.py --algorithm cgcnn --level std

# Competition mode
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --competition
```

Common overrides:

```bash
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level pro --epochs 700 --lr 0.0004
python scripts/phase2_multitask/run_phase2.py --algorithm cgcnn --level std --preset large --max-samples 12000
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --competition --epochs 320
```

## 6. Inference

```bash
python scripts/phase2_multitask/inference_multitask.py --cif data/your_structure.cif
python scripts/phase2_multitask/inference_multitask.py --dir data/structures --output phase2_preds.csv
```

## 7. Expected Outputs

- E3NN:
  - `models/multitask_std_e3nn/run_*/`
  - `models/multitask_pro_e3nn/run_*/`
- CGCNN baseline:
  - `models/multitask_cgcnn/run_*/`
- Key artifacts:
  - `best.pt`, `checkpoint.pt`, `history.json`, `results.json`, `run_manifest.json`.

## 8. Team Handoff Checklist

- Note algorithm (`e3nn` / `cgcnn`) and mode (`level` or `competition`).
- Record whether `--all-properties` is enabled.
- Save final validation/test MAE table by property.
- Keep run directory path in experiment registry.
