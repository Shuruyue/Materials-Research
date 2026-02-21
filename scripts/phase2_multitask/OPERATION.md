# Phase 2 Operation Manual (Multi-Task Learning)

This is the execution guide for teammates running multi-task material-property training.

## 1. What This Phase Does

- Objective: multi-property prediction with shared representation learning.
- Algorithms:
  - `e3nn`: equivariant GNN (main track).
  - `cgcnn`: CGCNN multi-task baseline (comparison track).
- Recommended entry point: `scripts/phase2_multitask/run_phase2.py`.

## 2. Pre-step (Data Processing)

Run once before training:

```bash
python scripts/phase2_multitask/process_data_phase2.py
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

## 4. Standard Commands (Recommended)

```bash
# E3NN standard development run
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level std

# E3NN production on all 9 properties
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level pro --all-properties

# Resume E3NN run
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level pro --resume

# CGCNN baseline run
python scripts/phase2_multitask/run_phase2.py --algorithm cgcnn --level std
```

Common overrides:

```bash
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level pro --epochs 700 --lr 0.0004
python scripts/phase2_multitask/run_phase2.py --algorithm cgcnn --level std --preset large --max-samples 12000
```

## 5. Inference

```bash
python scripts/phase2_multitask/inference_multitask.py --cif data/your_structure.cif
python scripts/phase2_multitask/inference_multitask.py --dir data/structures --output phase2_preds.csv
```

## 6. Expected Outputs

- E3NN:
  - `models/multitask_std_e3nn/run_*/`
  - `models/multitask_pro_e3nn/run_*/`
- CGCNN baseline:
  - `models/multitask_cgcnn_*`
- Key artifacts:
  - `best.pt`, `checkpoint.pt`, `history.json`, metrics JSON/CSV logs.

## 7. Team Handoff Checklist

- Note algorithm (`e3nn` / `cgcnn`) and level.
- Record whether `--all-properties` is enabled.
- Save final validation/test MAE table by property.
- Keep run directory path in experiment registry.

