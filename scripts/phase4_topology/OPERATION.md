# Phase 4 Operation Manual (Topological Classification)

Phase 4 now supports algorithm switching for practical training handoff:

- `topognn`: structure-graph GNN classifier.
- `rf`: random-forest baseline from composition features.

Recommended entry point: `scripts/phase4_topology/run_phase4.py`.

## 1. What This Phase Does

- Objective: classify materials into topological vs trivial classes.
- Label rule: `spillage > 0.5` => topological (1), else trivial (0).
- Data source: JARVIS DFT-3D with available spillage values.

## 2. Hyperparameter Levels (5 Levels)

### TopoGNN Track

| Level | Backend Script | Default Main Hyperparameters | Use Case |
| :--- | :--- | :--- | :--- |
| `smoke` | `train_topo_classifier.py` | `epochs=10`, `max=1000`, `batch_size=16`, `hidden=64` | Fast sanity test |
| `lite` | `train_topo_classifier.py` | `epochs=30`, `max=2500`, `batch_size=24`, `hidden=96` | Lightweight training |
| `std` | `train_topo_classifier.py` | `epochs=100`, `max=5000`, `batch_size=32`, `hidden=128` | Standard run |
| `pro` | `train_topo_classifier.py` | `epochs=180`, `max=8000`, `batch_size=48`, `hidden=160` | Production |
| `max` | `train_topo_classifier.py` | `epochs=260`, `max=12000`, `batch_size=64`, `hidden=192` | Maximum precision |

### RF Baseline Track

| Level | Backend Script | Default Main Hyperparameters | Use Case |
| :--- | :--- | :--- | :--- |
| `smoke` | `train_topo_classifier_rf.py` | `max_samples=1000`, `n_estimators=120`, `max_depth=12` | Fast baseline |
| `lite` | `train_topo_classifier_rf.py` | `max_samples=2500`, `n_estimators=300`, `max_depth=16` | Quick baseline |
| `std` | `train_topo_classifier_rf.py` | `max_samples=5000`, `n_estimators=600`, `max_depth=24` | Standard baseline |
| `pro` | `train_topo_classifier_rf.py` | `max_samples=8000`, `n_estimators=900`, `max_depth=28` | Strong baseline |
| `max` | `train_topo_classifier_rf.py` | `max_samples=12000`, `n_estimators=1200`, `max_depth=32` | Maximum baseline |

## 3. Competition Profile (Independent Mode)

Competition mode is independent from levels and targets accuracy/runtime balance.

### TopoGNN competition profile

- backend: `train_topo_classifier.py`
- defaults: `epochs=120`, `max=6000`, `batch_size=40`, `hidden=144`, `lr=9e-4`

```bash
python scripts/phase4_topology/run_phase4.py --algorithm topognn --competition
```

### RF competition profile

- backend: `train_topo_classifier_rf.py`
- defaults: `max_samples=7000`, `n_estimators=700`, `max_depth=22`, `min_samples_leaf=2`

```bash
python scripts/phase4_topology/run_phase4.py --algorithm rf --competition
```

## 4. Standard Commands (Recommended)

```bash
# Initialize known topological seed database (optional but recommended)
python scripts/phase4_topology/init_topo_db.py

# TopoGNN standard run
python scripts/phase4_topology/run_phase4.py --algorithm topognn --level std

# RF baseline run
python scripts/phase4_topology/run_phase4.py --algorithm rf --level std

# Highest-precision TopoGNN run
python scripts/phase4_topology/run_phase4.py --algorithm topognn --level max --max-samples 15000

# Competition mode
python scripts/phase4_topology/run_phase4.py --algorithm topognn --competition
```

Common overrides:

```bash
python scripts/phase4_topology/run_phase4.py --algorithm topognn --level std --epochs 140 --hidden 160
python scripts/phase4_topology/run_phase4.py --algorithm rf --level pro --n-estimators 1200 --max-depth 30
python scripts/phase4_topology/run_phase4.py --algorithm rf --competition --n-estimators 900
```

## 5. Expected Outputs

- TopoGNN model folder: `models/topo_classifier/`
- RF baseline folder: `models/topo_classifier_rf/`
- Both save metrics in `training_info.json`.

## 6. Team Handoff Checklist

- Record algorithm + mode used (`level` or `competition`).
- Save val/test metrics (accuracy, precision, recall, F1).
- Keep confusion matrix and model path in project tracking sheet.
