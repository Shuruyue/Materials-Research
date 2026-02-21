# Phase 4 — Topology Classification

Phase 4 now supports two switchable algorithms:

- `TopoGNN` (structure graph model)
- `RF baseline` (composition-feature random forest)

## Main Entry (Recommended)

```bash
python scripts/phase4_topology/run_phase4.py --algorithm topognn --level std
python scripts/phase4_topology/run_phase4.py --algorithm rf --level std
```

## Full Operation Guide

See `scripts/phase4_topology/OPERATION.md` for:

- algorithm switching details
- 5-level hyperparameter presets
- teammate handoff checklist
- output artifact locations

