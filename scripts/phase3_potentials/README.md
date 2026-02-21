# Phase 3 — Potentials & Specialist Training

Phase 3 combines:

- Interatomic potential learning (`MACE`)
- High-precision single-property specialist training (`EquivariantGNN`)

## Main Entry (Recommended)

```bash
python scripts/phase3_potentials/run_phase3.py --algorithm mace --level std --prepare-mace-data
python scripts/phase3_potentials/run_phase3.py --algorithm equivariant --level pro --property band_gap
python scripts/phase3_potentials/run_phase3.py --algorithm equivariant --competition --property band_gap
```

## Full Operation Guide

See `scripts/phase3_potentials/OPERATION.md` for:

- algorithm switching
- 5-level hyperparameter presets
- independent competition profile
- teammate handoff procedure
- output locations and post-training steps
