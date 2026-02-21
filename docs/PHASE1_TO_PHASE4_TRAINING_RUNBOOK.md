# Phase 1-4 Training Runbook (Team Delivery)

This runbook is for teammate execution without reading long source code.

## 1. Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/dev_tools/check_env.py
```

## 2. Recommended Execution Order

```bash
# Phase 1
python scripts/phase1_baseline/download_data.py
python scripts/phase1_baseline/run_phase1.py --level std --property formation_energy

# Phase 2
python scripts/phase2_multitask/process_data_phase2.py
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level std

# Phase 3
python scripts/phase3_potentials/run_phase3.py --algorithm mace --level std --prepare-mace-data
python scripts/phase3_potentials/run_phase3.py --algorithm equivariant --level pro --property band_gap

# Phase 4
python scripts/phase4_topology/init_topo_db.py
python scripts/phase4_topology/run_phase4.py --algorithm topognn --level std
python scripts/phase4_topology/run_phase4.py --algorithm rf --level std
```

## 3. Phase-by-Phase Operation Manuals

- Phase 1: `scripts/phase1_baseline/OPERATION.md`
- Phase 2: `scripts/phase2_multitask/OPERATION.md`
- Phase 3: `scripts/phase3_potentials/OPERATION.md`
- Phase 4: `scripts/phase4_topology/OPERATION.md`

## 4. Unified Level Convention

All phases now align to 5 levels:

- `smoke`: end-to-end sanity check.
- `lite`: quick debug training.
- `std`: default development training.
- `pro`: production-level training.
- `max`: highest precision / longest training.

## 5. Required Handoff Record (per run)

For every run, capture:

- phase (`phase1` / `phase2` / `phase3` / `phase4`)
- algorithm
- level
- full command
- key overrides (epochs/lr/batch/property/etc.)
- output model directory
- final validation/test metrics
- notes on failures or anomalies

## 6. Output Locations (Default)

- Phase 1: `models/cgcnn_*`
- Phase 2: `models/multitask_*`
- Phase 3:
  - `models/mace`
  - `models/specialist_*`
- Phase 4:
  - `models/topo_classifier`
  - `models/topo_classifier_rf`

