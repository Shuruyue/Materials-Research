# Internal Handoff Checklist (Phase 1-4)

Purpose: internal collaboration handoff only.  
Owner split:
- Base code owner: core + phase scripts maintenance
- Training owner: run training, debug runtime issues, summarize models

---

## A. Run Metadata

- Handoff date:
- Code commit hash used:
- Runner name:
- Machine/GPU:
- Python version:
- Environment install command used:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/dev_tools/check_env.py
```

---

## B. Required Execution Commands

Notes:
- Choose either `level` mode or `competition` mode.
- Keep the exact command string in the execution record.

### Phase 1 (CGCNN)

Standard:
```bash
python scripts/phase1_baseline/download_data.py
python scripts/phase1_baseline/run_phase1.py --level std --property formation_energy
```

Competition:
```bash
python scripts/phase1_baseline/run_phase1.py --competition --property formation_energy
```

Expected output path:
- `models/cgcnn_*`

---

### Phase 2 (Multitask)

Preprocess:
```bash
python scripts/phase2_multitask/process_data_phase2.py
```

Standard:
```bash
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level std
python scripts/phase2_multitask/run_phase2.py --algorithm cgcnn --level std
```

Competition:
```bash
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --competition
python scripts/phase2_multitask/run_phase2.py --algorithm cgcnn --competition
```

Expected output path:
- `models/multitask_std_e3nn/run_*`
- `models/multitask_pro_e3nn/run_*`
- `models/multitask_cgcnn_*`

---

### Phase 3 (Potentials + Specialist)

Standard:
```bash
python scripts/phase3_potentials/run_phase3.py --algorithm mace --level std --prepare-mace-data
python scripts/phase3_potentials/run_phase3.py --algorithm equivariant --level pro --property band_gap
```

Competition:
```bash
python scripts/phase3_potentials/run_phase3.py --algorithm mace --competition --prepare-mace-data
python scripts/phase3_potentials/run_phase3.py --algorithm equivariant --competition --property band_gap
```

Expected output path:
- `models/mace`
- `models/specialist_*`

---

### Phase 4 (Topology)

Standard:
```bash
python scripts/phase4_topology/init_topo_db.py
python scripts/phase4_topology/run_phase4.py --algorithm topognn --level std
python scripts/phase4_topology/run_phase4.py --algorithm rf --level std
```

Competition:
```bash
python scripts/phase4_topology/run_phase4.py --algorithm topognn --competition
python scripts/phase4_topology/run_phase4.py --algorithm rf --competition
```

Expected output path:
- `models/topo_classifier`
- `models/topo_classifier_rf`

---

## C. Acceptance Fields (Fill Per Run)

Use one row per training run.

| Run ID | Phase | Algorithm | Mode (`level/competition`) | Command | Exit Code | Output Dir | Main Artifact Exists (`best.pt`/`training_info.json`) | Key Metric(s) | Pass/Fail | Notes |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- | :---: | :--- | :---: | :--- |
| run-001 | phase1 | cgcnn | level |  |  |  |  |  |  |  |
| run-002 | phase2 | e3nn | competition |  |  |  |  |  |  |  |
| run-003 | phase2 | cgcnn | level |  |  |  |  |  |  |  |
| run-004 | phase3 | mace | competition |  |  |  |  |  |  |  |
| run-005 | phase3 | equivariant | level |  |  |  |  |  |  |  |
| run-006 | phase4 | topognn | level |  |  |  |  |  |  |  |
| run-007 | phase4 | rf | competition |  |  |  |  |  |  |  |

---

## D. Model Summary for Final Aggregation

Training owner should provide final model shortlist in this table.

| Phase | Algorithm | Selected Run ID | Model Path | Why Selected | Risks / Caveats |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phase1 | cgcnn |  |  |  |  |
| phase2 | e3nn |  |  |  |  |
| phase2 | cgcnn |  |  |  |  |
| phase3 | mace |  |  |  |  |
| phase3 | equivariant |  |  |  |  |
| phase4 | topognn |  |  |  |  |
| phase4 | rf |  |  |  |  |

---

## E. Minimum Handoff Criteria

- All required phase commands executed at least once (`standard` or `competition`).
- Every selected run has:
  - exact command string
  - output directory
  - key metric values
  - pass/fail decision
- Final shortlist table completed for model aggregation.

