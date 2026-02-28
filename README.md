# ATLAS - Accelerated Topological Learning And Screening

AI-driven workflow for materials ML research, focused on inorganic, metal, and semiconductor systems.

## Current Status
- Core package, training pipeline, active learning loop, and benchmark runner are implemented.
- Reproducibility layer is available (`workflow_reproducible_graph` + run manifest snapshots).
- Data governance pipeline includes `validate-data` and `make-splits`.
- Mandatory preflight is enabled for Phase1/2/5, benchmark, and full-project launchers.
- Run manifest v2 (`run_manifest.json` + `run_manifest.yaml`) supports internal/public visibility.
- Unit/integration tests are split and runnable with deterministic defaults.

See:
- `docs/ROADMAP_ACADEMIC.md`
- `docs/ROADMAP_PROJECT.md`
- `docs/ARCHITECTURE_NAMING.md`

## Quick Start

### Environment
```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

# Optional profiles
# pip install -r requirements-dev.txt
# pip install -r requirements-benchmark.txt
# pip install -r requirements-full.txt

python scripts/dev_tools/check_env.py
```

### Tests
```bash
# Default: unit-focused test run
pytest

# Integration tests (optional dependencies required)
pytest -m integration
```

### How to Run (Recommended Workflow)
```bash
# Step 1: Validate data
validate-data --output artifacts/validation_report.json

# Step 2: Generate deterministic splits
make-splits --strategy all --seed 42 --emit-assignment --output-dir artifacts/splits

# Step 3: Train (Phase 1 baseline)
python scripts/phase1_baseline/run_phase1.py --level smoke

# Step 4: Train (Phase 2 multi-task)
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level smoke

# Step 5: Active learning (Phase 5)
python scripts/phase5_active_learning/run_phase5.py --level smoke

# Step 6: Benchmark
benchmark --list-tasks
```

### Reproduce (Phase 1/2/5 minimal path)
```bash
python scripts/training/run_full_project.py --phase phase1 --level smoke
python scripts/training/run_full_project.py --phase phase2 --level smoke
python scripts/training/run_full_project.py --phase phase5 --level smoke
```

### All Phase Launchers
```bash
python scripts/phase1_baseline/run_phase1.py --level std
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level std
python scripts/phase3_potentials/run_phase3.py --algorithm equivariant --level std
python scripts/phase4_topology/run_phase4.py --algorithm topognn --level std
python scripts/phase5_active_learning/run_phase5.py --level std
python scripts/phase6_analysis/run_phase6.py --level std
python scripts/phase8_integration/run_phase8.py --level std

# Full-project orchestrator
python scripts/training/run_full_project.py --phase all --level std

# Note: phase7 is intentionally reserved.
# Legacy specialist path (not in full-project orchestration):
# scripts/phase3_singletask/train_singletask_pro.py
```

## Project Layout
```text
atlas/                    # core package
scripts/                  # runnable workflows by phase
tests/unit/               # fast and deterministic tests
tests/integration/        # optional dependency / heavier tests
data/                     # runtime data
models/                   # local model artifacts (gitignored)
references/               # external research repos (gitignored by default)
```

## Documentation
- `docs/ROADMAP_ACADEMIC.md`: public academic execution contract.
- `docs/ROADMAP_PROJECT.md`: public engineering/release contract.
- `docs/YEAR1_AUDIT_PLAN.md`: public redacted audit checklist.
- `docs/DATA_GOVERNANCE.md`: provenance, trust scoring, and split governance.
- `docs/REPRODUCIBILITY.md`: run manifest and replay instructions.
- `docs/ARCHITECTURE_NAMING.md`: phase/package naming contract and legacy-path policy.

## Governance Checks
```bash
python scripts/dev_tools/generate_dataset_manifest.py --property-group priority7 --max-samples 1000
python scripts/dev_tools/validate_split_stability.py --property-group core4 --seeds 42,52,62 --max-samples 1000
python scripts/dev_tools/validate_run_manifests.py --strict
python scripts/dev_tools/check_phase_interface_contract.py --strict
python scripts/dev_tools/create_version_snapshot.py
```

## License
MIT License - see `LICENSE`.
