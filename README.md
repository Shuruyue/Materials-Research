# ATLAS - Accelerated Topological Learning And Screening

AI-driven workflow for materials ML research, focused on inorganic, metal, and semiconductor systems.

## Current Status (2026-02-28)
- Core package, training pipeline, active learning loop, and benchmark runner are implemented.
- Reproducibility layer is available (`workflow_reproducible_graph` + run manifest snapshots).
- **Data governance pipeline** with `validate-data` and `make-splits` CLI tools.
- Mandatory preflight enforcement for Phase1/2/5, benchmark, and full-project launchers.
- Run manifest v2 (`run_manifest.json` + `run_manifest.yaml`) with public/internal visibility.
- Unit/integration tests are separated and runnable with stable defaults.
- Environment setup is standardized with profile-based requirements files.

See:
- `docs/PROJECT_STATUS.md`
- `docs/TODO.md`

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
# Step 1: Validate data (schema, provenance, leakage, outliers)
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

# Full-project orchestrator (Phase 1/2/3/4/5/6/8)
python scripts/training/run_full_project.py --phase all --level std
```

## Project Layout
```text
atlas/                    # core package
scripts/                  # runnable workflows by phase
tests/unit/               # fast and deterministic tests
tests/integration/        # optional dependency / heavy tests
data/                     # runtime data
models/                   # local model artifacts (gitignored)
references/               # external research repos (gitignored by default)
```

## Documentation

### Governance & Roadmaps
- `docs/ROADMAP_ACADEMIC.md`: Academic publication track (A0–A10 phases) with micro-tasks and arXiv checklist.
- `docs/ROADMAP_PROJECT.md`: Open-source project track (B0–B10 phases) with CI/release strategy.
- `docs/YEAR1_AUDIT_PLAN.md`: Year 1 remaining 2-month audit closure plan (P0 must close).
- `docs/DATA_GOVERNANCE.md`: Data provenance taxonomy, trust scoring rubric (0–100), validation gates, split governance.
- `docs/REPRODUCIBILITY.md`: Run manifest schema, environment locking, end-to-end reproduce instructions.

### Project Status
- `docs/PROJECT_STATUS.md`: current implementation progress and known constraints.
- `docs/TODO.md`: prioritized TODO list for next development steps.
- `docs/PHASE1_TO_PHASE4_TRAINING_RUNBOOK.md`: teammate-ready Phase 1-4 execution guide.

### Research 
- `docs/research_preparation/`: dataset/paper/repo trackers and integration status.
- `docs/program_plan/`: dual-track execution plans and batch deliverables.
- `docs/adr/`: architecture decision records (template + accepted ADRs).

### Program Governance Checks
```bash
python scripts/dev_tools/generate_dataset_manifest.py --property-group priority7 --max-samples 1000
python scripts/dev_tools/validate_split_stability.py --property-group core4 --seeds 42,52,62 --max-samples 1000
python scripts/dev_tools/validate_run_manifests.py --strict
# Optional audit-hard mode:
# python scripts/dev_tools/validate_run_manifests.py --strict --strict-completeness --strict-legacy
python scripts/dev_tools/check_phase_interface_contract.py --strict
python scripts/dev_tools/create_version_snapshot.py
```

### Research Index Build
```bash
python scripts/dev_tools/build_research_index.py
```

### Research Batch Reading (5 at a time)
```bash
python scripts/dev_tools/read_research_in_batches.py --source repo --batch-size 5 --mark-reviewed
python scripts/dev_tools/read_research_in_batches.py --source paper --batch-size 5 --mark-reviewed
```

## License
MIT License - see `LICENSE`.
