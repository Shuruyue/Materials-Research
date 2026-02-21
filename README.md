# ATLAS - Accelerated Topological Learning And Screening

AI-driven workflow for materials ML research, focused on inorganic, metal, and semiconductor systems.

## Current Status (2026-02-21)
- Core package, training pipeline, active learning loop, and benchmark runner are implemented.
- Reproducibility layer is available (`workflow_reproducible_graph` + run manifest snapshots).
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

### Common Pipeline Entrypoints
```bash
# Phase 1
python scripts/phase1_baseline/01_download_data.py
python scripts/phase1_baseline/10_train_cgcnn_lite.py

# Phase 2
python scripts/phase2_multitask/process_data_phase2.py
python scripts/phase2_multitask/20_train_multitask_lite.py

# Phase 5
python scripts/phase5_active_learning/06_run_discovery.py

# Benchmark CLI
benchmark --list-tasks
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
- `docs/PROJECT_STATUS.md`: current implementation progress and known constraints.
- `docs/TODO.md`: prioritized TODO list for next development steps.

## License
MIT License - see `LICENSE`.
