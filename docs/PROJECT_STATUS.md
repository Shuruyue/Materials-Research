# ATLAS Project Status

Last updated: 2026-02-21

## Scope
ATLAS is a research-first platform for combining materials science and machine learning, currently oriented to:
- inorganic crystals
- metal systems
- semiconductor-related screening

## Implemented

### Core workflow
- Config-driven project settings (`atlas/config.py`)
- Data source and method registry (`atlas/data/source_registry.py`, `atlas/research/method_registry.py`)
- Reproducibility utility and runtime metadata capture (`atlas/utils/reproducibility.py`)

### Modeling and training
- Phase 1 baseline (CGCNN family scripts)
- Phase 2 multitask/equivariant branch
- Phase 1-4 unified launchers with 5-level profiles:
  - `scripts/phase1_baseline/run_phase1.py`
  - `scripts/phase2_multitask/run_phase2.py`
  - `scripts/phase3_potentials/run_phase3.py`
  - `scripts/phase4_topology/run_phase4.py`
- Phase 4 baseline algorithm switch support:
  - `TopoGNN` (graph-based)
  - `RandomForest` composition baseline (`train_topo_classifier_rf.py`)
- M3GNet-style modeling components

### Discovery loop
- Active learning controller with:
  - checkpoint resume
  - stage timing snapshots
  - workflow manifest integration
- Optional GP surrogate branch (`gp_active_learning`)

### Benchmarking
- Matbench runner supports:
  - fold-level metrics
  - aggregate metrics
  - JSON report artifacts
- Benchmark CLI available through `benchmark` script entrypoint.

### Test system
- Test suite is categorized into:
  - `tests/unit/*`
  - `tests/integration/*`
- Default `pytest` excludes integration tests.
- Integration tests skip cleanly when optional dependencies are unavailable.

### Environment/tooling
- Requirement profiles:
  - `requirements.txt`
  - `requirements-dev.txt`
  - `requirements-benchmark.txt`
  - `requirements-full.txt`
- Environment validation script:
  - `scripts/dev_tools/check_env.py`
- Team operation runbook:
  - `docs/PHASE1_TO_PHASE4_TRAINING_RUNBOOK.md`

## Known Constraints
- `torch-scatter` and `torch-sparse` may not have prebuilt wheels for all Python/Torch/CUDA combinations.
- Some integration paths rely on external repos or optional packages (`openmm`, `mepin`, `liflow`, alchemical stack).
- MACE ecosystem dependency compatibility (notably around `e3nn`) requires explicit profile management.

## Recommended Default Development Mode
1. Use `requirements.txt` + `pip install -e .`
2. Run `pytest` for daily development
3. Run `pytest -m integration` only when optional dependencies are prepared
