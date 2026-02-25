# Changelog

All notable changes to the ATLAS project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `ruff.toml` — Centralized lint configuration (PEP 8, import sorting, bugbear, complexity).
- `.github/workflows/ci.yml` — GitHub Actions CI with lint, unit test (Python 3.10–3.12 matrix), and integration test (allowed-fail).
- `CHANGELOG.md` — This file.
- `CONTRIBUTING.md` — Developer guide with setup, code quality, testing, and PR process.
- `docs/PERFORMANCE_ANALYSIS.md` — Data pipeline, training, and memory performance analysis.
- `tests/unit/training/test_trainer.py` — 15 tests for Trainer (init, epoch, validation, checkpointing, early stopping, fit).
- `tests/unit/training/test_losses.py` — 15 tests for PropertyLoss, EvidentialLoss, MultiTaskLoss.
- `tests/unit/models/test_uncertainty.py` — 12 tests for EvidentialRegression, EnsembleUQ, MCDropoutUQ.
- `tests/unit/topology/test_classifier.py` — 10 tests for TopoGNN (forward, predict_proba, save/load, gradient flow).

### Changed
- `setup.py` — Replaced 48-line duplicate dependency list with 2-line shim delegating to `pyproject.toml`.
- `atlas/topology/classifier.py` — Removed duplicate `CrystalGraphBuilder` (~100 lines), now imports from `atlas.models.graph_builder`. Renamed local `MessagePassingLayer` to `_TopoMessagePassingLayer` with documentation. Updated `TopoGNN` default `node_dim` from 69 to 91.
- `atlas/models/layers.py` — Added exception chaining (`from e`) to `ImportError` raise.
- `atlas/data/crystal_dataset.py` — Replaced `try/except/pass` with `contextlib.suppress`.
- `atlas/training/trainer.py` — Removed unused walrus operator variable `model_call`.
- `atlas/thermo/calphad.py` — Added exception chaining to `ImportError` raise.

### Fixed
- ~942 auto-fixed lint violations across the codebase (whitespace, unused imports, import sorting, PEP 585/604 annotations).
