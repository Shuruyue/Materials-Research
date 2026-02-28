# ARCHITECTURE_NAMING

Public naming and classification contract for ATLAS repository structure.

## 1) Package and Directory Roles
- `atlas/`: core package code (models, data, training, benchmark, active learning).
- `scripts/phase*/`: runnable phase entrypoints.
- `scripts/training/run_full_project.py`: orchestrator for supported public phases.
- `scripts/dev_tools/`: governance, validation, and reproducibility utilities.
- `tests/unit/`: fast deterministic tests.
- `tests/integration/`: broader pipeline checks with optional/heavier dependencies.

## 2) Phase Naming Policy
- Public orchestrated phases are: `phase1`, `phase2`, `phase3`, `phase4`, `phase5`, `phase6`, `phase8`.
- `phase7` is intentionally reserved for future expansion and is currently non-runnable.
- Public orchestration order is defined in `scripts/training/run_full_project.py`.

## 3) Phase 3 Naming Clarification
- `scripts/phase3_potentials/` is the canonical Phase 3 pipeline used by the orchestrator.
- `scripts/phase3_singletask/` is a legacy/specialist path for focused experiments.
- Legacy/specialist paths are allowed, but they must be explicitly marked as non-orchestrated.

## 4) CLI Naming Policy
- User-facing CLIs should be short and verb-based: e.g., `validate-data`, `make-splits`, `benchmark`.
- New user-facing CLIs should be registered in `pyproject.toml`.
- Internal helper scripts in `scripts/dev_tools/` do not need global CLI registration.

## 5) Change Policy
- Year-1 audit period: prefer additive/non-breaking naming changes only.
- Any major rename/restructure should include:
  - compatibility note,
  - migration note,
  - docs update in README and this file.
