# REPRODUCIBILITY

Reproducibility contract for ATLAS.

## 1) Canonical Run Manifest Format
- Canonical source of truth: `run_manifest.json`.
- Human-readable mirror: `run_manifest.yaml` (generated from same payload).

### Run Manifest v2 required top-level keys
- `schema_version`
- `visibility` (`internal` or `public`)
- `created_at`
- `updated_at`
- `run_id`
- `runtime`
- `args`
- `dataset`
- `split`
- `environment_lock`
- `artifacts`
- `metrics`
- `seeds`
- `configs`

### Required reproducibility linkage
Each run must be traceable by:
- git commit SHA / branch / dirty state
- dataset snapshot id/fingerprint
- split id/hash
- seed values
- config references
- artifact paths

## 2) Visibility and Privacy
- `internal`: full runtime metadata.
- `public`: redacted runtime/path identifiers.
- Public manifests must redact hostname/cwd/pid.

## 3) Environment Locking Strategy
### Default lightweight lock
- install from `requirements.txt`.
- record lock file hash in manifest.

### Strict publication lock
- use `requirements-lock.txt` (hash-locked snapshot).
- set strict mode (`ATLAS_STRICT_LOCK=1` or strict lock option in tooling).
- strict lock metadata must be written into `environment_lock` block.

## 4) Phase Reproduce Instructions (RTX 3060)

### Step 1: setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Step 2: governance preflight
```bash
validate-data --output artifacts/validation_report.json
make-splits --strategy all --seed 42 --emit-assignment --output-dir artifacts/splits
```

### Step 3: phase reproduce path
```bash
python scripts/phase1_baseline/run_phase1.py --level smoke
python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level smoke
python scripts/phase5_active_learning/run_phase5.py --level smoke
```

### Step 4: full orchestrated reproduce
```bash
python scripts/training/run_full_project.py --phase phase1 --level smoke
python scripts/training/run_full_project.py --phase phase2 --level smoke
python scripts/training/run_full_project.py --phase phase5 --level smoke
```

## 5) Strict Manifest Validation
```bash
python scripts/dev_tools/validate_run_manifests.py --strict
# Optional audit-hard mode:
# python scripts/dev_tools/validate_run_manifests.py --strict --strict-completeness --strict-legacy
```

## 6) Reproducibility Bundle Requirements
A publication bundle must include:
- environment lock metadata
- dataset/split/run manifest chain
- IID and OOD metrics with CI
- ablation outputs and negative results
- limitations/failure mode/domain-shift artifacts
- public redacted artifact package for open release
