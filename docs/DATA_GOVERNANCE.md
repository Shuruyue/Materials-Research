# DATA_GOVERNANCE

Data governance contract for ATLAS (Year 1 audit baseline).

## 1) Provenance Taxonomy
Each sample must include provenance metadata with explicit type and version.

### Allowed provenance_type values
- `dft_primary`
- `experimental`
- `db_import`
- `literature`
- `synthetic`

### Required provenance fields
- `provenance_type`
- `source_key`
- `source_version`
- `source_id`

Rule: missing provenance (`provenance_type`) is a hard failure.

## 2) Trust Score Rubric (0-100)
Trust score is continuous and used as sample weight (`weight = trust_score / 100`) when enabled.

### Scoring Components
- Provenance type: up to 30
- Metadata completeness: up to 20
- Schema validity: up to 20
- Plausibility against distribution: up to 15
- Cross-source consistency: up to 15

### Trust Tier Thresholds
- `raw`: 0-39
- `curated`: 40-69
- `benchmark`: 70-100

### Data lifecycle thresholds
- raw -> curated: trust >= 40 and all hard gates pass.
- curated -> benchmark-grade: trust >= 70, complete provenance, clean split governance evidence.

## 3) Validation Gates
Run `validate-data` before training/benchmark.

### Hard-fail gates
- schema violations = 0
- provenance missing = 0
- leakage = 0
- unit violations = 0

### Warning gates (default warning, strict mode can fail)
- duplicates
- outliers
- drift

### Deletion vs Downweight Policy
- Keep questionable points by default and downweight with trust score.
- Delete only extreme outliers with explicit audit record.
- Never silently delete samples.

## 4) How to Run Validation
```bash
validate-data --output artifacts/validation_report.json
validate-data --units-spec configs/units_spec.json --baseline-report artifacts/validation_report_prev.json
validate-data --strict --output artifacts/validation_report_strict.json
```

## 5) Split Governance
Run `make-splits` to generate deterministic IID and OOD splits.

### Split definitions
- IID: random deterministic split by seed.
- compositional OOD: grouped by chemical system.
- prototype OOD: grouped by structure prototype (spacegroup proxy).

### Mandatory outputs
- `split_manifest_<strategy>.json`
- `split_assignment_<strategy>.json` and `.csv` (when `--emit-assignment`)
- stable `split_hash` and `assignment_hash`

### How to Run Split Generation
```bash
make-splits --strategy all --seed 42 --emit-assignment --output-dir artifacts/splits
make-splits --strategy compositional --seed 42 --split-id comp_v1 --group-definition-version 1 --emit-assignment
make-splits --strategy prototype --seed 42 --emit-assignment
```

## 6) OOD Reporting Requirement
Every experiment report must include IID and OOD separately.

### Mandatory table template
- metric name
- IID value (+CI)
- compositional OOD value (+CI)
- prototype OOD value (+CI)
- sample count per split
- run_id / split_id / dataset snapshot id

## 7) Privacy and Visibility
- Internal artifacts: full metadata allowed.
- Public artifacts: must be de-identified.
- Public manifests must not expose personal host/user/path/pid/internal network details.

## 8) Audit Artifacts
Minimum required artifacts for review:
- validation report JSON + Markdown summary
- split manifests and assignment files
- run manifest strict validation report
- limitations/failure modes/domain-shift risk block
