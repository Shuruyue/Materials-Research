# YEAR1_AUDIT_PLAN (Public Redacted)

This public file keeps only the minimum audit-ready contract.
Detailed reviewer timelines, internal checkpoints, and submission logistics are maintained in private planning records.

## Audit Definition of Done (Public)
1. `validate-data` hard gates are executable and enforced.
2. `make-splits` outputs deterministic manifests and assignments.
3. Phase launchers/benchmark/full-project run mandatory preflight.
4. Run manifest v2 is complete and strict-valid.
5. Local replay path can run `validate -> split -> smoke` end-to-end.
6. Public artifacts are de-identified.

## Stage Checklist
- Stage A: Data correctness gates and trust scoring are stable.
- Stage B: Deterministic split governance is stable.
- Stage C: Preflight enforcement is stable across all required launchers.
- Stage D: Reproducibility manifests pass strict validation.
- Stage E: Public artifact redaction and privacy checks pass.

## Minimum Evidence Bundle
- `artifacts/validation_report.json`
- `artifacts/splits/split_manifest_*.json`
- `artifacts/splits/split_assignment_*.csv` (or `.json`)
- smoke run logs with preflight enabled
- strict manifest validation output
- limitations/failure-modes/domain-shift report block
- privacy/redaction scan output

## Confidentiality Rule
- Algorithms and implementation can be open.
- Personal identifiers, internal infrastructure details, and internal review plans stay private.
