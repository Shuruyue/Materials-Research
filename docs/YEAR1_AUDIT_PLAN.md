# YEAR1_AUDIT_PLAN

Audit-focused closeout plan for the remaining two months of Year 1.

## Timeline
- Window: 2026-03-01 to 2026-04-30.
- Goal: clear all P0 audit blockers before Year 2 starts.

## P0 Closure Targets
1. Hard gates are executable and enforced (`validate-data`).
2. OOD split governance is deterministic and reproducible (`make-splits`).
3. Preflight is mandatory for Phase 1/2/5, benchmark, full-project launcher.
4. Run manifest v2 is complete and strict-valid.
5. Local RTX 3060 replay path is available end-to-end.
6. Public artifact path is de-identified.

## Weekly Milestones
- W1: schema/provenance hard-fail correctness.
- W2: units and drift baseline checks.
- W3: split manifest v2 and assignment hash stability.
- W4: preflight mandatory integration.
- W5: run_manifest v2 and strict validator.
- W6: privacy redaction and public/internal artifact split.
- W7: mock audit replay in clean environment.
- W8: final submission bundle and evidence index.

## Mandatory Evidence List
- `artifacts/validation_report.json`
- `artifacts/splits/split_manifest_*.json`
- `artifacts/splits/split_assignment_*.csv` (or `.json`)
- phase smoke run logs with preflight enabled
- strict run-manifest validation output
- limitations/failure-modes/domain-shift report block
- public bundle redaction proof

## Privacy Rule for Audit and Open Release
- Algorithms and code can be open.
- Personal identifiers and internal infra details must not be in public artifacts.
