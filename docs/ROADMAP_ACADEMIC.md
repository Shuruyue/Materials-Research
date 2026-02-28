# ROADMAP_ACADEMIC (Public Redacted)

This roadmap is the public-facing academic execution contract.
Detailed internal schedules and reviewer-facing milestones are intentionally omitted.

## Non-Negotiable Scientific Rules
- Provenance missing is a hard failure.
- Leakage must be zero.
- IID and OOD must be reported separately.
- JSON artifacts are canonical; Markdown is summary.
- Public outputs must be de-identified.

## A0-A10 Academic Phases (Public Contract)

## A0 Environment Baseline
- Purpose: freeze reproducible environment and version metadata.
- Public evidence: environment snapshot + git/version metadata in run manifests.

## A1 Data Provenance
- Purpose: attach provenance taxonomy and source version per sample.
- Public evidence: dataset snapshot metadata with provenance coverage.

## A2 Data Correctness + Trust
- Purpose: run schema/provenance/units/leakage/outlier/drift gates and compute trust scores.
- Public evidence: validation JSON + summary + trust distribution.

## A3 Split Governance
- Purpose: deterministic IID/compositional/prototype splits.
- Public evidence: split manifests, assignment files, stable hashes.

## A4 Baseline Training (Phase1)
- Purpose: establish single-task baseline with reproducible manifests.
- Public evidence: phase outputs + strict-valid run manifests.

## A5 Multi-Task Training (Phase2)
- Purpose: evaluate multi-task gains and risks.
- Public evidence: per-task results + with/without trust weighting ablations.

## A6 OOD Evaluation
- Purpose: mandatory IID vs OOD side-by-side reporting.
- Public evidence: IID/compositional/prototype metrics with sample counts.

## A7 Active Learning (Phase5)
- Purpose: candidate screening strategy with traceable acquisition behavior.
- Public evidence: per-round selection outputs and strategy comparisons.

## A8 Ablations and Negative Results
- Purpose: preserve falsifiability and avoid cherry-picking.
- Public evidence: ablation tables + negative result logs.

## A9 Publication Artifacts
- Purpose: generate traceable figures/tables from machine-readable artifacts.
- Public evidence: figure/table bundles mapped to run/split/dataset ids.

## A10 Reproducibility Package
- Purpose: third-party rerunnable package.
- Public evidence: runbook + manifest chain + strict checks.

## Submission Checklist (Public)
- [ ] Provenance completeness and hard gates pass
- [ ] IID + compositional OOD + prototype OOD reported
- [ ] Trust weighting ablation included
- [ ] Confidence intervals/bootstraps included
- [ ] Negative results logged
- [ ] Run/split/dataset manifest chain complete
- [ ] Data/model cards included (template-based)
- [ ] Public bundle passes privacy checks
