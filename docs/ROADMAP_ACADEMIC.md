# ROADMAP_ACADEMIC

This roadmap is the Year-1-audit-safe replacement plan for ATLAS under strict publication standards.

## Time Calibration
- Current date: 2026-02-28.
- Year 1 closeout window: 2026-03-01 to 2026-04-30 (8 weeks).
- Year 2: 2026-05-01 to 2027-04-30.
- Year 3: 2027-05-01 to 2028-04-30.

## Non-negotiable Principles
- Provenance missing is a hard failure.
- Leakage must be zero.
- IID and OOD must be reported separately.
- JSON artifacts are canonical; Markdown is summary only.
- Public artifacts must be de-identified (no personal host/user/path/pid details).

## Year 1 (Final 2 Months) Audit-Pass Program
### Definition of Done
1. `validate-data` enforces schema/provenance/units/leakage/outlier/drift gates with deterministic output.
2. `make-splits` emits deterministic compositional/prototype manifests and assignment files.
3. Phase 1/2/5, benchmark, and full-project launchers enforce preflight by default.
4. Run manifest v2 is complete (JSON canonical + YAML mirror + strict validation).
5. RTX 3060 local path can re-run validate -> splits -> smoke training.
6. Public bundle output is redacted and privacy-safe.

### 8-Week Sprint Breakdown
- W1: Gate behavior hardening.
- W2: Units and drift baseline support.
- W3: Split manifest v2 + assignment determinism.
- W4: Mandatory preflight integration for launchers.
- W5: Run manifest v2 contract and strict validator.
- W6: Public/internal visibility split and redaction.
- W7: Mock audit replay in clean environment.
- W8: Submission bundle freeze and evidence index.

## A0-A10 Academic Macro Phases

## A0 Environment and Version Baseline
- Purpose: Freeze reproducible starting state.
- Inputs: `pyproject.toml`, requirements files, git repo.
- Outputs: environment snapshot, git SHA snapshot, lock policy selection.
- Acceptance: environment check passes and lock metadata is available in run manifests.
- Risks: dependency drift and CUDA mismatch.
- Failure modes: unresolved packages, non-deterministic seed behavior.
- Required logs: `env_snapshot.json`, `version_snapshot.json`.
- Micro tasks:
  - A0.001 verify environment profile and CUDA compatibility.
  - A0.002 record git SHA/dirty state.
  - A0.003 set deterministic seeds and audit defaults.

## A1 Data Acquisition and Provenance Tagging
- Purpose: collect data with explicit provenance and versioning.
- Inputs: DFT primary sources + optional experimental/synthetic.
- Outputs: raw dataset snapshot with provenance records.
- Acceptance: provenance coverage is 100%.
- Risks: upstream schema changes, partial data pulls.
- Failure modes: missing source_version/source_id metadata.
- Required logs: dataset snapshot manifest.
- Micro tasks:
  - A1.001 ingest source datasets into cache with snapshot id.
  - A1.002 attach provenance taxonomy fields to each sample.
  - A1.003 emit dataset fingerprint/hash.

## A2 Data Correctness and Trust Scoring
- Purpose: run data gates and compute trust score 0-100.
- Inputs: provenance-tagged dataset.
- Outputs: validation JSON + Markdown summary + trust distribution.
- Acceptance: schema/provenance/leakage/units hard gates pass.
- Risks: false positives in unit and outlier checks.
- Failure modes: silent gate bypass, non-audited deletions.
- Required logs: `validation_report.json`, `validation_report.md`.
- Micro tasks:
  - A2.001 run `validate-data` full mode.
  - A2.002 mark questionable points for downweighting, not deletion.
  - A2.003 keep deletion list only for extreme audited outliers.
  - A2.004 emit trust-weight ablation toggle plan.

## A3 Split Governance and OOD Protocol
- Purpose: deterministic IID/compositional/prototype split generation.
- Inputs: validated dataset snapshot.
- Outputs: split manifests, assignment tables, split hashes.
- Acceptance: identical seed yields identical split hash.
- Risks: tiny OOD subsets for sparse chemical systems.
- Failure modes: overlap leakage across train/val/test.
- Required logs: split manifests + assignment hash report.
- Micro tasks:
  - A3.001 run `make-splits --strategy compositional --emit-assignment`.
  - A3.002 run `make-splits --strategy prototype --emit-assignment`.
  - A3.003 archive split ids/hash in run manifests.

## A4 Phase 1 Baseline Experiments
- Purpose: establish single-task baseline.
- Inputs: validated dataset + deterministic splits.
- Outputs: trained models and IID metrics.
- Acceptance: run manifest v2 complete and strict-valid.
- Risks: overfitting on small subsets.
- Failure modes: unstable loss due outlier tails.
- Required logs: model checkpoints, history, run_manifest.
- Micro tasks:
  - A4.001 smoke run on RTX 3060.
  - A4.002 standard run for core properties.
  - A4.003 bootstrap CI for key metrics.

## A5 Phase 2 Multi-task Experiments
- Purpose: quantify multi-task gains and negative transfer.
- Inputs: same governed data/splits.
- Outputs: per-task IID metrics and ablations.
- Acceptance: with/without trust weighting comparison completed.
- Risks: task conflict and imbalanced gradients.
- Failure modes: hidden performance collapse for minority tasks.
- Required logs: per-task report with CI.
- Micro tasks:
  - A5.001 run phase2 baseline.
  - A5.002 compare against phase1 per-property.
  - A5.003 report transfer gains/losses.

## A6 OOD Evaluation
- Purpose: mandatory IID vs OOD reporting.
- Inputs: A4/A5 trained models and OOD manifests.
- Outputs: IID/compositional/prototype side-by-side tables.
- Acceptance: all tasks include IID and both OOD metrics.
- Risks: OOD sample count instability.
- Failure modes: accidental split misuse in evaluation.
- Required logs: OOD report JSON + table artifacts.
- Micro tasks:
  - A6.001 evaluate compositional OOD.
  - A6.002 evaluate prototype OOD.
  - A6.003 bootstrap CI for all major metrics.

## A7 Active Learning for Candidate Screening
- Purpose: optimize screening strategy for follow-up DFT.
- Inputs: trained predictor + pool data.
- Outputs: AL rounds, candidate queue, acquisition diagnostics.
- Acceptance: AL vs random baseline is fully reported.
- Risks: acquisition degeneracy.
- Failure modes: opaque ranking without score decomposition.
- Required logs: per-round selection and score components.
- Micro tasks:
  - A7.001 run phase5 strategy matrix.
  - A7.002 generate candidate queue for DFT follow-up.
  - A7.003 report positive and negative AL outcomes.

## A8 Ablations and Negative Results
- Purpose: prevent cherry-picking and support falsifiability.
- Inputs: A4-A7 outputs.
- Outputs: ablation tables and negative-result log.
- Acceptance: at least trust weighting and OOD split ablations included.
- Risks: insufficient compute budget.
- Failure modes: dropping null/negative results.
- Required logs: ablation JSON + failure taxonomy notes.
- Micro tasks:
  - A8.001 trust weighted vs uniform.
  - A8.002 IID-only vs IID+OOD protocol.
  - A8.003 single-task vs multi-task.
  - A8.004 AL strategy comparison vs random.

## A9 Publication Artifact Generation
- Purpose: create paper-ready figures/tables from machine-readable artifacts.
- Inputs: validated experiment outputs.
- Outputs: figure/table bundles and claims blocks.
- Acceptance: each figure/table maps to run_id + split_id + dataset snapshot.
- Risks: table drift from manual edits.
- Failure modes: untraceable claims.
- Required logs: figure source maps.
- Micro tasks:
  - A9.001 generate IID/OOD table pack.
  - A9.002 generate AL convergence plots.
  - A9.003 generate limitations/failure-modes section artifacts.

## A10 Reproducibility Package and Submission
- Purpose: arXiv-first, third-party rerunnable package.
- Inputs: A0-A9 outputs.
- Outputs: runbook, lock metadata, manifests, results package.
- Acceptance: third party can replay key path from clean environment.
- Risks: lock drift and missing artifact references.
- Failure modes: incomplete manifest fields.
- Required logs: submission readiness checklist report.
- Micro tasks:
  - A10.001 freeze release SHA and version tag.
  - A10.002 bundle dataset/split/run manifests.
  - A10.003 verify strict manifest validator pass.
  - A10.004 publish public-redacted package.

## Year 2-3 Quarterly Plan (Detailed)

### Year 2 Q1 (2026-05 to 2026-07): research baseline rebuild after Year1 pass
- Deliver experiment registry v1 (`task/split/seed/model/weighting`).
- Integrate trust weighting into Phase1/Phase2 with explicit ablation switches.
- Produce fixed IID/compositional/prototype evaluator outputs.
- Enforce bootstrap CI generation in evaluation jobs.
- Prepare local/HPC shared config schema without changing local baseline path.

### Year 2 Q2 (2026-08 to 2026-10): scale-up and method comparison
- Run multi-model matrix (CGCNN/equivariant/M3GNet-like baselines).
- Run single-task vs multi-task transfer diagnostics.
- Add uncertainty calibration pipeline (coverage/NLL/miscalibration).
- Run AL strategy baseline matrix (EI/UCB/Thompson/Hybrid/Random).
- Add data-version drift watch artifacts.

### Year 2 Q3 (2026-11 to 2027-01): HPC pilot and decision-aware AL
- Build local-HPC parity checks (trend consistency, schema parity).
- Add HPC job adapter metadata and collection policy.
- Upgrade acquisition scoring to value-risk-cost decomposition.
- Emit DFT-ready candidate queue with traceable score parts.
- Build AL failure taxonomy and mitigation registry.

### Year 2 Q4 (2027-02 to 2027-04): arXiv-first package hardening
- Freeze table/figure pipelines.
- Produce model/data cards (minimal required fields from your templates).
- Create REPRODUCE scripts for major tracks.
- Execute external reproduction pilot at least once.
- Perform claim consistency and risk wording review.

### Year 3 Q1 (2027-05 to 2027-07): API/CLI contract freeze
- Finalize public API inventory and compatibility boundary.
- Apply deprecation policy and compatibility windows.
- Expand tests: contract/privacy/performance smoke.
- Add docs-command contract checks.
- Harden packaging and install path consistency.

### Year 3 Q2 (2027-08 to 2027-10): release candidate and reproducibility scorecard
- Create `v1.0-rc` release workflow.
- Automate internal/public artifact split.
- Add privacy/security scanning for path/secret leaks.
- Emit reproducibility scorecard for each candidate release.
- Lock benchmark versions and drift checks.

### Year 3 Q3 (2027-11 to 2028-01): contributor ecosystem
- Build 15-30 minute contributor onboarding path.
- Provide issue/PR templates split by research/engineering tracks.
- Define governance cadence for triage/release review.
- Publish minimal reproducible example packs.
- Enforce quality gates for external PRs.

### Year 3 Q4 (2028-02 to 2028-04): v1.0 release and paper alignment
- Publish `v1.0.0` with verified install/train/eval/reproduce path.
- Publish citation mapping between paper commit/tag/release.
- Publish initial LTS policy.
- Publish Year4+ research backlog draft.
- Final privacy/compliance audit on public outputs.

## Journal/ArXiv Submission Checklist
- [ ] Provenance completeness: 100%.
- [ ] Validation hard gates: schema/provenance/leakage/units all zero violations.
- [ ] IID + compositional-OOD + prototype-OOD are all reported.
- [ ] Trust weighting ablations (with and without) are included.
- [ ] Confidence intervals/bootstraps included for primary metrics.
- [ ] Negative results logged and not omitted.
- [ ] Run manifests strict validation pass.
- [ ] Split manifests and assignment hashes are archived.
- [ ] Data/model cards included (using your provided templates).
- [ ] Limitations/failure modes/domain-shift risks generated in final package.
- [ ] Public release bundle is de-identified.
