# ROADMAP_PROJECT

This is the implementation roadmap for open-source engineering maturity and release governance.

## Scope Guardrails
- Extend existing ATLAS structures only (no parallel framework rewrite).
- Keep defaults deterministic and lightweight.
- Preserve local RTX 3060 reproducibility even after HPC adoption.
- Public releases must hide personal/internal identifying information.

## B0-B10 Project Macro Phases

## B0 Repository Contract Baseline
- Purpose: lock package layout and script entrypoint contract.
- Inputs: current repo structure and pyproject metadata.
- Outputs: baseline interface inventory.
- Acceptance: all registered CLIs load with `--help`.
- Risks: hidden script drift from docs.
- Failure modes: missing entrypoint wiring.
- Required logs: interface contract report.
- Micro tasks:
  - B0.001 inventory package/modules/scripts/tests/docs.
  - B0.002 inventory public CLI entrypoints.
  - B0.003 baseline interface regression check.

## B1 Data Governance CLI Layer
- Purpose: operationalize validation/split governance CLIs.
- Inputs: governance specs.
- Outputs: `validate-data`, `make-splits`, machine-readable reports.
- Acceptance: hard gates and deterministic split hashes reproducible.
- Risks: upstream dataset schema drift.
- Failure modes: gate bypass in training launchers.
- Required logs: validation and split artifacts.
- Micro tasks:
  - B1.001 validate-data schema/provenance/leakage/units/outlier/drift.
  - B1.002 make-splits IID/compositional/prototype with assignment.
  - B1.003 add strict unit tests for gate behavior.

## B2 Training/Benchmark Preflight Enforcement
- Purpose: make governance checks mandatory before execution.
- Inputs: B1 CLIs.
- Outputs: phase/benchmark preflight stage.
- Acceptance: launcher exits non-zero when preflight fails.
- Risks: preflight runtime overhead.
- Failure modes: skip flags used incorrectly in real runs.
- Required logs: preflight command and result logs.
- Micro tasks:
  - B2.001 integrate preflight into Phase1/2/5 launchers.
  - B2.002 integrate preflight into full-project orchestrator.
  - B2.003 integrate preflight into benchmark CLI.

## B3 Reproducibility Manifest v2
- Purpose: produce complete auditable run records.
- Inputs: run_utils, launcher metadata, lock strategy.
- Outputs: run_manifest v2 JSON + YAML mirror.
- Acceptance: strict validator reaches 100% completeness.
- Risks: legacy manifests in old directories.
- Failure modes: missing split/dataset/environment lock metadata.
- Required logs: strict contract validation report.
- Micro tasks:
  - B3.001 add schema_version/visibility/dataset/split/environment_lock/artifacts blocks.
  - B3.002 implement public redaction path.
  - B3.003 update validator with strict completeness checks.

## B4 Test Strategy and Split (Unit vs Integration)
- Purpose: maintain fast default CI while preserving E2E checks.
- Inputs: existing tests + new governance code.
- Outputs: expanded unit/integration coverage.
- Acceptance: unit tests fast; integration tests marked and optional.
- Risks: flaky integration due data/network.
- Failure modes: unmarked heavy tests in default path.
- Required logs: pytest summary by marker.
- Micro tasks:
  - B4.001 unit: schema/leakage/determinism/manifest validation.
  - B4.002 integration: validate->split->preflight smoke path.
  - B4.003 ensure deterministic seed behavior in tests.

## B5 CI Policy
- Purpose: enforce quality gates with tiered runtime cost.
- Inputs: B4 tests, lint checks, optional smoke training.
- Outputs: CI policy and implementation-ready workflow spec.
- Acceptance: default CI stays fast and stable.
- Risks: optional dependency gaps across runners.
- Failure modes: integration jobs blocking routine PR flow.
- Required logs: CI quality gate reports.
- Micro tasks:
  - B5.001 Fast CI: lint + unit tests.
  - B5.002 Integration CI: marked tests + interface checks.
  - B5.003 Optional smoke training for release readiness.

## B6 Release Engineering Discipline
- Purpose: formalize release mechanics and compatibility policy.
- Inputs: changelog/versioning workflow.
- Outputs: SemVer rules, deprecation and compatibility policy.
- Acceptance: every release has changelog and compatibility statement.
- Risks: accidental breakage to public CLI/API contract.
- Failure modes: undocumented breaking changes.
- Required logs: release checklist artifact.
- Micro tasks:
  - B6.001 enforce SemVer progression.
  - B6.002 enforce changelog entries per release.
  - B6.003 maintain backward compatibility matrix.

## B7 Documentation and Runbook Contract
- Purpose: keep docs and executable commands synchronized.
- Inputs: README/docs index and script interfaces.
- Outputs: stable runbooks and docs checks.
- Acceptance: all documented commands are executable.
- Risks: stale docs after refactors.
- Failure modes: non-runnable README instructions.
- Required logs: docs contract report.
- Micro tasks:
  - B7.001 update README run order and minimal command set.
  - B7.002 index governance/repro roadmap docs.
  - B7.003 add docs consistency checks.

## B8 Privacy and Compliance for Open Releases
- Purpose: protect privacy while keeping algorithms open.
- Inputs: manifests, artifacts, logs.
- Outputs: internal/public visibility split policy.
- Acceptance: public artifacts pass redaction checks.
- Risks: path/hostname/pid leakage.
- Failure modes: accidental disclosure in runtime metadata.
- Required logs: privacy scan reports.
- Micro tasks:
  - B8.001 classify artifact fields as internal/public.
  - B8.002 automate redaction for public bundles.
  - B8.003 add privacy checks to release gate.

## B9 Release Candidate Readiness
- Purpose: verify end-to-end readiness before v1.0.
- Inputs: all previous phases.
- Outputs: release candidate and validation evidence.
- Acceptance: install + smoke run + manifest strict pass.
- Risks: unresolved dependency lock drift.
- Failure modes: release cannot reproduce baseline run.
- Required logs: RC validation report.
- Micro tasks:
  - B9.001 run full preflight and smoke workflows.
  - B9.002 run strict manifest validator.
  - B9.003 verify public artifact redaction.

## B10 GA Release and Long-term Maintenance
- Purpose: publish stable open-source release and support policy.
- Inputs: RC outputs and review sign-off.
- Outputs: tagged release, maintenance policy, next-cycle backlog.
- Acceptance: v1.0 package is reproducible and contributor-ready.
- Risks: support load spike after release.
- Failure modes: no deprecation discipline post-release.
- Required logs: final release compliance report.
- Micro tasks:
  - B10.001 publish v1.0 tag and release notes.
  - B10.002 publish support/deprecation timeline.
  - B10.003 publish Year4+ roadmap draft.

## Release Engineering Plan
### SemVer
- `0.x.y`: pre-stable research/engineering co-development.
- `1.0.0`: first stable release aligned to publication package.
- `1.x.y`: backward compatible features/fixes.
- `2.0.0+`: breaking public API/CLI changes.

### Changelog Discipline
- Follow Keep-a-Changelog categories: Added/Changed/Deprecated/Removed/Fixed/Security.
- Every release PR must update changelog.
- Every deprecation must include removal target version.

### Backward Compatibility Policy
- Public contract includes CLI entrypoints and documented API surface.
- Internal/dev scripts are best-effort and can evolve faster.
- Breaking changes require major bump or explicit compatibility bridge.

### Deprecation Policy
- Minimum one minor version warning window.
- Emit runtime warnings in deprecated paths.
- Document migration path and removal date.

## CI Strategy
- Fast default (required): lint + unit tests.
- Integration (required for merge to protected branch): marked integration tests + CLI contract checks.
- Optional smoke training (release/pre-merge hotfix): short Phase1/2/5 dry smoke on local-compatible settings.
