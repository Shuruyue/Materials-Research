# ATLAS TODO

Last updated: 2026-02-21

## P0 - Stability / Reproducibility
- [ ] Finalize optional dependency strategy for MACE stack (`e3nn` compatibility policy and lock file guidance).
- [ ] Add CI workflow for:
  - lint
  - unit tests
  - optional integration matrix (allowed-fail profile)
- [ ] Add artifact retention policy for benchmark/discovery outputs.

## P1 - Benchmark Quality
- [ ] Add training-mode Matbench protocol (current runner is inference-focused).
- [ ] Add benchmark comparison table generator (method/profile/date grouped report).
- [ ] Add baseline adapters:
  - composition baseline
  - descriptor baseline

## P1 - Discovery Pipeline
- [ ] Add chemistry prior gate before expensive relax/classify steps.
- [ ] Add explicit failure taxonomy in workflow manifests (dependency missing, model load fail, runtime fail).
- [ ] Add experiment templates for metal doping and semiconductor thin-film tracks.

## P2 - Data and Model Governance
- [ ] Define canonical dataset versions and split manifests.
- [ ] Add model registry metadata:
  - checkpoint lineage
  - training config hash
  - dependency profile
- [ ] Add reproducibility replay command for a specific manifest run.

## P2 - Documentation
- [ ] Keep `README.md`, `docs/PROJECT_STATUS.md`, and `docs/TODO.md` synchronized each milestone.
- [ ] Add concise developer guide for:
  - method switching (`method_key`)
  - data source switching (`data_source_key`)
  - benchmark CLI usage patterns

## Done Recently
- [x] Reorganized tests into `unit` and `integration`.
- [x] Added benchmark CLI and fold/aggregate JSON reports.
- [x] Added reproducibility workflow manifests and runtime metadata capture.
- [x] Standardized requirements profile files and environment checker.
