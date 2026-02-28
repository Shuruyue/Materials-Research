# ROADMAP_PROJECT (Public Redacted)

This roadmap defines open-source engineering maturity without exposing private planning details.

## Scope Guardrails
- Extend existing ATLAS structures only.
- Keep default workflows deterministic and lightweight.
- Maintain local reproducibility path even when scaling to larger compute.
- Protect privacy in all public artifacts.

## B0-B10 Project Phases (Public Contract)
- B0 Repository contract baseline (CLI/API inventory and stability checks)
- B1 Data governance CLI layer (`validate-data`, `make-splits`)
- B2 Mandatory preflight enforcement for training/benchmark launchers
- B3 Run manifest v2 contract and strict validation
- B4 Unit/integration split with deterministic defaults
- B5 CI policy (fast unit default + integration gates + optional smoke)
- B6 Release engineering discipline (SemVer/changelog/compatibility)
- B7 Documentation and runbook contract checks
- B8 Privacy/compliance for public artifact publication
- B9 Release candidate readiness verification
- B10 GA release and maintenance policy

## Release Engineering Policy
### SemVer
- `0.x.y`: active research/development
- `1.0.0`: first stable release aligned to reproducibility package
- `1.x.y`: backward-compatible changes
- `2.0.0+`: breaking changes

### Changelog Discipline
- Use Added/Changed/Deprecated/Removed/Fixed/Security categories.
- Every release must include migration notes for relevant changes.

### Compatibility + Deprecation
- Public CLI/API contract requires compatibility windows.
- Deprecations require warning period and removal target version.

## CI Strategy
- Required fast lane: lint + unit tests.
- Required merge gate: integration markers + interface contract checks.
- Optional release gate: lightweight smoke training.
