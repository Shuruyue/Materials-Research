# Optimization Report (2026-03-04)

This report summarizes the requested 3-round optimization loop plus final consolidation.

## Round 1 - Architecture Optimization

- Added `atlas/active_learning/objective_space.py` to centralize objective-space utilities:
  - clipping/float-safe conversion
  - objective matrix construction from score terms
  - history objective aggregation
  - feasibility masking
- Refactored `atlas/active_learning/controller.py` to use the shared objective-space helpers instead of repeating local logic in multiple methods.
- Result: less duplicated code in Pareto/HV paths and clearer separation between orchestration and objective-space math.

## Round 2 - Core Algorithm Optimization

- Optimized Pareto front extraction in `DiscoveryController._pareto_front` with vectorized dominance checks.
- Reduced repeated HV computations:
  - optimized 2D branch in `_mc_hv_improvements_shared` to compute baseline hypervolume once
  - in `_select_top_diverse`, reused shared-sample HV-improvement path for greedy batch selection when enabled
  - removed repeated stack/rebuild work in inner loops
- Result: lower compute overhead in selection-heavy active-learning rounds while keeping output behavior consistent with existing tests.

## Round 3 - Core Training Stack Optimization

- Updated `atlas/training/trainer.py`:
  - cached model forward signature once at init (instead of per batch)
  - unified autocast context via helper (`self._autocast()`)
  - switched to `optimizer.zero_grad(set_to_none=True)` for lower overhead
  - made gradient clipping configurable (`grad_clip_norm`)
  - strengthened checkpoint loading with explicit missing-file error
  - stored lightweight checkpoint config metadata payload
- Result: lower per-step overhead and improved checkpoint robustness.

## Final Consolidation

- Unified GP-UCB schedule implementation:
  - `atlas/active_learning/gp_surrogate.py` now uses `schedule_ucb_kappa` from `atlas/active_learning/acquisition.py`
  - avoids drift from maintaining duplicate schedule formulas.

## Round 4 - Architecture + Core Algorithm Refactor

- Added `atlas/active_learning/pareto_utils.py` and moved Pareto/HV math out of `DiscoveryController`:
  - Pareto front extraction
  - non-dominated sorting
  - crowding distance
  - Pareto rank scoring
  - 2D exact hypervolume + generic hypervolume
  - shared-sample HV-improvement estimation
- Updated `atlas/active_learning/controller.py` to call the new pure-function module, reducing controller method bloat and isolating numerical logic.
- Optimized non-dominated sorting path:
  - replaced repeated front-wise point elimination loop with dominance-matrix based front peeling (near O(n^2) behavior for fixed objective dimension).
- Optimized shared HV-improvement path:
  - vectorized candidate-vs-probe dominance checks in chunks for >=3 objectives.
  - keeps deterministic seed semantics while lowering Python-loop overhead.
- Optimized diversity-aware greedy selection:
  - precomputed pairwise Gaussian similarity matrix once per batch.
  - switched to incremental max-similarity updates and vectorized candidate scoring in each greedy step.
- Added dedicated unit tests:
  - `tests/unit/active_learning/test_pareto_utils.py`
  - validates rank correctness (vs brute force), exact 2D HV behavior, and shared-HV feasibility gating.

## Round 5 - Policy Engine + Runtime Hardening (博士級路線第一批落地)

- Added typed policy config/state:
  - `atlas/active_learning/policy_state.py`
  - `ActiveLearningPolicyConfig` replaces ad-hoc policy parameter reads.
  - `PolicyState` persists calibration scale factors and relaxer circuit-breaker state.
- Added decision-only strategy engine:
  - `atlas/active_learning/policy_engine.py`
  - Supports `legacy` and `cmoeic` policy routing.
  - `cmoeic` computes utility with objective/feasibility/risk/cost terms and writes per-candidate decision artifacts.
- Refactored controller integration:
  - `DiscoveryController` now accepts policy options and optional injected `policy_engine`.
  - `_score_and_select` delegates to policy engine; legacy behavior preserved in `_score_and_select_legacy`.
  - Added `_finalize_ranked_candidates` to unify winner registration/history updates.
- Added runtime robustness for relaxation:
  - timeout wrapper via `ThreadPoolExecutor` (`relax_timeout_sec`)
  - retry cap (`relax_max_retries`)
  - circuit breaker (`relax_circuit_breaker_failures`, `relax_circuit_breaker_cooldown_iters`)
  - per-iteration failure buckets persisted in workflow notes.
- Extended candidate artifact schema:
  - `calibrated_mean`, `calibrated_std`, `conformal_radius`, `risk_score`,
    `estimated_cost`, `gain_per_cost`, `reject_reason`
  - backward compatible (additive fields only).
- Added Phase5 CLI flags:
  - `--policy {legacy,cmoeic}`
  - `--risk-mode {soft,hard,hybrid}`
  - `--cost-aware`
  - `--calibration-window`
  - wired through both `run_phase5.py` and `run_discovery.py`.
- Added tests:
  - `tests/unit/active_learning/test_policy_state.py`
  - `tests/unit/active_learning/test_policy_engine.py`
  - `tests/unit/active_learning/test_phase5_cli.py`
  - `tests/unit/active_learning/test_controller_runtime_stability.py`

## Validation

- `ruff check atlas/active_learning/controller.py atlas/active_learning/objective_space.py atlas/active_learning/gp_surrogate.py atlas/training/trainer.py`
- `pytest tests/unit/active_learning/test_acquisition.py tests/unit/active_learning/test_controller_acquisition.py tests/unit/active_learning/test_gp_surrogate.py tests/unit/training/test_trainer.py -q`

All above checks passed.

Additional checks for this round:

- `python -m ruff check atlas/active_learning/controller.py atlas/active_learning/pareto_utils.py tests/unit/active_learning/test_pareto_utils.py`
- `python -m pytest tests/unit/active_learning/test_acquisition.py tests/unit/active_learning/test_controller_acquisition.py tests/unit/active_learning/test_gp_surrogate.py tests/unit/active_learning/test_pareto_utils.py -q`

## External References Reviewed (for architecture/algorithm direction)

- BoTorch (Bayesian optimization toolkit): https://github.com/pytorch/botorch
- qEHVI paper: Daulton et al. (2020), https://arxiv.org/abs/2006.05078
- qNEHVI paper: Daulton et al. (2021), https://arxiv.org/abs/2105.08195
- Batch BO via local penalization: Gonzalez et al. (2016), https://proceedings.mlr.press/v51/gonzalez16a.html
- Constrained BO: Gardner et al. (2014), https://proceedings.mlr.press/v32/gardner14.html
- NSGA-II: Deb et al. (2002), https://doi.org/10.1109/4235.996017
- Fast non-dominated sorting complexity reduction: Jensen (2003), https://doi.org/10.1109/TEVC.2003.817234
- Materials Project reaction-network repository: https://github.com/materialsproject/reaction-network
