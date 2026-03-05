
# ATLAS Sequential Optimization Todo (Temp)

Last updated: 2026-03-05

## Batch 1 (max 5 files)

- [X] `atlas/data/split_governance.py` - reviewed + optimized
- [X] `tests/unit/data/test_split_governance.py` - reviewed + optimized
- [X] `atlas/data/topo_db.py` - reviewed + optimized
- [X] `tests/unit/data/test_topo_db.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - created + updated

## Batch 1 optimization goals

- Refactor high-complexity split/inference routines into smaller helpers.
- Remove local `# noqa: C901` suppressions where practical.
- Preserve deterministic behavior and split/calibration semantics.
- Add regression tests for determinism and calibration diagnostics.

## Batch 1 outcomes

- `topo_db.infer_topology_probabilities` no longer relies on `# noqa: C901`; config-validation logic extracted into reusable helpers.
- Split governance now sanitizes similarity matrices and guards optimizer hyperparameters (`n_restarts`, `local_moves`) against invalid values.
- Added regression tests for:

  - split optimizer guard behavior (`n_restarts=0`)
  - similarity-matrix sanitization with NaN/Inf values
  - invalid `weight_constraint` handling
  - invalid `base_weights` shape handling

## Research references used in this batch

- Kernighan, Lin (1970), graph partition local-improvement heuristic: https://doi.org/10.1002/j.1538-7305.1970.tb01770.x
- Fiduccia, Mattheyses (1982), linear-time partition improvement: https://doi.org/10.1145/800263.809204
- Guo et al. (ICML 2017), temperature scaling calibration: https://proceedings.mlr.press/v70/guo17a.html
- Ledoit, Wolf (2004), covariance shrinkage/conditioning: https://doi.org/10.1016/S0047-259X(03)00096-4
- Joeres et al. (Nature Communications 2025), leakage-aware splitting (DataSAIL): https://www.nature.com/articles/s41467-025-58606-8

## Batch 2 (max 5 files)

- [X] `atlas/active_learning/policy_engine.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_phase5.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_discovery.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_phase5_cli.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 2 optimization goals

- Improve numeric stability in CMOEIC policy scoring path.
- Harden Phase5 CLI argument validation and override behavior.
- Add regression tests for CLI construction and argument guards.

## Batch 2 outcomes

- `PolicyEngine` now uses explicit helper functions for probability clamping, calibrated energy extraction, and base utility construction.
- Added safety guard for conformal denominator (`max_conformal_radius`) to avoid divide-by-zero behavior under malformed config.
- `run_phase5.py` now:

  - validates key numeric arguments before execution,
  - replaces profile default flags instead of appending duplicate flag/value pairs.
- `run_discovery.py` now validates core AL CLI arguments early (non-negative/finite/range checks).
- Added CLI regression tests covering:

  - policy-flag injection,
  - profile override replacement semantics,
  - invalid `--top > --candidates` guard.

## Research references used in batch 2

- Python argparse docs (official): https://docs.python.org/3/library/argparse.html
- Python subprocess docs (official): https://docs.python.org/3/library/subprocess.html
- Jones et al. (1998), Efficient Global Optimization / EI: https://doi.org/10.1023/A:1008306431147
- Gardner et al. (2014), Bayesian optimization with inequality constraints: https://proceedings.mlr.press/v32/gardner14.html
- Angelopoulos, Bates (2021), conformal prediction tutorial: https://arxiv.org/abs/2107.07511

## Batch 3 (max 5 files)

- [X] `atlas/active_learning/policy_engine.py` - reviewed + optimized (numerical stability pass)
- [X] `scripts/phase5_active_learning/run_phase5.py` - reviewed + optimized (CLI hardening pass)
- [X] `scripts/phase5_active_learning/run_discovery.py` - reviewed + optimized (CLI guard pass)
- [X] `tests/unit/active_learning/test_phase5_cli.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 3 optimization goals

- Strengthen policy-scoring numeric guards and helper decomposition.
- Make Phase5 CLI behavior deterministic under overrides.
- Enforce argument-domain constraints early (before heavy runtime init).

## Batch 3 outcomes

- `PolicyEngine`:

  - added `_clamp01` for consistent bounded probabilities,
  - extracted calibrated-stat and utility helpers to reduce scoring-path duplication,
  - added protected denominator for conformal scaling.
- `run_phase5.py`:

  - added `_validate_args` with domain checks,
  - added `_set_or_replace_flag` so profile defaults are replaced, not duplicated.
- `run_discovery.py`:

  - added `_validate_discovery_args` with fast-fail checks (`top<=candidates`, finite/positive constraints).
- `test_phase5_cli.py`:

  - added regression tests for override replacement semantics,
  - added guards tests for invalid `top/candidates` in both launchers.

## Research references used in batch 3

- Python `concurrent.futures` docs (timeouts/executor semantics): https://docs.python.org/3/library/concurrent.futures.html
- PEP 3148 (futures design rationale): https://peps.python.org/pep-3148/
- Bayesian optimization with constraints (Gardner et al., 2014): https://proceedings.mlr.press/v32/gardner14.html
- EI / Efficient Global Optimization (Jones et al., 1998): https://doi.org/10.1023/A:1008306431147
- Conformal prediction tutorial (Angelopoulos & Bates, 2021): https://arxiv.org/abs/2107.07511

## Batch 4 (max 5 files)

- [X] `atlas/active_learning/controller.py` - reviewed + optimized
- [X] `atlas/active_learning/policy_state.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_controller_runtime_stability.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_policy_state.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 4 optimization goals

- Harden policy/state config parsing against malformed profile and resume payload values.
- Stabilize runtime retry behavior with bounded exponential backoff + jitter controls.
- Add deterministic regression tests for retry/backoff and state sanitization paths.
- Keep compatibility with existing legacy-policy defaults.

## Batch 4 outcomes

- `policy_state.py`:

  - added typed coercion helpers (`_coerce_bool/_coerce_int/_coerce_float`) to safely parse profile values,
  - added policy retry-backoff config fields (`relax_retry_backoff_sec`, `relax_retry_backoff_max_sec`, `relax_retry_jitter`),
  - added `PolicyState.validated()` and strict `from_dict` sanitization to prevent invalid resume state (NaN/Inf/negative counters) from propagating.
- `controller.py`:

  - wired retry-backoff fields from policy config into runtime,
  - added `_retry_sleep_seconds()` implementing capped exponential backoff with optional jitter,
  - applied retry sleep between failed relax attempts and exposed backoff settings in workflow/report metadata.
- Runtime tests:

  - added retry recovery test proving transient failures can recover with bounded sleep,
  - added direct cap test for exponential retry schedule,
  - preserved existing timeout and circuit-breaker behavior tests.

## Research references used in batch 4

- Python `concurrent.futures` docs (timeouts and cancellation semantics): https://docs.python.org/3/library/concurrent.futures.html
- PEP 3148 (Futures design and retry-oriented execution model): https://peps.python.org/pep-3148/
- AWS Architecture Blog, "Exponential Backoff and Jitter": https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
- Dean and Barroso (2013), "The Tail at Scale": https://research.google/pubs/the-tail-at-scale/

## Batch 5 (max 5 files)

- [X] `atlas/active_learning/acquisition.py` - reviewed + optimized
- [X] `atlas/active_learning/gp_surrogate.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_acquisition.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_gp_surrogate.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 5 optimization goals

- Improve acquisition numeric robustness under malformed hyperparameters and noisy observations.
- Reduce NEI bias from invalid historical observations (NaN/Inf) by filtering, not zero-imputing.
- Add explicit GP surrogate config validation to prevent runtime instability from bad config values.
- Add regression tests for the new sanitization and stability behavior.

## Batch 5 outcomes

- `acquisition.py`:

  - added numeric coercion helpers for robust schedule/MC parameter parsing,
  - improved `_prepare_observed` to filter non-finite observation pairs before NEI sampling,
  - hardened `schedule_ucb_kappa` against invalid/non-finite inputs while preserving existing valid-path semantics.
- `gp_surrogate.py`:

  - added `GPSurrogateConfig.validated()` to sanitize numeric ranges and categorical modes,
  - enforced config validation at `GPSurrogateAcquirer` initialization.
- Tests:

  - `test_acquisition.py` now verifies:

    - dirty observations (NaN/Inf) are filtered consistently in NEI,
    - UCB kappa schedule stays finite under invalid input.
  - `test_gp_surrogate.py` now verifies invalid config fields are normalized to safe values and produce finite `current_kappa`.

## Research references used in batch 5

- BoTorch acquisition docs (`qNoisyExpectedImprovement`, MC noisy BO semantics): https://botorch.readthedocs.io/en/stable/acquisition.html
- scikit-learn `GaussianProcessRegressor` docs (`alpha` and noise handling): https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
- PyTorch `torch.isfinite` API (finite-value screening semantics): https://pytorch.org/docs/stable/generated/torch.isfinite.html
- Ament et al. (2023), LogEI numerical stabilization: https://arxiv.org/abs/2310.20708

## Batch 6 (max 5 files)

- [X] `atlas/active_learning/objective_space.py` - reviewed + optimized
- [X] `atlas/active_learning/pareto_utils.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_objective_space.py` - added + optimized
- [X] `tests/unit/active_learning/test_pareto_utils.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 6 optimization goals

- Harden objective-space conversion against NaN/Inf/out-of-range values.
- Make Pareto/HV helpers robust to malformed arrays and empty high-dimensional inputs.
- Preserve ranking semantics while preventing invalid points from polluting front extraction.
- Expand tests to cover numeric edge cases and new sanitization paths.

## Batch 6 outcomes

- `objective_space.py`:

  - `clip01` now safely handles non-finite inputs,
  - added internal row sanitizer for objective-map conversion (`NaN/Inf` -> bounded finite values),
  - strengthened dimensional coercion and feasibility-mask guards for 1D/non-finite arrays.
- `pareto_utils.py`:

  - added generic 2D-shape normalization helper for point matrices,
  - fixed empty Pareto-front return shape to preserve input objective dimension (not hard-coded to 2D),
  - added finite-value guards in non-dominated sorting and hypervolume estimation,
  - added defensive casting for MC HV sampling parameters (`samples/seed/chunk`).
- Tests:

  - new `test_objective_space.py` for map sanitization, joint-feasibility filtering, and non-finite mask behavior,
  - expanded `test_pareto_utils.py` for empty-shape preservation, non-finite ranking behavior, and HV finite-row filtering.

## Research references used in batch 6

- Deb et al. (2002), NSGA-II fast non-dominated sorting and crowding distance: https://doi.org/10.1109/4235.996017
- BoTorch acquisition docs (MC BO stability and constrained utility context): https://botorch.readthedocs.io/en/stable/acquisition.html
- NumPy `isfinite` reference (finite-value masking semantics): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy `nan_to_num` reference (controlled NaN/Inf replacement): https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html

## Batch 7 (max 5 files)

- [X] `atlas/active_learning/generator.py` - reviewed + optimized
- [X] `atlas/active_learning/synthesizability.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_generator.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_synthesizability.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 7 optimization goals

- Improve generator robustness under non-finite logits and single-worker runtime environments.
- Remove hash-randomization-induced nondeterminism from generator fallback Monte Carlo paths.
- Harden synthesizability evaluator configuration against malformed numeric/bool inputs.
- Add regression coverage for new sanitization and deterministic behavior.

## Batch 7 outcomes

- `generator.py`:

  - strengthened `_softmax` to handle NaN/Inf/empty logits safely and always return a valid probability vector,
  - replaced fallback MC seeding based on Python `hash()` with stable `blake2b`-derived seed material,
  - added single-worker synchronous execution path in `generate_batch` to avoid unnecessary `ProcessPoolExecutor` overhead,
  - cached seed fingerprints incrementally to reduce repeated fingerprint recomputation.
- `synthesizability.py`:

  - added typed coercion helpers (`_coerce_int/_coerce_float/_coerce_bool`) for config parsing,
  - normalized/clamped thresholds and repaired inverted threshold bounds (`threshold_min > threshold_max`),
  - hardened objective-weight normalization against NaN/non-finite values,
  - added finite guards in energy prior and final score composition.
- Tests:

  - `test_generator.py` now verifies softmax non-finite handling, charge-neutrality fallback determinism, and single-worker generation path.
  - `test_synthesizability.py` now verifies invalid config sanitization and finite score outputs for non-finite energy inputs.

## Research references used in batch 7

- Python `PYTHONHASHSEED` docs (hash randomization behavior): https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
- Python `hashlib` docs (`blake2b` stable digesting): https://docs.python.org/3/library/hashlib.html
- Python `ProcessPoolExecutor` docs (process-spawn/serialization model): https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor
- NumPy `SeedSequence` docs (reproducible seed material rationale): https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html

## Batch 8 (max 5 files)

- [X] `atlas/active_learning/crabnet_native.py` - reviewed + optimized
- [X] `atlas/active_learning/rxn_network_native.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_crabnet_native.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_rxn_network_native.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 8 optimization goals

- Harden uncertainty-calibration loops against NaN/Inf/degenerate std inputs.
- Improve ambiguous tensor input-order detection robustness in CrabNet wrapper.
- Stabilize reaction-network risk ranking under extreme cost scales and non-finite values.
- Add targeted regression tests proving finite-safe behavior and fallback consistency.

## Batch 8 outcomes

- `crabnet_native.py`:

  - added finite-safe coercion helpers for positive/non-negative scalar config values,
  - improved `_normalize_input_order` with fraction-likelihood scoring + finite-aware max fallback,
  - hardened uncertainty calibration by filtering invalid rows before quantile estimation,
  - grouped calibration now preserves prior calibration state when current calibration data is fully invalid.
- `rxn_network_native.py`:

  - `_safe_float` now rejects NaN/Inf and falls back deterministically,
  - `_normalize_weights` now ignores unknown/non-finite weight entries to prevent accidental metric dilution,
  - `_path_step_costs` now filters invalid cost vectors and falls back to reaction energies safely,
  - `_entropic_risk` now uses a numerically stable log-mean-exp formulation (shifted/log-sum-exp style),
  - Pareto objective arrays are sanitized before non-dominated sorting/crowding computations.
- Tests:

  - `test_crabnet_native.py` now covers non-finite input-order ambiguity and invalid-row calibration behavior.
  - `test_rxn_network_native.py` now covers non-finite float handling, weight sanitization, extreme-scale entropic risk, and invalid-cost fallback.

## Research references used in batch 8

- Blanchard, Higham, Higham (2019), accurate log-sum-exp/softmax computation: https://arxiv.org/abs/1909.03469
- PyTorch `torch.isfinite` docs: https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- PyTorch `torch.quantile` docs: https://docs.pytorch.org/docs/stable/generated/torch.quantile.html
- NumPy `nan_to_num` docs: https://numpy.org/doc/2.4/reference/generated/numpy.nan_to_num.html
- Ahmadi-Javid (2012), entropic risk measure: https://doi.org/10.1016/j.ejor.2011.11.016

## Batch 9 (max 5 files)

- [X] `atlas/config.py` - reviewed + optimized
- [X] `atlas/benchmark/runner.py` - reviewed + optimized
- [X] `tests/unit/config/test_config.py` - reviewed + optimized
- [X] `tests/unit/benchmark/test_benchmark_runner.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 9 optimization goals

- Strengthen configuration path/device handling for CI and heterogeneous runtime environments.
- Make benchmark-runner initialization and bootstrap routines finite-safe under malformed numeric inputs.
- Preserve existing benchmark semantics while preventing avoidable runtime crashes.
- Add regression tests for the new sanitization behavior.

## Batch 9 outcomes

- `config.py`:

  - `PathConfig` now preserves explicitly supplied `raw_dir/processed_dir/artifacts_dir` values instead of always overriding from `data_dir`,
  - added path normalization (`expanduser`, project-root-relative resolution) for env and explicit inputs,
  - `Config._set_device` now safely falls back to CPU on invalid device strings or torch availability issues,
  - `get_device` now surfaces invalid device strings clearly.
- `benchmark/runner.py`:

  - added typed coercion helpers for `batch_size`, `n_jobs`, bootstrap config, coverage values, and conformal calibration limits,
  - made bootstrap CI utilities robust to invalid `confidence/n_bootstrap/seed` inputs,
  - hardened `conformal_max_calibration_samples` parsing inside conformal metric path.
- Tests:

  - `test_config.py` now covers explicit-subdir preservation, env-relative path resolution, and invalid device fallback,
  - `test_benchmark_runner.py` now covers bootstrap invalid-parameter sanitization and invalid runtime-parameter coercion at runner init.

## Research references used in batch 9

- Joblib Parallel docs (`n_jobs` semantics including `n_jobs=0` invalid): https://joblib.readthedocs.io/en/stable/parallel.html
- PyTorch data loading reference (`torch.utils.data` / DataLoader API): https://docs.pytorch.org/docs/stable/data.html
- PyTorch `torch.device` API reference: https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch-device
- NumPy `RandomState.randint` reference (bootstrap resampling primitive): https://numpy.org/doc/stable/reference/random/generated/numpy.random.RandomState.randint.html
- Angelopoulos & Bates (2021), conformal prediction tutorial: https://arxiv.org/abs/2107.07511

## Batch 10 (max 5 files)

- [X] `atlas/active_learning/crabnet_screener.py` - reviewed + optimized
- [X] `atlas/active_learning/controller.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_crabnet_screener.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_controller_acquisition.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 10 optimization goals

- Harden composition screener numeric/config behavior under malformed runtime hyperparameters.
- Make training-loss path robust to non-finite supervision labels (without crashing rounds).
- Prevent active-learning acquisition instability when historical energy traces contain NaN/Inf.
- Add regression tests for new finite-value guards and fallback behavior.

## Batch 10 outcomes

- `crabnet_screener.py`:

  - added finite-safe coercion for critical hyperparameters (`simplex_blend`, transform temperatures, uncertainty floor, ensemble/mc counts),
  - hardened uncertainty decode path using `torch.nan_to_num` + clamping to prevent std explosions from bad raw heads,
  - sanitized aggregated aleatoric/epistemic variance tensors before sqrt,
  - updated `compute_training_loss` to ignore non-finite target rows and return stable zero scalar when no valid supervision exists.
- `controller.py`:

  - `_current_best_f` now filters non-finite historical observations before min/quantile,
  - `_historical_energy_observations` now drops non-finite means and clamps invalid std inputs,
  - `_current_acquisition_kappa` now falls back to base kappa when scheduler output is non-finite/non-positive,
  - `_stability_component` now falls back to deterministic score when candidate UQ stats are non-finite.
- Tests:

  - `test_crabnet_screener.py` now verifies invalid numeric hyperparameter sanitization and non-finite target loss masking,
  - `test_controller_acquisition.py` now verifies finite filtering in history, `best_f` fallback, kappa fallback, and non-finite candidate UQ fallback.

## Research references used in batch 10

- PyTorch `GaussianNLLLoss` docs (variance positivity and numerical epsilon): https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.GaussianNLLLoss.html
- PyTorch `torch.nan_to_num` docs (NaN/Inf replacement semantics): https://docs.pytorch.org/docs/stable/generated/torch.nan_to_num.html
- PyTorch `torch.isfinite` docs (finite-mask behavior): https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- Srinivas et al. (2010), GP-UCB schedule rationale: https://arxiv.org/abs/0912.3995
- Ament et al. (2023), numerical pathologies in EI-family acquisitions: https://arxiv.org/abs/2310.20708

## Batch 11 (max 5 files)

- [X] `atlas/training/run_utils.py` - reviewed + optimized
- [X] `atlas/utils/reproducibility.py` - reviewed + optimized
- [X] `tests/unit/training/test_run_utils_manifest.py` - reviewed + optimized
- [X] `tests/unit/research/test_reproducibility.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 11 optimization goals

- Improve manifest determinism and strict JSON compatibility for reproducibility artifacts.
- Harden run-manifest merge behavior when prior manifest sections have corrupted types.
- Make global-seed helper robust to malformed/non-finite seed inputs and cross-platform hash-seed constraints.
- Add regression tests for strict serialization and seed normalization behavior.

## Batch 11 outcomes

- `run_utils.py`:

  - `_json_safe` now sanitizes non-finite floats (`NaN/Inf` -> `null`) and emits deterministic ordering for dict/set payloads,
  - replaced eager file hashing with chunked SHA256 streaming to reduce peak memory on large manifests/locks,
  - added `_ensure_manifest_section` to repair malformed manifest sections during merge (`list/str` -> `{}`),
  - simplified manifest writing pipeline to a single canonical payload write pass (JSON + YAML mirror), with deterministic key ordering and `allow_nan=False`.
- `reproducibility.py`:

  - added robust seed coercion to uint32 domain, including non-finite input fallback,
  - normalized `PYTHONHASHSEED` handling and surfaced original input in metadata (`seed_input`),
  - runtime metadata now includes `python_hash_seed`.
- Tests:

  - `test_run_utils_manifest.py` now covers non-finite serialization sanitization, deterministic set ordering, and corrupted-section repair on merge,
  - `test_reproducibility.py` now covers non-finite seed fallback and negative seed normalization.

## Research references used in batch 11

- Python `json` module docs (`allow_nan` and standard-compliant output): https://docs.python.org/3/library/json.html
- Python `PYTHONHASHSEED` docs: https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
- Python `random.seed` docs: https://docs.python.org/3/library/random.html#random.seed
- NumPy `random.seed` docs: https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
- NIST FIPS 180-4 (SHA-256 standard): https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf

## Batch 12 (max 5 files)

- [X] `atlas/data/data_validation.py` - reviewed + optimized
- [X] `atlas/data/source_registry.py` - reviewed + optimized
- [X] `tests/unit/data/test_data_validation.py` - reviewed + optimized
- [X] `tests/unit/data/test_source_registry.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 12 optimization goals

- Strengthen data-validation output interoperability by enforcing strict JSON-safe serialization.
- Harden source-registry reliability/correlation fusion against non-finite numeric inputs.
- Prevent duplicate detector from treating null IDs as a real duplicate key.
- Add regression coverage for non-finite guards and malformed-input recovery paths.

## Batch 12 outcomes

- `data_validation.py`:

  - added recursive `_json_safe` conversion for report serialization (`NaN/Inf -> null`, NumPy scalars -> Python scalars),
  - `ValidationReport.to_json` now emits deterministic strict JSON (`sort_keys=True`, `allow_nan=False`),
  - `check_duplicates` now ignores `None` IDs instead of collapsing them into `"None"` duplicate buckets.
- `source_registry.py`:

  - hardened Beta reliability stats (`mean/variance`) for degenerate/invalid priors,
  - `register` now sanitizes invalid reliability priors and rejects empty source keys,
  - `update_reliability` rejects non-finite updates,
  - drift-aware source scoring now guards non-finite drift inputs,
  - correlation/covariance normalization now sanitizes non-finite matrices,
  - residual-based correlation estimation filters non-finite residual traces,
  - GLS fusion now skips invalid estimates (non-finite value/std) and validates covariance denominator robustness.
- Tests:

  - `test_data_validation.py` now verifies null-ID duplicate handling and strict JSON non-finite sanitization.
  - `test_source_registry.py` now verifies invalid-prior sanitization, non-finite update rejection, finite-safe drift scoring, and invalid-estimate filtering in fusion.

## Research references used in batch 12

- Python `json` docs (`allow_nan=False` strictness): https://docs.python.org/3/library/json.html
- NumPy `isfinite` docs (finite filtering semantics): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy `linalg.pinv` docs (stable pseudo-inverse in GLS weighting): https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
- Ledoit & Wolf (2004), covariance shrinkage: https://doi.org/10.1016/S0047-259X(03)00096-4
- Ben-David et al. (2010), domain-shift theory: https://jmlr.org/papers/v10/ben-david09a.html

## Batch 13 (max 5 files)

- [X] `atlas/data/alloy_estimator.py` - reviewed + optimized
- [X] `atlas/data/property_estimator.py` - reviewed + optimized
- [X] `tests/unit/data/test_alloy_estimator.py` - reviewed + optimized
- [X] `tests/unit/data/test_property_estimator.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 13 optimization goals

- Harden alloy/property estimation pipelines against non-finite runtime inputs (`NaN/Inf`) in weights, densities, and uncertainty hyperparameters.
- Ensure Gaussian fusion uses only statistically valid observations (`value` finite and `sigma > 0`) and stays stable in correlated-GLS mode.
- Keep downstream search/summary utilities robust under malformed caller arguments.
- Add regression tests proving finite-safe behavior and fallback semantics.

## Batch 13 outcomes

- `alloy_estimator.py`:

  - `AlloyPhase.get` now enforces finite-safe scalar conversion and falls back cleanly for malformed properties.
  - added explicit finite coercion helpers for non-negative/positive inputs and normalized-fraction helper used across weighting paths.
  - `convert_wt_to_vol` and `_normalize_weight_fractions` now sanitize non-finite weight/density values before normalization.
  - Reuss/Wiener/entropy helpers now ignore invalid rows and return stable finite results even with dirty phase data.
  - `estimate_properties` now normalizes finite-safe `wf/vf` and guards density/thermal/melting channels from non-finite contamination.
  - `print_report` experimental comparison now skips invalid/non-positive experimental targets.
- `property_estimator.py`:

  - added finite-safe coercion for sigma/correlation/temperature/fallback-mass hyperparameters.
  - `_precision_fusion` validity mask now requires finite values plus finite positive sigmas.
  - correlated GLS branch now sanitizes covariance and adds small diagonal jitter for inversion stability.
  - `_normal_cdf` now handles non-finite z-scores deterministically via bounded `nan_to_num`.
  - `search` now validates `max_results`; `property_summary` now handles numeric-only describe output safely.
- Tests:

  - `test_alloy_estimator.py` now covers non-finite weight/density sanitization in volume conversion and custom-alloy normalization.
  - `test_property_estimator.py` now covers invalid hyperparameter sanitization, sigma-invalid fusion filtering, and `search` fallback limit behavior.

## Research references used in batch 13

- NumPy `isfinite` docs (finite-mask semantics): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy `nan_to_num` docs (deterministic NaN/Inf replacement): https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
- NumPy `linalg.pinv` docs (robust pseudo-inverse fallback for GLS): https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
- Ledoit & Wolf (2004), covariance conditioning rationale: https://doi.org/10.1016/S0047-259X(03)00096-4
- Anderson (1963), Debye temperature from elastic constants: https://doi.org/10.1016/0022-3697(63)90067-2

## Batch 14 (max 5 files)

- [X] `atlas/data/crystal_dataset.py` - reviewed + optimized
- [X] `atlas/data/jarvis_client.py` - reviewed + optimized
- [X] `tests/unit/data/test_crystal_dataset.py` - reviewed + optimized
- [X] `tests/unit/data/test_jarvis_client.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 14 optimization goals

- Harden data-pipeline split/sampling primitives against malformed runtime inputs (invalid split names, non-finite ratios/features, dirty manifest rows).
- Improve robustness of probabilistic stability/topology scoring when uncertainty hyperparameters contain NaN/Inf or invalid ranges.
- Preserve deterministic coreset behavior even when feature tensors include non-finite values.
- Add regression tests for newly hardened edge cases.

## Batch 14 outcomes

- `crystal_dataset.py`:

  - added explicit split validation (`train/val/test`) and early split-ratio schema checks (finite, non-negative, sum>0),
  - improved formula fallback parser to preserve stoichiometric counts (not symbol-only token counting),
  - hardened k-center coreset routine with 2D shape validation + `nan_to_num` sanitization,
  - manifest assignment loader now filters invalid rows (`sample_id` missing, split not in train/val/test),
  - strengthened constructor guards for `max_samples`, `graph_cutoff`, and non-integer `min_labeled_properties`.
- `jarvis_client.py`:

  - added finite-safe coercion helpers for non-negative, positive, and probability-bounded scalars,
  - `_normal_cdf` now handles non-finite z-scores deterministically,
  - ehull noise estimation now sanitizes `base_noise` and adaptive slope inputs,
  - k-center selection path now sanitizes non-finite feature matrices before distance updates,
  - `_sample_dataframe` now validates strategy first and handles non-positive sample count safely,
  - `get_stable_materials` and `get_topological_materials` now sanitize key hyperparameters (`ehull_max`, `noise`, `prob thresholds`, `fusion weights`, calibration temperature) before scoring.
- Tests:

  - `test_crystal_dataset.py` now covers invalid split/split_ratio rejection, non-finite k-center feature handling, and manifest assignment row filtering.
  - `test_jarvis_client.py` now covers invalid probabilistic parameter sanitization, topology fusion sanitization, and k-center sampling stability under non-finite feature inputs.

## Research references used in batch 14

- Gonzalez (1985), metric k-center farthest-first approximation: https://doi.org/10.1016/0304-3975(85)90224-5
- Sener & Savarese (2018), core-set active learning intuition: https://arxiv.org/abs/1708.00489
- Sun et al. (2016), metastability/Ehull scale in inorganic materials: https://doi.org/10.1126/sciadv.1600225
- NumPy `nan_to_num` docs (finite sanitization semantics): https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
- NumPy `isfinite` docs (finite mask semantics): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html

## Batch 15 (max 5 files)

- [X] `atlas/models/prediction_utils.py` - reviewed + optimized
- [X] `atlas/models/uncertainty.py` - reviewed + optimized
- [X] `tests/unit/models/test_prediction_utils.py` - reviewed + optimized
- [X] `tests/unit/models/test_uncertainty.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 15 optimization goals

- Harden prediction normalization utilities against malformed uncertainty payloads (`NaN/Inf`, missing std, uninspectable signatures).
- Remove avoidable UQ numerical pathologies (single-member/single-sample std producing unstable values, non-finite ensemble predictions).
- Improve MC-dropout runtime correctness by restoring model train/eval state after stochastic inference.
- Add regression tests for numeric stability and API edge-case behavior.

## Batch 15 outcomes

- `prediction_utils.py`:

  - added `_sanitize_std_like` to enforce finite, non-negative std tensors across tuple/dict payload formats,
  - evidential payload path now sanitizes non-finite `nu/alpha/beta` before variance composition to avoid unstable division,
  - tuple/list payloads with `None` std are now handled explicitly as deterministic outputs,
  - `forward_graph_model` now tolerates `inspect.signature` failures and falls back safely.
- `uncertainty.py`:

  - added unified payload normalization supporting both dict and tensor model outputs,
  - constructor guards now reject invalid `n_models` / `n_samples`,
  - ensemble/MC std now uses stable population estimator (`unbiased=False`) and sanitizes non-finite prediction stacks,
  - MC dropout now enables all dropout variants (`_DropoutNd`) and restores original model training state after inference,
  - evidential regression `total_std` and loss paths now sanitize non-finite values and clamp unsafe logarithm denominators.
- Tests:

  - `test_prediction_utils.py` now covers `None` std tuples, non-finite std sanitization, evidential non-finite payload stability, uninspectable signature fallback, and missing-edge-feature error path.
  - `test_uncertainty.py` now covers invalid constructor arguments, single-member/sample std stability, tensor-output compatibility, MC state restoration, and evidential loss robustness under non-finite targets.

## Research references used in batch 15

- PyTorch `torch.std` docs (Bessel correction / population std semantics): https://docs.pytorch.org/docs/stable/generated/torch.std.html
- PyTorch `torch.nan_to_num` docs (finite-value replacement semantics): https://docs.pytorch.org/docs/stable/generated/torch.nan_to_num.html
- Lakshminarayanan et al. (NeurIPS 2017), deep ensemble uncertainty: https://papers.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles
- Gal & Ghahramani (ICML 2016), MC dropout as approximate Bayesian inference: https://proceedings.mlr.press/v48/gal16.html
- Amini et al. (NeurIPS 2020), deep evidential regression: https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html

## Batch 16 (max 5 files)

- [X] `atlas/models/utils.py` - reviewed + optimized
- [X] `atlas/models/graph_builder.py` - reviewed + optimized
- [X] `tests/unit/models/test_model_utils.py` - reviewed + optimized
- [X] `tests/unit/models/test_structure_expansion.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 16 optimization goals

- Harden checkpoint loading and state-dict key normalization to avoid silent key-collision corruption.
- Improve graph construction robustness for malformed numerical inputs (non-finite distances, invalid Gaussian basis settings, zero-neighbor structures).
- Add stronger parameter validation at graph-builder/model-loading boundaries for earlier deterministic failures.
- Add regression tests for collision detection, malformed payload handling, and graph fallback behavior.

## Batch 16 outcomes

- `models/utils.py`:

  - added collision-safe state-dict key normalization with explicit error on conflicting normalized keys,
  - strengthened CGCNN config inference with tensor-rank validation for critical weights,
  - normalizer loading now validates finite numeric `(mean, std)` and rejects malformed normalizer payloads,
  - phase1/phase2 checkpoint loaders now fail early on invalid checkpoint payload structure (`model_state_dict` non-dict).
- `models/graph_builder.py`:

  - `gaussian_expansion` now validates `cutoff/n_gaussians`, sanitizes non-finite distances, and supports `n_gaussians=1` safely,
  - `CrystalGraphBuilder` constructor now validates finite positive `cutoff` and positive `max_neighbors`,
  - `element_features` no longer maps unknown elements to Hydrogen one-hot by default,
  - `structure_to_graph` now rejects empty structures and uses per-node self-loop fallback when no neighbors are found,
  - edge vectors and emitted PyG tensors are now explicitly sanitized/typed (`float32`/`long`) for stable downstream use.
- Tests:

  - `test_model_utils.py` now covers normalized-key collision errors, malformed normalizer payload fallback, and invalid phase2 state-dict payload rejection.
  - `test_structure_expansion.py` now covers single-basis Gaussian expansion, non-finite distance sanitization, invalid parameter rejection, graph-builder init validation, per-node self-loop fallback, and empty-structure guard.

## Research references used in batch 16

- PyTorch `torch.load` docs (checkpoint payload semantics): https://docs.pytorch.org/docs/stable/generated/torch.load.html
- PyTorch Geometric `Data` docs (graph tensor schema expectations): https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html
- pymatgen `Structure.get_all_neighbors` docs (neighbor construction API): https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure.get_all_neighbors
- Xie & Grossman (2018), CGCNN representation assumptions: https://doi.org/10.1103/PhysRevLett.120.145301
- Schütt et al. (2018), radial/Gaussian distance basis in atomistic GNNs: https://doi.org/10.1063/1.5019779

## Batch 17 (max 5 files)

- [X] `atlas/models/cgcnn.py` - reviewed + optimized
- [X] `atlas/models/multi_task.py` - reviewed + optimized
- [X] `tests/unit/models/test_cgcnn.py` - reviewed + optimized
- [X] `tests/unit/models/test_multi_task.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 17 optimization goals

- Strengthen CGCNN constructor/input guards to fail fast on malformed runtime tensors and invalid hyperparameters.
- Improve MultiTaskGNN task-schema validation and encoder-kwargs passthrough robustness under introspection failure.
- Remove subtle task-head consistency risks (unknown task types, duplicate task registration, invalid tensor types).
- Add regression tests for edge-case failures and new validation behavior.

## Batch 17 outcomes

- `cgcnn.py`:

  - added explicit numeric validation for model dimensions/layer counts and dropout range,
  - added `_validate_graph_inputs` with shape checks for `node_feats/edge_index/edge_feats/batch`,
  - added safe `batch` casting to `long` during pooling path,
  - attention pooling logits now use `torch.nan_to_num` before sparse softmax to reduce NaN/Inf propagation risk.
- `multi_task.py`:

  - added global task/type schema constants and validation (`scalar/evidential/tensor` and tensor subtype checks),
  - `TensorHead` now strictly validates `tensor_type`; removed duplicate conditional branch in tensor reconstruction,
  - `MultiTaskGNN` constructor now validates task config structure and rejects unknown task types early,
  - forward path now normalizes `tasks` input (`str` or list), ignores non-dict `encoder_kwargs`, and gracefully falls back when `inspect.signature` raises,
  - `add_task` now validates task type and rejects duplicate task registration.
- Tests:

  - `test_cgcnn.py` now covers invalid hyperparameter rejection and graph-input shape mismatch guards.
  - `test_multi_task.py` now covers unknown task/tensor type rejection, single-task string selection, duplicate add-task rejection, invalid add-task type rejection, and signature-failure fallback behavior.

## Research references used in batch 17

- Xie & Grossman (2018), CGCNN baseline architecture: https://doi.org/10.1103/PhysRevLett.120.145301
- Xu et al. (2018), Jumping Knowledge Networks motivation for multi-depth aggregation: https://arxiv.org/abs/1806.03536
- PyTorch Geometric `global_add_pool` docs (graph pooling contract): https://pytorch-geometric.readthedocs.io/en/2.5.2/generated/torch_geometric.nn.pool.global_add_pool.html
- PyTorch `torch.nan_to_num` docs (finite sanitization semantics): https://docs.pytorch.org/docs/stable/generated/torch.nan_to_num.html
- Python `inspect.signature` docs (introspection failure semantics): https://docs.python.org/3/library/inspect.html#inspect.signature

## Batch 18 (max 5 files)

- [X] `atlas/topology/classifier.py` - reviewed + optimized
- [X] `atlas/utils/structure.py` - reviewed + optimized
- [X] `tests/unit/topology/test_classifier.py` - reviewed + optimized
- [X] `tests/unit/models/test_structure_utils.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 18 optimization goals

- Eliminate topology-classifier pooling/probability edge-case failures (invalid scatter-mean semantics, multi-graph probability misuse, MC dropout state leakage).
- Strengthen runtime input validation for topology forward/proba paths to fail fast on malformed graph tensors.
- Improve structure utility robustness for empty structures and deterministic feature extraction.
- Add targeted regression tests for new guards and deterministic behavior.

## Batch 18 outcomes

- `topology/classifier.py`:

  - added strict constructor validation for dimensions/layer count/dropout domain,
  - added `_validate_inputs` for node/edge/batch shape consistency,
  - fixed pooling bias by setting `include_self=False` in `scatter_reduce_(reduce="mean")`,
  - added batch index safety checks (non-empty/non-negative),
  - `predict_proba` now validates single-graph contract, validates `n_samples`, and restores model train/eval state after MC dropout inference.
- `utils/structure.py`:

  - added deterministic site subsampling helper for nearest-neighbor feature estimation (removed stochastic `np.random.choice` variability),
  - `get_element_info` now handles empty structures safely and yields stable sorted element ordering,
  - `compute_structural_features` now handles empty structures with explicit zero/unknown fallback payloads,
  - sanitized scalar outputs (`float` conversions) for consistent serialization and downstream typing.
- Tests:

  - `test_classifier.py` now covers invalid hyperparameter rejection, multi-graph `predict_proba` rejection, MC dropout training-state restoration, and shape-mismatch validation.
  - `test_structure_utils.py` now covers empty-structure element/features handling and deterministic large-structure feature extraction.

## Research references used in batch 18

- PyTorch `Tensor.scatter_reduce_` docs (include_self semantics): https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html
- Gal & Ghahramani (ICML 2016), MC dropout Bayesian approximation rationale: https://proceedings.mlr.press/v48/gal16.html
- pymatgen `SpacegroupAnalyzer` docs (structure standardization): https://pymatgen.org/pymatgen.symmetry.html#pymatgen.symmetry.analyzer.SpacegroupAnalyzer
- pymatgen `Structure.get_neighbors` docs (local-neighbor extraction): https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure.get_neighbors
- spglib docs (symmetry detection backend used by pymatgen): https://spglib.readthedocs.io/

## Batch 19 (max 5 files)

- [X] `atlas/training/losses.py` - reviewed + optimized
- [X] `atlas/training/trainer.py` - reviewed + optimized
- [X] `tests/unit/training/test_losses.py` - reviewed + optimized
- [X] `tests/unit/training/test_trainer.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 19 optimization goals

- Harden training-loss paths against non-finite values and malformed configuration (loss type, task type, weighting strategy).
- Make trainer loop fail-fast for invalid runtime arguments and malformed loss outputs instead of silently continuing.
- Reduce hidden runtime coupling by narrowing fallback behavior (`TypeError`-only signature fallback).
- Add regression coverage to lock in the new guardrails under CI.

## Batch 19 outcomes

- `training/losses.py`:

  - added explicit supported-set validation for property loss types, multi-task strategies, and task types.
  - added finite-safe scalar validation for `constraint_weight`, `coeff`, and fixed task weights.
  - property loss now masks non-finite `pred/target` pairs (not only `NaN` targets) and validates unknown constraints explicitly.
  - evidential loss now validates required keys, skips invalid distribution rows (`nu/alpha/beta` domain + finite checks), and filters non-finite loss terms before reduction.
  - multi-task loss now validates empty task schema, infers device safely when prediction dict is sparse/empty, and skips non-finite per-task losses.
- `training/trainer.py`:

  - added finite non-negative validation for `grad_clip_norm` and `min_delta`.
  - `inspect.signature` failure is now handled gracefully with empty forward-param set fallback.
  - narrowed forward fallback exception scope to `TypeError` so runtime tensor/shape bugs are not silently swallowed.
  - `_compute_loss` now raises explicit error when a loss function returns an empty dict.
  - `fit` now validates `n_epochs/patience/checkpoint_name`, tracks last epoch/loss explicitly, and writes history with UTF-8 encoding.
- Tests:

  - `test_losses.py` now covers invalid config rejection, non-finite prediction masking, evidential missing-key/invalid-row behavior, and schema validation for `MultiTaskLoss`.
  - `test_trainer.py` now covers invalid grad-clip rejection, signature introspection fallback, invalid fit-arg rejection, and empty-loss-dict error path.
  - Verified with:

    - `python -m ruff check atlas/training/losses.py atlas/training/trainer.py tests/unit/training/test_losses.py tests/unit/training/test_trainer.py`
    - `python -m pytest tests/unit/training/test_losses.py tests/unit/training/test_trainer.py -q`
    - `python -m pytest tests/unit/training -q`

## Research references used in batch 19

- Kendall, Gal, Cipolla (CVPR 2018), uncertainty weighting for multi-task learning: https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html
- Amini et al. (NeurIPS 2020), deep evidential regression: https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html
- PyTorch `torch.nn.utils.clip_grad_norm_` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- PyTorch `torch.isfinite` docs: https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- Python `inspect.signature` docs: https://docs.python.org/3/library/inspect.html#inspect.signature

## Batch 20 (max 5 files)

- [X] `atlas/training/run_utils.py` - reviewed + optimized
- [X] `atlas/utils/reproducibility.py` - reviewed + optimized
- [X] `tests/unit/training/test_run_utils_manifest.py` - reviewed + optimized
- [X] `tests/unit/research/test_reproducibility.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 20 optimization goals

- Strengthen run-manifest generation under CI/runtime edge conditions (slow git commands, public-visibility redaction completeness, lock parsing).
- Prevent public manifests from leaking local absolute paths in `environment_lock`/default artifact fields.
- Improve deterministic-training controls so `deterministic=False` can explicitly disable deterministic mode after a prior deterministic run.
- Add regression tests for new manifest privacy and reproducibility guarantees.

## Batch 20 outcomes

- `training/run_utils.py`:

  - added hard timeout for git metadata subprocess calls to avoid manifest-generation hangs.
  - hardened strict-lock parsing to accept bool-like string values (`"true"/"1"/"yes"/"on"`).
  - switched default manifest artifact pointers to portable relative filenames (`run_manifest.json`, `run_manifest.yaml`) instead of absolute paths.
  - public visibility path redaction now explicitly re-sanitizes all core manifest sections (`dataset/split/environment_lock/artifacts/metrics/seeds/configs`) to prevent merge-existing path leakage.
  - `environment_lock` block now applies visibility-aware redaction for both default lock metadata and user-supplied overrides.
- `utils/reproducibility.py`:

  - deterministic policy now supports both enable and disable flows via `torch.use_deterministic_algorithms(deterministic_requested, warn_only=True)`.
  - deterministic mode now sets `CUBLAS_WORKSPACE_CONFIG` default when requested (per PyTorch reproducibility guidance).
  - added `deterministic_enabled` metadata to both seed-setting and runtime metadata paths.
  - cuDNN flags now switch consistently with deterministic request (`benchmark = not deterministic`, `deterministic = deterministic`).
- Tests:

  - `test_run_utils_manifest.py` now verifies public redaction covers `environment_lock.lock_file` and validates relative default artifact paths.
  - added strict-lock string parsing regression test (`strict_lock="true"`).
  - `test_reproducibility.py` now verifies deterministic metadata presence, cuBLAS workspace config behavior, and deterministic-algorithm toggle (`True -> False`) correctness.
  - Verified with:

    - `python -m ruff check atlas/training/run_utils.py atlas/utils/reproducibility.py tests/unit/training/test_run_utils_manifest.py tests/unit/research/test_reproducibility.py`
    - `python -m pytest tests/unit/training/test_run_utils_manifest.py tests/unit/research/test_reproducibility.py -q`

## Research references used in batch 20

- PyTorch reproducibility notes (deterministic algorithms + `CUBLAS_WORKSPACE_CONFIG`): https://pytorch.org/docs/stable/notes/randomness.html
- Python `subprocess.run` docs (`timeout` behavior): https://docs.python.org/3/library/subprocess.html#subprocess.run
- NumPy random seeding reference: https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
- Sandve et al. (2013), Ten Simple Rules for Reproducible Computational Research: https://doi.org/10.1371/journal.pcbi.1003285
- ACM Artifact Review and Badging v1.1 (reproducibility evidence expectations): https://www.acm.org/publications/policies/artifact-review-and-badging-current

## Batch 21 (max 5 files)

- [X] `scripts/phase5_active_learning/run_phase5.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_discovery.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_phase5_cli.py` - reviewed + optimized
- [X] `ruff.toml` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 21 optimization goals

- Strengthen Phase5/Discovery CLI input validation against non-finite numeric values and unsafe run identifiers.
- Prevent accidental path traversal style run-id payloads from reaching run-directory resolution logic.
- Expand CLI regression tests to lock the new guardrails in CI.
- Improve Ruff CI behavior consistency so explicit path lint invocations still respect exclude policy.

## Batch 21 outcomes

- `run_phase5.py`:

  - added finite-safe validation helper for acquisition numeric options (`kappa/jitter`) and finite check for `acq_best_f`.
  - added strict `run_id` safety check (disallow path separators and traversal-style values like `..`).
  - added guard for non-negative `preflight_split_seed`.
- `run_discovery.py`:

  - strengthened acquisition validation: `acq_kappa` and `acq_jitter` must be finite and non-negative.
  - added `run_id` safety validation and `results_dir` non-empty string guard.
- `test_phase5_cli.py`:

  - expanded argument-validation tests for non-finite floats (`NaN/Inf`) and unsafe run-id patterns (`../escape`, `..\\escape`).
  - updated discovery validation fixture payload to include optional fields covered by new guards.
- `ruff.toml`:

  - enabled `force-exclude = true` so direct-path CI lint invocations still honor configured excludes.
  - added explicit `extend-exclude` for cache/artifact/data directories to reduce lint noise and avoid accidental non-source lint scans.
- Verified with:

  - `python -m ruff check scripts/phase5_active_learning/run_phase5.py scripts/phase5_active_learning/run_discovery.py tests/unit/active_learning/test_phase5_cli.py`
  - `python -m pytest tests/unit/active_learning/test_phase5_cli.py -q`

## Research references used in batch 21

- Ruff configuration docs (`extend-exclude`, `force-exclude`): https://docs.astral.sh/ruff/configuration/
- Python `math.isfinite` docs: https://docs.python.org/3/library/math.html#math.isfinite
- Python `argparse` docs (type parsing and validation extension points): https://docs.python.org/3/library/argparse.html
- OWASP Path Traversal reference: https://owasp.org/www-community/attacks/Path_Traversal
- ACM Artifact Review and Badging v1.1 (repeatability and CI evidence expectations): https://www.acm.org/publications/policies/artifact-review-and-badging-current

## Batch 22 (max 5 files)

- [X] `scripts/phase5_active_learning/active_learning.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/structure_enumerator.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_phase6_active_learning.py` - added + optimized
- [X] `tests/unit/active_learning/test_structure_enumerator_script.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 22 optimization goals

- Harden Phase6 acquisition/Pareto helpers against non-finite numeric inputs and malformed argument shapes.
- Make active-learning loop state transitions safer (budget bounds, strategy validation, MC-dropout state restoration).
- Strengthen fallback structure enumerator input validation and incomplete-structure filtering behavior.
- Add dedicated unit tests for script-level algorithms that previously had no CI regression coverage.

## Batch 22 outcomes

- `active_learning.py`:

  - added explicit input validators for `batch_size`, acquisition shapes, and finite `best_so_far`.
  - `acquisition_uncertainty` now sanitizes NaN/Inf std values and handles zero/oversized batch requests safely.
  - `acquisition_expected_improvement` now enforces shape consistency, finite reference value, zero-variance fallback, and finite EI outputs.
  - `acquisition_random` now validates pool/query bounds and returns deterministic `default_rng` selections.
  - `ActiveLearningLoop` now validates strategy and budget parameters up front, clamps initial sampling to dataset size, and handles empty datasets gracefully.
  - `_predict_with_dropout` now validates `n_samples`, uses numerically stable std (`unbiased=False`), sanitizes outputs, and restores model/dropout training states after MC inference.
  - `pareto_frontier` now validates 2D objective shape, checks `maximize` length, filters non-finite rows, and maps Pareto indices back to original rows.
  - `multi_objective_screening` now validates non-empty objectives/positive `top_k`, handles empty prediction matrices, and uses zero-span-safe normalization.
- `structure_enumerator.py`:

  - added type checks for `base_structure` and substitution payload schema.
  - added `max_index > 0` guard and normalized substitution parsing helper.
  - `remove_incomplete=True` now actively filters DummySpecies-containing structures.
  - improved combinatorial-cap logging and implemented `_build_constraints` to expose variant-space diagnostics.
- Tests:

  - added `test_phase6_active_learning.py` for acquisition finite-safety, Pareto filtering/index mapping, and AL loop init validation.
  - added `test_structure_enumerator_script.py` for substitution schema validation, incomplete-structure filtering, and constraint-space diagnostics.
  - Verified with:

    - `python -m ruff check scripts/phase5_active_learning/active_learning.py scripts/phase5_active_learning/structure_enumerator.py tests/unit/active_learning/test_phase6_active_learning.py tests/unit/active_learning/test_structure_enumerator_script.py`
    - `python -m pytest tests/unit/active_learning/test_phase6_active_learning.py tests/unit/active_learning/test_structure_enumerator_script.py -q`

## Research references used in batch 22

- Jones, Schonlau, Welch (1998), Efficient Global Optimization / EI: https://doi.org/10.1023/A:1008306431147
- Deb et al. (2002), NSGA-II non-dominated sorting baseline: https://doi.org/10.1109/4235.996017
- SciPy `scipy.stats.norm` API docs (EI CDF/PDF terms): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
- pymatgen `StructureMatcher` docs (`group_structures` symmetry-equivalence dedup): https://pymatgen.org/pymatgen.analysis.structure_matcher.html
- NumPy `nan_to_num` docs (finite-sanitization behavior): https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html

## Batch 23 (max 5 files)

- [X] `scripts/phase5_active_learning/search_materials.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/test_enumeration.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_search_materials_cli.py` - added + optimized
- [X] `tests/unit/active_learning/test_test_enumeration_script.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 23 optimization goals

- Make `search_materials` CLI fail fast on malformed inputs before heavy dataset loading.
- Improve robustness of custom filter parsing and criteria composition (bound merging, finite validation).
- Refactor enumeration demo script into reusable/testable helpers instead of print-only one-off flow.
- Add script-level unit tests for both search and enumeration demos to keep these user-facing entrypoints stable under CI.

## Batch 23 outcomes

- `search_materials.py`:

  - added strict argument validation (`_validate_args`) for preset conflicts, finite numeric constraints, range consistency, and EM bounds.
  - added robust custom-filter parser (`<`, `<=`, `>`, `>=`) with explicit syntax/finite checks.
  - extracted criteria assembly into `_build_criteria` with bound merging logic for repeated filters.
  - extracted deterministic display-column selection helper with deduplication.
  - `main()` now exits with explicit status codes and validates inputs before initializing JARVIS clients.
- `test_enumeration.py`:

  - converted to reusable demo utilities (`_resolve_enumerator_class`, `run_demo`) plus CLI args (`--skip-vacancy`, `--quiet`).
  - added structured summary return payload to support automation and regression tests.
  - switched to explicit `SystemExit(main())` convention for predictable script exit handling.
- Tests:

  - added `test_search_materials_cli.py` to cover argument guardrails, custom filter parsing, criteria merging, and display-column dedup behavior.
  - added `test_test_enumeration_script.py` to cover demo helper execution with injected dummy enumerator and vacancy branch behavior.
  - Verified with:

    - `python -m ruff check scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/test_enumeration.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_test_enumeration_script.py`
    - `python -m pytest tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_test_enumeration_script.py -q`

## Research references used in batch 23

- Python `argparse` docs (CLI validation strategy): https://docs.python.org/3/library/argparse.html
- NumPy `nan_to_num` docs (finite sanitization): https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
- pandas display/options docs (deterministic tabular rendering): https://pandas.pydata.org/docs/user_guide/options.html
- pymatgen `StructureMatcher` docs (symmetry-aware dedup): https://pymatgen.org/pymatgen.analysis.structure_matcher.html
- Hart & Forcade (2008), derivative structure enumeration foundations: https://doi.org/10.1107/S0108767308027336

## Batch 24 (max 5 files)

- [X] `setup.py` - reviewed + optimized
- [X] `pyproject.toml` - reviewed + optimized
- [X] `requirements-dev.txt` - reviewed + optimized
- [X] `tests/unit/config/test_packaging_metadata.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 24 optimization goals

- Improve packaging metadata completeness and CI/dev environment consistency.
- Align editable dev install flow with a single canonical extra to reduce environment drift.
- Keep legacy `setup.py` compatibility while making behavior explicit and side-effect-safe.
- Add unit-level metadata regression checks so packaging drift is caught in CI.

## Batch 24 outcomes

- `setup.py`:

  - converted shim to explicit `if __name__ == "__main__": setup()` guard to avoid accidental side effects on import while keeping legacy tooling compatibility.
- `pyproject.toml`:

  - added richer project metadata (`classifiers`, `project.urls`) for package index discoverability and downstream tooling compatibility.
  - added `dev` optional dependency group consolidating test/notebook/lint/build tooling.
  - added `tool.setuptools.include-package-data = false` for explicit packaging behavior.
- `requirements-dev.txt`:

  - switched editable install target from `.[test,jupyter]` to `.[dev]` for one canonical development environment entrypoint.
- Tests:

  - added `test_packaging_metadata.py` to validate:

    - `dev` extra presence and key tooling dependencies,
    - `requirements-dev` points to `-e .[dev]`,
    - `setup.py` remains a pyproject-backed guarded shim.
  - Verified with:

    - `python -m ruff check setup.py tests/unit/config/test_packaging_metadata.py`
    - `python -m pytest tests/unit/config/test_packaging_metadata.py tests/unit/config/test_config.py -q`

## Research references used in batch 24

- PEP 621 (canonical `pyproject.toml` project metadata): https://peps.python.org/pep-0621/
- Setuptools `pyproject.toml` configuration guide: https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html
- Setuptools dependency management (`dependencies` / `optional-dependencies`): https://setuptools.pypa.io/en/stable/userguide/dependency_management.html
- pip local/editable installs guide: https://pip.pypa.io/en/stable/topics/local-project-installs/
- PyPA Packaging User Guide overview: https://packaging.python.org/en/latest/

## Batch 25 (max 5 files)

- [X] `atlas/training/preflight.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_phase5.py` - reviewed + optimized
- [X] `tests/unit/training/test_preflight.py` - added + optimized
- [X] `tests/unit/active_learning/test_phase5_cli.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 25 optimization goals

- Harden preflight gates to fail fast on invalid input and avoid indefinite subprocess hangs.
- Improve preflight diagnostics with explicit failure reason propagation.
- Expose preflight timeout control at Phase5 CLI layer and validate it before execution.
- Add dedicated unit tests for preflight orchestration edge cases (timeout/failure/missing artifacts/success path).

## Batch 25 outcomes

- `training/preflight.py`:

  - added strict input validation for `project_root`, `property_group`, `max_samples`, `split_seed`, and `timeout_sec`.
  - added bounded subprocess execution helper (`_run_command`) with timeout handling (`TimeoutExpired -> return code 124`).
  - centralized required split-manifest filenames in `_REQUIRED_SPLIT_MANIFESTS`.
  - expanded `PreflightResult` with `error_message` for clearer upstream diagnostics.
  - all failure exits now return structured error categories (`validate-data failed`, `make-splits failed`, `missing split manifests`).
- `run_phase5.py`:

  - added `--preflight-timeout-sec` argument (default `1800`) and validation guard (`>0`).
  - wired `timeout_sec` through to `run_preflight(...)`.
- Tests:

  - new `test_preflight.py` covers:

    - dry-run path creation,
    - argument validation failures,
    - command-timeout handling,
    - missing-manifest failure,
    - full success path with emitted manifests.
  - updated `test_phase5_cli.py` to include `preflight_timeout_sec` fields and added invalid-timeout guard test.
  - Verified with:

    - `python -m ruff check atlas/training/preflight.py scripts/phase5_active_learning/run_phase5.py tests/unit/active_learning/test_phase5_cli.py tests/unit/training/test_preflight.py`
    - `python -m pytest tests/unit/training/test_preflight.py tests/unit/active_learning/test_phase5_cli.py -q`

## Research references used in batch 25

- Python `subprocess.run` timeout semantics (official docs): https://docs.python.org/3/library/subprocess.html
- Python `argparse` docs (CLI argument validation patterns): https://docs.python.org/3/library/argparse.html
- The Tail at Scale (Dean & Barroso, 2013) DOI: https://doi.org/10.1145/2408776.2408794
- PEP 324 (`subprocess` design rationale): https://peps.python.org/pep-0324/
- Google Research publication page for Tail at Scale: https://research.google/pubs/the-tail-at-scale/

## Batch 26 (max 5 files)

- [X] `atlas/training/checkpoint.py` - reviewed + optimized
- [X] `atlas/training/filters.py` - reviewed + optimized
- [X] `tests/unit/training/test_checkpoint.py` - added + optimized
- [X] `tests/unit/training/test_filters.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 26 optimization goals

- Harden checkpoint retention logic against malformed hyperparameters and numeric edge cases (`NaN` metrics, negative epochs).
- Make rotating checkpoint behavior correct for small `keep_last_k` settings (especially `keep_last_k=1`).
- Improve outlier filtering robustness under non-finite/missing values and provide robust-statistics option.
- Add dedicated training utility tests for checkpointing and outlier filtering to lock behavior in CI.

## Batch 26 outcomes

- `training/checkpoint.py`:

  - added constructor validation for `top_k` and `keep_last_k` (>0).
  - `save_best` now validates finite `mae` and non-negative `epoch`.
  - added early reject path when candidate cannot enter top-k, avoiding unnecessary checkpoint writes.
  - stabilized best-model ranking with deterministic tie-break (`(mae, epoch)`).
  - fixed checkpoint rotation edge case for `keep_last_k=1` (no invalid `checkpoint_prev_0.pt` path).
  - ensured checkpoint payload always contains `epoch` when absent in state.
- `training/filters.py`:

  - added argument guards for finite positive `n_sigma` and supported methods.
  - added robust scalar extraction that skips non-scalar, non-finite, or malformed property values.
  - added `method` option: `"zscore"` (default) and `"modified_zscore"` (MAD-based robust alternative).
  - improved per-property iteration with deduplicated property list and explicit outlier metadata (`method`, `scale`).
  - preserved backward-compatible default behavior (`method="zscore"`).
- Tests:

  - new `test_checkpoint.py` covers constructor/input validation, top-k retention, best pointer update, rotation behavior, and `keep_last_k=1`.
  - new `test_filters.py` covers z-score filtering + CSV export, malformed/non-finite value handling, modified-zscore detection, and argument validation.
  - Verified with:

    - `python -m ruff check atlas/training/checkpoint.py atlas/training/filters.py tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py`
    - `python -m pytest tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py -q`

## Research references used in batch 26

- SciPy `median_abs_deviation` docs (robust dispersion for outlier handling): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html
- Iglewicz & Hoaglin (1993), outlier detection using robust z-scores: https://books.google.com/books/about/How_to_Detect_and_Handle_Outliers.html?id=FuuiEAAAQBAJ
- PyTorch `torch.save` docs: https://docs.pytorch.org/docs/stable/generated/torch.save.html
- PyTorch serialization semantics note: https://docs.pytorch.org/docs/stable/notes/serialization.html
- Python `pathlib` docs (path-safe checkpoint handling): https://docs.python.org/3/library/pathlib.html

## Batch 27 (max 5 files)

- [X] `atlas/training/metrics.py` - reviewed + optimized
- [X] `atlas/training/normalizers.py` - reviewed + optimized
- [X] `tests/unit/training/test_metrics.py` - added + optimized
- [X] `tests/unit/training/test_normalizers.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 27 optimization goals

- Make metric utilities numerically stable under NaN/Inf, empty intersections, single-class ROC-AUC edge cases, and degenerate eigenvalue statistics.
- Harden target normalizer fitting/loading paths against empty datasets, malformed state payloads, and non-finite scalar values.
- Add targeted unit tests for scalar/classification/tensor metrics and normalizer state lifecycle so CI catches regressions immediately.

## Batch 27 outcomes

- `training/metrics.py`:

  - validated paired finite input extraction for scalar/classification metrics and retained 0-safe behavior when no valid pairs exist.
  - hardened classification AUC fallback: now explicitly handles single-class labels and non-finite AUC results (`NaN` -> `0.5`).
  - improved tensor eigenvalue agreement stability for constant-spectrum cases by avoiding undefined Spearman correlations and using deterministic fallback scores.
- `training/normalizers.py`:

  - added finite scalar extraction helper for heterogeneous sample payloads (`tensor.item()`, numeric values, malformed objects).
  - fixed empty/invalid dataset path to default `(mean=0, std=1)` instead of propagating `nan` statistics.
  - added `load_state_dict` schema/type/value validation and safe std fallback for invalid/non-positive scales.
  - added explicit missing-property errors in multi-target normalization with available-property hints.
- Tests:

  - added `test_metrics.py` covering non-finite filtering, prefix schema, single-class AUC fallback, tensor finite-row filtering, and eigenvalue-agreement edge handling.
  - added `test_normalizers.py` covering finite-only stat fitting, empty fallback, round-trip normalization, state validation, and multi-target dynamic state loading.
  - Verified with:

    - `python -m ruff check atlas/training/metrics.py atlas/training/normalizers.py tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py`
    - `python -m pytest tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py -q`
    - `python -m pytest tests/unit/training/test_losses.py tests/unit/training/test_trainer.py tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py -q`

## Research references used in batch 27

- scikit-learn `roc_auc_score` docs (single-class undefined behavior and API semantics): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
- scikit-learn classification metrics docs (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`): https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
- SciPy `spearmanr` docs (constant input / undefined correlation caveats): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
- PyTorch `torch.isfinite` docs (finite-mask semantics): https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- PyTorch `torch.linalg.eigvalsh` docs (symmetric/Hermitian eigenspectrum API): https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigvalsh.html
- Fawcett (2006), ROC analysis foundations (Pattern Recognition Letters): https://doi.org/10.1016/j.patrec.2005.10.010

## Batch 28 (max 5 files)

- [X] `atlas/training/run_utils.py` - reviewed + optimized
- [X] `atlas/training/trainer.py` - reviewed + optimized
- [X] `tests/unit/training/test_run_utils_manifest.py` - reviewed + optimized
- [X] `tests/unit/training/test_trainer.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 28 optimization goals

- Harden run directory/manifest utilities against path traversal style run ids, timestamp collisions, and seed metadata overwrite bugs.
- Improve trainer stability by making non-finite losses fail fast and fixing `patience=0` early-stopping semantics.
- Lock the above behavior with regression tests so CI can catch future drift.

## Batch 28 outcomes

- `training/run_utils.py`:

  - added strict run-id normalization/validation (`_normalize_run_name`) to reject path separators, traversal patterns, and unsafe characters.
  - added timestamp collision-resistant run directory creation (`_create_timestamped_run_dir`) with deterministic suffix fallback.
  - added schema-version validation (`non-empty string`).
  - fixed seed precedence bug: default seed inference now uses `setdefault`, so explicit `seeds_block` values are no longer overwritten by fallback defaults.
- `training/trainer.py`:

  - added finite scalar guard helper (`_coerce_finite_float`).
  - `train_epoch` now raises on non-finite batch loss before backward pass.
  - `fit` now validates `val_loss` is finite and correctly handles `patience=0`:

    - continue while improving,
    - stop on first non-improvement.
  - `_save_checkpoint` now validates `epoch >= 0` and finite `val_loss`.
  - `_save_history` now writes JSON with `allow_nan=False` and sanitizes non-finite history values to `null`.
- Tests:

  - `test_run_utils_manifest.py` now covers:

    - explicit seed preservation (no fallback overwrite),
    - path-traversal run-id rejection,
    - timestamp collision suffix behavior.
  - `test_trainer.py` now covers:

    - non-finite training loss fail-fast,
    - non-finite validation loss rejection,
    - checkpoint input validation,
    - `patience=0` improving/non-improving semantics.
  - Verified with:

    - `python -m ruff check atlas/training/run_utils.py atlas/training/trainer.py tests/unit/training/test_run_utils_manifest.py tests/unit/training/test_trainer.py`
    - `python -m pytest tests/unit/training/test_run_utils_manifest.py tests/unit/training/test_trainer.py -q`

## Research references used in batch 28

- OWASP Path Traversal reference (run-id/path safety): https://owasp.org/www-community/attacks/Path_Traversal
- Python `json` docs (`allow_nan` behavior): https://docs.python.org/3/library/json.html
- Python `pathlib` docs (`Path.name` semantics): https://docs.python.org/3/library/pathlib.html
- PyTorch `torch.isfinite` docs (non-finite value detection): https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- Prechelt (1998), early stopping trade-off analysis: https://doi.org/10.1007/3-540-49430-8_3

## Batch 29 (max 5 files)

- [X] `atlas/utils/reproducibility.py` - reviewed + optimized
- [X] `atlas/utils/structure.py` - reviewed + optimized
- [X] `tests/unit/research/test_reproducibility.py` - reviewed + optimized
- [X] `tests/unit/models/test_structure_utils.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 29 optimization goals

- Strengthen reproducibility controls for seed parsing and deterministic-mode toggling across Python/NumPy/PyTorch call sites.
- Improve structure utility robustness for oxidation-state/disordered compositions and non-finite geometric values.
- Add CI guard tests for newly hardened edge cases.

## Batch 29 outcomes

- `utils/reproducibility.py`:

  - added `_coerce_bool` to robustly parse bool-like inputs (including CLI-style strings such as `"false"`/`"true"`).
  - improved `_coerce_seed` parsing to support base-prefixed strings (e.g. hex seeds like `"0x10"`).
  - added `_enable_torch_determinism` for version-tolerant deterministic configuration (`warn_only` fallback, cuDNN/TF32 settings).
  - expanded metadata payload with `cublas_workspace_config` and `cuda_device_count` for more complete reproducibility auditing.
- `utils/structure.py`:

  - added `_element_symbol_number_pairs` using composition-level element extraction to robustly handle oxidation states and disordered species.
  - added `_finite_float` helper and applied finite guards to volume/density/neighbor-distance aggregation.
  - hardened `_sample_site_indices` for non-positive `max_samples`.
  - preserved existing API and return schema while improving edge-case stability.
- Tests:

  - `test_reproducibility.py` now covers hex/scientific seed parsing, bool-like deterministic flags, and expanded runtime metadata fields.
  - `test_structure_utils.py` now covers oxidation+disorder element parsing and non-positive sampling bounds.
  - Verified with:

    - `python -m ruff check atlas/utils/reproducibility.py atlas/utils/structure.py tests/unit/research/test_reproducibility.py tests/unit/models/test_structure_utils.py`
    - `python -m pytest tests/unit/research/test_reproducibility.py tests/unit/models/test_structure_utils.py -q`

## Research references used in batch 29

- PyTorch reproducibility notes (deterministic behavior and CUDA caveats): https://pytorch.org/docs/stable/notes/randomness.html
- PyTorch `torch.use_deterministic_algorithms` API docs: https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
- NumPy `numpy.random.seed` docs: https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
- Python `random.seed` docs: https://docs.python.org/3/library/random.html#random.seed
- pymatgen core API docs (composition/structure utilities): https://pymatgen.org/pymatgen.core.html
- Spglib method paper (symmetry search context): https://doi.org/10.1038/s41524-018-0081-4

## Batch 30 (max 5 files)

- [X] `atlas/active_learning/objective_space.py` - reviewed + optimized
- [X] `atlas/active_learning/pareto_utils.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_objective_space.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_pareto_utils.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 30 optimization goals

- Harden objective-space preprocessing against ambiguous scalar-like inputs and malformed threshold values.
- Improve Pareto/non-dominated sorting semantics for non-finite rows and reduce hypervolume estimation variance in low-data 3D edge cases.
- Add regression tests for the new edge-case policies so CI guards selection logic stability.

## Batch 30 outcomes

- `active_learning/objective_space.py`:

  - `clip01` now routes through `safe_float` so scalar-like wrappers are handled consistently before clipping.
  - `safe_float` now supports single-value `ndarray`/list/tuple payloads, avoiding accidental default-to-zero on scalar containers.
  - `infer_objective_dimension` now uses `_coerce_obj_dim` for both inferred and fallback dimensions.
  - `feasibility_mask_from_points` now sanitizes thresholds with `safe_float` and enforces strict behavior for `use_joint_synthesis=True` when synthesis dimension is missing (returns infeasible mask).
- `active_learning/pareto_utils.py`:

  - added `_finite_row_mask` helper for consistent finite-row handling.
  - `non_dominated_sort` now sorts finite points first and appends non-finite rows as a terminal front, making ranking semantics explicit and stable.
  - `hypervolume` now uses exact box-volume for the `dim>=3` single-point case instead of Monte Carlo approximation.
  - `mc_hv_improvements_shared` now returns early when no candidate passes feasibility/reference filters, reducing unnecessary sampling work.
- Tests:

  - `test_objective_space.py` adds coverage for:

    - scalar-container clipping,
    - strict joint-synthesis feasibility gate on missing synthesis objective,
    - safe objective-dimension fallback,
    - shortest-length truncation + clipping in term-based objective construction.
  - `test_pareto_utils.py` adds coverage for:

    - all-nonfinite non-dominated sort behavior (single terminal front),
    - exact 3D single-point hypervolume path.
  - Verified with:

    - `python -m ruff check atlas/active_learning/objective_space.py atlas/active_learning/pareto_utils.py tests/unit/active_learning/test_objective_space.py tests/unit/active_learning/test_pareto_utils.py`
    - `python -m pytest tests/unit/active_learning/test_objective_space.py tests/unit/active_learning/test_pareto_utils.py -q`

## Research references used in batch 30

- Deb et al. (2002), NSGA-II fast non-dominated sorting baseline: https://doi.org/10.1109/4235.996017
- Auger et al. (2012), hypervolume indicator complexity foundations: https://doi.org/10.1016/j.tcs.2011.03.012
- NumPy `isfinite` docs (finite-mask semantics): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy broadcasting guide (vectorized dominance/HV operations): https://numpy.org/doc/stable/user/basics.broadcasting.html
- NumPy `nan_to_num` docs (stable non-finite replacement policy): https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html

## Batch 31 (max 5 files)

- [X] `atlas/active_learning/acquisition.py` - reviewed + optimized
- [X] `atlas/active_learning/policy_state.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_acquisition.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_policy_state.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 31 optimization goals

- Improve acquisition scoring stability under non-finite model outputs and malformed scalar hyperparameters (`best_f`, `jitter`, `kappa`).
- Harden policy config/state validation so `None`/unknown strings/non-finite numerics cannot silently corrupt runtime policy behavior.
- Add regression tests for the above edge cases to keep CI behavior explicit.

## Batch 31 outcomes

- `active_learning/acquisition.py`:

  - `_prepare_mean_std` now sanitizes non-finite means to finite neutral values.
  - added scalar sanitizers for `best_f`, `jitter` (non-negative clamp), and `kappa` (non-negative clamp).
  - wired sanitizers into `EI/LogEI/PI/LogPI/NEI/UCB/LCB` paths and unified `score_acquisition` entrypoint.
  - effect: acquisition utilities remain finite and directionally stable even with noisy upstream surrogate outputs or malformed config values.
- `active_learning/policy_state.py`:

  - `_coerce_bool` now falls back to default for unknown strings instead of Python truthiness of non-empty text.
  - added `_coerce_text` for safe/consistent enum-like string parsing.
  - strengthened `ActiveLearningPolicyConfig.validated()` by routing numeric fields through finite coercion before clipping/bounding.
  - strengthened `PolicyState.validated()` similarly to prevent `nan/inf` persistence in state checkpoints.
- Tests:

  - `test_acquisition.py` adds coverage for:

    - negative `kappa` clamp behavior,
    - non-finite `best_f` + negative `jitter` sanitization,
    - non-finite `mean` sanitization on `"mean"` strategy.
  - `test_policy_state.py` adds coverage for:

    - unknown bool-string fallback,
    - config-level non-finite numeric sanitization,
    - direct state-object non-finite sanitization.
  - Verified with:

    - `python -m ruff check atlas/active_learning/acquisition.py atlas/active_learning/policy_state.py tests/unit/active_learning/test_acquisition.py tests/unit/active_learning/test_policy_state.py`
    - `python -m pytest tests/unit/active_learning/test_acquisition.py tests/unit/active_learning/test_policy_state.py -q`

## Research references used in batch 31

- Ament et al. (2023), LogEI numerical stability: https://arxiv.org/abs/2310.20708
- Letham et al. (2019), Noisy Expected Improvement: https://arxiv.org/abs/1706.07094
- Srinivas et al. (2010), GP-UCB exploration schedule foundations: https://arxiv.org/abs/0912.3995
- BoTorch analytic acquisition documentation: https://botorch.readthedocs.io/en/stable/acquisition.html
- PyTorch `torch.nan_to_num` docs: https://docs.pytorch.org/docs/stable/generated/torch.nan_to_num.html
- NumPy `clip` docs (bounded-parameter sanitization): https://numpy.org/doc/stable/reference/generated/numpy.clip.html

## Batch 32 (max 5 files)

- [X] `atlas/active_learning/acquisition.py` - reviewed + optimized
- [X] `atlas/active_learning/policy_state.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_acquisition.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_policy_state.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 32 optimization goals

- Make acquisition scoring robust to malformed runtime scalar knobs and non-finite surrogate outputs.
- Strengthen policy config/state coercion so invalid text/non-finite numeric payloads cannot silently poison resume/state transitions.
- Extend regression tests to lock the new sanitization guarantees in CI.

## Batch 32 outcomes

- `active_learning/acquisition.py`:

  - added reusable scalar sanitizers for `best_f`, `jitter`, and `kappa`.
  - added mean sanitization in `_prepare_mean_std` (`nan/inf -> 0`) so downstream acquisition math stays finite.
  - integrated sanitization into `EI/LogEI/PI/LogPI/NEI/UCB/LCB` and `score_acquisition`.
  - improved `schedule_ucb_kappa` and NEI sampling input coercion with finite-safe defaults.
- `active_learning/policy_state.py`:

  - `_coerce_bool` now falls back to default for unknown strings instead of treating any non-empty string as `True`.
  - added `_coerce_text` for robust enum-like text parsing.
  - routed `ActiveLearningPolicyConfig.validated()` numeric fields through finite-safe coercion before clipping/bounding.
  - routed `PolicyState.validated()` through finite-safe coercion, preventing `nan/inf` persistence in serialized state.
  - `PolicyState.from_dict()` now returns already validated state.
- Tests:

  - `test_acquisition.py` adds:

    - negative-kappa clamp check,
    - non-finite `best_f` + negative `jitter` sanitization check,
    - non-finite mean sanitization check for mean strategy.
  - `test_policy_state.py` adds:

    - unknown bool-string fallback behavior,
    - config-level non-finite numeric sanitization checks,
    - direct state-object non-finite sanitization checks.
  - Verified with:

    - `python -m ruff check atlas/active_learning/acquisition.py atlas/active_learning/policy_state.py tests/unit/active_learning/test_acquisition.py tests/unit/active_learning/test_policy_state.py`
    - `python -m pytest tests/unit/active_learning/test_acquisition.py tests/unit/active_learning/test_policy_state.py -q`

## Research references used in batch 32

- Ament et al. (2023), LogEI stabilization: https://arxiv.org/abs/2310.20708
- Letham et al. (2019), Noisy Expected Improvement: https://arxiv.org/abs/1706.07094
- Srinivas et al. (2010), GP-UCB schedules and regret bounds: https://arxiv.org/abs/0912.3995
- BoTorch acquisition docs (analytic & log variants): https://botorch.readthedocs.io/en/stable/acquisition.html
- PyTorch `torch.nan_to_num` docs: https://docs.pytorch.org/docs/stable/generated/torch.nan_to_num.html
- NumPy `isfinite` docs: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html

## Batch 33 (max 5 files)

- [X] `atlas/active_learning/gp_surrogate.py` - reviewed + optimized
- [X] `atlas/models/uncertainty.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_gp_surrogate.py` - reviewed + optimized
- [X] `tests/unit/models/test_uncertainty.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 33 optimization goals

- Harden GP surrogate config/feature sanitization against malformed bool/float inputs and non-finite candidate rows.
- Improve uncertainty module contract guarantees (`dict[str, Tensor]` payload, consistent task keys) for ensemble/MC paths.
- Stabilize evidential uncertainty/loss under non-finite targets and non-finite coefficient values.
- Add regression tests that lock the above behavior for CI.

## Batch 33 outcomes

- `active_learning/gp_surrogate.py`:

  - restored typed config validation via `GPSurrogateConfig.validated()` and applied at acquirer init.
  - `_coerce_bool` now uses default fallback for unknown strings (no accidental `True` from non-empty text).
  - `_safe_float` now rejects non-finite values (`nan/inf`) and returns defaults.
  - `ei_jitter` now clamps to non-negative range.
  - `suggest_constrained_utility` now:

    - filters non-finite feature rows,
    - predicts only on finite rows,
    - emits finite fallback utilities for invalid rows,
    - avoids full-round failure from a few bad candidates.
- `models/uncertainty.py`:

  - strengthened `_normalize_prediction_payload` to enforce `dict[str, Tensor]` contract and reject empty/non-tensor payloads.
  - added `_validate_prediction_keys` to enforce consistent task keys across ensemble members / MC samples.
  - clamped std outputs to non-negative for ensemble/MC predictions.
  - evidential forward path now sanitizes `aleatoric/epistemic` to finite non-negative tensors.
  - `evidential_loss` now:

    - sanitizes `coeff` via finite-safe parsing,
    - masks non-finite targets/parameters,
    - returns deterministic zero when no finite supervision is available.
- Tests:

  - `test_gp_surrogate.py` adds:

    - unknown bool-string fallback checks,
    - non-finite `_safe_float` checks,
    - non-finite feature-row handling in constrained utility.
  - `test_uncertainty.py` adds:

    - inconsistent task-key regression tests for Ensemble/MC Dropout,
    - non-tensor payload rejection test,
    - all-non-finite-target evidential-loss zero fallback test.
  - Verified with:

    - `python -m ruff check atlas/active_learning/gp_surrogate.py atlas/models/uncertainty.py tests/unit/active_learning/test_gp_surrogate.py tests/unit/models/test_uncertainty.py`
    - `python -m pytest -q tests/unit/active_learning/test_gp_surrogate.py`
    - `python -m pytest -q tests/unit/models/test_uncertainty.py`

## Research references used in batch 33

- NumPy `isfinite` (official API): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy `nan_to_num` (official API): https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
- PyTorch `torch.isfinite` (official API): https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- PyTorch `torch.nan_to_num` (official API): https://docs.pytorch.org/docs/stable/generated/torch.nan_to_num.html
- scikit-learn `GaussianProcessRegressor` (official API): https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
- Amini et al. (NeurIPS 2020), Deep Evidential Regression: https://papers.nips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html
- Lakshminarayanan et al. (NeurIPS 2017), Deep Ensembles: https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles
- Gal and Ghahramani (ICML 2016), MC Dropout Bayesian approximation: https://proceedings.mlr.press/v48/gal16

## Progress snapshot (after Batch 33)

- Completed: Batch 1 through Batch 33.
- Pending: Batch 34 onward (next 5-file chunk to be selected sequentially).

## Batch 34 (max 5 files)

- [X] `atlas/benchmark/cli.py` - reviewed + optimized
- [X] `atlas/console_style.py` - reviewed + optimized
- [X] `tests/unit/benchmark/test_benchmark_cli.py` - added + optimized
- [X] `tests/unit/test_console_style.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 34 optimization goals

- Harden benchmark CLI argument validation so invalid numeric/range/path payloads fail fast before heavy runtime.
- Improve model-kwargs decoding diagnostics and fold handling determinism.
- Refine console color support logic for standard env flags and edge cases (`sep=None`, `file=None`).
- Add targeted unit tests for the new validation/styling behavior.

## Batch 34 outcomes

- `benchmark/cli.py`:

  - added `_parse_model_kwargs` with explicit JSON decoding error messages.
  - added `_validate_cli_args` to enforce:

    - task key membership,
    - checkpoint path existence/type,
    - positive/non-negative numeric bounds,
    - coverage domain bounds,
    - fold non-negativity plus deduplicate/sort normalization.
  - validation now runs in `main()` for non-`--list-tasks` code paths.
- `console_style.py`:

  - added env helpers for truthy/falsy color flags.
  - `_supports_color` now supports `FORCE_COLOR`, `CLICOLOR_FORCE`, `CLICOLOR`, `TERM=dumb`, and `NO_COLOR` precedence.
  - `styled_print` now handles `file=None` and `sep=None` safely.
- Tests:

  - added `test_benchmark_cli.py` for kwargs parsing, range/path validation, fold normalization, and list-task bypass behavior.
  - added `test_console_style.py` for color-env precedence and `sep=None` print path.
  - Verified with:

    - `python -m ruff check atlas/benchmark/cli.py atlas/console_style.py tests/unit/benchmark/test_benchmark_cli.py tests/unit/test_console_style.py`
    - `python -m pytest -q tests/unit/benchmark/test_benchmark_cli.py`
    - `python -m pytest -q tests/unit/test_console_style.py`

## Research references used in batch 34

- Python `argparse` docs (official): https://docs.python.org/3/library/argparse.html
- Python `importlib` docs (official): https://docs.python.org/3/library/importlib.html
- Python `json` docs (official): https://docs.python.org/3/library/json.html
- Python `pathlib` docs (official): https://docs.python.org/3/library/pathlib.html
- Python `print()` docs (`sep`/`file` semantics): https://docs.python.org/3/library/functions.html#print
- PyTorch `torch.load` docs: https://pytorch.org/docs/stable/generated/torch.load.html
- PyTorch serialization notes: https://pytorch.org/docs/stable/notes/serialization.html
- `NO_COLOR` community standard: https://no-color.org/
- pytest monkeypatch docs (official): https://docs.pytest.org/en/stable/how-to/monkeypatch.html

## Progress snapshot (after Batch 34)

- Completed: Batch 1 through Batch 34.
- Pending: Batch 35 onward (next 5-file chunk to be selected sequentially).

## Batch 35 (max 5 files)

- [X] `atlas/__init__.py` - reviewed + optimized
- [X] `atlas/active_learning/__init__.py` - reviewed + optimized
- [X] `atlas/benchmark/__init__.py` - reviewed + optimized
- [X] `atlas/data/__init__.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 35 optimization goals

- Standardize package-level public API exposure via explicit lazy-export maps.
- Reduce import-time coupling by avoiding eager heavy submodule imports in package `__init__`.
- Improve package introspection stability (`__all__`, `__dir__`) and preserve backward-compatible attribute access.
- Validate that benchmark/data package users continue to work under lazy export paths.

## Batch 35 outcomes

- `atlas/__init__.py`:

  - replaced eager `Config/get_config` import with module-level lazy export map.
  - added PEP-562-style `__getattr__` and deterministic `__dir__`.
  - kept public API contract (`__version__`, `Config`, `get_config`) unchanged.
- `atlas/active_learning/__init__.py`:

  - normalized `__all__` to immutable tuple.
  - tightened `__getattr__` signature and `__dir__` implementation using set-union to avoid duplicate names.
  - preserved all existing exports and lazy-load semantics.
- `atlas/benchmark/__init__.py`:

  - replaced eager `from .runner import ...` with lazy export map to reduce import-time overhead.
  - added package-level `__getattr__` / `__dir__` for stable dynamic resolution and introspection.
  - preserved exported benchmark symbols and compatibility for `from atlas.benchmark import ...`.
- `atlas/data/__init__.py`:

  - normalized `__all__` tuple and refined `__dir__` to stable deduplicated output.
  - kept existing lazy import behavior with explicit typed `__getattr__`.
  - maintained all previous public exported names.
- Verification:

  - `python -m ruff check atlas/__init__.py atlas/active_learning/__init__.py atlas/benchmark/__init__.py atlas/data/__init__.py`
  - `python -m pytest -q tests/unit/benchmark/test_benchmark_runner.py tests/unit/data/test_crystal_dataset.py`
  - import smoke test: `import atlas`, `from atlas.benchmark import MatbenchRunner`, `from atlas.data import DataSourceRegistry`, etc.

## Research references used in batch 35

- PEP 562: Module `__getattr__` and `__dir__`: https://peps.python.org/pep-0562/
- Python data model (module customization): https://docs.python.org/3/reference/datamodel.html
- Python `importlib` docs (`import_module`): https://docs.python.org/3/library/importlib.html
- Python language reference (`__all__` and import semantics): https://docs.python.org/3/reference/simple_stmts.html
- Python tutorial modules/packages (`__all__` in packages): https://docs.python.org/3/tutorial/modules.html
- PEP 8 public/internal interface guidance (`__all__`): https://peps.python.org/pep-0008/
- `pkgutil` docs (package import side effects context): https://docs.python.org/3/library/pkgutil.html

## Progress snapshot (after Batch 35)

- Completed: Batch 1 through Batch 35.
- Pending: Batch 36 onward.

## Batch 36 (max 5 files)

- [X] `atlas/discovery/alchemy/__init__.py` - reviewed + optimized
- [X] `atlas/discovery/alchemy/calculator.py` - reviewed + optimized
- [X] `atlas/discovery/alchemy/model.py` - reviewed + optimized
- [X] `atlas/discovery/alchemy/optimizer.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 36 optimization goals

- Reduce import-time fragility for optional alchemical dependencies (MACE/e3nn path).
- Improve calculator/runtime robustness for device handling, weight sanitization, and non-finite model outputs.
- Add stricter alchemical pair/weight validation in graph construction to fail fast on malformed inputs.
- Upgrade composition projection from ad-hoc normalization to mathematically grounded simplex projection.

## Batch 36 outcomes

- `discovery/alchemy/__init__.py`:

  - replaced eager import block with explicit lazy export map (`__getattr__`, `__dir__`).
  - optional dependency failures now produce clear lazy-time diagnostics instead of hard module import failure.
  - preserved fallback behavior for missing calculator dependency via on-demand unavailable-calculator factory.
- `discovery/alchemy/calculator.py`:

  - removed dead `contextlib` import path.
  - added device normalization (`cpu/cuda/cuda:*`) with CUDA-unavailable fallback to CPU.
  - added alchemical weight validation (`size`, `finite`, clipping to `[0, 1]`) for init and updates.
  - replaced `.data` assignment with `torch.no_grad()+copy_` to avoid unsafe parameter mutation patterns.
  - hardened `calculate` path with:

    - stable grad reset in `finally`,
    - non-finite energy detection,
    - explicit failure when model outputs are missing.
- `discovery/alchemy/model.py`:

  - added pair parsing helper with explicit index/atomic-number validation.
  - enforced `alchemical_weights` length match to group count and finite-value requirement.
  - improved fixed-atom lookup complexity by using set membership for non-alchemical index extraction.
- `discovery/alchemy/optimizer.py`:

  - removed import-time hard dependency on `mace` path by using `TYPE_CHECKING` for calculator typing.
  - validated optimizer hyperparameter (`learning_rate` finite positive).
  - added exact simplex projection helper (`_project_to_simplex`) and applied per constrained atom group.
  - hardened optimization step for missing atoms, gradient shape mismatch, and non-finite gradients.
  - normalized run-step validation (`steps >= 0`).
- Verification:

  - `python -m ruff check atlas/discovery/alchemy/__init__.py atlas/discovery/alchemy/calculator.py atlas/discovery/alchemy/model.py atlas/discovery/alchemy/optimizer.py`
  - alchemy import/projection smoke script covering:

    - lazy export availability,
    - simplex projection invariants (`sum=1`, bounded, finite) under dirty input vector.

## Research references used in batch 36

- PEP 562 (module `__getattr__`/`__dir__` lazy access): https://peps.python.org/pep-0562/
- ASE calculators documentation (calculator contract and properties): https://ase.gitlab.io/ase/ase/calculators/calculators.html
- PyTorch `torch.autograd.grad` API: https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html
- NumPy `nan_to_num` API: https://numpy.org/doc/2.1/reference/generated/numpy.nan_to_num.html
- NumPy `clip` API: https://numpy.org/doc/stable/reference/generated/numpy.clip.html
- Duchi et al. (ICML 2008), efficient projection algorithm basis: https://web.stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
- MACE repository (architecture/API context for alchemical integration): https://github.com/ACEsuit/mace
- alchemical-mlip repository (upstream alchemical formulation context): https://github.com/learningmatter-mit/alchemical-mlip

## Progress snapshot (after Batch 36)

- Completed: Batch 1 through Batch 36.
- Pending: Batch 37 onward.

## Batch 37 (max 5 files)

- [X] `atlas/discovery/stability/__init__.py` - reviewed + optimized
- [X] `atlas/discovery/stability/mepin.py` - reviewed + optimized
- [X] `atlas/discovery/transport/liflow.py` - reviewed + optimized
- [X] `atlas/explain/__init__.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 37 optimization goals

- Make stability/explain package entrypoints consistent with lazy-export architecture and optional-dependency behavior.
- Harden MEPIN and LiFlow wrappers against malformed runtime inputs (device, checkpoints, sizes, non-finite outputs).
- Reduce import-time failures from optional third-party repos by delaying heavy imports and improving diagnostics.
- Keep API compatibility while adding stricter invariants for trajectory shape and composition parameters.

## Batch 37 outcomes

- `discovery/stability/__init__.py`:

  - implemented explicit lazy export map with `__getattr__` and deterministic `__dir__`.
  - preserved public symbol surface (`MEPINStabilityEvaluator`) while avoiding eager heavy import.
- `discovery/stability/mepin.py`:

  - switched path resolution to `pathlib` and guarded repo-path insertion.
  - added device normalization (`cpu/cuda` with CUDA-unavailable fallback).
  - added checkpoint resolver with supported model-type validation.
  - added trajectory input/output guards:

    - `num_images >= 2`,
    - reactant/product atom-count match,
    - model output tensor finite check + exact shape/size validation.
  - standardized `__all__` and error messaging for optional backend failures.
- `discovery/transport/liflow.py`:

  - switched to `pathlib` path resolution and guarded repo-path insertion.
  - added device normalization and temperature-list validation.
  - added checkpoint resolver and element-index loading validation.
  - hardened simulation pipeline:

    - validates `atoms`, `steps`, `flow_steps`,
    - validates each frame shape and finite coordinates,
    - stabilizes diffusion estimate with non-negative finite output.
- `explain/__init__.py`:

  - converted eager imports to lazy export map with robust `ImportError` handling for optional latent-analysis deps.
  - preserved expected behavior: `LatentSpaceAnalyzer` resolves to `None` when optional stack is unavailable.
- Verification:

  - `python -m ruff check atlas/discovery/stability/__init__.py atlas/discovery/stability/mepin.py atlas/discovery/transport/liflow.py atlas/explain/__init__.py`
  - `python -m py_compile atlas/discovery/stability/__init__.py atlas/discovery/stability/mepin.py atlas/discovery/transport/liflow.py atlas/explain/__init__.py`
  - import smoke script for `atlas.discovery.stability`, `atlas.explain`, and helper/device normalization checks.

## Research references used in batch 37

- PEP 562 (module lazy attribute access): https://peps.python.org/pep-0562/
- Python import system reference (`sys.path` semantics): https://docs.python.org/3/reference/import.html
- Python `pathlib` docs: https://docs.python.org/3/library/pathlib.html
- Python logging HOWTO: https://docs.python.org/3/howto/logging.html
- PyTorch `torch.no_grad` docs: https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html
- PyTorch `torch.isfinite` docs: https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- NumPy `nan_to_num` docs: https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
- NumPy broadcasting/user guide (shape-safe vectorized ops context): https://numpy.org/doc/stable/user/basics.broadcasting.html
- ASE Atoms API reference: https://ase-lib.org/ase/atoms.html
- ASE calculators documentation: https://ase.gitlab.io/ase/ase/calculators/calculators.html
- Lipman et al. (ICLR 2023), Flow Matching for Generative Modeling: https://openreview.net/forum?id=PqvMRDCJT9t

## Progress snapshot (after Batch 37)

- Completed: Batch 1 through Batch 37.
- Pending: Batch 38 onward.

## Batch 38 (max 5 files)

- [X] `atlas/explain/gnn_explainer.py` - reviewed + optimized
- [X] `atlas/explain/latent_analysis.py` - reviewed + optimized
- [X] `atlas/models/__init__.py` - reviewed + optimized
- [X] `atlas/models/equivariant.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 38 optimization goals

- Harden explainability utilities against missing optional graph attributes and non-finite explanation outputs.
- Improve latent-space analysis robustness for empty loaders, method parameter edge-cases, and clustering dimensional constraints.
- Convert `atlas.models` package exports to lazy-loading architecture to reduce import-time dependency coupling.
- Stabilize equivariant model inputs (species indexing, radial basis cutoff, shape checks) and remove hard-coded species assumptions.

## Batch 38 outcomes

- `explain/gnn_explainer.py`:

  - added finite-safe node/edge importance normalization with support for attribute-level node masks.
  - improved atomic-number extraction fallback (`z` or `x`) and robust bond symbol derivation.
  - made explainer call resilient to missing `edge_attr`.
  - added strict size checks for structure plotting (`node_importance` length must match atom count).
  - hardened atom radius handling (`atomic_radius` fallback).
- `explain/latent_analysis.py`:

  - added device normalization with CUDA fallback warnings.
  - added robust embedding extraction from tensor or dict outputs (`embedding`/`latent`/`mean` keys).
  - now sanitizes non-finite embeddings/properties and fails fast on empty loaders.
  - dimensional reduction now validates sample/component counts and constrains t-SNE perplexity to valid domain.
  - clustering now validates `n_clusters` vs sample count and method domain.
  - plotting and cluster-analysis paths now validate shape/length consistency.
- `models/__init__.py`:

  - replaced eager imports with explicit lazy export map (`__getattr__`, `__dir__`), preserving public API names.
- `models/equivariant.py`:

  - added `_infer_species_indices` to support multiple node feature layouts and remove hard-coded `:86` slicing.
  - validated model/radial constructor parameters (`n_layers`, `n_species`, `max_radius`, etc.).
  - radial basis now zeroes outside cutoff and sanitizes non-finite distances.
  - strengthened `encode` with shape checks for `edge_index`/`edge_vectors` and safer batch coercion.
  - made output-head hidden width robust when scalar channel count is small.
- Verification:

  - `python -m ruff check atlas/explain/gnn_explainer.py atlas/explain/latent_analysis.py atlas/models/__init__.py atlas/models/equivariant.py`
  - `python -m py_compile atlas/explain/gnn_explainer.py atlas/explain/latent_analysis.py atlas/models/__init__.py atlas/models/equivariant.py`
  - `python -m pytest -q tests/unit/models/test_model_utils.py tests/unit/models/test_prediction_utils.py`
  - smoke script for:

    - lazy `atlas.models` exports,
    - species-index inference helper,
    - radial basis finite output checks.

## Research references used in batch 38

- PyTorch Geometric explainability API docs: https://pytorch-geometric.readthedocs.io/en/latest/modules/explain.html
- GNNExplainer paper (NeurIPS 2019): https://papers.neurips.cc/paper/9123-gnnexplainer-generating-explanations-for-graph-neural-networks
- scikit-learn TSNE docs (perplexity constraint): https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
- scikit-learn PCA docs: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- scikit-learn KMeans docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- scikit-learn DBSCAN docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
- UMAP API docs: https://umap-learn.readthedocs.io/en/latest/api.html
- PEP 562 (module-level lazy attribute access): https://peps.python.org/pep-0562/
- NequIP paper (Nature Communications 2022): https://www.nature.com/articles/s41467-022-29939-5
- e3nn documentation: https://docs.e3nn.org/
- DimeNet paper (smooth radial/cutoff basis context): https://arxiv.org/abs/2003.03123
- NumPy `nan_to_num` docs: https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html

## Progress snapshot (after Batch 38)

- Completed: Batch 1 through Batch 38.
- Pending: Batch 39 onward.

## Batch 39 (max 5 files)

- [X] `atlas/models/fast_tp.py` - reviewed + optimized
- [X] `atlas/models/layers.py` - reviewed + optimized
- [X] `atlas/models/m3gnet.py` - reviewed + optimized
- [X] `atlas/models/matgl_three_body.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 39 optimization goals

- Harden fused tensor-product scatter path against malformed edge tensors and invalid indexing.
- Tighten message-passing/equivariant layer input contracts and remove accidental extra nonlinearity in update path.
- Improve M3GNet numerical stability and shape robustness (RBF edge cases, 3-body indexing, species indexing, aggregation semantics).
- Validate three-body basis helpers (`max_n/max_l/n_basis`, shape/finite checks) for safer low-data and noisy-input regimes.

## Batch 39 outcomes

- `models/fast_tp.py`:

  - added strict shape/range checks for `edge_src/edge_dst/edge_attr/edge_weight`.
  - validated `edge_weight` second dimension against `weight_numel`.
  - added empty-edge fast path and non-finite message sanitization.
- `models/layers.py`:

  - added constructor validation (`node_dim`, `edge_dim`, `n_radial_basis`, `max_radius`).
  - added forward shape/range checks and empty-edge guard in both layers.
  - removed duplicated SiLU in `MessagePassingLayer` update path (MLP already applies SiLU), reducing over-smoothing risk.
  - added per-node degree normalization in M3GNet-style aggregation path where appropriate.
- `models/matgl_three_body.py`:

  - added parameter validation for basis sizes.
  - added shape consistency checks for `(r_ij, r_ik, cos_theta)`.
  - added non-finite sanitization (`nan_to_num`) for robust basis outputs.
- `models/m3gnet.py`:

  - `RBFExpansion` now handles `n_gaussians=1` and invalid cutoff safely.
  - hardened `ThreeBodyInteraction` basis setup (`max_l>=1`) and finite-safe feature composition.
  - fixed ambiguous/double node aggregation behavior by using a single `dst` aggregation with degree normalization.
  - added comprehensive shape checks for `edge_index`, `edge_attr`, `edge_vectors`, `edge_index_3body`.
  - removed hard-coded `:86` species assumption; now robustly infers species indices and clamps into `[0, n_species-1]`.
  - added edge feature dimension adaptation (truncate/pad) before edge embedding.
- Verification:

  - `python -m ruff check atlas/models/fast_tp.py atlas/models/layers.py atlas/models/m3gnet.py atlas/models/matgl_three_body.py`
  - `python -m py_compile atlas/models/fast_tp.py atlas/models/layers.py atlas/models/m3gnet.py atlas/models/matgl_three_body.py`
  - smoke checks for:

    - `RBFExpansion` finite output on `inf` distances,
    - `MessagePassingLayer` finite output on random graph,
    - `SphericalBesselHarmonicsExpansion` output shape,
    - `M3GNet.encode` finite embedding generation.

## Research references used in batch 39

- M3GNet paper (Nature Computational Science 2022): https://www.nature.com/articles/s43588-022-00349-3
- M3GNet arXiv preprint: https://arxiv.org/abs/2202.02450
- PyTorch `index_add` docs: https://docs.pytorch.org/docs/stable/generated/torch.index_add.html
- PyTorch Geometric explainability docs (API stability context): https://pytorch-geometric.readthedocs.io/en/latest/modules/explain.html
- Duchi et al. (ICML 2008), projection algorithms basis: https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
- e3nn documentation: https://docs.e3nn.org/
- DimeNet paper (radial/angular basis context): https://arxiv.org/abs/2003.03123
- NumPy `nan_to_num` docs: https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html

## Progress snapshot (after Batch 39)

- Completed: Batch 1 through Batch 39.
- Pending: Batch 40 onward.

## Batch 40 (max 5 files)

- [X] `atlas/ops/cpp_ops.py` - reviewed + optimized
- [X] `atlas/potentials/__init__.py` - reviewed + optimized
- [X] `atlas/potentials/mace_relaxer.py` - reviewed + optimized
- [X] `atlas/potentials/relaxers/mlip_arena_relaxer.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 40 optimization goals

- Stabilize native/C++ radius-graph path under mixed environments (no compiler, no Ninja, CUDA/CPU differences) while preserving deterministic fallback behavior.
- Reduce package import-time coupling in `atlas.potentials` and align with prior lazy-export architecture used across ATLAS packages.
- Harden MACE relaxation runtime against invalid devices, invalid optimization hyperparameters, unsupported cell filters, and non-finite energy/force outputs.
- Replace brittle MLIP-Arena wrapper imports with an ASE-backed robust implementation preserving registry compatibility.

## Batch 40 outcomes

- `ops/cpp_ops.py`:

  - rebuilt C++ kernel to enforce `max_num_neighbors` per source atom via partial sort.
  - removed hard dependency on `torch_cluster` fallback and replaced with pure-torch fallback using `torch.cdist` + per-node top-k pruning.
  - added strict input validation (`pos/batch` shapes, dtype/device consistency, finite `r_max`, positive neighbor budget).
  - added safe empty-output path, one-time PBC warning, and compile/runtime fallback logging.
  - made C++ invocation robust by CPU/contiguous casting and automatic return to caller device/dtype.
- `potentials/__init__.py`:

  - switched to explicit lazy-export map (`MACERelaxer`, `NativeMlipArenaRelaxer`) with deterministic `__dir__` and PEP-562-compatible `__getattr__`.
- `potentials/mace_relaxer.py`:

  - added strong constructor validation (`model_size`, `default_dtype`, normalized device policy with CUDA fallback handling).
  - hardened calculator bootstrap for missing/invalid custom model path and foundation-model fallback.
  - added relaxation input guards (`fmax`, `steps`, empty structure handling, trajectory path creation).
  - standardized cell-filter resolution with validated domain (`frechet|exp|unit|fixed|None`) and fallback behavior.
  - added finite checks for output energy/forces and safer `volume_change` computation when initial volume is invalid.
  - made batch mode explicit about serial execution semantics for GPU calculators.
- `potentials/relaxers/mlip_arena_relaxer.py`:

  - replaced fragile third-party imports with stable ASE-native optimizer/filter mapping while preserving registry key (`mlip_arena_native`).
  - added parameter schema validation (`fmax`, `steps`, `optimizer`, `cell_filter`).
  - implemented robust relax path with optional symmetry constraints, cell-filter selection, non-finite energy rejection, and standardized failure payload.
- Verification:

  - `python -m ruff check atlas/ops/cpp_ops.py atlas/potentials/__init__.py atlas/potentials/mace_relaxer.py atlas/potentials/relaxers/mlip_arena_relaxer.py`
  - `python -m py_compile atlas/ops/cpp_ops.py atlas/potentials/__init__.py atlas/potentials/mace_relaxer.py atlas/potentials/relaxers/mlip_arena_relaxer.py`
  - smoke script covering:

    - `fast_radius_graph` finite output and shape contract,
    - lazy import of `MACERelaxer` / `NativeMlipArenaRelaxer`,
    - constructor/runtime failure-path checks for `NativeMlipArenaRelaxer`.

## Research references used in batch 40

- PyTorch C++ extension docs (`load_inline`/extension build): https://pytorch.org/docs/stable/cpp_extension.html
- PyTorch `torch.cdist` API: https://pytorch.org/docs/stable/generated/torch.cdist.html
- PyTorch `torch.topk` API: https://pytorch.org/docs/stable/generated/torch.topk.html
- ASE optimizers documentation: https://ase.gitlab.io/ase/ase/optimize.html
- ASE constraints documentation: https://ase.gitlab.io/ase/ase/constraints.html
- ASE filters documentation (`FrechetCellFilter`, cell filters): https://ase.gitlab.io/ase/ase/filters.html
- MACE repository / calculator integration context: https://github.com/ACEsuit/mace
- MACE paper (ICLR 2022 Workshop): https://arxiv.org/abs/2206.07697
- PEP 562 (`__getattr__`/`__dir__` on modules): https://peps.python.org/pep-0562/
- Python `importlib` docs: https://docs.python.org/3/library/importlib.html
- Python warnings docs: https://docs.python.org/3/library/warnings.html

## Progress snapshot (after Batch 40)

- Completed: Batch 1 through Batch 40.
- Pending: Batch 41 onward.

## Batch 41 (max 5 files)

- [X] `atlas/research/__init__.py` - reviewed + optimized
- [X] `atlas/research/method_registry.py` - reviewed + optimized
- [X] `atlas/research/workflow_reproducible_graph.py` - reviewed + optimized
- [X] `atlas/thermo/__init__.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 41 optimization goals

- Standardize research/thermo package entrypoints with lazy-export architecture to reduce import-time coupling and optional-dependency breakage.
- Harden methodology registry contracts (immutability, schema validation, duplicate-key governance) for reproducible experiment selection.
- Upgrade reproducibility workflow manifest path with stronger data validation, run-id safety, and atomic persistence semantics.
- Back new behavior with focused unit tests so regressions are caught in CI.

## Batch 41 outcomes

- `research/__init__.py`:

  - replaced eager imports with explicit lazy export map (`__getattr__`, `__dir__`).
  - preserved public API while reducing side effects from importing heavy research modules.
- `research/method_registry.py`:

  - upgraded `MethodSpec` to immutable tuple-based `strengths/tradeoffs` with normalization/validation (`__post_init__`).
  - added registry duplicate-key guard (`replace=False` default) and type checks on `register`.
  - maintained backward-compatible helper API (`get_method`, `list_methods`, `recommended_method_order`).
- `research/workflow_reproducible_graph.py`:

  - added strict validators for iteration counters/timings and manifest stage-plan hygiene.
  - sanitized run-id tokens to prevent unsafe path characters in artifact filenames.
  - removed repeated `type: ignore` union access via internal `_ensure_manifest()` helper.
  - switched manifest writes to atomic temp-file replacement to avoid partial/corrupted JSON on interrupted writes.
  - added `schema_version` field for forward-compatible artifact parsing.
- `thermo/__init__.py`:

  - replaced eager optional-import block with lazy optional exports.
  - optional missing dependencies now resolve to `None` at attribute access time while keeping module import healthy.
- Tests:

  - `tests/unit/research/test_method_registry.py`: added normalization/immutability and duplicate-key rejection cases.
  - `tests/unit/research/test_workflow_reproducible_graph.py`: added non-monotonic iteration rejection and run-id sanitization checks.
  - `tests/unit/thermo/test_init_exports.py` (new): added lazy export contract tests and optional-import-failure behavior.
- Verification:

  - `python -m ruff check atlas/research/__init__.py atlas/research/method_registry.py atlas/research/workflow_reproducible_graph.py atlas/thermo/__init__.py tests/unit/research/test_method_registry.py tests/unit/research/test_workflow_reproducible_graph.py tests/unit/thermo/test_init_exports.py`
  - `python -m py_compile atlas/research/__init__.py atlas/research/method_registry.py atlas/research/workflow_reproducible_graph.py atlas/thermo/__init__.py tests/unit/research/test_method_registry.py tests/unit/research/test_workflow_reproducible_graph.py tests/unit/thermo/test_init_exports.py`
  - `python -m pytest -q tests/unit/research/test_method_registry.py tests/unit/research/test_workflow_reproducible_graph.py tests/unit/thermo/test_init_exports.py`

## Research references used in batch 41

- PEP 562 (module `__getattr__`/`__dir__`): https://peps.python.org/pep-0562/
- Python `importlib` docs: https://docs.python.org/3/library/importlib.html
- Python `dataclasses` docs: https://docs.python.org/3/library/dataclasses.html
- Python `tempfile` docs (atomic temp-file workflow): https://docs.python.org/3/library/tempfile.html
- Python `pathlib` docs: https://docs.python.org/3/library/pathlib.html
- W3C PROV family overview (provenance model for reproducible pipelines): https://www.w3.org/TR/prov-overview/
- NeurIPS Reproducibility Checklist (artifact/reporting norms): https://neurips.cc/public/guides/PaperChecklist
- ACM Artifact Review & Badging policy: https://www.acm.org/publications/policies/artifact-review-and-badging-current
- Software Heritage reproducibility resources (archival/provenance context): https://www.softwareheritage.org/
- PyPA entry points specification (plugin/registry extensibility reference): https://packaging.python.org/en/latest/specifications/entry-points/
- MLflow tracking docs (run metadata/artifact lineage reference): https://mlflow.org/docs/latest/tracking.html
- DVC experiment docs (reproducible experiment/version context): https://dvc.org/doc

## Progress snapshot (after Batch 41)

- Completed: Batch 1 through Batch 41.
- Pending: Batch 42 onward.

## Batch 42 (max 5 files)

- [X] `atlas/thermo/calphad.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/__init__.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/atomate2_wrapper.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/engine.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 42 optimization goals

- Harden CALPHAD path robustness (composition normalization, temperature/step validation, Scheil/equilibrium fallback behavior, and stable phase extraction).
- Convert OpenMM package entrypoint to lazy optional exports so missing optional dependencies do not break module import.
- Stabilize native Atomate2 OpenMM wrapper with explicit parameter schema and delayed optional import resolution.
- Improve OpenMM engine reliability for periodic box handling, simulation input validation, reporter lifecycle, and non-finite output defense.

## Batch 42 outcomes

- `thermo/calphad.py`:

  - added strict composition normalization/validation:

    - unknown component rejection,
    - non-finite/negative fraction rejection,
    - dependent-component completion + final normalization.
  - added temperature and `n_steps` validation for equilibrium/solidification paths.
  - strengthened equilibrium extraction from pycalphad outputs (`Phase`/`NP`) using finite checks and stable sorting.
  - made equilibrium path output semantics cleaner by separating `LIQUID` from `solid_phases`.
  - made Scheil path robust to optional module absence and malformed result payloads, with deterministic fallback to equilibrium path.
  - added safer transus detection and plotting guards (shape checks, finite annotation checks, output path creation).
- `thermo/openmm/__init__.py`:

  - replaced eager import with lazy export map (`OpenMMEngine`, `PymatgenTrajectoryReporter`).
  - optional import failures now return `None` instead of crashing package import, with debug-trace capture.
- `thermo/openmm/atomate2_wrapper.py`:

  - replaced brittle top-level imports with delayed `importlib` loading.
  - added constructor validation for `temperature`, `step_size`, `ensemble`.
  - standardized ensemble modes (`nvt`, `npt`, `minimize`) and step validation in maker construction.
  - improved error diagnostics for missing optional Atomate2/OpenMM stacks.
- `thermo/openmm/engine.py`:

  - added constructor-level numeric validation (`temperature`, `friction`, `step_size`).
  - added system setup guards (`atoms` non-empty, finite positions, mass validation).
  - improved periodic box setup with full vector handling (`setPeriodicBoxVectors` when available).
  - made forcefield handling robust:

    - best-effort MACE setup via openmm-ml,
    - deterministic LJ fallback with periodic/non-periodic method selection.
  - improved simulation run path:

    - validated `steps` and `trajectory_interval`,
    - ensured reporter is detached after run,
    - validated final potential energy finite before returning trajectory.
- Verification:

  - `python -m ruff check atlas/thermo/calphad.py atlas/thermo/openmm/__init__.py atlas/thermo/openmm/atomate2_wrapper.py atlas/thermo/openmm/engine.py`
  - `python -m py_compile atlas/thermo/calphad.py atlas/thermo/openmm/__init__.py atlas/thermo/openmm/atomate2_wrapper.py atlas/thermo/openmm/engine.py`
  - `python -m pytest -q tests/unit/thermo/test_init_exports.py`
  - `python -m pytest -q tests/integration/openmm/test_openmm_core.py tests/integration/openmm/test_openmm_mace.py` (both skipped because `openmm` not installed in this environment)
  - smoke script for:

    - CALPHAD helper normalization/transus,
    - lazy OpenMM export behavior under missing dependency,
    - wrapper argument-validation paths.

## Research references used in batch 42

- OpenMM User Guide (Simulation APIs): https://docs.openmm.org/latest/userguide/application/04_advanced_sim_examples.html
- OpenMM API docs (`Topology`, periodic box vectors): https://docs.openmm.org/latest/api-python/generated/openmm.app.topology.Topology.html
- OpenMM API docs (`NonbondedForce` methods): https://docs.openmm.org/latest/api-python/generated/openmm.openmm.NonbondedForce.html
- OpenMM API docs (`LangevinMiddleIntegrator`): https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LangevinMiddleIntegrator.html
- OpenMM framework paper (PLoS Comput Biol 2017): https://pmc.ncbi.nlm.nih.gov/articles/PMC5549999/
- pycalphad docs (`equilibrium` API): https://pycalphad.org/docs/latest/api/pycalphad.core.html#pycalphad.core.equilibrium.equilibrium
- pycalphad Scheil package docs: https://scheil.readthedocs.io/en/stable/
- pycalphad project docs: https://pycalphad.org/docs/latest/
- pycalphad JORS paper: https://openresearchsoftware.metajnl.com/articles/10.5334/jors.140
- CALPHAD method overview (NIST): https://www.nist.gov/publications/calphad-calculation-phase-diagrams-comprehensive-guide
- openmm-ml package (MACE/ML potential integration context): https://github.com/openmm/openmm-ml

## Progress snapshot (after Batch 42)

- Completed: Batch 1 through Batch 42.
- Pending: Batch 43 onward.

## Batch 43 (max 5 files)

- [X] `atlas/thermo/openmm/reporters.py` - reviewed + optimized
- [X] `atlas/thermo/stability.py` - reviewed + optimized
- [X] `atlas/topology/__init__.py` - reviewed + optimized
- [X] `atlas/topology/classifier.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 43 optimization goals

- Harden OpenMM trajectory reporting to ensure finite-safe frame extraction and stable pymatgen trajectory export semantics.
- Strengthen phase-stability analysis contracts (input validation, robust decomposition formatting, and clearer failure payloads).
- Align `atlas.topology` package entrypoint with lazy-export architecture to reduce import-time coupling.
- Improve topology classifier numerical robustness for sparse/non-contiguous batch IDs and invalid edge tensors.

## Batch 43 outcomes

- `thermo/openmm/reporters.py`:

  - added strict constructor validation (`reportInterval > 0`, `Structure` type check).
  - converted conversion constants to explicit symbols and finite-checks for positions/time/energy/forces.
  - hardened `describeNextReport` to always return at least one step.
  - improved exported trajectory payload:

    - consistent `frame_properties` keys (`energy_ev`, `forces_ev_per_ang`, `time_ps`),
    - computed scalar `time_step` from recorded times,
    - explicit failure if no frames were collected.
- `thermo/stability.py`:

  - added typed `StabilityResult` model and stronger validations for energies and chemical system input.
  - `ReferenceDatabase` now enforces key presence in `load_from_list` and finite energy values in `add_entry`.
  - `analyze_stability` now produces deterministic decomposition formatting and clearer fallback payloads.
  - `plot_phase_diagram` now returns `None` safely when no entries exist and avoids unexpected crashes.
- `topology/__init__.py`:

  - switched to explicit lazy-export map with `__getattr__`/`__dir__`.
  - preserved public API (`CrystalGraphBuilder`, `TopoGNN`) with lower import-time coupling.
- `topology/classifier.py`:

  - strengthened input guards:

    - finite checks for node/edge features,
    - edge index dtype/shape/range validation,
    - batch dtype and non-negativity checks.
  - added batch ID remapping to contiguous graph IDs before pooling, preventing empty-graph holes from propagating `-inf` into readout.
  - stabilized max-pooling path via `nan_to_num` and added empty-edge guard in message passing.
  - added checkpoint config type check in `load_model` for safer deserialization path.
- Verification:

  - `python -m ruff check atlas/thermo/openmm/reporters.py atlas/thermo/stability.py atlas/topology/__init__.py atlas/topology/classifier.py`
  - `python -m py_compile atlas/thermo/openmm/reporters.py atlas/thermo/stability.py atlas/topology/__init__.py atlas/topology/classifier.py`
  - `python -m pytest -q tests/unit/topology/test_classifier.py tests/unit/thermo/test_init_exports.py`
  - smoke script for:

    - `PhaseStabilityAnalyst` minimal hull path,
    - non-contiguous batch handling in `TopoGNN.forward`.

## Research references used in batch 43

- OpenMM Reporter API (StateDataReporter reference): https://docs.openmm.org/8.0.0/api-python/generated/openmm.app.statedatareporter.StateDataReporter.html
- OpenMM Cookbook (reporting workflow): https://openmm.github.io/openmm-cookbook/latest/notebooks/tutorials/loading_and_reporting.html
- pymatgen usage docs (phase diagram workflow): https://pymatgen.org/usage.html
- pymatgen analysis docs (`phase_diagram` methods): https://pymatgen.org/pymatgen.analysis
- pycalphad project docs: https://pycalphad.org/docs
- PyTorch `scatter_reduce` docs: https://docs.pytorch.org/docs/stable/generated/torch.scatter_reduce.html
- PyTorch `LayerNorm` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- GraphSAGE paper: https://arxiv.org/abs/1706.02216
- GIN / "How Powerful are Graph Neural Networks?": https://arxiv.org/abs/1810.00826
- PEP 562 (`__getattr__`/`__dir__` on modules): https://peps.python.org/pep-0562/

## Progress snapshot (after Batch 43)

- Completed: Batch 1 through Batch 43.
- Pending: Batch 44 onward.

## Batch 44 (max 5 files)

- [X] `atlas/training/__init__.py` - reviewed + optimized
- [X] `atlas/training/checkpoint.py` - reviewed + optimized
- [X] `atlas/training/filters.py` - reviewed + optimized
- [X] `atlas/training/losses.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 44 optimization goals

- Align `atlas.training` package exports with lazy-loading architecture to reduce import-time coupling and keep API introspection deterministic.
- Harden checkpoint persistence against partial writes and malformed checkpoint payloads.
- Strengthen outlier filtering robustness under missing optional CSV dependencies and heterogeneous scalar payload types.
- Tighten loss-module contracts (constraint validation, duplicate task-name rejection, shape mismatch diagnostics) without breaking existing trainer semantics.

## Batch 44 outcomes

- `training/__init__.py`:

  - replaced eager imports with explicit lazy export map (`__getattr__`, `__dir__`) while preserving public API symbols.
- `training/checkpoint.py`:

  - added atomic checkpoint write helper (`_atomic_torch_save`) using temp file + atomic replace to prevent partial/corrupted checkpoints.
  - added strict mapping validation for `state` payloads (`_coerce_state`) in `save_best` / `save_checkpoint`.
  - enriched checkpoint payload with default metadata (`epoch`, `val_mae`) and switched best pointer update to `copy2`.
- `training/filters.py`:

  - removed hard top-level dependency on pandas; CSV export now gracefully falls back to stdlib `csv` when pandas is unavailable.
  - added dataset/properties interface validation and empty-dataset fast path.
  - improved scalar extraction for single-value numpy arrays and normalized deduplicated property names.
  - added minimum-sample guard in sigma computation to avoid unstable one-point standardization.
- `training/losses.py`:

  - switched scalar-finite checks from tensor allocation to `math.isfinite` for clearer/cheaper validation.
  - moved constraint validation to `PropertyLoss.__init__` (fail-fast config error).
  - added tensor type/shape checks in `PropertyLoss.forward` with explicit mismatch diagnostics.
  - enforced unique non-empty task names in `MultiTaskLoss` and improved per-task shape mismatch errors.
- Tests:

  - updated `tests/unit/training/test_losses.py` for new fail-fast constraint validation and added duplicate-task/shape-mismatch coverage.
  - expanded `tests/unit/training/test_filters.py` for empty-dataset and property-name normalization behavior.
  - expanded `tests/unit/training/test_checkpoint.py` for non-mapping checkpoint payload rejection.
  - added `tests/unit/training/test_init_exports.py` to validate lazy-export package contract.
- Verification:

  - `python -m ruff check atlas/training/__init__.py atlas/training/checkpoint.py atlas/training/filters.py atlas/training/losses.py tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py tests/unit/training/test_losses.py tests/unit/training/test_init_exports.py`
  - `python -m py_compile atlas/training/__init__.py atlas/training/checkpoint.py atlas/training/filters.py atlas/training/losses.py tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py tests/unit/training/test_losses.py tests/unit/training/test_init_exports.py`
  - `python -m pytest -q tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py tests/unit/training/test_losses.py tests/unit/training/test_init_exports.py`

## Research references used in batch 44

- PEP 562 (`__getattr__` / `__dir__` for modules): https://peps.python.org/pep-0562/
- Python `importlib` docs: https://docs.python.org/3/library/importlib.html
- Python `tempfile` docs (safe temp-file writes): https://docs.python.org/3/library/tempfile.html
- Python `Path.replace` / atomic replace semantics (`os.replace` behavior): https://docs.python.org/3/library/pathlib.html#pathlib.Path.replace
- PyTorch serialization notes: https://pytorch.org/docs/stable/notes/serialization.html
- PyTorch `torch.save` docs: https://pytorch.org/docs/stable/generated/torch.save.html
- NIST/SEMATECH e-Handbook (MAD/robust scale background): https://www.itl.nist.gov/div898/handbook/
- Iglewicz & Hoaglin (1993), robust outlier labeling rule using modified Z-scores: https://books.google.com/books/about/How_to_Detect_and_Handle_Outliers.html?id=FuuiEAAAQBAJ
- Kendall, Gal, Cipolla (CVPR 2018), uncertainty-weighted multi-task loss: https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html
- PyTorch BCE-with-logits API (classification loss contract): https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html

## Progress snapshot (after Batch 44)

- Completed: Batch 1 through Batch 44.
- Pending: Batch 45 onward.

## Batch 45 (max 5 files)

- [X] `atlas/training/metrics.py` - reviewed + optimized
- [X] `atlas/training/normalizers.py` - reviewed + optimized
- [X] `atlas/training/physics_losses.py` - reviewed + optimized
- [X] `atlas/training/preflight.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 45 optimization goals

- Harden training metrics against non-finite tensors, shape drift, and optional dependency variability.
- Improve target normalizer robustness for iterable datasets, property-name hygiene, and deterministic state serialization.
- Stabilize physics-constrained loss path for singular elastic tensors and malformed prediction payloads.
- Strengthen preflight runtime diagnostics and artifact-integrity checks before expensive downstream stages.

## Batch 45 outcomes

- `training/metrics.py`:

  - added finite-safe tensor coercion and pair filtering utilities to keep scalar/classification metrics stable under NaN/Inf.
  - normalized matrix metrics to support both single-matrix and batched-matrix inputs.
  - added optional-SciPy Spearman fallback implementation to keep `eigenvalue_agreement` usable when SciPy is unavailable.
- `training/normalizers.py`:

  - added iterable dataset support (not only random-access datasets), mapping/attribute property extraction, and property-name normalization.
  - added stronger `state_dict`/`load_state_dict` schema guards and deterministic key ordering.
  - added explicit unknown-property errors with available-key context.
- `training/physics_losses.py`:

  - rebuilt Voigt-Reuss/Born loss path with finite filtering, pseudo-inverse compliance fallback (`pinv`) for singular tensors, and safe zero-loss fallbacks.
  - validated alpha/weight hyperparameters (finite, non-negative, known keys) and tightened type checks for prediction payloads.
  - ensured all loss terms remain finite under malformed/non-finite physics tensors.
- `training/preflight.py`:

  - added command execution OSError handling (`127`) and improved failure diagnostics.
  - added validation-report integrity gate (must exist and be non-empty after validate step).
  - upgraded split-manifest gate to require non-empty manifest files.
- Tests:

  - expanded `tests/unit/training/test_metrics.py` for single-matrix Frobenius behavior and SciPy-missing Spearman fallback.
  - expanded `tests/unit/training/test_normalizers.py` for iterable datasets, property-name normalization, and empty state-key rejection.
  - expanded `tests/unit/training/test_preflight.py` for missing validation report and OSError command-failure return codes.
  - added `tests/unit/training/test_physics_losses.py` for physics loss finite-safety and alpha validation.
- Verification:

  - `python -m ruff check atlas/training/metrics.py atlas/training/normalizers.py atlas/training/physics_losses.py atlas/training/preflight.py tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py tests/unit/training/test_preflight.py tests/unit/training/test_physics_losses.py`
  - `python -m py_compile atlas/training/metrics.py atlas/training/normalizers.py atlas/training/physics_losses.py atlas/training/preflight.py tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py tests/unit/training/test_preflight.py tests/unit/training/test_physics_losses.py`
  - `python -m pytest -q tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py tests/unit/training/test_preflight.py tests/unit/training/test_physics_losses.py`

## Research references used in batch 45

- SciPy `spearmanr` API docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
- PyTorch `torch.linalg.eigvalsh` docs: https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigvalsh.html
- PyTorch `torch.pinverse`/`torch.linalg.pinv` docs: https://docs.pytorch.org/docs/stable/generated/torch.pinverse.html
- scikit-learn `StandardScaler` docs (`std` convention and normalization semantics): https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- Python `subprocess.run` docs (timeout/return-code contract): https://docs.python.org/3/library/subprocess.html
- Materials Project elasticity methodology (Voigt/Reuss/VRH reporting): https://docs.materialsproject.org/methodology/materials-methodology/elasticity
- pymatgen elasticity API (VRH properties): https://pymatgen.org/pymatgen.analysis.elasticity.html
- Mouhat & Coudert (2014), elastic stability criteria overview: https://arxiv.org/abs/1410.0065
- PRB publication mirror for Born criteria details: https://www.coudert.name/papers/10.1103_PhysRevB.90.224104.pdf

## Progress snapshot (after Batch 45)

- Completed: Batch 1 through Batch 45.
- Pending: Batch 46 onward.

## Batch 46 (max 5 files)

- [X] `atlas/training/run_utils.py` - reviewed + optimized
- [X] `atlas/training/trainer.py` - reviewed + optimized
- [X] `tests/unit/training/test_run_utils_manifest.py` - reviewed + optimized
- [X] `tests/unit/training/test_trainer.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 46 optimization goals

- Strengthen run-manifest persistence integrity (atomic writes + schema-preserving merge semantics).
- Prevent accidental corruption of core manifest sections when callers pass `extra` payloads.
- Improve trainer reliability for checkpoint/history writes and validation-time numeric failures.
- Harden filename/checkpoint stem safety for CI/runtime reproducibility paths.

## Batch 46 outcomes

- `training/run_utils.py`:

  - upgraded `_dump_json_file` to atomic temp-file replace with flush+fsync (crash-safe write pattern).
  - added `_merge_extra_payload` to enforce schema integrity for reserved dict sections (`runtime/args/dataset/split/environment_lock/artifacts/metrics/seeds/configs`).
  - repaired invalid legacy `created_at` payloads during merge (non-string values are normalized to new UTC ISO timestamp).
  - extracted `_ensure_seed_and_config_sections` and `_redact_manifest_sections` to reduce complexity and make policy explicit.
- `training/trainer.py`:

  - added strict checkpoint-name/file-name validation to block path traversal/separator injection.
  - switched checkpoint/history persistence to atomic write helpers (`_atomic_torch_save`, `_atomic_json_dump`).
  - added validation-loop non-finite loss guard (parity with train-loop finite check).
  - strengthened checkpoint loading contract with explicit payload type/key validation.
  - added explicit error when dict predictions cannot resolve any target mapping.
- Tests:

  - `test_run_utils_manifest.py`:

    - added regression for rejecting non-mapping `extra` payload on reserved manifest sections.
    - added regression for repairing invalid `created_at` during merge.
  - `test_trainer.py`:

    - added validation non-finite loss failure test.
    - added checkpoint loader malformed-payload test.
    - added dict-prediction missing-target test.
    - added filename/path-like checkpoint guard tests.
- Verification:

  - `python -m ruff check atlas/training/run_utils.py atlas/training/trainer.py tests/unit/training/test_run_utils_manifest.py tests/unit/training/test_trainer.py`
  - `python -m py_compile atlas/training/run_utils.py atlas/training/trainer.py tests/unit/training/test_run_utils_manifest.py tests/unit/training/test_trainer.py`
  - `python -m pytest -q tests/unit/training/test_run_utils_manifest.py tests/unit/training/test_trainer.py`

## Research references used in batch 46

- Python `tempfile` docs (safe temp-file creation patterns): https://docs.python.org/3/library/tempfile.html
- Python `pathlib.Path.replace` docs (atomic replacement semantics): https://docs.python.org/3/library/pathlib.html#pathlib.Path.replace
- Python `json` docs (`allow_nan=False`, deterministic dump controls): https://docs.python.org/3/library/json.html
- PyTorch AMP examples (autocast/GradScaler behavior): https://pytorch.org/docs/stable/notes/amp_examples.html
- PyTorch `clip_grad_norm_` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- PyTorch checkpoint save/load tutorial: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- IETF RFC 8785 (JSON Canonicalization Scheme): https://datatracker.ietf.org/doc/html/rfc8785
- Python `subprocess` docs (timeout and robust command execution): https://docs.python.org/3/library/subprocess.html

## Progress snapshot (after Batch 46)

- Completed: Batch 1 through Batch 46.
- Pending: Batch 47 onward.

## Batch 47 (max 5 files)

- [X] `scripts/phase5_active_learning/active_learning.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_phase5.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_discovery.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/search_materials.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/structure_enumerator.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 47 optimization goals

- Harden script-level AL acquisition utilities against shape mismatch, optional dependency variance, and unstable numeric inputs.
- Improve launcher argument validation for safer preflight/profile execution paths.
- Strengthen discovery/classifier loading compatibility across torch versions and malformed checkpoint payloads.
- Upgrade materials-search query validation so invalid columns/types fail fast before expensive search pipelines.
- Reduce combinatorial enumeration overhead and improve substitution hygiene/deduplication determinism.

## Batch 47 outcomes

- `active_learning.py`:

  - added Gaussian CDF/PDF fallback path when SciPy is unavailable (`math.erf` based), keeping EI functional in minimal environments.
  - strengthened acquisition validation:

    - `acquisition_uncertainty` now enforces mean/std leading-dimension consistency,
    - `acquisition_random` now validates non-negative seed.
  - made AL loop robust to dict-model outputs via `_select_prediction_tensor` in train/eval/MC-dropout paths.
  - hardened multi-objective screening to handle missing objectives, non-finite predictions, and mixed tensor/dict model outputs.
- `run_phase5.py`:

  - added explicit validation for preflight property group schema (`[A-Za-z0-9._-]+`) and non-empty `--results-dir`.
- `run_discovery.py`:

  - improved classifier checkpoint loading compatibility:

    - try `torch.load(..., weights_only=True)` first,
    - fallback to legacy `torch.load(... )` for older torch builds,
    - explicit payload type guard before `load_state_dict`.
- `search_materials.py`:

  - added `_validate_query_columns` to fail fast on unknown criteria/sort columns.
  - added numeric-type enforcement for range filters (reject non-numeric columns with numeric bounds).
  - retained return-code based CLI exit semantics for CI integration.
- `structure_enumerator.py`:

  - strengthened substitution normalization with dedupe, empty-option rejection, and DummySpecies-safe handling.
  - fixed variant cap logic to respect true combinational size when below hard cap.
  - reduced duplicate-filtering overhead by bucketing candidates by reduced formula before `StructureMatcher.group_structures`.

## Tests updated in batch 47

- `tests/unit/active_learning/test_phase5_cli.py`

  - added coverage for invalid preflight property group and empty `--results-dir`.
- `tests/unit/active_learning/test_search_materials_cli.py`

  - added coverage for missing criteria columns, non-numeric range filters, invalid sort column.
- `tests/unit/active_learning/test_structure_enumerator_script.py`

  - added coverage for empty substitution option rejection and option deduplication.
- `tests/unit/active_learning/test_phase6_active_learning.py`

  - added coverage for uncertainty shape mismatch and negative random-seed rejection.

## Verification

- `python -m ruff check scripts/phase5_active_learning/active_learning.py scripts/phase5_active_learning/run_phase5.py scripts/phase5_active_learning/run_discovery.py scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/structure_enumerator.py tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_phase6_active_learning.py`
- `python -m py_compile scripts/phase5_active_learning/active_learning.py scripts/phase5_active_learning/run_phase5.py scripts/phase5_active_learning/run_discovery.py scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/structure_enumerator.py tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_phase6_active_learning.py`
- `python -m pytest -q tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_phase6_active_learning.py tests/unit/active_learning/test_test_enumeration_script.py`

## Research references used in batch 47

- SciPy normal distribution API (`cdf`/`pdf` contract): https://docs.scipy.org/doc/scipy-1.7.0/reference/generated/scipy.stats.norm.html
- Bayesian Optimization / Expected Improvement (EGO, 1998): https://doi.org/10.1023/A:1008306431147
- MC Dropout as Bayesian Approximation (Gal & Ghahramani, 2016): https://proceedings.mlr.press/v48/gal16.html
- NumPy modern RNG (`default_rng`) docs: https://numpy.org/doc/stable/reference/random/generator.html
- PyTorch model loading / serialization notes: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- pandas numeric dtype checks (`is_numeric_dtype`): https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_numeric_dtype.html
- pymatgen StructureMatcher API (`group_structures` semantics): https://pymatgen.org/pymatgen.analysis
- Python regex syntax/validation patterns: https://docs.python.org/3/library/re.html

## Progress snapshot (after Batch 47)

- Completed: Batch 1 through Batch 47.
- Pending: Batch 48 onward.

## Batch 48 (max 5 files)

- [X] `scripts/phase5_active_learning/test_enumeration.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/active_learning.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/search_materials.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/structure_enumerator.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_test_enumeration_script.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 48 optimization goals

- Strengthen script contract quality for `test_enumeration` so CLI/demo usage is machine-verifiable and CI-friendly.
- Improve AL script model-forward compatibility across heterogeneous graph model signatures.
- Enforce stricter query/output-column validation for search CLI to fail fast on invalid operator/column combinations.
- Make structure enumerator options more deterministic and semantically aligned with exposed parameters.

## Batch 48 outcomes

- `test_enumeration.py`:

  - added contract checks for injected enumerator (`generate(...)` callable required).
  - added summary payload validation (non-empty formula, non-negative integer counts).
  - added `--json` output mode and error-to-exit-code handling for scripting/automation.
- `active_learning.py`:

  - added `_forward_graph_model` dispatch helper to tolerate common model signatures and feature-attribute variants (`edge_attr`/`edge_vec`/none).
  - switched training finite guard from `isnan` to full `isfinite` to reject `inf` loss values as well.
  - stabilized entrypoint with explicit `main() -> int` and `SystemExit` return propagation.
- `search_materials.py`:

  - strengthened argument validation for `--save` and `--columns` (non-empty tokens).
  - extended `_validate_query_columns` to validate requested output columns in addition to criteria/sort fields.
  - made CSV save path robust by creating parent directories when needed.
- `structure_enumerator.py`:

  - normalized substitution keys with whitespace stripping and non-empty enforcement.
  - made `remove_superperiodic=False` behavior explicit by skipping duplicate filtering path.
  - improved determinism by iterating grouped formulas in sorted order before `StructureMatcher` grouping.
  - added explicit warning for `max_index>1` in fallback mode (capability transparency).
- Tests:

  - `test_test_enumeration_script.py` now includes enumerator contract validation path.

## Verification

- `python -m ruff check scripts/phase5_active_learning/test_enumeration.py scripts/phase5_active_learning/active_learning.py scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/structure_enumerator.py tests/unit/active_learning/test_test_enumeration_script.py`
- `python -m py_compile scripts/phase5_active_learning/test_enumeration.py scripts/phase5_active_learning/active_learning.py scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/structure_enumerator.py tests/unit/active_learning/test_test_enumeration_script.py`
- `python -m pytest -q tests/unit/active_learning/test_test_enumeration_script.py tests/unit/active_learning/test_phase6_active_learning.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_structure_enumerator_script.py`

## Research references used in batch 48

- SciPy normal distribution API (`cdf`/`pdf`): https://docs.scipy.org/doc/scipy-1.7.0/reference/generated/scipy.stats.norm.html
- Bayesian Optimization / EI (EGO, Jones et al., 1998): https://doi.org/10.1023/A:1008306431147
- MC Dropout as Bayesian Approximation (Gal & Ghahramani, 2016): https://proceedings.mlr.press/v48/gal16.html
- PyTorch finite-check API (`torch.isfinite`): https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- PyTorch Huber loss API: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.huber_loss.html
- NumPy Generator API (`default_rng`): https://numpy.org/doc/stable/reference/random/generator.html
- pandas numeric dtype guard (`is_numeric_dtype`): https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_numeric_dtype.html
- pymatgen `StructureMatcher.group_structures` docs: https://pymatgen.org/pymatgen.analysis.html
- StructureMatcher robustness paper (OpenReview 2025): https://openreview.net/forum?id=ss5taK9Iy6
- Python argparse docs (CLI contract design): https://docs.python.org/3/library/argparse.html

## Progress snapshot (after Batch 48)

- Completed: Batch 1 through Batch 48.
- Pending: Batch 49 onward.

## Batch 49 (max 5 files)

- [X] `scripts/phase5_active_learning/run_phase5.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_discovery.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_phase5_cli.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 49 optimization goals

- Strengthen Phase5/Discovery CLI run-directory semantics so conflicting flags fail fast before expensive runtime work.
- Improve classifier checkpoint loading compatibility across common PyTorch checkpoint formats (raw `state_dict`, nested `state_dict`, DataParallel prefix).
- Expand regression coverage for new validation gates and checkpoint-payload extraction behavior.
- Keep launcher behavior deterministic and CI-friendly while preserving backward-compatible default flows.

## Batch 49 outcomes

- `run_phase5.py`:

  - added conflict guard for `--resume` + `--results-dir` (now rejected early with explicit message).
- `run_discovery.py`:

  - added checkpoint parsing helpers:

    - `_looks_like_state_dict(...)`
    - `_extract_classifier_state_dict(...)`
  - classifier loader now supports:

    - direct state dict payloads,
    - nested payloads (`state_dict` / `model_state_dict` / `model`),
    - `module.` key-prefix normalization from DataParallel checkpoints.
  - added run-directory conflict guards in discovery validation:

    - `--run-id` + `--results-dir` rejected,
    - `--resume` + `--results-dir` rejected.
- `test_phase5_cli.py`:

  - added `_phase5_args(...)` helper for stable/complete Namespace construction in validation tests.
  - added regression tests for:

    - conflicting run-directory flags in both launchers,
    - conflicting preflight-mode flags,
    - classifier checkpoint extraction for nested/DataParallel payloads,
    - invalid checkpoint payload rejection.

## Verification

- `python -m ruff check scripts/phase5_active_learning/run_phase5.py scripts/phase5_active_learning/run_discovery.py tests/unit/active_learning/test_phase5_cli.py`
- `python -m py_compile scripts/phase5_active_learning/run_phase5.py scripts/phase5_active_learning/run_discovery.py tests/unit/active_learning/test_phase5_cli.py`
- `python -m pytest -q tests/unit/active_learning/test_phase5_cli.py`
- `python -m pytest -q tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_phase6_active_learning.py`

## Research references used in batch 49

- Python `argparse` docs (CLI contract and mutually exclusive behavior patterns): https://docs.python.org/3/library/argparse.html
- Python `pathlib` docs (path semantics used in launcher runtime resolution): https://docs.python.org/3/library/pathlib.html
- Python `subprocess` docs (launcher process execution contract): https://docs.python.org/3/library/subprocess.html
- PyTorch "Saving and Loading Models" tutorial (checkpoint format patterns): https://pytorch.org/tutorials/beginner/saving_loading_models.html
- PyTorch serialization notes (`torch.load` behavior and compatibility): https://pytorch.org/docs/stable/notes/serialization.html
- PyTorch `DataParallel` API reference (module wrapper behavior impacting checkpoint keys): https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
- OWASP Path Traversal overview (input hardening rationale for run-id/path-like flags): https://owasp.org/www-community/attacks/Path_Traversal
- CWE-22 (Path Traversal) reference taxonomy: https://cwe.mitre.org/data/definitions/22.html

## Progress snapshot (after Batch 49)

- Completed: Batch 1 through Batch 49.
- Pending: Batch 50 onward.

## Batch 50 (max 5 files)

- [X] `scripts/phase5_active_learning/test_enumeration.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/structure_enumerator.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_test_enumeration_script.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_structure_enumerator_script.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 50 optimization goals

- Improve fallback structure-enumeration quality under capped combinatorial spaces by reducing prefix-truncation bias.
- Strengthen script-level `test_enumeration` contract so injected enumerators must return valid `pymatgen.Structure` sequences.
- Add regression tests proving the new deterministic capped-space sampling and payload contract checks.
- Keep fallback behavior deterministic and CI-friendly for Windows/non-compiled environments.

## Batch 50 outcomes

- `structure_enumerator.py`:

  - introduced typed species-option alias for clearer substitution contract (`str | DummySpecies`).
  - added deterministic stratified ordinal sampler:

    - `_select_variant_ordinals(total_variants, limit)`
    - `_decode_variant_ordinal(ordinal, radices)`
  - replaced prefix `itertools.product` truncation with mixed-radix ordinal decoding so capped runs sample across higher-order substitution dimensions (not only early lexicographic prefix).
  - preserved existing optional duplicate filtering and fallback safety semantics.
- `test_enumeration.py`:

  - added `_validate_generated_structures(...)` to enforce `generate(...) -> sequence[Structure]` contract at runtime.
  - both simple substitution and vacancy substitution paths now validate generated payload types before summary construction.
- Tests:

  - `test_test_enumeration_script.py`:

    - added failure-path test for non-Structure `generate(...)` payloads.
  - `test_structure_enumerator_script.py`:

    - added direct regression for stratified ordinal selection.
    - added capped-space regression ensuring generated set covers both Ti and Zr substitutions when truncation is active.

## Verification

- `python -m ruff check scripts/phase5_active_learning/structure_enumerator.py scripts/phase5_active_learning/test_enumeration.py tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_test_enumeration_script.py`
- `python -m py_compile scripts/phase5_active_learning/structure_enumerator.py scripts/phase5_active_learning/test_enumeration.py tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_test_enumeration_script.py`
- `python -m pytest -q tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_test_enumeration_script.py`

## Research references used in batch 50

- Python `itertools.product` docs (Cartesian-product baseline behavior): https://docs.python.org/3/library/itertools.html#itertools.product
- Python typing `TypeAlias` docs: https://docs.python.org/3/library/typing.html#typing.TypeAlias
- pymatgen `Structure.replace` API docs: https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure.replace
- pymatgen `StructureMatcher.group_structures` API docs: https://pymatgen.org/pymatgen.analysis.html#pymatgen.analysis.structure_matcher.StructureMatcher.group_structures
- Bergstra & Bengio (2012), random search vs grid-search coverage rationale: https://jmlr.org/beta/papers/v13/bergstra12a.html
- McKay, Beckman, Conover (1979), space-filling/stratified sampling (Latin Hypercube): https://www.osti.gov/biblio/5236110
- Cheon et al. (2025), StructureMatcher robustness analysis: https://openreview.net/forum?id=ss5taK9Iy6

## Progress snapshot (after Batch 50)

- Completed: Batch 1 through Batch 50.
- Pending: Batch 51 onward.

## Batch 51 (max 5 files)

- [X] `atlas/benchmark/cli.py` - reviewed + optimized
- [X] `tests/unit/benchmark/test_benchmark_cli.py` - reviewed + optimized
- [X] `atlas/console_style.py` - reviewed + optimized
- [X] `tests/unit/test_console_style.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 51 optimization goals

- Harden benchmark CLI argument governance to fail fast on conflicting output/preflight modes and malformed preflight controls.
- Improve checkpoint loading robustness/security posture by supporting common checkpoint wrappers while preferring safer torch loading behavior.
- Expand benchmark CLI regression tests to lock new validation and checkpoint-compatibility behavior.
- Improve console styling consistency for multi-digit phase headers and add explicit environment kill-switch for style injection.

## Batch 51 outcomes

- `benchmark/cli.py`:

  - added stricter CLI validation for:

    - `--bootstrap-seed >= 0`
    - `--preflight-split-seed >= 0`
    - `--preflight-timeout-sec > 0`
    - non-empty + safe `--preflight-property-group` schema
    - `--folds` requires at least one index when provided
    - `--output` and `--output-dir` mutual exclusion
    - `--preflight-only` and `--skip-preflight` conflict rejection
  - added `--preflight-timeout-sec` CLI option and wired it into `run_preflight(...)`.
  - improved preflight failure logging with structured detail (`error_message`).
  - introduced robust checkpoint extraction helpers:

    - `_looks_like_state_dict(...)`
    - `_extract_state_dict(...)`
  - `_load_model(...)` now prefers `torch.load(..., weights_only=True)` with fallback for older torch, supports nested checkpoint containers, and normalizes DataParallel `module.` prefixes.
- `test_benchmark_cli.py`:

  - added regression tests for new validation gates (invalid property group, output conflict, preflight mode conflict).
  - added checkpoint extraction tests (nested/DataParallel payload success + invalid payload failure).
  - added end-to-end `_load_model(...)` compatibility test with synthetic nested DataParallel checkpoint.
- `console_style.py`:

  - phase header regex upgraded from single-digit (`[Phase1]`) to multi-digit (`[Phase10]`) coverage.
  - added `ATLAS_CONSOLE_STYLE=0/false/no` environment kill-switch to skip global print wrapping.
- `test_console_style.py`:

  - added regression test for style disable env behavior.
  - added regression test verifying multi-digit phase header styling.

## Verification

- `python -m ruff check atlas/benchmark/cli.py atlas/console_style.py tests/unit/benchmark/test_benchmark_cli.py tests/unit/test_console_style.py`
- `python -m py_compile atlas/benchmark/cli.py atlas/console_style.py tests/unit/benchmark/test_benchmark_cli.py tests/unit/test_console_style.py`
- `python -m pytest -q tests/unit/benchmark/test_benchmark_cli.py tests/unit/test_console_style.py`

## Research references used in batch 51

- Python argparse docs (CLI validation and parser error contracts): https://docs.python.org/3/library/argparse.html
- Python importlib docs (dynamic module loading semantics): https://docs.python.org/3/library/importlib.html
- Python pathlib docs (path validation semantics): https://docs.python.org/3/library/pathlib.html
- PyTorch serialization notes (`torch.load`, `weights_only` behavior): https://pytorch.org/docs/stable/notes/serialization.html
- PyTorch DataParallel docs (`module.` prefix behavior in state_dict): https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
- Matbench benchmark paper (objective/benchmarking context): https://arxiv.org/abs/2005.00707
- NO_COLOR convention (terminal color disable interoperability): https://no-color.org/
- ECMA-48 / ANSI escape sequence reference context: https://en.wikipedia.org/wiki/ANSI_escape_code

## Progress snapshot (after Batch 51)

- Completed: Batch 1 through Batch 51.
- Pending: Batch 52 onward.

## Batch 52 (max 5 files)

- [X] `atlas/training/checkpoint.py` - reviewed + optimized
- [X] `tests/unit/training/test_checkpoint.py` - reviewed + optimized
- [X] `atlas/training/filters.py` - reviewed + optimized
- [X] `tests/unit/training/test_filters.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 52 optimization goals

- Improve checkpoint-manager resume consistency so top-k state survives process restart and stays synchronized with on-disk artifacts.
- Harden checkpoint pointer update path to avoid partial `best.pt` writes.
- Fix outlier-filter API type hazard where string inputs for `properties` are silently treated as character sequences.
- Add regression tests for resume rehydration and stricter filter argument contracts.

## Batch 52 outcomes

- `training/checkpoint.py`:

  - added robust best-checkpoint filename parser and disk rehydration flow on manager init:

    - `_parse_best_filename(...)`
    - `_sync_best_models_from_disk(...)`
  - manager now discovers pre-existing `best_epoch_*_mae_*.pt`, sorts/prunes to `top_k`, and refreshes `best.pt` pointer accordingly.
  - added `_atomic_copy(...)` and replaced direct `shutil.copy2` for `best.pt` updates to keep pointer writes atomic.
- `training/filters.py`:

  - fixed `properties` type contract:

    - reject `str`/`bytes` as invalid top-level property sequence input,
    - enforce each property entry is a string before normalization.
  - prevents silent misconfiguration (e.g., `properties="target"` becoming `["t", "a", ...]`).
- Tests:

  - `test_checkpoint.py`:

    - added resume-state regression proving manager rehydrates from existing best files, prunes overflow, and points `best.pt` to the best epoch.
  - `test_filters.py`:

    - added argument-contract tests for invalid string payload and non-string property entries.

## Verification

- `python -m ruff check atlas/training/checkpoint.py atlas/training/filters.py tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py`
- `python -m py_compile atlas/training/checkpoint.py atlas/training/filters.py tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py`
- `python -m pytest -q tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py`

## Research references used in batch 52

- PyTorch serialization notes (`state_dict` best practice and `weights_only` loading guidance): https://docs.pytorch.org/docs/stable/notes/serialization.html
- PyTorch save/load tutorial (`state_dict` workflow): https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
- Python `tempfile` docs (`NamedTemporaryFile` behavior and safety notes): https://docs.python.org/3/library/tempfile.html
- Python `pathlib` docs (`Path.replace` semantics): https://docs.python.org/3/library/pathlib.html
- SciPy `median_abs_deviation` docs (MAD robustness rationale): https://docs.scipy.org/doc/scipy-1.9.0/reference/generated/scipy.stats.median_abs_deviation.html
- NIST/SEMATECH e-Handbook (robust scale measures, MAD motivation): https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm
- Iglewicz & Hoaglin (1993) reference summary for modified z-score thresholding context: https://rdrr.io/github/fosterlab/modern/man/iglewicz_hoaglin.html

## Progress snapshot (after Batch 52)

- Completed: Batch 1 through Batch 52.
- Pending: Batch 53 onward.

## Batch 53 (max 5 files)

- [X] `atlas/training/__init__.py` - reviewed + optimized
- [X] `tests/unit/training/test_init_exports.py` - reviewed + optimized
- [X] `atlas/models/__init__.py` - reviewed + optimized
- [X] `tests/unit/models/test_model_utils.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 53 optimization goals

- Improve lazy-export API stability for `atlas.training` and `atlas.models` package surfaces.
- Make lazy-export registries immutable to reduce accidental mutation risk during runtime.
- Ensure lazy exports are cached after first resolution to avoid repeated import/getattr overhead.
- Add regression tests for cache behavior, export-surface integrity, and clearer mismatch diagnostics.

## Batch 53 outcomes

- `training/__init__.py`:

  - switched export registry to `MappingProxyType` for immutable lazy-export mapping.
  - `__getattr__` now caches resolved exports into module globals and raises clearer mismatch errors when target modules miss expected attributes.
  - `__dir__` now uses explicit global key set union for stable symbol reporting.
- `models/__init__.py`:

  - switched export mapping to immutable `MappingProxyType`.
  - improved `__getattr__` mismatch diagnostics when target module export contract is violated.
  - retained and validated global-cache behavior for resolved exports.
- Tests:

  - `test_init_exports.py`:

    - added lazy-export cache test,
    - added `__all__` integrity check (known symbols + uniqueness),
    - added export-mismatch diagnostic test via monkeypatched import path.
  - `test_model_utils.py`:

    - added package-level lazy-export cache regression for `atlas.models`,
    - added unknown-attribute failure regression,
    - added export-mismatch diagnostic regression via monkeypatched module loader.

## Verification

- `python -m ruff check atlas/training/__init__.py atlas/models/__init__.py tests/unit/training/test_init_exports.py tests/unit/models/test_model_utils.py`
- `python -m py_compile atlas/training/__init__.py atlas/models/__init__.py tests/unit/training/test_init_exports.py tests/unit/models/test_model_utils.py`
- `python -m pytest -q tests/unit/training/test_init_exports.py tests/unit/models/test_model_utils.py`

## Research references used in batch 53

- Python Data Model docs (`module.__getattr__` / module customization): https://docs.python.org/3/reference/datamodel.html#customizing-module-attribute-access
- PEP 562 (module `__getattr__` and `__dir__`): https://peps.python.org/pep-0562/
- Python `importlib` docs (dynamic module import semantics): https://docs.python.org/3/library/importlib.html
- Python `types.MappingProxyType` docs (read-only mapping views): https://docs.python.org/3/library/stdtypes.html#types.MappingProxyType
- Python `__all__` and import system semantics: https://docs.python.org/3/reference/simple_stmts.html#the-import-statement
- PEP 8 public/internal interface guidance (`__all__`): https://peps.python.org/pep-0008/#public-and-internal-interfaces
- Scientific Python SPEC 1 (lazy loading rationale in scientific packages): https://scientific-python.org/specs/spec-0001/

## Progress snapshot (after Batch 53)

- Completed: Batch 1 through Batch 53.
- Pending: Batch 54 onward.

## Batch 54 (max 5 files)

- [X] `atlas/training/metrics.py` - reviewed + optimized
- [X] `tests/unit/training/test_metrics.py` - reviewed + optimized
- [X] `atlas/training/normalizers.py` - reviewed + optimized
- [X] `tests/unit/training/test_normalizers.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 54 optimization goals

- Improve metric API consistency by normalizing metric-key prefixes across scalar/classification/tensor outputs.
- Harden classification metric output stability against non-finite metric-library returns.
- Strengthen normalizer dataset/property contracts to prevent silent misuse (mapping datasets and non-string property names).
- Add regression tests covering the new contracts and deterministic key-prefix behavior.

## Batch 54 outcomes

- `training/metrics.py`:

  - added `_normalize_prefix(...)` to normalize/trim prefixes consistently across all metric groups.
  - added `_safe_float(...)` for classification metric outputs to guard against non-finite library returns.
  - applied normalized-prefix behavior to:

    - `scalar_metrics(...)`
    - `classification_metrics(...)`
    - `tensor_metrics(...)`
- `training/normalizers.py`:

  - `_iter_dataset(...)` now supports mapping-style datasets by iterating over `dataset.values()` instead of keys.
  - `_normalize_properties(...)` now enforces string-only property names (fails fast on invalid types).
  - `MultiTargetNormalizer.load_state_dict(...)` now:

    - iterates properties in sorted order for deterministic behavior,
    - validates each property state is a mapping and raises property-specific errors when malformed.
- Tests:

  - `test_metrics.py`:

    - updated prefix tests to validate trimming/normalization behavior.
    - added classification prefix normalization regression.
  - `test_normalizers.py`:

    - added mapping-dataset support regression for `TargetNormalizer`.
    - added constructor guard for non-string multi-property names.
    - added malformed per-property state rejection test.

## Verification

- `python -m ruff check atlas/training/metrics.py atlas/training/normalizers.py tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py`
- `python -m py_compile atlas/training/metrics.py atlas/training/normalizers.py tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py`
- `python -m pytest -q tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py`

## Research references used in batch 54

- scikit-learn `accuracy_score` API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
- scikit-learn `precision_score` API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
- scikit-learn `recall_score` API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
- scikit-learn `f1_score` API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- scikit-learn `roc_auc_score` API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
- SciPy `spearmanr` API reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
- NumPy `std` reference: https://numpy.org/doc/stable/reference/generated/numpy.std.html
- NumPy `isfinite` reference: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- Python Mapping ABC docs (`collections.abc.Mapping`): https://docs.python.org/3/library/collections.abc.html
- scikit-learn `StandardScaler` reference (normalization semantics): https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

## Progress snapshot (after Batch 54)

- Completed: Batch 1 through Batch 54.
- Pending: Batch 55 onward.

## Batch 55 (max 5 files)

- [X] `atlas/training/losses.py` - reviewed + optimized
- [X] `tests/unit/training/test_losses.py` - reviewed + optimized
- [X] `atlas/training/preflight.py` - reviewed + optimized
- [X] `tests/unit/training/test_preflight.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 55 optimization goals

- Improve multi-task loss configuration safety by rejecting misspelled/unknown task keys in `task_types/task_weights/constraints`.
- Stabilize zero-label/zero-task minibatch behavior so returned total loss remains backward-safe.
- Harden evidential-loss tensor alignment to prevent silent broadcasting bugs under mismatched shapes.
- Improve preflight diagnostics with structured stage-level error reasons (timeout/OS error) for faster CI triage.

## Batch 55 outcomes

- `training/losses.py`:

  - added mapping normalization + key validation helpers:

    - `_normalize_named_mapping(...)`
    - `_validate_known_task_keys(...)`
  - `MultiTaskLoss` now normalizes config-key names and fails fast on unknown tasks in:

    - `task_types`
    - `task_weights`
    - `constraints`
  - `MultiTaskLoss.forward(...)` now initializes `total` with `_zero_loss(...)` so empty-task batches still produce a grad-enabled scalar.
  - `EvidentialLoss.forward(...)` now validates `pred/target` types, flattens and length-aligns tensors across `gamma/nu/alpha/beta/target`, and handles empty aligned windows robustly.
  - `PropertyLoss` BCE path now clamps targets to `[0, 1]` before `binary_cross_entropy_with_logits` to prevent invalid-label blowups.
- `training/preflight.py`:

  - introduced command result model (`_CommandResult`) with `error_reason`.
  - `_run_command(...)` now returns structured outcomes (`return_code`, `error_reason`).
  - added `_format_stage_error(...)` + `_run_stage(...)` helpers to centralize stage execution and failure wrapping.
  - stage failures now emit detailed error messages:

    - `validate-data failed: timeout`
    - `validate-data failed: oserror:FileNotFoundError`
    - `make-splits failed: timeout`
- Tests:

  - `test_losses.py`:

    - added BCE clamping regression,
    - added evidential mismatched-shape alignment regression,
    - added unknown-task key rejection tests for `task_types/task_weights/constraints`,
    - strengthened empty-prediction loss expectation (`requires_grad=True`).
  - `test_preflight.py`:

    - updated existing assertions for new detailed stage error messages,
    - added split-stage timeout regression.

## Verification

- `python -m ruff check atlas/training/losses.py atlas/training/preflight.py tests/unit/training/test_losses.py tests/unit/training/test_preflight.py`
- `python -m py_compile atlas/training/losses.py atlas/training/preflight.py tests/unit/training/test_losses.py tests/unit/training/test_preflight.py`
- `python -m pytest -q tests/unit/training/test_losses.py tests/unit/training/test_preflight.py`

## Research references used in batch 55

- Kendall, Gal, Cipolla (CVPR 2018), uncertainty-based multi-task weighting: https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html
- Amini et al. (NeurIPS 2020), deep evidential regression: https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html
- PyTorch BCE-with-logits API contract: https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
- PyTorch `torch.isfinite` reference: https://pytorch.org/docs/stable/generated/torch.isfinite.html
- Python `subprocess.run` docs (timeout/error behavior): https://docs.python.org/3/library/subprocess.html
- Python `pathlib` docs (`Path` filesystem semantics): https://docs.python.org/3/library/pathlib.html
- Python `dataclasses` docs (structured result objects): https://docs.python.org/3/library/dataclasses.html
- PyTorch `torch.clamp` reference: https://pytorch.org/docs/stable/generated/torch.clamp.html

## Progress snapshot (after Batch 55)

- Completed: Batch 1 through Batch 55.
- Pending: Batch 56 onward.

## Batch 56 (max 5 files)

- [X] `atlas/training/trainer.py` - reviewed + optimized
- [X] `tests/unit/training/test_trainer.py` - reviewed + optimized
- [X] `atlas/training/run_utils.py` - reviewed + optimized
- [X] `tests/unit/training/test_run_utils_manifest.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 56 optimization goals

- Improve checkpoint resume quality by persisting trainer/scheduler/scaler state in a structured payload.
- Harden checkpoint loading with strict payload-shape validation and explicit optional-state restore flags.
- Fail fast on empty train/validation loaders to prevent silent 0-loss runs from contaminating CI signals.
- Strengthen run-directory + run-manifest schema safety (prefix/path validation and manifest root-type validation).

## Batch 56 outcomes

- `training/trainer.py`:

  - added checkpoint schema marker (`schema_version`) and richer resume payload:

    - `optimizer_state_dict`
    - `scheduler_state_dict`
    - `scaler_state_dict`
    - `trainer_state` (`best_val_loss`, `patience_counter`, `history`)
  - added `_validate_checkpoint_payload(...)` to enforce mapping/finite-value contracts before state restore.
  - upgraded `load_checkpoint(...)` with opt-in restoration flags:

    - `restore_optimizer`
    - `restore_scheduler`
    - `restore_scaler`
    - `restore_trainer_state`
    - `strict`
  - switched checkpoint load path to prefer `torch.load(..., weights_only=True)` with compatibility fallback.
  - added explicit empty-loader guards in `train_epoch(...)` and `validate(...)`.
- `test_trainer.py`:

  - added empty-loader rejection tests for both train/validate loops.
  - added checkpoint-resume coverage for persisted trainer state and optimizer restore.
  - added optional-state missing-key regression tests when restore flags are requested.
- `training/run_utils.py`:

  - added `_validate_run_prefix(...)` and wired it through run-dir helpers (`list_run_dirs`, `latest_run_dir`, `resolve_run_dir`, timestamp creation).
  - added `_validate_manifest_payload(...)` to enforce required root fields/types and visibility correctness before write.
  - `write_run_manifest(...)` now validates serialized manifest payload shape before emitting JSON/YAML mirror.
- `test_run_utils_manifest.py`:

  - added invalid-prefix rejection regression.
  - added runtime-context shape regression to ensure invalid runtime payloads fail fast.

## Verification

- `python -m ruff check atlas/training/trainer.py atlas/training/run_utils.py tests/unit/training/test_trainer.py tests/unit/training/test_run_utils_manifest.py`
- `python -m py_compile atlas/training/trainer.py atlas/training/run_utils.py tests/unit/training/test_trainer.py tests/unit/training/test_run_utils_manifest.py`
- `python -m pytest -q tests/unit/training/test_trainer.py tests/unit/training/test_run_utils_manifest.py`

## Research references used in batch 56

- PyTorch serialization notes (`weights_only`, state_dict best practices): https://pytorch.org/docs/stable/notes/serialization.html
- PyTorch save/load tutorial (checkpoint contracts): https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
- PyTorch `torch.load` API reference: https://pytorch.org/docs/stable/generated/torch.load.html
- Python `tempfile.NamedTemporaryFile` docs (safe temp-write patterns): https://docs.python.org/3/library/tempfile.html
- Python `pathlib.Path.replace` docs (atomic replace semantics): https://docs.python.org/3/library/pathlib.html#pathlib.Path.replace
- RFC 8785 (JSON Canonicalization Scheme): https://www.rfc-editor.org/rfc/rfc8785
- NeurIPS ML Reproducibility Checklist (experiment/manifest reporting standards): https://neurips.cc/public/guides/PaperChecklist

## Progress snapshot (after Batch 56)

- Completed: Batch 1 through Batch 56.
- Pending: Batch 57 onward.

## Batch 57 (max 5 files)

- [X] `atlas/utils/reproducibility.py` - reviewed + optimized
- [X] `tests/unit/research/test_reproducibility.py` - reviewed + optimized
- [X] `atlas/research/workflow_reproducible_graph.py` - reviewed + optimized
- [X] `tests/unit/research/test_workflow_reproducible_graph.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 57 optimization goals

- Align runtime determinism toggles with actual environment state to avoid stale deterministic env flags leaking across runs.
- Strengthen workflow manifest durability and strict-JSON compliance (atomic write + non-finite sanitization).
- Prevent run-manifest overwrite risk under same-second run-id collisions.
- Expand reproducibility tests for deterministic-env transitions and workflow manifest edge cases.

## Batch 57 outcomes

- `utils/reproducibility.py`:

  - added `_configure_cublas_workspace(...)` to keep `CUBLAS_WORKSPACE_CONFIG` consistent with deterministic mode:

    - deterministic on: set default `:4096:8` when absent,
    - deterministic off: clear only known deterministic defaults (`:16:8`, `:4096:8`) while preserving custom user values.
  - moved CUBLAS env handling out of torch-specific branch so CPU-only runs also maintain consistent metadata.
  - `set_global_seed(...)` now reports post-configuration CUBLAS state in returned metadata.
- `research/workflow_reproducible_graph.py`:

  - added strict manifest serialization helpers:

    - `_json_safe(...)` (non-finite float -> `None`, path-safe conversion, deterministic set ordering),
    - `_atomic_json_write(...)` (`allow_nan=False`, sorted keys, flush + `fsync`, atomic replace).
  - added stage/method normalization helpers:

    - `_normalize_stage_plan(...)`,
    - `_normalize_fallback_methods(...)` (prevents string payloads from being split char-by-char).
  - hardened `RunManifest.__post_init__` contracts:

    - validates/sanitizes `seed`, `started_at`, `ended_at`, `status`, `stage_plan`, `fallback_methods`.
  - added `_resolve_manifest_path(...)` to avoid filename collisions by suffixing `_01`, `_02`, ... when needed.
  - persistence now validates serialized manifest payload is mapping before write.
- Tests:

  - `test_reproducibility.py`:

    - added regression tests for deterministic-off behavior:

      - known deterministic CUBLAS config is cleared,
      - custom CUBLAS config is preserved.
  - `test_workflow_reproducible_graph.py`:

    - added run-manifest filename-collision test,
    - added non-finite metric sanitization test (`NaN` -> `null`),
    - added invalid finalize status rejection test.

## Verification

- `python -m ruff check atlas/utils/reproducibility.py atlas/research/workflow_reproducible_graph.py tests/unit/research/test_reproducibility.py tests/unit/research/test_workflow_reproducible_graph.py`
- `python -m py_compile atlas/utils/reproducibility.py atlas/research/workflow_reproducible_graph.py tests/unit/research/test_reproducibility.py tests/unit/research/test_workflow_reproducible_graph.py`
- `python -m pytest -q tests/unit/research/test_reproducibility.py tests/unit/research/test_workflow_reproducible_graph.py`

## Research references used in batch 57

- PyTorch Reproducibility notes: https://docs.pytorch.org/docs/stable/notes/randomness.html
- PyTorch deterministic algorithms API (`torch.use_deterministic_algorithms`): https://docs.pytorch.org/docs/2.9/generated/torch.use_deterministic_algorithms.html
- NumPy RNG seeding guidance (`numpy.random.seed`, legacy vs Generator): https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
- NumPy `SeedSequence` (reproducible entropy mixing/spawn): https://numpy.org/doc/2.1/reference/random/bit_generators/generated/numpy.random.SeedSequence.html
- Python `tempfile` docs (`NamedTemporaryFile` semantics): https://docs.python.org/3/library/tempfile.html
- Python `os.replace` docs (atomic replace guarantee on success): https://docs.python.org/3.13/library/os.html#os.replace
- RFC 8259 (JSON, NaN/Infinity not permitted): https://www.rfc-editor.org/rfc/rfc8259
- RFC 8785 (JSON canonicalization / deterministic representation): https://datatracker.ietf.org/doc/html/rfc8785
- Henderson et al., "Deep Reinforcement Learning that Matters" (AAAI 2018): https://arxiv.org/abs/1709.06560
- Pineau et al., "Improving Reproducibility in Machine Learning Research" (JMLR 2021): https://www.jmlr.org/papers/v22/20-303.html

## Progress snapshot (after Batch 57)

- Completed: Batch 1 through Batch 57.
- Pending: Batch 58 onward.

## Batch 58 (max 5 files)

- [X] `atlas/training/physics_losses.py` - reviewed + optimized
- [X] `tests/unit/training/test_physics_losses.py` - reviewed + optimized
- [X] `atlas/thermo/__init__.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_init_exports.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 58 optimization goals

- Eliminate finite-value alignment drift in Voigt/Reuss bound penalties so per-sample pairing remains physically valid.
- Keep physics-only loss terms numerically finite while preserving gradient connectivity for edge cases with all-invalid inputs.
- Harden thermo lazy-export behavior to avoid silently swallowing export-contract mismatches.
- Expand regression coverage for the above stability and API-contract improvements.

## Batch 58 outcomes

- `training/physics_losses.py`:

  - added `_aligned_finite_vectors(...)` for index-aligned finite filtering across `K/G` and Voigt/Reuss bounds.
  - replaced independent flatten+truncate logic with joint finite mask logic in `VoigtReussBoundsLoss.forward(...)` to prevent sample mispairing.
  - added `_zero_loss_from_inputs(...)` + `nan_to_num` sanitation so all-invalid paths return finite zero while preserving grad path.
  - introduced `_BOUND_TOL` constant to centralize bound tolerance.
  - `PhysicsConstraintLoss.forward(...)` now guarantees finite-safe fallback for inactive/invalid constraint paths without breaking autograd.
- `test_physics_losses.py`:

  - added joint-finite-mask regression that catches previous index-misalignment behavior.
  - added all-invalid finite fallback + grad-path regression.
- `thermo/__init__.py`:

  - `__getattr__` now:

    - reuses cached globals when already resolved,
    - raises explicit `AttributeError` if imported module misses expected export symbol,
    - caches successful lazy exports to avoid repeated imports.
  - preserves optional-dependency behavior (`ImportError` -> `None`) for backwards compatibility.
- `test_init_exports.py`:

  - added lazy-export cache regression.
  - added missing-export contract regression (must raise `AttributeError`, no silent `None`).

## Verification

- `python -m ruff check atlas/training/physics_losses.py tests/unit/training/test_physics_losses.py atlas/thermo/__init__.py tests/unit/thermo/test_init_exports.py`
- `python -m py_compile atlas/training/physics_losses.py tests/unit/training/test_physics_losses.py atlas/thermo/__init__.py tests/unit/thermo/test_init_exports.py`
- `python -m pytest -q tests/unit/training/test_physics_losses.py tests/unit/thermo/test_init_exports.py`

## Research references used in batch 58

- Mouhat & Coudert (2014), elastic stability criteria: https://doi.org/10.1103/PhysRevB.90.224104
- Hill (1952), elastic aggregate behavior / Voigt-Reuss-Hill context: https://doi.org/10.1088/0370-1298/65/5/307
- Reuss (1929), iso-stress lower-bound formulation: https://doi.org/10.1002/zamm.19290090104
- Voigt (1889), iso-strain upper-bound formulation: https://doi.org/10.1002/andp.18892741206
- PyTorch `torch.nan_to_num` API (finite replacement semantics): https://docs.pytorch.org/docs/stable/generated/torch.nan_to_num.html
- PEP 562 (module-level `__getattr__` / `__dir__` contract): https://peps.python.org/pep-0562/
- Python `importlib.import_module` docs: https://docs.python.org/3/library/importlib.html

## Progress snapshot (after Batch 58)

- Completed: Batch 1 through Batch 58.
- Pending: Batch 59 onward.

## Batch 59 (max 5 files)

- [X] `atlas/thermo/calphad.py` - reviewed + optimized
- [X] `atlas/thermo/stability.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_calphad.py` - added + optimized
- [X] `tests/unit/thermo/test_stability.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 59 optimization goals

- Stabilize CALPHAD phase-fraction extraction so equilibrium outputs remain physically interpretable under noisy solver outputs.
- Improve transus (liquidus/solidus) estimation robustness via finite filtering + interpolation.
- Strengthen phase-stability API contracts (element normalization, payload validation, non-finite PD result guards).
- Add dedicated thermo unit tests to lock deterministic behavior and regression-proof numeric edge cases.

## Batch 59 outcomes

- `thermo/calphad.py`:

  - added `_normalize_phase_fractions(...)`:

    - filters invalid phase labels/values,
    - merges duplicate phase labels,
    - renormalizes when total fraction exceeds physical bounds,
    - returns deterministic sorted phase map.
  - `equilibrium_at(...)` now routes raw pycalphad fractions through normalized phase-fraction post-processing before returning.
  - upgraded `_find_transus(...)`:

    - finite-mask filtering,
    - clipping to `[0, 1]`,
    - temperature sorting for cooling trajectory consistency,
    - linear interpolation at threshold crossings (`0.99`, `0.01`) for more stable liquidus/solidus estimation.
- `thermo/stability.py`:

  - added `_normalize_element_symbol(...)` and applied case normalization in `get_entries(...)` for case-insensitive chemical-system matching.
  - hardened `ReferenceDatabase` input contracts:

    - non-empty formula requirement in `add_entry(...)`,
    - mapping-type validation in `load_from_list(...)`.
  - `analyze_stability(...)` now:

    - uses reduced target formula as decomposition fallback when decomposition map is empty,
    - rejects non-finite `e_above_hull` / `formation_energy` outputs with explicit error path.
- Tests:

  - new `test_calphad.py`:

    - phase-fraction normalization/renormalization regression,
    - transus interpolation regression,
    - equilibrium result normalization regression via mocked `pycalphad` module.
  - new `test_stability.py`:

    - case-insensitive entry filtering regression,
    - invalid list-item type rejection,
    - decomposition fallback behavior,
    - non-finite phase-diagram output guard behavior.

## Verification

- `python -m ruff check atlas/thermo/calphad.py atlas/thermo/stability.py tests/unit/thermo/test_calphad.py tests/unit/thermo/test_stability.py`
- `python -m py_compile atlas/thermo/calphad.py atlas/thermo/stability.py tests/unit/thermo/test_calphad.py tests/unit/thermo/test_stability.py`
- `python -m pytest -q tests/unit/thermo/test_calphad.py tests/unit/thermo/test_stability.py`

## Research references used in batch 59

- Saunders & Miodownik, *CALPHAD (Calculation of Phase Diagrams): A Comprehensive Guide*: https://www.sciencedirect.com/book/9780080421292/calphad
- Lukas, Fries, Sundman, *Computational Thermodynamics* (CALPHAD methodology): https://doi.org/10.1017/CBO9780511804137
- pycalphad official documentation: https://pycalphad.org/docs/latest/
- pycalphad API docs: https://pycalphad.org/docs/latest/api/
- pymatgen phase diagram docs: https://pymatgen.org/pymatgen.analysis#module-pymatgen.analysis.phase_diagram
- Ong et al. (2013), pymatgen paper: https://doi.org/10.1016/j.commatsci.2012.10.028
- Sun et al. (2016), thermodynamic-scale stability in inorganic crystals: https://doi.org/10.1126/sciadv.1600225
- NumPy interpolation reference (`numpy.interp`/piecewise linear rationale): https://numpy.org/doc/stable/reference/generated/numpy.interp.html
- NumPy finite filtering (`numpy.isfinite`): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- Python dataclasses docs (result schema robustness): https://docs.python.org/3/library/dataclasses.html

## Progress snapshot (after Batch 59)

- Completed: Batch 1 through Batch 59.
- Pending: Batch 60 onward.

## Batch 60 (max 5 files)

- [X] `atlas/thermo/openmm/engine.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/reporters.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/__init__.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_openmm_stack.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 60 optimization goals

- Harden OpenMM fallback LJ runtime against invalid periodic boundary configurations and unstable cutoff choices.
- Strengthen trajectory reporter contracts to prevent silent frame-buffer corruption and shape mismatches.
- Improve openmm package lazy-export behavior with caching and strict export-contract checks.
- Add deterministic tests that do not require real OpenMM installation (via controlled fake-module injection).

## Batch 60 outcomes

- `thermo/openmm/engine.py`:

  - added `_is_periodic(...)` helper and enforced strict boundary condition policy:

    - non-periodic (all false) OR fully periodic (all true),
    - mixed partial-PBC now fails fast with explicit error.
  - improved LJ periodic cutoff logic:

    - validates finite positive cell lengths,
    - enforces cutoff strictly below half minimum box length (minimum-image safety),
    - rejects pathological tiny periodic boxes with actionable error message.
- `thermo/openmm/reporters.py`:

  - `PymatgenTrajectoryReporter` now validates non-empty structure and stores expected site count.
  - `report(...)` now validates position/force array shape against `(n_sites, 3)`.
  - added `_validate_collected_frames(...)` to guard against inconsistent buffer lengths before trajectory export.
  - `describeNextReport(...)` now handles missing/negative `currentStep` defensively.
  - `get_trajectory(...)` now derives `time_step` only from finite positive deltas (non-monotonic/degenerate times -> `0.0`).
- `thermo/openmm/__init__.py`:

  - `__getattr__` now:

    - returns cached export when already resolved,
    - raises explicit `AttributeError` if imported module lacks expected symbol,
    - caches resolved symbols in module globals.
- Tests (`test_openmm_stack.py`):

  - lazy-export cache regression,
  - missing expected export regression,
  - reporter position-shape mismatch guard,
  - reporter inconsistent-buffer guard,
  - engine partial-PBC rejection and periodic LJ cutoff safety regression.
  - tests run fully in CI without OpenMM installation by injecting fake `openmm/openmm.app/openmm.unit` modules.

## Verification

- `python -m ruff check atlas/thermo/openmm/engine.py atlas/thermo/openmm/reporters.py atlas/thermo/openmm/__init__.py tests/unit/thermo/test_openmm_stack.py`
- `python -m py_compile atlas/thermo/openmm/engine.py atlas/thermo/openmm/reporters.py atlas/thermo/openmm/__init__.py tests/unit/thermo/test_openmm_stack.py`
- `python -m pytest -q tests/unit/thermo/test_openmm_stack.py`

## Research references used in batch 60

- OpenMM User Guide (periodic boundaries / simulation model): https://docs.openmm.org/latest/userguide/
- OpenMM API docs (`NonbondedForce`, cutoff methods): https://docs.openmm.org/latest/api-python/generated/openmm.openmm.NonbondedForce.html
- OpenMM API docs (`Simulation`): https://docs.openmm.org/latest/api-python/generated/openmm.app.simulation.Simulation.html
- Allen & Tildesley, *Computer Simulation of Liquids* (minimum image / cutoff constraints): https://global.oup.com/academic/product/computer-simulation-of-liquids-9780198803201
- Frenkel & Smit, *Understanding Molecular Simulation* (PBC and short-range interactions): https://doi.org/10.1016/B978-0-12-267351-1.X5000-7
- pymatgen trajectory docs: https://pymatgen.org/pymatgen.core.html#module-pymatgen.core.trajectory
- PEP 562 (module lazy attribute hooks): https://peps.python.org/pep-0562/
- Python `importlib` docs: https://docs.python.org/3/library/importlib.html

## Progress snapshot (after Batch 60)

- Completed: Batch 1 through Batch 60.
- Pending: Batch 61 onward.

## Batch 61 (max 5 files)

- [X] `atlas/thermo/openmm/atomate2_wrapper.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/__init__.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_openmm_atomate2_wrapper.py` - added + optimized
- [X] `tests/unit/thermo/test_openmm_stack.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 61 optimization goals

- Harden native Atomate2 OpenMM wrapper contracts for maker selection, step validation, and optional module loading.
- Avoid repeated dynamic-import overhead in wrapper hot paths by caching imported jobs module.
- Tighten OpenMM lazy-export error policy to only degrade on optional dependency absence (ImportError), not internal runtime faults.
- Add deterministic unit tests to lock these behaviors without requiring full OpenMM runtime dependencies.

## Batch 61 outcomes

- `thermo/openmm/atomate2_wrapper.py`:

  - removed fragile eager import block and switched to explicit on-demand loading via `_load_atomate2_jobs_module(...)`.
  - added `_ATOMATE2_JOBS_MODULE` cache so repeated wrapper calls do not repeatedly import job module.
  - added `_coerce_non_negative_int(...)` for robust integer-step parsing and bool rejection.
  - strengthened maker construction semantics:

    - `nvt/npt` require `steps > 0`,
    - `minimize` still supports `steps=0`.
  - `run_simulation(...)` now validates maker interface contract (`callable make(...)`) before invocation.
- `thermo/openmm/__init__.py`:

  - lazy export now only swallows `ImportError` as optional dependency path.
  - runtime import faults (e.g. module init bug) are no longer silently downgraded to `None`.
  - retains symbol caching + explicit missing-export `AttributeError` behavior from previous hardening.
- Tests:

  - new `test_openmm_atomate2_wrapper.py`:

    - init argument guards,
    - dynamic-ensemble step contracts,
    - module-import caching behavior,
    - import-error wrapping behavior,
    - maker interface validation,
    - successful maker invocation path.
  - updated `test_openmm_stack.py`:

    - added regression test that runtime import errors are propagated (not suppressed) in lazy export path.

## Verification

- `python -m ruff check atlas/thermo/openmm/atomate2_wrapper.py atlas/thermo/openmm/__init__.py tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`
- `python -m py_compile atlas/thermo/openmm/atomate2_wrapper.py atlas/thermo/openmm/__init__.py tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`
- `python -m pytest -q tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`

## Research references used in batch 61

- OpenMM User Guide (simulation stack and API behavior): https://docs.openmm.org/latest/userguide/
- OpenMM Python API docs (`Simulation` / app layer): https://docs.openmm.org/latest/api-python/generated/openmm.app.simulation.Simulation.html
- OpenMM Python API docs (`LangevinMiddleIntegrator`): https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LangevinMiddleIntegrator.html
- atomate2 OpenMM docs (maker/workflow integration context): https://materialsproject.github.io/atomate2/
- Python import system docs (`importlib.import_module`): https://docs.python.org/3/library/importlib.html
- PEP 562 (module `__getattr__` lazy export semantics): https://peps.python.org/pep-0562/
- Python logging best practices (standard library docs): https://docs.python.org/3/library/logging.html

## Progress snapshot (after Batch 61)

- Completed: Batch 1 through Batch 61.
- Pending: Batch 62 onward.

## Batch 62 (max 5 files)

- [X] `atlas/thermo/openmm/atomate2_wrapper.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/__init__.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_openmm_atomate2_wrapper.py` - added + optimized
- [X] `tests/unit/thermo/test_openmm_stack.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 62 optimization goals

- Strengthen native Atomate2 OpenMM wrapper contracts (typed step parsing, maker selection, module caching, interface validation).
- Prevent silent masking of non-ImportError module failures in OpenMM lazy exports.
- Add deterministic tests around optional-import paths and wrapper execution contracts.
- Keep behavior backward-compatible for optional dependency absence while surfacing true runtime faults.

## Batch 62 outcomes

- `thermo/openmm/atomate2_wrapper.py`:

  - added module-level cache (`_ATOMATE2_JOBS_MODULE`) for dynamic import reuse.
  - added strict step parsing helper (`_coerce_non_negative_int`) that rejects bool payloads.
  - `_build_maker(...)` now enforces:

    - `nvt`/`npt` require `steps > 0`,
    - `minimize` supports `steps=0`.
  - `run_simulation(...)` now verifies maker exposes callable `make(...)` before invocation and returns through validated callable reference.
- `thermo/openmm/__init__.py`:

  - lazy export path now catches only `ImportError` as optional-dependency fallback.
  - runtime import failures (e.g., module init `RuntimeError`) are now propagated instead of being silently converted to `None`.
- Tests:

  - new `test_openmm_atomate2_wrapper.py` with coverage for:

    - init argument validation,
    - ensemble/steps contract checks,
    - module import caching,
    - import-error wrapping,
    - maker interface contract,
    - successful maker invocation payload.
  - updated `test_openmm_stack.py` with regression ensuring runtime import errors are not swallowed in lazy export path.

## Verification

- `python -m ruff check atlas/thermo/openmm/atomate2_wrapper.py atlas/thermo/openmm/__init__.py tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`
- `python -m py_compile atlas/thermo/openmm/atomate2_wrapper.py atlas/thermo/openmm/__init__.py tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`
- `python -m pytest -q tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`

## Research references used in batch 62

- atomate2 OpenMM install + architecture notes: https://materialsproject.github.io/atomate2/user/codes/openmm.html
- atomate2 OpenMM tutorial (NVTMaker/NPTMaker usage): https://materialsproject.github.io/atomate2/tutorials/openmm_tutorial.html
- atomate2 docs home (2026 docs index): https://materialsproject.github.io/atomate2/
- OpenMM Simulation API (latest): https://docs.openmm.org/latest/api-python/generated/openmm.app.simulation.Simulation.html
- OpenMM NonbondedForce API (latest): https://docs.openmm.org/latest/api-python/generated/openmm.openmm.NonbondedForce.html
- OpenMM app layer reference: https://docs.openmm.org/latest/api-python/app.html
- Python `importlib` docs (`import_module`): https://docs.python.org/3/library/importlib.html
- PEP 562 (module `__getattr__` lazy exports): https://peps.python.org/pep-0562/
- Python logging docs: https://docs.python.org/3/library/logging.html

## Progress snapshot (after Batch 62)

- Completed: Batch 1 through Batch 62.
- Pending: Batch 63 onward.

## Batch 63 (max 5 files)

- [X] `atlas/thermo/openmm/atomate2_wrapper.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/__init__.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_openmm_atomate2_wrapper.py` - added + optimized
- [X] `tests/unit/thermo/test_openmm_stack.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 63 optimization goals

- Enforce strict runtime contracts for native Atomate2 OpenMM wrapper (step typing, maker interface, and mode-specific constraints).
- Improve lazy import reliability so optional dependency fallback handles only true optional-missing paths.
- Add deterministic CI-safe tests for import-cache behavior, import failure modes, and maker-call contracts.

## Batch 63 outcomes

- `thermo/openmm/atomate2_wrapper.py`:

  - added module-level cache (`_ATOMATE2_JOBS_MODULE`) for atomate2 jobs import reuse.
  - added `_coerce_non_negative_int(...)` to reject invalid step payloads (including bool) and normalize integer steps.
  - tightened ensemble semantics:

    - `nvt` and `npt` require strictly positive `steps`,
    - `minimize` remains valid with `steps=0`.
  - `run_simulation(...)` now validates maker exposes callable `make(...)` before invocation.
  - optional dependency loading now wraps only `ImportError` into a clear runtime message.
- `thermo/openmm/__init__.py`:

  - lazy export fallback now catches only `ImportError` for optional dependency absence.
  - runtime faults during module import are now propagated, improving diagnosability.
- Tests:

  - new `test_openmm_atomate2_wrapper.py` covers input validation, module-cache behavior, import-error wrapping, maker-interface checks, and positive execution path.
  - `test_openmm_stack.py` adds regression to ensure non-ImportError lazy import failures are not suppressed.

## Verification

- `python -m ruff check atlas/thermo/openmm/atomate2_wrapper.py atlas/thermo/openmm/__init__.py tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`
- `python -m py_compile atlas/thermo/openmm/atomate2_wrapper.py atlas/thermo/openmm/__init__.py tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`
- `python -m pytest -q tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`

## Research references used in batch 63

- OpenMM paper (method + software architecture): https://doi.org/10.1371/journal.pcbi.1005659
- OpenMM User Guide: https://docs.openmm.org/latest/userguide/
- OpenMM Python API (`Simulation`): https://docs.openmm.org/latest/api-python/generated/openmm.app.simulation.Simulation.html
- atomate2 OpenMM user docs: https://materialsproject.github.io/atomate2/user/codes/openmm.html
- atomate2 OpenMM tutorial (`NVTMaker` / `NPTMaker`): https://materialsproject.github.io/atomate2/tutorials/openmm_tutorial.html
- Python `importlib` docs (`import_module`): https://docs.python.org/3/library/importlib.html
- PEP 562 (`module.__getattr__` lazy export semantics): https://peps.python.org/pep-0562/

## Progress snapshot (after Batch 63)

- Completed: Batch 1 through Batch 63.
- Pending: Batch 64 onward.

## Batch 64 (max 5 files)

- [X] `atlas/thermo/openmm/engine.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/reporters.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_openmm_stack.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_openmm_reporters.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 64 optimization goals

- Eliminate implicit numeric truncation risks in OpenMM runtime controls (`steps`, `trajectory_interval`).
- Fix trajectory metadata unit correctness (`time_step` in pymatgen requires femtoseconds).
- Add deterministic regression tests for integer contract enforcement and reporter time-axis behavior.
- Keep optional dependency fallback and testability stable for CI environments without full OpenMM stack.

## Batch 64 outcomes

- `thermo/openmm/engine.py`:

  - added `_coerce_positive_int(...)` to enforce strict positive-integer inputs and reject bool/non-integral floats.
  - `run(...)` now uses strict validation for both `steps` and `trajectory_interval` (no silent truncation).
  - introduced high-precision `_KJ_MOL_PER_EV` conversion constant to remove magic-number drift.
  - normalized `forcefield_path` handling (`strip/lower`) and made unsupported custom path fallback explicit in logs.
- `thermo/openmm/reporters.py`:

  - added monotonic-time guard during reporting (`time_ps` must be non-decreasing).
  - strengthened frame validation with finite and non-decreasing time-axis checks.
  - corrected `Trajectory.time_step` unit from ps to fs (`* 1000.0`) to match pymatgen contract.
  - switched `zip(..., strict=True)` for frame property assembly after buffer-length validation.
- Tests:

  - `test_openmm_stack.py`:

    - added regression that `engine.run(...)` rejects non-integral `steps`/`trajectory_interval`.
  - new `test_openmm_reporters.py`:

    - verifies `time_step` conversion from ps to fs,
    - verifies non-monotonic simulation time is rejected.

## Verification

- `python -m ruff check atlas/thermo/openmm/engine.py atlas/thermo/openmm/reporters.py tests/unit/thermo/test_openmm_stack.py tests/unit/thermo/test_openmm_reporters.py`
- `python -m py_compile atlas/thermo/openmm/engine.py atlas/thermo/openmm/reporters.py tests/unit/thermo/test_openmm_stack.py tests/unit/thermo/test_openmm_reporters.py`
- `python -m pytest -q tests/unit/thermo/test_openmm_stack.py tests/unit/thermo/test_openmm_reporters.py`

## Research references used in batch 64

- OpenMM `Simulation.step(steps: int)` API: https://docs.openmm.org/latest/api-python/generated/openmm.app.simulation.Simulation.html
- OpenMM reporter contract (`describeNextReport`): https://docs.openmm.org/development/api-python/generated/openmm.app.pdbreporter.PDBReporter.html
- OpenMM architecture paper: https://doi.org/10.1371/journal.pcbi.1005659
- pymatgen trajectory API (`time_step` in femtoseconds): https://pymatgen.org/pymatgen.core.html#pymatgen.core.trajectory.Trajectory
- Python stdtypes (`bool` is a subclass of `int`): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool

## Progress snapshot (after Batch 64)

- Completed: Batch 1 through Batch 64.
- Pending: Batch 65 onward.

## Batch 65 (max 5 files)

- [X] `atlas/thermo/calphad.py` - reviewed + optimized
- [X] `atlas/thermo/stability.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_calphad.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_stability.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 65 optimization goals

- Remove silent numeric coercion risk in CALPHAD path generation (`n_steps`) by enforcing strict integer contracts.
- Improve phase-fraction normalization robustness so output is consistently probability-like for downstream comparison/ranking.
- Harden phase stability decomposition serialization against invalid/non-finite decomposition coefficients.
- Expand unit tests to lock new numerical/contract behavior and keep backward compatibility for existing outputs.

## Batch 65 outcomes

- `thermo/calphad.py`:

  - added `_coerce_int_with_min(...)` to reject bool/non-integral float step values.
  - `solidification_path(...)` now enforces strict integer `n_steps >= 2` (no silent truncation).
  - added `_canonical_phase_name(...)` and normalized phase labels to canonical uppercase form.
  - `_normalize_phase_fractions(...)` now always renormalizes surviving positive fractions to sum to 1.0.
- `thermo/stability.py`:

  - introduced `_STABLE_EHULL_EPS` constant for explicit stability threshold governance.
  - added decomposition-map sanitization: non-finite/non-positive coefficients are ignored.
  - when decomposition coefficients are all invalid, fallback decomposition text is target reduced formula.
- Tests:

  - `test_calphad.py`:

    - verifies case-folded phase merge + renormalization,
    - verifies subunit phase totals are renormalized to 1.0,
    - verifies `solidification_path(...)` rejects non-integral step values.
  - `test_stability.py`:

    - verifies non-finite decomposition coefficients do not leak to output formatting,
    - verifies non-finite target energy is rejected early.

## Verification

- `python -m ruff check atlas/thermo/calphad.py atlas/thermo/stability.py tests/unit/thermo/test_calphad.py tests/unit/thermo/test_stability.py`
- `python -m py_compile atlas/thermo/calphad.py atlas/thermo/stability.py tests/unit/thermo/test_calphad.py tests/unit/thermo/test_stability.py`
- `python -m pytest -q tests/unit/thermo/test_calphad.py tests/unit/thermo/test_stability.py`

## Research references used in batch 65

- pycalphad docs (API and equilibrium modeling): https://pycalphad.org/docs/latest/
- pycalphad example gallery (equilibrium workflows): https://pycalphad.org/docs/latest/examples/index.html
- pycalphad-scheil package docs: https://scheil.readthedocs.io/en/latest/
- pycalphad-scheil implementation repository: https://github.com/pycalphad/scheil
- Bocklund et al., *pycalphad: CALPHAD-based computational thermodynamics in Python* (JORS, 2019): https://doi.org/10.5334/jors.140
- pymatgen phase diagram API docs (`PhaseDiagram`, `get_e_above_hull`): https://pymatgen.org/pymatgen.analysis#module-pymatgen.analysis.phase_diagram
- pymatgen project docs root (reference implementation context): https://pymatgen.org/

## Progress snapshot (after Batch 65)

- Completed: Batch 1 through Batch 65.
- Pending: Batch 66 onward.

## Batch 66 (max 5 files)

- [X] `scripts/phase5_active_learning/test_enumeration.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/structure_enumerator.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_test_enumeration_script.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_structure_enumerator_script.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 66 optimization goals

- Make the enumeration demo CLI machine-friendly (`--json` output must not be mixed with verbose text).
- Remove silent numeric coercion in fallback enumerator controls (`max_index`) to avoid hidden behavior drift.
- Harden substitution normalization and ordinal decoding contracts for deterministic fallback enumeration.
- Expand tests to lock CLI behavior, substitution-key normalization, and enumeration input contracts.

## Batch 66 outcomes

- `scripts/phase5_active_learning/test_enumeration.py`:

  - split parser creation into `_build_parser()` and upgraded `main(argv=None)` for testable CLI execution.
  - fixed JSON mode behavior: `--json` now forces non-verbose execution for parseable single-object output.
  - error path now writes to `stderr` and returns exit code `2` consistently.
- `scripts/phase5_active_learning/structure_enumerator.py`:

  - added strict `_coerce_positive_int(...)` and applied to `max_index` (rejects bool/non-integral float).
  - strengthened `_decode_variant_ordinal(...)` with explicit ordinal-range validation.
  - `_normalize_substitutions(...)` now merges semantically identical keys after whitespace normalization.
  - switched site substitution loop to `zip(..., strict=True)` for iterator contract safety.
- Tests:

  - `test_test_enumeration_script.py`:

    - added regression for JSON mode suppressing verbose output,
    - added regression that CLI errors go to `stderr`.
  - `test_structure_enumerator_script.py`:

    - added non-integral `max_index` rejection regression,
    - added normalized duplicate-key merge regression,
    - added out-of-range ordinal decode regression.

## Verification

- `python -m ruff check scripts/phase5_active_learning/test_enumeration.py scripts/phase5_active_learning/structure_enumerator.py tests/unit/active_learning/test_test_enumeration_script.py tests/unit/active_learning/test_structure_enumerator_script.py`
- `python -m py_compile scripts/phase5_active_learning/test_enumeration.py scripts/phase5_active_learning/structure_enumerator.py tests/unit/active_learning/test_test_enumeration_script.py tests/unit/active_learning/test_structure_enumerator_script.py`
- `python -m pytest -q tests/unit/active_learning/test_test_enumeration_script.py tests/unit/active_learning/test_structure_enumerator_script.py`

## Research references used in batch 66

- Python `argparse` reference (`parse_args`, CLI design): https://docs.python.org/3/library/argparse.html
- Python boolean type semantics (`bool` is `int` subclass): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool
- Python iterator `zip(..., strict=True)` semantics: https://docs.python.org/3/library/functions.html#zip
- pymatgen `StructureMatcher` API docs: https://pymatgen.org/pymatgen.analysis.html#pymatgen.analysis.structure_matcher.StructureMatcher
- pymatgen `DummySpecies` API docs: https://pymatgen.org/pymatgen.core.html#pymatgen.core.periodic_table.DummySpecies
- Ong et al., *Python Materials Genomics (pymatgen)*: https://www.sciencedirect.com/science/article/pii/S0927025612006295

## Progress snapshot (after Batch 66)

- Completed: Batch 1 through Batch 66.
- Pending: Batch 67 onward.

## Batch 67 (max 5 files)

- [X] `scripts/phase5_active_learning/search_materials.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/active_learning.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_search_materials_cli.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_phase6_active_learning.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 67 optimization goals

- Make search CLI validation stricter and more testable (especially integer contracts and conflicting bounds).
- Ensure CLI error channels are automation-friendly (`stderr` for invalid input paths).
- Prevent active-learning helper APIs from silently coercing invalid integer-like controls.
- Avoid hidden state carry-over across repeated active-learning runs.

## Batch 67 outcomes

- `scripts/phase5_active_learning/search_materials.py`:

  - added `_coerce_positive_int(...)` and strict `--max` validation (rejects bool/non-integral inputs).
  - introduced `_validate_criteria_bounds(...)` and applied it to built criteria; contradictory merged bounds now fail fast.
  - refactored CLI entry into `_build_parser()` + `main(argv=None)` for deterministic unit testing.
  - CLI validation and query-column errors now print to `stderr` with exit code `2`.
  - added `--desc` alias while preserving legacy `-desc`.
  - replaced global `pd.set_option(...)` mutations with scoped `pd.option_context(...)`.
- `scripts/phase5_active_learning/active_learning.py`:

  - tightened integer contracts for batch sizes, budgets, iteration counts, `n_samples`, and `top_k`.
  - `ActiveLearningLoop.run(...)` now resets history at run start to avoid stale-run contamination.
- Tests:

  - `test_search_materials_cli.py`:

    - added strict `--max` validation regression,
    - added inconsistent-bound merge regression,
    - added `main(...)` error-to-`stderr` regression.
  - `test_phase6_active_learning.py`:

    - added non-integral batch-size regression,
    - added non-integral AL budget regression,
    - added empty-dataset run-history reset regression,
    - added non-integral `top_k` regression.

## Verification

- `python -m ruff check scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/active_learning.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_phase6_active_learning.py`
- `python -m py_compile scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/active_learning.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_phase6_active_learning.py`
- `python -m pytest -q tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_phase6_active_learning.py`

## Research references used in batch 67

- Python `argparse` docs (`parse_args`, parser patterns): https://docs.python.org/3/library/argparse.html
- Python numeric type semantics (`bool` as `int` subclass): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool
- pandas `option_context` (scoped display settings): https://pandas.pydata.org/docs/reference/api/pandas.option_context.html
- NumPy random Generator API (`choice`, reproducibility): https://numpy.org/doc/stable/reference/random/generator.html
- Gal & Ghahramani, *Dropout as a Bayesian Approximation* (MC-dropout UQ basis): https://proceedings.mlr.press/v48/gal16.html
- Jensen, *Introduction to Pareto optimality* (multi-objective optimization background): https://doi.org/10.1007/978-0-387-74759-0_493

## Progress snapshot (after Batch 67)

- Completed: Batch 1 through Batch 67.
- Pending: Batch 68 onward.

## Batch 68 (max 5 files)

- [X] `scripts/phase5_active_learning/run_phase5.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_discovery.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_phase5_cli.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_controller_runtime_stability.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 68 optimization goals

- Strengthen CLI numeric contracts in Phase5 launchers to avoid silent coercion bugs (`bool` / non-integral floats).
- Improve script entrypoint testability via parser extraction and `main(argv=None)` patterns.
- Route validation/runtime errors to `stderr` for better CI/automation behavior.
- Add regression tests for strict integer validation and runtime backoff jitter behavior.

## Batch 68 outcomes

- `scripts/phase5_active_learning/run_phase5.py`:

  - added `_coerce_int_with_bounds(...)` and applied strict integer checks to launcher integer controls.
  - refactored parser creation into `_build_parser()` and converted entrypoint to `main(argv=None)`.
  - validation/preflight error messages now emit to `stderr`.
- `scripts/phase5_active_learning/run_discovery.py`:

  - added strict integer coercion helper for discovery controls (`iterations`, `candidates`, `top`, `seeds`, `calibration_window`).
  - refactored parser creation into `_build_parser()` and converted entrypoint to `main(argv=None)`.
  - validation and run-directory resolution errors now emit to `stderr`.
- `tests/unit/active_learning/test_phase5_cli.py`:

  - added regressions for non-integral integer controls in both phase5 and discovery validators.
  - added regressions asserting `main(...)` validation failures are written to `stderr`.
- `tests/unit/active_learning/test_controller_runtime_stability.py`:

  - added deterministic jitter/backoff regression for `_retry_sleep_seconds(...)` range behavior with injected RNG.

## Verification

- `python -m ruff check scripts/phase5_active_learning/run_phase5.py scripts/phase5_active_learning/run_discovery.py tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_controller_runtime_stability.py`
- `python -m py_compile scripts/phase5_active_learning/run_phase5.py scripts/phase5_active_learning/run_discovery.py tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_controller_runtime_stability.py`
- `python -m pytest -q tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_controller_runtime_stability.py`

## Research references used in batch 68

- Python `argparse` reference: https://docs.python.org/3/library/argparse.html
- Python `subprocess` reference: https://docs.python.org/3/library/subprocess.html
- PEP 389 (`argparse` design rationale): https://peps.python.org/pep-0389/
- OWASP Input Validation Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html
- CWE-20 Improper Input Validation: https://cwe.mitre.org/data/definitions/20.html
- Dropout as a Bayesian Approximation (UQ context for AL pipelines): https://proceedings.mlr.press/v48/gal16.html
- Settles, Active Learning Literature Survey: https://burrsettles.com/pub/settles.activelearning.pdf

## Progress snapshot (after Batch 68)

- Completed: Batch 1 through Batch 68.
- Pending: Batch 69 onward.

## Batch 69 (max 5 files)

- [X] `atlas/benchmark/cli.py` - reviewed + optimized
- [X] `atlas/benchmark/runner.py` - reviewed + optimized
- [X] `tests/unit/benchmark/test_benchmark_cli.py` - reviewed + optimized
- [X] `tests/unit/benchmark/test_benchmark_runner.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 69 optimization goals

- Tighten benchmark CLI integer contracts to prevent bool/non-integral values from slipping through argument validation.
- Improve benchmark preflight failure signaling for automation (stderr instead of stdout).
- Remove silent float truncation in benchmark runner integer coercion helpers.
- Add regression tests to lock strict validation and sanitization behavior.

## Batch 69 outcomes

- `atlas/benchmark/cli.py`:

  - added `_coerce_int_with_min(...)` for strict integer validation.
  - `_validate_cli_args(...)` now normalizes/validates integer controls (`batch_size`, `jobs`, bootstrap and preflight controls) with explicit error semantics.
  - fold validation now enforces integer entries and non-negative bounds through the same helper.
  - preflight failure path now writes errors to `stderr`.
- `atlas/benchmark/runner.py`:

  - hardened `_coerce_positive_int(...)` and `_coerce_int(...)`:

    - bool is no longer treated as valid integer input,
    - non-integral real values no longer silently truncate via `int(...)`.
  - preserves existing fallback-to-default behavior while removing hidden truncation.
- Tests:

  - `test_benchmark_cli.py`:

    - added non-integral integer-control validation regressions,
    - added preflight failure stderr regression.
  - `test_benchmark_runner.py`:

    - added regression proving non-integral/bool runtime controls are sanitized to conservative defaults without truncation.

## Verification

- `python -m ruff check atlas/benchmark/cli.py atlas/benchmark/runner.py tests/unit/benchmark/test_benchmark_cli.py tests/unit/benchmark/test_benchmark_runner.py`
- `python -m py_compile atlas/benchmark/cli.py atlas/benchmark/runner.py tests/unit/benchmark/test_benchmark_cli.py tests/unit/benchmark/test_benchmark_runner.py`
- `python -m pytest -q tests/unit/benchmark/test_benchmark_cli.py tests/unit/benchmark/test_benchmark_runner.py`

## Research references used in batch 69

- Python `argparse` documentation: https://docs.python.org/3/library/argparse.html
- Python type semantics (`bool` subclassing `int`): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool
- Matbench benchmark paper (Dunn et al., 2020): https://doi.org/10.1038/s41524-020-00406-3
- Matbench official documentation: https://hackingmaterials.lbl.gov/matbench/
- Kuleshov et al., calibrated regression uncertainty: https://proceedings.mlr.press/v80/kuleshov18a.html
- Angelopoulos & Bates, conformal prediction tutorial: https://arxiv.org/abs/2107.07511
- Efron & Tibshirani, bootstrap methodology: https://doi.org/10.1007/978-1-4899-4541-9

## Progress snapshot (after Batch 69)

- Completed: Batch 1 through Batch 69.
- Pending: Batch 70 onward.

## Batch 70 (max 5 files)

- [X] `atlas/benchmark/cli.py` - reviewed + optimized
- [X] `atlas/benchmark/runner.py` - reviewed + optimized
- [X] `tests/unit/benchmark/test_benchmark_cli.py` - reviewed + optimized
- [X] `tests/unit/benchmark/test_benchmark_runner.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 70 optimization goals

- Strengthen benchmark CLI integer validation semantics to reject non-integral controls consistently.
- Prevent bool/non-integral numeric inputs from silently truncating in benchmark runner parameter sanitization.
- Improve automation friendliness by routing benchmark preflight failures to stderr.
- Add regression tests to lock these contracts and avoid drift.

## Batch 70 outcomes

- `atlas/benchmark/cli.py`:

  - added `_coerce_int_with_min(...)` and integrated it into `_validate_cli_args(...)`.
  - integer controls (`batch-size`, `jobs`, bootstrap/preflight integer args, and fold entries) are now normalized with strict integer checks.
  - preflight-failure error line now emits to `stderr`.
- `atlas/benchmark/runner.py`:

  - hardened `_coerce_positive_int(...)` and `_coerce_int(...)`:

    - bool values no longer pass as valid integers,
    - non-integral real values no longer get silently truncated by `int(...)`.
  - fallback-to-default behavior remains unchanged for invalid inputs, but now explicit and safer.
- `tests/unit/benchmark/test_benchmark_cli.py`:

  - added non-integral integer-control rejection regressions.
  - added preflight-failure stderr regression.
- `tests/unit/benchmark/test_benchmark_runner.py`:

  - added regression verifying non-integral/bool runner params sanitize to defaults rather than truncating.

## Verification

- `python -m ruff check atlas/benchmark/cli.py atlas/benchmark/runner.py tests/unit/benchmark/test_benchmark_cli.py tests/unit/benchmark/test_benchmark_runner.py`
- `python -m py_compile atlas/benchmark/cli.py atlas/benchmark/runner.py tests/unit/benchmark/test_benchmark_cli.py tests/unit/benchmark/test_benchmark_runner.py`
- `python -m pytest -q tests/unit/benchmark/test_benchmark_cli.py tests/unit/benchmark/test_benchmark_runner.py`

## Research references used in batch 70

- Python argparse docs (official): https://docs.python.org/3.12/library/argparse.html
- Python bool/int semantics (official): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool
- Matbench documentation: https://hackingmaterials.lbl.gov/automatminer/datasets.html
- Matbench paper (Nature Computational Materials): https://doi.org/10.1038/s41524-020-00406-3
- Kuleshov et al. calibrated regression (PMLR 2018): https://proceedings.mlr.press/v80/kuleshov18a.html
- Angelopoulos & Bates conformal tutorial (arXiv): https://arxiv.org/abs/2107.07511
- Efron & Tibshirani bootstrap reference: https://doi.org/10.1007/978-1-4899-4541-9

## Progress snapshot (after Batch 70)

- Completed: Batch 1 through Batch 70.
- Pending: Batch 71 onward.

## Batch 71 (max 5 files)

- [X] `atlas/benchmark/__init__.py` - reviewed + optimized
- [X] `atlas/benchmark/runner.py` - reviewed + optimized
- [X] `tests/unit/benchmark/test_benchmark_init.py` - reviewed + optimized (new)
- [X] `tests/unit/benchmark/test_benchmark_runner.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 71 optimization goals

- Strengthen benchmark package lazy-export contracts so symbol resolution failures are explicit and testable.
- Eliminate silent fold-index coercion in benchmark runner (non-integral/negative fold IDs must fail fast).
- Harden probability sanitization against bool semantics and non-finite values.
- Lock these contracts with focused unit regressions.

## Batch 71 outcomes

- `atlas/benchmark/__init__.py`:

  - switched to explicit lazy export table (`_EXPORTS`) with `__getattr__` + `__dir__`.
  - added cache short-circuit (`if name in globals()`) for stable lazy resolution semantics.
  - added explicit missing-attribute check after module import with informative `AttributeError`.
- `atlas/benchmark/runner.py`:

  - `_coerce_probability(...)` now rejects bool semantics and falls back to defaults.
  - added `_coerce_non_negative_fold_id(...)` and used it in `run_task(...)` fold normalization.
  - non-integral fold IDs (e.g., `0.5`) and negative IDs now fail fast with clear `ValueError` instead of implicit truncation.
- `tests/unit/benchmark/test_benchmark_init.py` (new):

  - verifies expected exports in `dir(...)`, unknown attribute behavior, lazy caching behavior, and explicit error path when expected export is missing.
- `tests/unit/benchmark/test_benchmark_runner.py`:

  - added regressions for bool probability controls fallback.
  - added regressions for non-integral and negative fold ID rejection.

## Verification

- `python -m ruff check atlas/benchmark/__init__.py atlas/benchmark/runner.py tests/unit/benchmark/test_benchmark_init.py tests/unit/benchmark/test_benchmark_runner.py`
- `python -m py_compile atlas/benchmark/__init__.py atlas/benchmark/runner.py tests/unit/benchmark/test_benchmark_init.py tests/unit/benchmark/test_benchmark_runner.py`
- `python -m pytest -q tests/unit/benchmark/test_benchmark_init.py tests/unit/benchmark/test_benchmark_runner.py`

## Research references used in batch 71

- PEP 562 (`module __getattr__`, `__dir__`): https://peps.python.org/pep-0562/
- Python bool/int semantics (official): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool
- Matbench benchmark paper: https://doi.org/10.1038/s41524-020-00406-3
- Matbench official documentation: https://hackingmaterials.lbl.gov/matbench/
- Kuleshov et al. calibrated regression uncertainty: https://proceedings.mlr.press/v80/kuleshov18a.html
- Angelopoulos & Bates conformal prediction tutorial: https://arxiv.org/abs/2107.07511

## Progress snapshot (after Batch 71)

- Completed: Batch 1 through Batch 71.
- Pending: Batch 72 onward.

## Batch 72 (max 5 files)

- [X] `scripts/phase5_active_learning/test_enumeration.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/structure_enumerator.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/search_materials.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_structure_enumerator_script.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 72 optimization goals

- Remove script-level import side effects in enumeration demo (`sys.path` mutation) while preserving local fallback execution.
- Improve fallback enumerator truncation sampling coverage so reduced variant sets better span the combinatorial space.
- Reuse shared variant-space validation logic to reduce duplicated radix checks and avoid drift.
- Add runtime safety guard for large result requests in multi-property search CLI.

## Batch 72 outcomes

- `scripts/phase5_active_learning/test_enumeration.py`:

  - removed global `sys.path.insert(...)` side effect.
  - added `importlib.util`-based local fallback loader (`_load_local_enumerator_class`) for direct script execution without package path pollution.
  - tightened fallback error semantics (explicit `ImportError`/`TypeError` path).
  - replaced `hasattr(cls, "__call__")` with `callable(cls)` for clearer contract checking.
- `scripts/phase5_active_learning/structure_enumerator.py`:

  - upgraded `_select_variant_ordinals(...)` to deterministic near-uniform endpoint-inclusive sampling over `[0, total-1]`.
  - added `_variant_space(...)` helper and reused it in `generate(...)`, `_decode_variant_ordinal(...)`, and `_build_constraints(...)`.
  - reduced duplicated radix-validation logic and made combinatorial-space computation consistent across code paths.
- `scripts/phase5_active_learning/search_materials.py`:

  - added `_MAX_RESULTS_LIMIT = 5000` and validation guard (`--max cannot exceed 5000`) to prevent accidental overlarge query/output requests.
  - updated CLI help text to expose the hard cap explicitly.
- `tests/unit/active_learning/test_structure_enumerator_script.py`:

  - updated ordinal-stratification regression to match endpoint-inclusive deterministic sampling contract.

## Verification

- `python -m ruff check scripts/phase5_active_learning/test_enumeration.py scripts/phase5_active_learning/structure_enumerator.py scripts/phase5_active_learning/search_materials.py tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_test_enumeration_script.py tests/unit/active_learning/test_search_materials_cli.py`
- `python -m py_compile scripts/phase5_active_learning/test_enumeration.py scripts/phase5_active_learning/structure_enumerator.py scripts/phase5_active_learning/search_materials.py tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_test_enumeration_script.py tests/unit/active_learning/test_search_materials_cli.py`
- `python -m pytest -q tests/unit/active_learning/test_test_enumeration_script.py tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_search_materials_cli.py`

## Research references used in batch 72

- Python `importlib` official docs: https://docs.python.org/3/library/importlib.html
- Python `argparse` official docs: https://docs.python.org/3/library/argparse.html
- pymatgen installation docs (`enumlib` optional dependency): https://pymatgen.org/installation.html
- pymatgen API docs (`StructureMatcher`): https://pymatgen.org/pymatgen.analysis.html
- Hart & Forcade (derivative superstructures): https://doi.org/10.1107/S0108767308028503
- Hart, Nelson & Forcade (multicomponent derivative structures): https://doi.org/10.1016/j.commatsci.2012.02.015
- dsenum package docs: https://lan496.github.io/dsenum/
- JARVIS data paper (Nature Scientific Data): https://doi.org/10.1038/s41597-020-00723-1

## Progress snapshot (after Batch 72)

- Completed: Batch 1 through Batch 72.
- Pending: Batch 73 onward.

## Batch 73 (max 5 files)

- [X] `atlas/active_learning/policy_engine.py` - reviewed + optimized
- [X] `atlas/active_learning/policy_state.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_policy_engine.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_policy_state.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 73 optimization goals

- Tighten policy/state integer coercion contracts to eliminate silent truncation and bool-as-int ambiguity.
- Make conformal calibration scale estimation use finite-sample-aware split-conformal quantile level.
- Align CMOEIC utility decomposition with objective/feasibility separation (avoid repeated topology multiplication).
- Add regression tests for n_top sanitization and stricter integer-field handling.

## Batch 73 outcomes

- `atlas/active_learning/policy_state.py`:

  - hardened `_coerce_int(...)` to reject bool and non-integral real values (fallback to defaults).
  - added `_conformal_quantile_level(...)` with finite-sample split-conformal correction `ceil((n+1)*(1-alpha))/n`.
  - `update_calibration(...)` now uses the finite-sample-aware quantile level when estimating `conformal_scale`.
- `atlas/active_learning/policy_engine.py`:

  - added strict positive-int sanitization for `n_top` (`score_and_select(...)`) to prevent invalid selection requests.
  - revised `_base_utility(...)` so objective and feasibility are separated (objective from stability/diversity/cost; topology/synthesis applied in feasibility stage), reducing topology over-counting.
- `tests/unit/active_learning/test_policy_state.py`:

  - added regressions for non-integral integer control rejection in profile/state payloads.
  - added regression ensuring finite-sample conformal calibration path remains numerically stable.
- `tests/unit/active_learning/test_policy_engine.py`:

  - added regression proving non-positive and non-integral `n_top` are sanitized to safe positive integer behavior.

## Verification

- `python -m ruff check atlas/active_learning/policy_engine.py atlas/active_learning/policy_state.py tests/unit/active_learning/test_policy_engine.py tests/unit/active_learning/test_policy_state.py`
- `python -m py_compile atlas/active_learning/policy_engine.py atlas/active_learning/policy_state.py tests/unit/active_learning/test_policy_engine.py tests/unit/active_learning/test_policy_state.py`
- `python -m pytest -q tests/unit/active_learning/test_policy_engine.py tests/unit/active_learning/test_policy_state.py`
- `python -m pytest -q tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_controller_runtime_stability.py`

## Research references used in batch 73

- Angelopoulos & Bates, conformal prediction tutorial (split-conformal quantile guidance): https://arxiv.org/abs/2107.07511
- Kuleshov et al., calibrated regression uncertainty: https://proceedings.mlr.press/v80/kuleshov18a.html
- NumPy `quantile` official documentation: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
- Settles, Active Learning Literature Survey: https://burrsettles.com/pub/settles.activelearning.pdf
- Beygelzimer et al., importance-weighted active learning: https://www.jmlr.org/papers/v10/beygelzimer09a.html
- Python bool/int semantics (official): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool

## Progress snapshot (after Batch 73)

- Completed: Batch 1 through Batch 73.
- Pending: Batch 74 onward.

## Batch 74 (max 5 files)

- [X] `atlas/active_learning/acquisition.py` - reviewed + optimized
- [X] `atlas/active_learning/objective_space.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_acquisition.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_objective_space.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 74 optimization goals

- Harden acquisition parameter coercion to avoid bool/non-integral truncation drift in iterative AL loops.
- Reduce NEI Monte Carlo variance and improve deterministic behavior in zero-observation-noise regimes.
- Make objective-space threshold and dimensionality sanitization explicit and bounded for robust Pareto/feasibility preprocessing.
- Add regression tests for these numerical/contract guarantees.

## Batch 74 outcomes

- `atlas/active_learning/acquisition.py`:

  - `_coerce_int(...)` is now strict for bool/non-integral real inputs (fallback to defaults).
  - added `_sanitize_mc_samples(...)` with bounded range to avoid pathological sample-count blowups.
  - `_noisy_expected_improvement_prepared(...)` now:

    - falls back to deterministic EI when observed noise is effectively zero,
    - uses antithetic-normal pairing for MC samples to reduce estimator variance at fixed budget.
  - `score_acquisition(...)` and `schedule_ucb_kappa(...)` now sanitize non-finite/non-integral controls consistently.
- `atlas/active_learning/objective_space.py`:

  - hardened `clip01(...)` via `safe_float(...)` to handle scalar-like containers safely.
  - `_coerce_obj_dim(...)` no longer silently truncates non-integral objective dimensions.
  - `_coerce_threshold(...)` introduced and applied in history collection + feasibility masking to enforce unit-interval threshold contracts.
- `tests/unit/active_learning/test_acquisition.py`:

  - added regression proving NEI collapses to EI under zero observation noise.
  - added regressions for sanitization of non-integral `nei_mc_samples` and non-integral `iteration` controls.
- `tests/unit/active_learning/test_objective_space.py`:

  - added regression for non-integral `obj_dim` fallback behavior and threshold clipping behavior.

## Verification

- `python -m ruff check atlas/active_learning/acquisition.py atlas/active_learning/objective_space.py tests/unit/active_learning/test_acquisition.py tests/unit/active_learning/test_objective_space.py`
- `python -m py_compile atlas/active_learning/acquisition.py atlas/active_learning/objective_space.py tests/unit/active_learning/test_acquisition.py tests/unit/active_learning/test_objective_space.py`
- `python -m pytest -q tests/unit/active_learning/test_acquisition.py tests/unit/active_learning/test_objective_space.py`
- `python -m pytest -q tests/unit/active_learning/test_controller_acquisition.py`

## Research references used in batch 74

- Jones, Schonlau, Welch (EGO, EI origin; DOI in metadata): https://r7-www1.stat.ubc.ca/efficient-global-optimization-expensive-black-box-functions
- GP-UCB regret analysis: https://arxiv.org/abs/0912.3995
- LogEI stabilization paper: https://arxiv.org/abs/2310.20708
- Constrained/Noisy BO (NEI context): https://arxiv.org/abs/1706.07094
- BoTorch acquisition docs (analytic/MC and LogNEI recommendation): https://botorch.readthedocs.io/en/latest/acquisition.html
- BoTorch acquisition overview (QMC variance reduction discussion): https://botorch.org/docs/v0.14.0/acquisition
- BoTorch framework paper (MC BO tooling): https://proceedings.neurips.cc/paper/2020/file/f5b1b89d98b7286673128a5fb112cb9a-Paper.pdf
- Antithetic variates classic reference: https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/abs/new-monte-carlo-technique-antithetic-variates/69A9BBEDC6A4F1B1AF7E0764CD422E15

## Progress snapshot (after Batch 74)

- Completed: Batch 1 through Batch 74.
- Pending: Batch 75 onward.

## Batch 75 (max 5 files)

- [X] `atlas/active_learning/gp_surrogate.py` - reviewed + optimized
- [X] `atlas/active_learning/generator.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_gp_surrogate.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_generator.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 75 optimization goals

- Eliminate silent truncation/bool-as-int behavior in GP surrogate integer controls.
- Harden feature preprocessing for surrogate descriptors to preserve probability semantics.
- Improve generator hyperparameter sanitization under non-finite inputs (NaN/inf) to avoid unstable weight pipelines.
- Add regressions for strict coercion and non-finite-weight normalization behavior.

## Batch 75 outcomes

- `atlas/active_learning/gp_surrogate.py`:

  - hardened `_coerce_int(...)` for bool/non-integral real inputs (fallback to defaults).
  - added `_clip01(...)` and applied it to probability-like candidate descriptors in `candidate_to_features(...)`.
  - `_schedule_ucb_kappa(...)` now uses strict integer coercion for iteration.
- `atlas/active_learning/generator.py`:

  - introduced robust `_coerce_int(...)` / `_coerce_float(...)` utilities and applied them in constructor hyperparameter normalization.
  - constructor now handles non-finite inputs safely for RNG seed, archive limits, weights, and substitution decay.
  - `_normalize_weights(...)` now sanitizes non-finite entries before normalization.
  - substitution online stats update/sampling paths now sanitize corrupted/non-finite count/reward values before use.
- `tests/unit/active_learning/test_gp_surrogate.py`:

  - added regression for non-integral integer config fields falling back to validated defaults.
  - added regression ensuring `candidate_to_features(...)` clips probability-like descriptors into `[0, 1]`.
- `tests/unit/active_learning/test_generator.py`:

  - added regression covering constructor sanitization with NaN/inf hyperparameters.
  - added regression for `_normalize_weights(...)` with non-finite entries.

## Verification

- `python -m ruff check atlas/active_learning/gp_surrogate.py atlas/active_learning/generator.py tests/unit/active_learning/test_gp_surrogate.py tests/unit/active_learning/test_generator.py`
- `python -m py_compile atlas/active_learning/gp_surrogate.py atlas/active_learning/generator.py tests/unit/active_learning/test_gp_surrogate.py tests/unit/active_learning/test_generator.py`
- `python -m pytest -q tests/unit/active_learning/test_gp_surrogate.py tests/unit/active_learning/test_generator.py`

## Research references used in batch 75

- scikit-learn Gaussian Process documentation: https://scikit-learn.org/stable/modules/gaussian_process.html
- `GaussianProcessRegressor` API: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
- `GaussianProcessClassifier` API: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html
- `Matern` kernel API: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html
- `WhiteKernel` API: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.WhiteKernel.html
- GP-UCB regret analysis (Srinivas et al.): https://arxiv.org/abs/0912.3995
- Constrained Bayesian Optimization (Gardner et al.): https://proceedings.mlr.press/v32/gardner14.html
- Ionic substitution statistics (Hautier et al.): https://doi.org/10.1021/ic102031h
- SMACT / composition screening (Davies et al.): https://doi.org/10.1039/D2DD00028H

## Progress snapshot (after Batch 75)

- Completed: Batch 1 through Batch 75.
- Pending: Batch 76 onward.

## Batch 76 (max 5 files)

- [X] `atlas/active_learning/crabnet_native.py` - reviewed + optimized
- [X] `atlas/active_learning/synthesizability.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_crabnet_native.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_synthesizability.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 81 (max 5 files)

- [X] `atlas/training/checkpoint.py` - reviewed + optimized
- [X] `atlas/training/preflight.py` - reviewed + optimized
- [X] `atlas/training/run_utils.py` - reviewed + optimized
- [X] `tests/unit/training/test_checkpoint.py` - reviewed + optimized
- [X] `tests/unit/training/test_preflight.py` - reviewed + optimized

## Batch 76 optimization goals

- Eliminate silent integer truncation and bool-as-int ambiguity in CrabNet/UQ controls used inside iterative AL loops.
- Harden grouped uncertainty calibration so malformed group-temperature maps cannot poison per-group scaling.
- Align synthesizability integer-control sanitation with strict research reproducibility contracts.
- Add targeted regressions to lock down these runtime contracts.

## Batch 81 optimization goals

- Remove silent int truncation / bool-as-int behavior in checkpoint and preflight runtime controls.
- Make checkpoint metric coercion explicit so malformed MAE inputs fail fast.
- Harden run-manifest strict-lock parsing to avoid accidental truthy coercion.
- Add regression tests for strict coercion contracts.

## Batch 76 outcomes

- `atlas/active_learning/crabnet_native.py`:

  - added strict `_coerce_int(...)` that rejects bool and non-integral real values (fallback to defaults + bounds).
  - applied strict integer sanitation to `mean_dims`, `ensemble_size`, `mc_dropout_samples`, `q_steps`, grouped calibration `min_group_size`, and `predict_distribution(mc_samples=...)`.
  - added grouped-calibration table sanitation (`_coerce_group_id`, `_sanitize_group_temperature_table`) and automatic cleanup before applying per-group scaling.
- `atlas/active_learning/synthesizability.py`:

  - upgraded module `_coerce_int(...)` to strict integer semantics (no silent decimal truncation, bool rejected).
- `tests/unit/active_learning/test_crabnet_native.py`:

  - added regression for strict constructor integer controls.
  - added regression verifying grouped-calibration table cleanup removes invalid keys and keeps valid scaling behavior.
- `tests/unit/active_learning/test_synthesizability.py`:

  - added regression verifying integer controls reject bool/fractional inputs and still accept valid integer-like strings.

## Verification

- `python -m ruff check atlas/active_learning/crabnet_native.py atlas/active_learning/synthesizability.py tests/unit/active_learning/test_crabnet_native.py tests/unit/active_learning/test_synthesizability.py`
- `python -m py_compile atlas/active_learning/crabnet_native.py atlas/active_learning/synthesizability.py tests/unit/active_learning/test_crabnet_native.py tests/unit/active_learning/test_synthesizability.py`
- `python -m pytest -q tests/unit/active_learning/test_crabnet_native.py tests/unit/active_learning/test_synthesizability.py`

## Research references used in batch 76

- Python built-in types (`bool` is a subtype of `int`): https://docs.python.org/3/library/stdtypes.html
- PyTorch quantile API (calibration quantile contract): https://docs.pytorch.org/docs/stable/generated/torch.quantile.html
- CrabNet paper (npj Computational Materials): https://www.nature.com/articles/s41524-021-00545-1
- Aitchison compositional data geometry (JRSS-B, 1982): https://doi.org/10.1111/j.2517-6161.1982.tb01195.x
- Gal & Ghahramani (MC Dropout as Bayesian approximation, ICML 2016): https://proceedings.mlr.press/v48/gal16.html
- Lakshminarayanan et al. (Deep Ensembles, NeurIPS 2017): https://papers.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles
- Kendall & Gal (aleatoric/epistemic uncertainty): https://arxiv.org/abs/1703.04977
- Angelopoulos & Bates (conformal prediction tutorial): https://arxiv.org/abs/2107.07511

## Progress snapshot (after Batch 76)

- Completed: Batch 1 through Batch 76.
- Pending: Batch 77 onward.

## Batch 77 (max 5 files)

- [X] `atlas/active_learning/rxn_network_native.py` - reviewed + optimized
- [X] `atlas/active_learning/crabnet_screener.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_rxn_network_native.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_crabnet_screener.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 77 optimization goals

- Eliminate silent integer truncation and string-bool ambiguity in reaction-network and composition-screener runtime controls.
- Prevent `bool("false") -> True` style config drift in reaction-network solver switches.
- Keep uncertainty/MC-dropout controls numerically stable and reproducible under malformed integer-like inputs.
- Add contract tests for strict coercion behavior.

## Batch 77 outcomes

- `atlas/active_learning/rxn_network_native.py`:

  - added strict `_coerce_int(...)` (rejects bool + non-integral reals, uses bounded defaults).
  - added robust `_coerce_bool(...)` for string flags (`true/false`, `yes/no`, `on/off`, `1/0`) with safe default fallback.
  - applied coercion to: `max_num_pathways`, `k_shortest_paths`, `max_num_combos`, `chunk_size`, all solver booleans, and `require_native`.
  - hardened `fallback_mode` to controlled enum (`conservative` / `energy_prior`) with conservative fallback.
- `atlas/active_learning/crabnet_screener.py`:

  - upgraded `_coerce_int(...)` to strict integer semantics.
  - applied strict coercion to `out_dims`, `d_model`, MC-dropout sample controls (`_mc_dropout_epistemic_var`, `predict_distribution`).
  - removed implicit decimal truncation behavior for MC sample counts.
- `tests/unit/active_learning/test_rxn_network_native.py`:

  - added regressions for strict int/bool sanitization and fallback-mode normalization.
  - added regression proving integer-like strings remain accepted.
- `tests/unit/active_learning/test_crabnet_screener.py`:

  - added regressions proving bool/fractional integer controls are rejected to defaults.
  - added regression for non-integral `mc_samples` sanitization in `predict_distribution(...)`.

## Batch 81 outcomes

- `atlas/training/checkpoint.py`:

  - added strict integer/float coercion helpers and applied them to `top_k`, `keep_last_k`, `epoch`, and `mae`.
  - rejected bool/non-integral numeric inputs for integer fields to avoid implicit truncation (`1.9 -> 1`) in recovery metadata.
- `atlas/training/preflight.py`:

  - upgraded numeric coercion for `max_samples`, `split_seed`, and `timeout_sec` to strict integer semantics.
  - now fast-fails on bool or fractional integer-like parameters for gate controls.
- `atlas/training/run_utils.py`:

  - added `_coerce_bool_like(...)` and used it in `_resolve_environment_lock(...)` to avoid non-explicit truthy parsing for strict lock toggles.
- tests:

  - `test_checkpoint.py` now covers bool/fractional integer rejection and malformed MAE type rejection.
  - `test_preflight.py` now covers invalid fractional timeout and bool-valued `max_samples`.

## Verification

- `python -m ruff check atlas/active_learning/rxn_network_native.py atlas/active_learning/crabnet_screener.py tests/unit/active_learning/test_rxn_network_native.py tests/unit/active_learning/test_crabnet_screener.py`
- `python -m py_compile atlas/active_learning/rxn_network_native.py atlas/active_learning/crabnet_screener.py tests/unit/active_learning/test_rxn_network_native.py tests/unit/active_learning/test_crabnet_screener.py`
- `python -m pytest -q tests/unit/active_learning/test_rxn_network_native.py tests/unit/active_learning/test_crabnet_screener.py`

## Research references used in batch 77

- Python built-in types (`bool` subtype of `int`): https://docs.python.org/3/library/stdtypes.html
- PEP 285 (`bool` type design): https://peps.python.org/pep-0285/
- NumPy finite-value screening: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy finite sanitization: https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
- PyTorch `TransformerEncoderLayer` API: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
- PyTorch `GaussianNLLLoss` API: https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.GaussianNLLLoss.html
- CrabNet (npj Computational Materials, 2021): https://www.nature.com/articles/s41524-021-00545-1
- Deep Sets (NeurIPS 2017): https://arxiv.org/abs/1703.06114
- Dropout as Bayesian Approximation (ICML 2016): https://proceedings.mlr.press/v48/gal16.html
- Deep Ensembles (NeurIPS 2017): https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa85bce38a7-Abstract.html
- Bayesian aleatoric/epistemic uncertainty: https://arxiv.org/abs/1703.04977
- Reaction-network pathfinding in solid-state synthesis (Nat Commun 2021): https://www.nature.com/articles/s41467-021-23339-x
- K shortest loopless paths (Yen 1971): http://dx.doi.org/10.1287/mnsc.17.11.712
- Multi-criteria shortest path (Martins 1984): https://doi.org/10.1016/0377-2217(84)90077-8
- Conformal prediction tutorial: https://arxiv.org/abs/2107.07511

## Progress snapshot (after Batch 77)

- Completed: Batch 1 through Batch 77.
- Pending: Batch 78 onward.

## Batch 78 (max 5 files)

- [X] `scripts/phase5_active_learning/active_learning.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_discovery.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_phase6_active_learning.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_phase5_cli.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

- `python -m ruff check atlas/training/checkpoint.py atlas/training/preflight.py atlas/training/run_utils.py tests/unit/training/test_checkpoint.py tests/unit/training/test_preflight.py`
- `python -m pytest -q tests/unit/training/test_checkpoint.py tests/unit/training/test_preflight.py tests/unit/training/test_run_utils_manifest.py`

## Batch 78 optimization goals

- Eliminate silent integer truncation in random-acquisition controls (`n_pool`, `seed`) to keep AL sampling behavior reproducible and type-safe.
- Reject boolean values in discovery acquisition float controls (`acq_kappa`, `acq_jitter`, `acq_best_f`) to avoid implicit bool-to-float coercion.
- Keep CLI/runtime validation behavior backward compatible for valid numeric inputs while hard-failing malformed controls.
- Add regression tests locking strict coercion contracts.

## Research references used in batch 81

- Python `bool` type semantics (`bool` is subclass of `int`): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool
- Python `numbers` hierarchy (Integral/Real contracts): https://docs.python.org/3/library/numbers.html
- Python `subprocess.run` timeout/error behavior: https://docs.python.org/3/library/subprocess.html#subprocess.run
- Python `tempfile.NamedTemporaryFile` for safe temp-write patterns: https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile
- Python `pathlib.Path.replace` atomic rename semantics: https://docs.python.org/3/library/pathlib.html#pathlib.Path.replace
- PyTorch checkpoint serialization docs (`torch.save` / `torch.load`): https://pytorch.org/tutorials/beginner/saving_loading_models.html

## Batch 78 outcomes

- `scripts/phase5_active_learning/active_learning.py`:

  - added strict `_coerce_non_negative_int(...)` helper (rejects bool + non-integral reals).
  - `acquisition_random(...)` now uses strict coercion for `n_pool` and `seed`, removing implicit truncation (e.g., `3.5 -> 3`).
- `scripts/phase5_active_learning/run_discovery.py`:

  - added `_coerce_finite_float(...)` helper with bool rejection and finite/range checks.
  - `_validate_discovery_args(...)` now normalizes and validates `acq_kappa`, `acq_jitter`, and `acq_best_f` through the helper.
- `tests/unit/active_learning/test_phase6_active_learning.py`:

  - expanded `acquisition_random` regressions for non-integral and bool inputs (`n_pool`, `seed`).
- `tests/unit/active_learning/test_phase5_cli.py`:

  - added regression proving discovery validator rejects boolean acquisition controls.

## Verification

- `python -m ruff check scripts/phase5_active_learning/active_learning.py scripts/phase5_active_learning/run_discovery.py tests/unit/active_learning/test_phase6_active_learning.py tests/unit/active_learning/test_phase5_cli.py`
- `python -m py_compile scripts/phase5_active_learning/active_learning.py scripts/phase5_active_learning/run_discovery.py tests/unit/active_learning/test_phase6_active_learning.py tests/unit/active_learning/test_phase5_cli.py`
- `python -m pytest -q tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_phase6_active_learning.py`

## Research references used in batch 78

- Python built-in types (`bool` subtype of `int`): https://docs.python.org/3/library/stdtypes.html
- Python numeric tower (`numbers.Integral` / `numbers.Real`): https://docs.python.org/3/library/numbers.html
- Python argparse (official): https://docs.python.org/3/library/argparse.html
- Python subprocess (official): https://docs.python.org/3/library/subprocess.html
- NumPy Generator API (`default_rng`): https://numpy.org/doc/stable/reference/random/generator.html
- NumPy `Generator.choice` API: https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html
- Active Learning Literature Survey (Settles, 2009): https://burrsettles.com/pub/settles.activelearning_20090109.pdf
- EGO / Expected Improvement origin (Jones et al., 1998): https://r7-www1.stat.ubc.ca/efficient-global-optimization-expensive-black-box-functions
- Constrained Bayesian Optimization (Gardner et al., ICML 2014): https://proceedings.mlr.press/v32/gardner14.html
- Conformal prediction tutorial (Angelopoulos & Bates): https://arxiv.org/abs/2107.07511

## Progress snapshot (after Batch 78)

- Completed: Batch 1 through Batch 78.
- Pending: Batch 79 onward.

## Batch 79 (max 5 files)

- [X] `scripts/phase5_active_learning/search_materials.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/test_enumeration.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_search_materials_cli.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_test_enumeration_script.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 79 optimization goals

- Tighten CLI numeric-parameter governance in multi-property search to reject bool/non-finite payloads consistently.
- Eliminate implicit bool/float-to-int coercion leakage in demo summary-count validation.
- Preserve backward compatibility for valid numeric inputs while enforcing deterministic validation failures for malformed controls.
- Add regression tests locking these contracts.

## Batch 79 outcomes

- `scripts/phase5_active_learning/search_materials.py`:

  - added `_coerce_optional_finite_float(...)` to normalize optional numeric filters with explicit bool rejection.
  - `_validate_args(...)` now sanitizes all optional range fields and `ehull_max` through strict finite-value coercion.
  - simplified `--max` validation messaging by directly validating with canonical option name.
- `scripts/phase5_active_learning/test_enumeration.py`:

  - added strict `_coerce_non_negative_int(...)` helper for summary counts.
  - summary count fields now use strict non-negative integer coercion (rejects bool and fractional values).
- `tests/unit/active_learning/test_search_materials_cli.py`:

  - added regressions proving bool-valued numeric filters (`ehull_max`, `bandgap_min`) are rejected.
- `tests/unit/active_learning/test_test_enumeration_script.py`:

  - added regressions for `_coerce_non_negative_int(...)` strictness on bool/fractional inputs.

## Verification

- `python -m ruff check scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/test_enumeration.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_test_enumeration_script.py`
- `python -m py_compile scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/test_enumeration.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_test_enumeration_script.py`
- `python -m pytest -q tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_test_enumeration_script.py`

## Research references used in batch 79

- Python built-in types (`bool` subtype of `int`): https://docs.python.org/3/library/stdtypes.html
- Python numeric tower (`numbers.Integral` / `numbers.Real`): https://docs.python.org/3/library/numbers.html
- Python argparse (official): https://docs.python.org/3/library/argparse.html
- Pandas user guide (IO + table display/query context): https://pandas.pydata.org/docs/user_guide/index.html
- NumPy finite checks (`isfinite`): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- pymatgen StructureMatcher docs: https://pymatgen.org/pymatgen.analysis.html#pymatgen.analysis.structure_matcher.StructureMatcher
- pymatgen Structure API: https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure
- Hart & Forcade derivative-structure enumeration: https://doi.org/10.1107/S0108767308028503
- Hart, Nelson & Forcade multicomponent derivatives: https://doi.org/10.1016/j.commatsci.2012.02.015
- JARVIS-DFT database paper (Sci Data): https://doi.org/10.1038/s41597-020-00723-1

## Progress snapshot (after Batch 79)

- Completed: Batch 1 through Batch 79.
- Pending: Batch 80 onward.

## Batch 80 (max 5 files)

- [X] `atlas/models/prediction_utils.py` - reviewed + optimized
- [X] `atlas/models/utils.py` - reviewed + optimized
- [X] `atlas/utils/reproducibility.py` - reviewed + optimized
- [X] `tests/unit/models/test_prediction_utils.py` - reviewed + optimized
- [X] `tests/unit/research/test_reproducibility.py` - reviewed + optimized

## Batch 80 optimization goals

- Harden prediction payload normalization so non-finite mean outputs cannot leak into downstream ranking/metrics.
- Tighten reproducibility coercion semantics to avoid implicit bool/float truncation drift in `seed` and deterministic flags.
- Keep checkpoint normalizer validation lightweight and deterministic.
- Add regression tests that lock strict coercion and sanitized-mean contracts.

## Batch 80 outcomes

- `atlas/models/prediction_utils.py`:

  - added `_sanitize_mean_like(...)` and applied it to `mean/mu/gamma` extraction paths.
  - evidential payload parsing now sanitizes `gamma` before uncertainty derivation.
  - non-finite model means now map to bounded finite values (`nan->0`, `+inf->1e6`, `-inf->-1e6`).
- `atlas/utils/reproducibility.py`:

  - added `_is_integral_float(...)` and switched bool/seed coercion to strict integral semantics.
  - `_coerce_bool(...)` now rejects non-integral floats (e.g., `0.7`) and ambiguous numeric strings unless integral-like.
  - `_coerce_seed(...)` now rejects bool and non-integral float/string seeds (falls back to default), preserving uint32 normalization.
- `atlas/models/utils.py`:

  - replaced tensor-based finiteness check in scalar normalizer validation with `math.isfinite(...)` for lower overhead and clearer intent.
- Tests:

  - `test_prediction_utils.py` adds non-finite mean sanitization coverage for both direct and evidential payloads.
  - `test_reproducibility.py` adds strict coercion regressions for `_coerce_seed` and `_coerce_bool`.

## Verification

- `python -m ruff check atlas/models/prediction_utils.py atlas/models/utils.py atlas/utils/reproducibility.py tests/unit/models/test_prediction_utils.py tests/unit/research/test_reproducibility.py`
- `python -m pytest -q tests/unit/models/test_prediction_utils.py tests/unit/research/test_reproducibility.py`

## Research references used in batch 80

- Python built-in types (`bool` subtype of `int`): https://docs.python.org/3/library/stdtypes.html
- PEP 285 (`bool` type semantics): https://peps.python.org/pep-0285/
- Python numeric tower (`numbers`): https://docs.python.org/3/library/numbers.html
- NumPy finite-value checks: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy NaN/Inf sanitization: https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
- PyTorch deterministic behavior notes: https://pytorch.org/docs/stable/notes/randomness.html
- Python `random` module reproducibility notes: https://docs.python.org/3/library/random.html

## Progress snapshot (after Batch 80)

- Completed: Batch 1 through Batch 80.
- Pending: Batch 81 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 81.

## Progress snapshot (after Batch 81)

- Completed: Batch 1 through Batch 81.
- Pending: Batch 82 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 82.

# ATLAS Sequential Optimization Todo (Temp)

Last updated: 2026-03-05

## Batch 1 (max 5 files)

- [X] `atlas/data/split_governance.py` - reviewed + optimized
- [X] `tests/unit/data/test_split_governance.py` - reviewed + optimized
- [X] `atlas/data/topo_db.py` - reviewed + optimized
- [X] `tests/unit/data/test_topo_db.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - created + updated

## Batch 1 optimization goals

- Refactor high-complexity split/inference routines into smaller helpers.
- Remove local `# noqa: C901` suppressions where practical.
- Preserve deterministic behavior and split/calibration semantics.
- Add regression tests for determinism and calibration diagnostics.

## Batch 1 outcomes

- `topo_db.infer_topology_probabilities` no longer relies on `# noqa: C901`; config-validation logic extracted into reusable helpers.
- Split governance now sanitizes similarity matrices and guards optimizer hyperparameters (`n_restarts`, `local_moves`) against invalid values.
- Added regression tests for:

  - split optimizer guard behavior (`n_restarts=0`)
  - similarity-matrix sanitization with NaN/Inf values
  - invalid `weight_constraint` handling
  - invalid `base_weights` shape handling

## Research references used in this batch

- Kernighan, Lin (1970), graph partition local-improvement heuristic: https://doi.org/10.1002/j.1538-7305.1970.tb01770.x
- Fiduccia, Mattheyses (1982), linear-time partition improvement: https://doi.org/10.1145/800263.809204
- Guo et al. (ICML 2017), temperature scaling calibration: https://proceedings.mlr.press/v70/guo17a.html
- Ledoit, Wolf (2004), covariance shrinkage/conditioning: https://doi.org/10.1016/S0047-259X(03)00096-4
- Joeres et al. (Nature Communications 2025), leakage-aware splitting (DataSAIL): https://www.nature.com/articles/s41467-025-58606-8

## Batch 2 (max 5 files)

- [X] `atlas/active_learning/policy_engine.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_phase5.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_discovery.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_phase5_cli.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 2 optimization goals

- Improve numeric stability in CMOEIC policy scoring path.
- Harden Phase5 CLI argument validation and override behavior.
- Add regression tests for CLI construction and argument guards.

## Batch 2 outcomes

- `PolicyEngine` now uses explicit helper functions for probability clamping, calibrated energy extraction, and base utility construction.
- Added safety guard for conformal denominator (`max_conformal_radius`) to avoid divide-by-zero behavior under malformed config.
- `run_phase5.py` now:

  - validates key numeric arguments before execution,
  - replaces profile default flags instead of appending duplicate flag/value pairs.
- `run_discovery.py` now validates core AL CLI arguments early (non-negative/finite/range checks).
- Added CLI regression tests covering:

  - policy-flag injection,
  - profile override replacement semantics,
  - invalid `--top > --candidates` guard.

## Research references used in batch 2

- Python argparse docs (official): https://docs.python.org/3/library/argparse.html
- Python subprocess docs (official): https://docs.python.org/3/library/subprocess.html
- Jones et al. (1998), Efficient Global Optimization / EI: https://doi.org/10.1023/A:1008306431147
- Gardner et al. (2014), Bayesian optimization with inequality constraints: https://proceedings.mlr.press/v32/gardner14.html
- Angelopoulos, Bates (2021), conformal prediction tutorial: https://arxiv.org/abs/2107.07511

## Batch 3 (max 5 files)

- [X] `atlas/active_learning/policy_engine.py` - reviewed + optimized (numerical stability pass)
- [X] `scripts/phase5_active_learning/run_phase5.py` - reviewed + optimized (CLI hardening pass)
- [X] `scripts/phase5_active_learning/run_discovery.py` - reviewed + optimized (CLI guard pass)
- [X] `tests/unit/active_learning/test_phase5_cli.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 3 optimization goals

- Strengthen policy-scoring numeric guards and helper decomposition.
- Make Phase5 CLI behavior deterministic under overrides.
- Enforce argument-domain constraints early (before heavy runtime init).

## Batch 3 outcomes

- `PolicyEngine`:

  - added `_clamp01` for consistent bounded probabilities,
  - extracted calibrated-stat and utility helpers to reduce scoring-path duplication,
  - added protected denominator for conformal scaling.
- `run_phase5.py`:

  - added `_validate_args` with domain checks,
  - added `_set_or_replace_flag` so profile defaults are replaced, not duplicated.
- `run_discovery.py`:

  - added `_validate_discovery_args` with fast-fail checks (`top<=candidates`, finite/positive constraints).
- `test_phase5_cli.py`:

  - added regression tests for override replacement semantics,
  - added guards tests for invalid `top/candidates` in both launchers.

## Research references used in batch 3

- Python `concurrent.futures` docs (timeouts/executor semantics): https://docs.python.org/3/library/concurrent.futures.html
- PEP 3148 (futures design rationale): https://peps.python.org/pep-3148/
- Bayesian optimization with constraints (Gardner et al., 2014): https://proceedings.mlr.press/v32/gardner14.html
- EI / Efficient Global Optimization (Jones et al., 1998): https://doi.org/10.1023/A:1008306431147
- Conformal prediction tutorial (Angelopoulos & Bates, 2021): https://arxiv.org/abs/2107.07511

## Batch 4 (max 5 files)

- [X] `atlas/active_learning/controller.py` - reviewed + optimized
- [X] `atlas/active_learning/policy_state.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_controller_runtime_stability.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_policy_state.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 4 optimization goals

- Harden policy/state config parsing against malformed profile and resume payload values.
- Stabilize runtime retry behavior with bounded exponential backoff + jitter controls.
- Add deterministic regression tests for retry/backoff and state sanitization paths.
- Keep compatibility with existing legacy-policy defaults.

## Batch 4 outcomes

- `policy_state.py`:

  - added typed coercion helpers (`_coerce_bool/_coerce_int/_coerce_float`) to safely parse profile values,
  - added policy retry-backoff config fields (`relax_retry_backoff_sec`, `relax_retry_backoff_max_sec`, `relax_retry_jitter`),
  - added `PolicyState.validated()` and strict `from_dict` sanitization to prevent invalid resume state (NaN/Inf/negative counters) from propagating.
- `controller.py`:

  - wired retry-backoff fields from policy config into runtime,
  - added `_retry_sleep_seconds()` implementing capped exponential backoff with optional jitter,
  - applied retry sleep between failed relax attempts and exposed backoff settings in workflow/report metadata.
- Runtime tests:

  - added retry recovery test proving transient failures can recover with bounded sleep,
  - added direct cap test for exponential retry schedule,
  - preserved existing timeout and circuit-breaker behavior tests.

## Research references used in batch 4

- Python `concurrent.futures` docs (timeouts and cancellation semantics): https://docs.python.org/3/library/concurrent.futures.html
- PEP 3148 (Futures design and retry-oriented execution model): https://peps.python.org/pep-3148/
- AWS Architecture Blog, "Exponential Backoff and Jitter": https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
- Dean and Barroso (2013), "The Tail at Scale": https://research.google/pubs/the-tail-at-scale/

## Batch 5 (max 5 files)

- [X] `atlas/active_learning/acquisition.py` - reviewed + optimized
- [X] `atlas/active_learning/gp_surrogate.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_acquisition.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_gp_surrogate.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 5 optimization goals

- Improve acquisition numeric robustness under malformed hyperparameters and noisy observations.
- Reduce NEI bias from invalid historical observations (NaN/Inf) by filtering, not zero-imputing.
- Add explicit GP surrogate config validation to prevent runtime instability from bad config values.
- Add regression tests for the new sanitization and stability behavior.

## Batch 5 outcomes

- `acquisition.py`:

  - added numeric coercion helpers for robust schedule/MC parameter parsing,
  - improved `_prepare_observed` to filter non-finite observation pairs before NEI sampling,
  - hardened `schedule_ucb_kappa` against invalid/non-finite inputs while preserving existing valid-path semantics.
- `gp_surrogate.py`:

  - added `GPSurrogateConfig.validated()` to sanitize numeric ranges and categorical modes,
  - enforced config validation at `GPSurrogateAcquirer` initialization.
- Tests:

  - `test_acquisition.py` now verifies:

    - dirty observations (NaN/Inf) are filtered consistently in NEI,
    - UCB kappa schedule stays finite under invalid input.
  - `test_gp_surrogate.py` now verifies invalid config fields are normalized to safe values and produce finite `current_kappa`.

## Research references used in batch 5

- BoTorch acquisition docs (`qNoisyExpectedImprovement`, MC noisy BO semantics): https://botorch.readthedocs.io/en/stable/acquisition.html
- scikit-learn `GaussianProcessRegressor` docs (`alpha` and noise handling): https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
- PyTorch `torch.isfinite` API (finite-value screening semantics): https://pytorch.org/docs/stable/generated/torch.isfinite.html
- Ament et al. (2023), LogEI numerical stabilization: https://arxiv.org/abs/2310.20708

## Batch 6 (max 5 files)

- [X] `atlas/active_learning/objective_space.py` - reviewed + optimized
- [X] `atlas/active_learning/pareto_utils.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_objective_space.py` - added + optimized
- [X] `tests/unit/active_learning/test_pareto_utils.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 6 optimization goals

- Harden objective-space conversion against NaN/Inf/out-of-range values.
- Make Pareto/HV helpers robust to malformed arrays and empty high-dimensional inputs.
- Preserve ranking semantics while preventing invalid points from polluting front extraction.
- Expand tests to cover numeric edge cases and new sanitization paths.

## Batch 6 outcomes

- `objective_space.py`:

  - `clip01` now safely handles non-finite inputs,
  - added internal row sanitizer for objective-map conversion (`NaN/Inf` -> bounded finite values),
  - strengthened dimensional coercion and feasibility-mask guards for 1D/non-finite arrays.
- `pareto_utils.py`:

  - added generic 2D-shape normalization helper for point matrices,
  - fixed empty Pareto-front return shape to preserve input objective dimension (not hard-coded to 2D),
  - added finite-value guards in non-dominated sorting and hypervolume estimation,
  - added defensive casting for MC HV sampling parameters (`samples/seed/chunk`).
- Tests:

  - new `test_objective_space.py` for map sanitization, joint-feasibility filtering, and non-finite mask behavior,
  - expanded `test_pareto_utils.py` for empty-shape preservation, non-finite ranking behavior, and HV finite-row filtering.

## Research references used in batch 6

- Deb et al. (2002), NSGA-II fast non-dominated sorting and crowding distance: https://doi.org/10.1109/4235.996017
- BoTorch acquisition docs (MC BO stability and constrained utility context): https://botorch.readthedocs.io/en/stable/acquisition.html
- NumPy `isfinite` reference (finite-value masking semantics): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy `nan_to_num` reference (controlled NaN/Inf replacement): https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html

## Batch 7 (max 5 files)

- [X] `atlas/active_learning/generator.py` - reviewed + optimized
- [X] `atlas/active_learning/synthesizability.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_generator.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_synthesizability.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 7 optimization goals

- Improve generator robustness under non-finite logits and single-worker runtime environments.
- Remove hash-randomization-induced nondeterminism from generator fallback Monte Carlo paths.
- Harden synthesizability evaluator configuration against malformed numeric/bool inputs.
- Add regression coverage for new sanitization and deterministic behavior.

## Batch 7 outcomes

- `generator.py`:

  - strengthened `_softmax` to handle NaN/Inf/empty logits safely and always return a valid probability vector,
  - replaced fallback MC seeding based on Python `hash()` with stable `blake2b`-derived seed material,
  - added single-worker synchronous execution path in `generate_batch` to avoid unnecessary `ProcessPoolExecutor` overhead,
  - cached seed fingerprints incrementally to reduce repeated fingerprint recomputation.
- `synthesizability.py`:

  - added typed coercion helpers (`_coerce_int/_coerce_float/_coerce_bool`) for config parsing,
  - normalized/clamped thresholds and repaired inverted threshold bounds (`threshold_min > threshold_max`),
  - hardened objective-weight normalization against NaN/non-finite values,
  - added finite guards in energy prior and final score composition.
- Tests:

  - `test_generator.py` now verifies softmax non-finite handling, charge-neutrality fallback determinism, and single-worker generation path.
  - `test_synthesizability.py` now verifies invalid config sanitization and finite score outputs for non-finite energy inputs.

## Research references used in batch 7

- Python `PYTHONHASHSEED` docs (hash randomization behavior): https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
- Python `hashlib` docs (`blake2b` stable digesting): https://docs.python.org/3/library/hashlib.html
- Python `ProcessPoolExecutor` docs (process-spawn/serialization model): https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor
- NumPy `SeedSequence` docs (reproducible seed material rationale): https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html

## Batch 8 (max 5 files)

- [X] `atlas/active_learning/crabnet_native.py` - reviewed + optimized
- [X] `atlas/active_learning/rxn_network_native.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_crabnet_native.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_rxn_network_native.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 8 optimization goals

- Harden uncertainty-calibration loops against NaN/Inf/degenerate std inputs.
- Improve ambiguous tensor input-order detection robustness in CrabNet wrapper.
- Stabilize reaction-network risk ranking under extreme cost scales and non-finite values.
- Add targeted regression tests proving finite-safe behavior and fallback consistency.

## Batch 8 outcomes

- `crabnet_native.py`:

  - added finite-safe coercion helpers for positive/non-negative scalar config values,
  - improved `_normalize_input_order` with fraction-likelihood scoring + finite-aware max fallback,
  - hardened uncertainty calibration by filtering invalid rows before quantile estimation,
  - grouped calibration now preserves prior calibration state when current calibration data is fully invalid.
- `rxn_network_native.py`:

  - `_safe_float` now rejects NaN/Inf and falls back deterministically,
  - `_normalize_weights` now ignores unknown/non-finite weight entries to prevent accidental metric dilution,
  - `_path_step_costs` now filters invalid cost vectors and falls back to reaction energies safely,
  - `_entropic_risk` now uses a numerically stable log-mean-exp formulation (shifted/log-sum-exp style),
  - Pareto objective arrays are sanitized before non-dominated sorting/crowding computations.
- Tests:

  - `test_crabnet_native.py` now covers non-finite input-order ambiguity and invalid-row calibration behavior.
  - `test_rxn_network_native.py` now covers non-finite float handling, weight sanitization, extreme-scale entropic risk, and invalid-cost fallback.

## Research references used in batch 8

- Blanchard, Higham, Higham (2019), accurate log-sum-exp/softmax computation: https://arxiv.org/abs/1909.03469
- PyTorch `torch.isfinite` docs: https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- PyTorch `torch.quantile` docs: https://docs.pytorch.org/docs/stable/generated/torch.quantile.html
- NumPy `nan_to_num` docs: https://numpy.org/doc/2.4/reference/generated/numpy.nan_to_num.html
- Ahmadi-Javid (2012), entropic risk measure: https://doi.org/10.1016/j.ejor.2011.11.016

## Batch 9 (max 5 files)

- [X] `atlas/config.py` - reviewed + optimized
- [X] `atlas/benchmark/runner.py` - reviewed + optimized
- [X] `tests/unit/config/test_config.py` - reviewed + optimized
- [X] `tests/unit/benchmark/test_benchmark_runner.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 9 optimization goals

- Strengthen configuration path/device handling for CI and heterogeneous runtime environments.
- Make benchmark-runner initialization and bootstrap routines finite-safe under malformed numeric inputs.
- Preserve existing benchmark semantics while preventing avoidable runtime crashes.
- Add regression tests for the new sanitization behavior.

## Batch 9 outcomes

- `config.py`:

  - `PathConfig` now preserves explicitly supplied `raw_dir/processed_dir/artifacts_dir` values instead of always overriding from `data_dir`,
  - added path normalization (`expanduser`, project-root-relative resolution) for env and explicit inputs,
  - `Config._set_device` now safely falls back to CPU on invalid device strings or torch availability issues,
  - `get_device` now surfaces invalid device strings clearly.
- `benchmark/runner.py`:

  - added typed coercion helpers for `batch_size`, `n_jobs`, bootstrap config, coverage values, and conformal calibration limits,
  - made bootstrap CI utilities robust to invalid `confidence/n_bootstrap/seed` inputs,
  - hardened `conformal_max_calibration_samples` parsing inside conformal metric path.
- Tests:

  - `test_config.py` now covers explicit-subdir preservation, env-relative path resolution, and invalid device fallback,
  - `test_benchmark_runner.py` now covers bootstrap invalid-parameter sanitization and invalid runtime-parameter coercion at runner init.

## Research references used in batch 9

- Joblib Parallel docs (`n_jobs` semantics including `n_jobs=0` invalid): https://joblib.readthedocs.io/en/stable/parallel.html
- PyTorch data loading reference (`torch.utils.data` / DataLoader API): https://docs.pytorch.org/docs/stable/data.html
- PyTorch `torch.device` API reference: https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch-device
- NumPy `RandomState.randint` reference (bootstrap resampling primitive): https://numpy.org/doc/stable/reference/random/generated/numpy.random.RandomState.randint.html
- Angelopoulos & Bates (2021), conformal prediction tutorial: https://arxiv.org/abs/2107.07511

## Batch 10 (max 5 files)

- [X] `atlas/active_learning/crabnet_screener.py` - reviewed + optimized
- [X] `atlas/active_learning/controller.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_crabnet_screener.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_controller_acquisition.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 10 optimization goals

- Harden composition screener numeric/config behavior under malformed runtime hyperparameters.
- Make training-loss path robust to non-finite supervision labels (without crashing rounds).
- Prevent active-learning acquisition instability when historical energy traces contain NaN/Inf.
- Add regression tests for new finite-value guards and fallback behavior.

## Batch 10 outcomes

- `crabnet_screener.py`:

  - added finite-safe coercion for critical hyperparameters (`simplex_blend`, transform temperatures, uncertainty floor, ensemble/mc counts),
  - hardened uncertainty decode path using `torch.nan_to_num` + clamping to prevent std explosions from bad raw heads,
  - sanitized aggregated aleatoric/epistemic variance tensors before sqrt,
  - updated `compute_training_loss` to ignore non-finite target rows and return stable zero scalar when no valid supervision exists.
- `controller.py`:

  - `_current_best_f` now filters non-finite historical observations before min/quantile,
  - `_historical_energy_observations` now drops non-finite means and clamps invalid std inputs,
  - `_current_acquisition_kappa` now falls back to base kappa when scheduler output is non-finite/non-positive,
  - `_stability_component` now falls back to deterministic score when candidate UQ stats are non-finite.
- Tests:

  - `test_crabnet_screener.py` now verifies invalid numeric hyperparameter sanitization and non-finite target loss masking,
  - `test_controller_acquisition.py` now verifies finite filtering in history, `best_f` fallback, kappa fallback, and non-finite candidate UQ fallback.

## Research references used in batch 10

- PyTorch `GaussianNLLLoss` docs (variance positivity and numerical epsilon): https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.GaussianNLLLoss.html
- PyTorch `torch.nan_to_num` docs (NaN/Inf replacement semantics): https://docs.pytorch.org/docs/stable/generated/torch.nan_to_num.html
- PyTorch `torch.isfinite` docs (finite-mask behavior): https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- Srinivas et al. (2010), GP-UCB schedule rationale: https://arxiv.org/abs/0912.3995
- Ament et al. (2023), numerical pathologies in EI-family acquisitions: https://arxiv.org/abs/2310.20708

## Batch 11 (max 5 files)

- [X] `atlas/training/run_utils.py` - reviewed + optimized
- [X] `atlas/utils/reproducibility.py` - reviewed + optimized
- [X] `tests/unit/training/test_run_utils_manifest.py` - reviewed + optimized
- [X] `tests/unit/research/test_reproducibility.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 11 optimization goals

- Improve manifest determinism and strict JSON compatibility for reproducibility artifacts.
- Harden run-manifest merge behavior when prior manifest sections have corrupted types.
- Make global-seed helper robust to malformed/non-finite seed inputs and cross-platform hash-seed constraints.
- Add regression tests for strict serialization and seed normalization behavior.

## Batch 11 outcomes

- `run_utils.py`:

  - `_json_safe` now sanitizes non-finite floats (`NaN/Inf` -> `null`) and emits deterministic ordering for dict/set payloads,
  - replaced eager file hashing with chunked SHA256 streaming to reduce peak memory on large manifests/locks,
  - added `_ensure_manifest_section` to repair malformed manifest sections during merge (`list/str` -> `{}`),
  - simplified manifest writing pipeline to a single canonical payload write pass (JSON + YAML mirror), with deterministic key ordering and `allow_nan=False`.
- `reproducibility.py`:

  - added robust seed coercion to uint32 domain, including non-finite input fallback,
  - normalized `PYTHONHASHSEED` handling and surfaced original input in metadata (`seed_input`),
  - runtime metadata now includes `python_hash_seed`.
- Tests:

  - `test_run_utils_manifest.py` now covers non-finite serialization sanitization, deterministic set ordering, and corrupted-section repair on merge,
  - `test_reproducibility.py` now covers non-finite seed fallback and negative seed normalization.

## Research references used in batch 11

- Python `json` module docs (`allow_nan` and standard-compliant output): https://docs.python.org/3/library/json.html
- Python `PYTHONHASHSEED` docs: https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
- Python `random.seed` docs: https://docs.python.org/3/library/random.html#random.seed
- NumPy `random.seed` docs: https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
- NIST FIPS 180-4 (SHA-256 standard): https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf

## Batch 12 (max 5 files)

- [X] `atlas/data/data_validation.py` - reviewed + optimized
- [X] `atlas/data/source_registry.py` - reviewed + optimized
- [X] `tests/unit/data/test_data_validation.py` - reviewed + optimized
- [X] `tests/unit/data/test_source_registry.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 12 optimization goals

- Strengthen data-validation output interoperability by enforcing strict JSON-safe serialization.
- Harden source-registry reliability/correlation fusion against non-finite numeric inputs.
- Prevent duplicate detector from treating null IDs as a real duplicate key.
- Add regression coverage for non-finite guards and malformed-input recovery paths.

## Batch 12 outcomes

- `data_validation.py`:

  - added recursive `_json_safe` conversion for report serialization (`NaN/Inf -> null`, NumPy scalars -> Python scalars),
  - `ValidationReport.to_json` now emits deterministic strict JSON (`sort_keys=True`, `allow_nan=False`),
  - `check_duplicates` now ignores `None` IDs instead of collapsing them into `"None"` duplicate buckets.
- `source_registry.py`:

  - hardened Beta reliability stats (`mean/variance`) for degenerate/invalid priors,
  - `register` now sanitizes invalid reliability priors and rejects empty source keys,
  - `update_reliability` rejects non-finite updates,
  - drift-aware source scoring now guards non-finite drift inputs,
  - correlation/covariance normalization now sanitizes non-finite matrices,
  - residual-based correlation estimation filters non-finite residual traces,
  - GLS fusion now skips invalid estimates (non-finite value/std) and validates covariance denominator robustness.
- Tests:

  - `test_data_validation.py` now verifies null-ID duplicate handling and strict JSON non-finite sanitization.
  - `test_source_registry.py` now verifies invalid-prior sanitization, non-finite update rejection, finite-safe drift scoring, and invalid-estimate filtering in fusion.

## Research references used in batch 12

- Python `json` docs (`allow_nan=False` strictness): https://docs.python.org/3/library/json.html
- NumPy `isfinite` docs (finite filtering semantics): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy `linalg.pinv` docs (stable pseudo-inverse in GLS weighting): https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
- Ledoit & Wolf (2004), covariance shrinkage: https://doi.org/10.1016/S0047-259X(03)00096-4
- Ben-David et al. (2010), domain-shift theory: https://jmlr.org/papers/v10/ben-david09a.html

## Batch 13 (max 5 files)

- [X] `atlas/data/alloy_estimator.py` - reviewed + optimized
- [X] `atlas/data/property_estimator.py` - reviewed + optimized
- [X] `tests/unit/data/test_alloy_estimator.py` - reviewed + optimized
- [X] `tests/unit/data/test_property_estimator.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 13 optimization goals

- Harden alloy/property estimation pipelines against non-finite runtime inputs (`NaN/Inf`) in weights, densities, and uncertainty hyperparameters.
- Ensure Gaussian fusion uses only statistically valid observations (`value` finite and `sigma > 0`) and stays stable in correlated-GLS mode.
- Keep downstream search/summary utilities robust under malformed caller arguments.
- Add regression tests proving finite-safe behavior and fallback semantics.

## Batch 13 outcomes

- `alloy_estimator.py`:

  - `AlloyPhase.get` now enforces finite-safe scalar conversion and falls back cleanly for malformed properties.
  - added explicit finite coercion helpers for non-negative/positive inputs and normalized-fraction helper used across weighting paths.
  - `convert_wt_to_vol` and `_normalize_weight_fractions` now sanitize non-finite weight/density values before normalization.
  - Reuss/Wiener/entropy helpers now ignore invalid rows and return stable finite results even with dirty phase data.
  - `estimate_properties` now normalizes finite-safe `wf/vf` and guards density/thermal/melting channels from non-finite contamination.
  - `print_report` experimental comparison now skips invalid/non-positive experimental targets.
- `property_estimator.py`:

  - added finite-safe coercion for sigma/correlation/temperature/fallback-mass hyperparameters.
  - `_precision_fusion` validity mask now requires finite values plus finite positive sigmas.
  - correlated GLS branch now sanitizes covariance and adds small diagonal jitter for inversion stability.
  - `_normal_cdf` now handles non-finite z-scores deterministically via bounded `nan_to_num`.
  - `search` now validates `max_results`; `property_summary` now handles numeric-only describe output safely.
- Tests:

  - `test_alloy_estimator.py` now covers non-finite weight/density sanitization in volume conversion and custom-alloy normalization.
  - `test_property_estimator.py` now covers invalid hyperparameter sanitization, sigma-invalid fusion filtering, and `search` fallback limit behavior.

## Research references used in batch 13

- NumPy `isfinite` docs (finite-mask semantics): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy `nan_to_num` docs (deterministic NaN/Inf replacement): https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
- NumPy `linalg.pinv` docs (robust pseudo-inverse fallback for GLS): https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
- Ledoit & Wolf (2004), covariance conditioning rationale: https://doi.org/10.1016/S0047-259X(03)00096-4
- Anderson (1963), Debye temperature from elastic constants: https://doi.org/10.1016/0022-3697(63)90067-2

## Batch 14 (max 5 files)

- [X] `atlas/data/crystal_dataset.py` - reviewed + optimized
- [X] `atlas/data/jarvis_client.py` - reviewed + optimized
- [X] `tests/unit/data/test_crystal_dataset.py` - reviewed + optimized
- [X] `tests/unit/data/test_jarvis_client.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 14 optimization goals

- Harden data-pipeline split/sampling primitives against malformed runtime inputs (invalid split names, non-finite ratios/features, dirty manifest rows).
- Improve robustness of probabilistic stability/topology scoring when uncertainty hyperparameters contain NaN/Inf or invalid ranges.
- Preserve deterministic coreset behavior even when feature tensors include non-finite values.
- Add regression tests for newly hardened edge cases.

## Batch 14 outcomes

- `crystal_dataset.py`:

  - added explicit split validation (`train/val/test`) and early split-ratio schema checks (finite, non-negative, sum>0),
  - improved formula fallback parser to preserve stoichiometric counts (not symbol-only token counting),
  - hardened k-center coreset routine with 2D shape validation + `nan_to_num` sanitization,
  - manifest assignment loader now filters invalid rows (`sample_id` missing, split not in train/val/test),
  - strengthened constructor guards for `max_samples`, `graph_cutoff`, and non-integer `min_labeled_properties`.
- `jarvis_client.py`:

  - added finite-safe coercion helpers for non-negative, positive, and probability-bounded scalars,
  - `_normal_cdf` now handles non-finite z-scores deterministically,
  - ehull noise estimation now sanitizes `base_noise` and adaptive slope inputs,
  - k-center selection path now sanitizes non-finite feature matrices before distance updates,
  - `_sample_dataframe` now validates strategy first and handles non-positive sample count safely,
  - `get_stable_materials` and `get_topological_materials` now sanitize key hyperparameters (`ehull_max`, `noise`, `prob thresholds`, `fusion weights`, calibration temperature) before scoring.
- Tests:

  - `test_crystal_dataset.py` now covers invalid split/split_ratio rejection, non-finite k-center feature handling, and manifest assignment row filtering.
  - `test_jarvis_client.py` now covers invalid probabilistic parameter sanitization, topology fusion sanitization, and k-center sampling stability under non-finite feature inputs.

## Research references used in batch 14

- Gonzalez (1985), metric k-center farthest-first approximation: https://doi.org/10.1016/0304-3975(85)90224-5
- Sener & Savarese (2018), core-set active learning intuition: https://arxiv.org/abs/1708.00489
- Sun et al. (2016), metastability/Ehull scale in inorganic materials: https://doi.org/10.1126/sciadv.1600225
- NumPy `nan_to_num` docs (finite sanitization semantics): https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
- NumPy `isfinite` docs (finite mask semantics): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html

## Batch 15 (max 5 files)

- [X] `atlas/models/prediction_utils.py` - reviewed + optimized
- [X] `atlas/models/uncertainty.py` - reviewed + optimized
- [X] `tests/unit/models/test_prediction_utils.py` - reviewed + optimized
- [X] `tests/unit/models/test_uncertainty.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 15 optimization goals

- Harden prediction normalization utilities against malformed uncertainty payloads (`NaN/Inf`, missing std, uninspectable signatures).
- Remove avoidable UQ numerical pathologies (single-member/single-sample std producing unstable values, non-finite ensemble predictions).
- Improve MC-dropout runtime correctness by restoring model train/eval state after stochastic inference.
- Add regression tests for numeric stability and API edge-case behavior.

## Batch 15 outcomes

- `prediction_utils.py`:

  - added `_sanitize_std_like` to enforce finite, non-negative std tensors across tuple/dict payload formats,
  - evidential payload path now sanitizes non-finite `nu/alpha/beta` before variance composition to avoid unstable division,
  - tuple/list payloads with `None` std are now handled explicitly as deterministic outputs,
  - `forward_graph_model` now tolerates `inspect.signature` failures and falls back safely.
- `uncertainty.py`:

  - added unified payload normalization supporting both dict and tensor model outputs,
  - constructor guards now reject invalid `n_models` / `n_samples`,
  - ensemble/MC std now uses stable population estimator (`unbiased=False`) and sanitizes non-finite prediction stacks,
  - MC dropout now enables all dropout variants (`_DropoutNd`) and restores original model training state after inference,
  - evidential regression `total_std` and loss paths now sanitize non-finite values and clamp unsafe logarithm denominators.
- Tests:

  - `test_prediction_utils.py` now covers `None` std tuples, non-finite std sanitization, evidential non-finite payload stability, uninspectable signature fallback, and missing-edge-feature error path.
  - `test_uncertainty.py` now covers invalid constructor arguments, single-member/sample std stability, tensor-output compatibility, MC state restoration, and evidential loss robustness under non-finite targets.

## Research references used in batch 15

- PyTorch `torch.std` docs (Bessel correction / population std semantics): https://docs.pytorch.org/docs/stable/generated/torch.std.html
- PyTorch `torch.nan_to_num` docs (finite-value replacement semantics): https://docs.pytorch.org/docs/stable/generated/torch.nan_to_num.html
- Lakshminarayanan et al. (NeurIPS 2017), deep ensemble uncertainty: https://papers.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles
- Gal & Ghahramani (ICML 2016), MC dropout as approximate Bayesian inference: https://proceedings.mlr.press/v48/gal16.html
- Amini et al. (NeurIPS 2020), deep evidential regression: https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html

## Batch 16 (max 5 files)

- [X] `atlas/models/utils.py` - reviewed + optimized
- [X] `atlas/models/graph_builder.py` - reviewed + optimized
- [X] `tests/unit/models/test_model_utils.py` - reviewed + optimized
- [X] `tests/unit/models/test_structure_expansion.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 16 optimization goals

- Harden checkpoint loading and state-dict key normalization to avoid silent key-collision corruption.
- Improve graph construction robustness for malformed numerical inputs (non-finite distances, invalid Gaussian basis settings, zero-neighbor structures).
- Add stronger parameter validation at graph-builder/model-loading boundaries for earlier deterministic failures.
- Add regression tests for collision detection, malformed payload handling, and graph fallback behavior.

## Batch 16 outcomes

- `models/utils.py`:

  - added collision-safe state-dict key normalization with explicit error on conflicting normalized keys,
  - strengthened CGCNN config inference with tensor-rank validation for critical weights,
  - normalizer loading now validates finite numeric `(mean, std)` and rejects malformed normalizer payloads,
  - phase1/phase2 checkpoint loaders now fail early on invalid checkpoint payload structure (`model_state_dict` non-dict).
- `models/graph_builder.py`:

  - `gaussian_expansion` now validates `cutoff/n_gaussians`, sanitizes non-finite distances, and supports `n_gaussians=1` safely,
  - `CrystalGraphBuilder` constructor now validates finite positive `cutoff` and positive `max_neighbors`,
  - `element_features` no longer maps unknown elements to Hydrogen one-hot by default,
  - `structure_to_graph` now rejects empty structures and uses per-node self-loop fallback when no neighbors are found,
  - edge vectors and emitted PyG tensors are now explicitly sanitized/typed (`float32`/`long`) for stable downstream use.
- Tests:

  - `test_model_utils.py` now covers normalized-key collision errors, malformed normalizer payload fallback, and invalid phase2 state-dict payload rejection.
  - `test_structure_expansion.py` now covers single-basis Gaussian expansion, non-finite distance sanitization, invalid parameter rejection, graph-builder init validation, per-node self-loop fallback, and empty-structure guard.

## Research references used in batch 16

- PyTorch `torch.load` docs (checkpoint payload semantics): https://docs.pytorch.org/docs/stable/generated/torch.load.html
- PyTorch Geometric `Data` docs (graph tensor schema expectations): https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html
- pymatgen `Structure.get_all_neighbors` docs (neighbor construction API): https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure.get_all_neighbors
- Xie & Grossman (2018), CGCNN representation assumptions: https://doi.org/10.1103/PhysRevLett.120.145301
- Schütt et al. (2018), radial/Gaussian distance basis in atomistic GNNs: https://doi.org/10.1063/1.5019779

## Batch 17 (max 5 files)

- [X] `atlas/models/cgcnn.py` - reviewed + optimized
- [X] `atlas/models/multi_task.py` - reviewed + optimized
- [X] `tests/unit/models/test_cgcnn.py` - reviewed + optimized
- [X] `tests/unit/models/test_multi_task.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 17 optimization goals

- Strengthen CGCNN constructor/input guards to fail fast on malformed runtime tensors and invalid hyperparameters.
- Improve MultiTaskGNN task-schema validation and encoder-kwargs passthrough robustness under introspection failure.
- Remove subtle task-head consistency risks (unknown task types, duplicate task registration, invalid tensor types).
- Add regression tests for edge-case failures and new validation behavior.

## Batch 17 outcomes

- `cgcnn.py`:

  - added explicit numeric validation for model dimensions/layer counts and dropout range,
  - added `_validate_graph_inputs` with shape checks for `node_feats/edge_index/edge_feats/batch`,
  - added safe `batch` casting to `long` during pooling path,
  - attention pooling logits now use `torch.nan_to_num` before sparse softmax to reduce NaN/Inf propagation risk.
- `multi_task.py`:

  - added global task/type schema constants and validation (`scalar/evidential/tensor` and tensor subtype checks),
  - `TensorHead` now strictly validates `tensor_type`; removed duplicate conditional branch in tensor reconstruction,
  - `MultiTaskGNN` constructor now validates task config structure and rejects unknown task types early,
  - forward path now normalizes `tasks` input (`str` or list), ignores non-dict `encoder_kwargs`, and gracefully falls back when `inspect.signature` raises,
  - `add_task` now validates task type and rejects duplicate task registration.
- Tests:

  - `test_cgcnn.py` now covers invalid hyperparameter rejection and graph-input shape mismatch guards.
  - `test_multi_task.py` now covers unknown task/tensor type rejection, single-task string selection, duplicate add-task rejection, invalid add-task type rejection, and signature-failure fallback behavior.

## Research references used in batch 17

- Xie & Grossman (2018), CGCNN baseline architecture: https://doi.org/10.1103/PhysRevLett.120.145301
- Xu et al. (2018), Jumping Knowledge Networks motivation for multi-depth aggregation: https://arxiv.org/abs/1806.03536
- PyTorch Geometric `global_add_pool` docs (graph pooling contract): https://pytorch-geometric.readthedocs.io/en/2.5.2/generated/torch_geometric.nn.pool.global_add_pool.html
- PyTorch `torch.nan_to_num` docs (finite sanitization semantics): https://docs.pytorch.org/docs/stable/generated/torch.nan_to_num.html
- Python `inspect.signature` docs (introspection failure semantics): https://docs.python.org/3/library/inspect.html#inspect.signature

## Batch 18 (max 5 files)

- [X] `atlas/topology/classifier.py` - reviewed + optimized
- [X] `atlas/utils/structure.py` - reviewed + optimized
- [X] `tests/unit/topology/test_classifier.py` - reviewed + optimized
- [X] `tests/unit/models/test_structure_utils.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 18 optimization goals

- Eliminate topology-classifier pooling/probability edge-case failures (invalid scatter-mean semantics, multi-graph probability misuse, MC dropout state leakage).
- Strengthen runtime input validation for topology forward/proba paths to fail fast on malformed graph tensors.
- Improve structure utility robustness for empty structures and deterministic feature extraction.
- Add targeted regression tests for new guards and deterministic behavior.

## Batch 18 outcomes

- `topology/classifier.py`:

  - added strict constructor validation for dimensions/layer count/dropout domain,
  - added `_validate_inputs` for node/edge/batch shape consistency,
  - fixed pooling bias by setting `include_self=False` in `scatter_reduce_(reduce="mean")`,
  - added batch index safety checks (non-empty/non-negative),
  - `predict_proba` now validates single-graph contract, validates `n_samples`, and restores model train/eval state after MC dropout inference.
- `utils/structure.py`:

  - added deterministic site subsampling helper for nearest-neighbor feature estimation (removed stochastic `np.random.choice` variability),
  - `get_element_info` now handles empty structures safely and yields stable sorted element ordering,
  - `compute_structural_features` now handles empty structures with explicit zero/unknown fallback payloads,
  - sanitized scalar outputs (`float` conversions) for consistent serialization and downstream typing.
- Tests:

  - `test_classifier.py` now covers invalid hyperparameter rejection, multi-graph `predict_proba` rejection, MC dropout training-state restoration, and shape-mismatch validation.
  - `test_structure_utils.py` now covers empty-structure element/features handling and deterministic large-structure feature extraction.

## Research references used in batch 18

- PyTorch `Tensor.scatter_reduce_` docs (include_self semantics): https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html
- Gal & Ghahramani (ICML 2016), MC dropout Bayesian approximation rationale: https://proceedings.mlr.press/v48/gal16.html
- pymatgen `SpacegroupAnalyzer` docs (structure standardization): https://pymatgen.org/pymatgen.symmetry.html#pymatgen.symmetry.analyzer.SpacegroupAnalyzer
- pymatgen `Structure.get_neighbors` docs (local-neighbor extraction): https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure.get_neighbors
- spglib docs (symmetry detection backend used by pymatgen): https://spglib.readthedocs.io/

## Batch 19 (max 5 files)

- [X] `atlas/training/losses.py` - reviewed + optimized
- [X] `atlas/training/trainer.py` - reviewed + optimized
- [X] `tests/unit/training/test_losses.py` - reviewed + optimized
- [X] `tests/unit/training/test_trainer.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 19 optimization goals

- Harden training-loss paths against non-finite values and malformed configuration (loss type, task type, weighting strategy).
- Make trainer loop fail-fast for invalid runtime arguments and malformed loss outputs instead of silently continuing.
- Reduce hidden runtime coupling by narrowing fallback behavior (`TypeError`-only signature fallback).
- Add regression coverage to lock in the new guardrails under CI.

## Batch 19 outcomes

- `training/losses.py`:

  - added explicit supported-set validation for property loss types, multi-task strategies, and task types.
  - added finite-safe scalar validation for `constraint_weight`, `coeff`, and fixed task weights.
  - property loss now masks non-finite `pred/target` pairs (not only `NaN` targets) and validates unknown constraints explicitly.
  - evidential loss now validates required keys, skips invalid distribution rows (`nu/alpha/beta` domain + finite checks), and filters non-finite loss terms before reduction.
  - multi-task loss now validates empty task schema, infers device safely when prediction dict is sparse/empty, and skips non-finite per-task losses.
- `training/trainer.py`:

  - added finite non-negative validation for `grad_clip_norm` and `min_delta`.
  - `inspect.signature` failure is now handled gracefully with empty forward-param set fallback.
  - narrowed forward fallback exception scope to `TypeError` so runtime tensor/shape bugs are not silently swallowed.
  - `_compute_loss` now raises explicit error when a loss function returns an empty dict.
  - `fit` now validates `n_epochs/patience/checkpoint_name`, tracks last epoch/loss explicitly, and writes history with UTF-8 encoding.
- Tests:

  - `test_losses.py` now covers invalid config rejection, non-finite prediction masking, evidential missing-key/invalid-row behavior, and schema validation for `MultiTaskLoss`.
  - `test_trainer.py` now covers invalid grad-clip rejection, signature introspection fallback, invalid fit-arg rejection, and empty-loss-dict error path.
  - Verified with:

    - `python -m ruff check atlas/training/losses.py atlas/training/trainer.py tests/unit/training/test_losses.py tests/unit/training/test_trainer.py`
    - `python -m pytest tests/unit/training/test_losses.py tests/unit/training/test_trainer.py -q`
    - `python -m pytest tests/unit/training -q`

## Research references used in batch 19

- Kendall, Gal, Cipolla (CVPR 2018), uncertainty weighting for multi-task learning: https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html
- Amini et al. (NeurIPS 2020), deep evidential regression: https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html
- PyTorch `torch.nn.utils.clip_grad_norm_` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- PyTorch `torch.isfinite` docs: https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- Python `inspect.signature` docs: https://docs.python.org/3/library/inspect.html#inspect.signature

## Batch 20 (max 5 files)

- [X] `atlas/training/run_utils.py` - reviewed + optimized
- [X] `atlas/utils/reproducibility.py` - reviewed + optimized
- [X] `tests/unit/training/test_run_utils_manifest.py` - reviewed + optimized
- [X] `tests/unit/research/test_reproducibility.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 20 optimization goals

- Strengthen run-manifest generation under CI/runtime edge conditions (slow git commands, public-visibility redaction completeness, lock parsing).
- Prevent public manifests from leaking local absolute paths in `environment_lock`/default artifact fields.
- Improve deterministic-training controls so `deterministic=False` can explicitly disable deterministic mode after a prior deterministic run.
- Add regression tests for new manifest privacy and reproducibility guarantees.

## Batch 20 outcomes

- `training/run_utils.py`:

  - added hard timeout for git metadata subprocess calls to avoid manifest-generation hangs.
  - hardened strict-lock parsing to accept bool-like string values (`"true"/"1"/"yes"/"on"`).
  - switched default manifest artifact pointers to portable relative filenames (`run_manifest.json`, `run_manifest.yaml`) instead of absolute paths.
  - public visibility path redaction now explicitly re-sanitizes all core manifest sections (`dataset/split/environment_lock/artifacts/metrics/seeds/configs`) to prevent merge-existing path leakage.
  - `environment_lock` block now applies visibility-aware redaction for both default lock metadata and user-supplied overrides.
- `utils/reproducibility.py`:

  - deterministic policy now supports both enable and disable flows via `torch.use_deterministic_algorithms(deterministic_requested, warn_only=True)`.
  - deterministic mode now sets `CUBLAS_WORKSPACE_CONFIG` default when requested (per PyTorch reproducibility guidance).
  - added `deterministic_enabled` metadata to both seed-setting and runtime metadata paths.
  - cuDNN flags now switch consistently with deterministic request (`benchmark = not deterministic`, `deterministic = deterministic`).
- Tests:

  - `test_run_utils_manifest.py` now verifies public redaction covers `environment_lock.lock_file` and validates relative default artifact paths.
  - added strict-lock string parsing regression test (`strict_lock="true"`).
  - `test_reproducibility.py` now verifies deterministic metadata presence, cuBLAS workspace config behavior, and deterministic-algorithm toggle (`True -> False`) correctness.
  - Verified with:

    - `python -m ruff check atlas/training/run_utils.py atlas/utils/reproducibility.py tests/unit/training/test_run_utils_manifest.py tests/unit/research/test_reproducibility.py`
    - `python -m pytest tests/unit/training/test_run_utils_manifest.py tests/unit/research/test_reproducibility.py -q`

## Research references used in batch 20

- PyTorch reproducibility notes (deterministic algorithms + `CUBLAS_WORKSPACE_CONFIG`): https://pytorch.org/docs/stable/notes/randomness.html
- Python `subprocess.run` docs (`timeout` behavior): https://docs.python.org/3/library/subprocess.html#subprocess.run
- NumPy random seeding reference: https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
- Sandve et al. (2013), Ten Simple Rules for Reproducible Computational Research: https://doi.org/10.1371/journal.pcbi.1003285
- ACM Artifact Review and Badging v1.1 (reproducibility evidence expectations): https://www.acm.org/publications/policies/artifact-review-and-badging-current

## Batch 21 (max 5 files)

- [X] `scripts/phase5_active_learning/run_phase5.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_discovery.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_phase5_cli.py` - reviewed + optimized
- [X] `ruff.toml` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 21 optimization goals

- Strengthen Phase5/Discovery CLI input validation against non-finite numeric values and unsafe run identifiers.
- Prevent accidental path traversal style run-id payloads from reaching run-directory resolution logic.
- Expand CLI regression tests to lock the new guardrails in CI.
- Improve Ruff CI behavior consistency so explicit path lint invocations still respect exclude policy.

## Batch 21 outcomes

- `run_phase5.py`:

  - added finite-safe validation helper for acquisition numeric options (`kappa/jitter`) and finite check for `acq_best_f`.
  - added strict `run_id` safety check (disallow path separators and traversal-style values like `..`).
  - added guard for non-negative `preflight_split_seed`.
- `run_discovery.py`:

  - strengthened acquisition validation: `acq_kappa` and `acq_jitter` must be finite and non-negative.
  - added `run_id` safety validation and `results_dir` non-empty string guard.
- `test_phase5_cli.py`:

  - expanded argument-validation tests for non-finite floats (`NaN/Inf`) and unsafe run-id patterns (`../escape`, `..\\escape`).
  - updated discovery validation fixture payload to include optional fields covered by new guards.
- `ruff.toml`:

  - enabled `force-exclude = true` so direct-path CI lint invocations still honor configured excludes.
  - added explicit `extend-exclude` for cache/artifact/data directories to reduce lint noise and avoid accidental non-source lint scans.
- Verified with:

  - `python -m ruff check scripts/phase5_active_learning/run_phase5.py scripts/phase5_active_learning/run_discovery.py tests/unit/active_learning/test_phase5_cli.py`
  - `python -m pytest tests/unit/active_learning/test_phase5_cli.py -q`

## Research references used in batch 21

- Ruff configuration docs (`extend-exclude`, `force-exclude`): https://docs.astral.sh/ruff/configuration/
- Python `math.isfinite` docs: https://docs.python.org/3/library/math.html#math.isfinite
- Python `argparse` docs (type parsing and validation extension points): https://docs.python.org/3/library/argparse.html
- OWASP Path Traversal reference: https://owasp.org/www-community/attacks/Path_Traversal
- ACM Artifact Review and Badging v1.1 (repeatability and CI evidence expectations): https://www.acm.org/publications/policies/artifact-review-and-badging-current

## Batch 22 (max 5 files)

- [X] `scripts/phase5_active_learning/active_learning.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/structure_enumerator.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_phase6_active_learning.py` - added + optimized
- [X] `tests/unit/active_learning/test_structure_enumerator_script.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 22 optimization goals

- Harden Phase6 acquisition/Pareto helpers against non-finite numeric inputs and malformed argument shapes.
- Make active-learning loop state transitions safer (budget bounds, strategy validation, MC-dropout state restoration).
- Strengthen fallback structure enumerator input validation and incomplete-structure filtering behavior.
- Add dedicated unit tests for script-level algorithms that previously had no CI regression coverage.

## Batch 22 outcomes

- `active_learning.py`:

  - added explicit input validators for `batch_size`, acquisition shapes, and finite `best_so_far`.
  - `acquisition_uncertainty` now sanitizes NaN/Inf std values and handles zero/oversized batch requests safely.
  - `acquisition_expected_improvement` now enforces shape consistency, finite reference value, zero-variance fallback, and finite EI outputs.
  - `acquisition_random` now validates pool/query bounds and returns deterministic `default_rng` selections.
  - `ActiveLearningLoop` now validates strategy and budget parameters up front, clamps initial sampling to dataset size, and handles empty datasets gracefully.
  - `_predict_with_dropout` now validates `n_samples`, uses numerically stable std (`unbiased=False`), sanitizes outputs, and restores model/dropout training states after MC inference.
  - `pareto_frontier` now validates 2D objective shape, checks `maximize` length, filters non-finite rows, and maps Pareto indices back to original rows.
  - `multi_objective_screening` now validates non-empty objectives/positive `top_k`, handles empty prediction matrices, and uses zero-span-safe normalization.
- `structure_enumerator.py`:

  - added type checks for `base_structure` and substitution payload schema.
  - added `max_index > 0` guard and normalized substitution parsing helper.
  - `remove_incomplete=True` now actively filters DummySpecies-containing structures.
  - improved combinatorial-cap logging and implemented `_build_constraints` to expose variant-space diagnostics.
- Tests:

  - added `test_phase6_active_learning.py` for acquisition finite-safety, Pareto filtering/index mapping, and AL loop init validation.
  - added `test_structure_enumerator_script.py` for substitution schema validation, incomplete-structure filtering, and constraint-space diagnostics.
  - Verified with:

    - `python -m ruff check scripts/phase5_active_learning/active_learning.py scripts/phase5_active_learning/structure_enumerator.py tests/unit/active_learning/test_phase6_active_learning.py tests/unit/active_learning/test_structure_enumerator_script.py`
    - `python -m pytest tests/unit/active_learning/test_phase6_active_learning.py tests/unit/active_learning/test_structure_enumerator_script.py -q`

## Research references used in batch 22

- Jones, Schonlau, Welch (1998), Efficient Global Optimization / EI: https://doi.org/10.1023/A:1008306431147
- Deb et al. (2002), NSGA-II non-dominated sorting baseline: https://doi.org/10.1109/4235.996017
- SciPy `scipy.stats.norm` API docs (EI CDF/PDF terms): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
- pymatgen `StructureMatcher` docs (`group_structures` symmetry-equivalence dedup): https://pymatgen.org/pymatgen.analysis.structure_matcher.html
- NumPy `nan_to_num` docs (finite-sanitization behavior): https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html

## Batch 23 (max 5 files)

- [X] `scripts/phase5_active_learning/search_materials.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/test_enumeration.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_search_materials_cli.py` - added + optimized
- [X] `tests/unit/active_learning/test_test_enumeration_script.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 23 optimization goals

- Make `search_materials` CLI fail fast on malformed inputs before heavy dataset loading.
- Improve robustness of custom filter parsing and criteria composition (bound merging, finite validation).
- Refactor enumeration demo script into reusable/testable helpers instead of print-only one-off flow.
- Add script-level unit tests for both search and enumeration demos to keep these user-facing entrypoints stable under CI.

## Batch 23 outcomes

- `search_materials.py`:

  - added strict argument validation (`_validate_args`) for preset conflicts, finite numeric constraints, range consistency, and EM bounds.
  - added robust custom-filter parser (`<`, `<=`, `>`, `>=`) with explicit syntax/finite checks.
  - extracted criteria assembly into `_build_criteria` with bound merging logic for repeated filters.
  - extracted deterministic display-column selection helper with deduplication.
  - `main()` now exits with explicit status codes and validates inputs before initializing JARVIS clients.
- `test_enumeration.py`:

  - converted to reusable demo utilities (`_resolve_enumerator_class`, `run_demo`) plus CLI args (`--skip-vacancy`, `--quiet`).
  - added structured summary return payload to support automation and regression tests.
  - switched to explicit `SystemExit(main())` convention for predictable script exit handling.
- Tests:

  - added `test_search_materials_cli.py` to cover argument guardrails, custom filter parsing, criteria merging, and display-column dedup behavior.
  - added `test_test_enumeration_script.py` to cover demo helper execution with injected dummy enumerator and vacancy branch behavior.
  - Verified with:

    - `python -m ruff check scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/test_enumeration.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_test_enumeration_script.py`
    - `python -m pytest tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_test_enumeration_script.py -q`

## Research references used in batch 23

- Python `argparse` docs (CLI validation strategy): https://docs.python.org/3/library/argparse.html
- NumPy `nan_to_num` docs (finite sanitization): https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
- pandas display/options docs (deterministic tabular rendering): https://pandas.pydata.org/docs/user_guide/options.html
- pymatgen `StructureMatcher` docs (symmetry-aware dedup): https://pymatgen.org/pymatgen.analysis.structure_matcher.html
- Hart & Forcade (2008), derivative structure enumeration foundations: https://doi.org/10.1107/S0108767308027336

## Batch 24 (max 5 files)

- [X] `setup.py` - reviewed + optimized
- [X] `pyproject.toml` - reviewed + optimized
- [X] `requirements-dev.txt` - reviewed + optimized
- [X] `tests/unit/config/test_packaging_metadata.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 24 optimization goals

- Improve packaging metadata completeness and CI/dev environment consistency.
- Align editable dev install flow with a single canonical extra to reduce environment drift.
- Keep legacy `setup.py` compatibility while making behavior explicit and side-effect-safe.
- Add unit-level metadata regression checks so packaging drift is caught in CI.

## Batch 24 outcomes

- `setup.py`:

  - converted shim to explicit `if __name__ == "__main__": setup()` guard to avoid accidental side effects on import while keeping legacy tooling compatibility.
- `pyproject.toml`:

  - added richer project metadata (`classifiers`, `project.urls`) for package index discoverability and downstream tooling compatibility.
  - added `dev` optional dependency group consolidating test/notebook/lint/build tooling.
  - added `tool.setuptools.include-package-data = false` for explicit packaging behavior.
- `requirements-dev.txt`:

  - switched editable install target from `.[test,jupyter]` to `.[dev]` for one canonical development environment entrypoint.
- Tests:

  - added `test_packaging_metadata.py` to validate:

    - `dev` extra presence and key tooling dependencies,
    - `requirements-dev` points to `-e .[dev]`,
    - `setup.py` remains a pyproject-backed guarded shim.
  - Verified with:

    - `python -m ruff check setup.py tests/unit/config/test_packaging_metadata.py`
    - `python -m pytest tests/unit/config/test_packaging_metadata.py tests/unit/config/test_config.py -q`

## Research references used in batch 24

- PEP 621 (canonical `pyproject.toml` project metadata): https://peps.python.org/pep-0621/
- Setuptools `pyproject.toml` configuration guide: https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html
- Setuptools dependency management (`dependencies` / `optional-dependencies`): https://setuptools.pypa.io/en/stable/userguide/dependency_management.html
- pip local/editable installs guide: https://pip.pypa.io/en/stable/topics/local-project-installs/
- PyPA Packaging User Guide overview: https://packaging.python.org/en/latest/

## Batch 25 (max 5 files)

- [X] `atlas/training/preflight.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_phase5.py` - reviewed + optimized
- [X] `tests/unit/training/test_preflight.py` - added + optimized
- [X] `tests/unit/active_learning/test_phase5_cli.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 25 optimization goals

- Harden preflight gates to fail fast on invalid input and avoid indefinite subprocess hangs.
- Improve preflight diagnostics with explicit failure reason propagation.
- Expose preflight timeout control at Phase5 CLI layer and validate it before execution.
- Add dedicated unit tests for preflight orchestration edge cases (timeout/failure/missing artifacts/success path).

## Batch 25 outcomes

- `training/preflight.py`:

  - added strict input validation for `project_root`, `property_group`, `max_samples`, `split_seed`, and `timeout_sec`.
  - added bounded subprocess execution helper (`_run_command`) with timeout handling (`TimeoutExpired -> return code 124`).
  - centralized required split-manifest filenames in `_REQUIRED_SPLIT_MANIFESTS`.
  - expanded `PreflightResult` with `error_message` for clearer upstream diagnostics.
  - all failure exits now return structured error categories (`validate-data failed`, `make-splits failed`, `missing split manifests`).
- `run_phase5.py`:

  - added `--preflight-timeout-sec` argument (default `1800`) and validation guard (`>0`).
  - wired `timeout_sec` through to `run_preflight(...)`.
- Tests:

  - new `test_preflight.py` covers:

    - dry-run path creation,
    - argument validation failures,
    - command-timeout handling,
    - missing-manifest failure,
    - full success path with emitted manifests.
  - updated `test_phase5_cli.py` to include `preflight_timeout_sec` fields and added invalid-timeout guard test.
  - Verified with:

    - `python -m ruff check atlas/training/preflight.py scripts/phase5_active_learning/run_phase5.py tests/unit/active_learning/test_phase5_cli.py tests/unit/training/test_preflight.py`
    - `python -m pytest tests/unit/training/test_preflight.py tests/unit/active_learning/test_phase5_cli.py -q`

## Research references used in batch 25

- Python `subprocess.run` timeout semantics (official docs): https://docs.python.org/3/library/subprocess.html
- Python `argparse` docs (CLI argument validation patterns): https://docs.python.org/3/library/argparse.html
- The Tail at Scale (Dean & Barroso, 2013) DOI: https://doi.org/10.1145/2408776.2408794
- PEP 324 (`subprocess` design rationale): https://peps.python.org/pep-0324/
- Google Research publication page for Tail at Scale: https://research.google/pubs/the-tail-at-scale/

## Batch 26 (max 5 files)

- [X] `atlas/training/checkpoint.py` - reviewed + optimized
- [X] `atlas/training/filters.py` - reviewed + optimized
- [X] `tests/unit/training/test_checkpoint.py` - added + optimized
- [X] `tests/unit/training/test_filters.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 26 optimization goals

- Harden checkpoint retention logic against malformed hyperparameters and numeric edge cases (`NaN` metrics, negative epochs).
- Make rotating checkpoint behavior correct for small `keep_last_k` settings (especially `keep_last_k=1`).
- Improve outlier filtering robustness under non-finite/missing values and provide robust-statistics option.
- Add dedicated training utility tests for checkpointing and outlier filtering to lock behavior in CI.

## Batch 26 outcomes

- `training/checkpoint.py`:

  - added constructor validation for `top_k` and `keep_last_k` (>0).
  - `save_best` now validates finite `mae` and non-negative `epoch`.
  - added early reject path when candidate cannot enter top-k, avoiding unnecessary checkpoint writes.
  - stabilized best-model ranking with deterministic tie-break (`(mae, epoch)`).
  - fixed checkpoint rotation edge case for `keep_last_k=1` (no invalid `checkpoint_prev_0.pt` path).
  - ensured checkpoint payload always contains `epoch` when absent in state.
- `training/filters.py`:

  - added argument guards for finite positive `n_sigma` and supported methods.
  - added robust scalar extraction that skips non-scalar, non-finite, or malformed property values.
  - added `method` option: `"zscore"` (default) and `"modified_zscore"` (MAD-based robust alternative).
  - improved per-property iteration with deduplicated property list and explicit outlier metadata (`method`, `scale`).
  - preserved backward-compatible default behavior (`method="zscore"`).
- Tests:

  - new `test_checkpoint.py` covers constructor/input validation, top-k retention, best pointer update, rotation behavior, and `keep_last_k=1`.
  - new `test_filters.py` covers z-score filtering + CSV export, malformed/non-finite value handling, modified-zscore detection, and argument validation.
  - Verified with:

    - `python -m ruff check atlas/training/checkpoint.py atlas/training/filters.py tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py`
    - `python -m pytest tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py -q`

## Research references used in batch 26

- SciPy `median_abs_deviation` docs (robust dispersion for outlier handling): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html
- Iglewicz & Hoaglin (1993), outlier detection using robust z-scores: https://books.google.com/books/about/How_to_Detect_and_Handle_Outliers.html?id=FuuiEAAAQBAJ
- PyTorch `torch.save` docs: https://docs.pytorch.org/docs/stable/generated/torch.save.html
- PyTorch serialization semantics note: https://docs.pytorch.org/docs/stable/notes/serialization.html
- Python `pathlib` docs (path-safe checkpoint handling): https://docs.python.org/3/library/pathlib.html

## Batch 27 (max 5 files)

- [X] `atlas/training/metrics.py` - reviewed + optimized
- [X] `atlas/training/normalizers.py` - reviewed + optimized
- [X] `tests/unit/training/test_metrics.py` - added + optimized
- [X] `tests/unit/training/test_normalizers.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 27 optimization goals

- Make metric utilities numerically stable under NaN/Inf, empty intersections, single-class ROC-AUC edge cases, and degenerate eigenvalue statistics.
- Harden target normalizer fitting/loading paths against empty datasets, malformed state payloads, and non-finite scalar values.
- Add targeted unit tests for scalar/classification/tensor metrics and normalizer state lifecycle so CI catches regressions immediately.

## Batch 27 outcomes

- `training/metrics.py`:

  - validated paired finite input extraction for scalar/classification metrics and retained 0-safe behavior when no valid pairs exist.
  - hardened classification AUC fallback: now explicitly handles single-class labels and non-finite AUC results (`NaN` -> `0.5`).
  - improved tensor eigenvalue agreement stability for constant-spectrum cases by avoiding undefined Spearman correlations and using deterministic fallback scores.
- `training/normalizers.py`:

  - added finite scalar extraction helper for heterogeneous sample payloads (`tensor.item()`, numeric values, malformed objects).
  - fixed empty/invalid dataset path to default `(mean=0, std=1)` instead of propagating `nan` statistics.
  - added `load_state_dict` schema/type/value validation and safe std fallback for invalid/non-positive scales.
  - added explicit missing-property errors in multi-target normalization with available-property hints.
- Tests:

  - added `test_metrics.py` covering non-finite filtering, prefix schema, single-class AUC fallback, tensor finite-row filtering, and eigenvalue-agreement edge handling.
  - added `test_normalizers.py` covering finite-only stat fitting, empty fallback, round-trip normalization, state validation, and multi-target dynamic state loading.
  - Verified with:

    - `python -m ruff check atlas/training/metrics.py atlas/training/normalizers.py tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py`
    - `python -m pytest tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py -q`
    - `python -m pytest tests/unit/training/test_losses.py tests/unit/training/test_trainer.py tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py -q`

## Research references used in batch 27

- scikit-learn `roc_auc_score` docs (single-class undefined behavior and API semantics): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
- scikit-learn classification metrics docs (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`): https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
- SciPy `spearmanr` docs (constant input / undefined correlation caveats): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
- PyTorch `torch.isfinite` docs (finite-mask semantics): https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- PyTorch `torch.linalg.eigvalsh` docs (symmetric/Hermitian eigenspectrum API): https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigvalsh.html
- Fawcett (2006), ROC analysis foundations (Pattern Recognition Letters): https://doi.org/10.1016/j.patrec.2005.10.010

## Batch 28 (max 5 files)

- [X] `atlas/training/run_utils.py` - reviewed + optimized
- [X] `atlas/training/trainer.py` - reviewed + optimized
- [X] `tests/unit/training/test_run_utils_manifest.py` - reviewed + optimized
- [X] `tests/unit/training/test_trainer.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 28 optimization goals

- Harden run directory/manifest utilities against path traversal style run ids, timestamp collisions, and seed metadata overwrite bugs.
- Improve trainer stability by making non-finite losses fail fast and fixing `patience=0` early-stopping semantics.
- Lock the above behavior with regression tests so CI can catch future drift.

## Batch 28 outcomes

- `training/run_utils.py`:

  - added strict run-id normalization/validation (`_normalize_run_name`) to reject path separators, traversal patterns, and unsafe characters.
  - added timestamp collision-resistant run directory creation (`_create_timestamped_run_dir`) with deterministic suffix fallback.
  - added schema-version validation (`non-empty string`).
  - fixed seed precedence bug: default seed inference now uses `setdefault`, so explicit `seeds_block` values are no longer overwritten by fallback defaults.
- `training/trainer.py`:

  - added finite scalar guard helper (`_coerce_finite_float`).
  - `train_epoch` now raises on non-finite batch loss before backward pass.
  - `fit` now validates `val_loss` is finite and correctly handles `patience=0`:

    - continue while improving,
    - stop on first non-improvement.
  - `_save_checkpoint` now validates `epoch >= 0` and finite `val_loss`.
  - `_save_history` now writes JSON with `allow_nan=False` and sanitizes non-finite history values to `null`.
- Tests:

  - `test_run_utils_manifest.py` now covers:

    - explicit seed preservation (no fallback overwrite),
    - path-traversal run-id rejection,
    - timestamp collision suffix behavior.
  - `test_trainer.py` now covers:

    - non-finite training loss fail-fast,
    - non-finite validation loss rejection,
    - checkpoint input validation,
    - `patience=0` improving/non-improving semantics.
  - Verified with:

    - `python -m ruff check atlas/training/run_utils.py atlas/training/trainer.py tests/unit/training/test_run_utils_manifest.py tests/unit/training/test_trainer.py`
    - `python -m pytest tests/unit/training/test_run_utils_manifest.py tests/unit/training/test_trainer.py -q`

## Research references used in batch 28

- OWASP Path Traversal reference (run-id/path safety): https://owasp.org/www-community/attacks/Path_Traversal
- Python `json` docs (`allow_nan` behavior): https://docs.python.org/3/library/json.html
- Python `pathlib` docs (`Path.name` semantics): https://docs.python.org/3/library/pathlib.html
- PyTorch `torch.isfinite` docs (non-finite value detection): https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- Prechelt (1998), early stopping trade-off analysis: https://doi.org/10.1007/3-540-49430-8_3

## Batch 29 (max 5 files)

- [X] `atlas/utils/reproducibility.py` - reviewed + optimized
- [X] `atlas/utils/structure.py` - reviewed + optimized
- [X] `tests/unit/research/test_reproducibility.py` - reviewed + optimized
- [X] `tests/unit/models/test_structure_utils.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 29 optimization goals

- Strengthen reproducibility controls for seed parsing and deterministic-mode toggling across Python/NumPy/PyTorch call sites.
- Improve structure utility robustness for oxidation-state/disordered compositions and non-finite geometric values.
- Add CI guard tests for newly hardened edge cases.

## Batch 29 outcomes

- `utils/reproducibility.py`:

  - added `_coerce_bool` to robustly parse bool-like inputs (including CLI-style strings such as `"false"`/`"true"`).
  - improved `_coerce_seed` parsing to support base-prefixed strings (e.g. hex seeds like `"0x10"`).
  - added `_enable_torch_determinism` for version-tolerant deterministic configuration (`warn_only` fallback, cuDNN/TF32 settings).
  - expanded metadata payload with `cublas_workspace_config` and `cuda_device_count` for more complete reproducibility auditing.
- `utils/structure.py`:

  - added `_element_symbol_number_pairs` using composition-level element extraction to robustly handle oxidation states and disordered species.
  - added `_finite_float` helper and applied finite guards to volume/density/neighbor-distance aggregation.
  - hardened `_sample_site_indices` for non-positive `max_samples`.
  - preserved existing API and return schema while improving edge-case stability.
- Tests:

  - `test_reproducibility.py` now covers hex/scientific seed parsing, bool-like deterministic flags, and expanded runtime metadata fields.
  - `test_structure_utils.py` now covers oxidation+disorder element parsing and non-positive sampling bounds.
  - Verified with:

    - `python -m ruff check atlas/utils/reproducibility.py atlas/utils/structure.py tests/unit/research/test_reproducibility.py tests/unit/models/test_structure_utils.py`
    - `python -m pytest tests/unit/research/test_reproducibility.py tests/unit/models/test_structure_utils.py -q`

## Research references used in batch 29

- PyTorch reproducibility notes (deterministic behavior and CUDA caveats): https://pytorch.org/docs/stable/notes/randomness.html
- PyTorch `torch.use_deterministic_algorithms` API docs: https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
- NumPy `numpy.random.seed` docs: https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
- Python `random.seed` docs: https://docs.python.org/3/library/random.html#random.seed
- pymatgen core API docs (composition/structure utilities): https://pymatgen.org/pymatgen.core.html
- Spglib method paper (symmetry search context): https://doi.org/10.1038/s41524-018-0081-4

## Batch 30 (max 5 files)

- [X] `atlas/active_learning/objective_space.py` - reviewed + optimized
- [X] `atlas/active_learning/pareto_utils.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_objective_space.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_pareto_utils.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 30 optimization goals

- Harden objective-space preprocessing against ambiguous scalar-like inputs and malformed threshold values.
- Improve Pareto/non-dominated sorting semantics for non-finite rows and reduce hypervolume estimation variance in low-data 3D edge cases.
- Add regression tests for the new edge-case policies so CI guards selection logic stability.

## Batch 30 outcomes

- `active_learning/objective_space.py`:

  - `clip01` now routes through `safe_float` so scalar-like wrappers are handled consistently before clipping.
  - `safe_float` now supports single-value `ndarray`/list/tuple payloads, avoiding accidental default-to-zero on scalar containers.
  - `infer_objective_dimension` now uses `_coerce_obj_dim` for both inferred and fallback dimensions.
  - `feasibility_mask_from_points` now sanitizes thresholds with `safe_float` and enforces strict behavior for `use_joint_synthesis=True` when synthesis dimension is missing (returns infeasible mask).
- `active_learning/pareto_utils.py`:

  - added `_finite_row_mask` helper for consistent finite-row handling.
  - `non_dominated_sort` now sorts finite points first and appends non-finite rows as a terminal front, making ranking semantics explicit and stable.
  - `hypervolume` now uses exact box-volume for the `dim>=3` single-point case instead of Monte Carlo approximation.
  - `mc_hv_improvements_shared` now returns early when no candidate passes feasibility/reference filters, reducing unnecessary sampling work.
- Tests:

  - `test_objective_space.py` adds coverage for:

    - scalar-container clipping,
    - strict joint-synthesis feasibility gate on missing synthesis objective,
    - safe objective-dimension fallback,
    - shortest-length truncation + clipping in term-based objective construction.
  - `test_pareto_utils.py` adds coverage for:

    - all-nonfinite non-dominated sort behavior (single terminal front),
    - exact 3D single-point hypervolume path.
  - Verified with:

    - `python -m ruff check atlas/active_learning/objective_space.py atlas/active_learning/pareto_utils.py tests/unit/active_learning/test_objective_space.py tests/unit/active_learning/test_pareto_utils.py`
    - `python -m pytest tests/unit/active_learning/test_objective_space.py tests/unit/active_learning/test_pareto_utils.py -q`

## Research references used in batch 30

- Deb et al. (2002), NSGA-II fast non-dominated sorting baseline: https://doi.org/10.1109/4235.996017
- Auger et al. (2012), hypervolume indicator complexity foundations: https://doi.org/10.1016/j.tcs.2011.03.012
- NumPy `isfinite` docs (finite-mask semantics): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy broadcasting guide (vectorized dominance/HV operations): https://numpy.org/doc/stable/user/basics.broadcasting.html
- NumPy `nan_to_num` docs (stable non-finite replacement policy): https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html

## Batch 31 (max 5 files)

- [X] `atlas/active_learning/acquisition.py` - reviewed + optimized
- [X] `atlas/active_learning/policy_state.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_acquisition.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_policy_state.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 31 optimization goals

- Improve acquisition scoring stability under non-finite model outputs and malformed scalar hyperparameters (`best_f`, `jitter`, `kappa`).
- Harden policy config/state validation so `None`/unknown strings/non-finite numerics cannot silently corrupt runtime policy behavior.
- Add regression tests for the above edge cases to keep CI behavior explicit.

## Batch 31 outcomes

- `active_learning/acquisition.py`:

  - `_prepare_mean_std` now sanitizes non-finite means to finite neutral values.
  - added scalar sanitizers for `best_f`, `jitter` (non-negative clamp), and `kappa` (non-negative clamp).
  - wired sanitizers into `EI/LogEI/PI/LogPI/NEI/UCB/LCB` paths and unified `score_acquisition` entrypoint.
  - effect: acquisition utilities remain finite and directionally stable even with noisy upstream surrogate outputs or malformed config values.
- `active_learning/policy_state.py`:

  - `_coerce_bool` now falls back to default for unknown strings instead of Python truthiness of non-empty text.
  - added `_coerce_text` for safe/consistent enum-like string parsing.
  - strengthened `ActiveLearningPolicyConfig.validated()` by routing numeric fields through finite coercion before clipping/bounding.
  - strengthened `PolicyState.validated()` similarly to prevent `nan/inf` persistence in state checkpoints.
- Tests:

  - `test_acquisition.py` adds coverage for:

    - negative `kappa` clamp behavior,
    - non-finite `best_f` + negative `jitter` sanitization,
    - non-finite `mean` sanitization on `"mean"` strategy.
  - `test_policy_state.py` adds coverage for:

    - unknown bool-string fallback,
    - config-level non-finite numeric sanitization,
    - direct state-object non-finite sanitization.
  - Verified with:

    - `python -m ruff check atlas/active_learning/acquisition.py atlas/active_learning/policy_state.py tests/unit/active_learning/test_acquisition.py tests/unit/active_learning/test_policy_state.py`
    - `python -m pytest tests/unit/active_learning/test_acquisition.py tests/unit/active_learning/test_policy_state.py -q`

## Research references used in batch 31

- Ament et al. (2023), LogEI numerical stability: https://arxiv.org/abs/2310.20708
- Letham et al. (2019), Noisy Expected Improvement: https://arxiv.org/abs/1706.07094
- Srinivas et al. (2010), GP-UCB exploration schedule foundations: https://arxiv.org/abs/0912.3995
- BoTorch analytic acquisition documentation: https://botorch.readthedocs.io/en/stable/acquisition.html
- PyTorch `torch.nan_to_num` docs: https://docs.pytorch.org/docs/stable/generated/torch.nan_to_num.html
- NumPy `clip` docs (bounded-parameter sanitization): https://numpy.org/doc/stable/reference/generated/numpy.clip.html

## Batch 32 (max 5 files)

- [X] `atlas/active_learning/acquisition.py` - reviewed + optimized
- [X] `atlas/active_learning/policy_state.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_acquisition.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_policy_state.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 32 optimization goals

- Make acquisition scoring robust to malformed runtime scalar knobs and non-finite surrogate outputs.
- Strengthen policy config/state coercion so invalid text/non-finite numeric payloads cannot silently poison resume/state transitions.
- Extend regression tests to lock the new sanitization guarantees in CI.

## Batch 32 outcomes

- `active_learning/acquisition.py`:

  - added reusable scalar sanitizers for `best_f`, `jitter`, and `kappa`.
  - added mean sanitization in `_prepare_mean_std` (`nan/inf -> 0`) so downstream acquisition math stays finite.
  - integrated sanitization into `EI/LogEI/PI/LogPI/NEI/UCB/LCB` and `score_acquisition`.
  - improved `schedule_ucb_kappa` and NEI sampling input coercion with finite-safe defaults.
- `active_learning/policy_state.py`:

  - `_coerce_bool` now falls back to default for unknown strings instead of treating any non-empty string as `True`.
  - added `_coerce_text` for robust enum-like text parsing.
  - routed `ActiveLearningPolicyConfig.validated()` numeric fields through finite-safe coercion before clipping/bounding.
  - routed `PolicyState.validated()` through finite-safe coercion, preventing `nan/inf` persistence in serialized state.
  - `PolicyState.from_dict()` now returns already validated state.
- Tests:

  - `test_acquisition.py` adds:

    - negative-kappa clamp check,
    - non-finite `best_f` + negative `jitter` sanitization check,
    - non-finite mean sanitization check for mean strategy.
  - `test_policy_state.py` adds:

    - unknown bool-string fallback behavior,
    - config-level non-finite numeric sanitization checks,
    - direct state-object non-finite sanitization checks.
  - Verified with:

    - `python -m ruff check atlas/active_learning/acquisition.py atlas/active_learning/policy_state.py tests/unit/active_learning/test_acquisition.py tests/unit/active_learning/test_policy_state.py`
    - `python -m pytest tests/unit/active_learning/test_acquisition.py tests/unit/active_learning/test_policy_state.py -q`

## Research references used in batch 32

- Ament et al. (2023), LogEI stabilization: https://arxiv.org/abs/2310.20708
- Letham et al. (2019), Noisy Expected Improvement: https://arxiv.org/abs/1706.07094
- Srinivas et al. (2010), GP-UCB schedules and regret bounds: https://arxiv.org/abs/0912.3995
- BoTorch acquisition docs (analytic & log variants): https://botorch.readthedocs.io/en/stable/acquisition.html
- PyTorch `torch.nan_to_num` docs: https://docs.pytorch.org/docs/stable/generated/torch.nan_to_num.html
- NumPy `isfinite` docs: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html

## Batch 33 (max 5 files)

- [X] `atlas/active_learning/gp_surrogate.py` - reviewed + optimized
- [X] `atlas/models/uncertainty.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_gp_surrogate.py` - reviewed + optimized
- [X] `tests/unit/models/test_uncertainty.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 33 optimization goals

- Harden GP surrogate config/feature sanitization against malformed bool/float inputs and non-finite candidate rows.
- Improve uncertainty module contract guarantees (`dict[str, Tensor]` payload, consistent task keys) for ensemble/MC paths.
- Stabilize evidential uncertainty/loss under non-finite targets and non-finite coefficient values.
- Add regression tests that lock the above behavior for CI.

## Batch 33 outcomes

- `active_learning/gp_surrogate.py`:

  - restored typed config validation via `GPSurrogateConfig.validated()` and applied at acquirer init.
  - `_coerce_bool` now uses default fallback for unknown strings (no accidental `True` from non-empty text).
  - `_safe_float` now rejects non-finite values (`nan/inf`) and returns defaults.
  - `ei_jitter` now clamps to non-negative range.
  - `suggest_constrained_utility` now:

    - filters non-finite feature rows,
    - predicts only on finite rows,
    - emits finite fallback utilities for invalid rows,
    - avoids full-round failure from a few bad candidates.
- `models/uncertainty.py`:

  - strengthened `_normalize_prediction_payload` to enforce `dict[str, Tensor]` contract and reject empty/non-tensor payloads.
  - added `_validate_prediction_keys` to enforce consistent task keys across ensemble members / MC samples.
  - clamped std outputs to non-negative for ensemble/MC predictions.
  - evidential forward path now sanitizes `aleatoric/epistemic` to finite non-negative tensors.
  - `evidential_loss` now:

    - sanitizes `coeff` via finite-safe parsing,
    - masks non-finite targets/parameters,
    - returns deterministic zero when no finite supervision is available.
- Tests:

  - `test_gp_surrogate.py` adds:

    - unknown bool-string fallback checks,
    - non-finite `_safe_float` checks,
    - non-finite feature-row handling in constrained utility.
  - `test_uncertainty.py` adds:

    - inconsistent task-key regression tests for Ensemble/MC Dropout,
    - non-tensor payload rejection test,
    - all-non-finite-target evidential-loss zero fallback test.
  - Verified with:

    - `python -m ruff check atlas/active_learning/gp_surrogate.py atlas/models/uncertainty.py tests/unit/active_learning/test_gp_surrogate.py tests/unit/models/test_uncertainty.py`
    - `python -m pytest -q tests/unit/active_learning/test_gp_surrogate.py`
    - `python -m pytest -q tests/unit/models/test_uncertainty.py`

## Research references used in batch 33

- NumPy `isfinite` (official API): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy `nan_to_num` (official API): https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
- PyTorch `torch.isfinite` (official API): https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- PyTorch `torch.nan_to_num` (official API): https://docs.pytorch.org/docs/stable/generated/torch.nan_to_num.html
- scikit-learn `GaussianProcessRegressor` (official API): https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
- Amini et al. (NeurIPS 2020), Deep Evidential Regression: https://papers.nips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html
- Lakshminarayanan et al. (NeurIPS 2017), Deep Ensembles: https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles
- Gal and Ghahramani (ICML 2016), MC Dropout Bayesian approximation: https://proceedings.mlr.press/v48/gal16

## Progress snapshot (after Batch 33)

- Completed: Batch 1 through Batch 33.
- Pending: Batch 34 onward (next 5-file chunk to be selected sequentially).

## Batch 34 (max 5 files)

- [X] `atlas/benchmark/cli.py` - reviewed + optimized
- [X] `atlas/console_style.py` - reviewed + optimized
- [X] `tests/unit/benchmark/test_benchmark_cli.py` - added + optimized
- [X] `tests/unit/test_console_style.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 34 optimization goals

- Harden benchmark CLI argument validation so invalid numeric/range/path payloads fail fast before heavy runtime.
- Improve model-kwargs decoding diagnostics and fold handling determinism.
- Refine console color support logic for standard env flags and edge cases (`sep=None`, `file=None`).
- Add targeted unit tests for the new validation/styling behavior.

## Batch 34 outcomes

- `benchmark/cli.py`:

  - added `_parse_model_kwargs` with explicit JSON decoding error messages.
  - added `_validate_cli_args` to enforce:

    - task key membership,
    - checkpoint path existence/type,
    - positive/non-negative numeric bounds,
    - coverage domain bounds,
    - fold non-negativity plus deduplicate/sort normalization.
  - validation now runs in `main()` for non-`--list-tasks` code paths.
- `console_style.py`:

  - added env helpers for truthy/falsy color flags.
  - `_supports_color` now supports `FORCE_COLOR`, `CLICOLOR_FORCE`, `CLICOLOR`, `TERM=dumb`, and `NO_COLOR` precedence.
  - `styled_print` now handles `file=None` and `sep=None` safely.
- Tests:

  - added `test_benchmark_cli.py` for kwargs parsing, range/path validation, fold normalization, and list-task bypass behavior.
  - added `test_console_style.py` for color-env precedence and `sep=None` print path.
  - Verified with:

    - `python -m ruff check atlas/benchmark/cli.py atlas/console_style.py tests/unit/benchmark/test_benchmark_cli.py tests/unit/test_console_style.py`
    - `python -m pytest -q tests/unit/benchmark/test_benchmark_cli.py`
    - `python -m pytest -q tests/unit/test_console_style.py`

## Research references used in batch 34

- Python `argparse` docs (official): https://docs.python.org/3/library/argparse.html
- Python `importlib` docs (official): https://docs.python.org/3/library/importlib.html
- Python `json` docs (official): https://docs.python.org/3/library/json.html
- Python `pathlib` docs (official): https://docs.python.org/3/library/pathlib.html
- Python `print()` docs (`sep`/`file` semantics): https://docs.python.org/3/library/functions.html#print
- PyTorch `torch.load` docs: https://pytorch.org/docs/stable/generated/torch.load.html
- PyTorch serialization notes: https://pytorch.org/docs/stable/notes/serialization.html
- `NO_COLOR` community standard: https://no-color.org/
- pytest monkeypatch docs (official): https://docs.pytest.org/en/stable/how-to/monkeypatch.html

## Progress snapshot (after Batch 34)

- Completed: Batch 1 through Batch 34.
- Pending: Batch 35 onward (next 5-file chunk to be selected sequentially).

## Batch 35 (max 5 files)

- [X] `atlas/__init__.py` - reviewed + optimized
- [X] `atlas/active_learning/__init__.py` - reviewed + optimized
- [X] `atlas/benchmark/__init__.py` - reviewed + optimized
- [X] `atlas/data/__init__.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 35 optimization goals

- Standardize package-level public API exposure via explicit lazy-export maps.
- Reduce import-time coupling by avoiding eager heavy submodule imports in package `__init__`.
- Improve package introspection stability (`__all__`, `__dir__`) and preserve backward-compatible attribute access.
- Validate that benchmark/data package users continue to work under lazy export paths.

## Batch 35 outcomes

- `atlas/__init__.py`:

  - replaced eager `Config/get_config` import with module-level lazy export map.
  - added PEP-562-style `__getattr__` and deterministic `__dir__`.
  - kept public API contract (`__version__`, `Config`, `get_config`) unchanged.
- `atlas/active_learning/__init__.py`:

  - normalized `__all__` to immutable tuple.
  - tightened `__getattr__` signature and `__dir__` implementation using set-union to avoid duplicate names.
  - preserved all existing exports and lazy-load semantics.
- `atlas/benchmark/__init__.py`:

  - replaced eager `from .runner import ...` with lazy export map to reduce import-time overhead.
  - added package-level `__getattr__` / `__dir__` for stable dynamic resolution and introspection.
  - preserved exported benchmark symbols and compatibility for `from atlas.benchmark import ...`.
- `atlas/data/__init__.py`:

  - normalized `__all__` tuple and refined `__dir__` to stable deduplicated output.
  - kept existing lazy import behavior with explicit typed `__getattr__`.
  - maintained all previous public exported names.
- Verification:

  - `python -m ruff check atlas/__init__.py atlas/active_learning/__init__.py atlas/benchmark/__init__.py atlas/data/__init__.py`
  - `python -m pytest -q tests/unit/benchmark/test_benchmark_runner.py tests/unit/data/test_crystal_dataset.py`
  - import smoke test: `import atlas`, `from atlas.benchmark import MatbenchRunner`, `from atlas.data import DataSourceRegistry`, etc.

## Research references used in batch 35

- PEP 562: Module `__getattr__` and `__dir__`: https://peps.python.org/pep-0562/
- Python data model (module customization): https://docs.python.org/3/reference/datamodel.html
- Python `importlib` docs (`import_module`): https://docs.python.org/3/library/importlib.html
- Python language reference (`__all__` and import semantics): https://docs.python.org/3/reference/simple_stmts.html
- Python tutorial modules/packages (`__all__` in packages): https://docs.python.org/3/tutorial/modules.html
- PEP 8 public/internal interface guidance (`__all__`): https://peps.python.org/pep-0008/
- `pkgutil` docs (package import side effects context): https://docs.python.org/3/library/pkgutil.html

## Progress snapshot (after Batch 35)

- Completed: Batch 1 through Batch 35.
- Pending: Batch 36 onward.

## Batch 36 (max 5 files)

- [X] `atlas/discovery/alchemy/__init__.py` - reviewed + optimized
- [X] `atlas/discovery/alchemy/calculator.py` - reviewed + optimized
- [X] `atlas/discovery/alchemy/model.py` - reviewed + optimized
- [X] `atlas/discovery/alchemy/optimizer.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 36 optimization goals

- Reduce import-time fragility for optional alchemical dependencies (MACE/e3nn path).
- Improve calculator/runtime robustness for device handling, weight sanitization, and non-finite model outputs.
- Add stricter alchemical pair/weight validation in graph construction to fail fast on malformed inputs.
- Upgrade composition projection from ad-hoc normalization to mathematically grounded simplex projection.

## Batch 36 outcomes

- `discovery/alchemy/__init__.py`:

  - replaced eager import block with explicit lazy export map (`__getattr__`, `__dir__`).
  - optional dependency failures now produce clear lazy-time diagnostics instead of hard module import failure.
  - preserved fallback behavior for missing calculator dependency via on-demand unavailable-calculator factory.
- `discovery/alchemy/calculator.py`:

  - removed dead `contextlib` import path.
  - added device normalization (`cpu/cuda/cuda:*`) with CUDA-unavailable fallback to CPU.
  - added alchemical weight validation (`size`, `finite`, clipping to `[0, 1]`) for init and updates.
  - replaced `.data` assignment with `torch.no_grad()+copy_` to avoid unsafe parameter mutation patterns.
  - hardened `calculate` path with:

    - stable grad reset in `finally`,
    - non-finite energy detection,
    - explicit failure when model outputs are missing.
- `discovery/alchemy/model.py`:

  - added pair parsing helper with explicit index/atomic-number validation.
  - enforced `alchemical_weights` length match to group count and finite-value requirement.
  - improved fixed-atom lookup complexity by using set membership for non-alchemical index extraction.
- `discovery/alchemy/optimizer.py`:

  - removed import-time hard dependency on `mace` path by using `TYPE_CHECKING` for calculator typing.
  - validated optimizer hyperparameter (`learning_rate` finite positive).
  - added exact simplex projection helper (`_project_to_simplex`) and applied per constrained atom group.
  - hardened optimization step for missing atoms, gradient shape mismatch, and non-finite gradients.
  - normalized run-step validation (`steps >= 0`).
- Verification:

  - `python -m ruff check atlas/discovery/alchemy/__init__.py atlas/discovery/alchemy/calculator.py atlas/discovery/alchemy/model.py atlas/discovery/alchemy/optimizer.py`
  - alchemy import/projection smoke script covering:

    - lazy export availability,
    - simplex projection invariants (`sum=1`, bounded, finite) under dirty input vector.

## Research references used in batch 36

- PEP 562 (module `__getattr__`/`__dir__` lazy access): https://peps.python.org/pep-0562/
- ASE calculators documentation (calculator contract and properties): https://ase.gitlab.io/ase/ase/calculators/calculators.html
- PyTorch `torch.autograd.grad` API: https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html
- NumPy `nan_to_num` API: https://numpy.org/doc/2.1/reference/generated/numpy.nan_to_num.html
- NumPy `clip` API: https://numpy.org/doc/stable/reference/generated/numpy.clip.html
- Duchi et al. (ICML 2008), efficient projection algorithm basis: https://web.stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
- MACE repository (architecture/API context for alchemical integration): https://github.com/ACEsuit/mace
- alchemical-mlip repository (upstream alchemical formulation context): https://github.com/learningmatter-mit/alchemical-mlip

## Progress snapshot (after Batch 36)

- Completed: Batch 1 through Batch 36.
- Pending: Batch 37 onward.

## Batch 37 (max 5 files)

- [X] `atlas/discovery/stability/__init__.py` - reviewed + optimized
- [X] `atlas/discovery/stability/mepin.py` - reviewed + optimized
- [X] `atlas/discovery/transport/liflow.py` - reviewed + optimized
- [X] `atlas/explain/__init__.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 37 optimization goals

- Make stability/explain package entrypoints consistent with lazy-export architecture and optional-dependency behavior.
- Harden MEPIN and LiFlow wrappers against malformed runtime inputs (device, checkpoints, sizes, non-finite outputs).
- Reduce import-time failures from optional third-party repos by delaying heavy imports and improving diagnostics.
- Keep API compatibility while adding stricter invariants for trajectory shape and composition parameters.

## Batch 37 outcomes

- `discovery/stability/__init__.py`:

  - implemented explicit lazy export map with `__getattr__` and deterministic `__dir__`.
  - preserved public symbol surface (`MEPINStabilityEvaluator`) while avoiding eager heavy import.
- `discovery/stability/mepin.py`:

  - switched path resolution to `pathlib` and guarded repo-path insertion.
  - added device normalization (`cpu/cuda` with CUDA-unavailable fallback).
  - added checkpoint resolver with supported model-type validation.
  - added trajectory input/output guards:

    - `num_images >= 2`,
    - reactant/product atom-count match,
    - model output tensor finite check + exact shape/size validation.
  - standardized `__all__` and error messaging for optional backend failures.
- `discovery/transport/liflow.py`:

  - switched to `pathlib` path resolution and guarded repo-path insertion.
  - added device normalization and temperature-list validation.
  - added checkpoint resolver and element-index loading validation.
  - hardened simulation pipeline:

    - validates `atoms`, `steps`, `flow_steps`,
    - validates each frame shape and finite coordinates,
    - stabilizes diffusion estimate with non-negative finite output.
- `explain/__init__.py`:

  - converted eager imports to lazy export map with robust `ImportError` handling for optional latent-analysis deps.
  - preserved expected behavior: `LatentSpaceAnalyzer` resolves to `None` when optional stack is unavailable.
- Verification:

  - `python -m ruff check atlas/discovery/stability/__init__.py atlas/discovery/stability/mepin.py atlas/discovery/transport/liflow.py atlas/explain/__init__.py`
  - `python -m py_compile atlas/discovery/stability/__init__.py atlas/discovery/stability/mepin.py atlas/discovery/transport/liflow.py atlas/explain/__init__.py`
  - import smoke script for `atlas.discovery.stability`, `atlas.explain`, and helper/device normalization checks.

## Research references used in batch 37

- PEP 562 (module lazy attribute access): https://peps.python.org/pep-0562/
- Python import system reference (`sys.path` semantics): https://docs.python.org/3/reference/import.html
- Python `pathlib` docs: https://docs.python.org/3/library/pathlib.html
- Python logging HOWTO: https://docs.python.org/3/howto/logging.html
- PyTorch `torch.no_grad` docs: https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html
- PyTorch `torch.isfinite` docs: https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- NumPy `nan_to_num` docs: https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
- NumPy broadcasting/user guide (shape-safe vectorized ops context): https://numpy.org/doc/stable/user/basics.broadcasting.html
- ASE Atoms API reference: https://ase-lib.org/ase/atoms.html
- ASE calculators documentation: https://ase.gitlab.io/ase/ase/calculators/calculators.html
- Lipman et al. (ICLR 2023), Flow Matching for Generative Modeling: https://openreview.net/forum?id=PqvMRDCJT9t

## Progress snapshot (after Batch 37)

- Completed: Batch 1 through Batch 37.
- Pending: Batch 38 onward.

## Batch 38 (max 5 files)

- [X] `atlas/explain/gnn_explainer.py` - reviewed + optimized
- [X] `atlas/explain/latent_analysis.py` - reviewed + optimized
- [X] `atlas/models/__init__.py` - reviewed + optimized
- [X] `atlas/models/equivariant.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 38 optimization goals

- Harden explainability utilities against missing optional graph attributes and non-finite explanation outputs.
- Improve latent-space analysis robustness for empty loaders, method parameter edge-cases, and clustering dimensional constraints.
- Convert `atlas.models` package exports to lazy-loading architecture to reduce import-time dependency coupling.
- Stabilize equivariant model inputs (species indexing, radial basis cutoff, shape checks) and remove hard-coded species assumptions.

## Batch 38 outcomes

- `explain/gnn_explainer.py`:

  - added finite-safe node/edge importance normalization with support for attribute-level node masks.
  - improved atomic-number extraction fallback (`z` or `x`) and robust bond symbol derivation.
  - made explainer call resilient to missing `edge_attr`.
  - added strict size checks for structure plotting (`node_importance` length must match atom count).
  - hardened atom radius handling (`atomic_radius` fallback).
- `explain/latent_analysis.py`:

  - added device normalization with CUDA fallback warnings.
  - added robust embedding extraction from tensor or dict outputs (`embedding`/`latent`/`mean` keys).
  - now sanitizes non-finite embeddings/properties and fails fast on empty loaders.
  - dimensional reduction now validates sample/component counts and constrains t-SNE perplexity to valid domain.
  - clustering now validates `n_clusters` vs sample count and method domain.
  - plotting and cluster-analysis paths now validate shape/length consistency.
- `models/__init__.py`:

  - replaced eager imports with explicit lazy export map (`__getattr__`, `__dir__`), preserving public API names.
- `models/equivariant.py`:

  - added `_infer_species_indices` to support multiple node feature layouts and remove hard-coded `:86` slicing.
  - validated model/radial constructor parameters (`n_layers`, `n_species`, `max_radius`, etc.).
  - radial basis now zeroes outside cutoff and sanitizes non-finite distances.
  - strengthened `encode` with shape checks for `edge_index`/`edge_vectors` and safer batch coercion.
  - made output-head hidden width robust when scalar channel count is small.
- Verification:

  - `python -m ruff check atlas/explain/gnn_explainer.py atlas/explain/latent_analysis.py atlas/models/__init__.py atlas/models/equivariant.py`
  - `python -m py_compile atlas/explain/gnn_explainer.py atlas/explain/latent_analysis.py atlas/models/__init__.py atlas/models/equivariant.py`
  - `python -m pytest -q tests/unit/models/test_model_utils.py tests/unit/models/test_prediction_utils.py`
  - smoke script for:

    - lazy `atlas.models` exports,
    - species-index inference helper,
    - radial basis finite output checks.

## Research references used in batch 38

- PyTorch Geometric explainability API docs: https://pytorch-geometric.readthedocs.io/en/latest/modules/explain.html
- GNNExplainer paper (NeurIPS 2019): https://papers.neurips.cc/paper/9123-gnnexplainer-generating-explanations-for-graph-neural-networks
- scikit-learn TSNE docs (perplexity constraint): https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
- scikit-learn PCA docs: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- scikit-learn KMeans docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- scikit-learn DBSCAN docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
- UMAP API docs: https://umap-learn.readthedocs.io/en/latest/api.html
- PEP 562 (module-level lazy attribute access): https://peps.python.org/pep-0562/
- NequIP paper (Nature Communications 2022): https://www.nature.com/articles/s41467-022-29939-5
- e3nn documentation: https://docs.e3nn.org/
- DimeNet paper (smooth radial/cutoff basis context): https://arxiv.org/abs/2003.03123
- NumPy `nan_to_num` docs: https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html

## Progress snapshot (after Batch 38)

- Completed: Batch 1 through Batch 38.
- Pending: Batch 39 onward.

## Batch 39 (max 5 files)

- [X] `atlas/models/fast_tp.py` - reviewed + optimized
- [X] `atlas/models/layers.py` - reviewed + optimized
- [X] `atlas/models/m3gnet.py` - reviewed + optimized
- [X] `atlas/models/matgl_three_body.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 39 optimization goals

- Harden fused tensor-product scatter path against malformed edge tensors and invalid indexing.
- Tighten message-passing/equivariant layer input contracts and remove accidental extra nonlinearity in update path.
- Improve M3GNet numerical stability and shape robustness (RBF edge cases, 3-body indexing, species indexing, aggregation semantics).
- Validate three-body basis helpers (`max_n/max_l/n_basis`, shape/finite checks) for safer low-data and noisy-input regimes.

## Batch 39 outcomes

- `models/fast_tp.py`:

  - added strict shape/range checks for `edge_src/edge_dst/edge_attr/edge_weight`.
  - validated `edge_weight` second dimension against `weight_numel`.
  - added empty-edge fast path and non-finite message sanitization.
- `models/layers.py`:

  - added constructor validation (`node_dim`, `edge_dim`, `n_radial_basis`, `max_radius`).
  - added forward shape/range checks and empty-edge guard in both layers.
  - removed duplicated SiLU in `MessagePassingLayer` update path (MLP already applies SiLU), reducing over-smoothing risk.
  - added per-node degree normalization in M3GNet-style aggregation path where appropriate.
- `models/matgl_three_body.py`:

  - added parameter validation for basis sizes.
  - added shape consistency checks for `(r_ij, r_ik, cos_theta)`.
  - added non-finite sanitization (`nan_to_num`) for robust basis outputs.
- `models/m3gnet.py`:

  - `RBFExpansion` now handles `n_gaussians=1` and invalid cutoff safely.
  - hardened `ThreeBodyInteraction` basis setup (`max_l>=1`) and finite-safe feature composition.
  - fixed ambiguous/double node aggregation behavior by using a single `dst` aggregation with degree normalization.
  - added comprehensive shape checks for `edge_index`, `edge_attr`, `edge_vectors`, `edge_index_3body`.
  - removed hard-coded `:86` species assumption; now robustly infers species indices and clamps into `[0, n_species-1]`.
  - added edge feature dimension adaptation (truncate/pad) before edge embedding.
- Verification:

  - `python -m ruff check atlas/models/fast_tp.py atlas/models/layers.py atlas/models/m3gnet.py atlas/models/matgl_three_body.py`
  - `python -m py_compile atlas/models/fast_tp.py atlas/models/layers.py atlas/models/m3gnet.py atlas/models/matgl_three_body.py`
  - smoke checks for:

    - `RBFExpansion` finite output on `inf` distances,
    - `MessagePassingLayer` finite output on random graph,
    - `SphericalBesselHarmonicsExpansion` output shape,
    - `M3GNet.encode` finite embedding generation.

## Research references used in batch 39

- M3GNet paper (Nature Computational Science 2022): https://www.nature.com/articles/s43588-022-00349-3
- M3GNet arXiv preprint: https://arxiv.org/abs/2202.02450
- PyTorch `index_add` docs: https://docs.pytorch.org/docs/stable/generated/torch.index_add.html
- PyTorch Geometric explainability docs (API stability context): https://pytorch-geometric.readthedocs.io/en/latest/modules/explain.html
- Duchi et al. (ICML 2008), projection algorithms basis: https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
- e3nn documentation: https://docs.e3nn.org/
- DimeNet paper (radial/angular basis context): https://arxiv.org/abs/2003.03123
- NumPy `nan_to_num` docs: https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html

## Progress snapshot (after Batch 39)

- Completed: Batch 1 through Batch 39.
- Pending: Batch 40 onward.

## Batch 40 (max 5 files)

- [X] `atlas/ops/cpp_ops.py` - reviewed + optimized
- [X] `atlas/potentials/__init__.py` - reviewed + optimized
- [X] `atlas/potentials/mace_relaxer.py` - reviewed + optimized
- [X] `atlas/potentials/relaxers/mlip_arena_relaxer.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 40 optimization goals

- Stabilize native/C++ radius-graph path under mixed environments (no compiler, no Ninja, CUDA/CPU differences) while preserving deterministic fallback behavior.
- Reduce package import-time coupling in `atlas.potentials` and align with prior lazy-export architecture used across ATLAS packages.
- Harden MACE relaxation runtime against invalid devices, invalid optimization hyperparameters, unsupported cell filters, and non-finite energy/force outputs.
- Replace brittle MLIP-Arena wrapper imports with an ASE-backed robust implementation preserving registry compatibility.

## Batch 40 outcomes

- `ops/cpp_ops.py`:

  - rebuilt C++ kernel to enforce `max_num_neighbors` per source atom via partial sort.
  - removed hard dependency on `torch_cluster` fallback and replaced with pure-torch fallback using `torch.cdist` + per-node top-k pruning.
  - added strict input validation (`pos/batch` shapes, dtype/device consistency, finite `r_max`, positive neighbor budget).
  - added safe empty-output path, one-time PBC warning, and compile/runtime fallback logging.
  - made C++ invocation robust by CPU/contiguous casting and automatic return to caller device/dtype.
- `potentials/__init__.py`:

  - switched to explicit lazy-export map (`MACERelaxer`, `NativeMlipArenaRelaxer`) with deterministic `__dir__` and PEP-562-compatible `__getattr__`.
- `potentials/mace_relaxer.py`:

  - added strong constructor validation (`model_size`, `default_dtype`, normalized device policy with CUDA fallback handling).
  - hardened calculator bootstrap for missing/invalid custom model path and foundation-model fallback.
  - added relaxation input guards (`fmax`, `steps`, empty structure handling, trajectory path creation).
  - standardized cell-filter resolution with validated domain (`frechet|exp|unit|fixed|None`) and fallback behavior.
  - added finite checks for output energy/forces and safer `volume_change` computation when initial volume is invalid.
  - made batch mode explicit about serial execution semantics for GPU calculators.
- `potentials/relaxers/mlip_arena_relaxer.py`:

  - replaced fragile third-party imports with stable ASE-native optimizer/filter mapping while preserving registry key (`mlip_arena_native`).
  - added parameter schema validation (`fmax`, `steps`, `optimizer`, `cell_filter`).
  - implemented robust relax path with optional symmetry constraints, cell-filter selection, non-finite energy rejection, and standardized failure payload.
- Verification:

  - `python -m ruff check atlas/ops/cpp_ops.py atlas/potentials/__init__.py atlas/potentials/mace_relaxer.py atlas/potentials/relaxers/mlip_arena_relaxer.py`
  - `python -m py_compile atlas/ops/cpp_ops.py atlas/potentials/__init__.py atlas/potentials/mace_relaxer.py atlas/potentials/relaxers/mlip_arena_relaxer.py`
  - smoke script covering:

    - `fast_radius_graph` finite output and shape contract,
    - lazy import of `MACERelaxer` / `NativeMlipArenaRelaxer`,
    - constructor/runtime failure-path checks for `NativeMlipArenaRelaxer`.

## Research references used in batch 40

- PyTorch C++ extension docs (`load_inline`/extension build): https://pytorch.org/docs/stable/cpp_extension.html
- PyTorch `torch.cdist` API: https://pytorch.org/docs/stable/generated/torch.cdist.html
- PyTorch `torch.topk` API: https://pytorch.org/docs/stable/generated/torch.topk.html
- ASE optimizers documentation: https://ase.gitlab.io/ase/ase/optimize.html
- ASE constraints documentation: https://ase.gitlab.io/ase/ase/constraints.html
- ASE filters documentation (`FrechetCellFilter`, cell filters): https://ase.gitlab.io/ase/ase/filters.html
- MACE repository / calculator integration context: https://github.com/ACEsuit/mace
- MACE paper (ICLR 2022 Workshop): https://arxiv.org/abs/2206.07697
- PEP 562 (`__getattr__`/`__dir__` on modules): https://peps.python.org/pep-0562/
- Python `importlib` docs: https://docs.python.org/3/library/importlib.html
- Python warnings docs: https://docs.python.org/3/library/warnings.html

## Progress snapshot (after Batch 40)

- Completed: Batch 1 through Batch 40.
- Pending: Batch 41 onward.

## Batch 41 (max 5 files)

- [X] `atlas/research/__init__.py` - reviewed + optimized
- [X] `atlas/research/method_registry.py` - reviewed + optimized
- [X] `atlas/research/workflow_reproducible_graph.py` - reviewed + optimized
- [X] `atlas/thermo/__init__.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 41 optimization goals

- Standardize research/thermo package entrypoints with lazy-export architecture to reduce import-time coupling and optional-dependency breakage.
- Harden methodology registry contracts (immutability, schema validation, duplicate-key governance) for reproducible experiment selection.
- Upgrade reproducibility workflow manifest path with stronger data validation, run-id safety, and atomic persistence semantics.
- Back new behavior with focused unit tests so regressions are caught in CI.

## Batch 41 outcomes

- `research/__init__.py`:

  - replaced eager imports with explicit lazy export map (`__getattr__`, `__dir__`).
  - preserved public API while reducing side effects from importing heavy research modules.
- `research/method_registry.py`:

  - upgraded `MethodSpec` to immutable tuple-based `strengths/tradeoffs` with normalization/validation (`__post_init__`).
  - added registry duplicate-key guard (`replace=False` default) and type checks on `register`.
  - maintained backward-compatible helper API (`get_method`, `list_methods`, `recommended_method_order`).
- `research/workflow_reproducible_graph.py`:

  - added strict validators for iteration counters/timings and manifest stage-plan hygiene.
  - sanitized run-id tokens to prevent unsafe path characters in artifact filenames.
  - removed repeated `type: ignore` union access via internal `_ensure_manifest()` helper.
  - switched manifest writes to atomic temp-file replacement to avoid partial/corrupted JSON on interrupted writes.
  - added `schema_version` field for forward-compatible artifact parsing.
- `thermo/__init__.py`:

  - replaced eager optional-import block with lazy optional exports.
  - optional missing dependencies now resolve to `None` at attribute access time while keeping module import healthy.
- Tests:

  - `tests/unit/research/test_method_registry.py`: added normalization/immutability and duplicate-key rejection cases.
  - `tests/unit/research/test_workflow_reproducible_graph.py`: added non-monotonic iteration rejection and run-id sanitization checks.
  - `tests/unit/thermo/test_init_exports.py` (new): added lazy export contract tests and optional-import-failure behavior.
- Verification:

  - `python -m ruff check atlas/research/__init__.py atlas/research/method_registry.py atlas/research/workflow_reproducible_graph.py atlas/thermo/__init__.py tests/unit/research/test_method_registry.py tests/unit/research/test_workflow_reproducible_graph.py tests/unit/thermo/test_init_exports.py`
  - `python -m py_compile atlas/research/__init__.py atlas/research/method_registry.py atlas/research/workflow_reproducible_graph.py atlas/thermo/__init__.py tests/unit/research/test_method_registry.py tests/unit/research/test_workflow_reproducible_graph.py tests/unit/thermo/test_init_exports.py`
  - `python -m pytest -q tests/unit/research/test_method_registry.py tests/unit/research/test_workflow_reproducible_graph.py tests/unit/thermo/test_init_exports.py`

## Research references used in batch 41

- PEP 562 (module `__getattr__`/`__dir__`): https://peps.python.org/pep-0562/
- Python `importlib` docs: https://docs.python.org/3/library/importlib.html
- Python `dataclasses` docs: https://docs.python.org/3/library/dataclasses.html
- Python `tempfile` docs (atomic temp-file workflow): https://docs.python.org/3/library/tempfile.html
- Python `pathlib` docs: https://docs.python.org/3/library/pathlib.html
- W3C PROV family overview (provenance model for reproducible pipelines): https://www.w3.org/TR/prov-overview/
- NeurIPS Reproducibility Checklist (artifact/reporting norms): https://neurips.cc/public/guides/PaperChecklist
- ACM Artifact Review & Badging policy: https://www.acm.org/publications/policies/artifact-review-and-badging-current
- Software Heritage reproducibility resources (archival/provenance context): https://www.softwareheritage.org/
- PyPA entry points specification (plugin/registry extensibility reference): https://packaging.python.org/en/latest/specifications/entry-points/
- MLflow tracking docs (run metadata/artifact lineage reference): https://mlflow.org/docs/latest/tracking.html
- DVC experiment docs (reproducible experiment/version context): https://dvc.org/doc

## Progress snapshot (after Batch 41)

- Completed: Batch 1 through Batch 41.
- Pending: Batch 42 onward.

## Batch 42 (max 5 files)

- [X] `atlas/thermo/calphad.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/__init__.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/atomate2_wrapper.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/engine.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 42 optimization goals

- Harden CALPHAD path robustness (composition normalization, temperature/step validation, Scheil/equilibrium fallback behavior, and stable phase extraction).
- Convert OpenMM package entrypoint to lazy optional exports so missing optional dependencies do not break module import.
- Stabilize native Atomate2 OpenMM wrapper with explicit parameter schema and delayed optional import resolution.
- Improve OpenMM engine reliability for periodic box handling, simulation input validation, reporter lifecycle, and non-finite output defense.

## Batch 42 outcomes

- `thermo/calphad.py`:

  - added strict composition normalization/validation:

    - unknown component rejection,
    - non-finite/negative fraction rejection,
    - dependent-component completion + final normalization.
  - added temperature and `n_steps` validation for equilibrium/solidification paths.
  - strengthened equilibrium extraction from pycalphad outputs (`Phase`/`NP`) using finite checks and stable sorting.
  - made equilibrium path output semantics cleaner by separating `LIQUID` from `solid_phases`.
  - made Scheil path robust to optional module absence and malformed result payloads, with deterministic fallback to equilibrium path.
  - added safer transus detection and plotting guards (shape checks, finite annotation checks, output path creation).
- `thermo/openmm/__init__.py`:

  - replaced eager import with lazy export map (`OpenMMEngine`, `PymatgenTrajectoryReporter`).
  - optional import failures now return `None` instead of crashing package import, with debug-trace capture.
- `thermo/openmm/atomate2_wrapper.py`:

  - replaced brittle top-level imports with delayed `importlib` loading.
  - added constructor validation for `temperature`, `step_size`, `ensemble`.
  - standardized ensemble modes (`nvt`, `npt`, `minimize`) and step validation in maker construction.
  - improved error diagnostics for missing optional Atomate2/OpenMM stacks.
- `thermo/openmm/engine.py`:

  - added constructor-level numeric validation (`temperature`, `friction`, `step_size`).
  - added system setup guards (`atoms` non-empty, finite positions, mass validation).
  - improved periodic box setup with full vector handling (`setPeriodicBoxVectors` when available).
  - made forcefield handling robust:

    - best-effort MACE setup via openmm-ml,
    - deterministic LJ fallback with periodic/non-periodic method selection.
  - improved simulation run path:

    - validated `steps` and `trajectory_interval`,
    - ensured reporter is detached after run,
    - validated final potential energy finite before returning trajectory.
- Verification:

  - `python -m ruff check atlas/thermo/calphad.py atlas/thermo/openmm/__init__.py atlas/thermo/openmm/atomate2_wrapper.py atlas/thermo/openmm/engine.py`
  - `python -m py_compile atlas/thermo/calphad.py atlas/thermo/openmm/__init__.py atlas/thermo/openmm/atomate2_wrapper.py atlas/thermo/openmm/engine.py`
  - `python -m pytest -q tests/unit/thermo/test_init_exports.py`
  - `python -m pytest -q tests/integration/openmm/test_openmm_core.py tests/integration/openmm/test_openmm_mace.py` (both skipped because `openmm` not installed in this environment)
  - smoke script for:

    - CALPHAD helper normalization/transus,
    - lazy OpenMM export behavior under missing dependency,
    - wrapper argument-validation paths.

## Research references used in batch 42

- OpenMM User Guide (Simulation APIs): https://docs.openmm.org/latest/userguide/application/04_advanced_sim_examples.html
- OpenMM API docs (`Topology`, periodic box vectors): https://docs.openmm.org/latest/api-python/generated/openmm.app.topology.Topology.html
- OpenMM API docs (`NonbondedForce` methods): https://docs.openmm.org/latest/api-python/generated/openmm.openmm.NonbondedForce.html
- OpenMM API docs (`LangevinMiddleIntegrator`): https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LangevinMiddleIntegrator.html
- OpenMM framework paper (PLoS Comput Biol 2017): https://pmc.ncbi.nlm.nih.gov/articles/PMC5549999/
- pycalphad docs (`equilibrium` API): https://pycalphad.org/docs/latest/api/pycalphad.core.html#pycalphad.core.equilibrium.equilibrium
- pycalphad Scheil package docs: https://scheil.readthedocs.io/en/stable/
- pycalphad project docs: https://pycalphad.org/docs/latest/
- pycalphad JORS paper: https://openresearchsoftware.metajnl.com/articles/10.5334/jors.140
- CALPHAD method overview (NIST): https://www.nist.gov/publications/calphad-calculation-phase-diagrams-comprehensive-guide
- openmm-ml package (MACE/ML potential integration context): https://github.com/openmm/openmm-ml

## Progress snapshot (after Batch 42)

- Completed: Batch 1 through Batch 42.
- Pending: Batch 43 onward.

## Batch 43 (max 5 files)

- [X] `atlas/thermo/openmm/reporters.py` - reviewed + optimized
- [X] `atlas/thermo/stability.py` - reviewed + optimized
- [X] `atlas/topology/__init__.py` - reviewed + optimized
- [X] `atlas/topology/classifier.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 43 optimization goals

- Harden OpenMM trajectory reporting to ensure finite-safe frame extraction and stable pymatgen trajectory export semantics.
- Strengthen phase-stability analysis contracts (input validation, robust decomposition formatting, and clearer failure payloads).
- Align `atlas.topology` package entrypoint with lazy-export architecture to reduce import-time coupling.
- Improve topology classifier numerical robustness for sparse/non-contiguous batch IDs and invalid edge tensors.

## Batch 43 outcomes

- `thermo/openmm/reporters.py`:

  - added strict constructor validation (`reportInterval > 0`, `Structure` type check).
  - converted conversion constants to explicit symbols and finite-checks for positions/time/energy/forces.
  - hardened `describeNextReport` to always return at least one step.
  - improved exported trajectory payload:

    - consistent `frame_properties` keys (`energy_ev`, `forces_ev_per_ang`, `time_ps`),
    - computed scalar `time_step` from recorded times,
    - explicit failure if no frames were collected.
- `thermo/stability.py`:

  - added typed `StabilityResult` model and stronger validations for energies and chemical system input.
  - `ReferenceDatabase` now enforces key presence in `load_from_list` and finite energy values in `add_entry`.
  - `analyze_stability` now produces deterministic decomposition formatting and clearer fallback payloads.
  - `plot_phase_diagram` now returns `None` safely when no entries exist and avoids unexpected crashes.
- `topology/__init__.py`:

  - switched to explicit lazy-export map with `__getattr__`/`__dir__`.
  - preserved public API (`CrystalGraphBuilder`, `TopoGNN`) with lower import-time coupling.
- `topology/classifier.py`:

  - strengthened input guards:

    - finite checks for node/edge features,
    - edge index dtype/shape/range validation,
    - batch dtype and non-negativity checks.
  - added batch ID remapping to contiguous graph IDs before pooling, preventing empty-graph holes from propagating `-inf` into readout.
  - stabilized max-pooling path via `nan_to_num` and added empty-edge guard in message passing.
  - added checkpoint config type check in `load_model` for safer deserialization path.
- Verification:

  - `python -m ruff check atlas/thermo/openmm/reporters.py atlas/thermo/stability.py atlas/topology/__init__.py atlas/topology/classifier.py`
  - `python -m py_compile atlas/thermo/openmm/reporters.py atlas/thermo/stability.py atlas/topology/__init__.py atlas/topology/classifier.py`
  - `python -m pytest -q tests/unit/topology/test_classifier.py tests/unit/thermo/test_init_exports.py`
  - smoke script for:

    - `PhaseStabilityAnalyst` minimal hull path,
    - non-contiguous batch handling in `TopoGNN.forward`.

## Research references used in batch 43

- OpenMM Reporter API (StateDataReporter reference): https://docs.openmm.org/8.0.0/api-python/generated/openmm.app.statedatareporter.StateDataReporter.html
- OpenMM Cookbook (reporting workflow): https://openmm.github.io/openmm-cookbook/latest/notebooks/tutorials/loading_and_reporting.html
- pymatgen usage docs (phase diagram workflow): https://pymatgen.org/usage.html
- pymatgen analysis docs (`phase_diagram` methods): https://pymatgen.org/pymatgen.analysis
- pycalphad project docs: https://pycalphad.org/docs
- PyTorch `scatter_reduce` docs: https://docs.pytorch.org/docs/stable/generated/torch.scatter_reduce.html
- PyTorch `LayerNorm` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- GraphSAGE paper: https://arxiv.org/abs/1706.02216
- GIN / "How Powerful are Graph Neural Networks?": https://arxiv.org/abs/1810.00826
- PEP 562 (`__getattr__`/`__dir__` on modules): https://peps.python.org/pep-0562/

## Progress snapshot (after Batch 43)

- Completed: Batch 1 through Batch 43.
- Pending: Batch 44 onward.

## Batch 44 (max 5 files)

- [X] `atlas/training/__init__.py` - reviewed + optimized
- [X] `atlas/training/checkpoint.py` - reviewed + optimized
- [X] `atlas/training/filters.py` - reviewed + optimized
- [X] `atlas/training/losses.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 44 optimization goals

- Align `atlas.training` package exports with lazy-loading architecture to reduce import-time coupling and keep API introspection deterministic.
- Harden checkpoint persistence against partial writes and malformed checkpoint payloads.
- Strengthen outlier filtering robustness under missing optional CSV dependencies and heterogeneous scalar payload types.
- Tighten loss-module contracts (constraint validation, duplicate task-name rejection, shape mismatch diagnostics) without breaking existing trainer semantics.

## Batch 44 outcomes

- `training/__init__.py`:

  - replaced eager imports with explicit lazy export map (`__getattr__`, `__dir__`) while preserving public API symbols.
- `training/checkpoint.py`:

  - added atomic checkpoint write helper (`_atomic_torch_save`) using temp file + atomic replace to prevent partial/corrupted checkpoints.
  - added strict mapping validation for `state` payloads (`_coerce_state`) in `save_best` / `save_checkpoint`.
  - enriched checkpoint payload with default metadata (`epoch`, `val_mae`) and switched best pointer update to `copy2`.
- `training/filters.py`:

  - removed hard top-level dependency on pandas; CSV export now gracefully falls back to stdlib `csv` when pandas is unavailable.
  - added dataset/properties interface validation and empty-dataset fast path.
  - improved scalar extraction for single-value numpy arrays and normalized deduplicated property names.
  - added minimum-sample guard in sigma computation to avoid unstable one-point standardization.
- `training/losses.py`:

  - switched scalar-finite checks from tensor allocation to `math.isfinite` for clearer/cheaper validation.
  - moved constraint validation to `PropertyLoss.__init__` (fail-fast config error).
  - added tensor type/shape checks in `PropertyLoss.forward` with explicit mismatch diagnostics.
  - enforced unique non-empty task names in `MultiTaskLoss` and improved per-task shape mismatch errors.
- Tests:

  - updated `tests/unit/training/test_losses.py` for new fail-fast constraint validation and added duplicate-task/shape-mismatch coverage.
  - expanded `tests/unit/training/test_filters.py` for empty-dataset and property-name normalization behavior.
  - expanded `tests/unit/training/test_checkpoint.py` for non-mapping checkpoint payload rejection.
  - added `tests/unit/training/test_init_exports.py` to validate lazy-export package contract.
- Verification:

  - `python -m ruff check atlas/training/__init__.py atlas/training/checkpoint.py atlas/training/filters.py atlas/training/losses.py tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py tests/unit/training/test_losses.py tests/unit/training/test_init_exports.py`
  - `python -m py_compile atlas/training/__init__.py atlas/training/checkpoint.py atlas/training/filters.py atlas/training/losses.py tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py tests/unit/training/test_losses.py tests/unit/training/test_init_exports.py`
  - `python -m pytest -q tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py tests/unit/training/test_losses.py tests/unit/training/test_init_exports.py`

## Research references used in batch 44

- PEP 562 (`__getattr__` / `__dir__` for modules): https://peps.python.org/pep-0562/
- Python `importlib` docs: https://docs.python.org/3/library/importlib.html
- Python `tempfile` docs (safe temp-file writes): https://docs.python.org/3/library/tempfile.html
- Python `Path.replace` / atomic replace semantics (`os.replace` behavior): https://docs.python.org/3/library/pathlib.html#pathlib.Path.replace
- PyTorch serialization notes: https://pytorch.org/docs/stable/notes/serialization.html
- PyTorch `torch.save` docs: https://pytorch.org/docs/stable/generated/torch.save.html
- NIST/SEMATECH e-Handbook (MAD/robust scale background): https://www.itl.nist.gov/div898/handbook/
- Iglewicz & Hoaglin (1993), robust outlier labeling rule using modified Z-scores: https://books.google.com/books/about/How_to_Detect_and_Handle_Outliers.html?id=FuuiEAAAQBAJ
- Kendall, Gal, Cipolla (CVPR 2018), uncertainty-weighted multi-task loss: https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html
- PyTorch BCE-with-logits API (classification loss contract): https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html

## Progress snapshot (after Batch 44)

- Completed: Batch 1 through Batch 44.
- Pending: Batch 45 onward.

## Batch 45 (max 5 files)

- [X] `atlas/training/metrics.py` - reviewed + optimized
- [X] `atlas/training/normalizers.py` - reviewed + optimized
- [X] `atlas/training/physics_losses.py` - reviewed + optimized
- [X] `atlas/training/preflight.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 45 optimization goals

- Harden training metrics against non-finite tensors, shape drift, and optional dependency variability.
- Improve target normalizer robustness for iterable datasets, property-name hygiene, and deterministic state serialization.
- Stabilize physics-constrained loss path for singular elastic tensors and malformed prediction payloads.
- Strengthen preflight runtime diagnostics and artifact-integrity checks before expensive downstream stages.

## Batch 45 outcomes

- `training/metrics.py`:

  - added finite-safe tensor coercion and pair filtering utilities to keep scalar/classification metrics stable under NaN/Inf.
  - normalized matrix metrics to support both single-matrix and batched-matrix inputs.
  - added optional-SciPy Spearman fallback implementation to keep `eigenvalue_agreement` usable when SciPy is unavailable.
- `training/normalizers.py`:

  - added iterable dataset support (not only random-access datasets), mapping/attribute property extraction, and property-name normalization.
  - added stronger `state_dict`/`load_state_dict` schema guards and deterministic key ordering.
  - added explicit unknown-property errors with available-key context.
- `training/physics_losses.py`:

  - rebuilt Voigt-Reuss/Born loss path with finite filtering, pseudo-inverse compliance fallback (`pinv`) for singular tensors, and safe zero-loss fallbacks.
  - validated alpha/weight hyperparameters (finite, non-negative, known keys) and tightened type checks for prediction payloads.
  - ensured all loss terms remain finite under malformed/non-finite physics tensors.
- `training/preflight.py`:

  - added command execution OSError handling (`127`) and improved failure diagnostics.
  - added validation-report integrity gate (must exist and be non-empty after validate step).
  - upgraded split-manifest gate to require non-empty manifest files.
- Tests:

  - expanded `tests/unit/training/test_metrics.py` for single-matrix Frobenius behavior and SciPy-missing Spearman fallback.
  - expanded `tests/unit/training/test_normalizers.py` for iterable datasets, property-name normalization, and empty state-key rejection.
  - expanded `tests/unit/training/test_preflight.py` for missing validation report and OSError command-failure return codes.
  - added `tests/unit/training/test_physics_losses.py` for physics loss finite-safety and alpha validation.
- Verification:

  - `python -m ruff check atlas/training/metrics.py atlas/training/normalizers.py atlas/training/physics_losses.py atlas/training/preflight.py tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py tests/unit/training/test_preflight.py tests/unit/training/test_physics_losses.py`
  - `python -m py_compile atlas/training/metrics.py atlas/training/normalizers.py atlas/training/physics_losses.py atlas/training/preflight.py tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py tests/unit/training/test_preflight.py tests/unit/training/test_physics_losses.py`
  - `python -m pytest -q tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py tests/unit/training/test_preflight.py tests/unit/training/test_physics_losses.py`

## Research references used in batch 45

- SciPy `spearmanr` API docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
- PyTorch `torch.linalg.eigvalsh` docs: https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigvalsh.html
- PyTorch `torch.pinverse`/`torch.linalg.pinv` docs: https://docs.pytorch.org/docs/stable/generated/torch.pinverse.html
- scikit-learn `StandardScaler` docs (`std` convention and normalization semantics): https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- Python `subprocess.run` docs (timeout/return-code contract): https://docs.python.org/3/library/subprocess.html
- Materials Project elasticity methodology (Voigt/Reuss/VRH reporting): https://docs.materialsproject.org/methodology/materials-methodology/elasticity
- pymatgen elasticity API (VRH properties): https://pymatgen.org/pymatgen.analysis.elasticity.html
- Mouhat & Coudert (2014), elastic stability criteria overview: https://arxiv.org/abs/1410.0065
- PRB publication mirror for Born criteria details: https://www.coudert.name/papers/10.1103_PhysRevB.90.224104.pdf

## Progress snapshot (after Batch 45)

- Completed: Batch 1 through Batch 45.
- Pending: Batch 46 onward.

## Batch 46 (max 5 files)

- [X] `atlas/training/run_utils.py` - reviewed + optimized
- [X] `atlas/training/trainer.py` - reviewed + optimized
- [X] `tests/unit/training/test_run_utils_manifest.py` - reviewed + optimized
- [X] `tests/unit/training/test_trainer.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 46 optimization goals

- Strengthen run-manifest persistence integrity (atomic writes + schema-preserving merge semantics).
- Prevent accidental corruption of core manifest sections when callers pass `extra` payloads.
- Improve trainer reliability for checkpoint/history writes and validation-time numeric failures.
- Harden filename/checkpoint stem safety for CI/runtime reproducibility paths.

## Batch 46 outcomes

- `training/run_utils.py`:

  - upgraded `_dump_json_file` to atomic temp-file replace with flush+fsync (crash-safe write pattern).
  - added `_merge_extra_payload` to enforce schema integrity for reserved dict sections (`runtime/args/dataset/split/environment_lock/artifacts/metrics/seeds/configs`).
  - repaired invalid legacy `created_at` payloads during merge (non-string values are normalized to new UTC ISO timestamp).
  - extracted `_ensure_seed_and_config_sections` and `_redact_manifest_sections` to reduce complexity and make policy explicit.
- `training/trainer.py`:

  - added strict checkpoint-name/file-name validation to block path traversal/separator injection.
  - switched checkpoint/history persistence to atomic write helpers (`_atomic_torch_save`, `_atomic_json_dump`).
  - added validation-loop non-finite loss guard (parity with train-loop finite check).
  - strengthened checkpoint loading contract with explicit payload type/key validation.
  - added explicit error when dict predictions cannot resolve any target mapping.
- Tests:

  - `test_run_utils_manifest.py`:

    - added regression for rejecting non-mapping `extra` payload on reserved manifest sections.
    - added regression for repairing invalid `created_at` during merge.
  - `test_trainer.py`:

    - added validation non-finite loss failure test.
    - added checkpoint loader malformed-payload test.
    - added dict-prediction missing-target test.
    - added filename/path-like checkpoint guard tests.
- Verification:

  - `python -m ruff check atlas/training/run_utils.py atlas/training/trainer.py tests/unit/training/test_run_utils_manifest.py tests/unit/training/test_trainer.py`
  - `python -m py_compile atlas/training/run_utils.py atlas/training/trainer.py tests/unit/training/test_run_utils_manifest.py tests/unit/training/test_trainer.py`
  - `python -m pytest -q tests/unit/training/test_run_utils_manifest.py tests/unit/training/test_trainer.py`

## Research references used in batch 46

- Python `tempfile` docs (safe temp-file creation patterns): https://docs.python.org/3/library/tempfile.html
- Python `pathlib.Path.replace` docs (atomic replacement semantics): https://docs.python.org/3/library/pathlib.html#pathlib.Path.replace
- Python `json` docs (`allow_nan=False`, deterministic dump controls): https://docs.python.org/3/library/json.html
- PyTorch AMP examples (autocast/GradScaler behavior): https://pytorch.org/docs/stable/notes/amp_examples.html
- PyTorch `clip_grad_norm_` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- PyTorch checkpoint save/load tutorial: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- IETF RFC 8785 (JSON Canonicalization Scheme): https://datatracker.ietf.org/doc/html/rfc8785
- Python `subprocess` docs (timeout and robust command execution): https://docs.python.org/3/library/subprocess.html

## Progress snapshot (after Batch 46)

- Completed: Batch 1 through Batch 46.
- Pending: Batch 47 onward.

## Batch 47 (max 5 files)

- [X] `scripts/phase5_active_learning/active_learning.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_phase5.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_discovery.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/search_materials.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/structure_enumerator.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 47 optimization goals

- Harden script-level AL acquisition utilities against shape mismatch, optional dependency variance, and unstable numeric inputs.
- Improve launcher argument validation for safer preflight/profile execution paths.
- Strengthen discovery/classifier loading compatibility across torch versions and malformed checkpoint payloads.
- Upgrade materials-search query validation so invalid columns/types fail fast before expensive search pipelines.
- Reduce combinatorial enumeration overhead and improve substitution hygiene/deduplication determinism.

## Batch 47 outcomes

- `active_learning.py`:

  - added Gaussian CDF/PDF fallback path when SciPy is unavailable (`math.erf` based), keeping EI functional in minimal environments.
  - strengthened acquisition validation:

    - `acquisition_uncertainty` now enforces mean/std leading-dimension consistency,
    - `acquisition_random` now validates non-negative seed.
  - made AL loop robust to dict-model outputs via `_select_prediction_tensor` in train/eval/MC-dropout paths.
  - hardened multi-objective screening to handle missing objectives, non-finite predictions, and mixed tensor/dict model outputs.
- `run_phase5.py`:

  - added explicit validation for preflight property group schema (`[A-Za-z0-9._-]+`) and non-empty `--results-dir`.
- `run_discovery.py`:

  - improved classifier checkpoint loading compatibility:

    - try `torch.load(..., weights_only=True)` first,
    - fallback to legacy `torch.load(... )` for older torch builds,
    - explicit payload type guard before `load_state_dict`.
- `search_materials.py`:

  - added `_validate_query_columns` to fail fast on unknown criteria/sort columns.
  - added numeric-type enforcement for range filters (reject non-numeric columns with numeric bounds).
  - retained return-code based CLI exit semantics for CI integration.
- `structure_enumerator.py`:

  - strengthened substitution normalization with dedupe, empty-option rejection, and DummySpecies-safe handling.
  - fixed variant cap logic to respect true combinational size when below hard cap.
  - reduced duplicate-filtering overhead by bucketing candidates by reduced formula before `StructureMatcher.group_structures`.

## Tests updated in batch 47

- `tests/unit/active_learning/test_phase5_cli.py`

  - added coverage for invalid preflight property group and empty `--results-dir`.
- `tests/unit/active_learning/test_search_materials_cli.py`

  - added coverage for missing criteria columns, non-numeric range filters, invalid sort column.
- `tests/unit/active_learning/test_structure_enumerator_script.py`

  - added coverage for empty substitution option rejection and option deduplication.
- `tests/unit/active_learning/test_phase6_active_learning.py`

  - added coverage for uncertainty shape mismatch and negative random-seed rejection.

## Verification

- `python -m ruff check scripts/phase5_active_learning/active_learning.py scripts/phase5_active_learning/run_phase5.py scripts/phase5_active_learning/run_discovery.py scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/structure_enumerator.py tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_phase6_active_learning.py`
- `python -m py_compile scripts/phase5_active_learning/active_learning.py scripts/phase5_active_learning/run_phase5.py scripts/phase5_active_learning/run_discovery.py scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/structure_enumerator.py tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_phase6_active_learning.py`
- `python -m pytest -q tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_phase6_active_learning.py tests/unit/active_learning/test_test_enumeration_script.py`

## Research references used in batch 47

- SciPy normal distribution API (`cdf`/`pdf` contract): https://docs.scipy.org/doc/scipy-1.7.0/reference/generated/scipy.stats.norm.html
- Bayesian Optimization / Expected Improvement (EGO, 1998): https://doi.org/10.1023/A:1008306431147
- MC Dropout as Bayesian Approximation (Gal & Ghahramani, 2016): https://proceedings.mlr.press/v48/gal16.html
- NumPy modern RNG (`default_rng`) docs: https://numpy.org/doc/stable/reference/random/generator.html
- PyTorch model loading / serialization notes: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- pandas numeric dtype checks (`is_numeric_dtype`): https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_numeric_dtype.html
- pymatgen StructureMatcher API (`group_structures` semantics): https://pymatgen.org/pymatgen.analysis
- Python regex syntax/validation patterns: https://docs.python.org/3/library/re.html

## Progress snapshot (after Batch 47)

- Completed: Batch 1 through Batch 47.
- Pending: Batch 48 onward.

## Batch 48 (max 5 files)

- [X] `scripts/phase5_active_learning/test_enumeration.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/active_learning.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/search_materials.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/structure_enumerator.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_test_enumeration_script.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 48 optimization goals

- Strengthen script contract quality for `test_enumeration` so CLI/demo usage is machine-verifiable and CI-friendly.
- Improve AL script model-forward compatibility across heterogeneous graph model signatures.
- Enforce stricter query/output-column validation for search CLI to fail fast on invalid operator/column combinations.
- Make structure enumerator options more deterministic and semantically aligned with exposed parameters.

## Batch 48 outcomes

- `test_enumeration.py`:

  - added contract checks for injected enumerator (`generate(...)` callable required).
  - added summary payload validation (non-empty formula, non-negative integer counts).
  - added `--json` output mode and error-to-exit-code handling for scripting/automation.
- `active_learning.py`:

  - added `_forward_graph_model` dispatch helper to tolerate common model signatures and feature-attribute variants (`edge_attr`/`edge_vec`/none).
  - switched training finite guard from `isnan` to full `isfinite` to reject `inf` loss values as well.
  - stabilized entrypoint with explicit `main() -> int` and `SystemExit` return propagation.
- `search_materials.py`:

  - strengthened argument validation for `--save` and `--columns` (non-empty tokens).
  - extended `_validate_query_columns` to validate requested output columns in addition to criteria/sort fields.
  - made CSV save path robust by creating parent directories when needed.
- `structure_enumerator.py`:

  - normalized substitution keys with whitespace stripping and non-empty enforcement.
  - made `remove_superperiodic=False` behavior explicit by skipping duplicate filtering path.
  - improved determinism by iterating grouped formulas in sorted order before `StructureMatcher` grouping.
  - added explicit warning for `max_index>1` in fallback mode (capability transparency).
- Tests:

  - `test_test_enumeration_script.py` now includes enumerator contract validation path.

## Verification

- `python -m ruff check scripts/phase5_active_learning/test_enumeration.py scripts/phase5_active_learning/active_learning.py scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/structure_enumerator.py tests/unit/active_learning/test_test_enumeration_script.py`
- `python -m py_compile scripts/phase5_active_learning/test_enumeration.py scripts/phase5_active_learning/active_learning.py scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/structure_enumerator.py tests/unit/active_learning/test_test_enumeration_script.py`
- `python -m pytest -q tests/unit/active_learning/test_test_enumeration_script.py tests/unit/active_learning/test_phase6_active_learning.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_structure_enumerator_script.py`

## Research references used in batch 48

- SciPy normal distribution API (`cdf`/`pdf`): https://docs.scipy.org/doc/scipy-1.7.0/reference/generated/scipy.stats.norm.html
- Bayesian Optimization / EI (EGO, Jones et al., 1998): https://doi.org/10.1023/A:1008306431147
- MC Dropout as Bayesian Approximation (Gal & Ghahramani, 2016): https://proceedings.mlr.press/v48/gal16.html
- PyTorch finite-check API (`torch.isfinite`): https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- PyTorch Huber loss API: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.huber_loss.html
- NumPy Generator API (`default_rng`): https://numpy.org/doc/stable/reference/random/generator.html
- pandas numeric dtype guard (`is_numeric_dtype`): https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_numeric_dtype.html
- pymatgen `StructureMatcher.group_structures` docs: https://pymatgen.org/pymatgen.analysis.html
- StructureMatcher robustness paper (OpenReview 2025): https://openreview.net/forum?id=ss5taK9Iy6
- Python argparse docs (CLI contract design): https://docs.python.org/3/library/argparse.html

## Progress snapshot (after Batch 48)

- Completed: Batch 1 through Batch 48.
- Pending: Batch 49 onward.

## Batch 49 (max 5 files)

- [X] `scripts/phase5_active_learning/run_phase5.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_discovery.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_phase5_cli.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 49 optimization goals

- Strengthen Phase5/Discovery CLI run-directory semantics so conflicting flags fail fast before expensive runtime work.
- Improve classifier checkpoint loading compatibility across common PyTorch checkpoint formats (raw `state_dict`, nested `state_dict`, DataParallel prefix).
- Expand regression coverage for new validation gates and checkpoint-payload extraction behavior.
- Keep launcher behavior deterministic and CI-friendly while preserving backward-compatible default flows.

## Batch 49 outcomes

- `run_phase5.py`:

  - added conflict guard for `--resume` + `--results-dir` (now rejected early with explicit message).
- `run_discovery.py`:

  - added checkpoint parsing helpers:

    - `_looks_like_state_dict(...)`
    - `_extract_classifier_state_dict(...)`
  - classifier loader now supports:

    - direct state dict payloads,
    - nested payloads (`state_dict` / `model_state_dict` / `model`),
    - `module.` key-prefix normalization from DataParallel checkpoints.
  - added run-directory conflict guards in discovery validation:

    - `--run-id` + `--results-dir` rejected,
    - `--resume` + `--results-dir` rejected.
- `test_phase5_cli.py`:

  - added `_phase5_args(...)` helper for stable/complete Namespace construction in validation tests.
  - added regression tests for:

    - conflicting run-directory flags in both launchers,
    - conflicting preflight-mode flags,
    - classifier checkpoint extraction for nested/DataParallel payloads,
    - invalid checkpoint payload rejection.

## Verification

- `python -m ruff check scripts/phase5_active_learning/run_phase5.py scripts/phase5_active_learning/run_discovery.py tests/unit/active_learning/test_phase5_cli.py`
- `python -m py_compile scripts/phase5_active_learning/run_phase5.py scripts/phase5_active_learning/run_discovery.py tests/unit/active_learning/test_phase5_cli.py`
- `python -m pytest -q tests/unit/active_learning/test_phase5_cli.py`
- `python -m pytest -q tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_phase6_active_learning.py`

## Research references used in batch 49

- Python `argparse` docs (CLI contract and mutually exclusive behavior patterns): https://docs.python.org/3/library/argparse.html
- Python `pathlib` docs (path semantics used in launcher runtime resolution): https://docs.python.org/3/library/pathlib.html
- Python `subprocess` docs (launcher process execution contract): https://docs.python.org/3/library/subprocess.html
- PyTorch "Saving and Loading Models" tutorial (checkpoint format patterns): https://pytorch.org/tutorials/beginner/saving_loading_models.html
- PyTorch serialization notes (`torch.load` behavior and compatibility): https://pytorch.org/docs/stable/notes/serialization.html
- PyTorch `DataParallel` API reference (module wrapper behavior impacting checkpoint keys): https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
- OWASP Path Traversal overview (input hardening rationale for run-id/path-like flags): https://owasp.org/www-community/attacks/Path_Traversal
- CWE-22 (Path Traversal) reference taxonomy: https://cwe.mitre.org/data/definitions/22.html

## Progress snapshot (after Batch 49)

- Completed: Batch 1 through Batch 49.
- Pending: Batch 50 onward.

## Batch 50 (max 5 files)

- [X] `scripts/phase5_active_learning/test_enumeration.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/structure_enumerator.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_test_enumeration_script.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_structure_enumerator_script.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 50 optimization goals

- Improve fallback structure-enumeration quality under capped combinatorial spaces by reducing prefix-truncation bias.
- Strengthen script-level `test_enumeration` contract so injected enumerators must return valid `pymatgen.Structure` sequences.
- Add regression tests proving the new deterministic capped-space sampling and payload contract checks.
- Keep fallback behavior deterministic and CI-friendly for Windows/non-compiled environments.

## Batch 50 outcomes

- `structure_enumerator.py`:

  - introduced typed species-option alias for clearer substitution contract (`str | DummySpecies`).
  - added deterministic stratified ordinal sampler:

    - `_select_variant_ordinals(total_variants, limit)`
    - `_decode_variant_ordinal(ordinal, radices)`
  - replaced prefix `itertools.product` truncation with mixed-radix ordinal decoding so capped runs sample across higher-order substitution dimensions (not only early lexicographic prefix).
  - preserved existing optional duplicate filtering and fallback safety semantics.
- `test_enumeration.py`:

  - added `_validate_generated_structures(...)` to enforce `generate(...) -> sequence[Structure]` contract at runtime.
  - both simple substitution and vacancy substitution paths now validate generated payload types before summary construction.
- Tests:

  - `test_test_enumeration_script.py`:

    - added failure-path test for non-Structure `generate(...)` payloads.
  - `test_structure_enumerator_script.py`:

    - added direct regression for stratified ordinal selection.
    - added capped-space regression ensuring generated set covers both Ti and Zr substitutions when truncation is active.

## Verification

- `python -m ruff check scripts/phase5_active_learning/structure_enumerator.py scripts/phase5_active_learning/test_enumeration.py tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_test_enumeration_script.py`
- `python -m py_compile scripts/phase5_active_learning/structure_enumerator.py scripts/phase5_active_learning/test_enumeration.py tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_test_enumeration_script.py`
- `python -m pytest -q tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_test_enumeration_script.py`

## Research references used in batch 50

- Python `itertools.product` docs (Cartesian-product baseline behavior): https://docs.python.org/3/library/itertools.html#itertools.product
- Python typing `TypeAlias` docs: https://docs.python.org/3/library/typing.html#typing.TypeAlias
- pymatgen `Structure.replace` API docs: https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure.replace
- pymatgen `StructureMatcher.group_structures` API docs: https://pymatgen.org/pymatgen.analysis.html#pymatgen.analysis.structure_matcher.StructureMatcher.group_structures
- Bergstra & Bengio (2012), random search vs grid-search coverage rationale: https://jmlr.org/beta/papers/v13/bergstra12a.html
- McKay, Beckman, Conover (1979), space-filling/stratified sampling (Latin Hypercube): https://www.osti.gov/biblio/5236110
- Cheon et al. (2025), StructureMatcher robustness analysis: https://openreview.net/forum?id=ss5taK9Iy6

## Progress snapshot (after Batch 50)

- Completed: Batch 1 through Batch 50.
- Pending: Batch 51 onward.

## Batch 51 (max 5 files)

- [X] `atlas/benchmark/cli.py` - reviewed + optimized
- [X] `tests/unit/benchmark/test_benchmark_cli.py` - reviewed + optimized
- [X] `atlas/console_style.py` - reviewed + optimized
- [X] `tests/unit/test_console_style.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 51 optimization goals

- Harden benchmark CLI argument governance to fail fast on conflicting output/preflight modes and malformed preflight controls.
- Improve checkpoint loading robustness/security posture by supporting common checkpoint wrappers while preferring safer torch loading behavior.
- Expand benchmark CLI regression tests to lock new validation and checkpoint-compatibility behavior.
- Improve console styling consistency for multi-digit phase headers and add explicit environment kill-switch for style injection.

## Batch 51 outcomes

- `benchmark/cli.py`:

  - added stricter CLI validation for:

    - `--bootstrap-seed >= 0`
    - `--preflight-split-seed >= 0`
    - `--preflight-timeout-sec > 0`
    - non-empty + safe `--preflight-property-group` schema
    - `--folds` requires at least one index when provided
    - `--output` and `--output-dir` mutual exclusion
    - `--preflight-only` and `--skip-preflight` conflict rejection
  - added `--preflight-timeout-sec` CLI option and wired it into `run_preflight(...)`.
  - improved preflight failure logging with structured detail (`error_message`).
  - introduced robust checkpoint extraction helpers:

    - `_looks_like_state_dict(...)`
    - `_extract_state_dict(...)`
  - `_load_model(...)` now prefers `torch.load(..., weights_only=True)` with fallback for older torch, supports nested checkpoint containers, and normalizes DataParallel `module.` prefixes.
- `test_benchmark_cli.py`:

  - added regression tests for new validation gates (invalid property group, output conflict, preflight mode conflict).
  - added checkpoint extraction tests (nested/DataParallel payload success + invalid payload failure).
  - added end-to-end `_load_model(...)` compatibility test with synthetic nested DataParallel checkpoint.
- `console_style.py`:

  - phase header regex upgraded from single-digit (`[Phase1]`) to multi-digit (`[Phase10]`) coverage.
  - added `ATLAS_CONSOLE_STYLE=0/false/no` environment kill-switch to skip global print wrapping.
- `test_console_style.py`:

  - added regression test for style disable env behavior.
  - added regression test verifying multi-digit phase header styling.

## Verification

- `python -m ruff check atlas/benchmark/cli.py atlas/console_style.py tests/unit/benchmark/test_benchmark_cli.py tests/unit/test_console_style.py`
- `python -m py_compile atlas/benchmark/cli.py atlas/console_style.py tests/unit/benchmark/test_benchmark_cli.py tests/unit/test_console_style.py`
- `python -m pytest -q tests/unit/benchmark/test_benchmark_cli.py tests/unit/test_console_style.py`

## Research references used in batch 51

- Python argparse docs (CLI validation and parser error contracts): https://docs.python.org/3/library/argparse.html
- Python importlib docs (dynamic module loading semantics): https://docs.python.org/3/library/importlib.html
- Python pathlib docs (path validation semantics): https://docs.python.org/3/library/pathlib.html
- PyTorch serialization notes (`torch.load`, `weights_only` behavior): https://pytorch.org/docs/stable/notes/serialization.html
- PyTorch DataParallel docs (`module.` prefix behavior in state_dict): https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
- Matbench benchmark paper (objective/benchmarking context): https://arxiv.org/abs/2005.00707
- NO_COLOR convention (terminal color disable interoperability): https://no-color.org/
- ECMA-48 / ANSI escape sequence reference context: https://en.wikipedia.org/wiki/ANSI_escape_code

## Progress snapshot (after Batch 51)

- Completed: Batch 1 through Batch 51.
- Pending: Batch 52 onward.

## Batch 52 (max 5 files)

- [X] `atlas/training/checkpoint.py` - reviewed + optimized
- [X] `tests/unit/training/test_checkpoint.py` - reviewed + optimized
- [X] `atlas/training/filters.py` - reviewed + optimized
- [X] `tests/unit/training/test_filters.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 52 optimization goals

- Improve checkpoint-manager resume consistency so top-k state survives process restart and stays synchronized with on-disk artifacts.
- Harden checkpoint pointer update path to avoid partial `best.pt` writes.
- Fix outlier-filter API type hazard where string inputs for `properties` are silently treated as character sequences.
- Add regression tests for resume rehydration and stricter filter argument contracts.

## Batch 52 outcomes

- `training/checkpoint.py`:

  - added robust best-checkpoint filename parser and disk rehydration flow on manager init:

    - `_parse_best_filename(...)`
    - `_sync_best_models_from_disk(...)`
  - manager now discovers pre-existing `best_epoch_*_mae_*.pt`, sorts/prunes to `top_k`, and refreshes `best.pt` pointer accordingly.
  - added `_atomic_copy(...)` and replaced direct `shutil.copy2` for `best.pt` updates to keep pointer writes atomic.
- `training/filters.py`:

  - fixed `properties` type contract:

    - reject `str`/`bytes` as invalid top-level property sequence input,
    - enforce each property entry is a string before normalization.
  - prevents silent misconfiguration (e.g., `properties="target"` becoming `["t", "a", ...]`).
- Tests:

  - `test_checkpoint.py`:

    - added resume-state regression proving manager rehydrates from existing best files, prunes overflow, and points `best.pt` to the best epoch.
  - `test_filters.py`:

    - added argument-contract tests for invalid string payload and non-string property entries.

## Verification

- `python -m ruff check atlas/training/checkpoint.py atlas/training/filters.py tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py`
- `python -m py_compile atlas/training/checkpoint.py atlas/training/filters.py tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py`
- `python -m pytest -q tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py`

## Research references used in batch 52

- PyTorch serialization notes (`state_dict` best practice and `weights_only` loading guidance): https://docs.pytorch.org/docs/stable/notes/serialization.html
- PyTorch save/load tutorial (`state_dict` workflow): https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
- Python `tempfile` docs (`NamedTemporaryFile` behavior and safety notes): https://docs.python.org/3/library/tempfile.html
- Python `pathlib` docs (`Path.replace` semantics): https://docs.python.org/3/library/pathlib.html
- SciPy `median_abs_deviation` docs (MAD robustness rationale): https://docs.scipy.org/doc/scipy-1.9.0/reference/generated/scipy.stats.median_abs_deviation.html
- NIST/SEMATECH e-Handbook (robust scale measures, MAD motivation): https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm
- Iglewicz & Hoaglin (1993) reference summary for modified z-score thresholding context: https://rdrr.io/github/fosterlab/modern/man/iglewicz_hoaglin.html

## Progress snapshot (after Batch 52)

- Completed: Batch 1 through Batch 52.
- Pending: Batch 53 onward.

## Batch 53 (max 5 files)

- [X] `atlas/training/__init__.py` - reviewed + optimized
- [X] `tests/unit/training/test_init_exports.py` - reviewed + optimized
- [X] `atlas/models/__init__.py` - reviewed + optimized
- [X] `tests/unit/models/test_model_utils.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 53 optimization goals

- Improve lazy-export API stability for `atlas.training` and `atlas.models` package surfaces.
- Make lazy-export registries immutable to reduce accidental mutation risk during runtime.
- Ensure lazy exports are cached after first resolution to avoid repeated import/getattr overhead.
- Add regression tests for cache behavior, export-surface integrity, and clearer mismatch diagnostics.

## Batch 53 outcomes

- `training/__init__.py`:

  - switched export registry to `MappingProxyType` for immutable lazy-export mapping.
  - `__getattr__` now caches resolved exports into module globals and raises clearer mismatch errors when target modules miss expected attributes.
  - `__dir__` now uses explicit global key set union for stable symbol reporting.
- `models/__init__.py`:

  - switched export mapping to immutable `MappingProxyType`.
  - improved `__getattr__` mismatch diagnostics when target module export contract is violated.
  - retained and validated global-cache behavior for resolved exports.
- Tests:

  - `test_init_exports.py`:

    - added lazy-export cache test,
    - added `__all__` integrity check (known symbols + uniqueness),
    - added export-mismatch diagnostic test via monkeypatched import path.
  - `test_model_utils.py`:

    - added package-level lazy-export cache regression for `atlas.models`,
    - added unknown-attribute failure regression,
    - added export-mismatch diagnostic regression via monkeypatched module loader.

## Verification

- `python -m ruff check atlas/training/__init__.py atlas/models/__init__.py tests/unit/training/test_init_exports.py tests/unit/models/test_model_utils.py`
- `python -m py_compile atlas/training/__init__.py atlas/models/__init__.py tests/unit/training/test_init_exports.py tests/unit/models/test_model_utils.py`
- `python -m pytest -q tests/unit/training/test_init_exports.py tests/unit/models/test_model_utils.py`

## Research references used in batch 53

- Python Data Model docs (`module.__getattr__` / module customization): https://docs.python.org/3/reference/datamodel.html#customizing-module-attribute-access
- PEP 562 (module `__getattr__` and `__dir__`): https://peps.python.org/pep-0562/
- Python `importlib` docs (dynamic module import semantics): https://docs.python.org/3/library/importlib.html
- Python `types.MappingProxyType` docs (read-only mapping views): https://docs.python.org/3/library/stdtypes.html#types.MappingProxyType
- Python `__all__` and import system semantics: https://docs.python.org/3/reference/simple_stmts.html#the-import-statement
- PEP 8 public/internal interface guidance (`__all__`): https://peps.python.org/pep-0008/#public-and-internal-interfaces
- Scientific Python SPEC 1 (lazy loading rationale in scientific packages): https://scientific-python.org/specs/spec-0001/

## Progress snapshot (after Batch 53)

- Completed: Batch 1 through Batch 53.
- Pending: Batch 54 onward.

## Batch 54 (max 5 files)

- [X] `atlas/training/metrics.py` - reviewed + optimized
- [X] `tests/unit/training/test_metrics.py` - reviewed + optimized
- [X] `atlas/training/normalizers.py` - reviewed + optimized
- [X] `tests/unit/training/test_normalizers.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 54 optimization goals

- Improve metric API consistency by normalizing metric-key prefixes across scalar/classification/tensor outputs.
- Harden classification metric output stability against non-finite metric-library returns.
- Strengthen normalizer dataset/property contracts to prevent silent misuse (mapping datasets and non-string property names).
- Add regression tests covering the new contracts and deterministic key-prefix behavior.

## Batch 54 outcomes

- `training/metrics.py`:

  - added `_normalize_prefix(...)` to normalize/trim prefixes consistently across all metric groups.
  - added `_safe_float(...)` for classification metric outputs to guard against non-finite library returns.
  - applied normalized-prefix behavior to:

    - `scalar_metrics(...)`
    - `classification_metrics(...)`
    - `tensor_metrics(...)`
- `training/normalizers.py`:

  - `_iter_dataset(...)` now supports mapping-style datasets by iterating over `dataset.values()` instead of keys.
  - `_normalize_properties(...)` now enforces string-only property names (fails fast on invalid types).
  - `MultiTargetNormalizer.load_state_dict(...)` now:

    - iterates properties in sorted order for deterministic behavior,
    - validates each property state is a mapping and raises property-specific errors when malformed.
- Tests:

  - `test_metrics.py`:

    - updated prefix tests to validate trimming/normalization behavior.
    - added classification prefix normalization regression.
  - `test_normalizers.py`:

    - added mapping-dataset support regression for `TargetNormalizer`.
    - added constructor guard for non-string multi-property names.
    - added malformed per-property state rejection test.

## Verification

- `python -m ruff check atlas/training/metrics.py atlas/training/normalizers.py tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py`
- `python -m py_compile atlas/training/metrics.py atlas/training/normalizers.py tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py`
- `python -m pytest -q tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py`

## Research references used in batch 54

- scikit-learn `accuracy_score` API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
- scikit-learn `precision_score` API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
- scikit-learn `recall_score` API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
- scikit-learn `f1_score` API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- scikit-learn `roc_auc_score` API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
- SciPy `spearmanr` API reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
- NumPy `std` reference: https://numpy.org/doc/stable/reference/generated/numpy.std.html
- NumPy `isfinite` reference: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- Python Mapping ABC docs (`collections.abc.Mapping`): https://docs.python.org/3/library/collections.abc.html
- scikit-learn `StandardScaler` reference (normalization semantics): https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

## Progress snapshot (after Batch 54)

- Completed: Batch 1 through Batch 54.
- Pending: Batch 55 onward.

## Batch 55 (max 5 files)

- [X] `atlas/training/losses.py` - reviewed + optimized
- [X] `tests/unit/training/test_losses.py` - reviewed + optimized
- [X] `atlas/training/preflight.py` - reviewed + optimized
- [X] `tests/unit/training/test_preflight.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 55 optimization goals

- Improve multi-task loss configuration safety by rejecting misspelled/unknown task keys in `task_types/task_weights/constraints`.
- Stabilize zero-label/zero-task minibatch behavior so returned total loss remains backward-safe.
- Harden evidential-loss tensor alignment to prevent silent broadcasting bugs under mismatched shapes.
- Improve preflight diagnostics with structured stage-level error reasons (timeout/OS error) for faster CI triage.

## Batch 55 outcomes

- `training/losses.py`:

  - added mapping normalization + key validation helpers:

    - `_normalize_named_mapping(...)`
    - `_validate_known_task_keys(...)`
  - `MultiTaskLoss` now normalizes config-key names and fails fast on unknown tasks in:

    - `task_types`
    - `task_weights`
    - `constraints`
  - `MultiTaskLoss.forward(...)` now initializes `total` with `_zero_loss(...)` so empty-task batches still produce a grad-enabled scalar.
  - `EvidentialLoss.forward(...)` now validates `pred/target` types, flattens and length-aligns tensors across `gamma/nu/alpha/beta/target`, and handles empty aligned windows robustly.
  - `PropertyLoss` BCE path now clamps targets to `[0, 1]` before `binary_cross_entropy_with_logits` to prevent invalid-label blowups.
- `training/preflight.py`:

  - introduced command result model (`_CommandResult`) with `error_reason`.
  - `_run_command(...)` now returns structured outcomes (`return_code`, `error_reason`).
  - added `_format_stage_error(...)` + `_run_stage(...)` helpers to centralize stage execution and failure wrapping.
  - stage failures now emit detailed error messages:

    - `validate-data failed: timeout`
    - `validate-data failed: oserror:FileNotFoundError`
    - `make-splits failed: timeout`
- Tests:

  - `test_losses.py`:

    - added BCE clamping regression,
    - added evidential mismatched-shape alignment regression,
    - added unknown-task key rejection tests for `task_types/task_weights/constraints`,
    - strengthened empty-prediction loss expectation (`requires_grad=True`).
  - `test_preflight.py`:

    - updated existing assertions for new detailed stage error messages,
    - added split-stage timeout regression.

## Verification

- `python -m ruff check atlas/training/losses.py atlas/training/preflight.py tests/unit/training/test_losses.py tests/unit/training/test_preflight.py`
- `python -m py_compile atlas/training/losses.py atlas/training/preflight.py tests/unit/training/test_losses.py tests/unit/training/test_preflight.py`
- `python -m pytest -q tests/unit/training/test_losses.py tests/unit/training/test_preflight.py`

## Research references used in batch 55

- Kendall, Gal, Cipolla (CVPR 2018), uncertainty-based multi-task weighting: https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html
- Amini et al. (NeurIPS 2020), deep evidential regression: https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html
- PyTorch BCE-with-logits API contract: https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
- PyTorch `torch.isfinite` reference: https://pytorch.org/docs/stable/generated/torch.isfinite.html
- Python `subprocess.run` docs (timeout/error behavior): https://docs.python.org/3/library/subprocess.html
- Python `pathlib` docs (`Path` filesystem semantics): https://docs.python.org/3/library/pathlib.html
- Python `dataclasses` docs (structured result objects): https://docs.python.org/3/library/dataclasses.html
- PyTorch `torch.clamp` reference: https://pytorch.org/docs/stable/generated/torch.clamp.html

## Progress snapshot (after Batch 55)

- Completed: Batch 1 through Batch 55.
- Pending: Batch 56 onward.

## Batch 56 (max 5 files)

- [X] `atlas/training/trainer.py` - reviewed + optimized
- [X] `tests/unit/training/test_trainer.py` - reviewed + optimized
- [X] `atlas/training/run_utils.py` - reviewed + optimized
- [X] `tests/unit/training/test_run_utils_manifest.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 56 optimization goals

- Improve checkpoint resume quality by persisting trainer/scheduler/scaler state in a structured payload.
- Harden checkpoint loading with strict payload-shape validation and explicit optional-state restore flags.
- Fail fast on empty train/validation loaders to prevent silent 0-loss runs from contaminating CI signals.
- Strengthen run-directory + run-manifest schema safety (prefix/path validation and manifest root-type validation).

## Batch 56 outcomes

- `training/trainer.py`:

  - added checkpoint schema marker (`schema_version`) and richer resume payload:

    - `optimizer_state_dict`
    - `scheduler_state_dict`
    - `scaler_state_dict`
    - `trainer_state` (`best_val_loss`, `patience_counter`, `history`)
  - added `_validate_checkpoint_payload(...)` to enforce mapping/finite-value contracts before state restore.
  - upgraded `load_checkpoint(...)` with opt-in restoration flags:

    - `restore_optimizer`
    - `restore_scheduler`
    - `restore_scaler`
    - `restore_trainer_state`
    - `strict`
  - switched checkpoint load path to prefer `torch.load(..., weights_only=True)` with compatibility fallback.
  - added explicit empty-loader guards in `train_epoch(...)` and `validate(...)`.
- `test_trainer.py`:

  - added empty-loader rejection tests for both train/validate loops.
  - added checkpoint-resume coverage for persisted trainer state and optimizer restore.
  - added optional-state missing-key regression tests when restore flags are requested.
- `training/run_utils.py`:

  - added `_validate_run_prefix(...)` and wired it through run-dir helpers (`list_run_dirs`, `latest_run_dir`, `resolve_run_dir`, timestamp creation).
  - added `_validate_manifest_payload(...)` to enforce required root fields/types and visibility correctness before write.
  - `write_run_manifest(...)` now validates serialized manifest payload shape before emitting JSON/YAML mirror.
- `test_run_utils_manifest.py`:

  - added invalid-prefix rejection regression.
  - added runtime-context shape regression to ensure invalid runtime payloads fail fast.

## Verification

- `python -m ruff check atlas/training/trainer.py atlas/training/run_utils.py tests/unit/training/test_trainer.py tests/unit/training/test_run_utils_manifest.py`
- `python -m py_compile atlas/training/trainer.py atlas/training/run_utils.py tests/unit/training/test_trainer.py tests/unit/training/test_run_utils_manifest.py`
- `python -m pytest -q tests/unit/training/test_trainer.py tests/unit/training/test_run_utils_manifest.py`

## Research references used in batch 56

- PyTorch serialization notes (`weights_only`, state_dict best practices): https://pytorch.org/docs/stable/notes/serialization.html
- PyTorch save/load tutorial (checkpoint contracts): https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
- PyTorch `torch.load` API reference: https://pytorch.org/docs/stable/generated/torch.load.html
- Python `tempfile.NamedTemporaryFile` docs (safe temp-write patterns): https://docs.python.org/3/library/tempfile.html
- Python `pathlib.Path.replace` docs (atomic replace semantics): https://docs.python.org/3/library/pathlib.html#pathlib.Path.replace
- RFC 8785 (JSON Canonicalization Scheme): https://www.rfc-editor.org/rfc/rfc8785
- NeurIPS ML Reproducibility Checklist (experiment/manifest reporting standards): https://neurips.cc/public/guides/PaperChecklist

## Progress snapshot (after Batch 56)

- Completed: Batch 1 through Batch 56.
- Pending: Batch 57 onward.

## Batch 57 (max 5 files)

- [X] `atlas/utils/reproducibility.py` - reviewed + optimized
- [X] `tests/unit/research/test_reproducibility.py` - reviewed + optimized
- [X] `atlas/research/workflow_reproducible_graph.py` - reviewed + optimized
- [X] `tests/unit/research/test_workflow_reproducible_graph.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 57 optimization goals

- Align runtime determinism toggles with actual environment state to avoid stale deterministic env flags leaking across runs.
- Strengthen workflow manifest durability and strict-JSON compliance (atomic write + non-finite sanitization).
- Prevent run-manifest overwrite risk under same-second run-id collisions.
- Expand reproducibility tests for deterministic-env transitions and workflow manifest edge cases.

## Batch 57 outcomes

- `utils/reproducibility.py`:

  - added `_configure_cublas_workspace(...)` to keep `CUBLAS_WORKSPACE_CONFIG` consistent with deterministic mode:

    - deterministic on: set default `:4096:8` when absent,
    - deterministic off: clear only known deterministic defaults (`:16:8`, `:4096:8`) while preserving custom user values.
  - moved CUBLAS env handling out of torch-specific branch so CPU-only runs also maintain consistent metadata.
  - `set_global_seed(...)` now reports post-configuration CUBLAS state in returned metadata.
- `research/workflow_reproducible_graph.py`:

  - added strict manifest serialization helpers:

    - `_json_safe(...)` (non-finite float -> `None`, path-safe conversion, deterministic set ordering),
    - `_atomic_json_write(...)` (`allow_nan=False`, sorted keys, flush + `fsync`, atomic replace).
  - added stage/method normalization helpers:

    - `_normalize_stage_plan(...)`,
    - `_normalize_fallback_methods(...)` (prevents string payloads from being split char-by-char).
  - hardened `RunManifest.__post_init__` contracts:

    - validates/sanitizes `seed`, `started_at`, `ended_at`, `status`, `stage_plan`, `fallback_methods`.
  - added `_resolve_manifest_path(...)` to avoid filename collisions by suffixing `_01`, `_02`, ... when needed.
  - persistence now validates serialized manifest payload is mapping before write.
- Tests:

  - `test_reproducibility.py`:

    - added regression tests for deterministic-off behavior:

      - known deterministic CUBLAS config is cleared,
      - custom CUBLAS config is preserved.
  - `test_workflow_reproducible_graph.py`:

    - added run-manifest filename-collision test,
    - added non-finite metric sanitization test (`NaN` -> `null`),
    - added invalid finalize status rejection test.

## Verification

- `python -m ruff check atlas/utils/reproducibility.py atlas/research/workflow_reproducible_graph.py tests/unit/research/test_reproducibility.py tests/unit/research/test_workflow_reproducible_graph.py`
- `python -m py_compile atlas/utils/reproducibility.py atlas/research/workflow_reproducible_graph.py tests/unit/research/test_reproducibility.py tests/unit/research/test_workflow_reproducible_graph.py`
- `python -m pytest -q tests/unit/research/test_reproducibility.py tests/unit/research/test_workflow_reproducible_graph.py`

## Research references used in batch 57

- PyTorch Reproducibility notes: https://docs.pytorch.org/docs/stable/notes/randomness.html
- PyTorch deterministic algorithms API (`torch.use_deterministic_algorithms`): https://docs.pytorch.org/docs/2.9/generated/torch.use_deterministic_algorithms.html
- NumPy RNG seeding guidance (`numpy.random.seed`, legacy vs Generator): https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
- NumPy `SeedSequence` (reproducible entropy mixing/spawn): https://numpy.org/doc/2.1/reference/random/bit_generators/generated/numpy.random.SeedSequence.html
- Python `tempfile` docs (`NamedTemporaryFile` semantics): https://docs.python.org/3/library/tempfile.html
- Python `os.replace` docs (atomic replace guarantee on success): https://docs.python.org/3.13/library/os.html#os.replace
- RFC 8259 (JSON, NaN/Infinity not permitted): https://www.rfc-editor.org/rfc/rfc8259
- RFC 8785 (JSON canonicalization / deterministic representation): https://datatracker.ietf.org/doc/html/rfc8785
- Henderson et al., "Deep Reinforcement Learning that Matters" (AAAI 2018): https://arxiv.org/abs/1709.06560
- Pineau et al., "Improving Reproducibility in Machine Learning Research" (JMLR 2021): https://www.jmlr.org/papers/v22/20-303.html

## Progress snapshot (after Batch 57)

- Completed: Batch 1 through Batch 57.
- Pending: Batch 58 onward.

## Batch 58 (max 5 files)

- [X] `atlas/training/physics_losses.py` - reviewed + optimized
- [X] `tests/unit/training/test_physics_losses.py` - reviewed + optimized
- [X] `atlas/thermo/__init__.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_init_exports.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 58 optimization goals

- Eliminate finite-value alignment drift in Voigt/Reuss bound penalties so per-sample pairing remains physically valid.
- Keep physics-only loss terms numerically finite while preserving gradient connectivity for edge cases with all-invalid inputs.
- Harden thermo lazy-export behavior to avoid silently swallowing export-contract mismatches.
- Expand regression coverage for the above stability and API-contract improvements.

## Batch 58 outcomes

- `training/physics_losses.py`:

  - added `_aligned_finite_vectors(...)` for index-aligned finite filtering across `K/G` and Voigt/Reuss bounds.
  - replaced independent flatten+truncate logic with joint finite mask logic in `VoigtReussBoundsLoss.forward(...)` to prevent sample mispairing.
  - added `_zero_loss_from_inputs(...)` + `nan_to_num` sanitation so all-invalid paths return finite zero while preserving grad path.
  - introduced `_BOUND_TOL` constant to centralize bound tolerance.
  - `PhysicsConstraintLoss.forward(...)` now guarantees finite-safe fallback for inactive/invalid constraint paths without breaking autograd.
- `test_physics_losses.py`:

  - added joint-finite-mask regression that catches previous index-misalignment behavior.
  - added all-invalid finite fallback + grad-path regression.
- `thermo/__init__.py`:

  - `__getattr__` now:

    - reuses cached globals when already resolved,
    - raises explicit `AttributeError` if imported module misses expected export symbol,
    - caches successful lazy exports to avoid repeated imports.
  - preserves optional-dependency behavior (`ImportError` -> `None`) for backwards compatibility.
- `test_init_exports.py`:

  - added lazy-export cache regression.
  - added missing-export contract regression (must raise `AttributeError`, no silent `None`).

## Verification

- `python -m ruff check atlas/training/physics_losses.py tests/unit/training/test_physics_losses.py atlas/thermo/__init__.py tests/unit/thermo/test_init_exports.py`
- `python -m py_compile atlas/training/physics_losses.py tests/unit/training/test_physics_losses.py atlas/thermo/__init__.py tests/unit/thermo/test_init_exports.py`
- `python -m pytest -q tests/unit/training/test_physics_losses.py tests/unit/thermo/test_init_exports.py`

## Research references used in batch 58

- Mouhat & Coudert (2014), elastic stability criteria: https://doi.org/10.1103/PhysRevB.90.224104
- Hill (1952), elastic aggregate behavior / Voigt-Reuss-Hill context: https://doi.org/10.1088/0370-1298/65/5/307
- Reuss (1929), iso-stress lower-bound formulation: https://doi.org/10.1002/zamm.19290090104
- Voigt (1889), iso-strain upper-bound formulation: https://doi.org/10.1002/andp.18892741206
- PyTorch `torch.nan_to_num` API (finite replacement semantics): https://docs.pytorch.org/docs/stable/generated/torch.nan_to_num.html
- PEP 562 (module-level `__getattr__` / `__dir__` contract): https://peps.python.org/pep-0562/
- Python `importlib.import_module` docs: https://docs.python.org/3/library/importlib.html

## Progress snapshot (after Batch 58)

- Completed: Batch 1 through Batch 58.
- Pending: Batch 59 onward.

## Batch 59 (max 5 files)

- [X] `atlas/thermo/calphad.py` - reviewed + optimized
- [X] `atlas/thermo/stability.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_calphad.py` - added + optimized
- [X] `tests/unit/thermo/test_stability.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 59 optimization goals

- Stabilize CALPHAD phase-fraction extraction so equilibrium outputs remain physically interpretable under noisy solver outputs.
- Improve transus (liquidus/solidus) estimation robustness via finite filtering + interpolation.
- Strengthen phase-stability API contracts (element normalization, payload validation, non-finite PD result guards).
- Add dedicated thermo unit tests to lock deterministic behavior and regression-proof numeric edge cases.

## Batch 59 outcomes

- `thermo/calphad.py`:

  - added `_normalize_phase_fractions(...)`:

    - filters invalid phase labels/values,
    - merges duplicate phase labels,
    - renormalizes when total fraction exceeds physical bounds,
    - returns deterministic sorted phase map.
  - `equilibrium_at(...)` now routes raw pycalphad fractions through normalized phase-fraction post-processing before returning.
  - upgraded `_find_transus(...)`:

    - finite-mask filtering,
    - clipping to `[0, 1]`,
    - temperature sorting for cooling trajectory consistency,
    - linear interpolation at threshold crossings (`0.99`, `0.01`) for more stable liquidus/solidus estimation.
- `thermo/stability.py`:

  - added `_normalize_element_symbol(...)` and applied case normalization in `get_entries(...)` for case-insensitive chemical-system matching.
  - hardened `ReferenceDatabase` input contracts:

    - non-empty formula requirement in `add_entry(...)`,
    - mapping-type validation in `load_from_list(...)`.
  - `analyze_stability(...)` now:

    - uses reduced target formula as decomposition fallback when decomposition map is empty,
    - rejects non-finite `e_above_hull` / `formation_energy` outputs with explicit error path.
- Tests:

  - new `test_calphad.py`:

    - phase-fraction normalization/renormalization regression,
    - transus interpolation regression,
    - equilibrium result normalization regression via mocked `pycalphad` module.
  - new `test_stability.py`:

    - case-insensitive entry filtering regression,
    - invalid list-item type rejection,
    - decomposition fallback behavior,
    - non-finite phase-diagram output guard behavior.

## Verification

- `python -m ruff check atlas/thermo/calphad.py atlas/thermo/stability.py tests/unit/thermo/test_calphad.py tests/unit/thermo/test_stability.py`
- `python -m py_compile atlas/thermo/calphad.py atlas/thermo/stability.py tests/unit/thermo/test_calphad.py tests/unit/thermo/test_stability.py`
- `python -m pytest -q tests/unit/thermo/test_calphad.py tests/unit/thermo/test_stability.py`

## Research references used in batch 59

- Saunders & Miodownik, *CALPHAD (Calculation of Phase Diagrams): A Comprehensive Guide*: https://www.sciencedirect.com/book/9780080421292/calphad
- Lukas, Fries, Sundman, *Computational Thermodynamics* (CALPHAD methodology): https://doi.org/10.1017/CBO9780511804137
- pycalphad official documentation: https://pycalphad.org/docs/latest/
- pycalphad API docs: https://pycalphad.org/docs/latest/api/
- pymatgen phase diagram docs: https://pymatgen.org/pymatgen.analysis#module-pymatgen.analysis.phase_diagram
- Ong et al. (2013), pymatgen paper: https://doi.org/10.1016/j.commatsci.2012.10.028
- Sun et al. (2016), thermodynamic-scale stability in inorganic crystals: https://doi.org/10.1126/sciadv.1600225
- NumPy interpolation reference (`numpy.interp`/piecewise linear rationale): https://numpy.org/doc/stable/reference/generated/numpy.interp.html
- NumPy finite filtering (`numpy.isfinite`): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- Python dataclasses docs (result schema robustness): https://docs.python.org/3/library/dataclasses.html

## Progress snapshot (after Batch 59)

- Completed: Batch 1 through Batch 59.
- Pending: Batch 60 onward.

## Batch 60 (max 5 files)

- [X] `atlas/thermo/openmm/engine.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/reporters.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/__init__.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_openmm_stack.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 60 optimization goals

- Harden OpenMM fallback LJ runtime against invalid periodic boundary configurations and unstable cutoff choices.
- Strengthen trajectory reporter contracts to prevent silent frame-buffer corruption and shape mismatches.
- Improve openmm package lazy-export behavior with caching and strict export-contract checks.
- Add deterministic tests that do not require real OpenMM installation (via controlled fake-module injection).

## Batch 60 outcomes

- `thermo/openmm/engine.py`:

  - added `_is_periodic(...)` helper and enforced strict boundary condition policy:

    - non-periodic (all false) OR fully periodic (all true),
    - mixed partial-PBC now fails fast with explicit error.
  - improved LJ periodic cutoff logic:

    - validates finite positive cell lengths,
    - enforces cutoff strictly below half minimum box length (minimum-image safety),
    - rejects pathological tiny periodic boxes with actionable error message.
- `thermo/openmm/reporters.py`:

  - `PymatgenTrajectoryReporter` now validates non-empty structure and stores expected site count.
  - `report(...)` now validates position/force array shape against `(n_sites, 3)`.
  - added `_validate_collected_frames(...)` to guard against inconsistent buffer lengths before trajectory export.
  - `describeNextReport(...)` now handles missing/negative `currentStep` defensively.
  - `get_trajectory(...)` now derives `time_step` only from finite positive deltas (non-monotonic/degenerate times -> `0.0`).
- `thermo/openmm/__init__.py`:

  - `__getattr__` now:

    - returns cached export when already resolved,
    - raises explicit `AttributeError` if imported module lacks expected symbol,
    - caches resolved symbols in module globals.
- Tests (`test_openmm_stack.py`):

  - lazy-export cache regression,
  - missing expected export regression,
  - reporter position-shape mismatch guard,
  - reporter inconsistent-buffer guard,
  - engine partial-PBC rejection and periodic LJ cutoff safety regression.
  - tests run fully in CI without OpenMM installation by injecting fake `openmm/openmm.app/openmm.unit` modules.

## Verification

- `python -m ruff check atlas/thermo/openmm/engine.py atlas/thermo/openmm/reporters.py atlas/thermo/openmm/__init__.py tests/unit/thermo/test_openmm_stack.py`
- `python -m py_compile atlas/thermo/openmm/engine.py atlas/thermo/openmm/reporters.py atlas/thermo/openmm/__init__.py tests/unit/thermo/test_openmm_stack.py`
- `python -m pytest -q tests/unit/thermo/test_openmm_stack.py`

## Research references used in batch 60

- OpenMM User Guide (periodic boundaries / simulation model): https://docs.openmm.org/latest/userguide/
- OpenMM API docs (`NonbondedForce`, cutoff methods): https://docs.openmm.org/latest/api-python/generated/openmm.openmm.NonbondedForce.html
- OpenMM API docs (`Simulation`): https://docs.openmm.org/latest/api-python/generated/openmm.app.simulation.Simulation.html
- Allen & Tildesley, *Computer Simulation of Liquids* (minimum image / cutoff constraints): https://global.oup.com/academic/product/computer-simulation-of-liquids-9780198803201
- Frenkel & Smit, *Understanding Molecular Simulation* (PBC and short-range interactions): https://doi.org/10.1016/B978-0-12-267351-1.X5000-7
- pymatgen trajectory docs: https://pymatgen.org/pymatgen.core.html#module-pymatgen.core.trajectory
- PEP 562 (module lazy attribute hooks): https://peps.python.org/pep-0562/
- Python `importlib` docs: https://docs.python.org/3/library/importlib.html

## Progress snapshot (after Batch 60)

- Completed: Batch 1 through Batch 60.
- Pending: Batch 61 onward.

## Batch 61 (max 5 files)

- [X] `atlas/thermo/openmm/atomate2_wrapper.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/__init__.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_openmm_atomate2_wrapper.py` - added + optimized
- [X] `tests/unit/thermo/test_openmm_stack.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 61 optimization goals

- Harden native Atomate2 OpenMM wrapper contracts for maker selection, step validation, and optional module loading.
- Avoid repeated dynamic-import overhead in wrapper hot paths by caching imported jobs module.
- Tighten OpenMM lazy-export error policy to only degrade on optional dependency absence (ImportError), not internal runtime faults.
- Add deterministic unit tests to lock these behaviors without requiring full OpenMM runtime dependencies.

## Batch 61 outcomes

- `thermo/openmm/atomate2_wrapper.py`:

  - removed fragile eager import block and switched to explicit on-demand loading via `_load_atomate2_jobs_module(...)`.
  - added `_ATOMATE2_JOBS_MODULE` cache so repeated wrapper calls do not repeatedly import job module.
  - added `_coerce_non_negative_int(...)` for robust integer-step parsing and bool rejection.
  - strengthened maker construction semantics:

    - `nvt/npt` require `steps > 0`,
    - `minimize` still supports `steps=0`.
  - `run_simulation(...)` now validates maker interface contract (`callable make(...)`) before invocation.
- `thermo/openmm/__init__.py`:

  - lazy export now only swallows `ImportError` as optional dependency path.
  - runtime import faults (e.g. module init bug) are no longer silently downgraded to `None`.
  - retains symbol caching + explicit missing-export `AttributeError` behavior from previous hardening.
- Tests:

  - new `test_openmm_atomate2_wrapper.py`:

    - init argument guards,
    - dynamic-ensemble step contracts,
    - module-import caching behavior,
    - import-error wrapping behavior,
    - maker interface validation,
    - successful maker invocation path.
  - updated `test_openmm_stack.py`:

    - added regression test that runtime import errors are propagated (not suppressed) in lazy export path.

## Verification

- `python -m ruff check atlas/thermo/openmm/atomate2_wrapper.py atlas/thermo/openmm/__init__.py tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`
- `python -m py_compile atlas/thermo/openmm/atomate2_wrapper.py atlas/thermo/openmm/__init__.py tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`
- `python -m pytest -q tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`

## Research references used in batch 61

- OpenMM User Guide (simulation stack and API behavior): https://docs.openmm.org/latest/userguide/
- OpenMM Python API docs (`Simulation` / app layer): https://docs.openmm.org/latest/api-python/generated/openmm.app.simulation.Simulation.html
- OpenMM Python API docs (`LangevinMiddleIntegrator`): https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LangevinMiddleIntegrator.html
- atomate2 OpenMM docs (maker/workflow integration context): https://materialsproject.github.io/atomate2/
- Python import system docs (`importlib.import_module`): https://docs.python.org/3/library/importlib.html
- PEP 562 (module `__getattr__` lazy export semantics): https://peps.python.org/pep-0562/
- Python logging best practices (standard library docs): https://docs.python.org/3/library/logging.html

## Progress snapshot (after Batch 61)

- Completed: Batch 1 through Batch 61.
- Pending: Batch 62 onward.

## Batch 62 (max 5 files)

- [X] `atlas/thermo/openmm/atomate2_wrapper.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/__init__.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_openmm_atomate2_wrapper.py` - added + optimized
- [X] `tests/unit/thermo/test_openmm_stack.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 62 optimization goals

- Strengthen native Atomate2 OpenMM wrapper contracts (typed step parsing, maker selection, module caching, interface validation).
- Prevent silent masking of non-ImportError module failures in OpenMM lazy exports.
- Add deterministic tests around optional-import paths and wrapper execution contracts.
- Keep behavior backward-compatible for optional dependency absence while surfacing true runtime faults.

## Batch 62 outcomes

- `thermo/openmm/atomate2_wrapper.py`:

  - added module-level cache (`_ATOMATE2_JOBS_MODULE`) for dynamic import reuse.
  - added strict step parsing helper (`_coerce_non_negative_int`) that rejects bool payloads.
  - `_build_maker(...)` now enforces:

    - `nvt`/`npt` require `steps > 0`,
    - `minimize` supports `steps=0`.
  - `run_simulation(...)` now verifies maker exposes callable `make(...)` before invocation and returns through validated callable reference.
- `thermo/openmm/__init__.py`:

  - lazy export path now catches only `ImportError` as optional-dependency fallback.
  - runtime import failures (e.g., module init `RuntimeError`) are now propagated instead of being silently converted to `None`.
- Tests:

  - new `test_openmm_atomate2_wrapper.py` with coverage for:

    - init argument validation,
    - ensemble/steps contract checks,
    - module import caching,
    - import-error wrapping,
    - maker interface contract,
    - successful maker invocation payload.
  - updated `test_openmm_stack.py` with regression ensuring runtime import errors are not swallowed in lazy export path.

## Verification

- `python -m ruff check atlas/thermo/openmm/atomate2_wrapper.py atlas/thermo/openmm/__init__.py tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`
- `python -m py_compile atlas/thermo/openmm/atomate2_wrapper.py atlas/thermo/openmm/__init__.py tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`
- `python -m pytest -q tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`

## Research references used in batch 62

- atomate2 OpenMM install + architecture notes: https://materialsproject.github.io/atomate2/user/codes/openmm.html
- atomate2 OpenMM tutorial (NVTMaker/NPTMaker usage): https://materialsproject.github.io/atomate2/tutorials/openmm_tutorial.html
- atomate2 docs home (2026 docs index): https://materialsproject.github.io/atomate2/
- OpenMM Simulation API (latest): https://docs.openmm.org/latest/api-python/generated/openmm.app.simulation.Simulation.html
- OpenMM NonbondedForce API (latest): https://docs.openmm.org/latest/api-python/generated/openmm.openmm.NonbondedForce.html
- OpenMM app layer reference: https://docs.openmm.org/latest/api-python/app.html
- Python `importlib` docs (`import_module`): https://docs.python.org/3/library/importlib.html
- PEP 562 (module `__getattr__` lazy exports): https://peps.python.org/pep-0562/
- Python logging docs: https://docs.python.org/3/library/logging.html

## Progress snapshot (after Batch 62)

- Completed: Batch 1 through Batch 62.
- Pending: Batch 63 onward.

## Batch 63 (max 5 files)

- [X] `atlas/thermo/openmm/atomate2_wrapper.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/__init__.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_openmm_atomate2_wrapper.py` - added + optimized
- [X] `tests/unit/thermo/test_openmm_stack.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 63 optimization goals

- Enforce strict runtime contracts for native Atomate2 OpenMM wrapper (step typing, maker interface, and mode-specific constraints).
- Improve lazy import reliability so optional dependency fallback handles only true optional-missing paths.
- Add deterministic CI-safe tests for import-cache behavior, import failure modes, and maker-call contracts.

## Batch 63 outcomes

- `thermo/openmm/atomate2_wrapper.py`:

  - added module-level cache (`_ATOMATE2_JOBS_MODULE`) for atomate2 jobs import reuse.
  - added `_coerce_non_negative_int(...)` to reject invalid step payloads (including bool) and normalize integer steps.
  - tightened ensemble semantics:

    - `nvt` and `npt` require strictly positive `steps`,
    - `minimize` remains valid with `steps=0`.
  - `run_simulation(...)` now validates maker exposes callable `make(...)` before invocation.
  - optional dependency loading now wraps only `ImportError` into a clear runtime message.
- `thermo/openmm/__init__.py`:

  - lazy export fallback now catches only `ImportError` for optional dependency absence.
  - runtime faults during module import are now propagated, improving diagnosability.
- Tests:

  - new `test_openmm_atomate2_wrapper.py` covers input validation, module-cache behavior, import-error wrapping, maker-interface checks, and positive execution path.
  - `test_openmm_stack.py` adds regression to ensure non-ImportError lazy import failures are not suppressed.

## Verification

- `python -m ruff check atlas/thermo/openmm/atomate2_wrapper.py atlas/thermo/openmm/__init__.py tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`
- `python -m py_compile atlas/thermo/openmm/atomate2_wrapper.py atlas/thermo/openmm/__init__.py tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`
- `python -m pytest -q tests/unit/thermo/test_openmm_atomate2_wrapper.py tests/unit/thermo/test_openmm_stack.py`

## Research references used in batch 63

- OpenMM paper (method + software architecture): https://doi.org/10.1371/journal.pcbi.1005659
- OpenMM User Guide: https://docs.openmm.org/latest/userguide/
- OpenMM Python API (`Simulation`): https://docs.openmm.org/latest/api-python/generated/openmm.app.simulation.Simulation.html
- atomate2 OpenMM user docs: https://materialsproject.github.io/atomate2/user/codes/openmm.html
- atomate2 OpenMM tutorial (`NVTMaker` / `NPTMaker`): https://materialsproject.github.io/atomate2/tutorials/openmm_tutorial.html
- Python `importlib` docs (`import_module`): https://docs.python.org/3/library/importlib.html
- PEP 562 (`module.__getattr__` lazy export semantics): https://peps.python.org/pep-0562/

## Progress snapshot (after Batch 63)

- Completed: Batch 1 through Batch 63.
- Pending: Batch 64 onward.

## Batch 64 (max 5 files)

- [X] `atlas/thermo/openmm/engine.py` - reviewed + optimized
- [X] `atlas/thermo/openmm/reporters.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_openmm_stack.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_openmm_reporters.py` - added + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 64 optimization goals

- Eliminate implicit numeric truncation risks in OpenMM runtime controls (`steps`, `trajectory_interval`).
- Fix trajectory metadata unit correctness (`time_step` in pymatgen requires femtoseconds).
- Add deterministic regression tests for integer contract enforcement and reporter time-axis behavior.
- Keep optional dependency fallback and testability stable for CI environments without full OpenMM stack.

## Batch 64 outcomes

- `thermo/openmm/engine.py`:

  - added `_coerce_positive_int(...)` to enforce strict positive-integer inputs and reject bool/non-integral floats.
  - `run(...)` now uses strict validation for both `steps` and `trajectory_interval` (no silent truncation).
  - introduced high-precision `_KJ_MOL_PER_EV` conversion constant to remove magic-number drift.
  - normalized `forcefield_path` handling (`strip/lower`) and made unsupported custom path fallback explicit in logs.
- `thermo/openmm/reporters.py`:

  - added monotonic-time guard during reporting (`time_ps` must be non-decreasing).
  - strengthened frame validation with finite and non-decreasing time-axis checks.
  - corrected `Trajectory.time_step` unit from ps to fs (`* 1000.0`) to match pymatgen contract.
  - switched `zip(..., strict=True)` for frame property assembly after buffer-length validation.
- Tests:

  - `test_openmm_stack.py`:

    - added regression that `engine.run(...)` rejects non-integral `steps`/`trajectory_interval`.
  - new `test_openmm_reporters.py`:

    - verifies `time_step` conversion from ps to fs,
    - verifies non-monotonic simulation time is rejected.

## Verification

- `python -m ruff check atlas/thermo/openmm/engine.py atlas/thermo/openmm/reporters.py tests/unit/thermo/test_openmm_stack.py tests/unit/thermo/test_openmm_reporters.py`
- `python -m py_compile atlas/thermo/openmm/engine.py atlas/thermo/openmm/reporters.py tests/unit/thermo/test_openmm_stack.py tests/unit/thermo/test_openmm_reporters.py`
- `python -m pytest -q tests/unit/thermo/test_openmm_stack.py tests/unit/thermo/test_openmm_reporters.py`

## Research references used in batch 64

- OpenMM `Simulation.step(steps: int)` API: https://docs.openmm.org/latest/api-python/generated/openmm.app.simulation.Simulation.html
- OpenMM reporter contract (`describeNextReport`): https://docs.openmm.org/development/api-python/generated/openmm.app.pdbreporter.PDBReporter.html
- OpenMM architecture paper: https://doi.org/10.1371/journal.pcbi.1005659
- pymatgen trajectory API (`time_step` in femtoseconds): https://pymatgen.org/pymatgen.core.html#pymatgen.core.trajectory.Trajectory
- Python stdtypes (`bool` is a subclass of `int`): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool

## Progress snapshot (after Batch 64)

- Completed: Batch 1 through Batch 64.
- Pending: Batch 65 onward.

## Batch 65 (max 5 files)

- [X] `atlas/thermo/calphad.py` - reviewed + optimized
- [X] `atlas/thermo/stability.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_calphad.py` - reviewed + optimized
- [X] `tests/unit/thermo/test_stability.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 65 optimization goals

- Remove silent numeric coercion risk in CALPHAD path generation (`n_steps`) by enforcing strict integer contracts.
- Improve phase-fraction normalization robustness so output is consistently probability-like for downstream comparison/ranking.
- Harden phase stability decomposition serialization against invalid/non-finite decomposition coefficients.
- Expand unit tests to lock new numerical/contract behavior and keep backward compatibility for existing outputs.

## Batch 65 outcomes

- `thermo/calphad.py`:

  - added `_coerce_int_with_min(...)` to reject bool/non-integral float step values.
  - `solidification_path(...)` now enforces strict integer `n_steps >= 2` (no silent truncation).
  - added `_canonical_phase_name(...)` and normalized phase labels to canonical uppercase form.
  - `_normalize_phase_fractions(...)` now always renormalizes surviving positive fractions to sum to 1.0.
- `thermo/stability.py`:

  - introduced `_STABLE_EHULL_EPS` constant for explicit stability threshold governance.
  - added decomposition-map sanitization: non-finite/non-positive coefficients are ignored.
  - when decomposition coefficients are all invalid, fallback decomposition text is target reduced formula.
- Tests:

  - `test_calphad.py`:

    - verifies case-folded phase merge + renormalization,
    - verifies subunit phase totals are renormalized to 1.0,
    - verifies `solidification_path(...)` rejects non-integral step values.
  - `test_stability.py`:

    - verifies non-finite decomposition coefficients do not leak to output formatting,
    - verifies non-finite target energy is rejected early.

## Verification

- `python -m ruff check atlas/thermo/calphad.py atlas/thermo/stability.py tests/unit/thermo/test_calphad.py tests/unit/thermo/test_stability.py`
- `python -m py_compile atlas/thermo/calphad.py atlas/thermo/stability.py tests/unit/thermo/test_calphad.py tests/unit/thermo/test_stability.py`
- `python -m pytest -q tests/unit/thermo/test_calphad.py tests/unit/thermo/test_stability.py`

## Research references used in batch 65

- pycalphad docs (API and equilibrium modeling): https://pycalphad.org/docs/latest/
- pycalphad example gallery (equilibrium workflows): https://pycalphad.org/docs/latest/examples/index.html
- pycalphad-scheil package docs: https://scheil.readthedocs.io/en/latest/
- pycalphad-scheil implementation repository: https://github.com/pycalphad/scheil
- Bocklund et al., *pycalphad: CALPHAD-based computational thermodynamics in Python* (JORS, 2019): https://doi.org/10.5334/jors.140
- pymatgen phase diagram API docs (`PhaseDiagram`, `get_e_above_hull`): https://pymatgen.org/pymatgen.analysis#module-pymatgen.analysis.phase_diagram
- pymatgen project docs root (reference implementation context): https://pymatgen.org/

## Progress snapshot (after Batch 65)

- Completed: Batch 1 through Batch 65.
- Pending: Batch 66 onward.

## Batch 66 (max 5 files)

- [X] `scripts/phase5_active_learning/test_enumeration.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/structure_enumerator.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_test_enumeration_script.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_structure_enumerator_script.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 66 optimization goals

- Make the enumeration demo CLI machine-friendly (`--json` output must not be mixed with verbose text).
- Remove silent numeric coercion in fallback enumerator controls (`max_index`) to avoid hidden behavior drift.
- Harden substitution normalization and ordinal decoding contracts for deterministic fallback enumeration.
- Expand tests to lock CLI behavior, substitution-key normalization, and enumeration input contracts.

## Batch 66 outcomes

- `scripts/phase5_active_learning/test_enumeration.py`:

  - split parser creation into `_build_parser()` and upgraded `main(argv=None)` for testable CLI execution.
  - fixed JSON mode behavior: `--json` now forces non-verbose execution for parseable single-object output.
  - error path now writes to `stderr` and returns exit code `2` consistently.
- `scripts/phase5_active_learning/structure_enumerator.py`:

  - added strict `_coerce_positive_int(...)` and applied to `max_index` (rejects bool/non-integral float).
  - strengthened `_decode_variant_ordinal(...)` with explicit ordinal-range validation.
  - `_normalize_substitutions(...)` now merges semantically identical keys after whitespace normalization.
  - switched site substitution loop to `zip(..., strict=True)` for iterator contract safety.
- Tests:

  - `test_test_enumeration_script.py`:

    - added regression for JSON mode suppressing verbose output,
    - added regression that CLI errors go to `stderr`.
  - `test_structure_enumerator_script.py`:

    - added non-integral `max_index` rejection regression,
    - added normalized duplicate-key merge regression,
    - added out-of-range ordinal decode regression.

## Verification

- `python -m ruff check scripts/phase5_active_learning/test_enumeration.py scripts/phase5_active_learning/structure_enumerator.py tests/unit/active_learning/test_test_enumeration_script.py tests/unit/active_learning/test_structure_enumerator_script.py`
- `python -m py_compile scripts/phase5_active_learning/test_enumeration.py scripts/phase5_active_learning/structure_enumerator.py tests/unit/active_learning/test_test_enumeration_script.py tests/unit/active_learning/test_structure_enumerator_script.py`
- `python -m pytest -q tests/unit/active_learning/test_test_enumeration_script.py tests/unit/active_learning/test_structure_enumerator_script.py`

## Research references used in batch 66

- Python `argparse` reference (`parse_args`, CLI design): https://docs.python.org/3/library/argparse.html
- Python boolean type semantics (`bool` is `int` subclass): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool
- Python iterator `zip(..., strict=True)` semantics: https://docs.python.org/3/library/functions.html#zip
- pymatgen `StructureMatcher` API docs: https://pymatgen.org/pymatgen.analysis.html#pymatgen.analysis.structure_matcher.StructureMatcher
- pymatgen `DummySpecies` API docs: https://pymatgen.org/pymatgen.core.html#pymatgen.core.periodic_table.DummySpecies
- Ong et al., *Python Materials Genomics (pymatgen)*: https://www.sciencedirect.com/science/article/pii/S0927025612006295

## Progress snapshot (after Batch 66)

- Completed: Batch 1 through Batch 66.
- Pending: Batch 67 onward.

## Batch 67 (max 5 files)

- [X] `scripts/phase5_active_learning/search_materials.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/active_learning.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_search_materials_cli.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_phase6_active_learning.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 67 optimization goals

- Make search CLI validation stricter and more testable (especially integer contracts and conflicting bounds).
- Ensure CLI error channels are automation-friendly (`stderr` for invalid input paths).
- Prevent active-learning helper APIs from silently coercing invalid integer-like controls.
- Avoid hidden state carry-over across repeated active-learning runs.

## Batch 67 outcomes

- `scripts/phase5_active_learning/search_materials.py`:

  - added `_coerce_positive_int(...)` and strict `--max` validation (rejects bool/non-integral inputs).
  - introduced `_validate_criteria_bounds(...)` and applied it to built criteria; contradictory merged bounds now fail fast.
  - refactored CLI entry into `_build_parser()` + `main(argv=None)` for deterministic unit testing.
  - CLI validation and query-column errors now print to `stderr` with exit code `2`.
  - added `--desc` alias while preserving legacy `-desc`.
  - replaced global `pd.set_option(...)` mutations with scoped `pd.option_context(...)`.
- `scripts/phase5_active_learning/active_learning.py`:

  - tightened integer contracts for batch sizes, budgets, iteration counts, `n_samples`, and `top_k`.
  - `ActiveLearningLoop.run(...)` now resets history at run start to avoid stale-run contamination.
- Tests:

  - `test_search_materials_cli.py`:

    - added strict `--max` validation regression,
    - added inconsistent-bound merge regression,
    - added `main(...)` error-to-`stderr` regression.
  - `test_phase6_active_learning.py`:

    - added non-integral batch-size regression,
    - added non-integral AL budget regression,
    - added empty-dataset run-history reset regression,
    - added non-integral `top_k` regression.

## Verification

- `python -m ruff check scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/active_learning.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_phase6_active_learning.py`
- `python -m py_compile scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/active_learning.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_phase6_active_learning.py`
- `python -m pytest -q tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_phase6_active_learning.py`

## Research references used in batch 67

- Python `argparse` docs (`parse_args`, parser patterns): https://docs.python.org/3/library/argparse.html
- Python numeric type semantics (`bool` as `int` subclass): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool
- pandas `option_context` (scoped display settings): https://pandas.pydata.org/docs/reference/api/pandas.option_context.html
- NumPy random Generator API (`choice`, reproducibility): https://numpy.org/doc/stable/reference/random/generator.html
- Gal & Ghahramani, *Dropout as a Bayesian Approximation* (MC-dropout UQ basis): https://proceedings.mlr.press/v48/gal16.html
- Jensen, *Introduction to Pareto optimality* (multi-objective optimization background): https://doi.org/10.1007/978-0-387-74759-0_493

## Progress snapshot (after Batch 67)

- Completed: Batch 1 through Batch 67.
- Pending: Batch 68 onward.

## Batch 68 (max 5 files)

- [X] `scripts/phase5_active_learning/run_phase5.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_discovery.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_phase5_cli.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_controller_runtime_stability.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 68 optimization goals

- Strengthen CLI numeric contracts in Phase5 launchers to avoid silent coercion bugs (`bool` / non-integral floats).
- Improve script entrypoint testability via parser extraction and `main(argv=None)` patterns.
- Route validation/runtime errors to `stderr` for better CI/automation behavior.
- Add regression tests for strict integer validation and runtime backoff jitter behavior.

## Batch 68 outcomes

- `scripts/phase5_active_learning/run_phase5.py`:

  - added `_coerce_int_with_bounds(...)` and applied strict integer checks to launcher integer controls.
  - refactored parser creation into `_build_parser()` and converted entrypoint to `main(argv=None)`.
  - validation/preflight error messages now emit to `stderr`.
- `scripts/phase5_active_learning/run_discovery.py`:

  - added strict integer coercion helper for discovery controls (`iterations`, `candidates`, `top`, `seeds`, `calibration_window`).
  - refactored parser creation into `_build_parser()` and converted entrypoint to `main(argv=None)`.
  - validation and run-directory resolution errors now emit to `stderr`.
- `tests/unit/active_learning/test_phase5_cli.py`:

  - added regressions for non-integral integer controls in both phase5 and discovery validators.
  - added regressions asserting `main(...)` validation failures are written to `stderr`.
- `tests/unit/active_learning/test_controller_runtime_stability.py`:

  - added deterministic jitter/backoff regression for `_retry_sleep_seconds(...)` range behavior with injected RNG.

## Verification

- `python -m ruff check scripts/phase5_active_learning/run_phase5.py scripts/phase5_active_learning/run_discovery.py tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_controller_runtime_stability.py`
- `python -m py_compile scripts/phase5_active_learning/run_phase5.py scripts/phase5_active_learning/run_discovery.py tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_controller_runtime_stability.py`
- `python -m pytest -q tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_controller_runtime_stability.py`

## Research references used in batch 68

- Python `argparse` reference: https://docs.python.org/3/library/argparse.html
- Python `subprocess` reference: https://docs.python.org/3/library/subprocess.html
- PEP 389 (`argparse` design rationale): https://peps.python.org/pep-0389/
- OWASP Input Validation Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html
- CWE-20 Improper Input Validation: https://cwe.mitre.org/data/definitions/20.html
- Dropout as a Bayesian Approximation (UQ context for AL pipelines): https://proceedings.mlr.press/v48/gal16.html
- Settles, Active Learning Literature Survey: https://burrsettles.com/pub/settles.activelearning.pdf

## Progress snapshot (after Batch 68)

- Completed: Batch 1 through Batch 68.
- Pending: Batch 69 onward.

## Batch 69 (max 5 files)

- [X] `atlas/benchmark/cli.py` - reviewed + optimized
- [X] `atlas/benchmark/runner.py` - reviewed + optimized
- [X] `tests/unit/benchmark/test_benchmark_cli.py` - reviewed + optimized
- [X] `tests/unit/benchmark/test_benchmark_runner.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 69 optimization goals

- Tighten benchmark CLI integer contracts to prevent bool/non-integral values from slipping through argument validation.
- Improve benchmark preflight failure signaling for automation (stderr instead of stdout).
- Remove silent float truncation in benchmark runner integer coercion helpers.
- Add regression tests to lock strict validation and sanitization behavior.

## Batch 69 outcomes

- `atlas/benchmark/cli.py`:

  - added `_coerce_int_with_min(...)` for strict integer validation.
  - `_validate_cli_args(...)` now normalizes/validates integer controls (`batch_size`, `jobs`, bootstrap and preflight controls) with explicit error semantics.
  - fold validation now enforces integer entries and non-negative bounds through the same helper.
  - preflight failure path now writes errors to `stderr`.
- `atlas/benchmark/runner.py`:

  - hardened `_coerce_positive_int(...)` and `_coerce_int(...)`:

    - bool is no longer treated as valid integer input,
    - non-integral real values no longer silently truncate via `int(...)`.
  - preserves existing fallback-to-default behavior while removing hidden truncation.
- Tests:

  - `test_benchmark_cli.py`:

    - added non-integral integer-control validation regressions,
    - added preflight failure stderr regression.
  - `test_benchmark_runner.py`:

    - added regression proving non-integral/bool runtime controls are sanitized to conservative defaults without truncation.

## Verification

- `python -m ruff check atlas/benchmark/cli.py atlas/benchmark/runner.py tests/unit/benchmark/test_benchmark_cli.py tests/unit/benchmark/test_benchmark_runner.py`
- `python -m py_compile atlas/benchmark/cli.py atlas/benchmark/runner.py tests/unit/benchmark/test_benchmark_cli.py tests/unit/benchmark/test_benchmark_runner.py`
- `python -m pytest -q tests/unit/benchmark/test_benchmark_cli.py tests/unit/benchmark/test_benchmark_runner.py`

## Research references used in batch 69

- Python `argparse` documentation: https://docs.python.org/3/library/argparse.html
- Python type semantics (`bool` subclassing `int`): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool
- Matbench benchmark paper (Dunn et al., 2020): https://doi.org/10.1038/s41524-020-00406-3
- Matbench official documentation: https://hackingmaterials.lbl.gov/matbench/
- Kuleshov et al., calibrated regression uncertainty: https://proceedings.mlr.press/v80/kuleshov18a.html
- Angelopoulos & Bates, conformal prediction tutorial: https://arxiv.org/abs/2107.07511
- Efron & Tibshirani, bootstrap methodology: https://doi.org/10.1007/978-1-4899-4541-9

## Progress snapshot (after Batch 69)

- Completed: Batch 1 through Batch 69.
- Pending: Batch 70 onward.

## Batch 70 (max 5 files)

- [X] `atlas/benchmark/cli.py` - reviewed + optimized
- [X] `atlas/benchmark/runner.py` - reviewed + optimized
- [X] `tests/unit/benchmark/test_benchmark_cli.py` - reviewed + optimized
- [X] `tests/unit/benchmark/test_benchmark_runner.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 70 optimization goals

- Strengthen benchmark CLI integer validation semantics to reject non-integral controls consistently.
- Prevent bool/non-integral numeric inputs from silently truncating in benchmark runner parameter sanitization.
- Improve automation friendliness by routing benchmark preflight failures to stderr.
- Add regression tests to lock these contracts and avoid drift.

## Batch 70 outcomes

- `atlas/benchmark/cli.py`:

  - added `_coerce_int_with_min(...)` and integrated it into `_validate_cli_args(...)`.
  - integer controls (`batch-size`, `jobs`, bootstrap/preflight integer args, and fold entries) are now normalized with strict integer checks.
  - preflight-failure error line now emits to `stderr`.
- `atlas/benchmark/runner.py`:

  - hardened `_coerce_positive_int(...)` and `_coerce_int(...)`:

    - bool values no longer pass as valid integers,
    - non-integral real values no longer get silently truncated by `int(...)`.
  - fallback-to-default behavior remains unchanged for invalid inputs, but now explicit and safer.
- `tests/unit/benchmark/test_benchmark_cli.py`:

  - added non-integral integer-control rejection regressions.
  - added preflight-failure stderr regression.
- `tests/unit/benchmark/test_benchmark_runner.py`:

  - added regression verifying non-integral/bool runner params sanitize to defaults rather than truncating.

## Verification

- `python -m ruff check atlas/benchmark/cli.py atlas/benchmark/runner.py tests/unit/benchmark/test_benchmark_cli.py tests/unit/benchmark/test_benchmark_runner.py`
- `python -m py_compile atlas/benchmark/cli.py atlas/benchmark/runner.py tests/unit/benchmark/test_benchmark_cli.py tests/unit/benchmark/test_benchmark_runner.py`
- `python -m pytest -q tests/unit/benchmark/test_benchmark_cli.py tests/unit/benchmark/test_benchmark_runner.py`

## Research references used in batch 70

- Python argparse docs (official): https://docs.python.org/3.12/library/argparse.html
- Python bool/int semantics (official): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool
- Matbench documentation: https://hackingmaterials.lbl.gov/automatminer/datasets.html
- Matbench paper (Nature Computational Materials): https://doi.org/10.1038/s41524-020-00406-3
- Kuleshov et al. calibrated regression (PMLR 2018): https://proceedings.mlr.press/v80/kuleshov18a.html
- Angelopoulos & Bates conformal tutorial (arXiv): https://arxiv.org/abs/2107.07511
- Efron & Tibshirani bootstrap reference: https://doi.org/10.1007/978-1-4899-4541-9

## Progress snapshot (after Batch 70)

- Completed: Batch 1 through Batch 70.
- Pending: Batch 71 onward.

## Batch 71 (max 5 files)

- [X] `atlas/benchmark/__init__.py` - reviewed + optimized
- [X] `atlas/benchmark/runner.py` - reviewed + optimized
- [X] `tests/unit/benchmark/test_benchmark_init.py` - reviewed + optimized (new)
- [X] `tests/unit/benchmark/test_benchmark_runner.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 71 optimization goals

- Strengthen benchmark package lazy-export contracts so symbol resolution failures are explicit and testable.
- Eliminate silent fold-index coercion in benchmark runner (non-integral/negative fold IDs must fail fast).
- Harden probability sanitization against bool semantics and non-finite values.
- Lock these contracts with focused unit regressions.

## Batch 71 outcomes

- `atlas/benchmark/__init__.py`:

  - switched to explicit lazy export table (`_EXPORTS`) with `__getattr__` + `__dir__`.
  - added cache short-circuit (`if name in globals()`) for stable lazy resolution semantics.
  - added explicit missing-attribute check after module import with informative `AttributeError`.
- `atlas/benchmark/runner.py`:

  - `_coerce_probability(...)` now rejects bool semantics and falls back to defaults.
  - added `_coerce_non_negative_fold_id(...)` and used it in `run_task(...)` fold normalization.
  - non-integral fold IDs (e.g., `0.5`) and negative IDs now fail fast with clear `ValueError` instead of implicit truncation.
- `tests/unit/benchmark/test_benchmark_init.py` (new):

  - verifies expected exports in `dir(...)`, unknown attribute behavior, lazy caching behavior, and explicit error path when expected export is missing.
- `tests/unit/benchmark/test_benchmark_runner.py`:

  - added regressions for bool probability controls fallback.
  - added regressions for non-integral and negative fold ID rejection.

## Verification

- `python -m ruff check atlas/benchmark/__init__.py atlas/benchmark/runner.py tests/unit/benchmark/test_benchmark_init.py tests/unit/benchmark/test_benchmark_runner.py`
- `python -m py_compile atlas/benchmark/__init__.py atlas/benchmark/runner.py tests/unit/benchmark/test_benchmark_init.py tests/unit/benchmark/test_benchmark_runner.py`
- `python -m pytest -q tests/unit/benchmark/test_benchmark_init.py tests/unit/benchmark/test_benchmark_runner.py`

## Research references used in batch 71

- PEP 562 (`module __getattr__`, `__dir__`): https://peps.python.org/pep-0562/
- Python bool/int semantics (official): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool
- Matbench benchmark paper: https://doi.org/10.1038/s41524-020-00406-3
- Matbench official documentation: https://hackingmaterials.lbl.gov/matbench/
- Kuleshov et al. calibrated regression uncertainty: https://proceedings.mlr.press/v80/kuleshov18a.html
- Angelopoulos & Bates conformal prediction tutorial: https://arxiv.org/abs/2107.07511

## Progress snapshot (after Batch 71)

- Completed: Batch 1 through Batch 71.
- Pending: Batch 72 onward.

## Batch 72 (max 5 files)

- [X] `scripts/phase5_active_learning/test_enumeration.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/structure_enumerator.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/search_materials.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_structure_enumerator_script.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 72 optimization goals

- Remove script-level import side effects in enumeration demo (`sys.path` mutation) while preserving local fallback execution.
- Improve fallback enumerator truncation sampling coverage so reduced variant sets better span the combinatorial space.
- Reuse shared variant-space validation logic to reduce duplicated radix checks and avoid drift.
- Add runtime safety guard for large result requests in multi-property search CLI.

## Batch 72 outcomes

- `scripts/phase5_active_learning/test_enumeration.py`:

  - removed global `sys.path.insert(...)` side effect.
  - added `importlib.util`-based local fallback loader (`_load_local_enumerator_class`) for direct script execution without package path pollution.
  - tightened fallback error semantics (explicit `ImportError`/`TypeError` path).
  - replaced `hasattr(cls, "__call__")` with `callable(cls)` for clearer contract checking.
- `scripts/phase5_active_learning/structure_enumerator.py`:

  - upgraded `_select_variant_ordinals(...)` to deterministic near-uniform endpoint-inclusive sampling over `[0, total-1]`.
  - added `_variant_space(...)` helper and reused it in `generate(...)`, `_decode_variant_ordinal(...)`, and `_build_constraints(...)`.
  - reduced duplicated radix-validation logic and made combinatorial-space computation consistent across code paths.
- `scripts/phase5_active_learning/search_materials.py`:

  - added `_MAX_RESULTS_LIMIT = 5000` and validation guard (`--max cannot exceed 5000`) to prevent accidental overlarge query/output requests.
  - updated CLI help text to expose the hard cap explicitly.
- `tests/unit/active_learning/test_structure_enumerator_script.py`:

  - updated ordinal-stratification regression to match endpoint-inclusive deterministic sampling contract.

## Verification

- `python -m ruff check scripts/phase5_active_learning/test_enumeration.py scripts/phase5_active_learning/structure_enumerator.py scripts/phase5_active_learning/search_materials.py tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_test_enumeration_script.py tests/unit/active_learning/test_search_materials_cli.py`
- `python -m py_compile scripts/phase5_active_learning/test_enumeration.py scripts/phase5_active_learning/structure_enumerator.py scripts/phase5_active_learning/search_materials.py tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_test_enumeration_script.py tests/unit/active_learning/test_search_materials_cli.py`
- `python -m pytest -q tests/unit/active_learning/test_test_enumeration_script.py tests/unit/active_learning/test_structure_enumerator_script.py tests/unit/active_learning/test_search_materials_cli.py`

## Research references used in batch 72

- Python `importlib` official docs: https://docs.python.org/3/library/importlib.html
- Python `argparse` official docs: https://docs.python.org/3/library/argparse.html
- pymatgen installation docs (`enumlib` optional dependency): https://pymatgen.org/installation.html
- pymatgen API docs (`StructureMatcher`): https://pymatgen.org/pymatgen.analysis.html
- Hart & Forcade (derivative superstructures): https://doi.org/10.1107/S0108767308028503
- Hart, Nelson & Forcade (multicomponent derivative structures): https://doi.org/10.1016/j.commatsci.2012.02.015
- dsenum package docs: https://lan496.github.io/dsenum/
- JARVIS data paper (Nature Scientific Data): https://doi.org/10.1038/s41597-020-00723-1

## Progress snapshot (after Batch 72)

- Completed: Batch 1 through Batch 72.
- Pending: Batch 73 onward.

## Batch 73 (max 5 files)

- [X] `atlas/active_learning/policy_engine.py` - reviewed + optimized
- [X] `atlas/active_learning/policy_state.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_policy_engine.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_policy_state.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 73 optimization goals

- Tighten policy/state integer coercion contracts to eliminate silent truncation and bool-as-int ambiguity.
- Make conformal calibration scale estimation use finite-sample-aware split-conformal quantile level.
- Align CMOEIC utility decomposition with objective/feasibility separation (avoid repeated topology multiplication).
- Add regression tests for n_top sanitization and stricter integer-field handling.

## Batch 73 outcomes

- `atlas/active_learning/policy_state.py`:

  - hardened `_coerce_int(...)` to reject bool and non-integral real values (fallback to defaults).
  - added `_conformal_quantile_level(...)` with finite-sample split-conformal correction `ceil((n+1)*(1-alpha))/n`.
  - `update_calibration(...)` now uses the finite-sample-aware quantile level when estimating `conformal_scale`.
- `atlas/active_learning/policy_engine.py`:

  - added strict positive-int sanitization for `n_top` (`score_and_select(...)`) to prevent invalid selection requests.
  - revised `_base_utility(...)` so objective and feasibility are separated (objective from stability/diversity/cost; topology/synthesis applied in feasibility stage), reducing topology over-counting.
- `tests/unit/active_learning/test_policy_state.py`:

  - added regressions for non-integral integer control rejection in profile/state payloads.
  - added regression ensuring finite-sample conformal calibration path remains numerically stable.
- `tests/unit/active_learning/test_policy_engine.py`:

  - added regression proving non-positive and non-integral `n_top` are sanitized to safe positive integer behavior.

## Verification

- `python -m ruff check atlas/active_learning/policy_engine.py atlas/active_learning/policy_state.py tests/unit/active_learning/test_policy_engine.py tests/unit/active_learning/test_policy_state.py`
- `python -m py_compile atlas/active_learning/policy_engine.py atlas/active_learning/policy_state.py tests/unit/active_learning/test_policy_engine.py tests/unit/active_learning/test_policy_state.py`
- `python -m pytest -q tests/unit/active_learning/test_policy_engine.py tests/unit/active_learning/test_policy_state.py`
- `python -m pytest -q tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_controller_runtime_stability.py`

## Research references used in batch 73

- Angelopoulos & Bates, conformal prediction tutorial (split-conformal quantile guidance): https://arxiv.org/abs/2107.07511
- Kuleshov et al., calibrated regression uncertainty: https://proceedings.mlr.press/v80/kuleshov18a.html
- NumPy `quantile` official documentation: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
- Settles, Active Learning Literature Survey: https://burrsettles.com/pub/settles.activelearning.pdf
- Beygelzimer et al., importance-weighted active learning: https://www.jmlr.org/papers/v10/beygelzimer09a.html
- Python bool/int semantics (official): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool

## Progress snapshot (after Batch 73)

- Completed: Batch 1 through Batch 73.
- Pending: Batch 74 onward.

## Batch 74 (max 5 files)

- [X] `atlas/active_learning/acquisition.py` - reviewed + optimized
- [X] `atlas/active_learning/objective_space.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_acquisition.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_objective_space.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 74 optimization goals

- Harden acquisition parameter coercion to avoid bool/non-integral truncation drift in iterative AL loops.
- Reduce NEI Monte Carlo variance and improve deterministic behavior in zero-observation-noise regimes.
- Make objective-space threshold and dimensionality sanitization explicit and bounded for robust Pareto/feasibility preprocessing.
- Add regression tests for these numerical/contract guarantees.

## Batch 74 outcomes

- `atlas/active_learning/acquisition.py`:

  - `_coerce_int(...)` is now strict for bool/non-integral real inputs (fallback to defaults).
  - added `_sanitize_mc_samples(...)` with bounded range to avoid pathological sample-count blowups.
  - `_noisy_expected_improvement_prepared(...)` now:

    - falls back to deterministic EI when observed noise is effectively zero,
    - uses antithetic-normal pairing for MC samples to reduce estimator variance at fixed budget.
  - `score_acquisition(...)` and `schedule_ucb_kappa(...)` now sanitize non-finite/non-integral controls consistently.
- `atlas/active_learning/objective_space.py`:

  - hardened `clip01(...)` via `safe_float(...)` to handle scalar-like containers safely.
  - `_coerce_obj_dim(...)` no longer silently truncates non-integral objective dimensions.
  - `_coerce_threshold(...)` introduced and applied in history collection + feasibility masking to enforce unit-interval threshold contracts.
- `tests/unit/active_learning/test_acquisition.py`:

  - added regression proving NEI collapses to EI under zero observation noise.
  - added regressions for sanitization of non-integral `nei_mc_samples` and non-integral `iteration` controls.
- `tests/unit/active_learning/test_objective_space.py`:

  - added regression for non-integral `obj_dim` fallback behavior and threshold clipping behavior.

## Verification

- `python -m ruff check atlas/active_learning/acquisition.py atlas/active_learning/objective_space.py tests/unit/active_learning/test_acquisition.py tests/unit/active_learning/test_objective_space.py`
- `python -m py_compile atlas/active_learning/acquisition.py atlas/active_learning/objective_space.py tests/unit/active_learning/test_acquisition.py tests/unit/active_learning/test_objective_space.py`
- `python -m pytest -q tests/unit/active_learning/test_acquisition.py tests/unit/active_learning/test_objective_space.py`
- `python -m pytest -q tests/unit/active_learning/test_controller_acquisition.py`

## Research references used in batch 74

- Jones, Schonlau, Welch (EGO, EI origin; DOI in metadata): https://r7-www1.stat.ubc.ca/efficient-global-optimization-expensive-black-box-functions
- GP-UCB regret analysis: https://arxiv.org/abs/0912.3995
- LogEI stabilization paper: https://arxiv.org/abs/2310.20708
- Constrained/Noisy BO (NEI context): https://arxiv.org/abs/1706.07094
- BoTorch acquisition docs (analytic/MC and LogNEI recommendation): https://botorch.readthedocs.io/en/latest/acquisition.html
- BoTorch acquisition overview (QMC variance reduction discussion): https://botorch.org/docs/v0.14.0/acquisition
- BoTorch framework paper (MC BO tooling): https://proceedings.neurips.cc/paper/2020/file/f5b1b89d98b7286673128a5fb112cb9a-Paper.pdf
- Antithetic variates classic reference: https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/abs/new-monte-carlo-technique-antithetic-variates/69A9BBEDC6A4F1B1AF7E0764CD422E15

## Progress snapshot (after Batch 74)

- Completed: Batch 1 through Batch 74.
- Pending: Batch 75 onward.

## Batch 75 (max 5 files)

- [X] `atlas/active_learning/gp_surrogate.py` - reviewed + optimized
- [X] `atlas/active_learning/generator.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_gp_surrogate.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_generator.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 75 optimization goals

- Eliminate silent truncation/bool-as-int behavior in GP surrogate integer controls.
- Harden feature preprocessing for surrogate descriptors to preserve probability semantics.
- Improve generator hyperparameter sanitization under non-finite inputs (NaN/inf) to avoid unstable weight pipelines.
- Add regressions for strict coercion and non-finite-weight normalization behavior.

## Batch 75 outcomes

- `atlas/active_learning/gp_surrogate.py`:

  - hardened `_coerce_int(...)` for bool/non-integral real inputs (fallback to defaults).
  - added `_clip01(...)` and applied it to probability-like candidate descriptors in `candidate_to_features(...)`.
  - `_schedule_ucb_kappa(...)` now uses strict integer coercion for iteration.
- `atlas/active_learning/generator.py`:

  - introduced robust `_coerce_int(...)` / `_coerce_float(...)` utilities and applied them in constructor hyperparameter normalization.
  - constructor now handles non-finite inputs safely for RNG seed, archive limits, weights, and substitution decay.
  - `_normalize_weights(...)` now sanitizes non-finite entries before normalization.
  - substitution online stats update/sampling paths now sanitize corrupted/non-finite count/reward values before use.
- `tests/unit/active_learning/test_gp_surrogate.py`:

  - added regression for non-integral integer config fields falling back to validated defaults.
  - added regression ensuring `candidate_to_features(...)` clips probability-like descriptors into `[0, 1]`.
- `tests/unit/active_learning/test_generator.py`:

  - added regression covering constructor sanitization with NaN/inf hyperparameters.
  - added regression for `_normalize_weights(...)` with non-finite entries.

## Verification

- `python -m ruff check atlas/active_learning/gp_surrogate.py atlas/active_learning/generator.py tests/unit/active_learning/test_gp_surrogate.py tests/unit/active_learning/test_generator.py`
- `python -m py_compile atlas/active_learning/gp_surrogate.py atlas/active_learning/generator.py tests/unit/active_learning/test_gp_surrogate.py tests/unit/active_learning/test_generator.py`
- `python -m pytest -q tests/unit/active_learning/test_gp_surrogate.py tests/unit/active_learning/test_generator.py`

## Research references used in batch 75

- scikit-learn Gaussian Process documentation: https://scikit-learn.org/stable/modules/gaussian_process.html
- `GaussianProcessRegressor` API: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
- `GaussianProcessClassifier` API: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html
- `Matern` kernel API: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html
- `WhiteKernel` API: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.WhiteKernel.html
- GP-UCB regret analysis (Srinivas et al.): https://arxiv.org/abs/0912.3995
- Constrained Bayesian Optimization (Gardner et al.): https://proceedings.mlr.press/v32/gardner14.html
- Ionic substitution statistics (Hautier et al.): https://doi.org/10.1021/ic102031h
- SMACT / composition screening (Davies et al.): https://doi.org/10.1039/D2DD00028H

## Progress snapshot (after Batch 75)

- Completed: Batch 1 through Batch 75.
- Pending: Batch 76 onward.

## Batch 76 (max 5 files)

- [X] `atlas/active_learning/crabnet_native.py` - reviewed + optimized
- [X] `atlas/active_learning/synthesizability.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_crabnet_native.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_synthesizability.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 81 (max 5 files)

- [X] `atlas/training/checkpoint.py` - reviewed + optimized
- [X] `atlas/training/preflight.py` - reviewed + optimized
- [X] `atlas/training/run_utils.py` - reviewed + optimized
- [X] `tests/unit/training/test_checkpoint.py` - reviewed + optimized
- [X] `tests/unit/training/test_preflight.py` - reviewed + optimized

## Batch 76 optimization goals

- Eliminate silent integer truncation and bool-as-int ambiguity in CrabNet/UQ controls used inside iterative AL loops.
- Harden grouped uncertainty calibration so malformed group-temperature maps cannot poison per-group scaling.
- Align synthesizability integer-control sanitation with strict research reproducibility contracts.
- Add targeted regressions to lock down these runtime contracts.

## Batch 81 optimization goals

- Remove silent int truncation / bool-as-int behavior in checkpoint and preflight runtime controls.
- Make checkpoint metric coercion explicit so malformed MAE inputs fail fast.
- Harden run-manifest strict-lock parsing to avoid accidental truthy coercion.
- Add regression tests for strict coercion contracts.

## Batch 76 outcomes

- `atlas/active_learning/crabnet_native.py`:

  - added strict `_coerce_int(...)` that rejects bool and non-integral real values (fallback to defaults + bounds).
  - applied strict integer sanitation to `mean_dims`, `ensemble_size`, `mc_dropout_samples`, `q_steps`, grouped calibration `min_group_size`, and `predict_distribution(mc_samples=...)`.
  - added grouped-calibration table sanitation (`_coerce_group_id`, `_sanitize_group_temperature_table`) and automatic cleanup before applying per-group scaling.
- `atlas/active_learning/synthesizability.py`:

  - upgraded module `_coerce_int(...)` to strict integer semantics (no silent decimal truncation, bool rejected).
- `tests/unit/active_learning/test_crabnet_native.py`:

  - added regression for strict constructor integer controls.
  - added regression verifying grouped-calibration table cleanup removes invalid keys and keeps valid scaling behavior.
- `tests/unit/active_learning/test_synthesizability.py`:

  - added regression verifying integer controls reject bool/fractional inputs and still accept valid integer-like strings.

## Verification

- `python -m ruff check atlas/active_learning/crabnet_native.py atlas/active_learning/synthesizability.py tests/unit/active_learning/test_crabnet_native.py tests/unit/active_learning/test_synthesizability.py`
- `python -m py_compile atlas/active_learning/crabnet_native.py atlas/active_learning/synthesizability.py tests/unit/active_learning/test_crabnet_native.py tests/unit/active_learning/test_synthesizability.py`
- `python -m pytest -q tests/unit/active_learning/test_crabnet_native.py tests/unit/active_learning/test_synthesizability.py`

## Research references used in batch 76

- Python built-in types (`bool` is a subtype of `int`): https://docs.python.org/3/library/stdtypes.html
- PyTorch quantile API (calibration quantile contract): https://docs.pytorch.org/docs/stable/generated/torch.quantile.html
- CrabNet paper (npj Computational Materials): https://www.nature.com/articles/s41524-021-00545-1
- Aitchison compositional data geometry (JRSS-B, 1982): https://doi.org/10.1111/j.2517-6161.1982.tb01195.x
- Gal & Ghahramani (MC Dropout as Bayesian approximation, ICML 2016): https://proceedings.mlr.press/v48/gal16.html
- Lakshminarayanan et al. (Deep Ensembles, NeurIPS 2017): https://papers.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles
- Kendall & Gal (aleatoric/epistemic uncertainty): https://arxiv.org/abs/1703.04977
- Angelopoulos & Bates (conformal prediction tutorial): https://arxiv.org/abs/2107.07511

## Progress snapshot (after Batch 76)

- Completed: Batch 1 through Batch 76.
- Pending: Batch 77 onward.

## Batch 77 (max 5 files)

- [X] `atlas/active_learning/rxn_network_native.py` - reviewed + optimized
- [X] `atlas/active_learning/crabnet_screener.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_rxn_network_native.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_crabnet_screener.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 77 optimization goals

- Eliminate silent integer truncation and string-bool ambiguity in reaction-network and composition-screener runtime controls.
- Prevent `bool("false") -> True` style config drift in reaction-network solver switches.
- Keep uncertainty/MC-dropout controls numerically stable and reproducible under malformed integer-like inputs.
- Add contract tests for strict coercion behavior.

## Batch 77 outcomes

- `atlas/active_learning/rxn_network_native.py`:

  - added strict `_coerce_int(...)` (rejects bool + non-integral reals, uses bounded defaults).
  - added robust `_coerce_bool(...)` for string flags (`true/false`, `yes/no`, `on/off`, `1/0`) with safe default fallback.
  - applied coercion to: `max_num_pathways`, `k_shortest_paths`, `max_num_combos`, `chunk_size`, all solver booleans, and `require_native`.
  - hardened `fallback_mode` to controlled enum (`conservative` / `energy_prior`) with conservative fallback.
- `atlas/active_learning/crabnet_screener.py`:

  - upgraded `_coerce_int(...)` to strict integer semantics.
  - applied strict coercion to `out_dims`, `d_model`, MC-dropout sample controls (`_mc_dropout_epistemic_var`, `predict_distribution`).
  - removed implicit decimal truncation behavior for MC sample counts.
- `tests/unit/active_learning/test_rxn_network_native.py`:

  - added regressions for strict int/bool sanitization and fallback-mode normalization.
  - added regression proving integer-like strings remain accepted.
- `tests/unit/active_learning/test_crabnet_screener.py`:

  - added regressions proving bool/fractional integer controls are rejected to defaults.
  - added regression for non-integral `mc_samples` sanitization in `predict_distribution(...)`.

## Batch 81 outcomes

- `atlas/training/checkpoint.py`:

  - added strict integer/float coercion helpers and applied them to `top_k`, `keep_last_k`, `epoch`, and `mae`.
  - rejected bool/non-integral numeric inputs for integer fields to avoid implicit truncation (`1.9 -> 1`) in recovery metadata.
- `atlas/training/preflight.py`:

  - upgraded numeric coercion for `max_samples`, `split_seed`, and `timeout_sec` to strict integer semantics.
  - now fast-fails on bool or fractional integer-like parameters for gate controls.
- `atlas/training/run_utils.py`:

  - added `_coerce_bool_like(...)` and used it in `_resolve_environment_lock(...)` to avoid non-explicit truthy parsing for strict lock toggles.
- tests:

  - `test_checkpoint.py` now covers bool/fractional integer rejection and malformed MAE type rejection.
  - `test_preflight.py` now covers invalid fractional timeout and bool-valued `max_samples`.

## Verification

- `python -m ruff check atlas/active_learning/rxn_network_native.py atlas/active_learning/crabnet_screener.py tests/unit/active_learning/test_rxn_network_native.py tests/unit/active_learning/test_crabnet_screener.py`
- `python -m py_compile atlas/active_learning/rxn_network_native.py atlas/active_learning/crabnet_screener.py tests/unit/active_learning/test_rxn_network_native.py tests/unit/active_learning/test_crabnet_screener.py`
- `python -m pytest -q tests/unit/active_learning/test_rxn_network_native.py tests/unit/active_learning/test_crabnet_screener.py`

## Research references used in batch 77

- Python built-in types (`bool` subtype of `int`): https://docs.python.org/3/library/stdtypes.html
- PEP 285 (`bool` type design): https://peps.python.org/pep-0285/
- NumPy finite-value screening: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy finite sanitization: https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
- PyTorch `TransformerEncoderLayer` API: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
- PyTorch `GaussianNLLLoss` API: https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.GaussianNLLLoss.html
- CrabNet (npj Computational Materials, 2021): https://www.nature.com/articles/s41524-021-00545-1
- Deep Sets (NeurIPS 2017): https://arxiv.org/abs/1703.06114
- Dropout as Bayesian Approximation (ICML 2016): https://proceedings.mlr.press/v48/gal16.html
- Deep Ensembles (NeurIPS 2017): https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa85bce38a7-Abstract.html
- Bayesian aleatoric/epistemic uncertainty: https://arxiv.org/abs/1703.04977
- Reaction-network pathfinding in solid-state synthesis (Nat Commun 2021): https://www.nature.com/articles/s41467-021-23339-x
- K shortest loopless paths (Yen 1971): http://dx.doi.org/10.1287/mnsc.17.11.712
- Multi-criteria shortest path (Martins 1984): https://doi.org/10.1016/0377-2217(84)90077-8
- Conformal prediction tutorial: https://arxiv.org/abs/2107.07511

## Progress snapshot (after Batch 77)

- Completed: Batch 1 through Batch 77.
- Pending: Batch 78 onward.

## Batch 78 (max 5 files)

- [X] `scripts/phase5_active_learning/active_learning.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/run_discovery.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_phase6_active_learning.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_phase5_cli.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

- `python -m ruff check atlas/training/checkpoint.py atlas/training/preflight.py atlas/training/run_utils.py tests/unit/training/test_checkpoint.py tests/unit/training/test_preflight.py`
- `python -m pytest -q tests/unit/training/test_checkpoint.py tests/unit/training/test_preflight.py tests/unit/training/test_run_utils_manifest.py`

## Batch 78 optimization goals

- Eliminate silent integer truncation in random-acquisition controls (`n_pool`, `seed`) to keep AL sampling behavior reproducible and type-safe.
- Reject boolean values in discovery acquisition float controls (`acq_kappa`, `acq_jitter`, `acq_best_f`) to avoid implicit bool-to-float coercion.
- Keep CLI/runtime validation behavior backward compatible for valid numeric inputs while hard-failing malformed controls.
- Add regression tests locking strict coercion contracts.

## Research references used in batch 81

- Python `bool` type semantics (`bool` is subclass of `int`): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool
- Python `numbers` hierarchy (Integral/Real contracts): https://docs.python.org/3/library/numbers.html
- Python `subprocess.run` timeout/error behavior: https://docs.python.org/3/library/subprocess.html#subprocess.run
- Python `tempfile.NamedTemporaryFile` for safe temp-write patterns: https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile
- Python `pathlib.Path.replace` atomic rename semantics: https://docs.python.org/3/library/pathlib.html#pathlib.Path.replace
- PyTorch checkpoint serialization docs (`torch.save` / `torch.load`): https://pytorch.org/tutorials/beginner/saving_loading_models.html

## Batch 78 outcomes

- `scripts/phase5_active_learning/active_learning.py`:

  - added strict `_coerce_non_negative_int(...)` helper (rejects bool + non-integral reals).
  - `acquisition_random(...)` now uses strict coercion for `n_pool` and `seed`, removing implicit truncation (e.g., `3.5 -> 3`).
- `scripts/phase5_active_learning/run_discovery.py`:

  - added `_coerce_finite_float(...)` helper with bool rejection and finite/range checks.
  - `_validate_discovery_args(...)` now normalizes and validates `acq_kappa`, `acq_jitter`, and `acq_best_f` through the helper.
- `tests/unit/active_learning/test_phase6_active_learning.py`:

  - expanded `acquisition_random` regressions for non-integral and bool inputs (`n_pool`, `seed`).
- `tests/unit/active_learning/test_phase5_cli.py`:

  - added regression proving discovery validator rejects boolean acquisition controls.

## Verification

- `python -m ruff check scripts/phase5_active_learning/active_learning.py scripts/phase5_active_learning/run_discovery.py tests/unit/active_learning/test_phase6_active_learning.py tests/unit/active_learning/test_phase5_cli.py`
- `python -m py_compile scripts/phase5_active_learning/active_learning.py scripts/phase5_active_learning/run_discovery.py tests/unit/active_learning/test_phase6_active_learning.py tests/unit/active_learning/test_phase5_cli.py`
- `python -m pytest -q tests/unit/active_learning/test_phase5_cli.py tests/unit/active_learning/test_phase6_active_learning.py`

## Research references used in batch 78

- Python built-in types (`bool` subtype of `int`): https://docs.python.org/3/library/stdtypes.html
- Python numeric tower (`numbers.Integral` / `numbers.Real`): https://docs.python.org/3/library/numbers.html
- Python argparse (official): https://docs.python.org/3/library/argparse.html
- Python subprocess (official): https://docs.python.org/3/library/subprocess.html
- NumPy Generator API (`default_rng`): https://numpy.org/doc/stable/reference/random/generator.html
- NumPy `Generator.choice` API: https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html
- Active Learning Literature Survey (Settles, 2009): https://burrsettles.com/pub/settles.activelearning_20090109.pdf
- EGO / Expected Improvement origin (Jones et al., 1998): https://r7-www1.stat.ubc.ca/efficient-global-optimization-expensive-black-box-functions
- Constrained Bayesian Optimization (Gardner et al., ICML 2014): https://proceedings.mlr.press/v32/gardner14.html
- Conformal prediction tutorial (Angelopoulos & Bates): https://arxiv.org/abs/2107.07511

## Progress snapshot (after Batch 78)

- Completed: Batch 1 through Batch 78.
- Pending: Batch 79 onward.

## Batch 79 (max 5 files)

- [X] `scripts/phase5_active_learning/search_materials.py` - reviewed + optimized
- [X] `scripts/phase5_active_learning/test_enumeration.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_search_materials_cli.py` - reviewed + optimized
- [X] `tests/unit/active_learning/test_test_enumeration_script.py` - reviewed + optimized
- [X] `docs/todolist_temp.md` - updated

## Batch 79 optimization goals

- Tighten CLI numeric-parameter governance in multi-property search to reject bool/non-finite payloads consistently.
- Eliminate implicit bool/float-to-int coercion leakage in demo summary-count validation.
- Preserve backward compatibility for valid numeric inputs while enforcing deterministic validation failures for malformed controls.
- Add regression tests locking these contracts.

## Batch 79 outcomes

- `scripts/phase5_active_learning/search_materials.py`:

  - added `_coerce_optional_finite_float(...)` to normalize optional numeric filters with explicit bool rejection.
  - `_validate_args(...)` now sanitizes all optional range fields and `ehull_max` through strict finite-value coercion.
  - simplified `--max` validation messaging by directly validating with canonical option name.
- `scripts/phase5_active_learning/test_enumeration.py`:

  - added strict `_coerce_non_negative_int(...)` helper for summary counts.
  - summary count fields now use strict non-negative integer coercion (rejects bool and fractional values).
- `tests/unit/active_learning/test_search_materials_cli.py`:

  - added regressions proving bool-valued numeric filters (`ehull_max`, `bandgap_min`) are rejected.
- `tests/unit/active_learning/test_test_enumeration_script.py`:

  - added regressions for `_coerce_non_negative_int(...)` strictness on bool/fractional inputs.

## Verification

- `python -m ruff check scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/test_enumeration.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_test_enumeration_script.py`
- `python -m py_compile scripts/phase5_active_learning/search_materials.py scripts/phase5_active_learning/test_enumeration.py tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_test_enumeration_script.py`
- `python -m pytest -q tests/unit/active_learning/test_search_materials_cli.py tests/unit/active_learning/test_test_enumeration_script.py`

## Research references used in batch 79

- Python built-in types (`bool` subtype of `int`): https://docs.python.org/3/library/stdtypes.html
- Python numeric tower (`numbers.Integral` / `numbers.Real`): https://docs.python.org/3/library/numbers.html
- Python argparse (official): https://docs.python.org/3/library/argparse.html
- Pandas user guide (IO + table display/query context): https://pandas.pydata.org/docs/user_guide/index.html
- NumPy finite checks (`isfinite`): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- pymatgen StructureMatcher docs: https://pymatgen.org/pymatgen.analysis.html#pymatgen.analysis.structure_matcher.StructureMatcher
- pymatgen Structure API: https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure
- Hart & Forcade derivative-structure enumeration: https://doi.org/10.1107/S0108767308028503
- Hart, Nelson & Forcade multicomponent derivatives: https://doi.org/10.1016/j.commatsci.2012.02.015
- JARVIS-DFT database paper (Sci Data): https://doi.org/10.1038/s41597-020-00723-1

## Progress snapshot (after Batch 79)

- Completed: Batch 1 through Batch 79.
- Pending: Batch 80 onward.

## Batch 80 (max 5 files)

- [X] `atlas/models/prediction_utils.py` - reviewed + optimized
- [X] `atlas/models/utils.py` - reviewed + optimized
- [X] `atlas/utils/reproducibility.py` - reviewed + optimized
- [X] `tests/unit/models/test_prediction_utils.py` - reviewed + optimized
- [X] `tests/unit/research/test_reproducibility.py` - reviewed + optimized

## Batch 80 optimization goals

- Harden prediction payload normalization so non-finite mean outputs cannot leak into downstream ranking/metrics.
- Tighten reproducibility coercion semantics to avoid implicit bool/float truncation drift in `seed` and deterministic flags.
- Keep checkpoint normalizer validation lightweight and deterministic.
- Add regression tests that lock strict coercion and sanitized-mean contracts.

## Batch 80 outcomes

- `atlas/models/prediction_utils.py`:

  - added `_sanitize_mean_like(...)` and applied it to `mean/mu/gamma` extraction paths.
  - evidential payload parsing now sanitizes `gamma` before uncertainty derivation.
  - non-finite model means now map to bounded finite values (`nan->0`, `+inf->1e6`, `-inf->-1e6`).
- `atlas/utils/reproducibility.py`:

  - added `_is_integral_float(...)` and switched bool/seed coercion to strict integral semantics.
  - `_coerce_bool(...)` now rejects non-integral floats (e.g., `0.7`) and ambiguous numeric strings unless integral-like.
  - `_coerce_seed(...)` now rejects bool and non-integral float/string seeds (falls back to default), preserving uint32 normalization.
- `atlas/models/utils.py`:

  - replaced tensor-based finiteness check in scalar normalizer validation with `math.isfinite(...)` for lower overhead and clearer intent.
- Tests:

  - `test_prediction_utils.py` adds non-finite mean sanitization coverage for both direct and evidential payloads.
  - `test_reproducibility.py` adds strict coercion regressions for `_coerce_seed` and `_coerce_bool`.

## Verification

- `python -m ruff check atlas/models/prediction_utils.py atlas/models/utils.py atlas/utils/reproducibility.py tests/unit/models/test_prediction_utils.py tests/unit/research/test_reproducibility.py`
- `python -m pytest -q tests/unit/models/test_prediction_utils.py tests/unit/research/test_reproducibility.py`

## Research references used in batch 80

- Python built-in types (`bool` subtype of `int`): https://docs.python.org/3/library/stdtypes.html
- PEP 285 (`bool` type semantics): https://peps.python.org/pep-0285/
- Python numeric tower (`numbers`): https://docs.python.org/3/library/numbers.html
- NumPy finite-value checks: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy NaN/Inf sanitization: https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
- PyTorch deterministic behavior notes: https://pytorch.org/docs/stable/notes/randomness.html
- Python `random` module reproducibility notes: https://docs.python.org/3/library/random.html

## Progress snapshot (after Batch 80)

- Completed: Batch 1 through Batch 80.
- Pending: Batch 81 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 81.

## Progress snapshot (after Batch 81)

- Completed: Batch 1 through Batch 81.
- Pending: Batch 82 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 82.
