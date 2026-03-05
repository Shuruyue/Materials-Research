
## Batch 81 (max 5 files)
- [x] `atlas/training/checkpoint.py` - reviewed + optimized
- [x] `atlas/training/preflight.py` - reviewed + optimized
- [x] `atlas/training/run_utils.py` - reviewed + optimized
- [x] `tests/unit/training/test_checkpoint.py` - reviewed + optimized
- [x] `tests/unit/training/test_preflight.py` - reviewed + optimized

## Batch 81 optimization goals
- Remove silent int truncation / bool-as-int behavior in checkpoint and preflight runtime controls.
- Make checkpoint metric coercion explicit so malformed MAE inputs fail fast.
- Harden run-manifest strict-lock parsing to avoid accidental truthy coercion.
- Add regression tests for strict coercion contracts.

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
- `python -m ruff check atlas/training/checkpoint.py atlas/training/preflight.py atlas/training/run_utils.py tests/unit/training/test_checkpoint.py tests/unit/training/test_preflight.py`
- `python -m pytest -q tests/unit/training/test_checkpoint.py tests/unit/training/test_preflight.py tests/unit/training/test_run_utils_manifest.py`

## Research references used in batch 81
- Python `bool` type semantics (`bool` is subclass of `int`): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool
- Python `numbers` hierarchy (Integral/Real contracts): https://docs.python.org/3/library/numbers.html
- Python `subprocess.run` timeout/error behavior: https://docs.python.org/3/library/subprocess.html#subprocess.run
- Python `tempfile.NamedTemporaryFile` for safe temp-write patterns: https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile
- Python `pathlib.Path.replace` atomic rename semantics: https://docs.python.org/3/library/pathlib.html#pathlib.Path.replace
- PyTorch checkpoint serialization docs (`torch.save` / `torch.load`): https://pytorch.org/tutorials/beginner/saving_loading_models.html

## Progress snapshot (after Batch 81)
- Completed: Batch 1 through Batch 81.
- Pending: Batch 82 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 82.

## Batch 82 (max 5 files)
- [x] `atlas/training/losses.py` - reviewed + optimized
- [x] `atlas/training/physics_losses.py` - reviewed + optimized
- [x] `atlas/training/filters.py` - reviewed (no code change required in this batch)
- [x] `tests/unit/training/test_losses.py` - reviewed + optimized
- [x] `tests/unit/training/test_physics_losses.py` - reviewed + optimized

## Batch 82 optimization goals
- Strengthen scalar-parameter validation in training losses to block bool-as-number drift.
- Make evidential regression loss robust to integral target tensors (`torch.finfo` requires floating dtypes).
- Tighten physics-constraint hyperparameter validation for `weight`/`alpha` controls.
- Add regression tests for strict coercion and mixed-dtype robustness.

## Batch 82 outcomes
- `atlas/training/losses.py`:
  - `_require_finite_scalar(...)` now rejects bool and non-numeric values explicitly.
  - `EvidentialLoss.forward(...)` now normalizes target dtype to floating before `torch.finfo(...)` path, preventing integer-dtype runtime failures.
- `atlas/training/physics_losses.py`:
  - added `_coerce_non_negative_float(...)` and applied it to `VoigtReussBoundsLoss(weight=...)` and `PhysicsConstraintLoss(alpha=...)` parsing.
  - bool/non-finite/negative physics weights are now fast-fail validated.
- `atlas/training/filters.py`:
  - reviewed for statistical masking and CSV fallback path; no additional patch required this batch.
- tests:
  - `test_losses.py` adds coverage for bool rejection in scalar config, integer target support in evidential loss, and bool task-weight rejection.
  - `test_physics_losses.py` adds coverage for bool `weight` and invalid `alpha` values.

## Verification
- `python -m ruff check atlas/training/losses.py atlas/training/physics_losses.py tests/unit/training/test_losses.py tests/unit/training/test_physics_losses.py`
- `python -m pytest -q tests/unit/training/test_losses.py tests/unit/training/test_physics_losses.py`

## Research references used in batch 82
- Deep Evidential Regression (NeurIPS 2020): https://arxiv.org/abs/1910.02600
- Evidential Deep Learning to Quantify Classification Uncertainty (NeurIPS 2018): https://papers.nips.cc/paper/7580-evidential-deep-learning-to-quantify-classification-uncertainty
- Multi-Task Learning Using Uncertainty to Weigh Losses (CVPR 2018): https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html
- PyTorch `torch.finfo` dtype contract: https://docs.pytorch.org/docs/stable/type_info.html
- PyTorch `BCEWithLogitsLoss` reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
- Python bool/int semantics: https://docs.python.org/3/library/functions.html#bool
- PEP 285 (bool subtype rationale): https://peps.python.org/pep-0285/
- Elastic stability/Born criterion overview: https://pubmed.ncbi.nlm.nih.gov/22617724/

## Progress snapshot (after Batch 82)
- Completed: Batch 1 through Batch 82.
- Pending: Batch 83 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 83.

## Batch 83 (max 5 files)
- [x] `atlas/training/metrics.py` - reviewed + optimized
- [x] `atlas/training/normalizers.py` - reviewed + optimized
- [x] `atlas/training/filters.py` - reviewed (no code change required in this batch)
- [x] `tests/unit/training/test_metrics.py` - reviewed + optimized
- [x] `tests/unit/training/test_normalizers.py` - reviewed + optimized

## Batch 83 optimization goals
- Tighten metric serialization safety so boolean-return edge cases cannot silently leak into float metrics.
- Harden normalizer state parsing against bool-as-float coercion (`True -> 1.0`, `False -> 0.0`).
- Preserve robust outlier filtering behavior in `filters.py` after targeted review.
- Add regressions for strict coercion and mixed-type safety.

## Batch 83 outcomes
- `atlas/training/metrics.py`:
  - kept `_safe_float(...)` bool rejection and removed premature `float(...)` cast on `roc_auc_score(...)` output so bool-like returns still respect metric defaults.
- `atlas/training/normalizers.py`:
  - `_to_finite_float(...)` now rejects bool / `np.bool_` values explicitly.
  - prevents accidental boolean state payloads from silently becoming normalizer means/stds.
- `atlas/training/filters.py`:
  - reviewed sigma/MAD path and CSV fallback flow; no additional patch needed in this batch.
- tests:
  - `test_metrics.py` adds regression proving boolean metric outputs are mapped to safe defaults.
  - `test_normalizers.py` adds regressions for boolean `mean` rejection and boolean `std` fallback behavior.

## Verification
- `python -m ruff check atlas/training/metrics.py atlas/training/normalizers.py tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py`
- `python -m pytest -q tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py tests/unit/training/test_filters.py`

## Research references used in batch 83
- scikit-learn `r2_score` API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
- scikit-learn `roc_auc_score` API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
- scikit-learn `accuracy_score` API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
- SciPy `spearmanr` API: https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.stats.spearmanr.html
- NumPy `nan_to_num` API: https://numpy.org/doc/2.1/reference/generated/numpy.nan_to_num.html
- NumPy `isinf` / finite checks: https://numpy.org/doc/stable/reference/generated/numpy.isinf.html
- SciPy MAD API (`median_abs_deviation`): https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.stats.median_abs_deviation.html
- NISTIR 8526 (modified z-score discussion): https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=957454
- Python bool semantics: https://docs.python.org/3/library/stdtypes.html#boolean-type-bool

## Progress snapshot (after Batch 83)
- Completed: Batch 1 through Batch 83.
- Pending: Batch 84 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 84.

## Batch 84 (max 5 files)
- [x] `atlas/training/trainer.py` - reviewed + optimized
- [x] `atlas/training/run_utils.py` - reviewed (no code change required in this batch)
- [x] `atlas/training/__init__.py` - reviewed (no code change required in this batch)
- [x] `tests/unit/training/test_trainer.py` - reviewed + optimized
- [x] `tests/unit/training/test_init_exports.py` - reviewed (no code change required in this batch)

## Batch 84 optimization goals
- Eliminate silent bool/int/float coercion paths in training-control parameters.
- Prevent non-integral epoch/patience metadata from being silently truncated in checkpoint save/load.
- Preserve backward-compatible trainer API while making validation errors explicit and early.

## Batch 84 outcomes
- `atlas/training/trainer.py`:
  - added `_is_boolean_like(...)` and `_coerce_non_negative_int(...)` to enforce strict integer semantics.
  - upgraded `fit(...)` validation for `n_epochs` and `patience` using strict integer coercion.
  - hardened `_save_checkpoint(...)` epoch validation to reject bool/fractional values.
  - hardened `_validate_checkpoint_payload(...)` and trainer-state restore path to reject non-integral `epoch` / `patience_counter`.
  - tightened `_coerce_non_negative_float(...)` and `_coerce_finite_float(...)` to reject boolean-like inputs.
- `tests/unit/training/test_trainer.py`:
  - added regressions for fractional/boolean epoch rejection in checkpoint save.
  - added regressions for fractional epoch and fractional `trainer_state.patience_counter` rejection during checkpoint load.
  - added regressions for fractional `n_epochs` and boolean `patience` rejection in `fit(...)`.
- `atlas/training/run_utils.py`, `atlas/training/__init__.py`, `tests/unit/training/test_init_exports.py`:
  - reviewed for interface/schema/lazy-export consistency; no patch needed this batch.

## Verification
- `python -m ruff check atlas/training/trainer.py atlas/training/run_utils.py atlas/training/__init__.py tests/unit/training/test_trainer.py tests/unit/training/test_init_exports.py`
- `python -m pytest -q tests/unit/training/test_trainer.py tests/unit/training/test_init_exports.py tests/unit/training/test_run_utils_manifest.py`

## Research references used in batch 84
- Python numeric ABCs (`numbers.Integral` / `Real`): https://docs.python.org/3/library/numbers.html
- Python bool semantics (`bool` is subclass of `int`): https://docs.python.org/3/library/functions.html#bool
- Python bool C-API semantics: https://docs.python.org/3/c-api/bool.html
- PyTorch `torch.load` API (`weights_only`): https://docs.pytorch.org/docs/stable/generated/torch.load.html
- PyTorch serialization note (`weights_only` security behavior): https://docs.pytorch.org/docs/stable/notes/serialization.html
- PyTorch save/load best-practice tutorial (`state_dict`): https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
- PyTorch reproducibility note (determinism limits): https://docs.pytorch.org/docs/stable/notes/randomness.html
- Python mapping ABC reference (checkpoint payload type contracts): https://docs.python.org/3/library/collections.abc.html

## Progress snapshot (after Batch 84)
- Completed: Batch 1 through Batch 84.
- Pending: Batch 85 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 85.

## Batch 85 (max 5 files)
- [x] `atlas/models/uncertainty.py` - reviewed + optimized
- [x] `atlas/models/prediction_utils.py` - reviewed (no code change required in this batch)
- [x] `atlas/models/__init__.py` - reviewed (no code change required in this batch)
- [x] `tests/unit/models/test_uncertainty.py` - reviewed + optimized
- [x] `tests/unit/models/test_prediction_utils.py` - reviewed (no code change required in this batch)

## Batch 85 optimization goals
- Eliminate silent float/bool-to-integer drift in uncertainty ensemble/MC sample-count controls.
- Prevent runtime `range(float)` type failures by enforcing strict positive integer semantics at construction time.
- Improve evidential loss dtype robustness for integer targets while preserving non-finite safety behavior.

## Batch 85 outcomes
- `atlas/models/uncertainty.py`:
  - added `_coerce_positive_int(...)` with strict integer semantics (rejects bool, fractional, non-finite values) and applied to `n_models` / `n_samples`.
  - added `_coerce_non_negative_finite_float(...)` for robust `coeff` handling in evidential loss.
  - `EvidentialRegression.evidential_loss(...)` now normalizes target dtype/device to match predicted evidential tensors, avoiding integer-dtype arithmetic pitfalls.
- `tests/unit/models/test_uncertainty.py`:
  - added regression coverage for invalid `n_models` and `n_samples` inputs (`float`, `bool`, `inf`, `nan`).
  - added regression proving evidential loss remains finite for integer target tensors.
- `atlas/models/prediction_utils.py`, `atlas/models/__init__.py`, `tests/unit/models/test_prediction_utils.py`:
  - reviewed for payload-sanitization, forward-signature, and lazy-export consistency; no patch needed this batch.

## Verification
- `python -m ruff check atlas/models/uncertainty.py atlas/models/prediction_utils.py tests/unit/models/test_uncertainty.py tests/unit/models/test_prediction_utils.py`
- `python -m pytest -q tests/unit/models/test_uncertainty.py tests/unit/models/test_prediction_utils.py`

## Research references used in batch 85
- Deep Ensembles (NeurIPS 2017): https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa85bce38-Abstract.html
- Dropout as a Bayesian Approximation (ICML 2016): https://proceedings.mlr.press/v48/gal16.html
- Deep Evidential Regression (NeurIPS 2020): https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html
- PyTorch `torch.nan_to_num` API: https://pytorch.org/docs/stable/generated/torch.nan_to_num.html
- PyTorch `torch.stack` API: https://pytorch.org/docs/stable/generated/torch.stack.html
- PyTorch `torch.Tensor.std` API: https://pytorch.org/docs/stable/generated/torch.Tensor.std.html
- PyTorch Dropout API: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
- Python numeric ABCs (`Integral`/`Real`): https://docs.python.org/3/library/numbers.html
- Python bool semantics: https://docs.python.org/3/library/stdtypes.html#boolean-type-bool

## Progress snapshot (after Batch 85)
- Completed: Batch 1 through Batch 85.
- Pending: Batch 86 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 86.

## Batch 86 (max 5 files)
- [x] `atlas/models/graph_builder.py` - reviewed + optimized
- [x] `atlas/topology/classifier.py` - reviewed + optimized
- [x] `tests/unit/models/test_structure_expansion.py` - reviewed + optimized
- [x] `tests/unit/topology/test_classifier.py` - reviewed + optimized
- [x] `atlas/models/utils.py` + `tests/unit/models/test_model_utils.py` - reviewed (no code change required in this batch)

## Batch 86 optimization goals
- Remove silent bool/fractional coercion in graph construction and topology-classifier runtime knobs.
- Prevent invalid count parameters from propagating into neighbor loops / MC-dropout loops.
- Lock validation behavior with explicit regression tests for integer-semantics edge cases.

## Batch 86 outcomes
- `atlas/models/graph_builder.py`:
  - added strict scalar coercion helpers for positive int/float fields.
  - `gaussian_expansion(...)` now rejects boolean/fractional `n_gaussians` and non-finite cutoff values explicitly.
  - `CrystalGraphBuilder.__init__(...)` now enforces strict positive-integer `max_neighbors` and rejects boolean-like cutoff values.
- `atlas/topology/classifier.py`:
  - added strict positive-integer validation for `node_dim`, `edge_dim`, `hidden_dim`, `n_layers`, and `predict_proba(..., n_samples=...)`.
  - tightened dropout validation to reject boolean-like values.
  - MC-dropout sampling loop now uses validated integer `n_samples_i`.
- tests:
  - `tests/unit/models/test_structure_expansion.py` now covers fractional/bool rejection for `n_gaussians` and strict init validation for `max_neighbors`/`cutoff`.
  - `tests/unit/topology/test_classifier.py` now covers fractional/bool hyperparameter rejection and invalid `n_samples` payloads.
- `atlas/models/utils.py` and `tests/unit/models/test_model_utils.py`:
  - reviewed for state-dict normalization and load-path robustness; no additional patch required this batch.

## Verification
- `python -m ruff check atlas/models/graph_builder.py atlas/topology/classifier.py atlas/models/utils.py tests/unit/models/test_structure_expansion.py tests/unit/topology/test_classifier.py tests/unit/models/test_model_utils.py`
- `python -m pytest -q tests/unit/models/test_structure_expansion.py tests/unit/topology/test_classifier.py tests/unit/models/test_model_utils.py`

## Research references used in batch 86
- Python numeric ABCs (`numbers.Integral` / `numbers.Real`): https://docs.python.org/3/library/numbers.html
- Python bool semantics (`bool` subtype of `int`): https://docs.python.org/3/library/stdtypes.html
- PyTorch Dropout API (MC dropout execution behavior): https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout
- PyTorch `scatter_reduce_` API (graph pooling reduction semantics): https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html
- PyTorch `unique(..., return_inverse=True)` API (batch-id remapping semantics): https://docs.pytorch.org/docs/stable/generated/torch.Tensor.unique.html
- NumPy `nan_to_num` API (non-finite sanitization): https://numpy.org/doc/2.4/reference/generated/numpy.nan_to_num.html
- NumPy `clip` API (bounded distance handling): https://numpy.org/devdocs/reference/generated/numpy.clip.html
- Crystal Graph Convolutional Neural Networks (PRL 2018): https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301
- SchNet (NeurIPS 2017): https://proceedings.neurips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions.pdf
- Dropout as Bayesian Approximation (ICML 2016): https://proceedings.mlr.press/v48/gal16

## Progress snapshot (after Batch 86)
- Completed: Batch 1 through Batch 86.
- Pending: Batch 87 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 87.

## Batch 87 (max 5 files)
- [x] `atlas/models/layers.py` - reviewed + optimized
- [x] `atlas/models/cgcnn.py` - reviewed + optimized
- [x] `atlas/models/multi_task.py` - reviewed + optimized
- [x] `tests/unit/models/test_cgcnn.py` - reviewed + optimized
- [x] `tests/unit/models/test_multi_task.py` - reviewed + optimized

## Batch 87 optimization goals
- Eliminate silent bool/fractional coercion in model/layer hyperparameter validation.
- Prevent accidental float-index truncation in message passing paths.
- Convert multi-task inference from silent task dropping to explicit unknown-task failure for reproducibility.

## Batch 87 outcomes
- `atlas/models/layers.py`:
  - added strict scalar coercion helpers for positive integer/float validation.
  - `MessagePassingLayer` now rejects non-integer `edge_index` dtypes instead of silently casting.
  - `GatedEquivariantBlock` now enforces strict integer `n_radial_basis` and finite positive `max_radius`.
- `atlas/models/cgcnn.py`:
  - replaced permissive integer casting with strict integer semantics for dimensions/layer counts.
  - tightened dropout validation (finite + range + boolean-like rejection).
  - strengthened graph input validation with finite checks, edge-index dtype/range checks, and batch-index dtype/non-negative checks.
- `atlas/models/multi_task.py`:
  - normalized task-head keys to string consistently when registering heads.
  - added explicit unknown-task rejection in `forward(...)` (no more silent skip of missing tasks).
- tests:
  - `tests/unit/models/test_cgcnn.py` adds regressions for fractional/boolean hyperparameters, invalid edge-index dtype, out-of-range indices, non-finite node/edge features, and strict `MessagePassingLayer` checks.
  - `tests/unit/models/test_multi_task.py` adds regression for unknown selected task failure.

## Verification
- `python -m ruff check atlas/models/layers.py atlas/models/cgcnn.py atlas/models/multi_task.py tests/unit/models/test_cgcnn.py tests/unit/models/test_multi_task.py`
- `python -m pytest -q tests/unit/models/test_cgcnn.py tests/unit/models/test_multi_task.py`
- `python -m pytest -q tests/unit/models`

## Research references used in batch 87
- CGCNN (PRL 2018): https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301
- e3nn `FullyConnectedTensorProduct` API: https://docs.e3nn.org/en/stable/api/o3/o3_tp.html#e3nn.o3.FullyConnectedTensorProduct
- PyTorch `BatchNorm1d` API: https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
- PyTorch `index_add` API: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_add.html
- PyTorch `Dropout` API: https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
- PyTorch Geometric `global_mean_pool`: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_mean_pool.html
- PyTorch Geometric `global_max_pool`: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_max_pool.html
- PyTorch Geometric `global_add_pool`: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_add_pool.html
- Python numeric ABCs (`numbers.Integral` / `numbers.Real`): https://docs.python.org/3/library/numbers.html
- Python bool semantics: https://docs.python.org/3/library/stdtypes.html#boolean-type-bool
- Multi-Task Uncertainty Weighting (CVPR 2018): https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html

## Progress snapshot (after Batch 87)
- Completed: Batch 1 through Batch 87.
- Pending: Batch 88 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 88.

## Batch 88 (max 5 files)
- [x] `atlas/models/equivariant.py` - reviewed + optimized
- [x] `atlas/models/fast_tp.py` - reviewed + optimized
- [x] `atlas/models/m3gnet.py` - reviewed + optimized
- [x] `tests/unit/models/test_classifier.py` - reviewed + optimized
- [x] `tests/unit/models/test_fast_tp.py` - reviewed + optimized (new)

## Batch 88 optimization goals
- Remove silent bool/fractional coercion in equivariant/M3GNet hyperparameter and graph-index validation paths.
- Prevent hidden float-to-long index truncation in fused tensor-product scatter kernels.
- Add regression coverage for strict integer semantics and edge-index safety in equivariant-ecosystem modules.

## Batch 88 outcomes
- `atlas/models/equivariant.py`:
  - added strict scalar coercion helpers (`_coerce_non_negative_int`, `_coerce_positive_int`, `_coerce_positive_float`).
  - hardened `AtomRef`, `BesselRadialBasis`, and `EquivariantGNN` constructor validation against bool/fractional/non-finite inputs.
  - strengthened `encode(...)` validation: integer edge-index dtype, non-empty edges, edge-index range checks, and strict batch shape/dtype checks.
  - normalized usage of validated config values (`self.max_ell`, validated output dims) for internal module construction.
- `atlas/models/fast_tp.py`:
  - added strict integer coercion for `num_nodes` and explicit integer dtype validation for `edge_src`/`edge_dst`.
  - added shape and finite-value validation for `x`, `edge_attr`, and `edge_weight` before tensor-product execution.
  - added explicit guard for invalid `num_nodes==0` when edges are present.
- `atlas/models/m3gnet.py`:
  - added strict scalar coercion helpers and applied to `RBFExpansion`, `ThreeBodyInteraction`, and `M3GNet` constructors.
  - hardened `ThreeBodyInteraction.forward(...)` with shape/dtype/range checks for 3-body edge indices and edge-vector consistency.
  - hardened `M3GNetLayer.forward(...)` and `M3GNet.encode(...)` with integer edge-index checks, edge-index range checks, and strict batch/3body dtype checks.
- tests:
  - `tests/unit/models/test_classifier.py` adds regressions for fractional/boolean hyperparameter rejection and invalid edge-index dtype/range behavior in `M3GNet`.
  - new `tests/unit/models/test_fast_tp.py` adds coverage for fused TP scatter dtype and finite-value guards.

## Verification
- `python -m ruff check atlas/models/equivariant.py atlas/models/fast_tp.py atlas/models/m3gnet.py tests/unit/models/test_classifier.py tests/unit/models/test_fast_tp.py`
- `python -m pytest -q tests/unit/models/test_classifier.py tests/unit/models/test_fast_tp.py`
- `python -m pytest -q tests/unit/models`

## Research references used in batch 88
- NequIP paper (Nature Communications 2022): https://www.nature.com/articles/s41467-022-29939-5
- e3nn documentation (Irreps/TensorProduct APIs): https://docs.e3nn.org/
- M3GNet paper (Nature Computational Science 2022): https://www.nature.com/articles/s43588-022-00349-3
- M3GNet reference implementation docs: https://materialsvirtuallab.github.io/m3gnet/index.html
- PyTorch `index_add_` API: https://pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html
- PyTorch `nan_to_num` API: https://pytorch.org/docs/stable/generated/torch.nan_to_num.html
- Python numeric ABCs (`numbers.Integral` / `numbers.Real`): https://docs.python.org/3/library/numbers.html
- Python bool semantics: https://docs.python.org/3/library/stdtypes.html#boolean-type-bool

## Progress snapshot (after Batch 88)
- Completed: Batch 1 through Batch 88.
- Pending: Batch 89 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 89.

## Batch 89 (max 5 files)
- [x] `atlas/models/matgl_three_body.py` - reviewed + optimized
- [x] `atlas/models/m3gnet.py` - reviewed + optimized
- [x] `tests/unit/models/test_classifier.py` - reviewed + optimized
- [x] `tests/unit/models/test_fast_tp.py` - reviewed + optimized
- [x] `tests/unit/models/test_matgl_three_body.py` - reviewed + optimized (new)

## Batch 89 optimization goals
- Close remaining bool/fractional scalar-coercion gaps in M3GNet three-body basis helpers.
- Eliminate residual raw-parameter usage in `ThreeBodyInteraction` / `M3GNetLayer` constructors to prevent hidden type drift.
- Expand regression tests for three-body basis rank/shape checks and strict `num_nodes` validation in fused tensor-product paths.

## Batch 89 outcomes
- `atlas/models/matgl_three_body.py`:
  - added strict positive-integer coercion helper and applied to `n_basis`, `max_n`, `max_l`.
  - added rank checks (`rank-2`) for angle-expansion inputs before shape checks.
  - maintains non-finite sanitization path while now failing fast on invalid tensor layout.
- `atlas/models/m3gnet.py`:
  - `ThreeBodyInteraction` and `M3GNetLayer` now consistently use validated internal dimensions (`self.embed_dim`, `self.n_basis`) for all linear layers.
  - removes residual dependency on unvalidated constructor arguments, preventing silent float/bool propagation into layer construction.
- tests:
  - `tests/unit/models/test_classifier.py` adds regressions for invalid `edge_index_3body` dtype, invalid batch dtype, and strict init checks on `M3GNetLayer`.
  - `tests/unit/models/test_fast_tp.py` adds parameterized regression for invalid `num_nodes` payloads (`float`, `bool`, `inf`, `nan`).
  - new `tests/unit/models/test_matgl_three_body.py` adds constructor and forward-path coverage for strict scalar/rank/shape behavior and non-finite sanitization.

## Verification
- `python -m ruff check atlas/models/matgl_three_body.py atlas/models/m3gnet.py tests/unit/models/test_classifier.py tests/unit/models/test_matgl_three_body.py tests/unit/models/test_fast_tp.py`
- `python -m pytest -q tests/unit/models/test_classifier.py tests/unit/models/test_matgl_three_body.py tests/unit/models/test_fast_tp.py`
- `python -m pytest -q tests/unit/models`

## Research references used in batch 89
- M3GNet paper (Nature Computational Science 2022): https://www.nature.com/articles/s43588-022-00349-3
- MatGL documentation/index: https://matgl.ai/
- e3nn documentation (equivariant tensor products): https://docs.e3nn.org/
- NequIP paper (Nature Communications 2022): https://www.nature.com/articles/s41467-022-29939-5
- PyTorch `index_add_` API: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html
- PyTorch `nan_to_num` API: https://docs.pytorch.org/docs/stable/generated/torch.nan_to_num.html
- Python numeric ABCs (`numbers.Integral` / `numbers.Real`): https://docs.python.org/3/library/numbers.html
- Python bool semantics: https://docs.python.org/3/library/stdtypes.html#boolean-type-bool

## Progress snapshot (after Batch 89)
- Completed: Batch 1 through Batch 89.
- Pending: Batch 90 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 90.

## Batch 90 (max 5 files)
- [x] `atlas/research/method_registry.py` - reviewed + optimized
- [x] `atlas/research/workflow_reproducible_graph.py` - reviewed + optimized
- [x] `atlas/utils/reproducibility.py` - reviewed (no code changes in this batch)
- [x] `tests/unit/research/test_method_registry.py` - reviewed + optimized
- [x] `tests/unit/research/test_workflow_reproducible_graph.py` - reviewed + optimized

## Batch 90 optimization goals
- Harden method/strategy metadata normalization to eliminate whitespace and iterable-string pitfalls.
- Tighten reproducible workflow graph numeric semantics (no bool-as-int, no silent fractional truncation).
- Add regression coverage for strict integer validation and stage-plan typing constraints.

## Batch 90 outcomes
- `atlas/research/method_registry.py`:
  - added `_normalize_lookup_key(...)` and applied to `MethodSpec.__post_init__` and registry lookups.
  - fixed `_normalize_text_items(...)` to treat raw strings as single entries (instead of char-wise iteration).
  - `MethodRegistry.get(...)` now strips/normalizes keys; `recommended_method_order(...)` now normalizes `primary`.
- `atlas/research/workflow_reproducible_graph.py`:
  - strengthened scalar coercion with `numbers.Integral`/`numbers.Real` semantics.
  - rejected bool-like numeric payloads for count/cost fields.
  - rejected fractional counts and non-finite values; stage-plan now rejects raw string payloads.
- tests:
  - `tests/unit/research/test_method_registry.py` adds regressions for string normalization and whitespace-safe key lookup.
  - `tests/unit/research/test_workflow_reproducible_graph.py` adds regressions for bool/fractional count rejection and stage-plan type guard.

## Verification
- `python -m ruff check atlas/research/method_registry.py atlas/research/workflow_reproducible_graph.py atlas/utils/reproducibility.py tests/unit/research/test_method_registry.py tests/unit/research/test_workflow_reproducible_graph.py`
- `python -m pytest -q tests/unit/research/test_method_registry.py tests/unit/research/test_workflow_reproducible_graph.py tests/unit/research/test_reproducibility.py`

## Research references used in batch 90
- W3C PROV Overview: https://www.w3.org/TR/prov-overview/
- ACM Artifact Review and Badging: https://www.acm.org/publications/policies/artifact-review-and-badging-current
- NeurIPS Reproducibility Checklist: https://neurips.cc/public/guides/CodeSubmissionPolicy
- Python `numbers` module docs: https://docs.python.org/3/library/numbers.html
- Python `dataclasses` module docs: https://docs.python.org/3/library/dataclasses.html

## Progress snapshot (after Batch 90)
- Completed: Batch 1 through Batch 90.
- Pending: Batch 91 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 91.

## Batch 91 (max 5 files)
- [x] `atlas/utils/registry.py` - reviewed + optimized
- [x] `atlas/utils/reproducibility.py` - reviewed + optimized
- [x] `atlas/utils/__init__.py` - reviewed + optimized
- [x] `tests/unit/research/test_reproducibility.py` - reviewed + optimized
- [x] `tests/unit/research/test_utils_registry.py` - reviewed + optimized (new)

## Batch 91 optimization goals
- Make component registry key handling deterministic and strict (strip/validate/non-empty).
- Ensure deterministic mode always applies deterministic CUBLAS workspace config.
- Backfill regression tests for registry normalization and deterministic-runtime env behavior.

## Batch 91 outcomes
- `atlas/utils/registry.py`:
  - added strict key normalization (`_normalize_registry_key`) for registry and entry names.
  - improved overwrite logs to lazy logger formatting.
  - `get(...)` now returns deterministic sorted available entries in errors.
  - `build(...)` now supports positional args + kwargs.
  - added `registered_names()` for deterministic diagnostics.
  - hardened `__contains__` for non-string/whitespace payloads.
- `atlas/utils/reproducibility.py`:
  - deterministic mode now enforces known deterministic `CUBLAS_WORKSPACE_CONFIG` when env has incompatible custom values.
  - keeps compatibility by preserving known deterministic configs (`:16:8`, `:4096:8`).
- `atlas/utils/__init__.py`:
  - exports `Registry` and global registries (`MODELS`, `RELAXERS`, `FEATURE_EXTRACTORS`, `EVALUATORS`) through package public API.
- tests:
  - `tests/unit/research/test_reproducibility.py` adds deterministic CUBLAS override/preserve regressions.
  - new `tests/unit/research/test_utils_registry.py` validates key normalization, deterministic name listing, invalid-input guards, and error diagnostics.

## Verification
- `python -m ruff check atlas/utils/registry.py atlas/utils/reproducibility.py atlas/utils/__init__.py tests/unit/research/test_reproducibility.py tests/unit/research/test_utils_registry.py atlas/research/method_registry.py atlas/research/workflow_reproducible_graph.py tests/unit/research/test_method_registry.py tests/unit/research/test_workflow_reproducible_graph.py`
- `python -m pytest -q tests/unit/research/test_method_registry.py tests/unit/research/test_workflow_reproducible_graph.py tests/unit/research/test_reproducibility.py tests/unit/research/test_utils_registry.py`
- `python -m pytest -q tests/unit/research`

## Research references used in batch 91
- PyTorch Reproducibility notes: https://pytorch.org/docs/stable/notes/randomness.html
- NVIDIA cuBLAS reproducibility guidance: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
- Python logging cookbook (lazy formatting patterns): https://docs.python.org/3/howto/logging.html
- Pluggy plugin system docs (production registry/factory patterns): https://pluggy.readthedocs.io/en/stable/
- Python typing docs (`Any`, callables, runtime typing constraints): https://docs.python.org/3/library/typing.html

## Progress snapshot (after Batch 91)
- Completed: Batch 1 through Batch 91.
- Pending: Batch 92 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 92.

## Batch 92 (max 5 files)
- [x] `atlas/training/theory_tuning.py` - reviewed + optimized
- [x] `atlas/utils/structure.py` - reviewed + optimized
- [x] `atlas/research/__init__.py` - reviewed (no code change required)
- [x] `tests/unit/research/test_theory_tuning.py` - reviewed + optimized
- [x] `tests/unit/models/test_structure_utils.py` - reviewed + optimized

## Batch 92 optimization goals
- Harden theory-tuning objective/profile validation to prevent silent invalid configuration drift.
- Enforce finite-score semantics in adaptation and metric extraction paths (no bool-as-number, no non-finite leakage).
- Improve structural feature extraction robustness for sparse-neighbor cases and strict integer input contracts.

## Batch 92 outcomes
- `atlas/training/theory_tuning.py`:
  - `MetricObjective` now validates keys/mode/threshold in `__post_init__`.
  - `TheoryProfile` now normalizes phase/algorithm/stage and validates container fields.
  - `get_profile(...)` now supports whitespace/case normalization for stable lookup.
  - `adapt_params_for_next_round(...)` now validates payload types, handles failed runs early, and rejects non-finite scores in non-failed paths.
  - `extract_score_from_manifest(...)` now ignores boolean and non-finite metric values.
  - numeric adjustment path now avoids touching boolean params and no-op factor/delta calls.
- `atlas/utils/structure.py`:
  - strengthened type hints and integer coercion (`_coerce_non_negative_int`) for sampling utilities.
  - `_sample_site_indices(...)` now rejects bool/fractional/invalid integer payloads.
  - `compute_structural_features(...)` now uses multi-radius nearest-neighbor fallback (`4A -> 8A`) via `_closest_neighbor_distance(...)` to reduce false-zero neighbor-distance artifacts.
- tests:
  - `tests/unit/research/test_theory_tuning.py` adds regressions for case-insensitive profile lookup, invalid objective config rejection, bool/non-finite metric filtering, failed-run recovery behavior, and non-finite previous-score rejection.
  - `tests/unit/models/test_structure_utils.py` adds regressions for strict integer contracts in `_sample_site_indices(...)` and neighbor-radius fallback behavior in structural features.

## Verification
- `python -m ruff check atlas/training/theory_tuning.py atlas/utils/structure.py atlas/research/__init__.py tests/unit/research/test_theory_tuning.py tests/unit/models/test_structure_utils.py`
- `python -m pytest -q tests/unit/research/test_theory_tuning.py tests/unit/models/test_structure_utils.py`

## Research references used in batch 92
- Hyperband (JMLR 2018): https://jmlr.org/papers/v18/16-558.html
- Random Search for Hyper-Parameter Optimization (JMLR 2012): https://www.jmlr.org/papers/v13/bergstra12a.html
- Pymatgen `Structure` API docs: https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure
- Pymatgen `SpacegroupAnalyzer` API docs: https://pymatgen.org/pymatgen.symmetry.html#pymatgen.symmetry.analyzer.SpacegroupAnalyzer
- NumPy `isfinite` reference: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- Python bool semantics: https://docs.python.org/3/library/stdtypes.html#boolean-type-bool

## Progress snapshot (after Batch 92)
- Completed: Batch 1 through Batch 92.
- Pending: Batch 93 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 93.

## Batch 93 (max 5 files)
- [x] `atlas/data/split_governance.py` - reviewed + optimized
- [x] `atlas/data/source_registry.py` - reviewed + optimized
- [x] `atlas/data/data_validation.py` - reviewed + optimized
- [x] `tests/unit/data/test_split_governance.py` - reviewed + optimized
- [x] `tests/unit/data/test_source_registry.py` - reviewed + optimized

## Batch 93 optimization goals
- Tighten split-governance input contracts for IDs/ratios/hyperparameters to avoid silent nondeterminism.
- Harden source-registry key normalization and reliability/correlation stability under malformed inputs.
- Improve validation pipeline integrity hashing and JSONL diagnostics.

## Batch 93 outcomes
- `atlas/data/split_governance.py`:
  - added strict finite/non-boolean guards in `_normalized_ratios(...)`.
  - replaced raw ID uniqueness check with normalized ID canonicalization (`_normalize_and_validate_sample_ids(...)`) so whitespace collisions are rejected deterministically.
  - tightened search hyperparameter coercion in `_normalize_search_hyperparams(...)` (reject bool/fractional values; keep `n_restarts=0 -> 1` behavior).
  - normalized formula and spacegroup grouping inputs to reduce split drift from formatting noise.
  - `build_assignment_records(...)` now uses normalized sample-id keys when resolving groups.
  - `generate_manifest(...)` now validates strategy name and emits split blocks in deterministic order.
- `atlas/data/source_registry.py`:
  - added `DataSourceSpec.__post_init__` normalization/validation for key/name/domain/targets/url/citation.
  - added normalized key path in registry `register/get/get_reliability`.
  - `restore_reliability(...)` now sanitizes restored alpha/beta to positive finite values.
  - `estimate_correlation_matrix(...)` now enforces unique source keys.
  - `fuse_scalar_estimates(...)` now normalizes source keys and disambiguates duplicate-source weight keys via suffixing (`key#idx`) to prevent weight-map overwrite.
- `atlas/data/data_validation.py`:
  - upgraded `_stable_hash(...)` from MD5 to SHA-256 for stronger deterministic integrity fingerprints.
  - `_load_records_from_input(...)` now reports JSONL parse errors with exact line numbers.
- tests:
  - `tests/unit/data/test_split_governance.py` adds regressions for finite-ratio enforcement, whitespace-ID collision rejection, fractional `n_restarts` rejection, spacegroup normalization behavior, unknown strategy rejection, and strict hyperparameter coercion.
  - `tests/unit/data/test_source_registry.py` adds regressions for spec key/target normalization, duplicate-source-key rejection in correlation estimation, and duplicate-source weight disambiguation in fusion output.

## Verification
- `python -m ruff check atlas/data/split_governance.py atlas/data/source_registry.py atlas/data/data_validation.py tests/unit/data/test_split_governance.py tests/unit/data/test_source_registry.py`
- `python -m pytest -q tests/unit/data/test_split_governance.py tests/unit/data/test_source_registry.py tests/unit/data/test_data_validation.py`

## Research references used in batch 93
- DataSAIL (Nature Communications 2025): https://www.nature.com/articles/s41467-025-58606-8
- Domain adaptation theory (Ben-David et al., Machine Learning 2010): https://link.springer.com/article/10.1007/s10994-009-5152-4
- Kernel two-sample test / MMD (Gretton et al., JMLR 2012): https://www.jmlr.org/beta/papers/v13/gretton12a.html
- Group-aware splitting reference (`GroupShuffleSplit`): https://sklearn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
- NumPy deterministic permutation reference: https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html
- SciPy KS two-sample test reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
- JSON Lines format reference: https://jsonlines.org/
- Ledoit-Wolf shrinkage reference (original paper mirror): https://ledoit.net/ole1a.pdf

## Progress snapshot (after Batch 93)
- Completed: Batch 1 through Batch 93.
- Pending: Batch 94 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 94.

## Batch 94 (max 5 files)
- [x] `atlas/data/crystal_dataset.py` - reviewed + optimized
- [x] `atlas/data/jarvis_client.py` - reviewed + optimized
- [x] `atlas/data/property_estimator.py` - reviewed (no code change required)
- [x] `tests/unit/data/test_crystal_dataset.py` - reviewed + optimized
- [x] `tests/unit/data/test_jarvis_client.py` - reviewed + optimized

## Batch 94 optimization goals
- Tighten dataset split/config contracts to prevent hidden bool/fractional coercion drift in active-learning runs.
- Improve JARVIS download integrity and IO atomicity to avoid partial/corrupted cache artifacts under network instability.
- Strengthen manifest assignment validation so conflicting split labels fail fast instead of silent overwrite.

## Batch 94 outcomes
- `atlas/data/crystal_dataset.py`:
  - added strict coercion helpers for positive integers and seed handling (`_coerce_optional_positive_int`, `_coerce_seed`).
  - hardened `_normalize_split_ratio(...)` to reject boolean and non-finite ratio payloads.
  - constructor now uses explicit coercion for `max_samples`, `split_seed`, `min_labeled_properties`, and `graph_max_neighbors`.
  - `_load_manifest_assignment(...)` now raises `ValueError` when duplicate `sample_id` rows carry conflicting split labels.
- `atlas/data/jarvis_client.py`:
  - `_download_file(...)` now validates retry bounds, streams within context manager, writes via temporary `.part` file, atomically replaces destination, and validates `content-length` consistency.
  - strengthened cleanup semantics for failed downloads (temporary file removal in `finally`).
  - `get_structure(...)` now normalizes/validates `jid`, validates presence of `jid` column, and performs normalized lookup.
- `atlas/data/property_estimator.py`:
  - reviewed in this batch; behavior and tests already aligned with current contract, so no direct code change applied.
- tests:
  - `tests/unit/data/test_crystal_dataset.py` adds contract regressions for boolean split ratio, fractional integer-typed fields, and conflicting manifest assignments.
  - `tests/unit/data/test_jarvis_client.py` adds regressions for whitespace-normalized `jid` lookup and incomplete-download detection.

## Verification
- `python -m ruff check atlas/data/crystal_dataset.py atlas/data/jarvis_client.py tests/unit/data/test_crystal_dataset.py tests/unit/data/test_jarvis_client.py`
- `python -m pytest -q tests/unit/data/test_crystal_dataset.py tests/unit/data/test_jarvis_client.py`
- `python -m pytest -q tests/unit/data/test_property_estimator.py`

## Research references used in batch 94
- Requests developer API (`Response.iter_content`, streaming usage): https://docs.python-requests.org/en/latest/api/
- Python `tempfile.NamedTemporaryFile` (secure temporary file patterns): https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile
- Python `pathlib.Path.replace` (atomic replace-oriented semantics at API level): https://docs.python.org/3/library/pathlib.html#pathlib.Path.replace
- JARVIS infrastructure paper (npj Comput. Mater. 2020): https://doi.org/10.1038/s41524-020-00440-1
- NIST JARVIS-DFT project page: https://www.nist.gov/programs-projects/jarvis-dft

## Progress snapshot (after Batch 94)
- Completed: Batch 1 through Batch 94.
- Pending: Batch 95 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 95.


## Batch 95 (max 5 files)
- [x] `atlas/data/topo_db.py` - reviewed + optimized
- [x] `atlas/data/alloy_estimator.py` - reviewed + optimized
- [x] `atlas/data/__init__.py` - reviewed (no code change required)
- [x] `tests/unit/data/test_topo_db.py` - reviewed + optimized
- [x] `tests/unit/data/test_alloy_estimator.py` - reviewed + optimized

## Batch 95 optimization goals
- Strengthen TopoDB API input contracts to reduce silent coercion drift (thresholds/ranges/integer knobs).
- Harden material insertion and fuzzy/query paths against malformed identifiers and non-finite parameters.
- Improve AlloyEstimator custom-construction validation while preserving legacy sanitization semantics for noisy weights.

## Batch 95 outcomes
- `atlas/data/topo_db.py`:
  - added shared coercion helpers for robust numeric/int/string parsing (`_as_nonempty_str`, `_coerce_finite_float`, `_coerce_int_like`).
  - normalized string columns (`jid`, `formula`, `source`, `topo_class`) with `.str.strip()` in `_normalize_frame(...)` to reduce whitespace-induced drift.
  - `calibrate_channel_reliability(...)` now validates `min_samples` and `corr_shrinkage` range before calibration.
  - `infer_topology_probabilities(...)` now validates critical scalar and integer hyperparameters (`prior_alpha/beta`, `evidence_strength`, `corr_shrinkage`, `correlation_penalty`, `calibration_bins/folds/seed`, calibration sample thresholds, `decision_threshold`, `ood_scale`).
  - `rank_topological_candidates(...)` now validates probability/OOD ranges and `top_k` integer domain.
  - `fuzzy_search(...)` now validates `query` and `cutoff in [0,1]`.
  - `query(...)` now validates `band_gap_range` shape/order and normalizes/cleans element inputs.
  - `add_material(...)` now enforces non-empty formula, normalizes `jid/source/topo_class`, validates integer-like `space_group`, and handles missing/non-finite `band_gap` deterministically.
- `atlas/data/alloy_estimator.py`:
  - constructor now explicitly rejects empty phase lists early.
  - `from_preset(...)` now validates non-empty string preset input before lookup.
  - added `_coerce_finite_real(...)` for explicit numeric contract validation (with optional non-finite allowance).
  - `custom(...)` now validates phase payload structure (`name`, `formula`, `properties` mapping) and rejects non-numeric/bool property values with precise field paths.
  - keeps prior behavior for noisy/non-finite phase weights (still sanitized downstream), preserving existing regression intent.
- tests:
  - `tests/unit/data/test_topo_db.py` adds regressions for cutoff validation, band-gap range validation, empty-formula rejection, trim normalization in `add_material`, ranking parameter validation, and calibration integer-parameter validation.
  - `tests/unit/data/test_alloy_estimator.py` adds regressions for preset input validation, required non-empty custom phase list, and rejection of bool/non-numeric custom property payloads.

## Verification
- `python -m ruff check atlas/data/topo_db.py atlas/data/alloy_estimator.py tests/unit/data/test_topo_db.py tests/unit/data/test_alloy_estimator.py`
- `python -m pytest -q tests/unit/data/test_topo_db.py tests/unit/data/test_alloy_estimator.py`
- `python -m pytest -q tests/unit/data`

## Research references used in batch 95
- Python `difflib.get_close_matches` reference: https://docs.python.org/3/library/difflib.html#difflib.get_close_matches
- NumPy finite-value guard reference (`numpy.isfinite`): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- pandas numeric coercion reference (`pandas.to_numeric`): https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html
- Python bool type semantics (bool is int subtype): https://docs.python.org/3/library/stdtypes.html#boolean-type-bool
- Voigt-Reuss-Hill averaging background (Hill, 1952 DOI): https://doi.org/10.1088/0370-1298/65/5/307
- Scoring-rule calibration context (Guo et al., 2017): https://proceedings.mlr.press/v70/guo17a.html

## Progress snapshot (after Batch 95)
- Completed: Batch 1 through Batch 95.
- Pending: Batch 96 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 96.

## Batch 96 (max 5 files)
- [x] `atlas/data/property_estimator.py` - reviewed + optimized
- [x] `atlas/config.py` - reviewed + optimized
- [x] `atlas/benchmark/runner.py` - reviewed (no code change required in this batch)
- [x] `tests/unit/data/test_property_estimator.py` - reviewed + optimized
- [x] `tests/unit/config/test_config.py` - reviewed + optimized

## Batch 96 optimization goals
- Tighten estimator/search API contracts to prevent silent malformed-criteria behavior.
- Improve numerical robustness for derived mechanical features under non-physical or corrupted elastic inputs.
- Make configuration dataclasses more schema-driven via post-init coercion/normalization while preserving backward-compatible defaults.

## Batch 96 outcomes
- `atlas/data/property_estimator.py`:
  - added `_coerce_search_limit(...)` to normalize non-integer/bool/invalid `max_results` payloads.
  - hardened `search(...)` input contracts: `criteria` must be mapping, tuple-range must be exactly `(lo, hi)`, bounds must be finite numeric values, and `lo <= hi` is enforced.
  - improved hardness proxy stability: `hardness_chen` now computes only on physically valid `(K > 0, G > 0)` region and avoids non-physical power operations from invalid shear/bulk values.
- `atlas/config.py`:
  - added reusable coercion helpers (`_coerce_non_negative_int`, `_coerce_int`, `_coerce_positive_float`, `_coerce_bool`, `_coerce_nonempty_string`).
  - `PathConfig._normalize_path(...)` now rejects bool path candidates and treats blank string path overrides as default fallback instead of ambiguous relative paths.
  - added dataclass post-init normalization for `DFTConfig`, `MACEConfig`, `TrainConfig`, and `ProfileConfig` to stabilize types/ranges and reduce runtime drift from malformed external inputs.
  - `ProfileConfig.fallback_methods` normalized to non-empty canonical tuple with safe default fallback list.
- `atlas/benchmark/runner.py`:
  - reviewed for input coercion/reporting pipeline; existing coercion and uncertainty-reporting contract already aligned with current reliability goals, so no direct code change in this batch.
- tests:
  - `tests/unit/data/test_property_estimator.py` adds regressions for invalid criteria payload, malformed range tuple, descending numeric ranges, and non-physical hardness handling.
  - `tests/unit/config/test_config.py` adds regressions for bool path rejection, train-config coercion behavior, and profile fallback-method normalization.

## Verification
- `python -m ruff check atlas/data/property_estimator.py atlas/config.py tests/unit/data/test_property_estimator.py tests/unit/config/test_config.py`
- `python -m pytest -q tests/unit/data/test_property_estimator.py tests/unit/config/test_config.py`
- `python -m pytest -q tests/unit/config`
- `python -m pytest -q tests/unit/data`

## Research references used in batch 96
- Python dataclasses (`__post_init__` and dataclass semantics): https://docs.python.org/3/library/dataclasses.html
- Python pathlib (path normalization and expansion behavior): https://docs.python.org/3/library/pathlib.html
- NumPy finite-value guards (`isfinite`): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- pandas numeric coercion (`to_numeric`): https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html
- Calibration reference (temperature scaling): https://proceedings.mlr.press/v70/guo17a.html
- Thermal-conductivity scaling context (Slack-related literature): https://doi.org/10.1103/PhysRevB.7.5379

## Progress snapshot (after Batch 96)
- Completed: Batch 1 through Batch 96.
- Pending: Batch 97 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 97.

## Batch 97 (max 5 files)
- [x] `atlas/benchmark/cli.py` - reviewed + optimized
- [x] `atlas/benchmark/__init__.py` - reviewed (no code change required)
- [x] `tests/unit/benchmark/test_benchmark_cli.py` - reviewed + optimized
- [x] `tests/unit/benchmark/test_benchmark_runner.py` - reviewed (no code change required)
- [x] `tests/unit/benchmark/test_benchmark_init.py` - reviewed (no code change required)

## Batch 97 optimization goals
- Harden benchmark CLI argument contracts to reject bool/non-finite probability payloads and prevent ambiguous whitespace inputs.
- Reduce false validation coupling between preflight and non-preflight flows.
- Extend benchmark CLI regression tests to lock strict parser behavior and normalization guarantees.

## Batch 97 outcomes
- `atlas/benchmark/cli.py`:
  - added `_normalize_optional_text(...)` to canonicalize string-like CLI fields before validation.
  - added `_coerce_probability_in_range(...)` with strict finite-float validation (reject bool/NaN/Inf) and explicit inclusive/exclusive bound handling.
  - `_validate_cli_args(...)` now normalizes key string args (`task/property/model_module/model_class/checkpoint/output/output_dir/preflight_property_group`) before downstream checks.
  - probability controls now use strict coercion/validation for `--min-coverage-required` and `--conformal-coverage`.
  - preflight property-group validation now only applies when preflight is actually enabled (avoids irrelevant rejection when `--skip-preflight --dry-run`).
- `tests/unit/benchmark/test_benchmark_cli.py`:
  - added regressions for non-finite probability rejection, bool probability rejection, skip-preflight property-group bypass, and normalized whitespace task handling.
- `atlas/benchmark/__init__.py`, `tests/unit/benchmark/test_benchmark_runner.py`, `tests/unit/benchmark/test_benchmark_init.py`:
  - reviewed in this batch; existing behavior remained aligned with current benchmark-core contract, so no direct code edits.

## Verification
- `python -m ruff check atlas/benchmark/cli.py atlas/benchmark/__init__.py tests/unit/benchmark/test_benchmark_cli.py tests/unit/benchmark/test_benchmark_runner.py tests/unit/benchmark/test_benchmark_init.py`
- `python -m pytest -q tests/unit/benchmark/test_benchmark_cli.py tests/unit/benchmark/test_benchmark_runner.py tests/unit/benchmark/test_benchmark_init.py`
- `python -m pytest -q tests/unit/benchmark`

## Research references used in batch 97
- Python `argparse` documentation: https://docs.python.org/3/library/argparse.html
- Python `json` documentation: https://docs.python.org/3/library/json.html
- PyTorch serialization notes (`torch.load`, checkpoint safety): https://docs.pytorch.org/docs/stable/notes/serialization.html
- PyTorch `load_state_dict` API semantics: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict
- Matbench benchmark paper (Dunn et al., 2020): https://arxiv.org/abs/2007.04256
- Conformal prediction tutorial (Angelopoulos & Bates, 2021): https://arxiv.org/abs/2107.07511

## Progress snapshot (after Batch 97)
- Completed: Batch 1 through Batch 97.
- Pending: Batch 98 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 98.

## Batch 98 (max 5 files)
- [x] `atlas/models/prediction_utils.py` - reviewed + optimized
- [x] `atlas/models/uncertainty.py` - reviewed + optimized
- [x] `atlas/models/utils.py` - reviewed (no code change required)
- [x] `tests/unit/models/test_prediction_utils.py` - reviewed + optimized
- [x] `tests/unit/models/test_uncertainty.py` - reviewed + optimized

## Batch 98 optimization goals
- Strengthen prediction payload normalization so dtype/device consistency and uncertainty-shape semantics are explicit.
- Prevent silent/brittle stacking behavior in UQ paths by validating per-task output shape consistency across ensemble/MC samples.
- Add regression coverage for broadcast-safe std handling and shape-mismatch failure modes.

## Batch 98 outcomes
- `atlas/models/prediction_utils.py`:
  - `_to_tensor(...)` now aligns tensor dtype/device with reference tensors when provided.
  - added `_broadcast_to_reference(...)` to enforce controlled broadcast semantics (allow broadcast to mean-shape only; reject expansion that changes mean shape).
  - `_sanitize_std_like(...)` now uses strict broadcast-aware conversion, improving error diagnostics for malformed uncertainty tensors.
  - evidential path (`_from_evidential_payload`) now aligns/broadcast-validates `nu/alpha/beta` against `gamma` before uncertainty computation.
  - `extract_mean_and_std(...)` now sanitizes tensor/tuple/fallback mean payloads consistently (non-finite -> bounded finite values).
- `atlas/models/uncertainty.py`:
  - added `_validate_prediction_shapes(...)` to enforce consistent per-task tensor shapes across ensemble members / MC passes.
  - `EnsembleUQ.forward`, `EnsembleUQ.predict_with_uncertainty`, and `MCDropoutUQ.predict_with_uncertainty` now fail fast with explicit error when task-wise output shapes drift.
- `atlas/models/utils.py`:
  - reviewed in this batch; state-dict normalization and model reconstruction path already consistent with current runtime contracts, so no direct code change.
- tests:
  - `tests/unit/models/test_prediction_utils.py` adds regressions for non-finite tensor mean sanitization, scalar std broadcasting, rejection of std tensors that expand mean shape, and evidential aux-tensor dtype alignment.
  - `tests/unit/models/test_uncertainty.py` adds regressions for ensemble/MC shape-mismatch detection while preserving existing key-mismatch and payload-type checks.

## Verification
- `python -m ruff check atlas/models/prediction_utils.py atlas/models/uncertainty.py atlas/models/utils.py tests/unit/models/test_prediction_utils.py tests/unit/models/test_uncertainty.py`
- `python -m pytest -q tests/unit/models/test_prediction_utils.py tests/unit/models/test_uncertainty.py`
- `python -m pytest -q tests/unit/models`

## Research references used in batch 98
- PyTorch broadcasting semantics: https://docs.pytorch.org/docs/stable/notes/broadcasting.html
- PyTorch `torch.broadcast_tensors`: https://docs.pytorch.org/docs/stable/generated/torch.broadcast_tensors.html
- PyTorch `torch.stack`: https://docs.pytorch.org/docs/stable/generated/torch.stack.html
- Deep Ensembles (Lakshminarayanan et al., NeurIPS 2017): https://arxiv.org/abs/1612.01474
- MC Dropout as Bayesian Approximation (Gal & Ghahramani, ICML 2016): https://proceedings.mlr.press/v48/gal16.html
- Deep Evidential Regression (Amini et al., NeurIPS 2020): https://papers.nips.cc/paper/2020/file/aab085461de182608ee9f607f3f7d18f-Paper.pdf

## Progress snapshot (after Batch 98)
- Completed: Batch 1 through Batch 98.
- Pending: Batch 99 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 99.
## Batch 99 (max 5 files)
- [x] `atlas/models/cgcnn.py` - reviewed (no code change required)
- [x] `atlas/models/equivariant.py` - reviewed (no code change required)
- [x] `atlas/models/layers.py` - reviewed (no code change required)
- [x] `atlas/models/multi_task.py` - reviewed + optimized
- [x] `atlas/models/graph_builder.py` - reviewed + optimized

## Batch 99 optimization goals
- Tighten multi-task head contracts so malformed task selections and malformed tensor-component payloads fail fast with explicit diagnostics.
- Harden graph-builder target attachment so downstream training receives finite, numeric tensors only.
- Keep architecture-level behavior unchanged for CGCNN/equivariant/layers while confirming current boundary checks remain consistent.

## Batch 99 outcomes
- `atlas/models/multi_task.py`:
  - `_normalize_task_names(...)` now enforces non-empty names, rejects non-iterable payloads with explicit errors, and de-duplicates while preserving order.
  - `MultiTaskGNN.__init__` now validates that `encoder` exposes a callable `encode(...)` method (fail-fast contract instead of deferred runtime `AttributeError`).
  - `TensorHead.to_full_tensor(...)` now validates component tensor rank/shape/finite values before reconstruction.
  - tensor reconstruction now preserves input dtype/device (`elastic` and `dielectric` use dtype-aware zeros; `piezoelectric` uses `reshape` for safer non-contiguous handling).
- `atlas/models/graph_builder.py`:
  - added `_coerce_bool(...)`; `compute_3body` now accepts strict boolean-like inputs (bool, 0/1, true/false-style strings) and rejects ambiguous values.
  - added `_coerce_property_tensor(...)` for `structure_to_pyg(...)` target attachment.
  - property targets now support scalar/list/NumPy/Tensor inputs but require non-empty finite numeric tensors; non-finite payloads fail fast with key-specific error.
- `atlas/models/cgcnn.py`, `atlas/models/equivariant.py`, `atlas/models/layers.py`:
  - reviewed for this batch; existing shape/index/type checks remain coherent with current runtime contracts, so no direct code edits in this batch.
- tests:
  - `tests/unit/models/test_multi_task.py` adds regressions for encoder contract validation, task-name normalization/de-duplication, empty-task rejection, and tensor-head component validation + dtype preservation.
  - `tests/unit/models/test_structure_expansion.py` adds regressions for compute_3body coercion and `structure_to_pyg` finite target validation.

## Verification
- `python -m ruff check atlas/models/multi_task.py atlas/models/graph_builder.py atlas/models/cgcnn.py atlas/models/equivariant.py atlas/models/layers.py tests/unit/models/test_multi_task.py tests/unit/models/test_structure_expansion.py`
- `python -m pytest -q tests/unit/models/test_multi_task.py tests/unit/models/test_structure_expansion.py`
- `python -m pytest -q tests/unit/models/test_cgcnn.py tests/unit/models/test_classifier.py tests/unit/models/test_multi_task.py tests/unit/models/test_structure_expansion.py`
- `python -m pytest -q tests/unit/models`

## Research references used in batch 99
- PyTorch `torch.isfinite`: https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- PyTorch `torch.reshape`: https://docs.pytorch.org/docs/stable/generated/torch.reshape.html
- PyTorch `torch.zeros` (dtype/device semantics): https://docs.pytorch.org/docs/stable/generated/torch.zeros.html
- PyTorch `torch.as_tensor` (tensor conversion semantics): https://docs.pytorch.org/docs/stable/generated/torch.as_tensor.html
- PyTorch Geometric `Data` API (custom attribute contracts): https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html
- M3GNet paper (three-body interactions context): https://arxiv.org/abs/2202.02450
- Multi-task uncertainty weighting reference (shared multi-task context): https://arxiv.org/abs/1705.07115

## Progress snapshot (after Batch 99)
- Completed: Batch 1 through Batch 99.
- Pending: Batch 100 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 100.
## Batch 100 (max 5 files)
- [x] `atlas/models/fast_tp.py` - reviewed + optimized
- [x] `atlas/models/m3gnet.py` - reviewed + optimized
- [x] `atlas/models/matgl_three_body.py` - reviewed + optimized
- [x] `atlas/models/__init__.py` - reviewed (no code change required)
- [x] `atlas/topology/classifier.py` - reviewed (no code change required)

## Batch 100 optimization goals
- Tighten tensor-product and three-body interaction contracts to fail fast on dtype/device/shape misuse before low-level kernel failures.
- Stabilize M3GNet three-body path by validating boolean-mode flags, triplet-edge bounds, and batch index constraints.
- Keep public export and topology classifier behavior stable while confirming current boundaries are coherent.

## Batch 100 outcomes
- `atlas/models/fast_tp.py`:
  - added strict floating-point checks for `x/edge_attr/edge_weight`.
  - added dtype alignment checks (`edge_attr` and `edge_weight` must match `x.dtype`).
  - added device alignment checks (`edge_attr/edge_weight/edge_src/edge_dst` must be on `x.device`).
  - removed unreachable negative-node branch in empty-edge path; kept deterministic zero-output behavior for `E=0`.
- `atlas/models/matgl_three_body.py`:
  - `SimpleMLPAngleExpansion.forward(...)` now enforces exact `(T, 1)` input-channel contract for `r_ij/r_ik/cos_theta`.
  - `SphericalBesselHarmonicsExpansion.forward(...)` now enforces `(T, 1)` channel contract and uses `reshape(...)` for robust shape normalization on non-contiguous tensors.
- `atlas/models/m3gnet.py`:
  - added `_coerce_bool(...)` and applied it to `ThreeBodyInteraction(use_sh=...)` for strict boolean-like parsing.
  - `ThreeBodyInteraction.forward(...)` now validates `edge_attr` rank/feature width (`embed_dim`) and finite payloads for both `edge_attr` and `edge_vectors`.
  - `M3GNet.encode(...)` now validates `edge_index_3body` bounds against current edge count before propagation.
  - `M3GNet.encode(...)` now rejects negative batch indices explicitly.
- `atlas/models/__init__.py`, `atlas/topology/classifier.py`:
  - reviewed in this batch; lazy export behavior and topology classifier runtime contracts remain consistent, so no direct code edits.
- tests:
  - `tests/unit/models/test_fast_tp.py` adds regressions for dtype mismatch and non-floating payload rejection.
  - `tests/unit/models/test_matgl_three_body.py` adds regressions for strict `(T,1)` angle-input channel validation.
  - `tests/unit/models/test_classifier.py` (M3GNet suite) adds regressions for negative batch rejection, out-of-range 3-body edge ids, invalid `use_sh` payload, and non-finite edge-vector rejection in three-body interaction.

## Verification
- `python -m ruff check atlas/models/fast_tp.py atlas/models/m3gnet.py atlas/models/matgl_three_body.py atlas/models/__init__.py atlas/topology/classifier.py tests/unit/models/test_fast_tp.py tests/unit/models/test_matgl_three_body.py tests/unit/models/test_classifier.py`
- `python -m pytest -q tests/unit/models/test_fast_tp.py tests/unit/models/test_matgl_three_body.py tests/unit/models/test_classifier.py`
- `python -m pytest -q tests/unit/models`
- `python -m pytest -q tests/unit/topology/test_classifier.py`

## Research references used in batch 100
- e3nn `FullyConnectedTensorProduct` API: https://docs.e3nn.org/en/stable/api/o3/o3_tp.html
- e3nn batch operations / scatter patterns: https://docs.e3nn.org/en/stable/api/math/math.html
- M3GNet paper (Chen & Ong, 2022): https://arxiv.org/abs/2202.02450
- MatGL project docs: https://matgl.ai/
- PyTorch `torch.Tensor.index_add_` semantics: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html
- PyTorch `torch.isfinite` semantics: https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html
- PyTorch `torch.reshape` semantics: https://docs.pytorch.org/docs/stable/generated/torch.reshape.html

## Progress snapshot (after Batch 100)
- Completed: Batch 1 through Batch 100.
- Pending: Batch 101 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 101.
## Batch 101 (max 5 files)
- [x] `atlas/training/checkpoint.py` - reviewed + optimized
- [x] `atlas/training/filters.py` - reviewed + optimized
- [x] `atlas/training/metrics.py` - reviewed (no code change required)
- [x] `atlas/training/normalizers.py` - reviewed + optimized
- [x] `atlas/training/preflight.py` - reviewed + optimized

## Batch 101 optimization goals
- Harden bool-like input contracts across training persistence/preflight/filter pipelines to avoid implicit `True/False -> 1/0` coercion drift.
- Improve normalizer property-name access consistency (load-time normalization and runtime access normalization alignment).
- Keep metric core behavior unchanged while validating current finite-pair and safe-fallback semantics remain stable.

## Batch 101 outcomes
- `atlas/training/checkpoint.py`:
  - added `_is_boolean_like(...)` and applied it to `_coerce_int(...)` / `_coerce_finite_float(...)` to reject NumPy-style boolean scalars in the same way as Python bool.
  - `save_best(...)` now refreshes `best.pt` pointer from current sorted best-model ranking unconditionally after update/prune, reducing pointer drift risk in repeated updates.
- `atlas/training/filters.py`:
  - added `_is_boolean_like(...)` guard into `_extract_scalar(...)` for tensor/ndarray/scalar `.item()` paths.
  - boolean-like values are now ignored as non-numeric targets instead of being silently coerced to `0.0/1.0` and polluting outlier statistics.
- `atlas/training/normalizers.py`:
  - introduced `_normalize_property_name(...)` (strip + non-empty validation).
  - `MultiTargetNormalizer._get_normalizer(...)` now uses normalized property names for runtime access (`normalize/denormalize`), aligning with load-time normalization behavior.
- `atlas/training/preflight.py`:
  - added `_is_boolean_like(...)` and applied to `_coerce_non_negative_int(...)` so NumPy bool-like inputs are rejected consistently for integer controls (`max_samples/split_seed/timeout` conversion paths).
- `atlas/training/metrics.py`:
  - reviewed in this batch; scalar/classification/tensor metric finite-safety and fallback contracts were already coherent, so no direct code edits.
- tests:
  - `tests/unit/training/test_checkpoint.py` adds regression for NumPy bool-like epoch rejection.
  - `tests/unit/training/test_filters.py` adds regression that boolean-like payloads are ignored by outlier extraction.
  - `tests/unit/training/test_normalizers.py` adds runtime whitespace-property access normalization tests and empty-name rejection test.
  - `tests/unit/training/test_preflight.py` adds NumPy bool-like rejection test for `split_seed`.

## Verification
- `python -m ruff check atlas/training/checkpoint.py atlas/training/filters.py atlas/training/metrics.py atlas/training/normalizers.py atlas/training/preflight.py tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py tests/unit/training/test_preflight.py`
- `python -m pytest -q tests/unit/training/test_checkpoint.py tests/unit/training/test_filters.py tests/unit/training/test_metrics.py tests/unit/training/test_normalizers.py tests/unit/training/test_preflight.py`
- `python -m pytest -q tests/unit/training`

## Research references used in batch 101
- Python `subprocess.run` and timeout/error semantics: https://docs.python.org/3/library/subprocess.html
- PyTorch `Tensor.item()` behavior for scalar extraction: https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.item.html
- PyTorch floating-point tensor detection (`torch.is_floating_point`): https://docs.pytorch.org/docs/stable/generated/torch.is_floating_point.html
- NumPy scalar boolean type (`numpy.bool_`) semantics: https://numpy.org/doc/2.0/reference/arrays.scalars.html
- Modified z-score outlier guidance (NIST discussion, 0.6745 scaling / 3.5 threshold context): https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=957454

## Progress snapshot (after Batch 101)
- Completed: Batch 1 through Batch 101.
- Pending: Batch 102 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 102.
## Batch 102 (max 5 files)
- [x] `atlas/training/losses.py` - reviewed + optimized
- [x] `atlas/training/physics_losses.py` - reviewed + optimized
- [x] `atlas/training/run_utils.py` - reviewed + optimized
- [x] `atlas/training/theory_tuning.py` - reviewed + optimized
- [x] `atlas/training/trainer.py` - reviewed + optimized

## Batch 102 optimization goals
- Eliminate bool-like coercion edge cases (`numpy.bool`/`numpy.bool_`) across core training logic to prevent silent numeric drift.
- Tighten manifest-loading exception boundaries for better diagnosability while preserving recovery behavior.
- Reduce broad exception handling in trainer alignment path to avoid masking unrelated runtime faults.

## Batch 102 outcomes
- `atlas/training/losses.py`:
  - added `_is_boolean_like(...)` helper that handles both Python `bool` and NumPy bool-like scalars.
  - `_require_finite_scalar(...)` now rejects NumPy bool-like values consistently.
  - `_normalize_named_mapping(...)` now rejects duplicate task keys after normalization (e.g. `"task"` and `" task "`) instead of silently overriding.
- `atlas/training/physics_losses.py`:
  - added `_is_boolean_like(...)` and applied it in `_coerce_non_negative_float(...)` to reject NumPy bool-like values for numeric weights.
- `atlas/training/run_utils.py`:
  - added `_is_boolean_like(...)`; `_coerce_bool_like(...)` now explicitly normalizes bool-like scalars via `bool(...)`.
  - `_json_safe(...)` now preserves NumPy bool-like values as booleans (not accidental numeric/string coercions).
  - narrowed broad exception catches when reading split manifests and merging existing manifest payloads to explicit parse/IO error classes.
- `atlas/training/theory_tuning.py`:
  - added `_is_boolean_like(...)` and applied it to float coercion and metric extraction gates.
  - bool-like values are now consistently treated as non-numeric in objective thresholds and score extraction.
- `atlas/training/trainer.py`:
  - narrowed `_align_prediction_target(...)` fallback catch from broad `Exception` to explicit import/runtime/type/value errors around pooling alignment.

## Tests updated
- `tests/unit/training/test_losses.py`:
  - added regression for NumPy bool-like coeff rejection in `EvidentialLoss`.
  - added regression for duplicate normalized task keys in `MultiTaskLoss` mappings.
- `tests/unit/training/test_physics_losses.py`:
  - added regression for NumPy bool-like weight rejection in `VoigtReussBoundsLoss`.
- `tests/unit/training/test_run_utils_manifest.py`:
  - added regression for `strict_lock=np.bool_(True)` parsing behavior.
- `tests/unit/research/test_theory_tuning.py`:
  - added regressions for NumPy bool-like threshold rejection and manifest-score bool filtering.
- `tests/unit/training/test_trainer.py`:
  - added regression ensuring alignment path gracefully recovers when pooling raises a runtime error.

## Verification
- `python -m ruff check atlas/training/losses.py atlas/training/physics_losses.py atlas/training/run_utils.py atlas/training/theory_tuning.py atlas/training/trainer.py tests/unit/training/test_losses.py tests/unit/training/test_physics_losses.py tests/unit/training/test_run_utils_manifest.py tests/unit/training/test_trainer.py tests/unit/research/test_theory_tuning.py`
- `python -m pytest -q tests/unit/training/test_losses.py tests/unit/training/test_physics_losses.py tests/unit/training/test_run_utils_manifest.py tests/unit/training/test_trainer.py tests/unit/research/test_theory_tuning.py`
- `python -m pytest -q tests/unit/training tests/unit/research/test_theory_tuning.py`

## Research references used in batch 102
- NumPy scalar type semantics (`numpy.bool` / scalar hierarchy): https://numpy.org/doc/2.0/reference/arrays.scalars.html
- Python JSON parsing errors (`json.JSONDecodeError`): https://docs.python.org/3/library/json.html#json.JSONDecodeError
- PyTorch Geometric `global_mean_pool` contract: https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.pool.global_mean_pool.html
- Multi-task uncertainty weighting reference (Kendall et al., 2018): https://arxiv.org/abs/1705.07115
- Python exception hierarchy rationale (`Exception` vs explicit exceptions): https://docs.python.org/3/library/exceptions.html

## Progress snapshot (after Batch 102)
- Completed: Batch 1 through Batch 102.
- Pending: Batch 103 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 103.
## Batch 103 (max 5 files)
- [x] `atlas/research/method_registry.py` - reviewed + optimized
- [x] `atlas/research/workflow_reproducible_graph.py` - reviewed + optimized
- [x] `atlas/research/__init__.py` - reviewed + optimized
- [x] `atlas/utils/reproducibility.py` - reviewed + optimized
- [x] `atlas/utils/registry.py` - reviewed + optimized

## Batch 103 optimization goals
- Strengthen research-method metadata contracts (strict string fields + dedupe) to avoid silent coercion in experiment descriptors.
- Tighten reproducible-workflow invariants (iteration count ordering, explicit stage-plan semantics, fallback-method normalization).
- Improve lazy-export and registry behavior for diagnosability/perf (cached lazy exports, callable-only registrations, explicit override intent).

## Batch 103 outcomes
- `atlas/research/method_registry.py`:
  - `_normalize_lookup_key(...)` now rejects non-string/boolean inputs instead of silently `str(...)` coercing.
  - `MethodSpec._normalize_text_items(...)` now enforces string entries and de-duplicates normalized `strengths`/`tradeoffs`.
  - `recommended_method_order(...)` now validates `primary` via the same strict key normalizer.
- `atlas/research/workflow_reproducible_graph.py`:
  - bool-like detection widened to include both `numpy.bool` and `numpy.bool_` naming variants.
  - `_normalize_fallback_methods(...)` now de-duplicates normalized methods and rejects boolean payloads.
  - `IterationSnapshot.__post_init__(...)` now enforces monotonic count constraints: `selected <= relaxed <= unique <= generated`.
  - `start(...)` now distinguishes `stage_plan=None` (use default) from explicit `[]` (validation error), removing ambiguous fallback behavior.
- `atlas/research/__init__.py`:
  - lazy exports are now cached into module globals after first access.
  - lazy-export attr resolution now raises targeted errors when the source module lacks the expected symbol.
- `atlas/utils/reproducibility.py`:
  - introduced `_is_boolean_like(...)` and applied it to `_coerce_bool(...)` / `_coerce_seed(...)` for NumPy bool-like robustness.
  - narrowed seed parsing exception boundaries to explicit parse/overflow cases.
- `atlas/utils/registry.py`:
  - `Registry.register(...)` now validates registered objects are callable.
  - added explicit `replace` flag to declare intentional overrides (without changing existing overwrite capability).
  - `__contains__(...)` now reuses canonical key validation for consistent behavior.

## Tests updated
- `tests/unit/research/test_method_registry.py`:
  - added dedupe regression for normalized `strengths/tradeoffs`.
  - added non-string key/entry rejection regressions.
- `tests/unit/research/test_workflow_reproducible_graph.py`:
  - added count-order invariant regressions (`selected <= relaxed <= unique <= generated`).
  - added explicit empty `stage_plan` rejection regression.
  - added `RunManifest.fallback_methods` dedupe + boolean rejection regressions.
- `tests/unit/research/test_reproducibility.py`:
  - added NumPy bool-like regressions for `_coerce_seed` and `_coerce_bool`.
- `tests/unit/research/test_utils_registry.py`:
  - added non-callable registration rejection regression.
  - added `replace=True` override regression.
- `tests/unit/research/test_research_init.py` (new):
  - added lazy-export caching, `__dir__` coverage, and unknown export error-path tests.

## Verification
- `python -m ruff check atlas/research/method_registry.py atlas/research/workflow_reproducible_graph.py atlas/research/__init__.py atlas/utils/reproducibility.py atlas/utils/registry.py tests/unit/research/test_method_registry.py tests/unit/research/test_workflow_reproducible_graph.py tests/unit/research/test_reproducibility.py tests/unit/research/test_utils_registry.py tests/unit/research/test_research_init.py`
- `python -m pytest -q tests/unit/research/test_method_registry.py tests/unit/research/test_workflow_reproducible_graph.py tests/unit/research/test_reproducibility.py tests/unit/research/test_utils_registry.py tests/unit/research/test_research_init.py`
- `python -m pytest -q tests/unit/research`

## Research references used in batch 103
- PEP 562 (module `__getattr__` / `__dir__` lazy-export semantics): https://peps.python.org/pep-0562/
- Python `importlib` docs (runtime module import behavior): https://docs.python.org/3/library/importlib.html
- Python `os.replace` docs (atomic replacement semantics relevant to manifest persistence): https://docs.python.org/3/library/os.html#os.replace
- PyTorch reproducibility notes (determinism and randomness controls): https://docs.pytorch.org/docs/stable/notes/randomness.html
- NumPy scalar type docs (`numpy.bool` / `numpy.bool_` behavior): https://numpy.org/doc/2.0/reference/arrays.scalars.html

## Progress snapshot (after Batch 103)
- Completed: Batch 1 through Batch 103.
- Pending: Batch 104 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 104.
## Batch 104 (max 5 files)
- [x] `atlas/thermo/stability.py` - reviewed + optimized
- [x] `atlas/thermo/calphad.py` - reviewed + optimized
- [x] `atlas/thermo/openmm/engine.py` - reviewed + optimized
- [x] `atlas/thermo/openmm/reporters.py` - reviewed + optimized
- [x] `atlas/thermo/openmm/atomate2_wrapper.py` - reviewed + optimized

## Batch 104 optimization goals
- Harden thermo/OpenMM numeric-input contracts to eliminate implicit bool/float coercions in critical simulation paths.
- Strengthen thermodynamic workflow invariants (chemical symbols, composition fractions, and frame/report integrity).
- Improve optional-dependency wrappers and error signaling for reproducible CI/runtime behavior.

## Batch 104 outcomes
- `atlas/thermo/stability.py`:
  - added strict helpers for bool-like detection and finite-float coercion.
  - `_normalize_element_symbol(...)` now enforces string-only symbols and validates symbol token format.
  - `ReferenceDatabase.add_entry(...)` now rejects non-string formulas and bool-like energies.
  - `ReferenceDatabase.get_entries(...)` now validates all provided chemical-system symbols (no silent `str(...)` coercion).
  - `analyze_stability(...)` now validates `target_formula` type/content before `Composition(...)` and uses `logger.exception(...)` for richer failure traces.
- `atlas/thermo/calphad.py`:
  - `_normalize_alloy_key(...)` now enforces string/non-empty inputs.
  - `_coerce_int_with_min(...)` now rejects NumPy bool-like values and narrows parse exception handling.
  - `_normalize_phase_fractions(...)` now rejects bool-like fraction payloads instead of silently accepting `True/False` as numeric.
  - `_normalize_composition(...)` now rejects bool-like mole fractions explicitly.
- `atlas/thermo/openmm/engine.py`:
  - added bool-like helper for strict integer coercion in run controls.
  - `_coerce_positive_int(...)` now uses explicit parse-error boundaries.
  - `setup_system(...)` now wraps invalid element symbol resolution into a clear `ValueError` context.
  - preserved OpenMM topology periodic-box API compatibility while keeping periodic/box validation flow intact.
- `atlas/thermo/openmm/reporters.py`:
  - added strict `_coerce_positive_int(...)` (reject bool and non-integral numeric intervals).
  - added explicit `_coerce_bool(...)` parser for `enforcePeriodicBox` (`true/false/1/0/...`).
  - reporter now stores defensive copies of position/force arrays to avoid aliasing/mutation drift between frames.
- `atlas/thermo/openmm/atomate2_wrapper.py`:
  - added bool-like helper and stricter integer coercion (`_coerce_non_negative_int`) for float/integer-valued checks.
  - keeps `steps` validation consistent across `nvt/npt/minimize` maker paths.

## Tests updated
- `tests/unit/thermo/test_stability.py`:
  - added regressions for invalid chemical-system symbol tokens.
  - added regressions for bool-like composition/energy rejection in reference entries.
  - added regression for invalid `target_formula` type rejection.
- `tests/unit/thermo/test_calphad.py`:
  - added regression for bool-like phase-fraction rejection.
  - added regression for non-string alloy-name rejection in `get_composition(...)`.
- `tests/unit/thermo/test_openmm_stack.py`:
  - adjusted fake topology periodic-box API to accept OpenMM-style vector payload shape.
- `tests/unit/thermo/test_openmm_reporters.py`:
  - added regressions for rejecting bool/non-integral `reportInterval`.
  - added regressions for parsing/validating `enforcePeriodicBox` bool-like values.
- `tests/unit/thermo/test_openmm_atomate2_wrapper.py`:
  - added regression ensuring fractional `steps` are rejected as non-integer in maker construction.

## Verification
- `python -m ruff check atlas/thermo/stability.py atlas/thermo/calphad.py atlas/thermo/openmm/engine.py atlas/thermo/openmm/reporters.py atlas/thermo/openmm/atomate2_wrapper.py tests/unit/thermo/test_stability.py tests/unit/thermo/test_calphad.py tests/unit/thermo/test_openmm_stack.py tests/unit/thermo/test_openmm_reporters.py tests/unit/thermo/test_openmm_atomate2_wrapper.py`
- `python -m pytest -q tests/unit/thermo/test_stability.py tests/unit/thermo/test_calphad.py tests/unit/thermo/test_openmm_stack.py tests/unit/thermo/test_openmm_reporters.py tests/unit/thermo/test_openmm_atomate2_wrapper.py`
- `python -m pytest -q tests/unit/thermo`

## Research references used in batch 104
- Pymatgen usage docs (phase diagram section): https://pymatgen.org/usage.html
- Materials Project phase diagram methodology notes: https://docs.materialsproject.org/methodology/materials-methodology/thermodynamic-stability/phase-diagrams-pds
- pycalphad docs homepage: https://pycalphad.org/docs/latest/
- pycalphad core API docs (`equilibrium`): https://pycalphad.org/docs/latest/api/pycalphad.core.html
- pycalphad paper (Otis & Liu, 2017): https://doi.org/10.5334/jors.140
- OpenMM documentation portal: https://openmm.org/documentation
- OpenMM Topology API (`setPeriodicBoxVectors`): https://docs.openmm.org/6.3.0/api-python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html
- OpenMM NonbondedForce API (cutoff/periodic method semantics): https://docs.openmm.org/latest/api-python/generated/openmm.openmm.NonbondedForce.html
- ASE Atoms documentation (cell/PBC semantics): https://wiki.fysik.dtu.dk/ase/ase/atoms.html
- NumPy scalar type semantics (`numpy.bool`/`numpy.bool_`): https://numpy.org/doc/2.0/reference/arrays.scalars.html

## Progress snapshot (after Batch 104)
- Completed: Batch 1 through Batch 104.
- Pending: Batch 105 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 105.
## Batch 105 (max 5 files)
- [x] `atlas/thermo/__init__.py` - reviewed + optimized
- [x] `atlas/thermo/openmm/__init__.py` - reviewed + optimized
- [x] `atlas/discovery/alchemy/calculator.py` - reviewed + optimized
- [x] `atlas/discovery/alchemy/model.py` - reviewed + optimized
- [x] `atlas/discovery/alchemy/optimizer.py` - reviewed + optimized

## Batch 105 optimization goals
- Harden optional-import behavior for thermo/openmm export layers so missing dependencies are explicit and diagnosable without masking non-import runtime faults.
- Decouple alchemy calculator module import-time from heavy MACE dependencies (lazy load + actionable error message).
- Tighten alchemical input/gradient constraints (pair schema, weight bounds, step integer semantics) to reduce silent coercion drift in optimization loops.

## Batch 105 outcomes
- `atlas/thermo/__init__.py`:
  - optional import failures are now cached (`_OPTIONAL_UNAVAILABLE`) and returned quickly on repeated access.
  - import-failure capture narrowed to `ModuleNotFoundError` so non-import runtime errors are not silently swallowed.
  - added `get_optional_import_errors()` diagnostics API for tooling/reporting.
- `atlas/thermo/openmm/__init__.py`:
  - mirrored the same optional-import hardening and diagnostics API (`get_optional_import_errors()`).
  - preserves lazy-export cache and explicit attribute error behavior.
- `atlas/discovery/alchemy/calculator.py`:
  - removed import-time hard dependency on `.model`; added `_load_alchemy_api()` lazy resolver with actionable optional-dependency message.
  - added strict alchemical-pair schema validation (`_validate_alchemical_pairs`) including bool-like rejection and atom-index range checks.
  - `_as_weight_array(...)` now enforces bool-like rejection and explicit [0,1] bound checks (tolerant clipping only near boundaries).
  - model loading now uses normalized runtime device (`self.device`) rather than raw requested string.
  - added output-contract checks in `calculate(...)` for required keys and finite/shape-valid `energy/forces/stress`.
- `atlas/discovery/alchemy/model.py`:
  - `_pair_to_indices(...)` now validates pair shape/type and rejects bool-like index/atomic-number payloads.
  - `AlchemyManager.__init__(...)` now validates non-empty atoms/pairs, positive finite `r_max`, and 1D alchemical weight tensors.
- `atlas/discovery/alchemy/optimizer.py`:
  - added strict bool-like guards and integer coercion helper for `steps` (`run`) and learning-rate validation.
  - `_project_weights(...)` now validates constraint index bounds before projection.

## Tests updated
- `tests/unit/thermo/test_init_exports.py`:
  - updated optional-failure path to `ModuleNotFoundError` and added assertion on diagnostic cache output.
- `tests/unit/thermo/test_openmm_stack.py`:
  - added regression for openmm optional-import failure caching + diagnostics API.
- `tests/unit/discovery/test_alchemy_calculator.py` (new):
  - added regressions for pair/weight validation and lazy alchemy API loading path.
  - added regression for normalized device propagation into model loader.
  - added regression for missing required model output keys.
- `tests/unit/discovery/test_alchemy_optimizer.py` (new):
  - added regressions for bool-like learning-rate rejection, simplex-projected update behavior, and strict non-integral step rejection.

## Verification
- `python -m ruff check atlas/thermo/__init__.py atlas/thermo/openmm/__init__.py atlas/discovery/alchemy/calculator.py atlas/discovery/alchemy/model.py atlas/discovery/alchemy/optimizer.py tests/unit/thermo/test_init_exports.py tests/unit/thermo/test_openmm_stack.py tests/unit/discovery/test_alchemy_calculator.py tests/unit/discovery/test_alchemy_optimizer.py`
- `python -m pytest -q tests/unit/thermo/test_init_exports.py tests/unit/thermo/test_openmm_stack.py tests/unit/discovery/test_alchemy_calculator.py tests/unit/discovery/test_alchemy_optimizer.py`
- `python -m pytest -q tests/unit/thermo tests/unit/discovery`

## Research references used in batch 105
- PEP 562 (`module __getattr__`, `__dir__`, lazy export semantics): https://peps.python.org/pep-0562/
- Python exception docs (`ModuleNotFoundError` as `ImportError` subclass): https://docs.python.org/3/library/exceptions.html#ModuleNotFoundError
- ASE calculator interface documentation: https://ase-lib.org/ase/calculators/calculators.html
- PyTorch `autograd.grad` API contract: https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html
- NumPy `nan_to_num` behavior: https://numpy.org/doc/2.4/reference/generated/numpy.nan_to_num.html
- MACE paper (Batatia et al., 2022): https://arxiv.org/abs/2206.07697
- MACE project repository/docs: https://github.com/ACEsuit/mace
- Simplex projection reference (Duchi et al., ICML 2008): https://icml.cc/Conferences/2008/papers/361.pdf

## Progress snapshot (after Batch 105)
- Completed: Batch 1 through Batch 105.
- Pending: Batch 106 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 106.
## Batch 106 (max 5 files)
- [x] `atlas/discovery/alchemy/__init__.py` - reviewed + optimized
- [x] `atlas/discovery/stability/__init__.py` - reviewed + optimized
- [x] `atlas/discovery/stability/mepin.py` - reviewed + optimized
- [x] `atlas/discovery/transport/liflow.py` - reviewed + optimized
- [x] `atlas/discovery/alchemy/model.py` - reviewed + optimized

## Batch 106 optimization goals
- Harden discovery-layer lazy exports so optional-dependency missing paths are diagnosable and cached, while non-import runtime failures remain visible.
- Reduce import-time fragility for MEPIN/LiFlow wrappers by moving optional heavy dependencies to lazy loading paths.
- Tighten numerical and schema contracts for path/transport simulation controls (strict positive integer checks, bool-like rejection, finite coordinate guards).

## Batch 106 outcomes
- `atlas/discovery/alchemy/__init__.py`:
  - added cached optional import state (`_OPTIONAL_UNAVAILABLE`) and error map (`_OPTIONAL_IMPORT_ERRORS`).
  - narrowed optional dependency handling to `ModuleNotFoundError` so runtime errors are no longer swallowed by broad exception fallback.
  - added `get_optional_import_errors()` for diagnostics tooling.
- `atlas/discovery/stability/__init__.py`:
  - mirrored optional import caching + diagnostics behavior.
  - now returns `None` only for explicit missing-module scenarios; other failures are surfaced.
- `atlas/discovery/stability/mepin.py`:
  - replaced module-import-time optional dependency loading with `_load_mepin_api()` lazy resolver.
  - added strict model-type normalization (`cyclo_l`/`t1x_l` -> canonical checkpoint names).
  - added strict integer coercion helper for `num_images` and bool-like rejection.
  - strengthened `predict_path(...)` input guards (ase.Atoms type + finite coordinate validation).
- `atlas/discovery/transport/liflow.py`:
  - replaced module-import-time optional dependency loading with `_load_liflow_api()` lazy resolver.
  - tightened temperature list normalization (reject bool/non-integral inputs, preserve positive integer semantics).
  - tightened `steps`/`flow_steps` coercion via strict positive-integer helper.
  - element-index loading now validates non-empty arrays and normalizes to integer vectors.
- `atlas/discovery/alchemy/model.py`:
  - fixed cutoff assignment to validated float (`self.r_max = cutoff`) to avoid latent string/object propagation.
  - added per-group non-empty validation for `alchemical_pairs` in `AlchemyManager`.

## Tests updated
- `tests/unit/discovery/test_alchemy_init.py` (new):
  - lazy export surface checks, optional-import placeholder behavior, and non-import runtime error propagation.
- `tests/unit/discovery/test_stability_init.py` (new):
  - stability lazy export checks + optional import diagnostics cache behavior.
- `tests/unit/discovery/test_mepin.py` (new):
  - model-type normalization, integer coercion guards, and `predict_path` input validation under monkeypatched backend.
- `tests/unit/discovery/test_liflow.py` (new):
  - temperature/step coercion guard regressions and element-index fallback/validation behavior.

## Verification
- `python -m ruff check atlas/discovery/alchemy/__init__.py atlas/discovery/stability/__init__.py atlas/discovery/stability/mepin.py atlas/discovery/transport/liflow.py atlas/discovery/alchemy/model.py tests/unit/discovery/test_alchemy_init.py tests/unit/discovery/test_stability_init.py tests/unit/discovery/test_mepin.py tests/unit/discovery/test_liflow.py`
- `python -m pytest -q tests/unit/discovery/test_alchemy_init.py tests/unit/discovery/test_stability_init.py tests/unit/discovery/test_mepin.py tests/unit/discovery/test_liflow.py tests/unit/discovery/test_alchemy_calculator.py tests/unit/discovery/test_alchemy_optimizer.py`
- `python -m pytest -q tests/unit/discovery`
- `python -m pytest -q tests/unit/thermo/test_init_exports.py tests/unit/thermo/test_openmm_stack.py`

## Research references used in batch 106
- PEP 562 (`module __getattr__` / `__dir__`): https://peps.python.org/pep-0562/
- Python exception hierarchy (`ModuleNotFoundError` semantics): https://docs.python.org/3/library/exceptions.html#ModuleNotFoundError
- Python `importlib` runtime import behavior: https://docs.python.org/3/library/importlib.html
- ASE `Atoms` API contracts: https://wiki.fysik.dtu.dk/ase/ase/atoms.html
- NumPy finite checks (`np.isfinite`): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy loading arrays (`np.load`) notes: https://numpy.org/doc/stable/reference/generated/numpy.load.html
- PyTorch no-grad and inference context: https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html
- PyTorch tensor shape/view semantics: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.reshape.html

## Progress snapshot (after Batch 106)
- Completed: Batch 1 through Batch 106.
- Pending: Batch 107 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 107.
## Batch 107 (max 5 files)
- [x] `atlas/ops/cpp_ops.py` - reviewed + optimized
- [x] `atlas/potentials/__init__.py` - reviewed + optimized
- [x] `atlas/potentials/mace_relaxer.py` - reviewed + optimized
- [x] `atlas/potentials/relaxers/mlip_arena_relaxer.py` - reviewed + optimized
- [x] `atlas/discovery/transport/liflow.py` - reviewed + optimized

## Batch 107 optimization goals
- Tighten high-risk numeric/input contracts in CPU/C++ graph ops and relaxation runtime controls to avoid silent bool/float coercion drift.
- Standardize optional-import behavior for potential exports with explicit diagnostics and cache semantics.
- Strengthen LiFlow runtime preflight checks to fail early on invalid structure/index mapping inputs.

## Batch 107 outcomes
- `atlas/ops/cpp_ops.py`:
  - added `_is_boolean_like(...)` + `_coerce_positive_int(...)` for strict integer controls.
  - `_validate_inputs(...)` now rejects non-finite `pos`, bool-like `r_max`, and non-integral/invalid `max_num_neighbors`.
  - fallback neighbor truncation now uses validated integer `max_neighbors` consistently.
- `atlas/potentials/__init__.py`:
  - added lazy-export optional import cache (`_OPTIONAL_UNAVAILABLE`) and error map (`_OPTIONAL_IMPORT_ERRORS`).
  - narrowed optional dependency handling to `ModuleNotFoundError`; non-import runtime faults remain visible.
  - added `get_optional_import_errors()` diagnostics API.
- `atlas/potentials/mace_relaxer.py`:
  - added strict bool-like/integer coercion helper for `steps` and `n_jobs`.
  - `relax_structure(...)` now enforces integer-valued `steps` (no implicit float truncation).
  - `batch_relax(...)` now enforces integer-valued `n_jobs` and clearer validation path.
  - `_normalize_device(...)` now normalizes string input via `str(...).strip()` and rejects bool-like device payloads.
- `atlas/potentials/relaxers/mlip_arena_relaxer.py`:
  - added strict integer coercion for `steps` and bool-like parser for `symmetry`.
  - `symmetry` now accepts explicit boolean-like strings (`true/false/1/0/...`) and rejects ambiguous payloads.
- `atlas/discovery/transport/liflow.py`:
  - `simulate(...)` now validates atomic positions shape/finite values before simulator invocation.
  - added atomic-number bounds check against `element_idx` size for early, actionable failure.
  - reuses validated atomic-number array consistently in downstream simulation logic.

## Tests updated
- `tests/unit/ops/test_cpp_ops.py` (new):
  - validates finite/bool/integer guards for `_validate_inputs(...)`.
  - validates neighbor-cap behavior in `_torch_radius_graph_fallback(...)`.
- `tests/unit/potentials/test_init_exports.py` (new):
  - validates lazy export surface, optional-import fallback + diagnostics cache, and missing-attr error path.
- `tests/unit/potentials/test_mace_relaxer.py` (new):
  - validates integer/device normalizers and strict `n_jobs` handling.
- `tests/unit/potentials/test_mlip_arena_relaxer.py` (new):
  - validates strict `steps` and `symmetry` boolean-like parsing behavior.
- `tests/unit/discovery/test_liflow.py`:
  - added regression for atomic-number/index-map mismatch rejection in `simulate(...)`.

## Verification
- `python -m ruff check atlas/ops/cpp_ops.py atlas/potentials/__init__.py atlas/potentials/mace_relaxer.py atlas/potentials/relaxers/mlip_arena_relaxer.py atlas/discovery/transport/liflow.py tests/unit/ops/test_cpp_ops.py tests/unit/potentials/test_init_exports.py tests/unit/potentials/test_mace_relaxer.py tests/unit/potentials/test_mlip_arena_relaxer.py tests/unit/discovery/test_liflow.py`
- `python -m pytest -q tests/unit/ops/test_cpp_ops.py tests/unit/potentials/test_init_exports.py tests/unit/potentials/test_mace_relaxer.py tests/unit/potentials/test_mlip_arena_relaxer.py tests/unit/discovery/test_liflow.py`
- `python -m pytest -q tests/unit/discovery tests/unit/potentials tests/unit/ops`

## Research references used in batch 107
- PyTorch C++ extension docs (`load_inline`): https://docs.pytorch.org/docs/stable/cpp_extension.html
- PyTorch `torch.cdist` reference: https://docs.pytorch.org/docs/stable/generated/torch.cdist.html
- ASE structure optimization docs (`BFGSLineSearch`): https://ase.gitlab.io/ase/ase/optimize.html
- MACE architecture paper (Batatia et al., 2022): https://arxiv.org/abs/2206.07697
- MACE project repository/docs: https://github.com/ACEsuit/mace
- NumPy finite check reference (`np.isfinite`): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy load reference (`np.load`): https://numpy.org/doc/stable/reference/generated/numpy.load.html

## Progress snapshot (after Batch 107)
- Completed: Batch 1 through Batch 107.
- Pending: Batch 108 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 108.
## Batch 108 (max 5 files)
- [x] `atlas/__init__.py` - reviewed + optimized
- [x] `atlas/console_style.py` - reviewed + optimized
- [x] `atlas/explain/__init__.py` - reviewed + optimized
- [x] `atlas/explain/gnn_explainer.py` - reviewed + optimized
- [x] `atlas/explain/latent_analysis.py` - reviewed + optimized

## Batch 108 optimization goals
- Harden lazy export boundaries in top-level and explain modules so optional dependency failures remain diagnosable and non-import faults are not silently masked.
- Strengthen explainability pipeline numeric/shape contracts (strict integer semantics, consistent graph-size checks, safer plotting inputs).
- Improve latent-space analytics robustness for small-sample regimes and malformed batch/property payloads.
- Keep console styling predictable in CI logs by avoiding double ANSI styling and narrowing exception boundaries.

## Batch 108 outcomes
- `atlas/__init__.py`:
  - `__getattr__(...)` now checks cached globals first.
  - missing expected symbol in mapped module now raises explicit `AttributeError` context.
- `atlas/console_style.py`:
  - `_supports_color(...)` narrowed `isatty()` exception handling to explicit runtime parse errors.
  - `_style_line(...)` now avoids re-styling lines that already contain ANSI escape codes.
- `atlas/explain/__init__.py`:
  - added optional-import diagnostics cache (`_OPTIONAL_IMPORT_ERRORS`) and unavailable cache (`_OPTIONAL_UNAVAILABLE`).
  - optional dependency path is now handled via `ModuleNotFoundError` only; non-import runtime errors remain visible.
  - added `get_optional_import_errors()` for reproducibility/debug tooling.
- `atlas/explain/gnn_explainer.py`:
  - added strict positive-integer coercion for `n_epochs`/`n_samples` (rejects bool/fractional/non-finite values).
  - added safer graph device inference and stronger input contract checks (`num_nodes`, `x`, `edge_index`).
  - added node/edge importance length invariants and stricter plotting `view_angle` validation.
- `atlas/explain/latent_analysis.py`:
  - added strict helpers for integer controls and finite array shape normalization.
  - `extract_embeddings(...)` now handles missing batch index, normalizes embedding shape to 2D, and aligns property lengths (with NaN fill + warning on mismatch).
  - `reduce_dimensions(...)` now validates inputs and handles tiny-sample UMAP safely (PCA fallback for <=2 samples; bounded `n_neighbors` otherwise).
  - `perform_clustering(...)`, `plot_latent_space(...)`, `analyze_clusters(...)` now enforce finite/shape/length contracts and reject silent coercion paths.

## Verification
- `python -m ruff check atlas/__init__.py atlas/console_style.py atlas/explain/__init__.py atlas/explain/gnn_explainer.py atlas/explain/latent_analysis.py`
- `python -m pytest -q tests/unit/test_console_style.py`
- latent-analysis smoke check:
  - inline Python check covering `reduce_dimensions(...)`, `perform_clustering(...)`, and `analyze_clusters(...)` on synthetic inputs.

## Research references used in batch 108
- PEP 562 (`module __getattr__` / `__dir__` lazy export semantics): https://peps.python.org/pep-0562/
- Python exception hierarchy (`ModuleNotFoundError`): https://docs.python.org/3/library/exceptions.html#ModuleNotFoundError
- PyTorch Geometric explainability docs (`torch_geometric.explain`): https://pytorch-geometric.readthedocs.io/en/latest/modules/explain.html
- GNNExplainer (NeurIPS 2019): https://papers.nips.cc/paper_files/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html
- UMAP paper (McInnes et al., 2018): https://joss.theoj.org/papers/10.21105/joss.00861
- scikit-learn TSNE API: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
- scikit-learn PCA API: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- scikit-learn KMeans API: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- scikit-learn DBSCAN API: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
- NumPy finite sanitization (`nan_to_num`): https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html

## Progress snapshot (after Batch 108)
- Completed: Batch 1 through Batch 108.
- Pending: Batch 109 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 109.
## Batch 109 (max 5 files)
- [x] `atlas/research/__init__.py` - reviewed + optimized
- [x] `atlas/research/method_registry.py` - reviewed + optimized
- [x] `atlas/research/workflow_reproducible_graph.py` - reviewed + optimized
- [x] `atlas/utils/__init__.py` - reviewed + optimized
- [x] `atlas/utils/reproducibility.py` - reviewed + optimized

## Batch 109 optimization goals
- Strengthen lazy-export reliability in research/utils package surfaces so import failures are diagnosable and cached consistently.
- Tighten method/workflow schema contracts to reduce silent coercion risk in reproducibility manifests.
- Reduce import-time coupling in `atlas.utils` by switching to lazy exports.
- Harden deterministic/reproducibility boolean parsing to avoid ambiguous numeric truthiness (e.g., `2 -> True`).

## Batch 109 outcomes
- `atlas/research/__init__.py`:
  - added cached lazy import diagnostics via `_IMPORT_ERRORS` and `get_import_errors()`.
  - `__getattr__(...)` now checks cached globals first and raises explicit `ImportError` context on missing modules.
- `atlas/research/method_registry.py`:
  - lookup keys can be normalized to lowercase when needed; method keys are now normalized to lowercase at `MethodSpec` construction.
  - `MethodRegistry.register(...)` now enforces strict boolean type for `replace`.
  - `get(...)` and `recommended_method_order(...)` now support case-insensitive key lookup behavior through normalized keys.
- `atlas/research/workflow_reproducible_graph.py`:
  - added strict helpers for non-empty strings and bool-like coercion.
  - tightened `RunManifest.__post_init__(...)` validation for `run_id/method_key/data_source_key/model_name/relaxer_name/evaluator_name/schema_version/status`.
  - deterministic flag now validates as bool-like (`true/false/0/1/...`) instead of implicit truthiness.
  - stage plan entries now require explicit non-empty strings; fallback methods now reject non-string/bool payloads.
  - `set_metric(...)`/`finalize(...)` now enforce non-empty string keys/status.
- `atlas/utils/__init__.py`:
  - replaced eager imports with lazy export map + `__getattr__`/`__dir__`, reducing import-time overhead and optional-dependency friction.
- `atlas/utils/reproducibility.py`:
  - `_coerce_bool(...)` now accepts only binary numeric values (`0/1`) for numeric payloads; other numeric values fall back to explicit default.
  - removes ambiguous integer/float truthiness from reproducibility controls.

## Tests updated
- `tests/unit/research/test_research_init.py`:
  - added regression for lazy-import failure caching via `get_import_errors()`.
- `tests/unit/research/test_method_registry.py`:
  - added lowercase key normalization tests.
  - added case-insensitive lookup regression.
  - added strict `replace` boolean-type validation test.
- `tests/unit/research/test_workflow_reproducible_graph.py`:
  - added strict stage-entry type validation test.
  - added non-string `model_name` rejection test.
  - added deterministic bool-like string coercion regression.
- `tests/unit/research/test_reproducibility.py`:
  - added binary-only numeric bool coercion regressions.
- `tests/unit/research/test_utils_init.py` (new):
  - added lazy export cache, `dir(...)`, and unknown attribute behavior tests for `atlas.utils`.

## Verification
- `python -m ruff check atlas/research/__init__.py atlas/research/method_registry.py atlas/research/workflow_reproducible_graph.py atlas/utils/__init__.py atlas/utils/reproducibility.py tests/unit/research/test_research_init.py tests/unit/research/test_method_registry.py tests/unit/research/test_workflow_reproducible_graph.py tests/unit/research/test_reproducibility.py tests/unit/research/test_utils_init.py`
- `python -m pytest -q tests/unit/research/test_research_init.py tests/unit/research/test_method_registry.py tests/unit/research/test_workflow_reproducible_graph.py tests/unit/research/test_reproducibility.py tests/unit/research/test_utils_init.py`
- `python -m pytest -q tests/unit/research`

## Research references used in batch 109
- PEP 562 (module-level `__getattr__` / `__dir__`): https://peps.python.org/pep-0562/
- Python `importlib` docs: https://docs.python.org/3/library/importlib.html
- Python `dataclasses` docs (`__post_init__`): https://docs.python.org/3/library/dataclasses.html
- Python `json` docs (`json.dump`, `allow_nan`, `ensure_ascii`): https://docs.python.org/3/library/json.html
- Python `tempfile` docs (`NamedTemporaryFile`): https://docs.python.org/3/library/tempfile.html
- Python `pathlib` docs (`Path.replace`): https://docs.python.org/3/library/pathlib.html
- PyTorch reproducibility notes: https://docs.pytorch.org/docs/stable/notes/randomness.html
- NumPy RNG seeding docs (`numpy.random.seed` legacy guidance): https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
- ACM Artifact Review and Badging (current): https://www.acm.org/publications/policies/artifact-review-and-badging-current

## Progress snapshot (after Batch 109)
- Completed: Batch 1 through Batch 109.
- Pending: Batch 110 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 110.
## Batch 110 (max 5 files)
- [x] `atlas/utils/registry.py` - reviewed + optimized
- [x] `atlas/utils/structure.py` - reviewed + optimized
- [x] `atlas/models/prediction_utils.py` - reviewed + optimized
- [x] `atlas/models/utils.py` - reviewed + optimized
- [x] `atlas/data/__init__.py` - reviewed + optimized

## Batch 110 optimization goals
- Eliminate silent overwrite and weak type contracts in registry infrastructure.
- Tighten bool-like and shape/dtype safety in structure + prediction helper utilities.
- Improve checkpoint/model-loader diagnostics to fail fast with actionable errors.
- Align data-package lazy exports with robust import diagnostics and explicit error paths.

## Batch 110 outcomes
- `atlas/utils/registry.py`:
  - `register(..., replace=...)` now enforces boolean type.
  - duplicate registration now fails fast unless `replace=True` (no silent overwrite).
- `atlas/utils/structure.py`:
  - added strict bool-like coercion helper for `primitive` flag in `get_standardized_structure(...)`.
  - rejects ambiguous payloads (e.g., `"maybe"`) while still accepting explicit bool-like tokens.
- `atlas/models/prediction_utils.py`:
  - mean/std sanitization now enforces floating-point tensors before numeric cleanup.
  - `forward_graph_model(...)` now validates required batch attributes (`x`, `edge_index`), validates `tasks` and `encoder_kwargs` contracts, and safely infers `batch` indices when absent.
- `atlas/models/utils.py`:
  - added `_coerce_checkpoint_path(...)` to validate path-like input, file existence, and non-directory contract before load.
  - `_try_load_candidates(...)` now captures expected loader failures and raises a consolidated diagnostic instead of broad silent fallthrough.
- `atlas/data/__init__.py`:
  - added lazy import diagnostics cache (`_IMPORT_ERRORS`) + `get_import_errors()`.
  - `__getattr__(...)` now handles missing module dependencies with explicit `ImportError` context and checks expected export symbols.

## Tests updated
- `tests/unit/research/test_utils_registry.py`:
  - added duplicate registration rejection regression.
  - added `replace` type enforcement regression.
- `tests/unit/models/test_structure_utils.py`:
  - added bool-like primitive acceptance regression.
  - added invalid bool-like primitive rejection regression.
- `tests/unit/models/test_prediction_utils.py`:
  - added non-floating mean/std coercion regression.
  - added `forward_graph_model(...)` contract validation tests (`tasks`, `encoder_kwargs`, missing batch index path).
- `tests/unit/models/test_model_utils.py`:
  - added boolean checkpoint-path rejection regression.
  - added missing checkpoint file rejection regression.
- `tests/unit/data/test_init_exports.py` (new):
  - lazy export cache behavior, unknown symbol behavior, export mismatch behavior, and import-error cache behavior.

## Verification
- `python -m ruff check atlas/utils/registry.py atlas/utils/structure.py atlas/models/prediction_utils.py atlas/models/utils.py atlas/data/__init__.py tests/unit/research/test_utils_registry.py tests/unit/models/test_structure_utils.py tests/unit/models/test_prediction_utils.py tests/unit/models/test_model_utils.py tests/unit/data/test_init_exports.py`
- `python -m pytest -q tests/unit/research/test_utils_registry.py tests/unit/models/test_structure_utils.py tests/unit/models/test_prediction_utils.py tests/unit/models/test_model_utils.py tests/unit/data/test_init_exports.py`

## Research references used in batch 110
- PEP 562 (module-level lazy attributes): https://peps.python.org/pep-0562/
- Python `importlib` documentation: https://docs.python.org/3/library/importlib.html
- PyTorch `torch.broadcast_tensors` API: https://docs.pytorch.org/docs/stable/generated/torch.broadcast_tensors.html
- PyTorch broadcasting semantics note: https://docs.pytorch.org/docs/stable/notes/broadcasting.html
- PyTorch reproducibility notes (updated 2025-10-03): https://docs.pytorch.org/docs/stable/notes/randomness.html
- NumPy `random.seed` reference: https://numpy.org/devdocs/reference/random/generated/numpy.random.seed.html
- NumPy legacy RNG guidance: https://numpy.org/doc/1.25/reference/random/legacy.html
- Python `tempfile` docs (`NamedTemporaryFile`): https://docs.python.org/3/library/tempfile.html
- Python `pathlib` docs (`Path.replace`): https://docs.python.org/3/library/pathlib.html
- pymatgen API index (SpacegroupAnalyzer methods): https://pymatgen.org/pymatgen.html

## Progress snapshot (after Batch 110)
- Completed: Batch 1 through Batch 110.
- Pending: Batch 111 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 111.
## Batch 111 (max 5 files)
- [x] `atlas/models/__init__.py` - reviewed + optimized
- [x] `atlas/training/__init__.py` - reviewed + optimized
- [x] `atlas/benchmark/__init__.py` - reviewed + optimized
- [x] `atlas/data/source_registry.py` - reviewed + optimized
- [x] `atlas/data/split_governance.py` - reviewed + optimized

## Batch 111 optimization goals
- Standardize lazy-export error handling across model/training/benchmark package surfaces.
- Strengthen data-source registry schema and update contracts to avoid silent coercion/overwrite paths.
- Harden split governance normalization around boolean-like payloads and identifier validation for deterministic manifests.
- Preserve backward-compatible behavior while improving diagnostics and reproducibility guarantees.

## Batch 111 outcomes
- `atlas/models/__init__.py`:
  - added lazy import diagnostics cache (`_IMPORT_ERRORS`) and `get_import_errors()`.
  - now caches global exports early and raises explicit `ImportError` context for missing optional dependencies.
- `atlas/training/__init__.py`:
  - same lazy import hardening pattern (`_IMPORT_ERRORS`, explicit import-failure diagnostics, cache-first lookup).
- `atlas/benchmark/__init__.py`:
  - added lazy import diagnostics cache and explicit `ImportError` context for dependency failures.
  - preserved existing expected-attribute guard behavior.
- `atlas/data/source_registry.py`:
  - added strict text validation helpers for `DataSourceSpec` identity fields.
  - `DataSourceSpec.primary_targets` now handles string input as one target and rejects non-string/boolean entries.
  - `DataSourceRegistry.register(...)` now enforces `spec` type, `replace` boolean type, and duplicate key protection unless `replace=True`.
  - `update_reliability(...)` now uses strict finite/non-negative validation (rejects bool-like payloads).
  - `reset_reliability(...)` now normalizes provided keys before lookup.
- `atlas/data/split_governance.py`:
  - `_normalize_and_validate_sample_ids(...)` now rejects bool-like sample IDs.
  - `_coerce_non_negative_int(...)` now rejects NumPy bool-like payloads in addition to Python bool.
  - `_normalize_formula_value(...)` / `_normalize_spacegroup_value(...)` now map bool-like/None values to `"unknown"` instead of silently stringifying.

## Tests updated
- `tests/unit/models/test_model_utils.py`:
  - added regression for `atlas.models` lazy-import error cache behavior (`get_import_errors()`).
- `tests/unit/training/test_init_exports.py`:
  - added regression for `atlas.training` lazy-import error cache behavior.
- `tests/unit/benchmark/test_benchmark_init.py`:
  - added regression for `atlas.benchmark` lazy-import error cache behavior.
- `tests/unit/data/test_source_registry.py`:
  - added strict identity-field validation regressions for `DataSourceSpec`.
  - added duplicate registration rejection + `replace=True` overwrite behavior tests.
  - added boolean-input rejection regression for `update_reliability(...)`.
- `tests/unit/data/test_split_governance.py`:
  - added bool-like sample ID rejection regression.
  - added bool-like spacegroup normalization-to-`unknown` regression.

## Verification
- `python -m ruff check atlas/models/__init__.py atlas/training/__init__.py atlas/benchmark/__init__.py atlas/data/source_registry.py atlas/data/split_governance.py tests/unit/models/test_model_utils.py tests/unit/training/test_init_exports.py tests/unit/benchmark/test_benchmark_init.py tests/unit/data/test_source_registry.py tests/unit/data/test_split_governance.py`
- `python -m pytest -q tests/unit/models/test_model_utils.py tests/unit/training/test_init_exports.py tests/unit/benchmark/test_benchmark_init.py tests/unit/data/test_source_registry.py tests/unit/data/test_split_governance.py`

## Research references used in batch 111
- PEP 562 (module-level `__getattr__`/`__dir__`): https://peps.python.org/pep-0562/
- Python `importlib` documentation: https://docs.python.org/3/library/importlib.html
- Python `dataclasses` (`frozen=True`, `__post_init__`): https://docs.python.org/3/library/dataclasses.html
- Python `contextlib.contextmanager` docs: https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager
- NumPy `isfinite` docs: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy `linalg.pinv` docs: https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
- Dawid & Skene (1979) observer-error reliability modeling: https://doi.org/10.2307/2346806
- Ledoit & Wolf (2004) covariance shrinkage: https://doi.org/10.1016/S0047-259X(03)00096-4
- Ben-David et al. (2010) domain adaptation theory: https://doi.org/10.1007/s10994-009-5152-4

## Progress snapshot (after Batch 111)
- Completed: Batch 1 through Batch 111.
- Pending: Batch 112 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 112.
## Batch 112 (max 5 files)
- [x] `atlas/data/data_validation.py` - reviewed + optimized
- [x] `atlas/data/property_estimator.py` - reviewed + optimized
- [x] `atlas/data/alloy_estimator.py` - reviewed + optimized
- [x] `atlas/data/jarvis_client.py` - reviewed + optimized
- [x] `atlas/data/crystal_dataset.py` - reviewed + optimized

## Batch 112 optimization goals
- Improve numerical/parameter sanitization to avoid hidden runtime failures in validation and acquisition preprocessing paths.
- Reduce repeated formula/element parsing overhead in property/alloy estimators with safe caching.
- Strengthen dataset label-presence handling and graph-build observability for split preparation.
- Add fail-fast contract checks for invalid stability filter ranges and cover them with regression tests.

## Batch 112 outcomes
- `atlas/data/data_validation.py`:
  - added `_coerce_positive_int(...)` for strict positive-integer coercion used by statistical routines.
  - hardened `compute_mmd_rbf(...)` against invalid `max_points` values (sanitized lower bound, deterministic behavior preserved).
  - hardened `compute_drift(...)` against invalid `n_bins` values to prevent division/shape edge failures.
- `atlas/data/property_estimator.py`:
  - added class-level element-mass cache (`_ELEMENT_MASS_CACHE`) to avoid repeated optional dependency lookup overhead.
  - added per-instance stoichiometry cache (`_formula_stoich_cache`) with defensive copy semantics.
  - reused cached stoichiometry in mass/complexity paths to reduce repeated formula parsing.
- `atlas/data/alloy_estimator.py`:
  - added `functools.lru_cache` for `_element_atomic_mass(...)` and `_formula_molar_mass(...)`.
  - improved element symbol normalization before lookup to reduce fallback misses.
- `atlas/data/jarvis_client.py`:
  - added `_coerce_optional_finite(...)` helper for optional numeric filter parameters.
  - `get_stable_materials(...)` now fails fast when `min_band_gap > max_band_gap` instead of silently returning empty subsets.
- `atlas/data/crystal_dataset.py`:
  - added normalized missing-label handling (`_MISSING_LABEL_TOKENS`, `_is_present_label_value`, `_valid_label_mask`) for consistent presence detection across pipeline stages.
  - replaced ad-hoc property presence checks in split filtering and worker extraction with unified logic.
  - added per-prepare conversion summary tracking (`_prepare_summary` + `prepare_summary()`), including failed/dropped row counts and failure rate.
  - added warning log when graph conversion drops samples.

## Tests updated
- `tests/unit/data/test_data_validation.py`:
  - added regression for invalid `n_bins` sanitization in `compute_drift(...)`.
  - added regression for invalid `max_points` sanitization in `compute_mmd_rbf(...)`.
- `tests/unit/data/test_jarvis_client.py`:
  - added regression verifying `get_stable_materials(...)` rejects inverted band-gap windows.
- `tests/unit/data/test_crystal_dataset.py`:
  - expanded `min_labeled_properties` coverage to normalize `" NaN "`-style missing tokens.
  - added regression verifying `prepare_summary()` reports dropped graph rows.

## Verification
- `python -m ruff check .`
- `python -m pytest -q tests/unit/data/test_data_validation.py tests/unit/data/test_property_estimator.py tests/unit/data/test_alloy_estimator.py tests/unit/data/test_jarvis_client.py tests/unit/data/test_crystal_dataset.py`

## Research references used in batch 112
- Python `functools.lru_cache` docs: https://docs.python.org/3/library/functools.html#functools.lru_cache
- pandas missing-value detection (`Series.notna`): https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.notna.html
- NumPy finite-value semantics (`numpy.isfinite`): https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- NumPy pseudo-random API (`RandomState`): https://numpy.org/doc/stable/reference/random/generated/numpy.random.RandomState.html
- KS two-sample test reference implementation notes (SciPy): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
- Gretton et al. (2012), kernel two-sample test / MMD: https://www.jmlr.org/papers/v13/gretton12a.html
- Choudhary et al. (2020), JARVIS database overview: https://www.nature.com/articles/s41524-020-00440-1
- pymatgen usage/docs index: https://pymatgen.org/

## Progress snapshot (after Batch 112)
- Completed: Batch 1 through Batch 112.
- Pending: Batch 113 onward.
- Sequence position (reset plan): currently in `core` stage, next is `core` Batch 113.
