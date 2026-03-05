"""
Matbench benchmark runner with reproducible fold-level reports.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from atlas.config import get_config
from atlas.models.graph_builder import CrystalGraphBuilder
from atlas.models.prediction_utils import extract_mean_and_std


def _coerce_positive_int(value: Any, default: int, minimum: int = 1) -> int:
    if isinstance(value, bool):
        return max(int(default), int(minimum))
    if isinstance(value, Integral):
        out = int(value)
    elif isinstance(value, Real):
        out_f = float(value)
        if not math.isfinite(out_f) or not out_f.is_integer():
            return max(int(default), int(minimum))
        out = int(out_f)
    else:
        try:
            out = int(value)
        except Exception:
            return max(int(default), int(minimum))
    if out < minimum:
        return int(minimum)
    return int(out)


def _coerce_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(default)
    if isinstance(value, Integral):
        out = int(value)
    elif isinstance(value, Real):
        out_f = float(value)
        if not math.isfinite(out_f) or not out_f.is_integer():
            return int(default)
        out = int(out_f)
    else:
        try:
            out = int(value)
        except Exception:
            return int(default)
    return int(out)


def _coerce_probability(value: Any, default: float, *, lo: float = 0.0, hi: float = 1.0) -> float:
    if isinstance(value, bool):
        return float(default)
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if not math.isfinite(out):
        out = float(default)
    return float(min(max(out, lo), hi))


def _coerce_n_jobs(value: Any, default: int = -1) -> int:
    out = _coerce_int(value, default=default)
    if out == 0:
        return int(default if default != 0 else 1)
    return int(out)


def _coerce_non_negative_fold_id(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"fold indices must be non-negative integers, got {value!r}")
    if isinstance(value, Integral):
        idx = int(value)
    elif isinstance(value, Real):
        raw = float(value)
        if not math.isfinite(raw) or not raw.is_integer():
            raise ValueError(f"fold indices must be non-negative integers, got {value!r}")
        idx = int(raw)
    else:
        try:
            idx = int(value)
        except Exception as exc:
            raise ValueError(f"fold indices must be non-negative integers, got {value!r}") from exc
    if idx < 0:
        raise ValueError(f"fold indices must be non-negative integers, got {value!r}")
    return idx


def _sanitize_bootstrap_params(
    confidence: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, int, int]:
    conf = _coerce_probability(confidence, default=0.95, lo=0.5, hi=0.999)
    boots = _coerce_positive_int(n_bootstrap, default=400, minimum=64)
    seed_i = _coerce_int(seed, default=42)
    return conf, boots, seed_i


def _to_numeric_array(values: Sequence[Any]) -> np.ndarray:
    """Convert labels to a numeric 1D array, coercing invalid entries to NaN."""
    series = pd.Series(list(values))
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def _weighted_nan_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean over finite values/weights only."""
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if mask.sum() == 0:
        return np.nan
    v = values[mask]
    w = weights[mask]
    return float(np.sum(v * w) / np.sum(w))


def _normal_quantile_for_two_sided_coverage(coverage: float) -> float:
    """
    z-value for symmetric two-sided Gaussian interval with target coverage.

    Ref:
    - Kuleshov et al. (2018), calibrated regression:
      https://proceedings.mlr.press/v80/kuleshov18a.html
    """
    c = min(max(float(coverage), 1e-6), 1.0 - 1e-6)
    p = 0.5 + 0.5 * c
    # Use torch Normal.icdf to avoid extra scipy dependency.
    return float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(p, dtype=torch.float64)).item())


def _bootstrap_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 400,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for a 1D statistic sample.

    Ref:
    - Efron & Tibshirani (1994), An Introduction to the Bootstrap.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        v = float(arr[0])
        return v, v

    conf, b, seed_i = _sanitize_bootstrap_params(confidence, n_bootstrap, seed)
    alpha = 0.5 * (1.0 - conf)
    rng = np.random.RandomState(seed_i)
    idx = rng.randint(0, arr.size, size=(b, arr.size))
    means = np.mean(arr[idx], axis=1)
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return lo, hi


def _bootstrap_metric_ci_from_indices(
    n_samples: int,
    metric_fn,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 400,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap CI for a scalar metric defined on index resamples.

    `metric_fn` receives a 1D integer index array and returns a scalar.
    """
    n = int(n_samples)
    if n <= 0:
        return np.nan, np.nan
    if n == 1:
        v = float(metric_fn(np.array([0], dtype=np.int64)))
        return v, v

    conf, b, seed_i = _sanitize_bootstrap_params(confidence, n_bootstrap, seed)
    alpha = 0.5 * (1.0 - conf)
    rng = np.random.RandomState(seed_i)
    idx = rng.randint(0, n, size=(b, n))
    stats = np.asarray([float(metric_fn(row)) for row in idx], dtype=np.float64)
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return np.nan, np.nan
    lo = float(np.quantile(stats, alpha))
    hi = float(np.quantile(stats, 1.0 - alpha))
    return lo, hi


def _conformal_quantile_abs_residual(residuals: np.ndarray, coverage: float) -> float:
    """
    Split-conformal quantile for absolute residuals.

    Ref:
    - Angelopoulos & Bates (2021), conformal prediction tutorial:
      https://arxiv.org/abs/2107.07511
    """
    r = np.asarray(residuals, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan

    c = min(max(float(coverage), 1e-6), 1.0 - 1e-6)
    alpha = 1.0 - c
    n = int(r.size)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(1, k), n)
    return float(np.partition(r, k - 1)[k - 1])


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    z_t = torch.as_tensor(z, dtype=torch.float64)
    out = 0.5 * (1.0 + torch.erf(z_t / np.sqrt(2.0)))
    return out.detach().cpu().numpy().astype(np.float64)


def compute_regression_metrics(targets: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Compute regression metrics on finite target/prediction pairs."""
    mask = np.isfinite(targets) & np.isfinite(preds)
    if mask.sum() == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}

    y = targets[mask]
    y_hat = preds[mask]
    mae = float(np.mean(np.abs(y - y_hat)))
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))

    if y.size < 2:
        r2 = np.nan
    else:
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = np.nan if ss_tot <= 1e-12 else float(1.0 - (ss_res / ss_tot))

    return {"mae": mae, "rmse": rmse, "r2": r2}


def compute_uncertainty_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
    stds: np.ndarray | None,
) -> dict[str, float]:
    """
    Compute uncertainty diagnostics on finite target/pred/std triplets.

    Metrics are intentionally lightweight and dependency-free.
    """
    if stds is None:
        return {}

    mask = np.isfinite(targets) & np.isfinite(preds) & np.isfinite(stds)
    if mask.sum() == 0:
        return {
            "avg_std": np.nan,
            "gaussian_nll": np.nan,
            "pi90_coverage": np.nan,
            "pi95_coverage": np.nan,
            "pi95_miscalibration": np.nan,
            "regression_ece": np.nan,
            "interval_score_95": np.nan,
            "crps_gaussian": np.nan,
            "std_residual_mean": np.nan,
            "std_residual_std": np.nan,
        }

    y = targets[mask]
    y_hat = preds[mask]
    sigma = np.asarray(stds[mask], dtype=np.float64)
    sigma = np.clip(sigma, 1e-12, None)
    err = y - y_hat

    # Classical predictive interval coverage (normal approximation).
    # Ref: Kuleshov et al. (2018), calibrated regression.
    z90 = 1.6448536269514722
    z95 = 1.959963984540054
    pi90_cov = float(np.mean(np.abs(err) <= z90 * sigma))
    pi95_cov = float(np.mean(np.abs(err) <= z95 * sigma))
    gaussian_nll = float(np.mean(0.5 * np.log(2.0 * np.pi * sigma * sigma) + 0.5 * (err * err) / (sigma * sigma)))

    # Regression calibration error over multiple nominal coverages.
    # Ref: Kuleshov et al. (2018), expected calibration for regression.
    nominal_coverages = np.array([0.50, 0.70, 0.80, 0.90, 0.95], dtype=np.float64)
    cal_errors: list[float] = []
    abs_err = np.abs(err)
    for cov in nominal_coverages:
        z = _normal_quantile_for_two_sided_coverage(float(cov))
        empirical_cov = float(np.mean(abs_err <= z * sigma))
        cal_errors.append(abs(empirical_cov - float(cov)))
    regression_ece = float(np.mean(cal_errors)) if cal_errors else np.nan

    # Proper interval scoring rule for 95% PI (lower is better).
    # Ref: Gneiting & Raftery (2007), strictly proper scoring rules.
    alpha = 0.05
    lower = y_hat - z95 * sigma
    upper = y_hat + z95 * sigma
    below = (y < lower).astype(np.float64)
    above = (y > upper).astype(np.float64)
    interval_score_95 = float(
        np.mean(
            (upper - lower)
            + (2.0 / alpha) * (lower - y) * below
            + (2.0 / alpha) * (y - upper) * above
        )
    )

    # Closed-form CRPS for Gaussian predictive distributions.
    # Ref: Gneiting & Raftery (2007), proper scoring rules.
    z = err / sigma
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    Phi = _normal_cdf(z)
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - (1.0 / np.sqrt(np.pi)))

    z_residual = err / sigma
    return {
        "avg_std": float(np.mean(sigma)),
        "gaussian_nll": gaussian_nll,
        "pi90_coverage": pi90_cov,
        "pi95_coverage": pi95_cov,
        "pi95_miscalibration": float(abs(pi95_cov - 0.95)),
        "regression_ece": regression_ece,
        "interval_score_95": interval_score_95,
        "crps_gaussian": float(np.mean(crps)),
        "std_residual_mean": float(np.mean(z_residual)),
        "std_residual_std": float(np.std(z_residual)),
    }


@dataclass
class FoldResult:
    """Compact metrics for one benchmark fold."""

    fold: int
    n_test: int
    n_valid: int
    coverage: float
    mae: float = np.nan
    rmse: float = np.nan
    r2: float = np.nan
    avg_std: float = np.nan
    gaussian_nll: float = np.nan
    pi90_coverage: float = np.nan
    pi95_coverage: float = np.nan
    pi95_miscalibration: float = np.nan
    regression_ece: float = np.nan
    interval_score_95: float = np.nan
    crps_gaussian: float = np.nan
    std_residual_mean: float = np.nan
    std_residual_std: float = np.nan
    conformal_qhat: float = np.nan
    conformal_pi_coverage: float = np.nan
    conformal_miscalibration: float = np.nan
    conformal_interval_score: float = np.nan
    conformal_calibration_size: int = 0
    runtime_sec: float = 0.0
    status: str = "ok"
    error: str = ""


def aggregate_fold_results(fold_results: Sequence[FoldResult]) -> dict[str, float]:
    """Aggregate fold-level results into mean/std summaries."""
    aggregate: dict[str, float] = {
        "n_folds": len(fold_results),
        "n_successful_folds": sum(r.status == "ok" for r in fold_results),
        "n_low_coverage_folds": sum(r.status == "low_coverage" for r in fold_results),
        "n_failed_folds": sum(r.status == "failed" for r in fold_results),
        "total_test_samples": sum(r.n_test for r in fold_results),
        "total_valid_samples": sum(r.n_valid for r in fold_results),
        "total_conformal_calibration_samples": sum(int(getattr(r, "conformal_calibration_size", 0)) for r in fold_results),
    }
    n_folds = max(1, len(fold_results))
    aggregate["success_rate"] = float(aggregate["n_successful_folds"] / n_folds)
    valid_weights = np.asarray([max(0.0, float(r.n_valid)) for r in fold_results], dtype=np.float64)
    test_weights = np.asarray([max(0.0, float(r.n_test)) for r in fold_results], dtype=np.float64)

    # Keep legacy unweighted mean/std for compatibility while adding
    # sample-weighted means to reduce fold-size bias.
    # Ref: Matbench CV reporting practice (Dunn et al., 2020).
    for key in (
        "mae",
        "rmse",
        "r2",
        "coverage",
        "avg_std",
        "gaussian_nll",
        "pi90_coverage",
        "pi95_coverage",
        "pi95_miscalibration",
        "regression_ece",
        "interval_score_95",
        "crps_gaussian",
        "std_residual_mean",
        "std_residual_std",
        "conformal_qhat",
        "conformal_pi_coverage",
        "conformal_miscalibration",
        "conformal_interval_score",
    ):
        vals = np.asarray(
            [float(getattr(r, key)) for r in fold_results if np.isfinite(getattr(r, key))],
            dtype=np.float64,
        )
        aggregate[f"{key}_mean"] = float(np.mean(vals)) if vals.size else np.nan
        aggregate[f"{key}_std"] = float(np.std(vals)) if vals.size else np.nan

        raw_vals = np.asarray([float(getattr(r, key)) for r in fold_results], dtype=np.float64)
        weights = test_weights if key == "coverage" else valid_weights
        aggregate[f"{key}_weighted_mean"] = _weighted_nan_mean(raw_vals, weights)

    # Conformal metrics are already computed on test residuals; weighted mean
    # is the most stable global summary across folds with unequal sizes.
    aggregate["global_conformal_pi_coverage"] = aggregate.get("conformal_pi_coverage_weighted_mean", np.nan)
    aggregate["global_conformal_miscalibration"] = aggregate.get("conformal_miscalibration_weighted_mean", np.nan)
    return aggregate


@dataclass
class TaskReport:
    """Serializable report for one Matbench task run."""

    task_name: str
    property_name: str
    model_name: str
    device: str
    timestamp: float
    fold_results: list[FoldResult] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    report_path: str | None = None


class MatbenchRunner:
    """
    Run Matbench tasks with a trained ATLAS model in inference mode.

    This runner focuses on reproducible reporting:
    - fold-level metrics
    - aggregate summaries
    - JSON artifact output for thesis/benchmark tracking
    """

    TASKS = {
        "matbench_mp_e_form": "formation_energy",
        "matbench_mp_gap": "band_gap",
        "matbench_log_gvrh": "shear_modulus",
        "matbench_log_kvrh": "bulk_modulus",
        "matbench_dielectric": "dielectric",
        "matbench_phonons": "phonons",
        "matbench_jdft2d": "exfoliation_energy",
        "matbench_steels": "yield_strength",
    }
    CALL_VARIANT_NAMES = (
        "x_edge_attr_batch",
        "x_edge_vec_batch",
        "x_edge_attr_edge_vec_batch",
        "batch_object",
    )

    def __init__(
        self,
        model,
        property_name: str,
        device: str = "auto",
        batch_size: int = 32,
        n_jobs: int = -1,
        output_dir: Path | None = None,
        bootstrap_samples: int = 400,
        bootstrap_seed: int = 42,
        min_coverage_required: float = 0.0,
        use_conformal: bool = False,
        conformal_coverage: float = 0.95,
        conformal_max_calibration_samples: int = 0,
        structure_cache: bool = True,
        fail_on_fallback_signature: bool = False,
    ):
        self.model = model
        self.property_name = property_name
        self.cfg = get_config()
        self.graph_builder = CrystalGraphBuilder()
        self.batch_size = _coerce_positive_int(batch_size, default=32, minimum=1)
        self.n_jobs = _coerce_n_jobs(n_jobs, default=-1)
        self.bootstrap_samples = _coerce_positive_int(bootstrap_samples, default=400, minimum=64)
        self.bootstrap_seed = _coerce_int(bootstrap_seed, default=42)
        self.min_coverage_required = _coerce_probability(min_coverage_required, default=0.0, lo=0.0, hi=1.0)
        self.use_conformal = bool(use_conformal)
        self.conformal_coverage = _coerce_probability(
            conformal_coverage,
            default=0.95,
            lo=1e-6,
            hi=1.0 - 1e-6,
        )
        self.conformal_max_calibration_samples = _coerce_positive_int(
            conformal_max_calibration_samples,
            default=0,
            minimum=0,
        )
        self.structure_cache = bool(structure_cache)
        self.fail_on_fallback_signature = bool(fail_on_fallback_signature)
        self._structure_data_cache: dict[str, Any] = {}
        self._structure_cache_hits = 0
        self._structure_cache_misses = 0
        self._call_variant_counts = [0, 0, 0, 0]

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.output_dir = output_dir or (self.cfg.paths.artifacts_dir / "benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _structure_cache_key(structure: Any) -> str | None:
        if structure is None:
            return None
        try:
            if hasattr(structure, "as_dict"):
                payload = structure.as_dict()
            elif isinstance(structure, (dict, str, int, float, bool)):
                payload = structure
            else:
                payload = str(structure)
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
            return hashlib.sha1(raw.encode("utf-8")).hexdigest()
        except Exception:
            return None

    def load_task(self, task_name: str):
        """Lazy-load a Matbench task."""
        try:
            from matbench.bench import MatbenchBenchmark
        except ImportError as exc:
            raise ImportError("Matbench not installed. Run: pip install matbench") from exc

        mb = MatbenchBenchmark(autoload=False)
        try:
            task = mb.tasks_map[task_name]
        except KeyError as exc:
            raise ValueError(f"Unknown task: {task_name}") from exc
        task.load()
        return task

    def structure_to_data(self, structure):
        """Convert a pymatgen structure to a PyG graph Data object."""
        try:
            return self.graph_builder.structure_to_pyg(structure)
        except Exception:
            return None

    def _predict_batch(self, batch) -> tuple[np.ndarray, np.ndarray | None]:
        """Run model forward with resilient call signatures and output extraction."""
        call_variants = (
            lambda: self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch),
            lambda: self.model(batch.x, batch.edge_index, batch.edge_vec, batch.batch),
            lambda: self.model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_vec, batch.batch),
            lambda: self.model(batch),
        )

        output = None
        last_error: Exception | None = None
        chosen_variant = -1
        for variant_idx, call in enumerate(call_variants):
            try:
                output = call()
                chosen_variant = int(variant_idx)
                break
            except Exception as exc:  # pragma: no cover - only used for fallback probing
                last_error = exc
                continue

        if output is None:
            raise RuntimeError(f"Model forward failed for all call variants: {last_error}")
        if chosen_variant > 0 and bool(self.fail_on_fallback_signature):
            name = self.CALL_VARIANT_NAMES[chosen_variant]
            raise RuntimeError(
                "Model forward required fallback signature "
                f"'{name}' (index={chosen_variant}) while fail_on_fallback_signature=True."
            )
        if 0 <= chosen_variant < len(self._call_variant_counts):
            self._call_variant_counts[chosen_variant] += 1

        if isinstance(output, dict):
            if self.property_name in output:
                output = output[self.property_name]
            elif len(output) == 1:
                output = next(iter(output.values()))
            else:
                keys = ", ".join(sorted(output.keys()))
                raise KeyError(f"Property '{self.property_name}' not found in model output keys: {keys}")

        mean_t, std_t = extract_mean_and_std(output)
        means = mean_t.detach().cpu().reshape(-1).numpy().astype(np.float64)
        stds = None
        if std_t is not None:
            stds = std_t.detach().cpu().reshape(-1).numpy().astype(np.float64)
        return means, stds

    def _convert_structures(self, structures: Sequence[Any], show_progress: bool) -> list[Any]:
        items = list(structures)
        if not items:
            return []
        if not bool(self.structure_cache):
            iterable: Iterable[Any] = tqdm(items, leave=False, disable=not show_progress)
            if self.n_jobs == 1:
                return [self.structure_to_data(s) for s in iterable]
            return Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self.structure_to_data)(s) for s in iterable
            )

        results: list[Any] = [None] * len(items)
        missing: list[tuple[int, Any, str | None]] = []
        for idx, structure in enumerate(items):
            key = self._structure_cache_key(structure)
            if key is not None and key in self._structure_data_cache:
                results[idx] = self._structure_data_cache[key]
                self._structure_cache_hits += 1
            else:
                missing.append((idx, structure, key))
                self._structure_cache_misses += 1

        if not missing:
            return results

        miss_structures = [s for _, s, _ in missing]
        iterable_missing: Iterable[Any] = tqdm(
            miss_structures,
            leave=False,
            disable=not show_progress,
        )
        if self.n_jobs == 1:
            converted = [self.structure_to_data(s) for s in iterable_missing]
        else:
            converted = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self.structure_to_data)(s) for s in iterable_missing
            )

        for (idx, _structure, key), data in zip(missing, converted, strict=False):
            results[idx] = data
            if key is not None:
                self._structure_data_cache[key] = data
        return results

    def _predict_on_inputs(
        self,
        inputs: Sequence[Any],
        *,
        show_progress: bool,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Predict on a structure list and preserve original order with NaN padding.
        """
        n = len(inputs)
        predictions = np.full(n, np.nan, dtype=np.float64)
        pred_stds = np.full(n, np.nan, dtype=np.float64)

        converted = self._convert_structures(inputs, show_progress=show_progress)
        data_list: list[Any] = []
        valid_indices: list[int] = []
        for idx, data in enumerate(converted):
            if data is not None:
                data_list.append(data)
                valid_indices.append(idx)

        n_valid = len(valid_indices)
        if n_valid == 0:
            return predictions, pred_stds, 0

        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)
        cursor = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                batch_pred, batch_std = self._predict_batch(batch)

                n_graphs = int(getattr(batch, "num_graphs", len(batch_pred)))
                if batch_pred.size == 1 and n_graphs > 1:
                    batch_pred = np.repeat(batch_pred, n_graphs)
                if batch_std is not None and batch_std.size == 1 and n_graphs > 1:
                    batch_std = np.repeat(batch_std, n_graphs)
                if batch_pred.size != n_graphs and batch_pred.size == batch.batch.numel():
                    batch_indices = batch.batch.detach().cpu().numpy()
                    reduced = []
                    for g_idx in range(n_graphs):
                        node_mask = batch_indices == g_idx
                        reduced.append(float(np.mean(batch_pred[node_mask])))
                    batch_pred = np.asarray(reduced, dtype=np.float64)
                if batch_std is not None and batch_std.size != n_graphs and batch_std.size == batch.batch.numel():
                    batch_indices = batch.batch.detach().cpu().numpy()
                    reduced_std = []
                    for g_idx in range(n_graphs):
                        node_mask = batch_indices == g_idx
                        reduced_std.append(float(np.mean(batch_std[node_mask])))
                    batch_std = np.asarray(reduced_std, dtype=np.float64)

                if batch_pred.size != n_graphs:
                    raise ValueError(
                        f"Prediction size mismatch: got {batch_pred.size}, expected {n_graphs}"
                    )
                if batch_std is not None and batch_std.size != n_graphs:
                    raise ValueError(
                        f"Uncertainty size mismatch: got {batch_std.size}, expected {n_graphs}"
                    )

                idx_slice = valid_indices[cursor : cursor + n_graphs]
                for local_idx, (global_idx, pred) in enumerate(zip(idx_slice, batch_pred, strict=False)):
                    predictions[global_idx] = float(pred)
                    if batch_std is not None:
                        pred_stds[global_idx] = float(batch_std[local_idx])
                cursor += n_graphs

        return predictions, pred_stds, n_valid

    def _compute_conformal_metrics(
        self,
        task: Any,
        fold: int,
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> dict[str, float]:
        """
        Split-conformal diagnostics from fold train/val residuals.

        Ref:
        - Angelopoulos & Bates (2021), conformal prediction.
        - Romano et al. (2019), CQR framing:
          https://arxiv.org/abs/1905.03222
        """
        default = {
            "conformal_qhat": np.nan,
            "conformal_pi_coverage": np.nan,
            "conformal_miscalibration": np.nan,
            "conformal_interval_score": np.nan,
            "conformal_calibration_size": 0.0,
        }
        if not bool(self.use_conformal):
            return default

        if not hasattr(task, "get_train_and_val_data"):
            return default

        try:
            cal_inputs_raw, cal_outputs = task.get_train_and_val_data(fold, include_target=True)
        except Exception:
            return default

        cal_inputs = list(cal_inputs_raw)
        cal_targets = _to_numeric_array(cal_outputs)
        n = min(len(cal_inputs), len(cal_targets))
        if n <= 0:
            return default
        cal_inputs = cal_inputs[:n]
        cal_targets = cal_targets[:n]

        limit = _coerce_positive_int(
            getattr(self, "conformal_max_calibration_samples", 0),
            default=0,
            minimum=0,
        )
        if limit > 0 and n > limit:
            rng = np.random.RandomState(int(self.bootstrap_seed) + int(fold) + 7919)
            picked = np.sort(rng.choice(n, size=limit, replace=False))
            cal_inputs = [cal_inputs[i] for i in picked]
            cal_targets = cal_targets[picked]

        try:
            cal_preds, _, _ = self._predict_on_inputs(cal_inputs, show_progress=False)
        except Exception:
            return default

        finite_cal = np.isfinite(cal_targets) & np.isfinite(cal_preds)
        residuals = np.abs(cal_targets[finite_cal] - cal_preds[finite_cal])
        qhat = _conformal_quantile_abs_residual(residuals, self.conformal_coverage)
        if not np.isfinite(qhat):
            return default

        finite_test = np.isfinite(targets) & np.isfinite(predictions)
        if finite_test.sum() == 0:
            out = dict(default)
            out["conformal_qhat"] = float(qhat)
            out["conformal_calibration_size"] = float(residuals.size)
            return out

        err = np.abs(targets[finite_test] - predictions[finite_test])
        cov = float(np.mean(err <= qhat))
        target_cov = float(self.conformal_coverage)
        alpha = max(1e-6, 1.0 - target_cov)
        # Symmetric interval score for conformal interval [mu-qhat, mu+qhat].
        interval_score = float(np.mean((2.0 * qhat) + (2.0 / alpha) * np.maximum(0.0, err - qhat)))
        return {
            "conformal_qhat": float(qhat),
            "conformal_pi_coverage": cov,
            "conformal_miscalibration": float(abs(cov - target_cov)),
            "conformal_interval_score": interval_score,
            "conformal_calibration_size": float(residuals.size),
        }

    def _run_fold_details(self, task, fold: int, show_progress: bool = True):
        """Run a single fold and return targets, predictions, and fold metrics."""
        t0 = time.time()
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
        inputs = list(test_inputs)
        targets = _to_numeric_array(test_outputs)

        if len(inputs) != len(targets):
            n = min(len(inputs), len(targets))
            inputs = inputs[:n]
            targets = targets[:n]

        n_test = len(inputs)
        try:
            predictions, pred_stds, n_valid = self._predict_on_inputs(inputs, show_progress=show_progress)
        except Exception as exc:
            result = FoldResult(
                fold=fold,
                n_test=n_test,
                n_valid=0,
                coverage=0.0 if n_test == 0 else np.nan,
                runtime_sec=time.time() - t0,
                status="failed",
                error=str(exc),
            )
            return targets, np.full(n_test, np.nan, dtype=np.float64), np.full(n_test, np.nan, dtype=np.float64), result
        coverage = 0.0 if n_test == 0 else float(n_valid / n_test)

        if n_valid == 0:
            result = FoldResult(
                fold=fold,
                n_test=n_test,
                n_valid=0,
                coverage=coverage,
                runtime_sec=time.time() - t0,
                status="no_valid_structures",
                error="structure_to_data failed for all samples",
            )
            return targets, predictions, pred_stds, result

        try:
            metrics = compute_regression_metrics(targets, predictions)
            uq_metrics = compute_uncertainty_metrics(targets, predictions, pred_stds)
            conformal_metrics = self._compute_conformal_metrics(task, fold, targets, predictions)
            status = "ok"
            error = ""
            if coverage < float(getattr(self, "min_coverage_required", 0.0)):
                status = "low_coverage"
                error = (
                    f"coverage {coverage:.4f} below minimum "
                    f"{float(getattr(self, 'min_coverage_required', 0.0)):.4f}"
                )
            result = FoldResult(
                fold=fold,
                n_test=n_test,
                n_valid=n_valid,
                coverage=coverage,
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                r2=metrics["r2"],
                avg_std=uq_metrics.get("avg_std", np.nan),
                gaussian_nll=uq_metrics.get("gaussian_nll", np.nan),
                pi90_coverage=uq_metrics.get("pi90_coverage", np.nan),
                pi95_coverage=uq_metrics.get("pi95_coverage", np.nan),
                pi95_miscalibration=uq_metrics.get("pi95_miscalibration", np.nan),
                regression_ece=uq_metrics.get("regression_ece", np.nan),
                interval_score_95=uq_metrics.get("interval_score_95", np.nan),
                crps_gaussian=uq_metrics.get("crps_gaussian", np.nan),
                std_residual_mean=uq_metrics.get("std_residual_mean", np.nan),
                std_residual_std=uq_metrics.get("std_residual_std", np.nan),
                conformal_qhat=conformal_metrics.get("conformal_qhat", np.nan),
                conformal_pi_coverage=conformal_metrics.get("conformal_pi_coverage", np.nan),
                conformal_miscalibration=conformal_metrics.get("conformal_miscalibration", np.nan),
                conformal_interval_score=conformal_metrics.get("conformal_interval_score", np.nan),
                conformal_calibration_size=int(conformal_metrics.get("conformal_calibration_size", 0.0)),
                runtime_sec=time.time() - t0,
                status=status,
                error=error,
            )
            return targets, predictions, pred_stds, result
        except Exception as exc:
            result = FoldResult(
                fold=fold,
                n_test=n_test,
                n_valid=n_valid,
                coverage=coverage,
                runtime_sec=time.time() - t0,
                status="failed",
                error=str(exc),
            )
            return targets, predictions, pred_stds, result

    def run_fold(self, task, fold: int):
        """
        Backward-compatible fold run API.

        Returns:
            tuple[np.ndarray, np.ndarray]: targets and predictions.
        """
        targets, preds, _, _ = self._run_fold_details(task, fold, show_progress=True)
        return targets, preds

    def _default_report_path(self, task_name: str) -> Path:
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        return self.output_dir / f"{task_name}_{self.property_name}_{ts}.json"

    def _write_report(self, report: TaskReport, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(report)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, default=float, sort_keys=True)

    def run_task(
        self,
        task_name: str,
        folds: Sequence[int] | None = None,
        save_path: Path | None = None,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """
        Run selected folds for one Matbench task and persist a JSON report.
        """
        task = self.load_task(task_name)
        call_counts_before = np.asarray(self._call_variant_counts, dtype=np.int64)
        cache_hits_before = int(self._structure_cache_hits)
        cache_misses_before = int(self._structure_cache_misses)
        fold_ids_raw = list(folds) if folds is not None else list(getattr(task, "folds", []))
        fold_ids = sorted({_coerce_non_negative_fold_id(f) for f in fold_ids_raw})
        if not fold_ids:
            raise ValueError(f"No folds available for task '{task_name}'")

        fold_results: list[FoldResult] = []
        all_targets: list[np.ndarray] = []
        all_preds: list[np.ndarray] = []
        all_stds: list[np.ndarray] = []

        for fold in fold_ids:
            targets, preds, stds, fold_result = self._run_fold_details(task, fold, show_progress=show_progress)
            fold_results.append(fold_result)
            all_targets.append(targets)
            all_preds.append(preds)
            all_stds.append(stds)

        aggregate = aggregate_fold_results(fold_results)
        if all_targets and all_preds:
            merged_targets = np.concatenate(all_targets)
            merged_preds = np.concatenate(all_preds)
            global_metrics = compute_regression_metrics(merged_targets, merged_preds)
            for key, value in global_metrics.items():
                aggregate[f"global_{key}"] = value
            finite_global = np.isfinite(merged_targets) & np.isfinite(merged_preds)
            y_global = merged_targets[finite_global]
            yhat_global = merged_preds[finite_global]
            err = np.abs(y_global - yhat_global)
            lo, hi = _bootstrap_ci(
                err,
                confidence=0.95,
                n_bootstrap=self.bootstrap_samples,
                seed=self.bootstrap_seed,
            )
            aggregate["global_mae_ci95_lo"] = lo
            aggregate["global_mae_ci95_hi"] = hi
            if y_global.size > 0:
                rmse_lo, rmse_hi = _bootstrap_metric_ci_from_indices(
                    y_global.size,
                    lambda idx: float(np.sqrt(np.mean((y_global[idx] - yhat_global[idx]) ** 2))),
                    confidence=0.95,
                    n_bootstrap=self.bootstrap_samples,
                    seed=self.bootstrap_seed + 11,
                )
                aggregate["global_rmse_ci95_lo"] = rmse_lo
                aggregate["global_rmse_ci95_hi"] = rmse_hi
                r2_lo, r2_hi = _bootstrap_metric_ci_from_indices(
                    y_global.size,
                    lambda idx: (
                        np.nan
                        if idx.size < 2
                        else (
                            np.nan
                            if float(np.sum((y_global[idx] - np.mean(y_global[idx])) ** 2)) <= 1e-12
                            else float(
                                1.0
                                - (
                                    float(np.sum((y_global[idx] - yhat_global[idx]) ** 2))
                                    / float(np.sum((y_global[idx] - np.mean(y_global[idx])) ** 2))
                                )
                            )
                        )
                    ),
                    confidence=0.95,
                    n_bootstrap=self.bootstrap_samples,
                    seed=self.bootstrap_seed + 23,
                )
                aggregate["global_r2_ci95_lo"] = r2_lo
                aggregate["global_r2_ci95_hi"] = r2_hi

            if all_stds:
                merged_stds = np.concatenate(all_stds)
                global_uq = compute_uncertainty_metrics(merged_targets, merged_preds, merged_stds)
                for key, value in global_uq.items():
                    aggregate[f"global_{key}"] = value
                uq_mask = (
                    np.isfinite(merged_targets)
                    & np.isfinite(merged_preds)
                    & np.isfinite(merged_stds)
                )
                if int(np.sum(uq_mask)) > 0:
                    y_uq = merged_targets[uq_mask].astype(np.float64)
                    yhat_uq = merged_preds[uq_mask].astype(np.float64)
                    sigma_uq = np.clip(
                        merged_stds[uq_mask].astype(np.float64),
                        1e-12,
                        None,
                    )
                    err_uq = y_uq - yhat_uq
                    z95 = 1.959963984540054
                    pi95_hit = (np.abs(err_uq) <= z95 * sigma_uq).astype(np.float64)
                    pi95_lo, pi95_hi = _bootstrap_ci(
                        pi95_hit,
                        confidence=0.95,
                        n_bootstrap=self.bootstrap_samples,
                        seed=self.bootstrap_seed + 31,
                    )
                    aggregate["global_pi95_coverage_ci95_lo"] = pi95_lo
                    aggregate["global_pi95_coverage_ci95_hi"] = pi95_hi

                    nll_i = 0.5 * np.log(2.0 * np.pi * sigma_uq * sigma_uq) + 0.5 * (
                        (err_uq * err_uq) / (sigma_uq * sigma_uq)
                    )
                    nll_lo, nll_hi = _bootstrap_ci(
                        nll_i,
                        confidence=0.95,
                        n_bootstrap=self.bootstrap_samples,
                        seed=self.bootstrap_seed + 37,
                    )
                    aggregate["global_gaussian_nll_ci95_lo"] = nll_lo
                    aggregate["global_gaussian_nll_ci95_hi"] = nll_hi

                    z = err_uq / sigma_uq
                    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
                    Phi = _normal_cdf(z)
                    crps_i = sigma_uq * (
                        z * (2.0 * Phi - 1.0) + 2.0 * phi - (1.0 / np.sqrt(np.pi))
                    )
                    crps_lo, crps_hi = _bootstrap_ci(
                        crps_i,
                        confidence=0.95,
                        n_bootstrap=self.bootstrap_samples,
                        seed=self.bootstrap_seed + 41,
                    )
                    aggregate["global_crps_gaussian_ci95_lo"] = crps_lo
                    aggregate["global_crps_gaussian_ci95_hi"] = crps_hi

        call_counts_after = np.asarray(self._call_variant_counts, dtype=np.int64)
        call_counts_run = (call_counts_after - call_counts_before).astype(int)
        call_counts_dict = {
            name: int(call_counts_run[i])
            for i, name in enumerate(self.CALL_VARIANT_NAMES)
        }
        fallback_calls = int(np.sum(call_counts_run[1:])) if call_counts_run.size >= 2 else 0
        cache_hits_run = int(self._structure_cache_hits - cache_hits_before)
        cache_misses_run = int(self._structure_cache_misses - cache_misses_before)

        report = TaskReport(
            task_name=task_name,
            property_name=self.property_name,
            model_name=self.model.__class__.__name__,
            device=str(self.device),
            timestamp=time.time(),
            fold_results=fold_results,
            aggregate_metrics=aggregate,
            settings={
                "batch_size": self.batch_size,
                "n_jobs": self.n_jobs,
                "bootstrap_samples": self.bootstrap_samples,
                "bootstrap_seed": self.bootstrap_seed,
                "min_coverage_required": self.min_coverage_required,
                "use_conformal": self.use_conformal,
                "conformal_coverage": self.conformal_coverage,
                "conformal_max_calibration_samples": self.conformal_max_calibration_samples,
                "structure_cache": self.structure_cache,
                "structure_cache_hits": cache_hits_run,
                "structure_cache_misses": cache_misses_run,
                "structure_cache_size": int(len(self._structure_data_cache)),
                "fail_on_fallback_signature": self.fail_on_fallback_signature,
                "model_call_variant_counts": call_counts_dict,
                "model_call_fallback_count": fallback_calls,
                "seed": self.cfg.train.seed,
                "deterministic": self.cfg.train.deterministic,
                "method_key": self.cfg.profile.method_key,
                "data_source_key": self.cfg.profile.data_source_key,
            },
        )

        report_path = save_path or self._default_report_path(task_name)
        report.report_path = str(report_path)
        self._write_report(report, report_path)
        return asdict(report)

