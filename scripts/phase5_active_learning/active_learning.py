"""
Phase 6: Active Learning & Multi-Objective Materials Screening

Implements:
1. Uncertainty-driven active learning loop
2. Multi-objective Pareto screening for materials discovery
3. Acquisition functions: uncertainty, expected improvement, random

Usage:
    python scripts/phase5_active_learning/active_learning.py --strategy uncertainty --budget 500
    python scripts/phase5_active_learning/active_learning.py --strategy random --budget 500
    python scripts/phase5_active_learning/active_learning.py --pareto --objectives "high bulk_modulus" "low band_gap"
"""

import argparse
import math
import sys
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from atlas.config import get_config
from atlas.training.metrics import scalar_metrics

SUPPORTED_AL_STRATEGIES = {"uncertainty", "random", "expected_improvement"}
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

try:  # pragma: no cover - optional dependency
    from scipy.stats import norm as _scipy_norm
except Exception:  # pragma: no cover
    _scipy_norm = None


def _validate_batch_size(batch_size: int) -> int:
    if isinstance(batch_size, bool):
        raise ValueError("batch_size must be an integer >= 0")
    if isinstance(batch_size, Integral):
        size = int(batch_size)
    elif isinstance(batch_size, Real):
        size_f = float(batch_size)
        if not math.isfinite(size_f) or not size_f.is_integer():
            raise ValueError("batch_size must be an integer >= 0")
        size = int(size_f)
    else:
        try:
            size = int(batch_size)
        except Exception as exc:
            raise ValueError("batch_size must be an integer >= 0") from exc
    if size < 0:
        raise ValueError("batch_size must be >= 0")
    return size


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        number_f = float(value)
        if not math.isfinite(number_f) or not number_f.is_integer():
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        number = int(number_f)
    else:
        try:
            number = int(value)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return number


def _coerce_non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        number_f = float(value)
        if not math.isfinite(number_f) or not number_f.is_integer():
            raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
        number = int(number_f)
    else:
        try:
            number = int(value)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(f"{name} must be an integer >= 0, got {value!r}") from exc
    if number < 0:
        raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
    return number


def _to_numpy_1d(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be at least 1D")
    return arr.reshape(-1)


def _standard_normal_pdf(z: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * np.square(z))


def _standard_normal_cdf(z: np.ndarray) -> np.ndarray:
    if _scipy_norm is not None:
        return _scipy_norm.cdf(z)
    vec_erf = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + vec_erf(z / math.sqrt(2.0)))


def _forward_graph_model(model, batch):
    """Best-effort model forward across common graph model signatures."""
    if not hasattr(batch, "x") or not hasattr(batch, "edge_index"):
        return model(batch)

    x = batch.x
    edge_index = batch.edge_index
    batch_index = getattr(batch, "batch", None)
    edge_candidates = []
    for candidate in (getattr(batch, "edge_attr", None), getattr(batch, "edge_vec", None)):
        if candidate is not None and all(candidate is not seen for seen in edge_candidates):
            edge_candidates.append(candidate)
    edge_candidates.append(None)

    last_error: TypeError | None = None
    for edge_feats in edge_candidates:
        try:
            if edge_feats is None:
                return model(x, edge_index, batch=batch_index)
            return model(x, edge_index, edge_feats, batch_index)
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return model(batch)


# ─────────────────────────────────────────────────────────
#  Acquisition Functions
# ─────────────────────────────────────────────────────────

def acquisition_uncertainty(
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int = 50,
) -> np.ndarray:
    """
    Select samples with highest predictive uncertainty.

    Args:
        mean: (N,) or (N, d) predicted means
        std: (N,) or (N, d) predicted standard deviations
        batch_size: number of samples to select

    Returns:
        (batch_size,) indices of selected samples
    """
    mean_arr = np.asarray(mean, dtype=float)
    if mean_arr.ndim == 0:
        raise ValueError("mean must be at least 1D")
    k = _validate_batch_size(batch_size)
    if k == 0:
        return np.array([], dtype=int)

    std_arr = np.asarray(std, dtype=float)
    if std_arr.ndim == 0:
        raise ValueError("std must be at least 1D")
    if mean_arr.shape[0] != std_arr.shape[0]:
        raise ValueError("mean and std must have matching leading dimension")
    std_arr = np.nan_to_num(std_arr, nan=0.0, posinf=0.0, neginf=0.0)
    std_arr = np.abs(std_arr)

    # Use total uncertainty (sum over output dims if multi-dim)
    if std_arr.ndim > 1:
        total_unc = std_arr.sum(axis=-1)
    else:
        total_unc = std_arr
    if total_unc.size == 0:
        return np.array([], dtype=int)

    k = min(k, total_unc.shape[0])
    indices = np.argsort(total_unc, kind="mergesort")[-k:][::-1]
    return indices.astype(int, copy=False)


def acquisition_expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_so_far: float,
    batch_size: int = 50,
    maximize: bool = False,
) -> np.ndarray:
    """
    Expected Improvement acquisition function.

    EI(x) = E[max(0, f(x) - f*)]

    Args:
        mean: (N,) predicted means
        std: (N,) predicted stds
        best_so_far: best observed value
        batch_size: number of samples to select
        maximize: if True, maximize property; else minimize

    Returns:
        (batch_size,) indices of selected samples
    """
    k = _validate_batch_size(batch_size)
    if k == 0:
        return np.array([], dtype=int)
    if not np.isfinite(float(best_so_far)):
        raise ValueError("best_so_far must be finite")

    mean_arr = _to_numpy_1d(mean, name="mean")
    std_arr = _to_numpy_1d(std, name="std")
    if mean_arr.shape != std_arr.shape:
        raise ValueError("mean and std must have matching shapes")
    if mean_arr.size == 0:
        return np.array([], dtype=int)

    std_clean = np.nan_to_num(std_arr, nan=0.0, posinf=0.0, neginf=0.0)
    std_clean = np.clip(std_clean, a_min=0.0, a_max=None)
    denom = np.maximum(std_clean, 1e-8)

    if maximize:
        delta = mean_arr - float(best_so_far)
    else:
        delta = float(best_so_far) - mean_arr

    z = np.nan_to_num(delta / denom, nan=0.0, posinf=0.0, neginf=0.0)
    cdf = _standard_normal_cdf(z)
    pdf = _standard_normal_pdf(z)
    ei = denom * (z * cdf + pdf)
    zero_var = std_clean <= 1e-12
    if np.any(zero_var):
        ei[zero_var] = np.maximum(delta[zero_var], 0.0)
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

    k = min(k, ei.shape[0])
    indices = np.argsort(ei, kind="mergesort")[-k:][::-1]
    return indices.astype(int, copy=False)


def acquisition_random(
    n_pool: int,
    batch_size: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Random baseline acquisition."""
    pool_size = _coerce_non_negative_int(n_pool, name="n_pool")
    k = _validate_batch_size(batch_size)
    if pool_size == 0 or k == 0:
        return np.array([], dtype=int)
    seed_value = _coerce_non_negative_int(seed, name="seed")
    rng = np.random.default_rng(seed_value)
    return rng.choice(pool_size, size=min(k, pool_size), replace=False).astype(int, copy=False)


# ─────────────────────────────────────────────────────────
#  Active Learning Loop
# ─────────────────────────────────────────────────────────

class ActiveLearningLoop:
    """
    Uncertainty-driven active learning for materials property prediction.

    Iteratively:
    1. Train model on labeled data
    2. Predict + uncertainty on unlabeled pool
    3. Select most uncertain samples → oracle (DFT / MACE)
    4. Add to training set → retrain

    Args:
        model_class: GNN model constructor
        model_kwargs: model hyperparameters
        initial_budget: initial training set size
        query_budget: samples to query per iteration
        n_iterations: number of AL iterations
        strategy: "uncertainty", "random", or "expected_improvement"
    """

    def __init__(
        self,
        model_class: type,
        model_kwargs: dict,
        initial_budget: int = 1000,
        query_budget: int = 100,
        n_iterations: int = 10,
        strategy: str = "uncertainty",
        property_name: str = "formation_energy",
    ):
        initial_budget_i = _coerce_positive_int(initial_budget, name="initial_budget")
        query_budget_i = _coerce_positive_int(query_budget, name="query_budget")
        n_iterations_i = _coerce_positive_int(n_iterations, name="n_iterations")
        strategy_norm = str(strategy).strip().lower()
        if strategy_norm not in SUPPORTED_AL_STRATEGIES:
            raise ValueError(f"Unsupported strategy '{strategy}'. Supported: {sorted(SUPPORTED_AL_STRATEGIES)}")

        self.model_class = model_class
        self.model_kwargs = dict(model_kwargs or {})
        self.initial_budget = initial_budget_i
        self.query_budget = query_budget_i
        self.n_iterations = n_iterations_i
        self.strategy = strategy_norm
        self.property_name = property_name

        self.history = self._empty_history()

    @staticmethod
    def _empty_history() -> dict[str, list]:
        return {
            "iteration": [],
            "n_train": [],
            "val_mae": [],
            "test_mae": [],
            "selected_indices": [],
        }

    def _select_prediction_tensor(self, prediction):
        if isinstance(prediction, dict):
            if self.property_name in prediction:
                return prediction[self.property_name]
            if len(prediction) == 1:
                return next(iter(prediction.values()))
            raise KeyError(
                f"prediction dict missing '{self.property_name}' "
                f"and contains multiple outputs: {sorted(prediction)}"
            )
        return prediction

    def run(self, full_dataset, val_loader, test_loader, device="cuda"):
        """
        Execute the active learning loop.

        Args:
            full_dataset: complete dataset (labeled + unlabeled pool)
            val_loader: validation DataLoader
            test_loader: test DataLoader
            device: CUDA device

        Returns:
            history dict with results at each iteration
        """
        self.history = self._empty_history()

        n_total = len(full_dataset)
        if n_total == 0:
            return self.history

        all_indices = np.arange(n_total)
        rng = np.random.default_rng(42)

        # Initial random selection
        initial = min(self.initial_budget, n_total)
        labeled = set(rng.choice(n_total, size=initial, replace=False).tolist())
        pool = set(all_indices.tolist()) - labeled

        print(f"\n  Active Learning: {self.strategy}")
        print(f"  Initial: {len(labeled)}, Query: {self.query_budget}/iter, "
              f"Iterations: {self.n_iterations}")
        print("-" * 60)

        for iteration in range(self.n_iterations):
            # Create train subset
            train_indices = sorted(labeled)
            train_subset = torch.utils.data.Subset(full_dataset, train_indices)
            from torch_geometric.loader import DataLoader as PyGLoader
            train_loader = PyGLoader(train_subset, batch_size=64, shuffle=True)

            # Train model
            model = self.model_class(**self.model_kwargs).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

            model.train()
            for _epoch in range(50):  # Quick training per iteration
                for batch in train_loader:
                    batch = batch.to(device)
                    if not hasattr(batch, self.property_name):
                        continue
                    optimizer.zero_grad()
                    pred = self._select_prediction_tensor(
                        _forward_graph_model(model, batch)
                    )
                    target = getattr(batch, self.property_name).view(-1, 1)
                    loss = torch.nn.functional.huber_loss(pred, target)
                    if torch.isfinite(loss).all():
                        loss.backward()
                        optimizer.step()

            # Evaluate
            val_metrics = self._evaluate(model, val_loader, device)
            test_metrics = self._evaluate(model, test_loader, device)
            val_mae = val_metrics.get(f"{self.property_name}_MAE", float("inf"))
            test_mae = test_metrics.get(f"{self.property_name}_MAE", float("inf"))

            print(f"  Iter {iteration+1:3d} | "
                  f"n_train={len(labeled):5d} | "
                  f"val_MAE={val_mae:.4f} | "
                  f"test_MAE={test_mae:.4f}")

            self.history["iteration"].append(iteration)
            self.history["n_train"].append(len(labeled))
            self.history["val_mae"].append(val_mae)
            self.history["test_mae"].append(test_mae)

            # Select next batch from pool
            if len(pool) == 0:
                print("  Pool exhausted.")
                break

            pool_indices = sorted(pool)
            pool_subset = torch.utils.data.Subset(full_dataset, pool_indices)
            pool_loader = PyGLoader(pool_subset, batch_size=64, shuffle=False)

            if self.strategy == "uncertainty":
                means, stds = self._predict_with_dropout(model, pool_loader, device)
                selected = acquisition_uncertainty(means, stds, self.query_budget)
            elif self.strategy == "expected_improvement":
                means, stds = self._predict_with_dropout(model, pool_loader, device)
                selected = acquisition_expected_improvement(
                    means, stds, best_so_far=val_mae, batch_size=self.query_budget,
                )
            else:
                selected = acquisition_random(len(pool_indices), self.query_budget, seed=iteration)

            # Add selected to labeled
            new_indices = [pool_indices[i] for i in selected if i < len(pool_indices)]
            labeled.update(new_indices)
            pool -= set(new_indices)
            self.history["selected_indices"].append(new_indices)

        return self.history

    @torch.no_grad()
    def _evaluate(self, model, loader, device):
        model.eval()
        preds, targets = [], []
        for batch in loader:
            batch = batch.to(device)
            if not hasattr(batch, self.property_name):
                continue
            pred = self._select_prediction_tensor(
                _forward_graph_model(model, batch)
            )
            preds.append(pred.cpu())
            targets.append(getattr(batch, self.property_name).view(-1, 1).cpu())
        if not preds:
            return {}
        return scalar_metrics(torch.cat(preds), torch.cat(targets), prefix=self.property_name)

    def _predict_with_dropout(self, model, loader, device, n_samples=10):
        """MC-Dropout predictions for uncertainty."""
        n_mc = _coerce_positive_int(n_samples, name="n_samples")
        initial_mode = model.training
        dropout_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        dropout_states = [m.training for m in dropout_layers]

        model.eval()
        for layer in dropout_layers:
            layer.train()

        all_means, all_stds = [], []
        try:
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    preds = []
                    for _ in range(n_mc):
                        pred = self._select_prediction_tensor(
                            _forward_graph_model(model, batch)
                        )
                        preds.append(pred)
                    preds = torch.stack(preds)
                    all_means.append(preds.mean(0).cpu().numpy())
                    all_stds.append(preds.std(0, unbiased=False).cpu().numpy())
        finally:
            model.train(initial_mode)
            for layer, state in zip(dropout_layers, dropout_states):
                layer.train(state)

        if not all_means:
            return np.array([], dtype=float), np.array([], dtype=float)
        means = np.concatenate(all_means).squeeze()
        stds = np.concatenate(all_stds).squeeze()
        means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        stds = np.nan_to_num(stds, nan=0.0, posinf=0.0, neginf=0.0)
        return means, stds


# ─────────────────────────────────────────────────────────
#  Multi-Objective Pareto Screening
# ─────────────────────────────────────────────────────────

def pareto_frontier(
    objectives: np.ndarray,
    maximize: list[bool] | None = None,
) -> np.ndarray:
    """
    Find Pareto-optimal solutions.

    Args:
        objectives: (N, n_objectives) matrix
        maximize: list of bools (True=maximize, False=minimize)

    Returns:
        (K,) indices of Pareto-optimal points
    """
    obj_raw = np.asarray(objectives, dtype=float)
    if obj_raw.ndim != 2:
        raise ValueError("objectives must be a 2D array")
    n_rows, n_obj = obj_raw.shape
    if n_rows == 0:
        return np.array([], dtype=int)

    if maximize is None:
        maximize = [True] * n_obj
    elif len(maximize) != n_obj:
        raise ValueError("maximize length must match number of objectives")

    valid_mask = np.isfinite(obj_raw).all(axis=1)
    if not np.any(valid_mask):
        return np.array([], dtype=int)
    valid_indices = np.where(valid_mask)[0]
    obj = obj_raw[valid_mask].copy()
    n = obj.shape[0]

    # Flip sign for minimization objectives
    for i, mx in enumerate(maximize):
        if not mx:
            obj[:, i] = -obj[:, i]

    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i if j >= i in all objectives and j > i in at least one
            if np.all(obj[j] >= obj[i]) and np.any(obj[j] > obj[i]):
                is_pareto[i] = False
                break

    return valid_indices[np.where(is_pareto)[0]].astype(int, copy=False)


@torch.no_grad()
def multi_objective_screening(
    model,
    loader,
    objectives: dict[str, str],  # {"property": "high"/"low"}
    device: str = "cuda",
    top_k: int = 50,
):
    """
    Screen materials for multi-objective optimization.

    Args:
        model: trained multi-task model
        loader: DataLoader of candidate materials
        objectives: dict mapping property → "high" or "low"
        device: computation device
        top_k: number of top candidates to return

    Returns:
        Dict with Pareto-optimal candidates and their scores
    """
    if not objectives:
        raise ValueError("objectives must not be empty")
    top_k_i = _coerce_positive_int(top_k, name="top_k")

    print(f"\n  Multi-Objective Screening: {objectives}")

    prop_names = list(objectives.keys())
    maximize = [objectives[p] == "high" for p in prop_names]

    all_preds = {p: [] for p in prop_names}
    for batch in loader:
        batch = batch.to(device)

        predictions = _forward_graph_model(model, batch)

        if isinstance(predictions, dict):
            for p in prop_names:
                if p in predictions:
                    all_preds[p].append(predictions[p].cpu().numpy())
        elif len(prop_names) == 1 and isinstance(predictions, torch.Tensor):
            all_preds[prop_names[0]].append(predictions.cpu().numpy())

    # Stack predictions
    if any(len(all_preds[p]) == 0 for p in prop_names):
        return {
            "pareto_indices": [],
            "top_k_indices": [],
            "objectives": objectives,
            "scores": {p: [] for p in prop_names},
        }

    per_objective = []
    for p in prop_names:
        values = np.concatenate(all_preds[p]).reshape(-1)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        per_objective.append(values)
    n_rows = min((arr.size for arr in per_objective), default=0)
    if n_rows == 0:
        return {
            "pareto_indices": [],
            "top_k_indices": [],
            "objectives": objectives,
            "scores": {p: [] for p in prop_names},
        }
    obj_matrix = np.column_stack([arr[:n_rows] for arr in per_objective])

    # Find Pareto front
    pareto_idx = pareto_frontier(obj_matrix, maximize=maximize)
    print(f"  Pareto-optimal materials: {len(pareto_idx)}")

    # Rank by a combined score (normalized sum)
    spans = np.ptp(obj_matrix, axis=0)
    denom = np.where(spans <= 0.0, 1.0, spans)
    normed = (obj_matrix - obj_matrix.min(0)) / denom
    for i, mx in enumerate(maximize):
        if not mx:
            normed[:, i] = 1 - normed[:, i]

    combined_score = normed.sum(axis=1)
    k = min(top_k_i, obj_matrix.shape[0])
    top_indices = np.argsort(combined_score)[-k:][::-1]

    results = {
        "pareto_indices": pareto_idx.tolist(),
        "top_k_indices": top_indices.tolist(),
        "objectives": objectives,
        "scores": {
            p: obj_matrix[:, i].tolist() for i, p in enumerate(prop_names)
        },
    }

    print(f"\n  Top {top_k} candidates (combined score):")
    print(f"  {'Rank':>6s} {'Index':>8s}", end="")
    for p in prop_names:
        print(f" {p[:12]:>14s}", end="")
    print(f" {'Score':>8s}")
    for rank, idx in enumerate(top_indices[:10]):
        print(f"  {rank+1:>6d} {idx:>8d}", end="")
        for i, p in enumerate(prop_names):
            print(f" {obj_matrix[idx, i]:>14.3f}", end="")
        print(f" {combined_score[idx]:>8.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Active Learning & Screening")
    parser.add_argument("--strategy", type=str, default="uncertainty",
                        choices=["uncertainty", "random", "expected_improvement"])
    parser.add_argument("--budget", type=int, default=500,
                        help="Total query budget")
    parser.add_argument("--initial", type=int, default=1000,
                        help="Initial labeled set size")
    parser.add_argument("--query-size", type=int, default=100,
                        help="Samples per AL iteration")
    parser.add_argument("--pareto", action="store_true",
                        help="Run Pareto screening instead of AL")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print(f"║ {'Phase 6: Active Learning & Multi-Objective Screening'.center(64)} ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    config = get_config()
    save_dir = config.paths.models_dir / "active_learning"
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.pareto:
        print("\n  Pareto screening mode.")
        print("  Please load your trained multi-task model and run:")
        print("    results = multi_objective_screening(model, loader, objectives)")
    else:
        print(f"\n  Active Learning mode: {args.strategy}")
        print(f"  Initial: {args.initial}, Query: {args.query_size}/iter")
        print(f"  Total budget: {args.budget}")
        print("\n  Please integrate with your trained model class.")
        print("  Example:")
        print("    al = ActiveLearningLoop(CGCNN, model_kwargs, ...)")
        print("    history = al.run(full_dataset, val_loader, test_loader)")

    print(f"\n  Save directory: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
