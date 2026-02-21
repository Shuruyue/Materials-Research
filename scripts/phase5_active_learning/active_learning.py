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
import torch
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from atlas.config import get_config
from atlas.data.crystal_dataset import CrystalPropertyDataset, DEFAULT_PROPERTIES
from atlas.training.metrics import scalar_metrics


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
    # Use total uncertainty (sum over output dims if multi-dim)
    if std.ndim > 1:
        total_unc = std.sum(axis=-1)
    else:
        total_unc = std

    indices = np.argsort(total_unc)[-batch_size:][::-1]
    return indices


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
    from scipy.stats import norm

    if maximize:
        z = (mean - best_so_far) / (std + 1e-8)
    else:
        z = (best_so_far - mean) / (std + 1e-8)

    ei = std * (z * norm.cdf(z) + norm.pdf(z))
    indices = np.argsort(ei)[-batch_size:][::-1]
    return indices


def acquisition_random(
    n_pool: int,
    batch_size: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Random baseline acquisition."""
    rng = np.random.RandomState(seed)
    return rng.choice(n_pool, size=min(batch_size, n_pool), replace=False)


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
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.initial_budget = initial_budget
        self.query_budget = query_budget
        self.n_iterations = n_iterations
        self.strategy = strategy
        self.property_name = property_name

        self.history = {
            "iteration": [],
            "n_train": [],
            "val_mae": [],
            "test_mae": [],
            "selected_indices": [],
        }

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
        n_total = len(full_dataset)
        all_indices = np.arange(n_total)
        rng = np.random.RandomState(42)

        # Initial random selection
        labeled = set(rng.choice(n_total, size=self.initial_budget, replace=False).tolist())
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
            for epoch in range(50):  # Quick training per iteration
                for batch in train_loader:
                    batch = batch.to(device)
                    if not hasattr(batch, self.property_name):
                        continue
                    optimizer.zero_grad()
                    pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    target = getattr(batch, self.property_name).view(-1, 1)
                    loss = torch.nn.functional.huber_loss(pred, target)
                    if not torch.isnan(loss):
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
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds.append(pred.cpu())
            targets.append(getattr(batch, self.property_name).view(-1, 1).cpu())
        if not preds:
            return {}
        return scalar_metrics(torch.cat(preds), torch.cat(targets), prefix=self.property_name)

    def _predict_with_dropout(self, model, loader, device, n_samples=10):
        """MC-Dropout predictions for uncertainty."""
        model.eval()
        # Enable dropout
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

        all_means, all_stds = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                preds = []
                for _ in range(n_samples):
                    pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    preds.append(pred)
                preds = torch.stack(preds)
                all_means.append(preds.mean(0).cpu().numpy())
                all_stds.append(preds.std(0).cpu().numpy())

        return np.concatenate(all_means).squeeze(), np.concatenate(all_stds).squeeze()


# ─────────────────────────────────────────────────────────
#  Multi-Objective Pareto Screening
# ─────────────────────────────────────────────────────────

def pareto_frontier(
    objectives: np.ndarray,
    maximize: Optional[List[bool]] = None,
) -> np.ndarray:
    """
    Find Pareto-optimal solutions.

    Args:
        objectives: (N, n_objectives) matrix
        maximize: list of bools (True=maximize, False=minimize)

    Returns:
        (K,) indices of Pareto-optimal points
    """
    n = objectives.shape[0]
    if maximize is None:
        maximize = [True] * objectives.shape[1]

    # Flip sign for minimization objectives
    obj = objectives.copy()
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

    return np.where(is_pareto)[0]


@torch.no_grad()
def multi_objective_screening(
    model,
    loader,
    objectives: Dict[str, str],  # {"property": "high"/"low"}
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
    print(f"\n  Multi-Objective Screening: {objectives}")

    prop_names = list(objectives.keys())
    maximize = [objectives[p] == "high" for p in prop_names]

    all_preds = {p: [] for p in prop_names}
    for batch in loader:
        batch = batch.to(device)

        if hasattr(batch, "edge_vec") and hasattr(model, "irreps_hidden"):
            edge_feats = batch.edge_vec
        else:
            edge_feats = batch.edge_attr

        predictions = model(batch.x, batch.edge_index, edge_feats, batch.batch)

        for p in prop_names:
            if p in predictions:
                all_preds[p].append(predictions[p].cpu().numpy())

    # Stack predictions
    obj_matrix = np.column_stack([
        np.concatenate(all_preds[p]).squeeze() for p in prop_names
    ])

    # Find Pareto front
    pareto_idx = pareto_frontier(obj_matrix, maximize=maximize)
    print(f"  Pareto-optimal materials: {len(pareto_idx)}")

    # Rank by a combined score (normalized sum)
    normed = (obj_matrix - obj_matrix.min(0)) / (obj_matrix.ptp(0) + 1e-8)
    for i, mx in enumerate(maximize):
        if not mx:
            normed[:, i] = 1 - normed[:, i]

    combined_score = normed.sum(axis=1)
    top_indices = np.argsort(combined_score)[-top_k:][::-1]

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


if __name__ == "__main__":
    main()
