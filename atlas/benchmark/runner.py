"""
Matbench benchmark runner with reproducible fold-level reports.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
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


def _to_numeric_array(values: Sequence[Any]) -> np.ndarray:
    """Convert labels to a numeric 1D array, coercing invalid entries to NaN."""
    series = pd.Series(list(values))
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


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
    runtime_sec: float = 0.0
    status: str = "ok"
    error: str = ""


def aggregate_fold_results(fold_results: Sequence[FoldResult]) -> dict[str, float]:
    """Aggregate fold-level results into mean/std summaries."""
    aggregate: dict[str, float] = {
        "n_folds": len(fold_results),
        "n_successful_folds": sum(r.status == "ok" for r in fold_results),
        "total_test_samples": sum(r.n_test for r in fold_results),
        "total_valid_samples": sum(r.n_valid for r in fold_results),
    }
    for key in ("mae", "rmse", "r2", "coverage"):
        vals = np.array(
            [float(getattr(r, key)) for r in fold_results if np.isfinite(getattr(r, key))],
            dtype=np.float64,
        )
        aggregate[f"{key}_mean"] = float(np.mean(vals)) if vals.size else np.nan
        aggregate[f"{key}_std"] = float(np.std(vals)) if vals.size else np.nan
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

    def __init__(
        self,
        model,
        property_name: str,
        device: str = "auto",
        batch_size: int = 32,
        n_jobs: int = -1,
        output_dir: Path | None = None,
    ):
        self.model = model
        self.property_name = property_name
        self.cfg = get_config()
        self.graph_builder = CrystalGraphBuilder()
        self.batch_size = batch_size
        self.n_jobs = n_jobs

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.output_dir = output_dir or (self.cfg.paths.artifacts_dir / "benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def _predict_batch(self, batch) -> np.ndarray:
        """Run model forward with resilient call signatures and output extraction."""
        call_variants = (
            lambda: self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch),
            lambda: self.model(batch.x, batch.edge_index, batch.edge_vec, batch.batch),
            lambda: self.model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_vec, batch.batch),
            lambda: self.model(batch),
        )

        output = None
        last_error: Exception | None = None
        for call in call_variants:
            try:
                output = call()
                break
            except Exception as exc:  # pragma: no cover - only used for fallback probing
                last_error = exc
                continue

        if output is None:
            raise RuntimeError(f"Model forward failed for all call variants: {last_error}")

        if isinstance(output, dict):
            if self.property_name in output:
                output = output[self.property_name]
            elif len(output) == 1:
                output = next(iter(output.values()))
            else:
                keys = ", ".join(sorted(output.keys()))
                raise KeyError(f"Property '{self.property_name}' not found in model output keys: {keys}")

        if isinstance(output, dict):
            if "gamma" in output:
                output = output["gamma"]
            elif "mu" in output:
                output = output["mu"]
            else:
                keys = ", ".join(sorted(output.keys()))
                raise KeyError(f"Unsupported nested output keys: {keys}")

        if torch.is_tensor(output):
            return output.detach().cpu().reshape(-1).numpy().astype(np.float64)
        return np.asarray(output, dtype=np.float64).reshape(-1)

    def _convert_structures(self, structures: Sequence[Any], show_progress: bool) -> list[Any]:
        iterable: Iterable[Any] = tqdm(structures, leave=False, disable=not show_progress)
        if self.n_jobs == 1:
            return [self.structure_to_data(s) for s in iterable]
        return Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(self.structure_to_data)(s) for s in iterable
        )

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
        predictions = np.full(n_test, np.nan, dtype=np.float64)

        converted = self._convert_structures(inputs, show_progress=show_progress)
        data_list = []
        valid_indices: list[int] = []
        for idx, data in enumerate(converted):
            if data is not None:
                data_list.append(data)
                valid_indices.append(idx)

        n_valid = len(valid_indices)
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
            return targets, predictions, result

        try:
            loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)
            cursor = 0
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    batch_pred = self._predict_batch(batch)

                    n_graphs = int(getattr(batch, "num_graphs", len(batch_pred)))
                    if batch_pred.size == 1 and n_graphs > 1:
                        batch_pred = np.repeat(batch_pred, n_graphs)
                    if batch_pred.size != n_graphs and batch_pred.size == batch.batch.numel():
                        batch_indices = batch.batch.detach().cpu().numpy()
                        reduced = []
                        for g_idx in range(n_graphs):
                            node_mask = batch_indices == g_idx
                            reduced.append(float(np.mean(batch_pred[node_mask])))
                        batch_pred = np.asarray(reduced, dtype=np.float64)
                    if batch_pred.size != n_graphs:
                        raise ValueError(
                            f"Prediction size mismatch: got {batch_pred.size}, expected {n_graphs}"
                        )

                    idx_slice = valid_indices[cursor : cursor + n_graphs]
                    for global_idx, pred in zip(idx_slice, batch_pred):
                        predictions[global_idx] = float(pred)
                    cursor += n_graphs

            metrics = compute_regression_metrics(targets, predictions)
            result = FoldResult(
                fold=fold,
                n_test=n_test,
                n_valid=n_valid,
                coverage=coverage,
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                r2=metrics["r2"],
                runtime_sec=time.time() - t0,
            )
            return targets, predictions, result
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
            return targets, predictions, result

    def run_fold(self, task, fold: int):
        """
        Backward-compatible fold run API.

        Returns:
            tuple[np.ndarray, np.ndarray]: targets and predictions.
        """
        targets, preds, _ = self._run_fold_details(task, fold, show_progress=True)
        return targets, preds

    def _default_report_path(self, task_name: str) -> Path:
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        return self.output_dir / f"{task_name}_{self.property_name}_{ts}.json"

    def _write_report(self, report: TaskReport, path: Path) -> None:
        payload = asdict(report)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, default=float)

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
        fold_ids = list(folds) if folds is not None else list(getattr(task, "folds", []))

        fold_results: list[FoldResult] = []
        all_targets: list[np.ndarray] = []
        all_preds: list[np.ndarray] = []

        for fold in fold_ids:
            targets, preds, fold_result = self._run_fold_details(task, fold, show_progress=show_progress)
            fold_results.append(fold_result)
            all_targets.append(targets)
            all_preds.append(preds)

        aggregate = aggregate_fold_results(fold_results)
        if all_targets and all_preds:
            merged_targets = np.concatenate(all_targets)
            merged_preds = np.concatenate(all_preds)
            global_metrics = compute_regression_metrics(merged_targets, merged_preds)
            for key, value in global_metrics.items():
                aggregate[f"global_{key}"] = value

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
                "seed": self.cfg.train.seed,
                "deterministic": self.cfg.train.deterministic,
                "method_key": self.cfg.profile.method_key,
                "data_source_key": self.cfg.profile.data_source_key,
            },
        )

        report_path = save_path or self._default_report_path(task_name)
        self._write_report(report, report_path)
        report.report_path = str(report_path)
        return asdict(report)

