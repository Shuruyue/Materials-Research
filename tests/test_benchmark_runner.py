"""Tests for atlas.benchmark runner/reporting utilities."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from atlas.benchmark import (
    FoldResult,
    MatbenchRunner,
    aggregate_fold_results,
    compute_regression_metrics,
)
from atlas.benchmark.cli import main as benchmark_cli_main


class _DummyModel(nn.Module):
    def forward(self, x, edge_index, edge_attr, batch):  # noqa: ARG002
        n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        out = torch.zeros((n_graphs, 1), dtype=x.dtype, device=x.device)
        for idx in range(n_graphs):
            out[idx, 0] = x[batch == idx, 0].mean()
        return out


class _FakeTask:
    folds = [0]

    def get_test_data(self, fold, include_target=True):  # noqa: ARG002
        return [1.0, 2.0, "bad"], pd.Series([1.0, 2.0, 3.0])


def _fake_data(v: float) -> Data:
    return Data(
        x=torch.tensor([[float(v)]], dtype=torch.float32),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        edge_attr=torch.tensor([[0.0]], dtype=torch.float32),
        edge_vec=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        num_nodes=1,
    )


def test_compute_regression_metrics_ignores_nan_pairs():
    target = torch.tensor([1.0, 2.0, 3.0]).numpy()
    pred = torch.tensor([1.0, float("nan"), 5.0]).numpy()
    metrics = compute_regression_metrics(target, pred)
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx((2.0**0.5))
    assert metrics["r2"] == pytest.approx(-1.0)


def test_aggregate_fold_results():
    folds = [
        FoldResult(fold=0, n_test=10, n_valid=9, coverage=0.9, mae=0.2, rmse=0.3, r2=0.8),
        FoldResult(fold=1, n_test=12, n_valid=12, coverage=1.0, mae=0.4, rmse=0.5, r2=0.6),
    ]
    agg = aggregate_fold_results(folds)
    assert agg["mae_mean"] == pytest.approx(0.3)
    assert agg["coverage_mean"] == pytest.approx(0.95)
    assert agg["total_test_samples"] == pytest.approx(22.0)
    assert agg["n_successful_folds"] == pytest.approx(2.0)


def test_runner_run_task_writes_report(tmp_path):
    runner = MatbenchRunner(
        model=_DummyModel(),
        property_name="formation_energy",
        device="cpu",
        batch_size=2,
        n_jobs=1,
        output_dir=tmp_path,
    )
    runner.load_task = lambda _: _FakeTask()
    runner.structure_to_data = lambda structure: None if structure == "bad" else _fake_data(structure)

    out_path = tmp_path / "fake_task_report.json"
    report = runner.run_task(
        task_name="fake_task",
        folds=[0],
        save_path=out_path,
        show_progress=False,
    )

    assert out_path.exists()
    assert report["task_name"] == "fake_task"
    assert report["aggregate_metrics"]["coverage_mean"] == pytest.approx(2.0 / 3.0)
    assert report["aggregate_metrics"]["global_mae"] == pytest.approx(0.0)
    assert report["fold_results"][0]["status"] == "ok"


def test_benchmark_cli_list_tasks(capsys):
    rc = benchmark_cli_main(["--list-tasks"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "matbench_mp_gap" in out
