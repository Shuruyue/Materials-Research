"""Tests for atlas.benchmark runner/reporting utilities."""

from __future__ import annotations

import numpy as np
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
    compute_uncertainty_metrics,
)
from atlas.benchmark import runner as runner_mod


class _DummyModel(nn.Module):
    def forward(self, x, edge_index, edge_attr, batch):  # noqa: ARG002
        n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        out = torch.zeros((n_graphs, 1), dtype=x.dtype, device=x.device)
        for idx in range(n_graphs):
            out[idx, 0] = x[batch == idx, 0].mean()
        return out


class _DummyUncertainModel(nn.Module):
    def forward(self, x, edge_index, edge_attr, batch):  # noqa: ARG002
        n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        out = torch.zeros((n_graphs, 1), dtype=x.dtype, device=x.device)
        for idx in range(n_graphs):
            out[idx, 0] = x[batch == idx, 0].mean()
        std = torch.full_like(out, 0.1)
        return {"formation_energy": {"mean": out, "std": std}}


class _BatchOnlyModel(nn.Module):
    def forward(self, batch):
        x = batch.x
        b = batch.batch
        n_graphs = int(b.max().item()) + 1 if b.numel() > 0 else 1
        out = torch.zeros((n_graphs, 1), dtype=x.dtype, device=x.device)
        for idx in range(n_graphs):
            out[idx, 0] = x[b == idx, 0].mean()
        return out


class _FakeTask:
    folds = [0]

    def get_test_data(self, fold, include_target=True):  # noqa: ARG002
        return [1.0, 2.0, "bad"], pd.Series([1.0, 2.0, 3.0])

    def get_train_and_val_data(self, fold, include_target=True):  # noqa: ARG002
        return [1.0, 2.0, 3.0], pd.Series([1.0, 2.0, 3.0])


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
    assert metrics["rmse"] == pytest.approx(2.0**0.5)
    assert metrics["r2"] == pytest.approx(-1.0)


def test_compute_uncertainty_metrics():
    target = torch.tensor([1.0, 2.0, 3.0]).numpy()
    pred = torch.tensor([1.1, 1.9, 3.2]).numpy()
    std = torch.tensor([0.2, 0.2, 0.2]).numpy()
    metrics = compute_uncertainty_metrics(target, pred, std)
    assert metrics["avg_std"] == pytest.approx(0.2)
    assert 0.0 <= metrics["pi95_coverage"] <= 1.0
    assert np.isfinite(metrics["gaussian_nll"])
    assert np.isfinite(metrics["regression_ece"])
    assert np.isfinite(metrics["interval_score_95"])
    assert np.isfinite(metrics["crps_gaussian"])


def test_bootstrap_ci_sanitizes_invalid_parameters():
    lo, hi = runner_mod._bootstrap_ci(  # noqa: SLF001
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        confidence=float("nan"),
        n_bootstrap=-5,
        seed=float("nan"),
    )
    assert np.isfinite(lo)
    assert np.isfinite(hi)
    assert hi >= lo


def test_aggregate_fold_results():
    folds = [
        FoldResult(fold=0, n_test=10, n_valid=9, coverage=0.9, mae=0.2, rmse=0.3, r2=0.8),
        FoldResult(fold=1, n_test=12, n_valid=12, coverage=1.0, mae=0.4, rmse=0.5, r2=0.6),
    ]
    agg = aggregate_fold_results(folds)
    assert agg["mae_mean"] == pytest.approx(0.3)
    assert agg["mae_weighted_mean"] == pytest.approx((0.2 * 9.0 + 0.4 * 12.0) / (9.0 + 12.0))
    assert agg["coverage_mean"] == pytest.approx(0.95)
    assert agg["coverage_weighted_mean"] == pytest.approx((0.9 * 10.0 + 1.0 * 12.0) / (10.0 + 12.0))
    assert agg["total_test_samples"] == pytest.approx(22.0)
    assert agg["n_successful_folds"] == pytest.approx(2.0)
    assert agg["n_low_coverage_folds"] == pytest.approx(0.0)


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
    assert report["report_path"] == str(out_path)
    assert report["aggregate_metrics"]["coverage_mean"] == pytest.approx(2.0 / 3.0)
    assert report["aggregate_metrics"]["global_mae"] == pytest.approx(0.0)
    assert np.isfinite(report["aggregate_metrics"]["global_mae_ci95_lo"])
    assert np.isfinite(report["aggregate_metrics"]["global_mae_ci95_hi"])
    assert np.isfinite(report["aggregate_metrics"]["global_rmse_ci95_lo"])
    assert np.isfinite(report["aggregate_metrics"]["global_rmse_ci95_hi"])
    assert np.isfinite(report["aggregate_metrics"]["global_r2_ci95_lo"])
    assert np.isfinite(report["aggregate_metrics"]["global_r2_ci95_hi"])
    assert report["fold_results"][0]["status"] == "ok"


def test_runner_init_sanitizes_invalid_runtime_parameters(tmp_path):
    runner = MatbenchRunner(
        model=_DummyModel(),
        property_name="formation_energy",
        device="cpu",
        batch_size=0,
        n_jobs=0,
        output_dir=tmp_path,
        bootstrap_samples=float("nan"),
        bootstrap_seed=float("nan"),
        min_coverage_required=float("nan"),
        conformal_coverage=float("nan"),
        conformal_max_calibration_samples=-9,
    )
    assert runner.batch_size == 1
    assert runner.n_jobs == -1
    assert runner.bootstrap_samples >= 64
    assert runner.bootstrap_seed == 42
    assert runner.min_coverage_required == pytest.approx(0.0)
    assert 0.0 < runner.conformal_coverage < 1.0
    assert runner.conformal_max_calibration_samples == 0


def test_runner_init_avoids_silent_truncation_for_non_integral_inputs(tmp_path):
    runner = MatbenchRunner(
        model=_DummyModel(),
        property_name="formation_energy",
        device="cpu",
        batch_size=3.7,
        n_jobs=True,
        output_dir=tmp_path,
        bootstrap_samples=99.9,
        bootstrap_seed=7.5,
    )
    assert runner.batch_size == 32
    assert runner.n_jobs == -1
    assert runner.bootstrap_samples == 400
    assert runner.bootstrap_seed == 42


def test_runner_init_probability_controls_reject_bool_semantics(tmp_path):
    runner = MatbenchRunner(
        model=_DummyModel(),
        property_name="formation_energy",
        device="cpu",
        output_dir=tmp_path,
        min_coverage_required=True,
        conformal_coverage=True,
    )
    assert runner.min_coverage_required == pytest.approx(0.0)
    assert runner.conformal_coverage == pytest.approx(0.95)


def test_runner_uncertainty_outputs_are_reported(tmp_path):
    runner = MatbenchRunner(
        model=_DummyUncertainModel(),
        property_name="formation_energy",
        device="cpu",
        batch_size=2,
        n_jobs=1,
        output_dir=tmp_path,
    )
    runner.load_task = lambda _: _FakeTask()
    runner.structure_to_data = lambda structure: None if structure == "bad" else _fake_data(structure)

    report = runner.run_task(
        task_name="fake_task_unc",
        folds=[0],
        show_progress=False,
    )
    agg = report["aggregate_metrics"]
    assert agg["pi95_coverage_mean"] == pytest.approx(1.0)
    assert np.isfinite(agg["gaussian_nll_mean"])
    assert agg["global_pi95_coverage"] == pytest.approx(1.0)
    assert np.isfinite(agg["global_crps_gaussian"])
    assert np.isfinite(agg["global_pi95_coverage_ci95_lo"])
    assert np.isfinite(agg["global_pi95_coverage_ci95_hi"])
    assert np.isfinite(agg["global_gaussian_nll_ci95_lo"])
    assert np.isfinite(agg["global_gaussian_nll_ci95_hi"])
    assert np.isfinite(agg["global_crps_gaussian_ci95_lo"])
    assert np.isfinite(agg["global_crps_gaussian_ci95_hi"])


def test_runner_conformal_metrics_are_reported(tmp_path):
    runner = MatbenchRunner(
        model=_DummyModel(),
        property_name="formation_energy",
        device="cpu",
        batch_size=2,
        n_jobs=1,
        output_dir=tmp_path,
        use_conformal=True,
        conformal_coverage=0.95,
    )
    runner.load_task = lambda _: _FakeTask()
    runner.structure_to_data = lambda structure: None if structure == "bad" else _fake_data(structure)

    report = runner.run_task(
        task_name="fake_task_conformal",
        folds=[0],
        show_progress=False,
    )
    agg = report["aggregate_metrics"]
    assert np.isfinite(agg["conformal_qhat_mean"])
    assert np.isfinite(agg["conformal_pi_coverage_mean"])
    assert np.isfinite(agg["global_conformal_pi_coverage"])
    settings = report["settings"]
    assert settings["structure_cache"] is True
    assert int(settings["structure_cache_hits"]) >= 1
    assert "model_call_variant_counts" in settings


def test_runner_marks_low_coverage_when_below_threshold(tmp_path):
    runner = MatbenchRunner(
        model=_DummyModel(),
        property_name="formation_energy",
        device="cpu",
        batch_size=2,
        n_jobs=1,
        output_dir=tmp_path,
        min_coverage_required=0.9,
    )
    runner.load_task = lambda _: _FakeTask()
    runner.structure_to_data = lambda structure: None if structure == "bad" else _fake_data(structure)

    report = runner.run_task(
        task_name="fake_task_gate",
        folds=[0],
        show_progress=False,
    )
    assert report["fold_results"][0]["status"] == "low_coverage"
    assert report["aggregate_metrics"]["n_low_coverage_folds"] == pytest.approx(1.0)
    assert report["aggregate_metrics"]["success_rate"] == pytest.approx(0.0)


def test_runner_fail_on_fallback_signature(tmp_path):
    runner = MatbenchRunner(
        model=_BatchOnlyModel(),
        property_name="formation_energy",
        device="cpu",
        batch_size=2,
        n_jobs=1,
        output_dir=tmp_path,
        fail_on_fallback_signature=True,
    )
    runner.load_task = lambda _: _FakeTask()
    runner.structure_to_data = lambda structure: None if structure == "bad" else _fake_data(structure)

    report = runner.run_task(
        task_name="fake_task_signature",
        folds=[0],
        show_progress=False,
    )
    assert report["fold_results"][0]["status"] == "failed"
    assert "fallback signature" in str(report["fold_results"][0]["error"]).lower()


def test_runner_run_task_rejects_non_integral_fold_indices(tmp_path):
    runner = MatbenchRunner(
        model=_DummyModel(),
        property_name="formation_energy",
        device="cpu",
        batch_size=2,
        n_jobs=1,
        output_dir=tmp_path,
    )
    runner.load_task = lambda _: _FakeTask()
    with pytest.raises(ValueError, match="non-negative integers"):
        runner.run_task(
            task_name="fake_task_invalid_fold",
            folds=[0.5],
            show_progress=False,
        )


def test_runner_run_task_rejects_negative_fold_indices(tmp_path):
    runner = MatbenchRunner(
        model=_DummyModel(),
        property_name="formation_energy",
        device="cpu",
        batch_size=2,
        n_jobs=1,
        output_dir=tmp_path,
    )
    runner.load_task = lambda _: _FakeTask()
    with pytest.raises(ValueError, match="non-negative integers"):
        runner.run_task(
            task_name="fake_task_negative_fold",
            folds=[-1],
            show_progress=False,
        )


def test_benchmark_cli_list_tasks(capsys):
    try:
        from atlas.benchmark.cli import main as benchmark_cli_main
    except Exception as exc:  # pragma: no cover - environment-dependent optional import path
        pytest.skip(f"benchmark CLI import unavailable in this env: {exc}")
    rc = benchmark_cli_main(["--list-tasks"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "matbench_mp_gap" in out
