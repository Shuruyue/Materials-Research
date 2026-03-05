"""Unit tests for atlas.training.metrics."""

import pytest
import torch

import atlas.training.metrics as metrics_module
from atlas.training.metrics import (
    classification_metrics,
    eigenvalue_agreement,
    frobenius_error,
    mae,
    max_ae,
    r2_score,
    rmse,
    scalar_metrics,
    symmetry_violation,
    tensor_metrics,
)


def test_scalar_metrics_ignore_non_finite_pairs():
    pred = torch.tensor([1.0, float("nan"), float("inf")])
    target = torch.tensor([1.0, 2.0, 3.0])
    assert mae(pred, target) == pytest.approx(0.0)
    assert rmse(pred, target) == pytest.approx(0.0)
    assert max_ae(pred, target) == pytest.approx(0.0)
    assert r2_score(pred, target) == pytest.approx(0.0)


def test_scalar_metrics_prefix_keys():
    out = scalar_metrics(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), prefix=" val ")
    assert set(out) == {"val_MAE", "val_RMSE", "val_R2", "val_MaxAE"}


def test_classification_metrics_empty_returns_safe_defaults():
    out = classification_metrics(
        torch.tensor([float("nan"), float("inf")]),
        torch.tensor([0.0, 1.0]),
    )
    assert out["Accuracy"] == pytest.approx(0.0)
    assert out["Precision"] == pytest.approx(0.0)
    assert out["Recall"] == pytest.approx(0.0)
    assert out["F1"] == pytest.approx(0.0)
    assert out["AUC"] == pytest.approx(0.5)


def test_classification_metrics_single_class_auc_fallback():
    out = classification_metrics(
        torch.tensor([0.1, 0.2, 0.3]),
        torch.tensor([1.0, 1.0, 1.0]),
    )
    assert out["AUC"] == pytest.approx(0.5)
    for key in ("Accuracy", "Precision", "Recall", "F1"):
        assert 0.0 <= out[key] <= 1.0


def test_frobenius_error_skips_non_finite_rows():
    pred = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[float("inf"), 0.0], [0.0, 1.0]],
        ]
    )
    target = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    assert frobenius_error(pred, target) == pytest.approx(0.0)


def test_frobenius_error_accepts_single_square_matrix():
    pred = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert frobenius_error(pred, target) == pytest.approx(1.0)


def test_symmetry_violation_non_square_returns_zero():
    assert symmetry_violation(torch.ones(2, 3)) == pytest.approx(0.0)


def test_eigenvalue_agreement_skips_non_finite_samples():
    pred = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 1.0]],
            [[float("nan"), 0.0], [0.0, 1.0]],
        ]
    )
    target = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 1.0]],
        ]
    )
    assert eigenvalue_agreement(pred, target) == pytest.approx(1.0)


def test_eigenvalue_agreement_fallback_without_scipy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(metrics_module, "_scipy_spearmanr", None)
    pred = torch.tensor([[[3.0, 0.0], [0.0, 1.0]]])
    target = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]])
    assert eigenvalue_agreement(pred, target) == pytest.approx(1.0)


def test_tensor_metrics_prefix_keys():
    pred = torch.eye(3).unsqueeze(0)
    out = tensor_metrics(pred, pred, prefix=" t ")
    assert set(out) == {"t_Frobenius", "t_SymViolation", "t_EigAgreement"}


def test_classification_metrics_prefix_normalization():
    out = classification_metrics(
        torch.tensor([0.1, 0.9, -0.2]),
        torch.tensor([0.0, 1.0, 0.0]),
        prefix=" cls ",
    )
    assert set(out) == {"cls_Accuracy", "cls_Precision", "cls_Recall", "cls_F1", "cls_AUC"}


def test_classification_metrics_rejects_boolean_metric_outputs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(metrics_module, "accuracy_score", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(metrics_module, "roc_auc_score", lambda *_args, **_kwargs: False)

    out = classification_metrics(
        torch.tensor([0.2, 0.8, 0.1, 0.9]),
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
    )
    # bool outputs should not leak into metric payload as 1.0/0.0 via implicit casting.
    assert out["Accuracy"] == pytest.approx(0.0)
    assert out["AUC"] == pytest.approx(0.5)
