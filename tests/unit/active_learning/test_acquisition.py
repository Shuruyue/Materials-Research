"""Tests for acquisition strategy normalization and scoring."""

import pytest
import torch

from atlas.active_learning.acquisition import (
    DISCOVERY_ACQUISITION_STRATEGIES,
    normalize_acquisition_strategy,
    score_acquisition,
)


def test_strategy_normalization_aliases():
    assert normalize_acquisition_strategy("expected_improvement") == "ei"
    assert normalize_acquisition_strategy("upper-confidence-bound") == "ucb"
    assert normalize_acquisition_strategy("probability_of_improvement") == "pi"
    assert normalize_acquisition_strategy("thompson_sampling") == "thompson"
    assert normalize_acquisition_strategy("std") == "uncertainty"


def test_strategy_normalization_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown acquisition strategy"):
        normalize_acquisition_strategy("foo_bar")


def test_strategy_normalization_controller_only_hint():
    with pytest.raises(ValueError, match="Controller-only strategies: hybrid, stability"):
        normalize_acquisition_strategy("hybrid")


def test_mean_strategy_respects_minimization_direction():
    mean = torch.tensor([-1.2, -0.2], dtype=torch.float32)
    std = torch.tensor([0.1, 0.1], dtype=torch.float32)
    score = score_acquisition(mean, std, strategy="mean", maximize=False)
    assert score.shape == mean.shape
    assert score[0] > score[1]


def test_lcb_strategy_maximization_differs_from_ucb():
    mean = torch.tensor([1.0], dtype=torch.float32)
    std = torch.tensor([0.5], dtype=torch.float32)
    ucb_score = score_acquisition(mean, std, strategy="ucb", maximize=True, kappa=2.0)
    lcb_score = score_acquisition(mean, std, strategy="lcb", maximize=True, kappa=2.0)
    assert ucb_score.item() == pytest.approx(2.0)
    assert lcb_score.item() == pytest.approx(0.0)
    assert ucb_score.item() > lcb_score.item()


def test_ucb_strategy_minimization_prefers_lower_lcb():
    mean = torch.tensor([-1.0, -0.3], dtype=torch.float32)
    std = torch.tensor([0.1, 0.1], dtype=torch.float32)
    score = score_acquisition(mean, std, strategy="ucb", maximize=False, kappa=2.0)
    assert score[0] > score[1]


def test_thompson_strategy_is_reproducible_with_seeded_generators():
    mean = torch.tensor([-0.5, -0.5], dtype=torch.float32)
    std = torch.tensor([0.2, 0.2], dtype=torch.float32)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    s1 = score_acquisition(mean, std, strategy="thompson", maximize=False, generator=g1)
    s2 = score_acquisition(mean, std, strategy="thompson", maximize=False, generator=g2)
    assert torch.allclose(s1, s2)


def test_discovery_strategy_catalog_includes_hybrid_switch_modes():
    assert "hybrid" in DISCOVERY_ACQUISITION_STRATEGIES
    assert "stability" in DISCOVERY_ACQUISITION_STRATEGIES
    assert "ei" in DISCOVERY_ACQUISITION_STRATEGIES


def test_score_acquisition_aligns_std_dtype_to_mean_dtype():
    mean = torch.tensor([0.1], dtype=torch.float64)
    std = torch.tensor([0.2], dtype=torch.float32)
    score = score_acquisition(mean, std, strategy="ucb", maximize=True)
    assert score.dtype == mean.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available in this environment")
def test_score_acquisition_aligns_std_device_to_mean_device():
    mean = torch.tensor([0.1], dtype=torch.float32, device="cuda")
    std = torch.tensor([0.2], dtype=torch.float32, device="cpu")
    score = score_acquisition(mean, std, strategy="ucb", maximize=True)
    assert score.device == mean.device


def test_score_acquisition_handles_nonpositive_and_nonfinite_std():
    mean = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    std = torch.tensor([0.0, -0.2, float("nan"), float("inf")], dtype=torch.float32)
    score = score_acquisition(mean, std, strategy="pi", maximize=True, best_f=0.0)
    assert torch.isfinite(score).all()
    assert torch.all((score >= 0.0) & (score <= 1.0))


def test_expected_improvement_with_tiny_std_is_finite():
    mean = torch.tensor([1.0], dtype=torch.float32)
    std = torch.tensor([0.0], dtype=torch.float32)
    score = score_acquisition(mean, std, strategy="ei", maximize=True, best_f=0.5)
    assert torch.isfinite(score).all()
    assert score.item() >= 0.0


def test_score_acquisition_supports_broadcast_shapes():
    mean = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)
    std = torch.tensor([[0.01, 0.02, 0.03, 0.04]], dtype=torch.float32)
    score = score_acquisition(mean, std, strategy="ucb", maximize=True, kappa=1.0)
    assert score.shape == (3, 4)
