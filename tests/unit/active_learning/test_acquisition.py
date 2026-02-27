"""Tests for acquisition strategy normalization and scoring."""

import torch
import pytest

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


def test_mean_strategy_respects_minimization_direction():
    mean = torch.tensor([-1.2, -0.2], dtype=torch.float32)
    std = torch.tensor([0.1, 0.1], dtype=torch.float32)
    score = score_acquisition(mean, std, strategy="mean", maximize=False)
    assert score.shape == mean.shape
    assert score[0] > score[1]


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
