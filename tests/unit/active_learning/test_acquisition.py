"""Tests for acquisition strategy normalization and scoring."""

import math

import pytest
import torch

from atlas.active_learning.acquisition import (
    DISCOVERY_ACQUISITION_STRATEGIES,
    normalize_acquisition_strategy,
    schedule_ucb_kappa,
    score_acquisition,
)


def test_strategy_normalization_aliases():
    assert normalize_acquisition_strategy("expected_improvement") == "ei"
    assert normalize_acquisition_strategy("log-expected-improvement") == "log_ei"
    assert normalize_acquisition_strategy("upper-confidence-bound") == "ucb"
    assert normalize_acquisition_strategy("probability_of_improvement") == "pi"
    assert normalize_acquisition_strategy("log_pi") == "log_pi"
    assert normalize_acquisition_strategy("noisy_expected_improvement") == "nei"
    assert normalize_acquisition_strategy("log_noisy_expected_improvement") == "log_nei"
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
    assert "log_ei" in DISCOVERY_ACQUISITION_STRATEGIES
    assert "log_pi" in DISCOVERY_ACQUISITION_STRATEGIES
    assert "nei" in DISCOVERY_ACQUISITION_STRATEGIES
    assert "log_nei" in DISCOVERY_ACQUISITION_STRATEGIES


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


def test_log_ei_stays_finite_in_extreme_negative_tail():
    mean = torch.tensor([-100.0], dtype=torch.float64)
    std = torch.tensor([1.0], dtype=torch.float64)
    score = score_acquisition(mean, std, strategy="log_ei", maximize=True, best_f=0.0)
    assert torch.isfinite(score).all()
    assert score.item() < 0.0


def test_log_ei_matches_ei_in_regular_regime():
    mean = torch.tensor([0.7], dtype=torch.float64)
    std = torch.tensor([0.2], dtype=torch.float64)
    ei = score_acquisition(mean, std, strategy="ei", maximize=True, best_f=0.3)
    log_ei = score_acquisition(mean, std, strategy="log_ei", maximize=True, best_f=0.3)
    assert torch.exp(log_ei).item() == pytest.approx(ei.item(), rel=1e-3, abs=1e-8)


def test_log_pi_stays_finite_and_non_positive():
    mean = torch.tensor([-80.0], dtype=torch.float64)
    std = torch.tensor([1.0], dtype=torch.float64)
    log_pi = score_acquisition(mean, std, strategy="log_pi", maximize=True, best_f=0.0)
    assert torch.isfinite(log_pi).all()
    assert log_pi.item() <= 0.0


def test_nei_is_finite_with_observation_uncertainty():
    mean = torch.tensor([-0.8], dtype=torch.float64)
    std = torch.tensor([0.2], dtype=torch.float64)
    obs_mean = torch.tensor([-1.1, -0.9, -1.0], dtype=torch.float64)
    obs_std = torch.tensor([0.05, 0.10, 0.08], dtype=torch.float64)
    nei = score_acquisition(
        mean,
        std,
        strategy="nei",
        maximize=False,
        observed_mean=obs_mean,
        observed_std=obs_std,
        nei_mc_samples=64,
    )
    assert torch.isfinite(nei).all()
    assert nei.item() >= 0.0


def test_log_nei_is_finite_with_observation_uncertainty():
    mean = torch.tensor([-0.8], dtype=torch.float64)
    std = torch.tensor([0.2], dtype=torch.float64)
    obs_mean = torch.tensor([-1.1, -0.9, -1.0], dtype=torch.float64)
    obs_std = torch.tensor([0.05, 0.10, 0.08], dtype=torch.float64)
    log_nei = score_acquisition(
        mean,
        std,
        strategy="log_nei",
        maximize=False,
        observed_mean=obs_mean,
        observed_std=obs_std,
        nei_mc_samples=64,
    )
    assert torch.isfinite(log_nei).all()
    assert log_nei.item() <= 0.0


def test_nei_reduces_to_ei_when_observation_noise_is_zero():
    mean = torch.tensor([-0.8], dtype=torch.float64)
    std = torch.tensor([0.2], dtype=torch.float64)
    obs_mean = torch.tensor([-1.1, -0.9, -1.0], dtype=torch.float64)
    obs_std = torch.zeros_like(obs_mean)

    nei = score_acquisition(
        mean,
        std,
        strategy="nei",
        maximize=False,
        observed_mean=obs_mean,
        observed_std=obs_std,
        nei_mc_samples=1024,
    )
    ei_ref = score_acquisition(
        mean,
        std,
        strategy="ei",
        maximize=False,
        best_f=-1.1,
    )
    assert torch.allclose(nei, ei_ref)


def test_nei_observation_std_shape_mismatch_raises():
    mean = torch.tensor([-0.8], dtype=torch.float64)
    std = torch.tensor([0.2], dtype=torch.float64)
    with pytest.raises(ValueError, match="same number of elements"):
        score_acquisition(
            mean,
            std,
            strategy="nei",
            maximize=False,
            observed_mean=torch.tensor([-1.0, -0.9], dtype=torch.float64),
            observed_std=torch.tensor([0.1], dtype=torch.float64),
        )


def test_nei_filters_nonfinite_observations_before_sampling():
    mean = torch.tensor([-0.8], dtype=torch.float64)
    std = torch.tensor([0.2], dtype=torch.float64)
    dirty_obs_mean = torch.tensor([-1.1, float("nan"), -0.9, float("inf")], dtype=torch.float64)
    dirty_obs_std = torch.tensor([0.05, 0.10, 0.08, 0.07], dtype=torch.float64)
    clean_obs_mean = torch.tensor([-1.1, -0.9], dtype=torch.float64)
    clean_obs_std = torch.tensor([0.05, 0.08], dtype=torch.float64)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)

    dirty = score_acquisition(
        mean,
        std,
        strategy="nei",
        maximize=False,
        observed_mean=dirty_obs_mean,
        observed_std=dirty_obs_std,
        nei_mc_samples=64,
        generator=g1,
    )
    clean = score_acquisition(
        mean,
        std,
        strategy="nei",
        maximize=False,
        observed_mean=clean_obs_mean,
        observed_std=clean_obs_std,
        nei_mc_samples=64,
        generator=g2,
    )
    assert torch.isfinite(dirty).all()
    assert torch.allclose(dirty, clean)


def test_score_acquisition_supports_broadcast_shapes():
    mean = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)
    std = torch.tensor([[0.01, 0.02, 0.03, 0.04]], dtype=torch.float32)
    score = score_acquisition(mean, std, strategy="ucb", maximize=True, kappa=1.0)
    assert score.shape == (3, 4)


def test_schedule_ucb_kappa_anneal_is_monotonic_decreasing():
    k1 = schedule_ucb_kappa(1, base_kappa=3.0, mode="anneal", kappa_min=1.0, decay=0.2)
    k10 = schedule_ucb_kappa(10, base_kappa=3.0, mode="anneal", kappa_min=1.0, decay=0.2)
    assert k1 > k10
    assert k10 >= 1.0


def test_schedule_ucb_kappa_gp_ucb_grows_with_time():
    k1 = schedule_ucb_kappa(1, base_kappa=2.0, mode="gp_ucb", dimension=1, delta=0.1, kappa_min=1.0)
    k50 = schedule_ucb_kappa(50, base_kappa=2.0, mode="gp_ucb", dimension=1, delta=0.1, kappa_min=1.0)
    assert k50 >= k1


def test_schedule_ucb_kappa_sanitizes_nonfinite_inputs():
    out = schedule_ucb_kappa(
        3,
        base_kappa=float("nan"),
        mode="gp_ucb",
        dimension=0,
        delta=float("inf"),
        kappa_min=-1.0,
        decay=float("nan"),
    )
    assert math.isfinite(out)
    assert out >= 0.0


def test_upper_confidence_bound_clamps_negative_kappa_to_zero():
    mean = torch.tensor([1.5], dtype=torch.float32)
    std = torch.tensor([0.4], dtype=torch.float32)
    score = score_acquisition(mean, std, strategy="ucb", maximize=True, kappa=-3.0)
    assert score.item() == pytest.approx(mean.item(), rel=0.0, abs=1e-12)


def test_score_acquisition_sanitizes_nonfinite_bestf_and_negative_jitter():
    mean = torch.tensor([0.5], dtype=torch.float64)
    std = torch.tensor([0.1], dtype=torch.float64)
    score = score_acquisition(
        mean,
        std,
        strategy="ei",
        maximize=True,
        best_f=float("nan"),
        jitter=-1.0,
    )
    assert torch.isfinite(score).all()
    assert score.item() >= 0.0


def test_score_acquisition_sanitizes_nonfinite_mean_for_mean_strategy():
    mean = torch.tensor([float("nan"), float("inf"), float("-inf")], dtype=torch.float32)
    std = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)
    score = score_acquisition(mean, std, strategy="mean", maximize=True)
    assert torch.isfinite(score).all()
    assert torch.all(score == 0.0)


def test_score_acquisition_uses_dynamic_kappa_schedule():
    mean = torch.tensor([0.5], dtype=torch.float64)
    std = torch.tensor([0.1], dtype=torch.float64)
    fixed = score_acquisition(mean, std, strategy="ucb", maximize=True, kappa=2.0, kappa_schedule="fixed", iteration=10)
    annealed = score_acquisition(
        mean,
        std,
        strategy="ucb",
        maximize=True,
        kappa=3.0,
        kappa_schedule="anneal",
        kappa_min=1.0,
        kappa_decay=0.4,
        iteration=10,
    )
    assert annealed.item() != pytest.approx(fixed.item())


def test_score_acquisition_sanitizes_non_integral_controls():
    mean = torch.tensor([-0.8], dtype=torch.float64)
    std = torch.tensor([0.2], dtype=torch.float64)
    obs_mean = torch.tensor([-1.1, -0.9, -1.0], dtype=torch.float64)
    obs_std = torch.tensor([0.05, 0.10, 0.08], dtype=torch.float64)

    g1 = torch.Generator().manual_seed(17)
    g2 = torch.Generator().manual_seed(17)
    sampled = score_acquisition(
        mean,
        std,
        strategy="nei",
        maximize=False,
        observed_mean=obs_mean,
        observed_std=obs_std,
        nei_mc_samples=63.7,
        generator=g1,
    )
    baseline = score_acquisition(
        mean,
        std,
        strategy="nei",
        maximize=False,
        observed_mean=obs_mean,
        observed_std=obs_std,
        nei_mc_samples=128,
        generator=g2,
    )
    assert torch.allclose(sampled, baseline)

    score_bad_iter = score_acquisition(
        torch.tensor([0.5], dtype=torch.float64),
        torch.tensor([0.1], dtype=torch.float64),
        strategy="ucb",
        maximize=True,
        kappa=3.0,
        kappa_schedule="anneal",
        kappa_min=1.0,
        kappa_decay=0.2,
        iteration=10.5,
    )
    score_iter_1 = score_acquisition(
        torch.tensor([0.5], dtype=torch.float64),
        torch.tensor([0.1], dtype=torch.float64),
        strategy="ucb",
        maximize=True,
        kappa=3.0,
        kappa_schedule="anneal",
        kappa_min=1.0,
        kappa_decay=0.2,
        iteration=1,
    )
    assert torch.allclose(score_bad_iter, score_iter_1)
