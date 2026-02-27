"""Focused tests for DiscoveryController acquisition strategy behavior."""

import torch

from atlas.active_learning.controller import Candidate, DiscoveryController


def _controller_stub(strategy: str) -> DiscoveryController:
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.acquisition_strategy = strategy
    controller.acquisition_kappa = 2.0
    controller.acquisition_best_f = -0.5
    controller.acquisition_jitter = 0.01
    controller._acquisition_generator = torch.Generator().manual_seed(7)
    return controller


def test_stability_component_uses_direct_stability_mode():
    controller = _controller_stub("stability")
    cand = Candidate(
        structure=None,
        formula="A",
        stability_score=0.8,
        energy_mean=-1.2,
        energy_std=0.2,
    )
    assert controller._stability_component(cand) == 0.8


def test_stability_component_mean_strategy_prefers_lower_energy():
    controller = _controller_stub("mean")
    low_energy = Candidate(structure=None, formula="A", stability_score=0.5, energy_mean=-1.0, energy_std=0.2)
    high_energy = Candidate(structure=None, formula="B", stability_score=0.5, energy_mean=-0.2, energy_std=0.2)

    low_score = controller._stability_component(low_energy)
    high_score = controller._stability_component(high_energy)

    assert low_score > high_score


def test_stability_component_hybrid_falls_back_without_uncertainty():
    controller = _controller_stub("hybrid")
    cand = Candidate(
        structure=None,
        formula="C",
        stability_score=0.61,
        energy_mean=-0.8,
        energy_std=None,
    )
    assert controller._stability_component(cand) == 0.61
