"""Focused tests for DiscoveryController acquisition strategy behavior."""

import torch

from atlas.active_learning.controller import Candidate, DiscoveryController


def _controller_stub(strategy: str) -> DiscoveryController:
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.acquisition_strategy = strategy
    controller.acquisition_kappa = 2.0
    controller.acquisition_kappa_schedule = "fixed"
    controller.acquisition_kappa_min = 1.0
    controller.acquisition_kappa_decay = 0.08
    controller.acquisition_ucb_delta = 0.1
    controller.acquisition_ucb_dimension = 1
    controller.acquisition_best_f = -0.5
    controller.acquisition_jitter = 0.01
    controller.use_noisy_ei = True
    controller.noisy_ei_mc_samples = 64
    controller.batch_diversity_strength = 0.15
    controller.batch_diversity_sigma = 1.0
    controller.use_constrained_acquisition = True
    controller.weights = {"topo": 0.4, "stability": 0.3, "heuristic": 0.15, "novelty": 0.15}
    controller.known_formulas = set()
    controller.top_candidates = []
    controller.gp_acquirer = None
    controller.synthesizability_evaluator = None
    controller.iteration = 1
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


def test_stability_component_hybrid_with_uq_is_positive():
    controller = _controller_stub("hybrid")
    cand = Candidate(
        structure=None,
        formula="D",
        stability_score=0.2,
        energy_mean=-0.9,
        energy_std=0.15,
    )
    assert controller._stability_component(cand) > 0.0


def test_score_and_select_uses_feasibility_constraint_term():
    controller = _controller_stub("hybrid")
    a = Candidate(
        structure=None,
        formula="A",
        topo_probability=1.0,
        heuristic_topo_score=0.0,
        stability_score=0.5,
        energy_mean=-1.0,
        energy_std=0.2,
    )
    b = Candidate(
        structure=None,
        formula="B",
        topo_probability=0.0,
        heuristic_topo_score=0.0,
        stability_score=0.5,
        energy_mean=-1.0,
        energy_std=0.2,
    )
    ranked = controller._score_and_select([a, b], n_top=2)
    assert ranked[0].formula == "A"
    assert a.acquisition_value > b.acquisition_value
