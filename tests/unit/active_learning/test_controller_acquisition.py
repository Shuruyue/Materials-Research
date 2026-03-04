"""Focused tests for DiscoveryController acquisition strategy behavior."""

from collections import OrderedDict
from unittest.mock import patch

import numpy as np
import torch

from atlas.active_learning.controller import Candidate, DiscoveryController


class _FakeComposition:
    def __init__(self, formula: str):
        self.reduced_formula = formula


class _FakeStructure:
    def __init__(self, formula: str, sid: int):
        self.composition = _FakeComposition(formula)
        self.sid = sid


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
    controller.batch_diversity_space = "performance"
    controller.use_constrained_acquisition = True
    controller.dynamic_best_f = True
    controller.dynamic_best_f_quantile = 0.0
    controller.max_observation_history = 5000
    controller.use_pareto_rank_bonus = False
    controller.pareto_rank_bonus_weight = 0.2
    controller.use_pareto_hv_bonus = False
    controller.pareto_hv_bonus_weight = 0.25
    controller.pareto_feasibility_threshold = 0.05
    controller.use_hv_batch_greedy = False
    controller.hv_batch_weight = 0.35
    controller.hv_mc_samples = 4096
    controller.hv_mc_seed = 17
    controller.hv_use_shared_samples = True
    controller.hv_candidate_pool_limit = 96
    controller.hv_chunk_size = 256
    controller.experimental_algorithms_enabled = True
    controller.use_synthesis_objective = False
    controller.synthesis_objective_weight = 0.15
    controller.synthesis_eval_topk = 128
    controller.synthesis_gate_topo_threshold = 0.2
    controller.synthesis_score_floor = 0.0
    controller.synthesis_eval_strategy = "hybrid_topk_uncertain"
    controller.synthesis_uncertainty_weight = 0.35
    controller.synthesis_uncertainty_decay = 0.8
    controller.synthesis_time_budget_sec = 1.25
    controller.synthesis_cache_max_size = 50000
    controller.pareto_joint_feasibility = True
    controller.pareto_synthesis_feasibility_threshold = 0.10
    controller.use_ood_penalty = False
    controller.ood_penalty_weight = 0.15
    controller.ood_history_min_points = 24
    controller.ood_quantile = 0.95
    controller.ood_space = "chemistry"
    controller.max_pathway_annotations_per_iter = 8
    controller.enable_structure_dedup = False
    controller._structure_matcher = None
    controller.weights = {"topo": 0.4, "stability": 0.3, "heuristic": 0.15, "novelty": 0.15}
    controller.known_formulas = set()
    controller.known_structures_by_formula = {}
    controller.all_candidates = []
    controller.top_candidates = []
    controller.gp_acquirer = None
    controller.synthesizability_evaluator = None
    controller._synthesis_cache = OrderedDict()
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


def test_dynamic_best_f_changes_hybrid_stability_score():
    dynamic_controller = _controller_stub("hybrid")
    dynamic_controller.use_noisy_ei = False
    dynamic_controller.dynamic_best_f = True
    dynamic_controller.all_candidates = [
        Candidate(structure=None, formula="H1", energy_mean=-1.2, energy_std=0.0),
        Candidate(structure=None, formula="H2", energy_mean=-1.0, energy_std=0.0),
    ]
    fixed_controller = _controller_stub("hybrid")
    fixed_controller.use_noisy_ei = False
    fixed_controller.dynamic_best_f = False
    fixed_controller.all_candidates = list(dynamic_controller.all_candidates)

    query = Candidate(
        structure=None,
        formula="Q",
        stability_score=0.2,
        energy_mean=-0.95,
        energy_std=0.1,
    )
    dynamic_score = dynamic_controller._stability_component(query)
    fixed_score = fixed_controller._stability_component(query)
    assert dynamic_score < fixed_score


def test_score_and_select_updates_gp_with_all_evaluated_candidates():
    class _DummyGP:
        class config:  # noqa: N801 - mirrors runtime object shape
            blend_ratio = 0.0

        def __init__(self):
            self.last_update_count = 0

        def suggest_constrained_utility(self, candidates):
            return np.zeros(len(candidates), dtype=float)

        def suggest_ucb(self, candidates):
            return np.zeros(len(candidates), dtype=float)

        def update(self, candidates):
            self.last_update_count = len(candidates)

    controller = _controller_stub("hybrid")
    controller.gp_acquirer = _DummyGP()
    controller.use_pareto_hv_bonus = False

    candidates = [
        Candidate(structure=None, formula="A", topo_probability=0.9, stability_score=0.8, energy_mean=-1.0, energy_std=0.1),
        Candidate(structure=None, formula="B", topo_probability=0.4, stability_score=0.6, energy_mean=-0.8, energy_std=0.1),
        Candidate(structure=None, formula="C", topo_probability=0.2, stability_score=0.5, energy_mean=-0.6, energy_std=0.1),
    ]
    controller._score_and_select(candidates, n_top=1)
    assert controller.gp_acquirer.last_update_count == len(candidates)


def test_structure_aware_dedup_does_not_collapse_polymorphs():
    controller = _controller_stub("hybrid")
    controller.enable_structure_dedup = True
    controller._structures_match = lambda a, b: getattr(a, "sid", None) == getattr(b, "sid", None)

    known = _FakeStructure("SiO2", sid=1)
    same_polymorph = _FakeStructure("SiO2", sid=1)
    different_polymorph = _FakeStructure("SiO2", sid=2)
    controller._register_known_structure(known, formula_hint="SiO2")

    assert controller._is_duplicate_structure(same_polymorph) is True
    assert controller._is_duplicate_structure(different_polymorph) is False


def test_pareto_rank_bonus_demotes_dominated_candidate():
    controller = _controller_stub("hybrid")
    controller.weights = {"topo": 0.0, "stability": 0.0, "heuristic": 0.0, "novelty": 0.0}
    controller.use_pareto_rank_bonus = True
    controller.use_pareto_hv_bonus = False
    controller.use_hv_batch_greedy = False
    controller.pareto_feasibility_threshold = 0.0
    controller._stability_component = lambda c: float(c.stability_score)

    a = Candidate(structure=None, formula="A", topo_probability=0.95, stability_score=0.2)
    b = Candidate(structure=None, formula="B", topo_probability=0.2, stability_score=0.95)
    d = Candidate(structure=None, formula="D", topo_probability=0.1, stability_score=0.1)

    ranked = controller._score_and_select([a, b, d], n_top=3)
    assert ranked[-1].formula == "D"
    assert d.acquisition_value < a.acquisition_value
    assert d.acquisition_value < b.acquisition_value


def test_hv_batch_greedy_prefers_complementary_objective_point():
    controller = _controller_stub("hybrid")
    controller.batch_diversity_strength = 0.0
    controller.use_hv_batch_greedy = True
    controller.hv_batch_weight = 1.0
    controller.pareto_feasibility_threshold = 0.0

    c0 = Candidate(structure=None, formula="A", acquisition_value=1.0, topo_probability=1.0, stability_score=0.6)
    c1 = Candidate(structure=None, formula="B", acquisition_value=1.0, topo_probability=0.6, stability_score=1.0)
    c2 = Candidate(structure=None, formula="C", acquisition_value=1.0, topo_probability=0.8, stability_score=0.5)
    candidates = [c0, c1, c2]
    objective_map = {
        id(c0): np.array([1.0, 0.6], dtype=float),
        id(c1): np.array([0.6, 1.0], dtype=float),
        id(c2): np.array([0.8, 0.5], dtype=float),
    }

    selected = controller._select_top_diverse(candidates, n_top=2, objective_map=objective_map)
    selected_formulas = {c.formula for c in selected}
    assert "A" in selected_formulas
    assert "B" in selected_formulas


def test_synthesis_objective_biases_ranking_when_enabled():
    class _DummySynth:
        def evaluate(self, formula, energy):
            score = {"A": 0.2, "B": 0.9}.get(str(formula), 0.0)
            return {"synthesizable": score >= 0.5, "score": score, "pathway": [f"X -> {formula}"]}

    controller = _controller_stub("hybrid")
    controller.use_synthesis_objective = True
    controller.synthesis_objective_weight = 1.0
    controller.synthesis_eval_topk = 8
    controller.synthesis_gate_topo_threshold = 0.0
    controller.synthesizability_evaluator = _DummySynth()
    controller._stability_component = lambda c: 0.5
    controller.use_pareto_rank_bonus = False
    controller.use_pareto_hv_bonus = False
    controller.use_hv_batch_greedy = False

    a = Candidate(structure=None, formula="A", topo_probability=1.0, stability_score=0.5)
    b = Candidate(structure=None, formula="B", topo_probability=1.0, stability_score=0.5)

    ranked = controller._score_and_select([a, b], n_top=2)
    assert ranked[0].formula == "B"
    assert b.synthesis_score > a.synthesis_score


def test_hypervolume_supports_three_objectives():
    controller = _controller_stub("hybrid")
    controller.hv_mc_samples = 12000
    controller.hv_mc_seed = 3
    points = np.array(
        [
            [0.80, 0.30, 0.40],
            [0.40, 0.80, 0.50],
        ],
        dtype=float,
    )
    ref = np.array([0.0, 0.0, 0.0], dtype=float)
    hv = controller._hypervolume(points, ref)
    assert hv > 0.0
    assert hv < 0.40


def test_hv_batch_greedy_supports_three_objective_map():
    controller = _controller_stub("hybrid")
    controller.batch_diversity_strength = 0.0
    controller.use_hv_batch_greedy = True
    controller.hv_batch_weight = 1.0
    controller.pareto_feasibility_threshold = 0.0
    controller.hv_mc_samples = 10000
    controller.hv_mc_seed = 9

    c0 = Candidate(structure=None, formula="A", acquisition_value=1.0, topo_probability=1.0, stability_score=0.60, synthesis_score=0.20)
    c1 = Candidate(structure=None, formula="B", acquisition_value=1.0, topo_probability=0.60, stability_score=1.00, synthesis_score=0.90)
    c2 = Candidate(structure=None, formula="C", acquisition_value=1.0, topo_probability=0.95, stability_score=0.55, synthesis_score=0.15)
    candidates = [c0, c1, c2]
    objective_map = {
        id(c0): np.array([1.0, 0.6, 0.2], dtype=float),
        id(c1): np.array([0.6, 1.0, 0.9], dtype=float),
        id(c2): np.array([0.95, 0.55, 0.15], dtype=float),
    }

    selected = controller._select_top_diverse(candidates, n_top=2, objective_map=objective_map)
    selected_formulas = {c.formula for c in selected}
    assert "A" in selected_formulas
    assert "B" in selected_formulas


def test_synthesis_hybrid_budget_selects_uncertain_candidate():
    class _DummySynth:
        def evaluate(self, formula, energy):
            return {"synthesizable": True, "score": 0.8 if formula == "C" else 0.1, "pathway": [f"P->{formula}"]}

    controller = _controller_stub("hybrid")
    controller.use_synthesis_objective = True
    controller.synthesis_eval_topk = 2
    controller.synthesis_eval_strategy = "hybrid_topk_uncertain"
    controller.synthesis_uncertainty_weight = 0.5
    controller.synthesis_gate_topo_threshold = 0.0
    controller.synthesizability_evaluator = _DummySynth()
    controller.use_pareto_rank_bonus = False
    controller.use_pareto_hv_bonus = False
    controller.use_hv_batch_greedy = False
    controller._stability_component = lambda c: 0.4

    # A, B have higher provisional score; C has highest uncertainty and should be evaluated.
    a = Candidate(structure=None, formula="A", topo_probability=0.9, stability_score=0.5, energy_std=0.1)
    b = Candidate(structure=None, formula="B", topo_probability=0.8, stability_score=0.5, energy_std=0.2)
    c = Candidate(structure=None, formula="C", topo_probability=0.7, stability_score=0.5, energy_std=1.5)
    controller._score_and_select([a, b, c], n_top=3)

    assert c.synthesis_score > 0.0


def test_synthesis_cache_lru_capacity_is_enforced():
    class _DummySynth:
        def evaluate(self, formula, energy):
            return {"synthesizable": True, "score": 0.1, "pathway": []}

    controller = _controller_stub("hybrid")
    controller.synthesizability_evaluator = _DummySynth()
    controller.synthesis_cache_max_size = 2

    c1 = Candidate(structure=None, formula="A", energy_mean=-1.0)
    c2 = Candidate(structure=None, formula="B", energy_mean=-1.0)
    c3 = Candidate(structure=None, formula="C", energy_mean=-1.0)
    controller._evaluate_synthesizability(c1)
    controller._evaluate_synthesizability(c2)
    controller._evaluate_synthesizability(c3)

    assert len(controller._synthesis_cache) == 2
    assert ("A", -1.0) not in controller._synthesis_cache


def test_joint_feasibility_gate_blocks_low_synthesis_in_hv():
    controller = _controller_stub("hybrid")
    controller.use_pareto_hv_bonus = True
    controller.pareto_feasibility_threshold = 0.0
    controller.pareto_joint_feasibility = True
    controller.pareto_synthesis_feasibility_threshold = 0.4
    controller.hv_use_shared_samples = True
    controller.hv_mc_samples = 8000

    a = Candidate(structure=None, formula="A", acquisition_value=0.5, topo_probability=1.0, stability_score=0.6, synthesis_score=0.9)
    b = Candidate(structure=None, formula="B", acquisition_value=0.5, topo_probability=1.0, stability_score=0.7, synthesis_score=0.1)
    cands = [a, b]
    topo_terms = [1.0, 1.0]
    stability_terms = [0.6, 0.7]
    synthesis_terms = [0.9, 0.1]

    controller._apply_pareto_hv_bonus(cands, topo_terms, stability_terms, synthesis_terms)
    assert a.acquisition_value > b.acquisition_value


def test_synthesis_time_budget_stops_remaining_evaluations():
    controller = _controller_stub("hybrid")
    controller.use_synthesis_objective = True
    controller.synthesizability_evaluator = object()
    controller.synthesis_eval_topk = 3
    controller.synthesis_gate_topo_threshold = 0.0
    controller.synthesis_time_budget_sec = 0.01
    controller.synthesis_eval_strategy = "topk"

    call_count = {"n": 0}

    def _fake_eval(_cand):
        call_count["n"] += 1
        return {"synthesizable": True, "score": 0.7, "pathway": []}

    controller._evaluate_synthesizability = _fake_eval
    cands = [
        Candidate(structure=None, formula="A", topo_probability=0.9, acquisition_value=0.9),
        Candidate(structure=None, formula="B", topo_probability=0.8, acquisition_value=0.8),
        Candidate(structure=None, formula="C", topo_probability=0.7, acquisition_value=0.7),
    ]
    with patch(
        "atlas.active_learning.controller.time.perf_counter",
        side_effect=[0.0, 0.0, 0.02, 0.03, 0.04],
    ):
        scores = controller._apply_synthesis_objective(cands)

    assert call_count["n"] == 1
    assert scores[0] > 0.0
    assert scores[1] == 0.0
    assert scores[2] == 0.0


def test_synthesis_uncertainty_decay_penalizes_high_std_candidate():
    controller = _controller_stub("hybrid")
    controller.use_synthesis_objective = True
    controller.synthesis_eval_topk = 2
    controller.synthesis_gate_topo_threshold = 0.0
    controller.synthesis_eval_strategy = "topk"
    controller.synthesis_uncertainty_decay = 2.0
    controller.synthesizability_evaluator = object()
    controller._evaluate_synthesizability = lambda _cand: {
        "synthesizable": True,
        "score": 1.0,
        "pathway": [],
    }

    low_std = Candidate(structure=None, formula="A", topo_probability=0.8, acquisition_value=0.9, energy_std=0.0)
    high_std = Candidate(structure=None, formula="B", topo_probability=0.8, acquisition_value=0.8, energy_std=1.0)
    controller._apply_synthesis_objective([low_std, high_std])

    assert low_std.synthesis_score > high_std.synthesis_score
    assert high_std.synthesis_score < 1.0


def test_ood_penalty_reduces_outlier_utility():
    controller = _controller_stub("hybrid")
    controller.use_ood_penalty = True
    controller.ood_penalty_weight = 1.0
    controller.ood_history_min_points = 4
    controller.ood_quantile = 0.8
    controller.ood_space = "performance"
    controller.use_pareto_rank_bonus = False
    controller.use_pareto_hv_bonus = False
    controller.use_hv_batch_greedy = False
    controller.use_constrained_acquisition = False
    controller._stability_component = lambda _c: 0.5
    controller.weights = {"topo": 0.5, "stability": 0.5, "heuristic": 0.0, "novelty": 0.0}

    controller.all_candidates = [
        Candidate(
            structure=None,
            formula=f"H{i}",
            topo_probability=0.52 + 0.01 * (i % 2),
            stability_score=0.5,
            heuristic_topo_score=0.0,
            novelty_score=0.0,
            energy_mean=-1.0 + 0.01 * i,
            energy_std=0.05,
        )
        for i in range(6)
    ]

    inlier = Candidate(
        structure=None,
        formula="IN",
        topo_probability=0.53,
        stability_score=0.5,
        heuristic_topo_score=0.0,
        energy_mean=-0.98,
        energy_std=0.05,
    )
    outlier = Candidate(
        structure=None,
        formula="OUT",
        topo_probability=0.99,
        stability_score=0.5,
        heuristic_topo_score=0.0,
        energy_mean=3.0,
        energy_std=3.0,
    )

    ranked = controller._score_and_select([inlier, outlier], n_top=2)

    by_formula = {c.formula: c for c in ranked}
    assert by_formula["OUT"].ood_score > by_formula["IN"].ood_score
    assert by_formula["OUT"].acquisition_value < by_formula["IN"].acquisition_value
