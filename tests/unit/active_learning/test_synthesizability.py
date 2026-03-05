"""Algorithm-focused tests for synthesizability evaluator."""

import math

from atlas.active_learning.synthesizability import SynthesisPathfinder


def test_heuristic_graph_produces_ranked_pathways():
    ev = SynthesisPathfinder(max_steps=6)
    out = ev.evaluate("LiFePO4", -1.05)

    assert out["graph_nodes"] > 0
    assert out["graph_edges"] > 0
    assert out["pathway_count"] > 0
    assert len(out["pathway"]) >= 1
    assert out["score"] > 0.0


def test_more_exothermic_candidate_gets_higher_score():
    ev = SynthesisPathfinder(max_steps=6)

    strong = ev.evaluate("LiFePO4", -1.20)
    weak = ev.evaluate("LiFePO4", -0.05)

    assert strong["score"] > weak["score"]


def test_custom_pathways_are_pareto_ranked():
    ev = SynthesisPathfinder(max_steps=6)

    bad_route = {
        "nodes": ["R", "X", "AB2"],
        "delta_g": [0.10, -0.10],
        "activation": [0.90, 0.80],
        "uncertainty": [0.08, 0.08],
    }
    good_route = {
        "nodes": ["R", "Y", "AB2"],
        "delta_g": [-0.40, -0.40],
        "activation": [0.25, 0.20],
        "uncertainty": [0.03, 0.03],
    }

    out = ev.evaluate("AB2", -0.80, pathways=[bad_route, good_route])

    assert out["pathway_count"] >= 1
    assert out["pathway"][0] == "R -> Y -> AB2"


def test_activation_constraint_can_invalidate_all_paths():
    ev = SynthesisPathfinder(max_steps=6, max_total_activation=0.05, max_total_delta_g=1.0)
    out = ev.evaluate("LiFePO4", -1.10)

    assert out["synthesizable"] is False
    assert out["pathway_count"] == 0
    assert out["pathway"] == []


def test_custom_pathways_must_terminate_at_target():
    ev = SynthesisPathfinder(max_steps=6)
    wrong = {
        "nodes": ["R", "X", "ZZ2"],
        "delta_g": [-0.20, -0.20],
        "activation": [0.10, 0.10],
        "uncertainty": [0.01, 0.01],
    }
    out = ev.evaluate("AB2", -0.9, pathways=[wrong])

    assert out["synthesizable"] is False
    assert out["pathway_count"] == 0


def test_energy_prior_stable_under_extreme_inputs():
    ev = SynthesisPathfinder(max_steps=6)

    very_low = ev.evaluate("LiFePO4", -1e6)
    very_high = ev.evaluate("LiFePO4", 1e6)

    assert 0.0 <= very_low["score"] <= 1.0
    assert 0.0 <= very_high["score"] <= 1.0
    assert very_low["score"] >= very_high["score"]


def test_empty_result_metrics_are_json_friendly():
    ev = SynthesisPathfinder(max_steps=6, max_total_activation=0.0, max_total_delta_g=0.0)
    out = ev.evaluate("LiFePO4", -1.0)

    assert out["metrics"]["scalar_penalty"] is None
    assert isinstance(out["metrics"]["decision_threshold"], float)


def test_adaptive_threshold_increases_with_path_uncertainty():
    ev = SynthesisPathfinder(
        max_steps=6,
        score_threshold=0.10,
        threshold_uncertainty_weight=0.8,
        threshold_step_weight=0.0,
    )
    low_uncertainty = {
        "nodes": ["R", "I", "AB2"],
        "delta_g": [-0.50, -0.35],
        "activation": [0.10, 0.10],
        "uncertainty": [0.01, 0.01],
    }
    high_uncertainty = {
        "nodes": ["R", "I", "AB2"],
        "delta_g": [-0.50, -0.35],
        "activation": [0.10, 0.10],
        "uncertainty": [0.30, 0.30],
    }

    low = ev.evaluate("AB2", -0.8, pathways=[low_uncertainty])
    high = ev.evaluate("AB2", -0.8, pathways=[high_uncertainty])

    assert high["metrics"]["decision_threshold"] > low["metrics"]["decision_threshold"]


def test_invalid_config_inputs_are_sanitized():
    ev = SynthesisPathfinder(
        max_pathways=0,
        max_steps="bad",  # type: ignore[arg-type]
        max_subset_nodes=-1,
        max_path_expansions=-100,
        max_rank_candidates=0,
        max_total_activation=float("nan"),
        max_total_delta_g=float("inf"),
        score_threshold=float("nan"),
        risk_aversion=float("nan"),
        objective_weights={
            "thermo": float("nan"),
            "activation": -1.0,
            "steps": "oops",  # type: ignore[dict-item]
            "risk": 2.0,
        },
        allow_jump_edges="yes",  # type: ignore[arg-type]
        use_adaptive_threshold="no",  # type: ignore[arg-type]
        threshold_min=0.8,
        threshold_max=0.2,
        threshold_uncertainty_weight=float("nan"),
        threshold_step_weight=-0.2,
        energy_prior_center=float("nan"),
        energy_prior_temperature=0.0,
    )

    assert ev.max_pathways >= 1
    assert ev.max_steps >= 2
    assert ev.max_subset_nodes >= 8
    assert ev.max_path_expansions >= 100
    assert ev.max_rank_candidates >= 16
    assert ev.risk_aversion > 0.0
    assert 0.0 <= ev.score_threshold <= 1.0
    assert 0.0 <= ev.threshold_min <= ev.threshold_max <= 1.0
    assert ev.threshold_uncertainty_weight >= 0.0
    assert ev.threshold_step_weight >= 0.0
    assert ev.energy_prior_temperature > 0.0
    assert math.isclose(sum(ev.objective_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert all(v >= 0.0 for v in ev.objective_weights.values())


def test_integer_controls_reject_bool_and_fractional_inputs():
    ev = SynthesisPathfinder(
        max_pathways=False,  # type: ignore[arg-type]
        max_steps=3.7,  # type: ignore[arg-type]
        max_subset_nodes="20.4",  # type: ignore[arg-type]
        max_path_expansions=800.2,  # type: ignore[arg-type]
    )

    assert ev.max_pathways == 5
    assert ev.max_steps == 6
    assert ev.max_subset_nodes == 96
    assert ev.max_path_expansions == 120000

    parsed = SynthesisPathfinder(max_pathways="2", max_steps="4")
    assert parsed.max_pathways == 2
    assert parsed.max_steps == 4


def test_nonfinite_energy_inputs_produce_finite_scores():
    ev = SynthesisPathfinder(max_steps=6)
    out_nan = ev.evaluate("LiFePO4", float("nan"))
    out_inf = ev.evaluate("LiFePO4", float("inf"))

    assert 0.0 <= out_nan["score"] <= 1.0
    assert 0.0 <= out_inf["score"] <= 1.0
