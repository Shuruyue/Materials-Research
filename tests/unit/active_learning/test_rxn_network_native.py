"""Algorithm-focused tests for native reaction-network evaluator."""

from dataclasses import dataclass

import numpy as np

from atlas.active_learning.rxn_network_native import NativeReactionNetworkEvaluator


@dataclass
class _FakeReaction:
    name: str
    energy_per_atom: float
    energy_uncertainty_per_atom: float = 0.0
    products: tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.name


@dataclass
class _FakePath:
    reactions: list[_FakeReaction]
    costs: list[float]
    intermediates: set[str]

    @property
    def average_cost(self) -> float:
        return sum(self.costs) / max(1, len(self.costs))

    @property
    def energy_per_atom(self) -> float:
        return sum(r.energy_per_atom for r in self.reactions)

    @property
    def products(self) -> set[str]:
        out: set[str] = set()
        for rxn in self.reactions:
            out.update(rxn.products)
        return out


def test_fallback_conservative_without_context():
    ev = NativeReactionNetworkEvaluator(fallback_mode="conservative")
    out = ev.evaluate("SiO2", -1.0)
    assert out["synthesizable"] is False
    assert out["pathway"] == []
    assert out["pathway_count"] == 0


def test_alias_soft_mish_maps_to_softplus():
    ev = NativeReactionNetworkEvaluator(cost_function="soft_mish")
    assert ev.cost_function == "softplus"


def test_safe_float_rejects_non_finite_values():
    assert NativeReactionNetworkEvaluator._safe_float(float("nan"), default=0.7) == 0.7
    assert NativeReactionNetworkEvaluator._safe_float(float("inf"), default=1.2) == 1.2


def test_normalize_weights_ignores_unknown_and_non_finite_values():
    weights = NativeReactionNetworkEvaluator._normalize_weights(
        {
            "average_cost": float("nan"),
            "driving_force": 0.5,
            "unknown_metric": 10.0,
        }
    )
    assert set(weights) == {
        "average_cost",
        "driving_force",
        "num_steps",
        "uncertainty",
        "intermediate_complexity",
        "entropic_risk",
    }
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert "unknown_metric" not in weights


def test_entropic_risk_is_finite_under_extreme_scales():
    score = NativeReactionNetworkEvaluator._entropic_risk(np.asarray([0.0, 1.0e6], dtype=float), risk_aversion=500.0)
    assert np.isfinite(score)
    assert score >= 0.0


def test_pareto_ranking_prefers_better_cost_and_driving_force():
    ev = NativeReactionNetworkEvaluator(
        max_num_pathways=2,
        objective_weights={
            "average_cost": 0.55,
            "driving_force": 0.35,
            "num_steps": 0.05,
            "uncertainty": 0.03,
            "intermediate_complexity": 0.01,
            "entropic_risk": 0.01,
        },
    )
    good = _FakePath(
        reactions=[
            _FakeReaction("A->B", -0.60, products=("B",)),
            _FakeReaction("B->T", -0.40, products=("T",)),
        ],
        costs=[0.08, 0.12],
        intermediates={"B"},
    )
    poor = _FakePath(
        reactions=[
            _FakeReaction("A->C", -0.20, products=("C",)),
            _FakeReaction("C->T", -0.10, products=("T",)),
        ],
        costs=[0.35, 0.30],
        intermediates={"C"},
    )
    out = ev.evaluate("T", -0.4, pathways=[poor, good], precursors=["A"])
    assert out["synthesizable"] is True
    assert out["pathway_count"] == 2
    assert "A->B" in out["best_path_repr"]


def test_entropic_risk_penalizes_high_variance_cost_path():
    ev = NativeReactionNetworkEvaluator(
        max_num_pathways=2,
        risk_aversion=6.0,
        objective_weights={
            "average_cost": 0.10,
            "driving_force": 0.10,
            "num_steps": 0.00,
            "uncertainty": 0.00,
            "intermediate_complexity": 0.00,
            "entropic_risk": 0.80,
        },
    )
    low_risk = _FakePath(
        reactions=[
            _FakeReaction("R1", -0.20, products=("I",)),
            _FakeReaction("R2", -0.20, products=("T",)),
        ],
        costs=[0.20, 0.20],
        intermediates=set(),
    )
    high_risk = _FakePath(
        reactions=[
            _FakeReaction("R3", -0.20, products=("I",)),
            _FakeReaction("R4", -0.20, products=("T",)),
        ],
        costs=[0.01, 0.39],
        intermediates=set(),
    )
    out = ev.evaluate("T", -0.2, pathways=[high_risk, low_risk], precursors=["A"])
    assert "R1" in out["best_path_repr"]
    assert out["metrics"]["entropic_risk"] <= 0.25


def test_target_validation_rejects_pathways_without_target_product():
    ev = NativeReactionNetworkEvaluator(fallback_mode="conservative")
    wrong = _FakePath(
        reactions=[_FakeReaction("A->B", -0.2, products=("B",)), _FakeReaction("B->C", -0.2, products=("C",))],
        costs=[0.2, 0.2],
        intermediates={"B"},
    )
    out = ev.evaluate("T", -0.3, pathways=[wrong], precursors=["A"])
    assert out["synthesizable"] is False
    assert out["pathway_count"] == 0


def test_path_step_costs_falls_back_to_reaction_energies_when_costs_invalid():
    ev = NativeReactionNetworkEvaluator()
    path = _FakePath(
        reactions=[
            _FakeReaction("R1", -0.30, products=("I",)),
            _FakeReaction("R2", -0.20, products=("T",)),
        ],
        costs=[float("nan"), float("inf")],
        intermediates={"I"},
    )
    out = ev.evaluate("T", -0.5, pathways=[path], precursors=["A"])
    assert out["synthesizable"] is True
    assert np.isfinite(out["metrics"]["entropic_risk"])


def test_integer_and_bool_controls_are_sanitized():
    ev = NativeReactionNetworkEvaluator(
        max_num_pathways=False,  # type: ignore[arg-type]
        k_shortest_paths=3.7,  # type: ignore[arg-type]
        max_num_combos="2.5",  # type: ignore[arg-type]
        chunk_size=float("nan"),
        use_balanced_solver="false",  # type: ignore[arg-type]
        find_intermediate_rxns="no",  # type: ignore[arg-type]
        use_basic_enumerator="0",  # type: ignore[arg-type]
        use_minimize_enumerator="yes",  # type: ignore[arg-type]
        filter_interdependent="off",  # type: ignore[arg-type]
        fallback_mode="unknown",
    )
    assert ev.max_num_pathways == 5
    assert ev.k_shortest_paths == 25
    assert ev.max_num_combos == 4
    assert ev.chunk_size == 100000
    assert ev.use_balanced_solver is False
    assert ev.find_intermediate_rxns is False
    assert ev.use_basic_enumerator is False
    assert ev.use_minimize_enumerator is True
    assert ev.filter_interdependent is False
    assert ev.fallback_mode == "conservative"


def test_integer_like_string_controls_are_accepted():
    ev = NativeReactionNetworkEvaluator(max_num_pathways="3", k_shortest_paths="8")
    assert ev.max_num_pathways == 3
    assert ev.k_shortest_paths == 8
