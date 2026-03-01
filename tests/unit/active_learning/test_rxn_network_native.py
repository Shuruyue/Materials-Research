"""Algorithm-focused tests for native reaction-network evaluator."""

from dataclasses import dataclass

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
