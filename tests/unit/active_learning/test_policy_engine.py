"""Tests for policy-engine behavior (legacy/cmoeic)."""

from __future__ import annotations

from types import SimpleNamespace

from atlas.active_learning.controller import Candidate
from atlas.active_learning.policy_engine import PolicyEngine
from atlas.active_learning.policy_state import ActiveLearningPolicyConfig, PolicyState


class _DummyController:
    def __init__(self):
        self.all_candidates = [
            SimpleNamespace(energy_mean=-1.0, energy_per_atom=-1.1, energy_std=0.1),
            SimpleNamespace(energy_mean=-0.8, energy_per_atom=-0.9, energy_std=0.1),
            SimpleNamespace(energy_mean=-0.7, energy_per_atom=-0.72, energy_std=0.08),
            SimpleNamespace(energy_mean=-0.9, energy_per_atom=-0.95, energy_std=0.05),
            SimpleNamespace(energy_mean=-0.6, energy_per_atom=-0.62, energy_std=0.04),
            SimpleNamespace(energy_mean=-0.5, energy_per_atom=-0.55, energy_std=0.05),
            SimpleNamespace(energy_mean=-1.2, energy_per_atom=-1.25, energy_std=0.1),
            SimpleNamespace(energy_mean=-1.1, energy_per_atom=-1.18, energy_std=0.08),
            SimpleNamespace(energy_mean=-0.4, energy_per_atom=-0.45, energy_std=0.07),
            SimpleNamespace(energy_mean=-0.3, energy_per_atom=-0.35, energy_std=0.08),
            SimpleNamespace(energy_mean=-0.2, energy_per_atom=-0.3, energy_std=0.09),
            SimpleNamespace(energy_mean=-0.1, energy_per_atom=-0.2, energy_std=0.09),
        ]
        self.use_synthesis_objective = True
        self.iteration = 3
        self.relaxer = None

    def _score_and_select_legacy(self, candidates, n_top):
        out = list(candidates)
        out.sort(key=lambda c: c.formula)
        return out

    def _is_duplicate_structure(self, _structure):
        return False

    def _stability_component(self, candidate):
        return float(candidate.stability_score)

    def _apply_synthesis_objective(self, candidates):
        scores = []
        for c in candidates:
            if c.formula == "A":
                c.synthesis_score = 0.9
            else:
                c.synthesis_score = 0.4
            c.synthesis_feasibility = 1.0 if c.synthesis_score >= 0.5 else 0.0
            scores.append(c.synthesis_score)
        return scores

    def _estimate_ood_scores(self, candidates):
        return [0.05 if c.formula == "A" else 0.95 for c in candidates]

    def _apply_pareto_rank_bonus(self, *_args, **_kwargs):
        return None

    def _apply_pareto_hv_bonus(self, *_args, **_kwargs):
        return None

    def _select_top_diverse(self, candidates, n_top, *, objective_map=None):
        del objective_map
        return list(candidates[:n_top])

    def _finalize_ranked_candidates(self, candidates, top):
        selected_ids = {id(c) for c in top}
        return top + [c for c in candidates if id(c) not in selected_ids]


def test_policy_engine_legacy_delegates_to_controller_legacy():
    controller = _DummyController()
    cfg = ActiveLearningPolicyConfig(policy_name="legacy")
    engine = PolicyEngine(config=cfg, state=PolicyState())

    c1 = Candidate(structure=None, formula="B")
    c2 = Candidate(structure=None, formula="A")
    ranked = engine.score_and_select(controller, [c1, c2], n_top=1)

    assert [c.formula for c in ranked] == ["A", "B"]


def test_policy_engine_cmoeic_sets_artifact_fields_and_risk_gate():
    controller = _DummyController()
    cfg = ActiveLearningPolicyConfig(
        policy_name="cmoeic",
        risk_mode="hard",
        cost_aware=True,
        calibration_window=64,
        ood_gate_threshold=0.8,
        conformal_gate_threshold=0.95,
        ood_combination="or",
    )
    engine = PolicyEngine(config=cfg, state=PolicyState())

    a = Candidate(
        structure=None,
        formula="A",
        topo_probability=0.95,
        stability_score=0.8,
        heuristic_topo_score=0.2,
        energy_mean=-1.0,
        energy_std=0.1,
    )
    b = Candidate(
        structure=None,
        formula="B",
        topo_probability=0.95,
        stability_score=0.9,
        heuristic_topo_score=0.2,
        energy_mean=-1.0,
        energy_std=0.1,
    )

    ranked = engine.score_and_select(controller, [a, b], n_top=2)

    by_formula = {c.formula: c for c in ranked}
    assert by_formula["A"].calibrated_std is not None
    assert by_formula["A"].conformal_radius >= 0.0
    assert by_formula["A"].estimated_cost > 0.0
    assert by_formula["A"].gain_per_cost != 0.0

    assert by_formula["B"].reject_reason == "risk_gate"
    assert by_formula["B"].acquisition_value < by_formula["A"].acquisition_value
