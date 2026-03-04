"""Runtime-stability tests for relaxation timeout/retry/circuit breaker."""

from __future__ import annotations

import time

from atlas.active_learning.controller import DiscoveryController
from atlas.active_learning.policy_state import PolicyState


class _FakeComposition:
    def __init__(self, formula: str):
        self.reduced_formula = formula


class _FakeStructure:
    def __init__(self, formula: str = "SiO2"):
        self.composition = _FakeComposition(formula)
        self.num_sites = 4


class _SlowRelaxer:
    def relax_structure(self, _struct, steps=100):
        del steps
        time.sleep(0.05)
        return {"relaxed_structure": _struct, "energy_per_atom": -1.0, "converged": True}

    def score_stability(self, _struct):
        return 0.9


class _ErrorRelaxer:
    def relax_structure(self, _struct, steps=100):
        del steps
        raise RuntimeError("boom")

    def score_stability(self, _struct):
        return 0.0


def _runtime_stub(relaxer):
    c = DiscoveryController.__new__(DiscoveryController)
    c.relaxer = relaxer
    c.relax_timeout_sec = 0.01
    c.relax_max_retries = 0
    c.relax_circuit_breaker_failures = 1
    c.relax_circuit_breaker_cooldown_iters = 2
    c.policy_state = PolicyState()
    c.iteration = 1
    c._last_relax_stats = {}
    return c


def test_relax_timeout_sets_bucket_and_penalizes_stability():
    controller = _runtime_stub(_SlowRelaxer())
    raw = [{"structure": _FakeStructure("A"), "method": "substitute", "topo_score": 0.3}]

    out = controller._relax_candidates(raw)

    assert len(out) == 1
    assert out[0].stability_score == 0.0
    buckets = controller._last_relax_stats.get("buckets", {})
    assert buckets.get("timeout", 0) >= 1


def test_relax_circuit_breaker_opens_after_failures():
    controller = _runtime_stub(_ErrorRelaxer())
    raw = [{"structure": _FakeStructure("B"), "method": "substitute", "topo_score": 0.3}]

    controller._relax_candidates(raw)
    assert controller.policy_state.relax_circuit_open_until_iter >= controller.iteration + 1

    controller._relax_candidates(raw)
    buckets = controller._last_relax_stats.get("buckets", {})
    assert buckets.get("circuit_open", 0) >= 1
