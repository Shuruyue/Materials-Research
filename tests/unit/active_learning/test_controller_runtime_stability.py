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


class _FlakyRelaxer:
    def __init__(self, fail_times: int = 1):
        self.fail_times = int(max(0, fail_times))
        self.calls = 0

    def relax_structure(self, _struct, steps=100):
        del steps
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient failure")
        return {"relaxed_structure": _struct, "energy_per_atom": -1.1, "converged": True}

    def score_stability(self, _struct):
        return 0.8


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


def test_relax_retries_with_backoff_and_recovers(monkeypatch):
    relaxer = _FlakyRelaxer(fail_times=1)
    controller = _runtime_stub(relaxer)
    controller.relax_timeout_sec = 0.0
    controller.relax_max_retries = 2
    controller.relax_retry_backoff_sec = 0.02
    controller.relax_retry_backoff_max_sec = 0.02
    controller.relax_retry_jitter = 0.0
    controller._retry_rng = None

    sleep_calls: list[float] = []
    monkeypatch.setattr("atlas.active_learning.controller.time.sleep", lambda sec: sleep_calls.append(float(sec)))

    raw = [{"structure": _FakeStructure("C"), "method": "substitute", "topo_score": 0.3}]
    out = controller._relax_candidates(raw)

    assert len(out) == 1
    assert out[0].stability_score > 0.0
    buckets = controller._last_relax_stats.get("buckets", {})
    assert buckets.get("exception", 0) == 1
    assert controller._last_relax_stats.get("success", 0) == 1
    assert sleep_calls == [0.02]


def test_retry_sleep_seconds_caps_exponential_growth():
    controller = _runtime_stub(_ErrorRelaxer())
    controller.relax_retry_backoff_sec = 0.1
    controller.relax_retry_backoff_max_sec = 0.15
    controller.relax_retry_jitter = 0.0

    assert controller._retry_sleep_seconds(0) == 0.1
    assert controller._retry_sleep_seconds(3) == 0.15


def test_retry_sleep_seconds_jitter_uses_rng_and_clamps_range():
    class _DeterministicRng:
        def uniform(self, lo, hi):
            return (float(lo) + float(hi)) / 2.0

    controller = _runtime_stub(_ErrorRelaxer())
    controller.relax_retry_backoff_sec = 0.1
    controller.relax_retry_backoff_max_sec = 0.2
    controller.relax_retry_jitter = 0.5
    controller._retry_rng = _DeterministicRng()

    # attempt 1 => delay=min(0.2, 0.1*2)=0.2, jitter bounds [0.1, 0.3], midpoint=0.2
    assert controller._retry_sleep_seconds(1) == 0.2
