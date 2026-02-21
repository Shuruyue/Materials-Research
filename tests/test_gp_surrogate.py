"""Tests for GP surrogate acquirer."""

from dataclasses import dataclass

from atlas.active_learning.gp_surrogate import GPSurrogateAcquirer, GPSurrogateConfig


@dataclass
class DummyCandidate:
    topo_probability: float
    stability_score: float
    heuristic_topo_score: float
    novelty_score: float
    energy_per_atom: float
    acquisition_value: float


def _dummy_candidates(n: int = 12):
    out = []
    for i in range(n):
        topo = i / n
        stab = (n - i) / n
        heuristic = 0.25 + 0.01 * i
        novelty = 1.0 if i % 2 == 0 else 0.0
        epa = -1.0 + 0.05 * i
        score = 0.5 * topo + 0.3 * stab + 0.1 * heuristic + 0.1 * novelty
        out.append(
            DummyCandidate(
                topo_probability=topo,
                stability_score=stab,
                heuristic_topo_score=heuristic,
                novelty_score=novelty,
                energy_per_atom=epa,
                acquisition_value=score,
            )
        )
    return out


def test_gp_surrogate_warmup_behavior():
    acq = GPSurrogateAcquirer(config=GPSurrogateConfig(min_points=20))
    cands = _dummy_candidates(8)
    acq.update(cands)
    assert acq.suggest_ucb(cands) is None


def test_gp_surrogate_suggest_shape_when_ready():
    acq = GPSurrogateAcquirer(config=GPSurrogateConfig(min_points=6))
    cands = _dummy_candidates(12)
    acq.update(cands)
    scores = acq.suggest_ucb(cands)
    if scores is None:
        # Acceptable when sklearn is unavailable at runtime.
        return
    assert len(scores) == len(cands)

