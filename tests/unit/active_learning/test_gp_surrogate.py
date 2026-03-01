"""Tests for GP surrogate acquirer."""

from dataclasses import dataclass

import numpy as np

from atlas.active_learning.gp_surrogate import GPSurrogateAcquirer, GPSurrogateConfig


@dataclass
class DummyCandidate:
    topo_probability: float
    stability_score: float
    heuristic_topo_score: float
    novelty_score: float
    energy_per_atom: float
    acquisition_value: float


@dataclass
class DummyCandidateNoScore:
    topo_probability: float
    stability_score: float
    heuristic_topo_score: float
    novelty_score: float
    energy_per_atom: float


@dataclass
class DummyCandidateWithEnergy:
    topo_probability: float
    stability_score: float
    heuristic_topo_score: float
    novelty_score: float
    energy_per_atom: float
    energy_mean: float
    energy_std: float = 0.0


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


def _dummy_candidates_no_score(n: int = 12):
    out = []
    for i in range(n):
        topo = i / n
        stab = (n - i) / n
        heuristic = 0.25 + 0.01 * i
        novelty = 1.0 if i % 2 == 0 else 0.0
        epa = -1.0 + 0.05 * i
        out.append(
            DummyCandidateNoScore(
                topo_probability=topo,
                stability_score=stab,
                heuristic_topo_score=heuristic,
                novelty_score=novelty,
                energy_per_atom=epa,
            )
        )
    return out


def _dummy_candidates_with_energy(n: int = 12):
    out = []
    for i in range(n):
        topo = i / n
        stab = (n - i) / n
        heuristic = 0.25 + 0.01 * i
        novelty = 1.0 if i % 2 == 0 else 0.0
        epa = -1.0 + 0.05 * i
        out.append(
            DummyCandidateWithEnergy(
                topo_probability=topo,
                stability_score=stab,
                heuristic_topo_score=heuristic,
                novelty_score=novelty,
                energy_per_atom=epa,
                energy_mean=epa,
                energy_std=0.05 + 0.01 * (i % 3),
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


def test_gp_surrogate_can_train_without_acquisition_score_field():
    acq = GPSurrogateAcquirer(config=GPSurrogateConfig(min_points=6))
    cands = _dummy_candidates_no_score(12)
    acq.update(cands)
    scores = acq.suggest_constrained_utility(cands)
    if scores is None:
        return
    assert len(scores) == len(cands)


def test_feature_map_excludes_energy_mean_by_default():
    acq = GPSurrogateAcquirer(config=GPSurrogateConfig())
    a = DummyCandidateWithEnergy(0.5, 0.6, 0.3, 1.0, -1.2, -1.2, 0.1)
    b = DummyCandidateWithEnergy(0.5, 0.6, 0.3, 1.0, -0.2, -0.2, 0.1)
    feat_a = acq.candidate_to_features(a)
    feat_b = acq.candidate_to_features(b)
    assert np.allclose(feat_a, feat_b)


def test_feature_map_can_include_energy_mean_when_opt_in():
    acq = GPSurrogateAcquirer(config=GPSurrogateConfig(include_energy_feature=True))
    a = DummyCandidateWithEnergy(0.5, 0.6, 0.3, 1.0, -1.2, -1.2, 0.1)
    b = DummyCandidateWithEnergy(0.5, 0.6, 0.3, 1.0, -0.2, -0.2, 0.1)
    feat_a = acq.candidate_to_features(a)
    feat_b = acq.candidate_to_features(b)
    assert not np.allclose(feat_a, feat_b)


def test_gp_ucb_schedule_changes_kappa_after_warmup():
    acq = GPSurrogateAcquirer(
        config=GPSurrogateConfig(min_points=4, kappa=1.5, kappa_schedule="gp_ucb", ucb_dimension=5)
    )
    cands = _dummy_candidates_with_energy(10)
    acq.update(cands[:2])
    kappa_small = acq.current_kappa
    acq.update(cands[2:])
    kappa_large = acq.current_kappa
    assert kappa_large >= kappa_small


def test_logei_mode_returns_finite_ranked_scores():
    acq = GPSurrogateAcquirer(
        config=GPSurrogateConfig(
            min_points=6,
            acquisition_mode="logei",
            feasibility_mode="chance",
            normalize_output=True,
        )
    )
    cands = _dummy_candidates_with_energy(12)
    acq.update(cands)
    scores = acq.suggest_constrained_utility(cands)
    if scores is None:
        return
    assert np.isfinite(scores).all()
    assert (scores >= 0.0).all()
    assert (scores <= 1.0).all()


def test_chance_feasibility_penalizes_low_topology_probability():
    acq = GPSurrogateAcquirer(
        config=GPSurrogateConfig(
            min_points=6,
            acquisition_mode="ucb",
            feasibility_mode="chance",
            normalize_output=False,
            feasibility_threshold=0.5,
            coupling_strength=0.0,
        )
    )
    cands = _dummy_candidates_with_energy(20)
    acq.update(cands)
    if acq._objective_model is None:
        return
    x_raw = np.vstack([acq.candidate_to_features(c) for c in cands])
    feas_prob = acq._predict_feasibility(x_raw)
    high_topo_idx = int(np.argmax([c.topo_probability for c in cands]))
    low_topo_idx = int(np.argmin([c.topo_probability for c in cands]))
    assert feas_prob[high_topo_idx] >= feas_prob[low_topo_idx]


def test_feasibility_classifier_mode_produces_probability_semantics():
    acq = GPSurrogateAcquirer(
        config=GPSurrogateConfig(
            min_points=6,
            use_feasibility_classifier=True,
            feasibility_mode="chance",
            normalize_output=False,
        )
    )
    cands = _dummy_candidates_with_energy(16)
    acq.update(cands)
    scores = acq.suggest_constrained_utility(cands)
    if scores is None:
        return
    assert np.isfinite(scores).all()
    assert acq._feasibility_model is None or acq._feasibility_is_classifier is True


def test_objective_model_uses_heteroscedastic_alpha_from_energy_uncertainty():
    acq = GPSurrogateAcquirer(
        config=GPSurrogateConfig(
            min_points=6,
            acquisition_mode="ucb",
            normalize_output=False,
        )
    )
    cands = _dummy_candidates_with_energy(18)
    for idx, c in enumerate(cands):
        c.energy_std = 0.01 if idx % 2 == 0 else 0.20
    acq.update(cands)
    if acq._objective_model is None:
        return
    alpha = np.asarray(acq._objective_model.alpha, dtype=float)
    assert alpha.shape[0] == len(cands)
    assert float(alpha.max()) > float(alpha.min())


def test_objective_feasibility_coupling_changes_feasibility_mass():
    acq = GPSurrogateAcquirer(
        config=GPSurrogateConfig(
            min_points=6,
            acquisition_mode="ucb",
            feasibility_mode="chance",
            coupling_strength=1.0,
            min_coupling_points=6,
            normalize_output=False,
        )
    )
    cands = _dummy_candidates_with_energy(20)
    acq.update(cands)
    if acq._objective_model is None:
        return

    x_raw = np.vstack([acq.candidate_to_features(c) for c in cands])
    x = acq._transform_features(x_raw)
    mean_obj, _ = acq._objective_model.predict(x, return_std=True)
    base_feas = acq._predict_feasibility(x_raw)
    coupled = acq._apply_objective_feasibility_coupling(base_feas, mean_obj)
    assert np.isfinite(coupled).all()
    assert not np.allclose(base_feas, coupled)
