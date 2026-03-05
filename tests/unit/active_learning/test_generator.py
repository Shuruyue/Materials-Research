"""Tests for atlas.active_learning.generator module."""

import numpy as np
import pytest


@pytest.fixture
def seed_structures():
    """Create a set of seed structures for testing."""
    from pymatgen.core import Lattice, Structure

    structures = []

    # Bi2Se3-type (rhombohedral)
    lattice = Lattice.hexagonal(4.14, 28.64)
    species = ["Bi", "Bi", "Se", "Se", "Se"]
    coords = [
        [0.0, 0.0, 0.4],
        [0.0, 0.0, 0.6],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.21],
        [0.0, 0.0, 0.79],
    ]
    structures.append(Structure(lattice, species, coords))

    # SnTe-type (rocksalt)
    lattice = Lattice.cubic(6.32)
    species = ["Sn", "Te"]
    coords = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    structures.append(Structure(lattice, species, coords))

    return structures


@pytest.fixture
def generator(seed_structures):
    """StructureGenerator with seed structures."""
    from atlas.active_learning.generator import StructureGenerator
    return StructureGenerator(seed_structures=seed_structures, rng_seed=42)


def test_generator_init(generator, seed_structures):
    """Generator should initialize with seeds."""
    assert len(generator.seeds) == len(seed_structures)
    assert len(generator.generated) == 0


def test_generate_batch(generator):
    """generate_batch should produce the requested number of candidates."""
    candidates = generator.generate_batch(n_candidates=10)
    assert len(candidates) <= 10
    assert len(candidates) > 0

    # Each candidate should have required keys
    for cand in candidates:
        assert "structure" in cand
        assert "method" in cand
        assert "topo_score" in cand
        assert "generator_score" in cand
        assert "novelty_score" in cand
        assert "feasibility_score" in cand
        assert "selection_utility" in cand
        assert "adaptive_weights" in cand


def test_substitute_method(generator):
    """Substitution should produce valid structures."""
    candidates = generator.generate_batch(
        n_candidates=5, methods=["substitute"]
    )
    for cand in candidates:
        assert cand["method"] == "substitute"
        struct = cand["structure"]
        assert len(struct) > 0


def test_strain_method(generator):
    """Strain should produce valid structures with modified lattice."""
    candidates = generator.generate_batch(
        n_candidates=5, methods=["strain"]
    )
    for cand in candidates:
        assert cand["method"] == "strain"
        struct = cand["structure"]
        assert len(struct) > 0
        assert struct.volume > 0


def test_mix_method(generator):
    """Mix should produce valid structures."""
    candidates = generator.generate_batch(
        n_candidates=5, methods=["mix"]
    )
    # mix may fail if element counts don't match; at least some should succeed
    for cand in candidates:
        assert cand["method"] in ("mix", "substitute")  # fallback to substitute
        struct = cand["structure"]
        assert len(struct) > 0


def test_heuristic_topo_score_range(generator):
    """Score should be in [0, 1]."""
    candidates = generator.generate_batch(n_candidates=20)
    for cand in candidates:
        assert 0.0 <= cand["topo_score"] <= 1.0
        assert cand["generator_score"] >= 0.0
        assert 0.0 <= cand["novelty_score"] <= 1.0
        assert 0.0 <= cand["feasibility_score"] <= 1.0


def test_add_seeds(generator):
    """add_seeds should increase seed count."""
    from pymatgen.core import Lattice, Structure

    n_before = len(generator.seeds)
    new_struct = Structure(
        Lattice.cubic(4.0), ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    generator.add_seeds([new_struct])
    assert len(generator.seeds) == n_before + 1


def test_get_top_candidates(generator):
    """get_top_candidates should return sorted results."""
    generator.generate_batch(n_candidates=20)
    top = generator.get_top_candidates(n=5)
    assert len(top) <= 5

    # Should be sorted by utility descending after diversity-aware ranking.
    scores = [c.get("selection_utility", c.get("generator_score", c["topo_score"])) for c in top]
    assert scores == sorted(scores, reverse=True)


def test_adaptive_weights_are_normalized(generator):
    generator.generate_batch(n_candidates=12)
    weights = generator.last_adaptive_weights
    assert set(weights.keys()) == {"topo", "novelty", "feasibility", "strain"}
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert all(v >= 0.0 for v in weights.values())


def test_substitution_stats_updates_after_generation(generator):
    generator.generate_batch(n_candidates=8, methods=["substitute"])
    assert len(generator.substitution_stats) > 0
    any_count = any(v.get("count", 0.0) > 0.0 for v in generator.substitution_stats.values())
    assert any_count


def test_structure_fingerprint_distinguishes_polymorph_like_variants():
    from pymatgen.core import Lattice, Structure

    from atlas.active_learning.generator import _structure_fingerprint

    cubic = Structure(Lattice.cubic(6.32), ["Sn", "Te"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    tetr = Structure(Lattice.tetragonal(4.5, 7.1), ["Sn", "Te"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    f1 = _structure_fingerprint(cubic)
    f2 = _structure_fingerprint(tetr)
    assert f1.shape == f2.shape
    assert not (f1 == f2).all()


def test_no_seeds_raises():
    """Generator without seeds should raise on generate."""
    from atlas.active_learning.generator import StructureGenerator

    gen = StructureGenerator(seed_structures=[], rng_seed=42)
    with pytest.raises(ValueError, match="No seed"):
        gen.generate_batch(n_candidates=5)


def test_softmax_handles_nonfinite_logits():
    from atlas.active_learning.generator import _softmax

    probs = _softmax(np.asarray([0.0, np.nan, np.inf, -np.inf], dtype=float))
    assert np.isfinite(probs).all()
    assert probs.shape == (4,)
    assert probs.sum() == pytest.approx(1.0, abs=1e-12)
    assert (probs >= 0.0).all()


def test_charge_neutrality_score_is_stable_under_monte_carlo_fallback():
    from pymatgen.core import Lattice, Structure

    from atlas.active_learning.generator import _charge_neutrality_score

    struct = Structure(
        Lattice.cubic(4.2),
        ["Fe", "O", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )
    s1 = _charge_neutrality_score(struct, max_combinations=1)
    s2 = _charge_neutrality_score(struct, max_combinations=1)
    assert s1 == pytest.approx(s2, abs=1e-12)
    assert 0.0 <= s1 <= 1.0


def test_generate_batch_single_worker_fallback(generator):
    generator.n_workers = 1
    cands = generator.generate_batch(n_candidates=8, methods=["substitute", "strain"])
    assert 0 < len(cands) <= 8


def test_generator_init_sanitizes_nonfinite_hyperparameters(seed_structures):
    from atlas.active_learning.generator import StructureGenerator

    gen = StructureGenerator(
        seed_structures=seed_structures,
        rng_seed=7.9,  # type: ignore[arg-type]
        w_topo=float("nan"),
        w_novelty=float("inf"),
        w_feasibility=float("-inf"),
        w_strain=float("nan"),
        archive_limit=float("nan"),  # type: ignore[arg-type]
        substitution_stat_decay=float("nan"),
    )
    assert gen.rng_seed == 42
    assert gen.archive_limit >= 32
    assert np.isfinite(gen.substitution_stat_decay)
    assert 0.90 <= gen.substitution_stat_decay <= 1.0
    weights = gen._normalize_weights(gen.weights)
    assert np.isfinite(np.asarray(list(weights.values()), dtype=float)).all()
    assert pytest.approx(sum(weights.values()), abs=1e-12) == 1.0


def test_normalize_weights_handles_nonfinite_entries(generator):
    weights = generator._normalize_weights({"topo": float("nan"), "novelty": float("inf"), "x": -5.0})
    assert np.isfinite(np.asarray(list(weights.values()), dtype=float)).all()
    assert pytest.approx(sum(weights.values()), abs=1e-12) == 1.0
    assert all(v >= 0.0 for v in weights.values())
