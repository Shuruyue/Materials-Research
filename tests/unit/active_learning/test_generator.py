"""Tests for atlas.active_learning.generator module."""

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

    # Should be sorted by topo_score descending
    scores = [c["topo_score"] for c in top]
    assert scores == sorted(scores, reverse=True)


def test_no_seeds_raises():
    """Generator without seeds should raise on generate."""
    from atlas.active_learning.generator import StructureGenerator

    gen = StructureGenerator(seed_structures=[], rng_seed=42)
    with pytest.raises(ValueError, match="No seed"):
        gen.generate_batch(n_candidates=5)
