"""Tests for atlas.utils.structure module."""

import numpy as np
import pytest


@pytest.fixture
def si_structure():
    """Create a simple Si diamond structure for testing."""
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(5.43)
    structure = Structure(
        lattice=lattice,
        species=["Si", "Si"],
        coords=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    return structure


@pytest.fixture
def bi2se3_structure():
    """Create a Bi2Se3 structure for testing (heavy elements)."""
    from pymatgen.core import Lattice, Structure

    # Simplified rhombohedral Bi2Se3
    lattice = Lattice.hexagonal(4.14, 28.64)
    species = ["Bi", "Bi", "Se", "Se", "Se"]
    coords = [
        [0.0, 0.0, 0.4],
        [0.0, 0.0, 0.6],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.21],
        [0.0, 0.0, 0.79],
    ]
    return Structure(lattice, species, coords)


@pytest.fixture
def empty_structure():
    from pymatgen.core import Lattice, Structure

    return Structure(lattice=Lattice.cubic(3.0), species=[], coords=[])


def test_pymatgen_to_ase(si_structure):
    """Conversion to ASE should preserve atom count and PBC."""
    from atlas.utils.structure import pymatgen_to_ase

    atoms = pymatgen_to_ase(si_structure)
    assert len(atoms) == 2
    assert list(atoms.get_chemical_symbols()) == ["Si", "Si"]
    assert all(atoms.pbc)


def test_ase_to_pymatgen(si_structure):
    """Conversion from ASE back to pymatgen should work."""
    from atlas.utils.structure import ase_to_pymatgen, pymatgen_to_ase

    atoms = pymatgen_to_ase(si_structure)
    struct = ase_to_pymatgen(atoms)
    assert len(struct) == 2
    assert str(struct[0].specie) == "Si"


def test_roundtrip_conversion(si_structure):
    """Round-trip pymatgen → ASE → pymatgen should preserve structure."""
    from atlas.utils.structure import ase_to_pymatgen, pymatgen_to_ase

    struct_out = ase_to_pymatgen(pymatgen_to_ase(si_structure))

    assert len(struct_out) == len(si_structure)
    # Lattice should be very close
    np.testing.assert_allclose(
        struct_out.lattice.matrix,
        si_structure.lattice.matrix,
        atol=1e-6,
    )


def test_structure_from_dict(si_structure):
    """structure_from_dict should reconstruct from dict."""
    from atlas.utils.structure import structure_from_dict

    d = si_structure.as_dict()
    reconstructed = structure_from_dict(d)
    assert len(reconstructed) == len(si_structure)
    assert reconstructed.composition == si_structure.composition


def test_get_element_info_silicon(si_structure):
    """get_element_info should identify Si as not heavy."""
    from atlas.utils.structure import get_element_info

    info = get_element_info(si_structure)
    assert "Si" in info["elements"]
    assert info["num_elements"] == 1
    assert info["has_heavy_elements"] is False
    assert info["max_atomic_number"] == 14  # Si


def test_get_element_info_bi2se3(bi2se3_structure):
    """get_element_info should identify Bi as heavy."""
    from atlas.utils.structure import get_element_info

    info = get_element_info(bi2se3_structure)
    assert info["has_heavy_elements"] is True
    assert "Bi" in info["heavy_elements"]
    assert info["max_atomic_number"] == 83  # Bi


def test_compute_structural_features(si_structure):
    """compute_structural_features should return expected keys."""
    from atlas.utils.structure import compute_structural_features

    feats = compute_structural_features(si_structure)
    assert "volume_per_atom" in feats
    assert "density" in feats
    assert "space_group_number" in feats
    assert "crystal_system" in feats
    assert "num_sites" in feats
    assert feats["num_sites"] == 2
    assert feats["volume_per_atom"] > 0
    assert feats["density"] > 0


def test_get_element_info_empty_structure(empty_structure):
    from atlas.utils.structure import get_element_info

    info = get_element_info(empty_structure)
    assert info["num_elements"] == 0
    assert info["max_atomic_number"] == 0
    assert info["has_heavy_elements"] is False


def test_compute_structural_features_empty_structure(empty_structure):
    from atlas.utils.structure import compute_structural_features

    feats = compute_structural_features(empty_structure)
    assert feats["num_sites"] == 0
    assert feats["volume_per_atom"] == 0.0
    assert feats["density"] == 0.0


def test_compute_structural_features_is_deterministic_for_large_structure():
    from pymatgen.core import Lattice, Structure

    from atlas.utils.structure import compute_structural_features

    lattice = Lattice.cubic(10.0)
    species = ["Si"] * 60
    coords = [[(i % 5) / 5, ((i // 5) % 4) / 4, (i // 20) / 3] for i in range(60)]
    structure = Structure(lattice=lattice, species=species, coords=coords)

    f1 = compute_structural_features(structure)
    f2 = compute_structural_features(structure)
    assert f1["avg_nn_distance"] == f2["avg_nn_distance"]


def test_get_element_info_handles_oxidation_and_disorder():
    from pymatgen.core import Lattice, Structure

    from atlas.utils.structure import get_element_info

    structure = Structure(
        lattice=Lattice.cubic(4.2),
        species=[{"Fe2+": 0.5, "Mn2+": 0.5}, "O2-"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    info = get_element_info(structure)

    assert info["elements"] == ["Fe", "Mn", "O"]
    assert info["num_elements"] == 3
    assert info["atomic_numbers"] == [26, 25, 8]
    assert info["has_heavy_elements"] is False


def test_sample_site_indices_handles_non_positive_max_samples():
    from atlas.utils.structure import _sample_site_indices

    assert _sample_site_indices(10, max_samples=0).size == 0
    with pytest.raises(ValueError, match="max_samples"):
        _sample_site_indices(10, max_samples=-5)


def test_sample_site_indices_rejects_non_integer_payloads():
    from atlas.utils.structure import _sample_site_indices

    with pytest.raises(ValueError, match="n_sites"):
        _sample_site_indices(10.5, max_samples=4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_samples"):
        _sample_site_indices(10, max_samples=True)  # type: ignore[arg-type]


def test_compute_structural_features_uses_neighbor_radius_fallback():
    from pymatgen.core import Lattice, Structure

    from atlas.utils.structure import compute_structural_features

    # Nearest neighbor is >4A but <8A; fallback radius should recover non-zero distance.
    structure = Structure(
        lattice=Lattice.cubic(9.0),
        species=["Si", "Si"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
    )
    feats = compute_structural_features(structure)
    assert feats["avg_nn_distance"] > 0.0


def test_get_standardized_structure_accepts_bool_like_string(si_structure):
    from atlas.utils.structure import get_standardized_structure

    standardized = get_standardized_structure(si_structure, primitive="false")  # type: ignore[arg-type]
    assert len(standardized) > 0


def test_get_standardized_structure_rejects_invalid_bool_like(si_structure):
    from atlas.utils.structure import get_standardized_structure

    with pytest.raises(ValueError, match="primitive"):
        get_standardized_structure(si_structure, primitive="maybe")  # type: ignore[arg-type]
