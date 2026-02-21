"""Tests for atlas.utils.structure module."""

import pytest
import numpy as np


@pytest.fixture
def si_structure():
    """Create a simple Si diamond structure for testing."""
    from pymatgen.core import Structure, Lattice

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
    from pymatgen.core import Structure, Lattice

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


def test_pymatgen_to_ase(si_structure):
    """Conversion to ASE should preserve atom count and PBC."""
    from atlas.utils.structure import pymatgen_to_ase

    atoms = pymatgen_to_ase(si_structure)
    assert len(atoms) == 2
    assert list(atoms.get_chemical_symbols()) == ["Si", "Si"]
    assert all(atoms.pbc)


def test_ase_to_pymatgen(si_structure):
    """Conversion from ASE back to pymatgen should work."""
    from atlas.utils.structure import pymatgen_to_ase, ase_to_pymatgen

    atoms = pymatgen_to_ase(si_structure)
    struct = ase_to_pymatgen(atoms)
    assert len(struct) == 2
    assert str(struct[0].specie) == "Si"


def test_roundtrip_conversion(si_structure):
    """Round-trip pymatgen → ASE → pymatgen should preserve structure."""
    from atlas.utils.structure import pymatgen_to_ase, ase_to_pymatgen

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
