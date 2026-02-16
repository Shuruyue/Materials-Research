"""
Structure Conversion Utilities

Convert between pymatgen, ASE, and MACE data formats.
Provides helpers for structural analysis and featurization.
"""

from typing import Optional
import numpy as np


def pymatgen_to_ase(structure):
    """
    Convert pymatgen Structure → ASE Atoms.

    Args:
        structure: pymatgen Structure object

    Returns:
        ASE Atoms object
    """
    from ase import Atoms

    atoms = Atoms(
        symbols=[str(site.specie) for site in structure],
        positions=[site.coords for site in structure],
        cell=structure.lattice.matrix,
        pbc=True,
    )
    return atoms


def ase_to_pymatgen(atoms):
    """
    Convert ASE Atoms → pymatgen Structure.

    Args:
        atoms: ASE Atoms object

    Returns:
        pymatgen Structure object
    """
    from pymatgen.core import Structure, Lattice

    lattice = Lattice(atoms.cell[:])
    species = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    return Structure(
        lattice=lattice,
        species=species,
        coords=positions,
        coords_are_cartesian=True,
    )


def structure_from_dict(d: dict):
    """Reconstruct a pymatgen Structure from its dict representation."""
    from pymatgen.core import Structure
    return Structure.from_dict(d)


def get_element_info(structure) -> dict:
    """
    Get element-level information about a structure.

    Returns:
        Dict with keys: elements, num_elements, has_heavy_elements,
        max_atomic_number, avg_atomic_number
    """
    from pymatgen.core import Element

    elements = list(set(str(site.specie) for site in structure))
    atomic_numbers = [Element(e).Z for e in elements]

    # Heavy elements relevant for strong spin-orbit coupling
    heavy_threshold = 50  # Z >= 50 (Sn and heavier)

    return {
        "elements": sorted(elements),
        "num_elements": len(elements),
        "has_heavy_elements": any(z >= heavy_threshold for z in atomic_numbers),
        "max_atomic_number": max(atomic_numbers),
        "avg_atomic_number": np.mean(atomic_numbers),
        "heavy_elements": [
            e for e, z in zip(elements, atomic_numbers) if z >= heavy_threshold
        ],
    }


def compute_structural_features(structure) -> dict:
    """
    Compute basic structural features for ML featurization.

    Returns:
        Dict with: volume_per_atom, density, avg_nn_distance,
        space_group_number, crystal_system
    """
    from pymatgen.analysis.local_env import CrystalNN

    n = len(structure)
    vol = structure.volume

    # Symmetry
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        sga = SpacegroupAnalyzer(structure)
        sg_num = sga.get_space_group_number()
        crystal_sys = sga.get_crystal_system()
    except Exception:
        sg_num = 0
        crystal_sys = "unknown"

    # Average nearest-neighbor distance
    try:
        cnn = CrystalNN()
        nn_dists = []
        for i in range(min(n, 10)):  # Sample first 10 sites
            nn_info = cnn.get_nn_info(structure, i)
            nn_dists.extend([nn["weight"] for nn in nn_info])
        avg_nn = np.mean(nn_dists) if nn_dists else 0.0
    except Exception:
        avg_nn = 0.0

    return {
        "volume_per_atom": vol / n,
        "density": structure.density,
        "avg_nn_distance": avg_nn,
        "space_group_number": sg_num,
        "crystal_system": crystal_sys,
        "num_sites": n,
    }
