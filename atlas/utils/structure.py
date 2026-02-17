"""
Structure Conversion Utilities

Convert between pymatgen, ASE, and MACE data formats.
Provides helpers for structural analysis and featurization.

Optimization:
- Standardization: Convert to conventional/primitive cells for consistent GNN input.
- Robust MACE Support: Handle MACE Atoms conversion explicitly if needed.
"""

from typing import Optional, Dict, Any, Union
import numpy as np

def pymatgen_to_ase(structure):
    """
    Convert pymatgen Structure → ASE Atoms.
    """
    from ase import Atoms
    from pymatgen.io.ase import AseAtomsAdaptor
    
    # Use official adaptor for best compatibility (magmoms, etc.)
    return AseAtomsAdaptor.get_atoms(structure)

def ase_to_pymatgen(atoms):
    """
    Convert ASE Atoms → pymatgen Structure.
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor.get_structure(atoms)

def structure_from_dict(d: dict):
    """Reconstruct a pymatgen Structure from its dict representation."""
    from pymatgen.core import Structure
    return Structure.from_dict(d)

def get_standardized_structure(structure, primitive: bool = False):
    """
    Get the standardized (conventional or primitive) structure.
    Crucial for GNNs to ensure invariant inputs for the same material.
    
    Args:
        structure: pymatgen Structure
        primitive: If True, return primitive cell. Else conventional.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.01)
        if primitive:
            return sga.get_primitive_standard_structure()
        else:
            return sga.get_conventional_standard_structure()
    except Exception:
        # Fallback if symmetry detection fails (e.g. too distorted)
        return structure

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

    # Heavy elements relevant for strong spin-orbit coupling (SOC)
    # Important for Topological Materials
    heavy_threshold = 50  # Z >= 50 (Sn and heavier)

    return {
        "elements": sorted(elements),
        "num_elements": len(elements),
        "has_heavy_elements": any(z >= heavy_threshold for z in atomic_numbers),
        "max_atomic_number": max(atomic_numbers),
        "avg_atomic_number": float(np.mean(atomic_numbers)),
        "atomic_numbers": atomic_numbers,
        "heavy_elements": [
            e for e, z in zip(elements, atomic_numbers) if z >= heavy_threshold
        ],
    }

def compute_structural_features(structure) -> dict:
    """
    Compute basic structural features for ML featurization/analytics.
    
    Returns:
        Dict with: volume_per_atom, density, avg_nn_distance,
        space_group_number, crystal_system, dimensionality_score
    """
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    n = len(structure)
    vol = structure.volume

    # Symmetry
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        sg_num = sga.get_space_group_number()
        crystal_sys = sga.get_crystal_system()
    except Exception:
        sg_num = 0
        crystal_sys = "unknown"

    # Average nearest-neighbor distance (Bond Length proxy)
    try:
        # Distance to nearest neighbor
        # Faster than CrystalNN for just distance
        dists = []
        # Sample a few atoms if large
        indices = range(n) if n < 50 else np.random.choice(n, 50, replace=False)
        for i in indices:
            # get_neighbors returns list of Neighbor objects
            # we just want the closest one
            nbrs = structure.get_neighbors(structure[i], r=4.0)
            if nbrs:
                dists.append(min(n.nn_distance for n in nbrs))
        avg_nn = float(np.mean(dists)) if dists else 0.0
    except Exception:
        avg_nn = 0.0

    return {
        "volume_per_atom": vol / n,
        "density": structure.density,
        "avg_nn_distance": avg_nn,
        "space_group_number": sg_num,
        "crystal_system": crystal_sys,
        "num_sites": n,
        "formula": structure.composition.reduced_formula
    }
