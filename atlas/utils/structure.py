"""
Structure Conversion Utilities

Convert between pymatgen, ASE, and MACE data formats.
Provides helpers for structural analysis and featurization.

Optimization:
- Standardization: Convert to conventional/primitive cells for consistent GNN input.
- Robust MACE Support: Handle MACE Atoms conversion explicitly if needed.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(scalar):
        return float(default)
    return scalar


def _element_symbol_number_pairs(structure: Any) -> list[tuple[str, int]]:
    """Extract unique (symbol, Z) element pairs robustly."""
    from pymatgen.core import Element

    try:
        composition = structure.composition
        element_comp = composition.element_composition
        elements = list(getattr(element_comp, "elements", []))
    except Exception:
        elements = []

    pairs: dict[str, int] = {}
    for el in elements:
        symbol = getattr(el, "symbol", None)
        z = getattr(el, "Z", None)
        if symbol is None or z is None:
            try:
                parsed = Element(str(el))
            except Exception:
                continue
            symbol = parsed.symbol
            z = parsed.Z
        pairs[str(symbol)] = int(z)

    return sorted(pairs.items(), key=lambda item: item[0])


def pymatgen_to_ase(structure: Any) -> Any:
    """
    Convert pymatgen Structure → ASE Atoms.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    # Use official adaptor for best compatibility (magmoms, etc.)
    return AseAtomsAdaptor.get_atoms(structure)


def ase_to_pymatgen(atoms: Any) -> Any:
    """
    Convert ASE Atoms → pymatgen Structure.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_structure(atoms)


def structure_from_dict(d: dict[str, Any]) -> Any:
    """Reconstruct a pymatgen Structure from its dict representation."""
    from pymatgen.core import Structure

    return Structure.from_dict(d)


def get_standardized_structure(structure: Any, primitive: bool = False) -> Any:
    """
    Get the standardized (conventional or primitive) structure.
    Crucial for GNNs to ensure invariant inputs for the same material.

    Args:
        structure: pymatgen Structure
        primitive: If True, return primitive cell. Else conventional.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    primitive_flag = _coerce_bool_like(primitive, "primitive")

    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.01)
        if primitive_flag:
            return sga.get_primitive_standard_structure()
        return sga.get_conventional_standard_structure()
    except Exception:
        # Fallback if symmetry detection fails (e.g. too distorted)
        return structure


def get_element_info(structure: Any) -> dict[str, Any]:
    """
    Get element-level information about a structure.

    Returns:
        Dict with keys: elements, num_elements, has_heavy_elements,
        max_atomic_number, avg_atomic_number
    """
    if len(structure) == 0:
        return {
            "elements": [],
            "num_elements": 0,
            "has_heavy_elements": False,
            "max_atomic_number": 0,
            "avg_atomic_number": 0.0,
            "atomic_numbers": [],
            "heavy_elements": [],
        }
    element_pairs = _element_symbol_number_pairs(structure)
    elements = [symbol for symbol, _ in element_pairs]
    atomic_numbers = [z for _, z in element_pairs]

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
            e for e, z in zip(elements, atomic_numbers, strict=True) if z >= heavy_threshold
        ],
    }


def _coerce_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise ValueError(f"{name} must be integer-valued, not boolean")
    if isinstance(value, Integral):
        coerced = int(value)
    else:
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if coerced < 0:
        raise ValueError(f"{name} must be >= 0")
    return coerced


def _coerce_bool_like(value: Any, name: str) -> bool:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return bool(value)
    if isinstance(value, Integral):
        integer = int(value)
        if integer in (0, 1):
            return bool(integer)
        raise ValueError(f"{name} integer payload must be 0/1")
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be bool-like")


def _sample_site_indices(n_sites: int, max_samples: int = 50) -> np.ndarray:
    n_sites = _coerce_non_negative_int(n_sites, "n_sites")
    max_samples = _coerce_non_negative_int(max_samples, "max_samples")
    if n_sites <= 0:
        return np.zeros(0, dtype=int)
    if max_samples <= 0:
        return np.zeros(0, dtype=int)
    if n_sites <= max_samples:
        return np.arange(n_sites, dtype=int)
    step = max(1, n_sites // max_samples)
    indices = np.arange(0, n_sites, step, dtype=int)
    return indices[:max_samples]


def _closest_neighbor_distance(structure: Any, site_index: int, radii: tuple[float, ...]) -> float | None:
    site = structure[site_index]
    for radius in radii:
        nbrs = structure.get_neighbors(site, r=radius)
        if not nbrs:
            continue
        finite_dists = [
            _finite_float(getattr(neigh, "nn_distance", np.nan), default=np.nan)
            for neigh in nbrs
        ]
        finite_dists = [dist for dist in finite_dists if np.isfinite(dist) and dist > 0.0]
        if finite_dists:
            return float(min(finite_dists))
    return None


def compute_structural_features(structure: Any) -> dict[str, Any]:
    """
    Compute basic structural features for ML featurization/analytics.

    Returns:
        Dict with: volume_per_atom, density, avg_nn_distance,
        space_group_number, crystal_system, dimensionality_score
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    n = int(len(structure))
    if n <= 0:
        return {
            "volume_per_atom": 0.0,
            "density": 0.0,
            "avg_nn_distance": 0.0,
            "space_group_number": 0,
            "crystal_system": "unknown",
            "num_sites": 0,
            "formula": "",
        }
    vol = _finite_float(getattr(structure, "volume", 0.0), default=0.0)

    # Symmetry
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        sg_num = int(sga.get_space_group_number())
        crystal_sys = str(sga.get_crystal_system())
    except Exception:
        sg_num = 0
        crystal_sys = "unknown"

    # Average nearest-neighbor distance (Bond Length proxy)
    try:
        # Distance to nearest neighbor
        # Faster than CrystalNN for just distance
        dists = []
        # Sample a few atoms if large
        indices = _sample_site_indices(n, max_samples=50)
        for i in indices:
            nearest = _closest_neighbor_distance(structure, int(i), radii=(4.0, 8.0))
            if nearest is not None:
                dists.append(nearest)
        avg_nn = float(np.mean(dists)) if dists else 0.0
    except Exception:
        avg_nn = 0.0

    try:
        formula = str(structure.composition.reduced_formula)
    except Exception:
        formula = ""

    return {
        "volume_per_atom": _finite_float(vol / max(n, 1), default=0.0),
        "density": _finite_float(getattr(structure, "density", 0.0), default=0.0),
        "avg_nn_distance": avg_nn,
        "space_group_number": sg_num,
        "crystal_system": crystal_sys,
        "num_sites": n,
        "formula": formula,
    }
