"""
atlas.utils — Structure conversion and featurization utilities.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "pymatgen_to_ase": ("atlas.utils.structure", "pymatgen_to_ase"),
    "ase_to_pymatgen": ("atlas.utils.structure", "ase_to_pymatgen"),
    "structure_from_dict": ("atlas.utils.structure", "structure_from_dict"),
    "get_element_info": ("atlas.utils.structure", "get_element_info"),
    "compute_structural_features": ("atlas.utils.structure", "compute_structural_features"),
    "collect_runtime_metadata": ("atlas.utils.reproducibility", "collect_runtime_metadata"),
    "set_global_seed": ("atlas.utils.reproducibility", "set_global_seed"),
    "Registry": ("atlas.utils.registry", "Registry"),
    "MODELS": ("atlas.utils.registry", "MODELS"),
    "RELAXERS": ("atlas.utils.registry", "RELAXERS"),
    "FEATURE_EXTRACTORS": ("atlas.utils.registry", "FEATURE_EXTRACTORS"),
    "EVALUATORS": ("atlas.utils.registry", "EVALUATORS"),
}

__all__ = tuple(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in globals():
        return globals()[name]
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    try:
        value = getattr(module, attr_name)
    except AttributeError as exc:
        raise AttributeError(
            f"module {module_name!r} does not define expected attribute {attr_name!r}"
        ) from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
