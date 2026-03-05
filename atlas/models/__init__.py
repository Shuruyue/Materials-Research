"""
ATLAS Models Module

Graph Neural Network architectures for crystal property prediction:
- CGCNN: Baseline Crystal Graph CNN (Xie et al. 2018)
- EquivariantGNN: E(3)-Equivariant GNN (NequIP-inspired)
- MultiTaskGNN: Shared encoder with multi-head prediction
- GraphBuilder: Crystal structure to graph conversion
"""

from __future__ import annotations

import logging
from importlib import import_module
from types import MappingProxyType
from typing import Any

logger = logging.getLogger(__name__)

_EXPORTS = MappingProxyType({
    "CGCNN": ("atlas.models.cgcnn", "CGCNN"),
    "EquivariantGNN": ("atlas.models.equivariant", "EquivariantGNN"),
    "M3GNet": ("atlas.models.m3gnet", "M3GNet"),
    "MultiTaskGNN": ("atlas.models.multi_task", "MultiTaskGNN"),
    "ScalarHead": ("atlas.models.multi_task", "ScalarHead"),
    "TensorHead": ("atlas.models.multi_task", "TensorHead"),
    "EvidentialHead": ("atlas.models.multi_task", "EvidentialHead"),
    "CrystalGraphBuilder": ("atlas.models.graph_builder", "CrystalGraphBuilder"),
    "MessagePassingLayer": ("atlas.models.layers", "MessagePassingLayer"),
    "GatedEquivariantBlock": ("atlas.models.layers", "GatedEquivariantBlock"),
})

__all__ = tuple(_EXPORTS.keys())
_IMPORT_ERRORS: dict[str, str] = {}


def __getattr__(name: str) -> Any:
    if name in globals():
        return globals()[name]
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        _IMPORT_ERRORS[name] = f"{type(exc).__name__}: {exc}"
        logger.debug("Failed to import lazy model export %s from %s: %s", name, module_name, exc)
        raise ImportError(
            f"Unable to import dependency for {name!r} from {module_name!r}: {exc}"
        ) from exc
    try:
        value = getattr(module, attr_name)
    except AttributeError as exc:
        raise AttributeError(
            f"module {module_name!r} does not define expected export {attr_name!r} for {__name__!r}"
        ) from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))


def get_import_errors() -> dict[str, str]:
    """Return cached lazy-import errors for diagnostics."""
    return dict(_IMPORT_ERRORS)
