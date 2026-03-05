"""
atlas.topology — Topological invariant computation and GNN classification.

Note: CrystalGraphBuilder is imported from atlas.models.graph_builder (canonical location).
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS = {
    "CrystalGraphBuilder": ("atlas.models.graph_builder", "CrystalGraphBuilder"),
    "TopoGNN": ("atlas.topology.classifier", "TopoGNN"),
}

__all__ = tuple(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
