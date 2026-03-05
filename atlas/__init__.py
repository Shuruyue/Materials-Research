"""
ATLAS: Accelerated Topological Learning And Screening

A platform for AI-driven discovery of topological quantum materials,
combining equivariant neural network potentials, DFT-based topological
classification, and active learning in a closed-loop framework.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_EXPORTS: dict[str, tuple[str, str]] = {
    "Config": ("atlas.config", "Config"),
    "get_config": ("atlas.config", "get_config"),
}

__all__ = ("__version__", *tuple(_EXPORTS.keys()))


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
    return sorted(set(globals().keys()) | set(__all__))
