"""
Research method registry for switchable project methodologies.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

_LAZY_EXPORTS = {
    "MethodSpec": ("atlas.research.method_registry", "MethodSpec"),
    "METHODS": ("atlas.research.method_registry", "METHODS"),
    "get_method": ("atlas.research.method_registry", "get_method"),
    "list_methods": ("atlas.research.method_registry", "list_methods"),
    "recommended_method_order": (
        "atlas.research.method_registry",
        "recommended_method_order",
    ),
    "IterationSnapshot": (
        "atlas.research.workflow_reproducible_graph",
        "IterationSnapshot",
    ),
    "RunManifest": ("atlas.research.workflow_reproducible_graph", "RunManifest"),
    "WorkflowReproducibleGraph": (
        "atlas.research.workflow_reproducible_graph",
        "WorkflowReproducibleGraph",
    ),
}

__all__ = tuple(_LAZY_EXPORTS)
_IMPORT_ERRORS: dict[str, str] = {}


def __getattr__(name: str) -> Any:
    if name in globals():
        return globals()[name]
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        _IMPORT_ERRORS[name] = f"{type(exc).__name__}: {exc}"
        logger.debug("Failed to import lazy research export %s from %s: %s", name, module_name, exc)
        raise ImportError(
            f"Unable to import dependency for {name!r} from {module_name!r}: {exc}"
        ) from exc
    try:
        value = getattr(module, attr_name)
    except AttributeError as exc:
        raise AttributeError(
            f"module {module_name!r} does not define lazy export {attr_name!r}"
        ) from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def get_import_errors() -> dict[str, str]:
    """Return cached lazy-import errors for diagnostics."""
    return dict(_IMPORT_ERRORS)
