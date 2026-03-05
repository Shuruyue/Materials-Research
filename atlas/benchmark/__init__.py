"""
ATLAS Benchmark Module

Integration with Matbench for standardized model evaluation.
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

logger = logging.getLogger(__name__)

_EXPORTS: dict[str, tuple[str, str]] = {
    "MatbenchRunner": ("atlas.benchmark.runner", "MatbenchRunner"),
    "FoldResult": ("atlas.benchmark.runner", "FoldResult"),
    "TaskReport": ("atlas.benchmark.runner", "TaskReport"),
    "compute_regression_metrics": ("atlas.benchmark.runner", "compute_regression_metrics"),
    "compute_uncertainty_metrics": ("atlas.benchmark.runner", "compute_uncertainty_metrics"),
    "aggregate_fold_results": ("atlas.benchmark.runner", "aggregate_fold_results"),
}

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
        logger.debug("Failed to import lazy benchmark export %s from %s: %s", name, module_name, exc)
        raise ImportError(
            f"Unable to import dependency for {name!r} from {module_name!r}: {exc}"
        ) from exc
    if not hasattr(module, attr_name):
        raise AttributeError(
            f"module {module_name!r} does not define expected attribute {attr_name!r}"
        )
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))


def get_import_errors() -> dict[str, str]:
    """Return cached lazy-import errors for diagnostics."""
    return dict(_IMPORT_ERRORS)
