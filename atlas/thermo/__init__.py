"""Thermodynamics helpers with optional lazy imports."""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

_LAZY_EXPORTS = {
    "CalphadCalculator": ("atlas.thermo.calphad", "CalphadCalculator"),
    "PhaseStabilityAnalyst": ("atlas.thermo.stability", "PhaseStabilityAnalyst"),
    "ReferenceDatabase": ("atlas.thermo.stability", "ReferenceDatabase"),
}

_OPTIONAL_IMPORT_ERRORS: dict[str, str] = {}
_OPTIONAL_UNAVAILABLE: set[str] = set()

__all__ = tuple(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in globals():
        return globals()[name]
    if name in _OPTIONAL_UNAVAILABLE:
        return None
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        _OPTIONAL_IMPORT_ERRORS[name] = f"{type(exc).__name__}: {exc}"
        _OPTIONAL_UNAVAILABLE.add(name)
        globals()[name] = None
        logger.debug("Optional thermo dependency unavailable for %s: %s", name, exc)
        return None
    if not hasattr(module, attr_name):
        raise AttributeError(
            f"module {module_name!r} does not define expected attribute {attr_name!r}"
        )
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def get_optional_import_errors() -> dict[str, str]:
    """Return cached optional-import errors for diagnostics."""
    return dict(_OPTIONAL_IMPORT_ERRORS)
