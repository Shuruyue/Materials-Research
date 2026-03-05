"""
Alchemical Discovery Module for ATLAS.

This module integrates the Alchemical-MLIP logic for continuous chemical space exploration.
Ported and optimized from recisic/alchemical-mlip.
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

logger = logging.getLogger(__name__)

_EXPORTS = {
    "AlchemicalModel": ("atlas.discovery.alchemy.model", "AlchemicalModel"),
    "AlchemyManager": ("atlas.discovery.alchemy.model", "AlchemyManager"),
    "AlchemicalMACECalculator": ("atlas.discovery.alchemy.calculator", "AlchemicalMACECalculator"),
}
_OPTIONAL_IMPORT_ERRORS: dict[str, str] = {}
_OPTIONAL_UNAVAILABLE: set[str] = set()


def _missing_dependency_message(symbol: str, exc: Exception) -> str:
    return (
        f"{symbol} is unavailable because optional alchemical dependencies are missing "
        f"or failed to initialize: {exc}"
    )


def _missing_calculator_factory(message: str):
    class _UnavailableAlchemicalMACECalculator:  # pragma: no cover - optional dependency path
        def __init__(self, *args, **kwargs):
            raise ImportError(message)

    _UnavailableAlchemicalMACECalculator.__name__ = "AlchemicalMACECalculator"
    return _UnavailableAlchemicalMACECalculator


__all__ = tuple(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in globals():
        return globals()[name]
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _OPTIONAL_UNAVAILABLE:
        return globals().get(name)

    module_name, attr_name = _EXPORTS[name]
    try:
        module = import_module(module_name)
        value = getattr(module, attr_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - optional heavy dependency path
        message = _missing_dependency_message(attr_name, exc)
        _OPTIONAL_IMPORT_ERRORS[name] = f"{type(exc).__name__}: {exc}"
        _OPTIONAL_UNAVAILABLE.add(name)
        logger.debug("Optional alchemy dependency unavailable for %s: %s", name, exc)
        if attr_name == "AlchemicalMACECalculator":
            value = _missing_calculator_factory(message)
        else:
            value = None
    except AttributeError as exc:
        raise AttributeError(
            f"module {module_name!r} does not define expected attribute {attr_name!r}"
        ) from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))


def get_optional_import_errors() -> dict[str, str]:
    """Return cached optional-import errors for diagnostics."""
    return dict(_OPTIONAL_IMPORT_ERRORS)
