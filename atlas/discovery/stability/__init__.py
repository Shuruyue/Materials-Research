"""
Reaction Stability Module
Focuses on assessing synthesis feasibility and reaction pathways using MEPIN techniques.
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

logger = logging.getLogger(__name__)

_EXPORTS = {
    "MEPINStabilityEvaluator": ("atlas.discovery.stability.mepin", "MEPINStabilityEvaluator"),
}
_OPTIONAL_IMPORT_ERRORS: dict[str, str] = {}
_OPTIONAL_UNAVAILABLE: set[str] = set()

__all__ = tuple(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in globals():
        return globals()[name]
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _OPTIONAL_UNAVAILABLE:
        return None
    module_name, attr_name = _EXPORTS[name]
    try:
        module = import_module(module_name)
        value = getattr(module, attr_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
        _OPTIONAL_IMPORT_ERRORS[name] = f"{type(exc).__name__}: {exc}"
        _OPTIONAL_UNAVAILABLE.add(name)
        logger.debug("Optional stability dependency unavailable for %s: %s", name, exc)
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
