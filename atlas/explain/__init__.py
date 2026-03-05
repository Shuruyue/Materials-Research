"""
ATLAS Explain Module

Interpretability analysis for crystal GNN models:
- GNNExplainer: identify important substructures per property
- Latent space: t-SNE/UMAP visualization of materials property space
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

logger = logging.getLogger(__name__)

_EXPORTS = {
    "GNNExplainerWrapper": ("atlas.explain.gnn_explainer", "GNNExplainerWrapper"),
    "LatentSpaceAnalyzer": ("atlas.explain.latent_analysis", "LatentSpaceAnalyzer"),
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
        return globals().get(name)
    module_name, attr_name = _EXPORTS[name]
    try:
        module = import_module(module_name)
        value = getattr(module, attr_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
        _OPTIONAL_IMPORT_ERRORS[name] = f"{type(exc).__name__}: {exc}"
        _OPTIONAL_UNAVAILABLE.add(name)
        logger.debug("Optional explain dependency unavailable for %s: %s", name, exc)
        if name == "LatentSpaceAnalyzer":
            value = None
        else:
            raise ImportError(
                f"{attr_name} is unavailable because optional explainability dependencies are missing: {exc}"
            ) from exc
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
