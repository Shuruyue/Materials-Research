"""
ATLAS Training Module

Training infrastructure for crystal property prediction:
- Trainer: Robust single/multi-task training loop with AMP and Gradient Clipping
- MultiTaskLoss: Uncertainty-weighted or fixed-weight multi-task loss
- PhysicsConstraintLoss: Physics-informed loss functions (Positivity, Bounds)
- Metrics: Comprehensive evaluation metrics (MAE, RMSE, R², AUC, F1)
- Normalizers: Target normalization for single and multi-property training
- Filters: Statistical outlier detection and removal
- CheckpointManager: Top-k best + rotating checkpoint persistence
"""

from __future__ import annotations

import importlib
import logging
from types import MappingProxyType
from typing import Any

logger = logging.getLogger(__name__)

_LAZY_EXPORTS = MappingProxyType({
    "Trainer": ("atlas.training.trainer", "Trainer"),
    "MultiTaskLoss": ("atlas.training.losses", "MultiTaskLoss"),
    "PropertyLoss": ("atlas.training.losses", "PropertyLoss"),
    "PhysicsConstraintLoss": ("atlas.training.physics_losses", "PhysicsConstraintLoss"),
    "VoigtReussBoundsLoss": ("atlas.training.physics_losses", "VoigtReussBoundsLoss"),
    "scalar_metrics": ("atlas.training.metrics", "scalar_metrics"),
    "tensor_metrics": ("atlas.training.metrics", "tensor_metrics"),
    "classification_metrics": ("atlas.training.metrics", "classification_metrics"),
    "TargetNormalizer": ("atlas.training.normalizers", "TargetNormalizer"),
    "MultiTargetNormalizer": ("atlas.training.normalizers", "MultiTargetNormalizer"),
    "filter_outliers": ("atlas.training.filters", "filter_outliers"),
    "CheckpointManager": ("atlas.training.checkpoint", "CheckpointManager"),
})

__all__ = tuple(_LAZY_EXPORTS.keys())
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
        logger.debug("Failed to import lazy training export %s from %s: %s", name, module_name, exc)
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
