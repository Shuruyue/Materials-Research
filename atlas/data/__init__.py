"""
atlas.data — Data loading, databases, and property estimation.

This package intentionally uses lazy imports so light-weight modules (e.g.
`source_registry`) can be imported without pulling in optional/heavy
dependencies from dataset clients.
"""

from __future__ import annotations

from importlib import import_module
import logging
from typing import Any

logger = logging.getLogger(__name__)

_EXPORTS = {
    "JARVISClient": ("atlas.data.jarvis_client", "JARVISClient"),
    "TopoDB": ("atlas.data.topo_db", "TopoDB"),
    "TopoMaterial": ("atlas.data.topo_db", "TopoMaterial"),
    "TOPO_CLASSES": ("atlas.data.topo_db", "TOPO_CLASSES"),
    "PropertyEstimator": ("atlas.data.property_estimator", "PropertyEstimator"),
    "AlloyEstimator": ("atlas.data.alloy_estimator", "AlloyEstimator"),
    "AlloyPhase": ("atlas.data.alloy_estimator", "AlloyPhase"),
    "CrystalPropertyDataset": ("atlas.data.crystal_dataset", "CrystalPropertyDataset"),
    "DATA_SOURCES": ("atlas.data.source_registry", "DATA_SOURCES"),
    "DataSourceRegistry": ("atlas.data.source_registry", "DataSourceRegistry"),
    "DataSourceSpec": ("atlas.data.source_registry", "DataSourceSpec"),
    "SourceReliability": ("atlas.data.source_registry", "SourceReliability"),
    "ReliabilitySnapshot": ("atlas.data.source_registry", "ReliabilitySnapshot"),
    "SourceEstimate": ("atlas.data.source_registry", "SourceEstimate"),
    "FusedEstimate": ("atlas.data.source_registry", "FusedEstimate"),
    "validate_dataset": ("atlas.data.data_validation", "validate_dataset"),
    "compute_trust_score": ("atlas.data.data_validation", "compute_trust_score"),
    "ValidationReport": ("atlas.data.data_validation", "ValidationReport"),
    "TrustScore": ("atlas.data.data_validation", "TrustScore"),
    "TrustScoreBreakdown": ("atlas.data.data_validation", "TrustScoreBreakdown"),
    "ProvenanceRecord": ("atlas.data.data_validation", "ProvenanceRecord"),
    "iid_split": ("atlas.data.split_governance", "iid_split"),
    "compositional_split": ("atlas.data.split_governance", "compositional_split"),
    "prototype_split": ("atlas.data.split_governance", "prototype_split"),
    "SplitManifest": ("atlas.data.split_governance", "SplitManifest"),
    "SplitManifestV2": ("atlas.data.split_governance", "SplitManifestV2"),
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
        logger.debug("Failed to import lazy data export %s from %s: %s", name, module_name, exc)
        raise ImportError(
            f"Unable to import dependency for {name!r} from {module_name!r}: {exc}"
        ) from exc
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


def get_import_errors() -> dict[str, str]:
    """Return cached lazy-import errors for diagnostics."""
    return dict(_IMPORT_ERRORS)

