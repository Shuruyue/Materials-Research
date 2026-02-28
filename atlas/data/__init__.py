"""
atlas.data — Data loading, databases, and property estimation.

This package intentionally uses lazy imports so light-weight modules (e.g.
`source_registry`) can be imported without pulling in optional/heavy
dependencies from dataset clients.
"""

from __future__ import annotations

from importlib import import_module

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

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)

