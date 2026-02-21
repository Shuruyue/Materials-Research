"""
atlas.active_learning — Bayesian optimization discovery loop.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DiscoveryController": ("atlas.active_learning.controller", "DiscoveryController"),
    "Candidate": ("atlas.active_learning.controller", "Candidate"),
    "StructureGenerator": ("atlas.active_learning.generator", "StructureGenerator"),
    "expected_improvement": ("atlas.active_learning.acquisition", "expected_improvement"),
    "upper_confidence_bound": ("atlas.active_learning.acquisition", "upper_confidence_bound"),
    "GPSurrogateAcquirer": ("atlas.active_learning.gp_surrogate", "GPSurrogateAcquirer"),
    "GPSurrogateConfig": ("atlas.active_learning.gp_surrogate", "GPSurrogateConfig"),
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
