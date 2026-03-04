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
    "log_expected_improvement": ("atlas.active_learning.acquisition", "log_expected_improvement"),
    "probability_of_improvement": ("atlas.active_learning.acquisition", "probability_of_improvement"),
    "log_probability_of_improvement": ("atlas.active_learning.acquisition", "log_probability_of_improvement"),
    "noisy_expected_improvement": ("atlas.active_learning.acquisition", "noisy_expected_improvement"),
    "log_noisy_expected_improvement": ("atlas.active_learning.acquisition", "log_noisy_expected_improvement"),
    "upper_confidence_bound": ("atlas.active_learning.acquisition", "upper_confidence_bound"),
    "schedule_ucb_kappa": ("atlas.active_learning.acquisition", "schedule_ucb_kappa"),
    "score_acquisition": ("atlas.active_learning.acquisition", "score_acquisition"),
    "normalize_acquisition_strategy": ("atlas.active_learning.acquisition", "normalize_acquisition_strategy"),
    "BASE_ACQUISITION_STRATEGIES": ("atlas.active_learning.acquisition", "BASE_ACQUISITION_STRATEGIES"),
    "DISCOVERY_ACQUISITION_STRATEGIES": ("atlas.active_learning.acquisition", "DISCOVERY_ACQUISITION_STRATEGIES"),
    "GPSurrogateAcquirer": ("atlas.active_learning.gp_surrogate", "GPSurrogateAcquirer"),
    "GPSurrogateConfig": ("atlas.active_learning.gp_surrogate", "GPSurrogateConfig"),
    "ActiveLearningPolicyConfig": ("atlas.active_learning.policy_state", "ActiveLearningPolicyConfig"),
    "PolicyState": ("atlas.active_learning.policy_state", "PolicyState"),
    "PolicyEngine": ("atlas.active_learning.policy_engine", "PolicyEngine"),
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
