"""
atlas.active_learning — Bayesian optimization discovery loop.
"""

from atlas.active_learning.controller import DiscoveryController, Candidate
from atlas.active_learning.controller import DiscoveryController, Candidate
from atlas.active_learning.generator import StructureGenerator
from atlas.active_learning.acquisition import expected_improvement, upper_confidence_bound

__all__ = [
    "DiscoveryController", 
    "Candidate", 
    "StructureGenerator",
    "expected_improvement",
    "upper_confidence_bound",
]
