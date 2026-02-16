"""
atlas.active_learning — Bayesian optimization discovery loop.
"""

from atlas.active_learning.controller import DiscoveryController, Candidate
from atlas.active_learning.generator import StructureGenerator

__all__ = ["DiscoveryController", "Candidate", "StructureGenerator"]
