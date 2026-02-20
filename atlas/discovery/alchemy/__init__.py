"""
Alchemical Discovery Module for ATLAS.

This module integrates the Alchemical-MLIP logic for continuous chemical space exploration.
Ported and optimized from recisic/alchemical-mlip.
"""

from .model import AlchemicalModel, AlchemyManager
from .calculator import AlchemicalMACECalculator

__all__ = ["AlchemicalModel", "AlchemyManager", "AlchemicalMACECalculator"]
