"""
ATLAS: Accelerated Topological Learning And Screening

A platform for AI-driven discovery of topological quantum materials,
combining equivariant neural network potentials, DFT-based topological
classification, and active learning in a closed-loop framework.
"""

__version__ = "0.1.0"

from atlas.config import Config, get_config

__all__ = ["__version__", "get_config", "Config"]
