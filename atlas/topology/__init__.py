"""
atlas.topology — Topological invariant computation and GNN classification.

Note: CrystalGraphBuilder is imported from atlas.models.graph_builder (canonical location).
"""

from atlas.models.graph_builder import CrystalGraphBuilder
from atlas.topology.classifier import TopoGNN

__all__ = ["CrystalGraphBuilder", "TopoGNN"]
