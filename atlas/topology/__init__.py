"""
atlas.topology — Topological invariant computation and GNN classification.

Note: CrystalGraphBuilder is also available from atlas.models (canonical location).
"""

from atlas.topology.classifier import CrystalGraphBuilder, TopoGNN, MessagePassingLayer

__all__ = ["CrystalGraphBuilder", "TopoGNN", "MessagePassingLayer"]
