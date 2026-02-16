"""
ATLAS Models Module

Neural network architectures for crystal property prediction:
- CrystalGraphBuilder: crystal structure → graph representation
- CGCNN: Crystal Graph Convolutional Neural Network (baseline)
- EquivariantGNN: E(3)-equivariant GNN using e3nn (Phase 2)
- MultiTaskGNN: multi-task wrapper with shared encoder + task-specific heads
"""

from atlas.models.graph_builder import CrystalGraphBuilder
from atlas.models.layers import MessagePassingLayer, GatedEquivariantBlock
from atlas.models.cgcnn import CGCNN
from atlas.models.multi_task import MultiTaskGNN, ScalarHead, TensorHead

# EquivariantGNN requires e3nn — import lazily to avoid hard dependency
try:
    from atlas.models.equivariant import EquivariantGNN
except ImportError:
    EquivariantGNN = None  # e3nn not installed

__all__ = [
    "CrystalGraphBuilder",
    "MessagePassingLayer",
    "GatedEquivariantBlock",
    "CGCNN",
    "EquivariantGNN",
    "MultiTaskGNN",
    "ScalarHead",
    "TensorHead",
]
