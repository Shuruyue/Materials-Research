"""
ATLAS Models Module

Graph Neural Network architectures for crystal property prediction:
- CGCNN: Baseline Crystal Graph CNN (Xie et al. 2018)
- EquivariantGNN: E(3)-Equivariant GNN (NequIP-inspired)
- MultiTaskGNN: Shared encoder with multi-head prediction
- GraphBuilder: Crystal structure to graph conversion
"""

from atlas.models.cgcnn import CGCNN
from atlas.models.equivariant import EquivariantGNN
from atlas.models.graph_builder import CrystalGraphBuilder
from atlas.models.layers import GatedEquivariantBlock, MessagePassingLayer
from atlas.models.m3gnet import M3GNet
from atlas.models.multi_task import EvidentialHead, MultiTaskGNN, ScalarHead, TensorHead

__all__ = [
    "CGCNN",
    "EquivariantGNN",
    "M3GNet",
    "MultiTaskGNN",
    "ScalarHead",
    "TensorHead",
    "EvidentialHead",
    "CrystalGraphBuilder",
    "MessagePassingLayer",
    "GatedEquivariantBlock",
]
