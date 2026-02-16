"""
CGCNN: Crystal Graph Convolutional Neural Network

Baseline model for crystal property prediction.
Reference: Xie & Grossman, PRL 120, 145301 (2018)

Architecture:
    Crystal Graph → N × ConvLayers → Pooling → MLP → Property
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from atlas.models.layers import MessagePassingLayer


class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network.

    Predicts a single scalar property from crystal structure graph.
    Used as baseline in Phase 1 validation.

    Args:
        node_dim: input node feature dimension
        edge_dim: input edge feature dimension
        hidden_dim: hidden layer dimension
        n_conv: number of convolution layers
        n_fc: number of fully-connected layers after pooling
        output_dim: output dimension (1 for single property)
        dropout: dropout rate
    """

    def __init__(
        self,
        node_dim: int = 91,
        edge_dim: int = 20,
        hidden_dim: int = 128,
        n_conv: int = 3,
        n_fc: int = 2,
        output_dim: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.node_embed = nn.Linear(node_dim, hidden_dim)

        # Convolutional layers
        self.convs = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_dim) for _ in range(n_conv)
        ])
        self.conv_bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(n_conv)
        ])

        # Fully-connected layers
        fc_layers = []
        for i in range(n_fc - 1):
            fc_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        fc_layers.append(nn.Linear(hidden_dim, output_dim))
        self.fc = nn.Sequential(*fc_layers)

        self.dropout = nn.Dropout(dropout)

    def encode(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode crystal graph into graph-level embedding.

        Args:
            node_feats: (N, node_dim)
            edge_index: (2, E)
            edge_feats: (E, edge_dim)
            batch: (N,) batch assignment

        Returns:
            (B, hidden_dim) graph-level embedding
        """
        h = self.node_embed(node_feats)

        for conv, bn in zip(self.convs, self.conv_bns):
            h_new = conv(h, edge_index, edge_feats)
            h_new = bn(h_new)
            h = F.relu(h + h_new)  # residual connection
            h = self.dropout(h)

        # Global pooling
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        from torch_geometric.nn import global_mean_pool
        graph_emb = global_mean_pool(h, batch)

        return graph_emb

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: crystal graph → predicted property.

        Returns:
            (B, output_dim) predicted property values
        """
        graph_emb = self.encode(node_feats, edge_index, edge_feats, batch)
        return self.fc(graph_emb)
