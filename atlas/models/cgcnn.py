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
        pooling: str = "mean_max",
        jk: str = "concat",
        message_aggr: str = "mean",
        use_edge_gates: bool = True,
    ):
        super().__init__()
        if pooling not in {"mean", "sum", "max", "mean_max", "attn"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        if jk not in {"last", "mean", "concat"}:
            raise ValueError(f"Unsupported JK mode: {jk}")
        self.pooling = pooling
        self.jk = jk
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.graph_dim = hidden_dim * 2 if pooling == "mean_max" else hidden_dim

        # Input projection
        self.node_embed = nn.Linear(node_dim, hidden_dim)

        # Convolutional layers
        self.convs = nn.ModuleList([
            MessagePassingLayer(
                hidden_dim,
                edge_dim,
                aggr=message_aggr,
                use_edge_gates=use_edge_gates,
            )
            for _ in range(n_conv)
        ])
        self.conv_bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(n_conv)
        ])

        if jk == "concat":
            self.jk_proj = nn.Linear(hidden_dim * (n_conv + 1), hidden_dim)
        else:
            self.jk_proj = None

        if pooling == "attn":
            self.attn_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.attn_gate = None

        # Fully-connected layers
        fc_layers = []
        in_dim = self.graph_dim
        for _i in range(n_fc - 1):
            fc_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        fc_layers.append(nn.Linear(in_dim, output_dim))
        self.fc = nn.Sequential(*fc_layers)

        self.dropout = nn.Dropout(dropout)

    def encode(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        batch: torch.Tensor | None = None,
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
        layer_states = [h]

        for conv, bn in zip(self.convs, self.conv_bns):
            h_new = conv(h, edge_index, edge_feats)
            h_new = bn(h_new)
            h = F.silu(h + h_new)  # residual connection
            h = self.dropout(h)
            layer_states.append(h)

        if self.jk == "concat":
            h = self.jk_proj(torch.cat(layer_states, dim=-1))
        elif self.jk == "mean":
            h = torch.stack(layer_states, dim=0).mean(dim=0)

        # Global pooling
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
        if self.pooling == "mean":
            graph_emb = global_mean_pool(h, batch)
        elif self.pooling == "sum":
            graph_emb = global_add_pool(h, batch)
        elif self.pooling == "max":
            graph_emb = global_max_pool(h, batch)
        elif self.pooling == "attn":
            from torch_geometric.utils import softmax

            attn_logits = self.attn_gate(h).squeeze(-1)
            attn = softmax(attn_logits, batch).unsqueeze(-1)
            graph_emb = global_add_pool(h * attn, batch)
        else:  # mean_max
            graph_emb = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)

        return graph_emb

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass: crystal graph → predicted property.

        Returns:
            (B, output_dim) predicted property values
        """
        graph_emb = self.encode(node_feats, edge_index, edge_feats, batch)
        return self.fc(graph_emb)
