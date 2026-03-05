"""
CGCNN: Crystal Graph Convolutional Neural Network

Baseline model for crystal property prediction.
Reference: Xie & Grossman, PRL 120, 145301 (2018)

Architecture:
    Crystal Graph → N × ConvLayers → Pooling → MLP → Property
"""

import math
from numbers import Integral, Real

import torch
import torch.nn as nn
import torch.nn.functional as F

from atlas.models.layers import MessagePassingLayer


def _is_boolean_like(value: object) -> bool:
    return isinstance(value, bool) or type(value).__name__ == "bool_"


def _coerce_positive_int(value: object, name: str) -> int:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be integer-valued, not boolean.")
    if isinstance(value, Integral):
        integer = int(value)
    elif isinstance(value, Real):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be finite.")
        rounded = round(scalar)
        if abs(scalar - rounded) > 1e-9:
            raise ValueError(f"{name} must be integer-valued.")
        integer = int(rounded)
    else:
        raise ValueError(f"{name} must be integer-valued.")
    if integer <= 0:
        raise ValueError(f"{name} must be > 0.")
    return integer


def _coerce_dropout(value: object) -> float:
    if _is_boolean_like(value):
        raise ValueError("dropout must be in [0, 1).")
    dropout_v = float(value)
    if not math.isfinite(dropout_v):
        raise ValueError("dropout must be finite.")
    if not (0.0 <= dropout_v < 1.0):
        raise ValueError("dropout must be in [0, 1).")
    return dropout_v


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
        node_dim = _coerce_positive_int(node_dim, "node_dim")
        edge_dim = _coerce_positive_int(edge_dim, "edge_dim")
        hidden_dim = _coerce_positive_int(hidden_dim, "hidden_dim")
        n_conv = _coerce_positive_int(n_conv, "n_conv")
        n_fc = _coerce_positive_int(n_fc, "n_fc")
        output_dim = _coerce_positive_int(output_dim, "output_dim")
        dropout = _coerce_dropout(dropout)
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

    @staticmethod
    def _validate_graph_inputs(
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> None:
        if node_feats.ndim != 2:
            raise ValueError(f"node_feats must be 2D [N, F], got shape {tuple(node_feats.shape)}")
        if node_feats.shape[0] <= 0:
            raise ValueError("node_feats must contain at least one node")
        if not torch.isfinite(node_feats).all():
            raise ValueError("node_feats contains NaN or Inf values")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2, E], got shape {tuple(edge_index.shape)}")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"edge_index must be integer tensor, got dtype {edge_index.dtype}")
        if edge_index.shape[1] <= 0:
            raise ValueError("edge_index must contain at least one edge")
        if edge_feats.ndim != 2:
            raise ValueError(f"edge_feats must be 2D [E, D], got shape {tuple(edge_feats.shape)}")
        if edge_feats.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_feats first dim must match edge count E ({edge_index.shape[1]}), "
                f"got {edge_feats.shape[0]}"
            )
        if not torch.isfinite(edge_feats).all():
            raise ValueError("edge_feats contains NaN or Inf values")

        n_nodes = int(node_feats.shape[0])
        edge_min = int(edge_index.min().item())
        edge_max = int(edge_index.max().item())
        if edge_min < 0 or edge_max >= n_nodes:
            raise ValueError(
                f"edge_index contains out-of-range node ids: min={edge_min}, max={edge_max}, N={n_nodes}"
            )

        if batch is not None:
            if batch.ndim != 1:
                raise ValueError(f"batch must be 1D [N], got shape {tuple(batch.shape)}")
            if batch.shape[0] != node_feats.shape[0]:
                raise ValueError(
                    f"batch size must match number of nodes ({node_feats.shape[0]}), got {batch.shape[0]}"
                )
            if batch.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"batch must be integer tensor, got dtype {batch.dtype}")
            if batch.numel() > 0 and int(batch.min().item()) < 0:
                raise ValueError("batch indices must be non-negative")

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
        self._validate_graph_inputs(node_feats, edge_index, edge_feats, batch)
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
        elif batch.dtype != torch.long:
            batch = batch.to(dtype=torch.long)

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
            attn_logits = torch.nan_to_num(attn_logits, nan=0.0, posinf=30.0, neginf=-30.0)
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
