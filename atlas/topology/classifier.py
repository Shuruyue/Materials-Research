"""
Topological Material GNN Classifier

A Graph Neural Network that classifies crystal structures as
topological or trivial based on their atomic structure.

Architecture:
    Crystal Structure → Graph (atoms=nodes, bonds=edges)
    → Message Passing (3 layers) → Global Pooling → MLP → P(topological)

Improved:
- Model Persistence: save_model/load_model methods.
- Uncertainty: Monte Carlo Dropout for Bayesian approximation.
"""

import logging
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Configure logging
logger = logging.getLogger(__name__)


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
    if not (0.0 <= dropout_v < 1.0):
        raise ValueError("dropout must be in [0, 1).")
    return dropout_v



class TopoGNN(nn.Module):
    """
    Graph Neural Network for topological material classification.
    """

    def __init__(
        self,
        node_dim: int = 91,   # 86 one-hot + 5 properties (from CrystalGraphBuilder)
        edge_dim: int = 20,   # Gaussian expansion
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        node_dim = _coerce_positive_int(node_dim, "node_dim")
        edge_dim = _coerce_positive_int(edge_dim, "edge_dim")
        hidden_dim = _coerce_positive_int(hidden_dim, "hidden_dim")
        n_layers = _coerce_positive_int(n_layers, "n_layers")
        dropout = _coerce_dropout(dropout)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout

        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)

        # Message passing layers
        self.mp_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.mp_layers.append(_TopoMessagePassingLayer(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Readout MLP
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _validate_inputs(
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> None:
        if node_feats.ndim != 2:
            raise ValueError(f"node_feats must be 2D [N, F], got shape {tuple(node_feats.shape)}")
        if node_feats.shape[0] <= 0:
            raise ValueError("node_feats must contain at least one node.")
        if not torch.isfinite(node_feats).all():
            raise ValueError("node_feats contains NaN or Inf values.")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2, E], got shape {tuple(edge_index.shape)}")
        if edge_index.shape[1] <= 0:
            raise ValueError("edge_index must contain at least one edge.")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"edge_index must be integer tensor, got dtype {edge_index.dtype}")
        if edge_feats.ndim != 2:
            raise ValueError(f"edge_feats must be 2D [E, D], got shape {tuple(edge_feats.shape)}")
        if edge_feats.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_feats first dim must match edge count E ({edge_index.shape[1]}), "
                f"got {edge_feats.shape[0]}"
            )
        if not torch.isfinite(edge_feats).all():
            raise ValueError("edge_feats contains NaN or Inf values.")

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
            if int(batch.min().item()) < 0:
                raise ValueError("batch indices must be non-negative.")

    def forward(self, node_feats, edge_index, edge_feats, batch=None):
        """Forward pass."""
        self._validate_inputs(node_feats, edge_index, edge_feats, batch)
        h = self.node_embed(node_feats)
        e = self.edge_embed(edge_feats)

        # Message passing
        for mp, norm in zip(self.mp_layers, self.norms, strict=True):
            h_new = mp(h, edge_index, e)
            h = norm(h + h_new)  # residual + layernorm

        # Global pooling
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        elif batch.dtype != torch.long:
            batch = batch.to(dtype=torch.long)
        if batch.numel() == 0:
            raise ValueError("batch cannot be empty.")

        # Remap possibly sparse/non-contiguous batch ids to contiguous [0, n_graphs).
        unique_batch, inverse = torch.unique(batch, sorted=True, return_inverse=True)
        if unique_batch.numel() <= 0:
            raise ValueError("batch must contain at least one graph id.")
        batch = inverse

        # Mean and max pooling
        n_graphs = int(unique_batch.numel())
        h_mean = torch.zeros(n_graphs, h.size(1), device=h.device)

        h_mean.scatter_reduce_(
            0,
            batch.unsqueeze(1).expand_as(h),
            h,
            reduce="mean",
            include_self=False,
        )

        h_max_vals = torch.full((n_graphs, h.size(1)), float("-inf"), device=h.device)
        h_max_vals.scatter_reduce_(
            0,
            batch.unsqueeze(1).expand_as(h),
            h,
            reduce="amax",
            include_self=False,
        )
        h_max = torch.nan_to_num(h_max_vals, neginf=0.0, posinf=0.0)

        h_global = torch.cat([h_mean, h_max], dim=1)

        return self.readout(h_global).squeeze(-1)

    def predict_proba(
        self,
        node_feats,
        edge_index,
        edge_feats,
        batch=None,
        mc_dropout: bool = False,
        n_samples: int = 10
    ) -> float | tuple[float, float]:
        """
        Get probability of being topological.

        Args:
            mc_dropout: Enable Monte Carlo Dropout for uncertainty estimation.
            n_samples: Number of MC samples if enabled.

        Returns:
            prob (float) or (mean_prob, std_prob)
        """
        n_samples_i = _coerce_positive_int(n_samples, "n_samples")

        def _single_prob(logits: torch.Tensor) -> float:
            if logits.numel() != 1:
                raise ValueError(
                    "predict_proba expects exactly one graph in batch. "
                    f"Got {logits.numel()} logits."
                )
            return float(torch.sigmoid(logits.reshape(-1)[0]).item())

        if not mc_dropout:
            self.eval()
            with torch.no_grad():
                logits = self.forward(node_feats, edge_index, edge_feats, batch)
                return _single_prob(logits)
        else:
            # MC Dropout: keep dropout active during inference
            was_training = self.training
            self.train() # Enable dropout
            probs = []
            try:
                with torch.no_grad():
                    for _ in range(n_samples_i):
                        logits = self.forward(node_feats, edge_index, edge_feats, batch)
                        probs.append(_single_prob(logits))
            finally:
                self.train(was_training)

            probs = np.array(probs)
            return float(probs.mean()), float(probs.std())

    def save_model(self, path: str | Path):
        """Save model weights and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save state dict along with init params to easy reconstruction
        state = {
            'state_dict': self.state_dict(),
            'config': {
                'node_dim': self.node_dim,
                'edge_dim': self.edge_dim,
                'hidden_dim': self.hidden_dim,
                'n_layers': self.n_layers,
                'dropout': self.dropout_rate
            }
        }
        torch.save(state, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str | Path, device: str = "cpu") -> "TopoGNN":
        """Load model from file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})
        if not isinstance(config, dict):
            raise ValueError("Checkpoint config must be a dict.")

        # Instantiate with saved config
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()

        logger.info(f"Model loaded from {path}")
        return model


class _TopoMessagePassingLayer(nn.Module):
    """Bidirectional message passing for topology classification.

    Unlike ``atlas.models.layers.MessagePassingLayer`` which concatenates
    ``(h_src, e)``, this variant concatenates ``(h_src, h_dst, e)`` to
    capture both endpoint features — useful for topology-aware tasks.
    """

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
        )

    def forward(self, h, edge_index, e):
        src, dst = edge_index[0], edge_index[1]
        if src.numel() == 0:
            return torch.zeros_like(h)

        # Messages: concat(h_src, h_dst, e)
        msg_input = torch.cat([h[src], h[dst], e], dim=1)
        messages = self.msg_mlp(msg_input)
        messages = torch.nan_to_num(messages, nan=0.0, posinf=0.0, neginf=0.0)

        # Aggregate (sum)
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)

        # Update
        h_new = self.update_mlp(torch.cat([h, agg], dim=1))
        return h_new
