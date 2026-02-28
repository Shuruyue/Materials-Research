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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Configure logging
logger = logging.getLogger(__name__)



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

    def forward(self, node_feats, edge_index, edge_feats, batch=None):
        """Forward pass."""
        h = self.node_embed(node_feats)
        e = self.edge_embed(edge_feats)

        # Message passing
        for mp, norm in zip(self.mp_layers, self.norms, strict=True):
            h_new = mp(h, edge_index, e)
            h = norm(h + h_new)  # residual + layernorm

        # Global pooling
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        # Mean and max pooling
        n_graphs = batch.max().item() + 1
        h_mean = torch.zeros(n_graphs, h.size(1), device=h.device)
        h_max = torch.zeros(n_graphs, h.size(1), device=h.device)

        h_mean.scatter_reduce_(0, batch.unsqueeze(1).expand_as(h), h, reduce="mean")

        h_max_vals = torch.full_like(h_max, float("-inf"))
        h_max_vals.scatter_reduce_(0, batch.unsqueeze(1).expand_as(h), h, reduce="amax")
        h_max = h_max_vals

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
        if not mc_dropout:
            self.eval()
            with torch.no_grad():
                logits = self.forward(node_feats, edge_index, edge_feats, batch)
                return torch.sigmoid(logits).item()
        else:
            # MC Dropout: keep dropout active during inference
            self.train() # Enable dropout
            probs = []
            with torch.no_grad():
                for _ in range(n_samples):
                    logits = self.forward(node_feats, edge_index, edge_feats, batch)
                    probs.append(torch.sigmoid(logits).item())

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

        # Messages: concat(h_src, h_dst, e)
        msg_input = torch.cat([h[src], h[dst], e], dim=1)
        messages = self.msg_mlp(msg_input)

        # Aggregate (sum)
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)

        # Update
        h_new = self.update_mlp(torch.cat([h, agg], dim=1))
        return h_new
