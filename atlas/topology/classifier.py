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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrystalGraphBuilder:
    """
    Converts a crystal structure to a graph representation.
    
    Nodes: atoms (features = one-hot element + atomic properties)
    Edges: bonds within cutoff radius (features = distance + direction)
    """

    # Common elements in topological materials
    ELEMENTS = [
        "H", "Li", "Be", "B", "C", "N", "O", "F",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl",
        "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
        "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd",
        "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
        "Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
        "Au", "Hg", "Tl", "Pb", "Bi",
    ]
    ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}

    def __init__(self, cutoff: float = 5.0, max_neighbors: int = 12):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

    def element_features(self, symbol: str) -> np.ndarray:
        """
        Get element features: one-hot encoding + physical properties.
        """
        idx = self.ELEMENT_TO_IDX.get(symbol, len(self.ELEMENTS) - 1)
        one_hot = np.zeros(len(self.ELEMENTS))
        one_hot[idx] = 1.0

        # Basic atomic properties
        try:
            from pymatgen.core import Element
            elem = Element(symbol)
            atomic_num = elem.Z / 100.0  # normalize
            electronegativity = (elem.X or 2.0) / 4.0  # normalize
            atomic_radius = (elem.atomic_radius or 1.5) / 3.0  # normalize
            is_metal = 1.0 if elem.is_metal else 0.0
            is_heavy = 1.0 if elem.Z >= 50 else 0.0
        except Exception:
            atomic_num = 0.5
            electronegativity = 0.5
            atomic_radius = 0.5
            is_metal = 0.0
            is_heavy = 0.0

        return np.concatenate([
            one_hot,
            [atomic_num, electronegativity, atomic_radius, is_metal, is_heavy],
        ]).astype(np.float32)

    def structure_to_graph(self, structure) -> dict:
        """
        Convert pymatgen Structure to graph dict.
        """
        n_atoms = len(structure)

        # Node features
        node_feats = []
        for site in structure:
            feats = self.element_features(str(site.specie))
            node_feats.append(feats)
        node_feats = np.array(node_feats)

        # Edge features (bonds within cutoff)
        src_list, dst_list, dist_list = [], [], []

        for i in range(n_atoms):
            neighbors = structure.get_neighbors(structure[i], self.cutoff)
            # Sort by distance and take nearest max_neighbors
            neighbors.sort(key=lambda x: x.nn_distance)
            for nn in neighbors[:self.max_neighbors]:
                j = nn.index
                d = nn.nn_distance
                src_list.append(i)
                dst_list.append(j)
                dist_list.append(d)

        if len(src_list) == 0:
            # Fallback: at least self-loops
            src_list = list(range(n_atoms))
            dst_list = list(range(n_atoms))
            dist_list = [0.0] * n_atoms

        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        
        # Edge features: Gaussian expansion of distance
        distances = np.array(dist_list, dtype=np.float32)
        edge_feats = self._gaussian_expansion(distances)

        return {
            "node_features": torch.FloatTensor(node_feats),
            "edge_index": torch.LongTensor(edge_index),
            "edge_features": torch.FloatTensor(edge_feats),
            "num_nodes": n_atoms,
        }

    def _gaussian_expansion(
        self, distances: np.ndarray, n_gaussians: int = 20
    ) -> np.ndarray:
        """Expand distances into Gaussian basis functions."""
        centers = np.linspace(0, self.cutoff, n_gaussians)
        width = 0.5 * (centers[1] - centers[0])
        expanded = np.exp(-((distances[:, None] - centers[None, :]) ** 2) / (2 * width ** 2))
        return expanded.astype(np.float32)


class TopoGNN(nn.Module):
    """
    Graph Neural Network for topological material classification.
    """

    def __init__(
        self,
        node_dim: int = 69,   # 64 one-hot + 5 properties
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
            self.mp_layers.append(MessagePassingLayer(hidden_dim, hidden_dim))
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
        for mp, norm in zip(self.mp_layers, self.norms):
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
    ) -> Union[float, Tuple[float, float]]:
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

    def save_model(self, path: Union[str, Path]):
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
    def load_model(cls, path: Union[str, Path], device: str = "cpu") -> "TopoGNN":
        """Load model from file."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint.get('config', {})
        
        # Instantiate with saved config
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded from {path}")
        return model


class MessagePassingLayer(nn.Module):
    """Simple message passing: aggregate neighbor features weighted by edge features."""

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
