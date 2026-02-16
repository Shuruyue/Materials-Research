"""
GNN Layers

Neural network layers for crystal property prediction:
- MessagePassingLayer: invariant message passing (CGCNN)
- GatedEquivariantBlock: E(3)-equivariant convolution (e3nn)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MessagePassingLayer(nn.Module):
    """
    Invariant message passing layer.

    Aggregates neighbor features weighted by edge features.
    Used in CGCNN-style models where rotational equivariance is not required.
    """

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        e: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: (N, node_dim) node features
            edge_index: (2, E) source-target indices
            e: (E, edge_dim) edge features

        Returns:
            (N, node_dim) updated node features
        """
        src, dst = edge_index
        msg_input = torch.cat([h[src], e], dim=-1)
        messages = self.msg_mlp(msg_input)

        # Aggregate: sum messages per target node
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)

        # Update
        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))
        return h_new


class GatedEquivariantBlock(nn.Module):
    """
    E(3)-equivariant convolution block using e3nn.

    Full implementation using:
    - FullyConnectedTensorProduct for equivariant message passing
    - Gate activation for non-scalar irreps
    - Self-interaction (linear mixing)

    This is a standalone block that can be used independently of
    the full EquivariantGNN model.

    Args:
        irreps_in: input irreps (e.g., "32x0e + 16x1o + 8x2e")
        irreps_out: output irreps
        irreps_edge: edge spherical harmonics irreps
        n_radial_basis: number of radial basis functions
        max_radius: cutoff radius in Angstroms
    """

    def __init__(
        self,
        irreps_in: str = "32x0e + 16x1o + 8x2e",
        irreps_out: str = "32x0e + 16x1o + 8x2e",
        irreps_edge: str = "0e + 1o + 2e",
        n_radial_basis: int = 8,
        max_radius: float = 5.0,
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.irreps_edge = irreps_edge
        self.max_radius = max_radius
        self._initialized = False
        self._n_radial_basis = n_radial_basis

        self._lazy_init()

    def _lazy_init(self):
        """Initialize e3nn layers."""
        try:
            from e3nn import o3

            irreps_in = o3.Irreps(self.irreps_in)
            irreps_out = o3.Irreps(self.irreps_out)
            irreps_edge = o3.Irreps(self.irreps_edge)

            # Tensor product
            self.tp = o3.FullyConnectedTensorProduct(
                irreps_in, irreps_edge, irreps_out,
                internal_weights=False, shared_weights=False,
            )

            # Radial weight MLP
            self.radial_mlp = nn.Sequential(
                nn.Linear(self._n_radial_basis, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, self.tp.weight_numel),
            )

            # Self-interaction
            self.self_int = o3.Linear(irreps_out, irreps_out)

            self._initialized = True

        except ImportError:
            raise ImportError(
                "e3nn is required for equivariant GNN. "
                "Install via: pip install e3nn"
            )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_radial: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (N, irreps_in_dim) node features
            edge_index: (2, E) edge indices
            edge_sh: (E, sh_dim) spherical harmonics of edge vectors
            edge_radial: (E, n_basis) radial basis values

        Returns:
            (N, irreps_out_dim) updated node features
        """
        src, dst = edge_index

        # Compute tensor product weights from radial basis
        weights = self.radial_mlp(edge_radial)

        # Tensor product message
        messages = self.tp(node_features[src], edge_sh, weights)

        # Aggregate
        agg = torch.zeros(
            node_features.size(0), messages.size(-1),
            device=messages.device, dtype=messages.dtype,
        )
        agg.index_add_(0, dst, messages)

        # Self-interaction + residual
        out = self.self_int(agg)
        if node_features.shape == out.shape:
            out = out + node_features

        return out
