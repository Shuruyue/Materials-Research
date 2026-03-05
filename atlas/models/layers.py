"""
GNN Layers

Neural network layers for crystal property prediction:
- MessagePassingLayer: invariant message passing (CGCNN)
- GatedEquivariantBlock: E(3)-equivariant convolution (e3nn)
"""

import math
from numbers import Integral, Real

import torch
import torch.nn as nn


def _is_boolean_like(value: object) -> bool:
    return isinstance(value, bool) or type(value).__name__ == "bool_"


def _coerce_positive_int(value: object, name: str) -> int:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be integer-valued, not boolean")
    if isinstance(value, Integral):
        integer = int(value)
    elif isinstance(value, Real):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        rounded = round(scalar)
        if abs(scalar - rounded) > 1e-9:
            raise ValueError(f"{name} must be integer-valued")
        integer = int(rounded)
    else:
        raise ValueError(f"{name} must be integer-valued")
    if integer <= 0:
        raise ValueError(f"{name} must be > 0")
    return integer


def _coerce_positive_float(value: object, name: str) -> float:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be > 0, not boolean")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if scalar <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return scalar


class MessagePassingLayer(nn.Module):
    """
    Invariant message passing layer.

    Aggregates neighbor features weighted by edge features.
    Used in CGCNN-style models where rotational equivariance is not required.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        *,
        aggr: str = "mean",
        use_edge_gates: bool = True,
    ):
        super().__init__()
        node_dim_i = _coerce_positive_int(node_dim, "node_dim")
        edge_dim_i = _coerce_positive_int(edge_dim, "edge_dim")
        if aggr not in {"sum", "mean"}:
            raise ValueError(f"Unsupported aggregation: {aggr}")
        self.aggr = aggr
        self.use_edge_gates = use_edge_gates

        msg_in = node_dim_i * 2 + edge_dim_i
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_in, node_dim_i),
            nn.SiLU(),
            nn.Linear(node_dim_i, node_dim_i),
        )
        self.gate_mlp = nn.Linear(msg_in, node_dim_i) if use_edge_gates else None
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim_i * 2, node_dim_i),
            nn.SiLU(),
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
        if h.ndim != 2:
            raise ValueError(f"h must be rank-2 tensor, got shape {tuple(h.shape)}")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"edge_index must be integer tensor, got dtype {edge_index.dtype}")
        if e.ndim != 2:
            raise ValueError(f"e must be rank-2 tensor, got shape {tuple(e.shape)}")
        if e.size(0) != edge_index.size(1):
            raise ValueError(
                f"Edge feature count mismatch: got {e.size(0)}, expected {edge_index.size(1)}"
            )

        src, dst = edge_index[0].to(dtype=torch.long), edge_index[1].to(dtype=torch.long)
        if src.numel() == 0:
            agg = torch.zeros_like(h)
            return self.update_mlp(torch.cat([h, agg], dim=-1))
        if int(src.min()) < 0 or int(src.max()) >= h.size(0):
            raise ValueError("edge_index src contains out-of-range node indices")
        if int(dst.min()) < 0 or int(dst.max()) >= h.size(0):
            raise ValueError("edge_index dst contains out-of-range node indices")

        msg_input = torch.cat([h[src], h[dst], e], dim=-1)
        messages = self.msg_mlp(msg_input)
        if self.gate_mlp is not None:
            gates = torch.sigmoid(self.gate_mlp(msg_input))
            messages = messages * gates

        # Aggregate: sum messages per target node
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)
        if self.aggr == "mean":
            deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
            deg.index_add_(0, dst, torch.ones(dst.size(0), device=h.device, dtype=h.dtype))
            agg = agg / deg.clamp_min(1.0).unsqueeze(-1)

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
        n_radial_basis_i = _coerce_positive_int(n_radial_basis, "n_radial_basis")
        max_radius_f = _coerce_positive_float(max_radius, "max_radius")
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.irreps_edge = irreps_edge
        self.max_radius = max_radius_f
        self._initialized = False
        self._n_radial_basis = n_radial_basis_i

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

        except ImportError as e:
            raise ImportError(
                "e3nn is required for equivariant GNN. "
                "Install via: pip install e3nn"
            ) from e

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
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}")
        if edge_sh.ndim != 2 or edge_radial.ndim != 2:
            raise ValueError("edge_sh and edge_radial must be rank-2 tensors")
        if edge_sh.size(0) != edge_index.size(1) or edge_radial.size(0) != edge_index.size(1):
            raise ValueError("edge_sh/edge_radial row counts must match edge count")

        src, dst = edge_index
        src = src.long()
        dst = dst.long()
        if src.numel() == 0:
            return node_features
        if int(src.min()) < 0 or int(src.max()) >= node_features.size(0):
            raise ValueError("edge_index src contains out-of-range node indices")
        if int(dst.min()) < 0 or int(dst.max()) >= node_features.size(0):
            raise ValueError("edge_index dst contains out-of-range node indices")

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
