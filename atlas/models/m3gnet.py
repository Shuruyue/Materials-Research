"""
M3GNet-inspired Graph Neural Network

Implements a Many-Body Graph Neural Network with 3-body interactions (bond angles).
Based on: Chen & Ong, "A universal graph deep learning interatomic potential for the periodic table", Nature Computational Science (2022).

Key Features:
- Edge update based on bond distances (2-body)
- Angle update based on bond triplets (3-body)
- Gated updates for information flow control
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


def _coerce_bool(value: object, name: str) -> bool:
    if _is_boolean_like(value):
        return bool(value)
    if isinstance(value, Integral):
        integer = int(value)
        if integer in (0, 1):
            return bool(integer)
        raise ValueError(f"{name} integer value must be 0 or 1, got {integer}")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{name} must be boolean-like (bool, 0/1, true/false), got {value!r}")


class RBFExpansion(nn.Module):
    """Gaussian Radial Basis Function expansion."""
    def __init__(self, n_gaussians: int = 20, cutoff: float = 5.0):
        super().__init__()
        n_basis = _coerce_positive_int(n_gaussians, "n_gaussians")
        cutoff_value = _coerce_positive_float(cutoff, "cutoff")
        offset = torch.linspace(0, cutoff_value, n_basis)
        self.register_buffer("offset", offset)
        if n_basis > 1:
            width = 0.5 * float(offset[1] - offset[0])
        else:
            width = max(cutoff_value * 0.5, 1e-3)
        self.width = max(width, 1e-6)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        if dist.ndim != 1:
            dist = dist.reshape(-1)
        d = torch.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0).unsqueeze(-1)
        return torch.exp(-((d - self.offset) ** 2) / (self.width**2))



class ThreeBodyInteraction(nn.Module):
    """
    Computes 3-body interactions using bond angles.
    """
    def __init__(self, embed_dim: int, n_basis: int = 20, use_sh: bool = True):
        super().__init__()
        self.embed_dim = _coerce_positive_int(embed_dim, "embed_dim")
        self.n_basis = _coerce_positive_int(n_basis, "n_basis")
        self.use_sh = _coerce_bool(use_sh, "use_sh")

        # Basis for angles
        from .matgl_three_body import SimpleMLPAngleExpansion, SphericalBesselHarmonicsExpansion
        if use_sh:
            # MatGL style (Spherical Harmonics + Bessel Expansion)
            # max_n * max_l must equal n_basis
            max_n = 4
            max_l = max(1, self.n_basis // max_n)
            n_basis = _coerce_positive_int(max_n * max_l, "n_basis") # adjust if division wasn't perfect
            self.angle_expansion = SphericalBesselHarmonicsExpansion(max_n=max_n, max_l=max_l)
        else:
            # Simple Original MLP style
            self.angle_expansion = SimpleMLPAngleExpansion(self.n_basis)

        self.n_basis = _coerce_positive_int(n_basis, "n_basis") # Update n_basis in case it changed

        # Mixing weights
        self.phi_3b = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + self.n_basis, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Sigmoid()
        )

        self.update_edge = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        edge_attr: torch.Tensor,
        three_body_indices: torch.Tensor,     # (N_triplets, 3) [j, i, k]
        three_body_edge_indices: torch.Tensor, # (N_triplets, 2) [edge_ij, edge_ik]
        edge_vectors: torch.Tensor,           # (E, 3)
    ) -> torch.Tensor:
        """
        Args:
            edge_attr: (E, embed_dim) current edge embeddings
            three_body_indices: atoms involved in triplets (unused here, using edges)
            three_body_edge_indices: indices of edges forming the angle
            edge_vectors: vector describing edge i->j

        Returns:
            (E, embed_dim) contribution to edge features from 3-body
        """
        if edge_attr.ndim != 2:
            raise ValueError(f"edge_attr must be rank-2 tensor, got shape {tuple(edge_attr.shape)}")
        if edge_attr.size(1) != self.embed_dim:
            raise ValueError(
                f"edge_attr second dimension must match embed_dim={self.embed_dim}, got {edge_attr.size(1)}"
            )
        if three_body_edge_indices.ndim != 2 or three_body_edge_indices.size(1) != 2:
            raise ValueError(
                "three_body_edge_indices must have shape (T, 2) when triplets are provided"
            )
        if three_body_edge_indices.size(0) == 0:
            return torch.zeros_like(edge_attr)
        if three_body_edge_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"three_body_edge_indices must be integer tensor, got dtype {three_body_edge_indices.dtype}"
            )
        if edge_vectors.ndim != 2 or edge_vectors.size(1) != 3:
            raise ValueError(f"edge_vectors must have shape (E, 3), got {tuple(edge_vectors.shape)}")
        if edge_vectors.size(0) != edge_attr.size(0):
            raise ValueError("edge_vectors row count must match edge count")
        if not torch.isfinite(edge_attr).all():
            raise ValueError("edge_attr contains NaN or Inf values")
        if not torch.isfinite(edge_vectors).all():
            raise ValueError("edge_vectors contains NaN or Inf values")

        e_ij_idx = three_body_edge_indices[:, 0].long()
        e_ik_idx = three_body_edge_indices[:, 1].long()
        n_edges = int(edge_attr.size(0))
        if int(e_ij_idx.min().item()) < 0 or int(e_ij_idx.max().item()) >= n_edges:
            raise ValueError("three_body_edge_indices[:,0] contains out-of-range edge ids")
        if int(e_ik_idx.min().item()) < 0 or int(e_ik_idx.max().item()) >= n_edges:
            raise ValueError("three_body_edge_indices[:,1] contains out-of-range edge ids")

        # Calculate angles
        vec_ij = -edge_vectors[e_ij_idx] # Vector i->j
        vec_ik = -edge_vectors[e_ik_idx] # Vector i->k (wait, standard is center i? edge_vectors are j->i or i->j?)
        # Standard: edge_vectors[e] = coords[dst] - coords[src]. So src->dst.
        # If center is i (src), then we have i->j and i->k.
        # We want angle j-i-k.
        # Vectors are correct.

        length_ij = torch.norm(vec_ij, dim=1, keepdim=True).clamp(min=1e-6)
        length_ik = torch.norm(vec_ik, dim=1, keepdim=True).clamp(min=1e-6)

        cos_theta = (vec_ij * vec_ik).sum(dim=1, keepdim=True) / (length_ij * length_ik)
        cos_theta = cos_theta.clamp(-1, 1)

        # Expand angle (using either SH or MLP)
        if self.use_sh:
            angle_feats = self.angle_expansion(length_ij, length_ik, cos_theta) # (T, max_n * max_l)
        else:
            angle_feats = self.angle_expansion(length_ij, length_ik, cos_theta) # (T, n_basis)

        # Combine with edge features
        # We use edge features at the previous step (e_ij, e_ik)
        h_3b = torch.cat([edge_attr[e_ij_idx], edge_attr[e_ik_idx], angle_feats], dim=-1)
        h_3b = torch.nan_to_num(h_3b, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute 3-body message
        m_ijk = self.phi_3b(h_3b) # (T, embed_dim)

        # Filter by gate (optional, M3GNet uses sigmoidal gating)
        # m_ijk = m_ijk * self.gate(m_ijk)
        # (Simplified)

        # Aggregate back to edges i->j
        # For each triplet (j, i, k), it contributes to edge (i, j) update?
        # M3GNet: update e_ij with sum_k m_ijk
        delta_e_ij = torch.zeros_like(edge_attr)
        delta_e_ij.index_add_(0, e_ij_idx, m_ijk)

        # Also symmetric? e_ik update? Usually handled because (k, i, j) is another triplet.
        # Assuming builder produces all permutations.

        return torch.nan_to_num(self.update_edge(delta_e_ij), nan=0.0, posinf=0.0, neginf=0.0)


class M3GNetLayer(nn.Module):
    def __init__(self, embed_dim: int, n_basis: int = 20):
        super().__init__()
        self.embed_dim = _coerce_positive_int(embed_dim, "embed_dim")
        self.n_basis = _coerce_positive_int(n_basis, "n_basis")
        # 1. Edge Update (2-body)
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim), # atom_i + atom_j + edge
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.edge_gate = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.Sigmoid())

        # 2. Three-Body Update
        self.three_body = ThreeBodyInteraction(self.embed_dim, self.n_basis)
        self.three_body_gate = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.Sigmoid())

        # 3. Node Update
        self.node_mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim), # atom + aggregated_edges
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        three_body_indices: torch.Tensor,
        three_body_edge_indices: torch.Tensor,
        edge_vectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"edge_index must be integer tensor, got dtype {edge_index.dtype}")
        if edge_feats.ndim != 2 or node_feats.ndim != 2:
            raise ValueError("node_feats and edge_feats must be rank-2 tensors")
        if edge_feats.size(0) != edge_index.size(1):
            raise ValueError("edge feature count must match edge count")

        src, dst = edge_index
        src = src.long()
        dst = dst.long()
        if src.numel() > 0 and (int(src.min()) < 0 or int(src.max()) >= node_feats.size(0)):
            raise ValueError("edge_index src contains out-of-range node indices")
        if dst.numel() > 0 and (int(dst.min()) < 0 or int(dst.max()) >= node_feats.size(0)):
            raise ValueError("edge_index dst contains out-of-range node indices")

        # 1. Edge Update (2-body)
        edge_input = torch.cat([node_feats[src], node_feats[dst], edge_feats], dim=-1)
        e_prime = self.edge_mlp(edge_input) * self.edge_gate(edge_feats)
        edge_feats = edge_feats + e_prime # Residual

        # 2. Three-Body Interaction
        if three_body_edge_indices.size(0) > 0:
            e_3b = self.three_body(
                edge_feats, three_body_indices, three_body_edge_indices, edge_vectors
            )
            edge_feats = edge_feats + e_3b * self.three_body_gate(edge_feats)

        # 3. Node Update
        # Aggregate edges
        msg = torch.zeros_like(node_feats)
        msg.index_add_(0, dst, edge_feats)
        deg = torch.zeros(node_feats.size(0), dtype=node_feats.dtype, device=node_feats.device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), dtype=node_feats.dtype, device=node_feats.device))
        msg = msg / deg.clamp_min(1.0).unsqueeze(-1)

        node_input = torch.cat([node_feats, msg], dim=-1)
        v_prime = self.node_mlp(node_input)
        node_feats = node_feats + v_prime # Residual

        return node_feats, edge_feats


class M3GNet(nn.Module):
    """
    Main M3GNet Module.
    """
    def __init__(
        self,
        n_species: int = 86,
        embed_dim: int = 64,
        n_layers: int = 3,
        n_rbf: int = 20,
        max_radius: float = 5.0
    ):
        super().__init__()
        self.n_species = _coerce_positive_int(n_species, "n_species")
        self.embed_dim = _coerce_positive_int(embed_dim, "embed_dim")
        n_layers_i = _coerce_positive_int(n_layers, "n_layers")
        n_rbf_i = _coerce_positive_int(n_rbf, "n_rbf")
        max_radius_f = _coerce_positive_float(max_radius, "max_radius")

        self.embedding = nn.Embedding(self.n_species, self.embed_dim)
        self.rbf = RBFExpansion(n_gaussians=n_rbf_i, cutoff=max_radius_f)
        self.edge_embedding = nn.Linear(n_rbf_i, self.embed_dim)

        self.layers = nn.ModuleList([
            M3GNetLayer(self.embed_dim, n_basis=min(10, n_rbf_i)) for _ in range(n_layers_i)
        ])

        # Final pooling will be done by MultiTaskGNN or here if needed.
        # Assuming used as Encoder for MultiTaskGNN, we need to implement `encode`.

    def encode(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor, # This is usually RBF from builder, but M3GNet can recompute or use it.
        batch: torch.Tensor | None = None,
        edge_vectors: torch.Tensor | None = None,
        edge_index_3body: torch.Tensor | None = None, # (2, T)
        **kwargs
    ) -> torch.Tensor:
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"edge_index must be integer tensor, got dtype {edge_index.dtype}")
        if node_feats.ndim not in {1, 2}:
            raise ValueError(f"Unsupported node_feats shape: {tuple(node_feats.shape)}")
        if edge_attr.ndim != 2:
            raise ValueError(f"edge_attr must have shape (E, F), got {tuple(edge_attr.shape)}")
        if edge_attr.size(0) != edge_index.size(1):
            raise ValueError("edge_attr row count must match edge count")
        if edge_index.size(1) <= 0:
            raise ValueError("edge_index must contain at least one edge")
        n_nodes = int(node_feats.size(0))
        edge_min = int(edge_index.min().item())
        edge_max = int(edge_index.max().item())
        if edge_min < 0 or edge_max >= n_nodes:
            raise ValueError(
                f"edge_index contains out-of-range node ids: min={edge_min}, max={edge_max}, N={n_nodes}"
            )

        # 1. Embed Nodes
        if node_feats.ndim == 1:
            species_idx = node_feats.long()
        elif node_feats.size(1) == 1:
            species_idx = node_feats[:, 0].long()
        else:
            n_channels = min(node_feats.size(1), self.n_species)
            species_idx = node_feats[:, :n_channels].argmax(dim=-1).long()
        if species_idx.numel() > 0:
            idx_min = int(species_idx.min().item())
            idx_max = int(species_idx.max().item())
            if idx_min >= 1 and idx_max <= self.n_species:
                species_idx = species_idx - 1
        species_idx = species_idx.clamp(min=0, max=self.n_species - 1)
        x = self.embedding(species_idx)

        # 2. Embed Edges
        # edge_attr from builder is already RBF-expanded.
        edge_feat = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
        in_dim = self.edge_embedding.in_features
        if edge_feat.size(-1) != in_dim:
            if edge_feat.size(-1) > in_dim:
                edge_feat = edge_feat[:, :in_dim]
            else:
                pad = torch.zeros(
                    edge_feat.size(0),
                    in_dim - edge_feat.size(-1),
                    device=edge_feat.device,
                    dtype=edge_feat.dtype,
                )
                edge_feat = torch.cat([edge_feat, pad], dim=-1)
        e = self.edge_embedding(edge_feat)

        # Prepare 3-body indices
        # Graph builder gives (2, T) Transposed as proper PyG tensor
        if edge_index_3body is not None:
            if edge_index_3body.ndim != 2:
                raise ValueError("edge_index_3body must be rank-2 tensor")
            if edge_index_3body.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    f"edge_index_3body must be integer tensor, got dtype {edge_index_3body.dtype}"
                )
            if edge_index_3body.size(0) == 2:
                tb_edge_indices = edge_index_3body.T
            elif edge_index_3body.size(1) == 2:
                tb_edge_indices = edge_index_3body
            else:
                raise ValueError(
                    f"edge_index_3body must have shape (2,T) or (T,2), got {tuple(edge_index_3body.shape)}"
                )
            tb_edge_indices = tb_edge_indices.long()
            if tb_edge_indices.numel() > 0:
                max_edge = int(edge_index.size(1))
                if int(tb_edge_indices.min().item()) < 0 or int(tb_edge_indices.max().item()) >= max_edge:
                    raise ValueError(
                        "edge_index_3body contains out-of-range edge ids for current edge_index"
                    )
            tb_indices = None
        else:
            tb_edge_indices = torch.empty((0, 2), dtype=torch.long, device=x.device)
            tb_indices = None

        if edge_vectors is None:
            # Cannot compute angles without vectors
            # Fallback to pure 2-body
            tb_edge_indices = torch.empty((0, 2), dtype=torch.long, device=x.device)
            edge_vectors = torch.empty((0, 3), dtype=x.dtype, device=x.device)
        else:
            if edge_vectors.ndim != 2 or edge_vectors.size(1) != 3:
                raise ValueError(f"edge_vectors must have shape (E,3), got {tuple(edge_vectors.shape)}")
            if edge_vectors.size(0) != edge_index.size(1):
                raise ValueError("edge_vectors row count must match edge count")
            edge_vectors = torch.nan_to_num(edge_vectors, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Message Passing
        for layer in self.layers:
            x, e = layer(x, e, edge_index, tb_indices, tb_edge_indices, edge_vectors)

        # 4. Global Pool
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            if batch.ndim != 1:
                raise ValueError(f"batch must be 1D [N], got shape {tuple(batch.shape)}")
            if batch.shape[0] != x.shape[0]:
                raise ValueError(
                    f"batch size must match number of nodes ({x.shape[0]}), got {batch.shape[0]}"
                )
            if batch.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"batch must be integer tensor, got dtype {batch.dtype}")
            if batch.numel() > 0 and int(batch.min().item()) < 0:
                raise ValueError("batch indices must be non-negative")
            batch = batch.long()

        from torch_geometric.nn import global_mean_pool
        return global_mean_pool(x, batch)

    def forward(self, *args, **kwargs):
        # Compatibility with MultiTaskGNN which calls encode
        return self.encode(*args, **kwargs)
