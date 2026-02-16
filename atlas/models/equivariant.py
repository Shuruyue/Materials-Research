"""
E(3)-Equivariant Graph Neural Network

NequIP-inspired architecture using e3nn for crystal property prediction.
Respects rotational and translational symmetry of crystal structures,
enabling physically consistent predictions.

Architecture:
    Species embedding (scalars, 0e)
    → Radial basis (Bessel) + Spherical harmonics (edge)
    → N × Equivariant Interaction blocks (TensorProduct + Gate)
    → Invariant pooling (extract 0e scalars)
    → MLP → Property prediction

Reference:
    - NequIP: Batzner et al., Nature Comms 2022
    - e3nn: Geiger & Smidt, arXiv:2207.09453
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Dict


# ─────────────────────────────────────────────────────────
#  Radial Basis Functions
# ─────────────────────────────────────────────────────────

class BesselRadialBasis(nn.Module):
    """
    Bessel radial basis functions with smooth polynomial cutoff.

    More physically motivated than Gaussian expansion:
    φ_n(r) = sqrt(2/r_c) * sin(nπr/r_c) / r * f_cut(r)

    Args:
        r_max: cutoff radius in Angstroms
        n_basis: number of Bessel basis functions
        trainable: whether basis frequencies are trainable
    """

    def __init__(self, r_max: float = 5.0, n_basis: int = 8, trainable: bool = True):
        super().__init__()
        self.r_max = r_max
        self.n_basis = n_basis

        # Bessel function frequencies: n*π/r_max
        freqs = torch.arange(1, n_basis + 1).float() * math.pi / r_max
        if trainable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: (E,) edge distances in Angstroms

        Returns:
            (E, n_basis) radial basis values
        """
        d = distances.unsqueeze(-1)  # (E, 1)

        # Bessel functions
        basis = torch.sin(self.freqs * d) / d.clamp(min=1e-6)  # (E, n_basis)

        # Smooth polynomial cutoff: p(r) = 1 - 6(r/rc)^5 + 15(r/rc)^4 - 10(r/rc)^3
        x = distances / self.r_max
        cutoff = 1.0 - 6.0 * x**5 + 15.0 * x**4 - 10.0 * x**3
        cutoff = cutoff.unsqueeze(-1)  # (E, 1)

        return basis * cutoff


class RadialMLP(nn.Module):
    """
    MLP that maps radial basis to tensor product weights.

    Takes Bessel basis values and produces weights for the
    FullyConnectedTensorProduct.

    Args:
        n_basis: number of input radial basis functions
        n_out: number of output weights (determined by tensor product)
        hidden_dim: intermediate MLP width
    """

    def __init__(self, n_basis: int, n_out: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_basis, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_out),
        )

    def forward(self, basis: torch.Tensor) -> torch.Tensor:
        return self.net(basis)


# ─────────────────────────────────────────────────────────
#  Interaction Block
# ─────────────────────────────────────────────────────────

class InteractionBlock(nn.Module):
    """
    Equivariant interaction block with gated activation.

    Performs message passing with e3nn tensor products:
    1. Tensor product: node_features ⊗ edge_spherical_harmonics
       weighted by radial MLP
    2. Gated nonlinearity: scalar gate controls L>0 irreps
    3. Self-interaction (linear mixing of irreps)
    4. e3nn BatchNorm (stable normalization)
    5. Residual connection (if irreps match)

    Args:
        irreps_in: input irreducible representations
        irreps_out: output irreducible representations
        irreps_sh: spherical harmonics irreps for edges
        n_radial_basis: number of radial basis functions
        hidden_dim: radial MLP hidden dimension
    """

    def __init__(
        self,
        irreps_in: str,
        irreps_out: str,
        irreps_sh: str,
        n_radial_basis: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        from e3nn import o3, nn as e3nn_nn

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)

        # ── Gate activation setup ──
        # For each non-scalar irrep in output, we need an extra scalar as gate
        gate_scalars = []   # scalar irreps for gating
        gated_irreps = []   # non-scalar irreps being gated
        act_scalars = []    # activations for scalar channels
        act_gates = []      # activations for gate channels

        for mul, ir in self.irreps_out:
            if ir.l == 0:
                gate_scalars.append((mul, ir))
                act_scalars.append(torch.nn.SiLU())
            else:
                gated_irreps.append((mul, ir))
                # Each gated irrep needs 'mul' scalar gates
                gate_scalars.append((mul, (0, 1)))  # 0e
                act_gates.append(torch.sigmoid)

        irreps_scalars = o3.Irreps(gate_scalars[:len([m for m, ir in self.irreps_out if ir.l == 0])])
        irreps_gate_scalars = o3.Irreps(gate_scalars[len([m for m, ir in self.irreps_out if ir.l == 0]):])
        irreps_gated = o3.Irreps(gated_irreps)

        # TP output needs: original scalars + gate scalars + gated non-scalars
        self.irreps_tp_out = irreps_scalars + irreps_gate_scalars + irreps_gated

        # Build Gate if there are non-scalar irreps
        if len(irreps_gated) > 0 and len(act_gates) > 0:
            self.gate = e3nn_nn.Gate(
                irreps_scalars, act_scalars,
                irreps_gate_scalars, act_gates,
                irreps_gated,
            )
            # Gate output should match irreps_out
        else:
            self.gate = None

        # Tensor product: node ⊗ edge → message
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_tp_out,
            internal_weights=False,
            shared_weights=False,
        )

        # Radial MLP: produces weights for tensor product
        self.radial_mlp = RadialMLP(
            n_basis=n_radial_basis,
            n_out=self.tp.weight_numel,
            hidden_dim=hidden_dim,
        )

        # Self-interaction: linear mixing of irreps (like 1×1 convolution)
        self.self_interaction = o3.Linear(self.irreps_out, self.irreps_out)

        # BatchNorm: e3nn's proven equivariant normalization
        # - L=0: standard batch norm (affine)
        # - L>0: normalize by RMS of the norm, no shift (preserves equivariance)
        self.batch_norm = e3nn_nn.BatchNorm(self.irreps_out)

        # Residual connection possible if irreps match
        self.residual = (self.irreps_in == self.irreps_out)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_basis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (N, irreps_in_dim) node features
            edge_index: (2, E) source-target indices
            edge_sh: (E, irreps_sh_dim) spherical harmonics of edges
            edge_basis: (E, n_radial_basis) radial basis values

        Returns:
            (N, irreps_out_dim) updated node features
        """
        src, dst = edge_index

        # Compute tensor product weights from radial basis
        tp_weights = self.radial_mlp(edge_basis)  # (E, weight_numel)

        # Tensor product: source_node ⊗ edge_sh, weighted by radial
        messages = self.tp(node_features[src], edge_sh, tp_weights)  # (E, irreps_tp_out_dim)

        # Aggregate messages per target node
        agg = torch.zeros(
            node_features.size(0), messages.size(-1),
            device=messages.device, dtype=messages.dtype,
        )
        agg.index_add_(0, dst, messages)

        # Gated activation (SiLU for scalars, sigmoid·value for L>0)
        if self.gate is not None:
            agg = self.gate(agg)
        else:
            # Fallback: SiLU on all (scalars only)
            agg = torch.nn.functional.silu(agg)

        # Self-interaction
        out = self.self_interaction(agg)

        # Batch normalization (equivariant)
        out = self.batch_norm(out)

        # Residual connection if shapes match
        if self.residual:
            out = out + node_features

        return out


# ─────────────────────────────────────────────────────────
#  Main Model
# ─────────────────────────────────────────────────────────

class EquivariantGNN(nn.Module):
    """
    E(3)-equivariant GNN for crystal property prediction.

    NequIP-inspired architecture:
        1. Species embedding → scalar features (L=0)
        2. Edge: spherical harmonics Y_l(r̂) + Bessel radial basis φ_n(|r|)
        3. N interaction blocks with tensor products
        4. Extract invariant (scalar) features via pooling
        5. MLP property head

    Args:
        irreps_hidden: hidden irreducible representations
            e.g., "32x0e + 16x1o + 8x2e"
        max_ell: maximum spherical harmonic order
        n_layers: number of interaction layers
        max_radius: cutoff radius in Angstroms
        n_species: number of chemical species
        n_radial_basis: number of Bessel basis functions
        radial_hidden: hidden dimension of radial MLP
        output_dim: final output dimension
        embed_dim: species embedding dimension (must match scalar count in irreps_hidden)
    """

    def __init__(
        self,
        irreps_hidden: str = "32x0e + 16x1o + 8x2e",
        max_ell: int = 2,
        n_layers: int = 3,
        max_radius: float = 5.0,
        n_species: int = 86,
        n_radial_basis: int = 8,
        radial_hidden: int = 64,
        output_dim: int = 1,
        embed_dim: Optional[int] = None,
    ):
        super().__init__()
        from e3nn import o3

        self.max_ell = max_ell
        self.n_layers = n_layers
        self.max_radius = max_radius
        self.n_species = n_species
        self.irreps_hidden_str = irreps_hidden

        irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_hidden = irreps_hidden

        # Count scalar multiplicity for embedding dimension
        scalar_mul = sum(mul for mul, ir in irreps_hidden if ir.l == 0)
        self._embed_dim = embed_dim or scalar_mul

        # ── Species embedding ──
        # Maps atomic number to scalar features that seed the irreps
        self.species_embed = nn.Embedding(n_species, self._embed_dim)

        # Linear projection from scalar embedding to full irreps
        # Scalars go into 0e slots, higher-L initialized to zero
        irreps_input = o3.Irreps(f"{self._embed_dim}x0e")
        self.input_proj = o3.Linear(irreps_input, irreps_hidden)

        # ── Edge features ──
        self.sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.radial_basis = BesselRadialBasis(
            r_max=max_radius, n_basis=n_radial_basis
        )

        # ── Interaction blocks ──
        self.interactions = nn.ModuleList()
        for _ in range(n_layers):
            block = InteractionBlock(
                irreps_in=str(irreps_hidden),
                irreps_out=str(irreps_hidden),
                irreps_sh=str(self.sh_irreps),
                n_radial_basis=n_radial_basis,
                hidden_dim=radial_hidden,
            )
            self.interactions.append(block)

        # ── Output head ──
        # Extract only scalar (0e) features for invariant prediction
        self.scalar_dim = scalar_mul
        self.output_head = nn.Sequential(
            nn.Linear(scalar_mul, scalar_mul),
            nn.SiLU(),
            nn.Linear(scalar_mul, scalar_mul // 2),
            nn.SiLU(),
            nn.Linear(scalar_mul // 2, output_dim),
        )

    def _extract_scalars(self, features: torch.Tensor) -> torch.Tensor:
        """Extract L=0 (scalar) components from irreps features."""
        scalars = []
        idx = 0
        for mul, ir in self.irreps_hidden:
            dim = ir.dim
            if ir.l == 0:
                scalars.append(features[:, idx:idx + mul * dim])
            idx += mul * dim
        return torch.cat(scalars, dim=-1)

    def encode(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode crystal to graph-level invariant embedding.

        Args:
            node_feats: (N, node_dim) — species index extracted from first column
            edge_index: (2, E) edge indices
            edge_vectors: (E, 3) relative displacement vectors
            batch: (N,) batch assignment

        Returns:
            (B, scalar_dim) graph-level invariant embedding
        """
        from e3nn import o3

        # ── Species embedding ──
        # Extract species index from one-hot encoded node features
        species_idx = node_feats[:, :86].argmax(dim=-1)  # (N,)
        h_scalar = self.species_embed(species_idx)  # (N, embed_dim)
        h = self.input_proj(h_scalar)  # (N, irreps_hidden_dim)

        # ── Edge features ──
        distances = edge_vectors.norm(dim=-1)  # (E,)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps,
            edge_vectors,
            normalize=True,
            normalization="component",
        )  # (E, sh_dim)
        edge_basis = self.radial_basis(distances)  # (E, n_basis)

        # ── Interaction blocks ──
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_sh, edge_basis)

        # ── Extract scalar features ──
        h_scalars = self._extract_scalars(h)  # (N, scalar_dim)

        # ── Global mean pooling ──
        if batch is None:
            batch = torch.zeros(h_scalars.size(0), dtype=torch.long, device=h_scalars.device)

        from torch_geometric.nn import global_mean_pool
        graph_emb = global_mean_pool(h_scalars, batch)  # (B, scalar_dim)

        return graph_emb

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: crystal graph → predicted property.

        Args:
            node_feats: (N, node_dim) node features (one-hot + properties)
            edge_index: (2, E) edge indices
            edge_vectors: (E, 3) displacement vectors (NOT Gaussian distances)
            batch: (N,) batch assignment

        Returns:
            (B, output_dim) predicted property values
        """
        graph_emb = self.encode(node_feats, edge_index, edge_vectors, batch)
        return self.output_head(graph_emb)
