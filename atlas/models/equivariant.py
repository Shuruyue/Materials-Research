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
from numbers import Integral, Real

import torch
import torch.nn as nn

# Common architecture presets used by phase scripts.
SMALL_PRESET = {
    "irreps": "16x0e + 8x1o",
    "max_ell": 1,
    "n_layers": 2,
    "n_radial": 10,
    "radial_hidden": 32,
}

MEDIUM_PRESET = {
    "irreps": "64x0e + 32x1o + 16x2e",
    "max_ell": 2,
    "n_layers": 3,
    "n_radial": 20,
    "radial_hidden": 128,
}

LARGE_PRESET = {
    "irreps": "128x0e + 64x1o + 32x2e",
    "max_ell": 2,
    "n_layers": 4,
    "n_radial": 32,
    "radial_hidden": 256,
}


def _is_boolean_like(value: object) -> bool:
    return isinstance(value, bool) or type(value).__name__ == "bool_"


def _coerce_non_negative_int(value: object, name: str) -> int:
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
    if integer < 0:
        raise ValueError(f"{name} must be >= 0")
    return integer


def _coerce_positive_int(value: object, name: str) -> int:
    integer = _coerce_non_negative_int(value, name)
    if integer <= 0:
        raise ValueError(f"{name} must be > 0")
    return integer


def _coerce_positive_float(value: object, name: str) -> float:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be finite and > 0, not boolean")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if scalar <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return scalar


def _infer_species_indices(node_feats: torch.Tensor, n_species: int) -> torch.Tensor:
    """
    Infer per-node species indices from common node feature layouts.

    Supported inputs:
    - `(N,)` integer indices
    - `(N, 1)` integer-like indices
    - `(N, C)` one-hot / logits (uses argmax over first `min(C, n_species)` channels)
    """
    if node_feats.ndim == 1:
        idx = node_feats.long()
    elif node_feats.ndim == 2 and node_feats.size(1) == 1:
        idx = node_feats[:, 0].long()
    elif node_feats.ndim == 2:
        n_channels = int(min(node_feats.size(1), n_species))
        if n_channels <= 0:
            raise ValueError("node_feats has no valid channels to infer species indices")
        idx = node_feats[:, :n_channels].argmax(dim=-1).long()
    else:
        raise ValueError(f"Unsupported node_feats shape: {tuple(node_feats.shape)}")

    # Handle one-based atomic number style input (1..n_species).
    if idx.numel() > 0:
        idx_min = int(idx.min().item())
        idx_max = int(idx.max().item())
        if idx_min >= 1 and idx_max <= n_species:
            idx = idx - 1
    return idx.clamp(min=0, max=max(n_species - 1, 0))

class AtomRef(nn.Module):
    """
    Atomic Reference Energy.

    Learnable per-species offset that is added to the final prediction.
    Helps the model capture systematic species-dependent energy shifts.
    """
    def __init__(self, n_species: int = 86, output_dim: int = 1):
        super().__init__()
        self.n_species = _coerce_positive_int(n_species, "n_species")
        self.output_dim = _coerce_positive_int(output_dim, "output_dim")
        # Initialize with zeros, let the model learn the offsets
        self.ref = nn.Embedding(self.n_species, self.output_dim)
        nn.init.zeros_(self.ref.weight)

    def forward(self, node_feats: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            node_feats: (N, node_dim) — species index extracted from first column
            batch: (N,) or None
        Returns:
            (B, output_dim) summed atomic reference energy per graph
        """
        species_idx = _infer_species_indices(node_feats, self.n_species)

        # Get per-atom reference values
        atom_refs = self.ref(species_idx) # (N, output_dim)

        # Sum per graph
        if batch is None:
            batch = torch.zeros(atom_refs.size(0), dtype=torch.long, device=atom_refs.device)

        from torch_geometric.nn import global_add_pool
        return global_add_pool(atom_refs, batch) # (B, output_dim)



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
        cutoff = _coerce_positive_float(r_max, "r_max")
        basis_count = _coerce_positive_int(n_basis, "n_basis")
        self.r_max = cutoff
        self.n_basis = basis_count

        # Bessel function frequencies: n*π/r_max
        freqs = torch.arange(1, basis_count + 1).float() * math.pi / cutoff
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
        if distances.ndim != 1:
            distances = distances.reshape(-1)
        distances = torch.nan_to_num(distances, nan=0.0, posinf=self.r_max, neginf=0.0).clamp_min(0.0)
        d = distances.unsqueeze(-1)  # (E, 1)

        # Bessel functions
        basis = torch.sin(self.freqs * d) / d.clamp(min=1e-6)  # (E, n_basis)

        # Smooth polynomial cutoff: p(r) = 1 - 6(r/rc)^5 + 15(r/rc)^4 - 10(r/rc)^3
        x = (distances / self.r_max).clamp(min=0.0, max=1.0)
        cutoff = 1.0 - 6.0 * x**5 + 15.0 * x**4 - 10.0 * x**3
        cutoff = torch.where(distances <= self.r_max, cutoff, torch.zeros_like(cutoff))
        cutoff = cutoff.unsqueeze(-1)  # (E, 1)

        return torch.nan_to_num(basis * cutoff, nan=0.0, posinf=0.0, neginf=0.0)


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
        from e3nn import nn as e3nn_nn
        from e3nn import o3

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

        # Tensor product: node ⊗ edge → message (Fused for VRAM optimization)
        from .fast_tp import FusedTensorProductScatter
        self.tp = FusedTensorProductScatter(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_tp_out,
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
        # Using the unified FusedTensorProductScatter (NequIP style)
        messages = self.tp(
            x=node_features,
            edge_attr=edge_sh,
            edge_weight=tp_weights,
            edge_src=src,
            edge_dst=dst,
            num_nodes=node_features.size(0)
        )

        agg = messages

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
        embed_dim: int | None = None,
    ):
        super().__init__()
        from e3nn import o3

        self.max_ell = _coerce_non_negative_int(max_ell, "max_ell")
        self.n_layers = _coerce_positive_int(n_layers, "n_layers")
        self.max_radius = _coerce_positive_float(max_radius, "max_radius")
        self.n_species = _coerce_positive_int(n_species, "n_species")
        output_dim_i = _coerce_positive_int(output_dim, "output_dim")
        n_radial_basis_i = _coerce_positive_int(n_radial_basis, "n_radial_basis")
        radial_hidden_i = _coerce_positive_int(radial_hidden, "radial_hidden")
        self.irreps_hidden_str = irreps_hidden

        irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_hidden = irreps_hidden

        # Count scalar multiplicity for embedding dimension
        scalar_mul = sum(mul for mul, ir in irreps_hidden if ir.l == 0)
        if scalar_mul <= 0:
            raise ValueError("irreps_hidden must include at least one scalar (0e) channel")
        self._embed_dim = (
            scalar_mul if embed_dim is None else _coerce_positive_int(embed_dim, "embed_dim")
        )

        # ── AtomRef (Learnable Atomic Reference) ──
        # Adds a baseline energy based on composition
        self.atom_ref = AtomRef(n_species=self.n_species, output_dim=output_dim_i)

        # ── Species embedding ──
        # Maps atomic number to scalar features that seed the irreps
        self.species_embed = nn.Embedding(self.n_species, self._embed_dim)

        # Linear projection from scalar embedding to full irreps
        # Scalars go into 0e slots, higher-L initialized to zero
        irreps_input = o3.Irreps(f"{self._embed_dim}x0e")
        self.input_proj = o3.Linear(irreps_input, irreps_hidden)

        # ── Edge features ──
        self.sh_irreps = o3.Irreps.spherical_harmonics(self.max_ell)
        self.radial_basis = BesselRadialBasis(
            r_max=self.max_radius, n_basis=n_radial_basis_i
        )

        # ── Interaction blocks ──
        self.interactions = nn.ModuleList()
        for _ in range(self.n_layers):
            block = InteractionBlock(
                irreps_in=str(irreps_hidden),
                irreps_out=str(irreps_hidden),
                irreps_sh=str(self.sh_irreps),
                n_radial_basis=n_radial_basis_i,
                hidden_dim=radial_hidden_i,
            )
            self.interactions.append(block)

        # ── Output head ──
        # Extract only scalar (0e) features for invariant prediction
        self.scalar_dim = int(scalar_mul)
        hidden_out = max(1, self.scalar_dim // 2)
        self.output_head = nn.Sequential(
            nn.Linear(self.scalar_dim, self.scalar_dim),
            nn.SiLU(),
            nn.Linear(self.scalar_dim, hidden_out),
            nn.SiLU(),
            nn.Linear(hidden_out, output_dim_i),
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
        batch: torch.Tensor | None = None,
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

        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"edge_index must be integer tensor, got dtype {edge_index.dtype}")
        if edge_index.size(1) <= 0:
            raise ValueError("edge_index must contain at least one edge")
        if edge_vectors.ndim != 2 or edge_vectors.size(1) != 3:
            raise ValueError(f"edge_vectors must have shape (E, 3), got {tuple(edge_vectors.shape)}")
        if edge_vectors.size(0) != edge_index.size(1):
            raise ValueError(
                f"edge count mismatch between edge_index and edge_vectors: {edge_index.size(1)} != {edge_vectors.size(0)}"
            )
        n_nodes = int(node_feats.size(0))
        edge_min = int(edge_index.min().item())
        edge_max = int(edge_index.max().item())
        if edge_min < 0 or edge_max >= n_nodes:
            raise ValueError(
                f"edge_index contains out-of-range node ids: min={edge_min}, max={edge_max}, N={n_nodes}"
            )

        # ── Species embedding ──
        species_idx = _infer_species_indices(node_feats, self.n_species)  # (N,)
        h_scalar = self.species_embed(species_idx)  # (N, embed_dim)
        h = self.input_proj(h_scalar)  # (N, irreps_hidden_dim)

        # ── Edge features ──
        distances = torch.nan_to_num(edge_vectors, nan=0.0, posinf=0.0, neginf=0.0).norm(dim=-1)  # (E,)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps,
            torch.nan_to_num(edge_vectors, nan=0.0, posinf=0.0, neginf=0.0),
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
        else:
            if batch.ndim != 1:
                raise ValueError(f"batch must be 1D [N], got shape {tuple(batch.shape)}")
            if batch.shape[0] != h_scalars.shape[0]:
                raise ValueError(
                    f"batch size must match number of nodes ({h_scalars.shape[0]}), got {batch.shape[0]}"
                )
            if batch.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"batch must be integer tensor, got dtype {batch.dtype}")
            batch = batch.long()

        from torch_geometric.nn import global_mean_pool
        graph_emb = global_mean_pool(h_scalars, batch)  # (B, scalar_dim)

        return graph_emb

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        batch: torch.Tensor | None = None,
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
        interaction_energy = self.output_head(graph_emb) # (B, output_dim)

        # Add atomic reference energy
        atomic_energy = self.atom_ref(node_feats, batch) # (B, output_dim)

        return interaction_energy + atomic_energy
