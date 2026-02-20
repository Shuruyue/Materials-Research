"""
M3GNet-inspired Graph Neural Network

Implements a Many-Body Graph Neural Network with 3-body interactions (bond angles).
Based on: Chen & Ong, "A universal graph deep learning interatomic potential for the periodic table", Nature Computational Science (2022).

Key Features:
- Edge update based on bond distances (2-body)
- Angle update based on bond triplets (3-body)
- Gated updates for information flow control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RBFExpansion(nn.Module):
    """Gaussian Radial Basis Function expansion."""
    def __init__(self, n_gaussians: int = 20, cutoff: float = 5.0):
        super().__init__()
        offset = torch.linspace(0, cutoff, n_gaussians)
        self.register_buffer("offset", offset)
        self.width = 0.5 * (offset[1] - offset[0])

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(dist.unsqueeze(-1) - self.offset)**2 / self.width**2)



class ThreeBodyInteraction(nn.Module):
    """
    Computes 3-body interactions using bond angles.
    """
    def __init__(self, embed_dim: int, n_basis: int = 20, use_sh: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_basis = n_basis
        self.use_sh = use_sh
        
        # Basis for angles 
        from .matgl_three_body import SimpleMLPAngleExpansion, SphericalBesselHarmonicsExpansion
        if use_sh:
            # MatGL style (Spherical Harmonics + Bessel Expansion)
            # max_n * max_l must equal n_basis
            max_n = 4
            max_l = n_basis // max_n
            n_basis = max_n * max_l # adjust if division wasn't perfect
            self.angle_expansion = SphericalBesselHarmonicsExpansion(max_n=max_n, max_l=max_l)
        else:
            # Simple Original MLP style
            self.angle_expansion = SimpleMLPAngleExpansion(n_basis)
        
        self.n_basis = n_basis # Update n_basis in case it changed
        
        # Mixing weights
        self.phi_3b = nn.Sequential(
            nn.Linear(embed_dim * 2 + n_basis, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        self.update_edge = nn.Linear(embed_dim, embed_dim)

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
        if three_body_edge_indices.size(0) == 0:
            return torch.zeros_like(edge_attr)
            
        e_ij_idx = three_body_edge_indices[:, 0]
        e_ik_idx = three_body_edge_indices[:, 1]
        
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
        h_3b = torch.cat([
            edge_attr[e_ij_idx],
            edge_attr[e_ik_idx],
            angle_feats
        ], dim=-1)
        
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
        
        return self.update_edge(delta_e_ij)


class M3GNetLayer(nn.Module):
    def __init__(self, embed_dim: int, n_basis: int = 20):
        super().__init__()
        # 1. Edge Update (2-body)
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim), # atom_i + atom_j + edge
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.edge_gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        
        # 2. Three-Body Update
        self.three_body = ThreeBodyInteraction(embed_dim, n_basis)
        self.three_body_gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())

        # 3. Node Update
        self.node_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), # atom + aggregated_edges
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
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
        
        src, dst = edge_index
        
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
        msg.index_add_(0, src, edge_feats) # Aggregate outgoing edges? Or incoming? 
        # Usually incoming (dst). Let's assume dst is the receiver.
        # msg.index_add_(0, dst, edge_feats) 
        # Standard GNN: update target node using source nodes. i <- j.
        # Here src=i, dst=j. So if we update i, we aggregate over j (dst).
        # Wait, PyG edge_index is [src, dst]. Flow is src -> dst.
        # To update dst, we sum over src.
        msg.index_add_(0, dst, edge_feats)
        
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
        self.embed_dim = embed_dim
        
        self.embedding = nn.Embedding(n_species, embed_dim)
        self.rbf = RBFExpansion(n_gaussians=n_rbf, cutoff=max_radius)
        self.edge_embedding = nn.Linear(n_rbf, embed_dim)
        
        self.layers = nn.ModuleList([
            M3GNetLayer(embed_dim, n_basis=10) for _ in range(n_layers)
        ])
        
        # Final pooling will be done by MultiTaskGNN or here if needed.
        # Assuming used as Encoder for MultiTaskGNN, we need to implement `encode`.

    def encode(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor, # This is usually RBF from builder, but M3GNet can recompute or use it.
        batch: torch.Tensor = None,
        edge_vectors: torch.Tensor = None,
        edge_index_3body: torch.Tensor = None, # (2, T)
        **kwargs
    ) -> torch.Tensor:
        
        # 1. Embed Nodes
        species_idx = node_feats[:, :86].argmax(dim=-1)
        x = self.embedding(species_idx)
        
        # 2. Embed Edges
        # edge_attr from builder is already RBF-expanded.
        if edge_attr.size(-1) != self.edge_embedding.in_features:
            # Handle mismatch if builder RBF dim != n_rbf
            # For now assume mostly correct or rely on linear layer adaptation
             pass
        e = self.edge_embedding(edge_attr)
        
        # Prepare 3-body indices
        # Graph builder gives (2, T) Transposed as proper PyG tensor
        if edge_index_3body is not None:
            tb_edge_indices = edge_index_3body.T # (T, 2)
            tb_indices = None # Not strictly used in my impl if edge indices are enough
        else:
            tb_edge_indices = torch.empty((0, 2), dtype=torch.long, device=x.device)
            tb_indices = None

        if edge_vectors is None:
            # Cannot compute angles without vectors
            # Fallback to pure 2-body
            tb_edge_indices = torch.empty((0, 2), dtype=torch.long, device=x.device)

        # 3. Message Passing
        for layer in self.layers:
            x, e = layer(x, e, edge_index, tb_indices, tb_edge_indices, edge_vectors)
            
        # 4. Global Pool
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        from torch_geometric.nn import global_mean_pool
        return global_mean_pool(x, batch)
        
    def forward(self, *args, **kwargs):
        # Compatibility with MultiTaskGNN which calls encode
        return self.encode(*args, **kwargs)
