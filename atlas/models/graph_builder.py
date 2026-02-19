"""
Crystal Graph Builder

Converts crystal structures (pymatgen Structure) into graph representations
suitable for GNN input.

Nodes: atoms with element features (one-hot + physical properties)
Edges: bonds within cutoff radius with distance features (Gaussian expansion)

This module was originally part of atlas.topology.classifier and has been
refactored into atlas.models for broader reuse across different GNN architectures.
"""

import numpy as np
import torch
from typing import Optional, Dict, Any

from atlas.config import get_config


# ──────────────────────────────────────────────────────────────
# Element property lookup tables
# ──────────────────────────────────────────────────────────────

# Pauling electronegativity (subset of common elements)
_ELECTRONEG = {
    "H": 2.20, "He": 0.00, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55,
    "N": 3.04, "O": 3.44, "F": 3.98, "Ne": 0.00, "Na": 0.93, "Mg": 1.31,
    "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16, "Ar": 0.00,
    "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66,
    "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65,
    "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 0.00,
    "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.60, "Mo": 2.16,
    "Tc": 1.90, "Ru": 2.20, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69,
    "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.10, "I": 2.66, "Xe": 2.60,
    "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Hf": 1.30, "Ta": 1.50, "W": 2.36,
    "Re": 1.90, "Os": 2.20, "Ir": 2.20, "Pt": 2.28, "Au": 2.54, "Hg": 2.00,
    "Tl": 1.62, "Pb": 2.33, "Bi": 2.02,
}

# Covalent radii (Å)
_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
    "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
    "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42,
    "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40, "Cs": 2.44,
    "Ba": 2.15, "La": 2.07, "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51,
    "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Tl": 1.45,
    "Pb": 1.46, "Bi": 1.48,
}

# Elements ordered for one-hot encoding (first 86)
_ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
]

_ELEM_TO_IDX = {e: i for i, e in enumerate(_ELEMENTS)}
_N_ELEMENTS = len(_ELEMENTS)


def gaussian_expansion(distances: np.ndarray, cutoff: float, n_gaussians: int = 20) -> np.ndarray:
    """
    Expand distances into Gaussian basis functions.
    Shared implementation for graph builders.
    """
    centers = np.linspace(0, cutoff, n_gaussians)
    width = 0.5 * (centers[1] - centers[0])
    return np.exp(-((distances[:, None] - centers[None, :]) ** 2) / width**2).astype(
        np.float32
    )


class CrystalGraphBuilder:
    """
    Converts a crystal structure to a graph representation.

    Nodes: atoms (features = one-hot element + atomic properties)
    Edges: bonds within cutoff radius (features = distance + Gaussian expansion)

    Args:
        cutoff: neighbor search cutoff radius in Angstroms
        max_neighbors: max number of neighbors per atom
    """

    def __init__(self, cutoff: float = 5.0, max_neighbors: int = 12, compute_3body: bool = True):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.compute_3body = compute_3body

    @property
    def node_dim(self) -> int:
        """Dimensionality of node features."""
        return _N_ELEMENTS + 5  # one-hot + [Z, electronegativity, radius, is_metal, is_heavy]

    def element_features(self, symbol: str) -> np.ndarray:
        """
        Get element features: one-hot encoding + physical properties.

        Features: [one_hot(86), atomic_number, electronegativity,
                   atomic_radius, is_metal, is_heavy(Z>=50)]
        """
        from pymatgen.core import Element

        elem = Element(symbol)
        one_hot = np.zeros(_N_ELEMENTS, dtype=np.float32)
        idx = _ELEM_TO_IDX.get(symbol, 0)
        one_hot[idx] = 1.0

        z = elem.Z / 100.0
        en = _ELECTRONEG.get(symbol, 1.5) / 4.0
        radius = _RADII.get(symbol, 1.5) / 3.0
        is_metal = 1.0 if elem.is_metal else 0.0
        is_heavy = 1.0 if elem.Z >= 50 else 0.0

        return np.concatenate([one_hot, [z, en, radius, is_metal, is_heavy]])

    def structure_to_graph(self, structure) -> Dict[str, Any]:
        """
        Convert pymatgen Structure to graph dict.

        Args:
            structure: pymatgen Structure object

        Returns:
            dict with keys: node_features, edge_index, edge_features,
                           num_nodes, edge_vectors (for equivariant models)
        """
        n_atoms = len(structure)

        # Node features
        node_feats = np.array(
            [self.element_features(str(site.specie)) for site in structure],
            dtype=np.float32,
        )

        # Find neighbors
        all_neighbors = structure.get_all_neighbors(self.cutoff, include_index=True)

        src, dst, dists, vectors = [], [], [], []
        for i, neighbors in enumerate(all_neighbors):
            sorted_n = sorted(neighbors, key=lambda x: x[1])[:self.max_neighbors]
            for neighbor in sorted_n:
                site, dist, j = neighbor[0], neighbor[1], neighbor[2]
                src.append(i)
                dst.append(j)
                dists.append(dist)
                # Direction vector for equivariant models
                vec = site.coords - structure[i].coords
                vectors.append(vec)

        if len(src) == 0:
            # Fallback: self-loops
            src, dst = [0], [0]
            dists = [0.0]
            vectors = [[0.0, 0.0, 0.0]]

        # 3-Body Indices (Edge pairs sharing source)
        # We need to map (src, dst) -> edge_idx
        # Since we construct edges sequentially, we can track their ranges.
        # But pymatgen neighbors are not sorted by index, so we need a mapping.
        
        edge_index = np.array([src, dst], dtype=np.int64)
        distances = np.array(dists, dtype=np.float32)
        edge_vectors = np.array(vectors, dtype=np.float32)
        
        three_body_indices = np.zeros((0, 3), dtype=np.int64) # Changed from (0,2) to (0,3) to match [d1, i, d2]
        three_body_edge_indices = np.zeros((0, 2), dtype=np.int64)

        if self.compute_3body and len(src) > 0:
            # Reconstruct adjacency for efficient triplet search
            # Group edges by source: src_idx -> list of (dst_idx, edge_array_idx)
            from collections import defaultdict
            adj = defaultdict(list)
            for e_idx, (s, d) in enumerate(zip(src, dst)):
                adj[s].append((d, e_idx))
            
            tb_indices = [] # (atom_j, atom_i, atom_k)
            tb_edge_indices = [] # (edge_ij, edge_ik)
            # M3GNet uses: for atom i, pairs of neighbors (j, k).
            # We want indices of edges (i->j) and (i->k).
            
            for i in range(n_atoms):
                 neighbors_i = adj[i]
                 n = len(neighbors_i)
                 if n < 2: continue
                 
                 # All pairs of neighbors (j, k)
                 # Limit to max neighbors to avoid explosion? 
                 # We already limited max_neighbors in getting neighbors.
                 for idx1 in range(n):
                     for idx2 in range(n):
                         if idx1 == idx2: continue
                         
                         d1, e_idx1 = neighbors_i[idx1] # d1 is neighbor j
                         d2, e_idx2 = neighbors_i[idx2] # d2 is neighbor k
                         
                         # Triplet: j(d1) - i - k(d2)
                         # We store edge indices corresponding to i->j and i->k
                         tb_edge_indices.append([e_idx1, e_idx2])
                         # We can also store atom indices if needed
                         tb_indices.append([d1, i, d2])

            if tb_edge_indices:
                three_body_edge_indices = np.array(tb_edge_indices, dtype=np.int64)
                three_body_indices = np.array(tb_indices, dtype=np.int64)

        # Edge features: Gaussian expansion of distances
        edge_feats = self._gaussian_expansion(distances)

        return {
            "node_features": node_feats,
            "edge_index": edge_index,
            "edge_features": edge_feats,
            "edge_vectors": edge_vectors,
            "three_body_indices": three_body_indices,          # (N_triplets, 3) [j, i, k]
            "three_body_edge_indices": three_body_edge_indices,# (N_triplets, 2) [edge_ij, edge_ik]
            "num_nodes": n_atoms,
        }

    def _gaussian_expansion(
        self, distances: np.ndarray, n_gaussians: int = 20
    ) -> np.ndarray:
        """Expand distances into Gaussian basis functions."""
        return gaussian_expansion(distances, self.cutoff, n_gaussians)

    def structure_to_pyg(self, structure, **properties) -> "torch_geometric.data.Data":
        """
        Convert to PyTorch Geometric Data object.

        Args:
            structure: pymatgen Structure
            **properties: target properties (e.g., band_gap=1.5, formation_energy=-0.3)

        Returns:
            torch_geometric.data.Data
        """
        from torch_geometric.data import Data

        graph = self.structure_to_graph(structure)

        data = Data(
            x=torch.tensor(graph["node_features"]),
            edge_index=torch.tensor(graph["edge_index"]),
            edge_attr=torch.tensor(graph["edge_features"]),
            edge_vec=torch.tensor(graph["edge_vectors"]),
            edge_index_3body=torch.tensor(graph["three_body_edge_indices"].T), # (2, N_triplets)
            num_nodes=graph["num_nodes"],
        )

        # Attach property targets
        for key, value in properties.items():
            if value is not None:
                # Allow NaNs (they will be masked in loss function)
                # if value is not None and not (isinstance(value, float) and np.isnan(value)):
                setattr(data, key, torch.tensor([value], dtype=torch.float))

        return data
