"""Tests for atlas.models.m3gnet and atlas.models.multi_task"""

import numpy as np
import torch
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from atlas.models.graph_builder import CrystalGraphBuilder
from atlas.models.m3gnet import M3GNet
from atlas.models.multi_task import MultiTaskGNN, EvidentialHead
from pymatgen.core import Structure, Lattice

class TestM3GNet(unittest.TestCase):
    
    def setUp(self):
        # Create a simple Si structure
        lattice = Lattice.cubic(5.43)
        self.si_structure = Structure(
            lattice=lattice,
            species=["Si", "Si"],
            coords=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        )
        self.builder = CrystalGraphBuilder(cutoff=5.0, max_neighbors=12, compute_3body=True)
        self.graph = self.builder.structure_to_pyg(self.si_structure)

    def test_m3gnet_forward(self):
        """Test M3GNet encoder forward pass."""
        model = M3GNet(n_species=86, embed_dim=16, n_layers=2)
        model.eval()
        
        with torch.no_grad():
            emb = model(
                node_feats=self.graph.x,
                edge_index=self.graph.edge_index,
                edge_attr=self.graph.edge_attr,
                batch=torch.zeros(self.graph.num_nodes, dtype=torch.long),
                edge_vectors=self.graph.edge_vec,
                edge_index_3body=self.graph.edge_index_3body
            )
            
        self.assertEqual(emb.shape, (1, 16))

    def test_evidential_head(self):
        """Test EvidentialHead output shape and constraints."""
        head = EvidentialHead(embed_dim=16, output_dim=1)
        x = torch.randn(10, 16)
        out = head(x)
        
        self.assertIn("gamma", out)
        self.assertIn("nu", out)
        self.assertIn("alpha", out)
        self.assertIn("beta", out)
        
        # Check constraints
        self.assertTrue(torch.all(out["nu"] > 0))
        self.assertTrue(torch.all(out["alpha"] > 1.0))
        self.assertTrue(torch.all(out["beta"] > 0))

    def test_multitask_gnn(self):
        """Test MultiTaskGNN with M3GNet encoder."""
        encoder = M3GNet(n_species=86, embed_dim=16)
        tasks = {
            "energy": {"type": "scalar"},
            "uncertainty": {"type": "evidential"}
        }
        model = MultiTaskGNN(encoder=encoder, tasks=tasks, embed_dim=16)
        
        with torch.no_grad():
            preds = model(
                node_feats=self.graph.x,
                edge_index=self.graph.edge_index,
                edge_feats=self.graph.edge_attr,
                batch=torch.zeros(self.graph.num_nodes, dtype=torch.long),
            )
            
        self.assertIn("energy", preds)
        self.assertIn("uncertainty", preds)
        self.assertEqual(preds["energy"].shape, (1, 1))
        self.assertIn("gamma", preds["uncertainty"])

if __name__ == "__main__":
    unittest.main()
