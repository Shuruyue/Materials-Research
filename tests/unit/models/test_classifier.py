"""Tests for atlas.models.m3gnet and atlas.models.multi_task"""

import unittest

import torch
from pymatgen.core import Lattice, Structure

from atlas.models.graph_builder import CrystalGraphBuilder
from atlas.models.m3gnet import M3GNet, M3GNetLayer, ThreeBodyInteraction
from atlas.models.multi_task import EvidentialHead, MultiTaskGNN


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

    def test_m3gnet_rejects_fractional_and_boolean_hyperparameters(self):
        with self.assertRaisesRegex(ValueError, "n_species must be integer-valued"):
            M3GNet(n_species=86.5)
        with self.assertRaisesRegex(ValueError, "embed_dim must be integer-valued, not boolean"):
            M3GNet(embed_dim=True)
        with self.assertRaisesRegex(ValueError, "n_layers must be integer-valued"):
            M3GNet(n_layers=2.2)
        with self.assertRaisesRegex(ValueError, "max_radius must be > 0, not boolean"):
            M3GNet(max_radius=False)

    def test_m3gnet_rejects_non_integer_edge_index_dtype(self):
        model = M3GNet(n_species=86, embed_dim=16, n_layers=1)
        with self.assertRaisesRegex(ValueError, "edge_index must be integer tensor"):
            model(
                node_feats=self.graph.x,
                edge_index=self.graph.edge_index.to(torch.float32),
                edge_attr=self.graph.edge_attr,
                batch=torch.zeros(self.graph.num_nodes, dtype=torch.long),
                edge_vectors=self.graph.edge_vec,
                edge_index_3body=self.graph.edge_index_3body,
            )

    def test_m3gnet_rejects_out_of_range_edge_index(self):
        model = M3GNet(n_species=86, embed_dim=16, n_layers=1)
        bad_edge_index = self.graph.edge_index.clone()
        bad_edge_index[0, 0] = self.graph.num_nodes
        with self.assertRaisesRegex(ValueError, "out-of-range node ids"):
            model(
                node_feats=self.graph.x,
                edge_index=bad_edge_index,
                edge_attr=self.graph.edge_attr,
                batch=torch.zeros(self.graph.num_nodes, dtype=torch.long),
                edge_vectors=self.graph.edge_vec,
                edge_index_3body=self.graph.edge_index_3body,
            )

    def test_m3gnet_rejects_non_integer_edge_index_3body_dtype(self):
        model = M3GNet(n_species=86, embed_dim=16, n_layers=1)
        with self.assertRaisesRegex(ValueError, "edge_index_3body must be integer tensor"):
            model(
                node_feats=self.graph.x,
                edge_index=self.graph.edge_index,
                edge_attr=self.graph.edge_attr,
                batch=torch.zeros(self.graph.num_nodes, dtype=torch.long),
                edge_vectors=self.graph.edge_vec,
                edge_index_3body=self.graph.edge_index_3body.to(torch.float32),
            )

    def test_m3gnet_rejects_invalid_batch_dtype(self):
        model = M3GNet(n_species=86, embed_dim=16, n_layers=1)
        with self.assertRaisesRegex(ValueError, "batch must be integer tensor"):
            model(
                node_feats=self.graph.x,
                edge_index=self.graph.edge_index,
                edge_attr=self.graph.edge_attr,
                batch=torch.zeros(self.graph.num_nodes, dtype=torch.float32),
                edge_vectors=self.graph.edge_vec,
                edge_index_3body=self.graph.edge_index_3body,
            )

    def test_m3gnet_rejects_negative_batch_indices(self):
        model = M3GNet(n_species=86, embed_dim=16, n_layers=1)
        bad_batch = torch.zeros(self.graph.num_nodes, dtype=torch.long)
        bad_batch[0] = -1
        with self.assertRaisesRegex(ValueError, "batch indices must be non-negative"):
            model(
                node_feats=self.graph.x,
                edge_index=self.graph.edge_index,
                edge_attr=self.graph.edge_attr,
                batch=bad_batch,
                edge_vectors=self.graph.edge_vec,
                edge_index_3body=self.graph.edge_index_3body,
            )

    def test_m3gnet_rejects_out_of_range_edge_index_3body(self):
        model = M3GNet(n_species=86, embed_dim=16, n_layers=1)
        bad_3body = self.graph.edge_index_3body.clone()
        if bad_3body.numel() == 0:
            self.skipTest("graph has no 3-body edges")
        bad_3body = bad_3body.long()
        bad_3body[0, 0] = self.graph.edge_index.shape[1]
        with self.assertRaisesRegex(ValueError, "out-of-range edge ids"):
            model(
                node_feats=self.graph.x,
                edge_index=self.graph.edge_index,
                edge_attr=self.graph.edge_attr,
                batch=torch.zeros(self.graph.num_nodes, dtype=torch.long),
                edge_vectors=self.graph.edge_vec,
                edge_index_3body=bad_3body,
            )

    def test_m3gnet_layer_rejects_fractional_or_boolean_init(self):
        with self.assertRaisesRegex(ValueError, "embed_dim must be integer-valued"):
            M3GNetLayer(embed_dim=16.5, n_basis=8)
        with self.assertRaisesRegex(ValueError, "n_basis must be integer-valued, not boolean"):
            M3GNetLayer(embed_dim=16, n_basis=True)

    def test_three_body_interaction_validates_use_sh_and_finite_inputs(self):
        with self.assertRaisesRegex(ValueError, "use_sh must be boolean-like"):
            ThreeBodyInteraction(embed_dim=8, n_basis=12, use_sh="maybe")

        module = ThreeBodyInteraction(embed_dim=8, n_basis=12, use_sh=True)
        edge_attr = torch.randn(6, 8)
        three_body_edge_indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        edge_vectors = torch.randn(6, 3)
        edge_vectors[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "edge_vectors contains NaN or Inf"):
            module(
                edge_attr=edge_attr,
                three_body_indices=torch.empty((0, 3), dtype=torch.long),
                three_body_edge_indices=three_body_edge_indices,
                edge_vectors=edge_vectors,
            )

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
