"""
Unit tests for atlas.topology.classifier

Tests:
- TopoGNN: forward pass shape, predict_proba, save/load roundtrip
- CrystalGraphBuilder integration (via atlas.topology import)
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from atlas.topology import CrystalGraphBuilder, TopoGNN


# ── TopoGNN ───────────────────────────────────────────────────


class TestTopoGNN:
    @pytest.fixture
    def model(self):
        return TopoGNN(node_dim=91, edge_dim=20, hidden_dim=32, n_layers=2, dropout=0.1)

    @pytest.fixture
    def fake_graph(self):
        n_nodes = 6
        n_edges = 12
        node_feats = torch.randn(n_nodes, 91)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        edge_feats = torch.randn(n_edges, 20)
        batch = torch.zeros(n_nodes, dtype=torch.long)
        return node_feats, edge_index, edge_feats, batch

    def test_forward_shape(self, model, fake_graph):
        out = model(*fake_graph)
        assert out.shape == (1,)  # 1 graph in batch

    def test_forward_multi_graph(self, model):
        n_nodes = 10
        node_feats = torch.randn(n_nodes, 91)
        edge_index = torch.randint(0, n_nodes, (2, 20))
        edge_feats = torch.randn(20, 20)
        batch = torch.cat([
            torch.zeros(5, dtype=torch.long),
            torch.ones(5, dtype=torch.long),
        ])
        out = model(node_feats, edge_index, edge_feats, batch)
        assert out.shape == (2,)  # 2 graphs in batch

    def test_forward_no_batch(self, model, fake_graph):
        """When batch is None, should default to single graph."""
        node_feats, edge_index, edge_feats, _ = fake_graph
        out = model(node_feats, edge_index, edge_feats, batch=None)
        assert out.shape == (1,)

    def test_predict_proba(self, model, fake_graph):
        prob = model.predict_proba(*fake_graph, mc_dropout=False)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_predict_proba_mc_dropout(self, model, fake_graph):
        result = model.predict_proba(*fake_graph, mc_dropout=True, n_samples=5)
        assert isinstance(result, tuple)
        mean_prob, std_prob = result
        assert 0.0 <= mean_prob <= 1.0
        assert std_prob >= 0.0

    def test_save_load_roundtrip(self, model, fake_graph):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "topo_model.pt"
            model.save_model(path)
            assert path.exists()

            loaded = TopoGNN.load_model(path)
            assert loaded.node_dim == 91
            assert loaded.hidden_dim == 32
            assert loaded.n_layers == 2

            # Predictions should match
            model.eval()
            with torch.no_grad():
                out_orig = model(*fake_graph)
                out_loaded = loaded(*fake_graph)
            assert torch.allclose(out_orig, out_loaded, atol=1e-5)

    def test_gradient_flows(self, model, fake_graph):
        node_feats, edge_index, edge_feats, batch = fake_graph
        node_feats = node_feats.clone().detach().requires_grad_(False)
        out = model(node_feats, edge_index, edge_feats, batch)
        loss = out.sum()
        loss.backward()
        # Should have gradients on model parameters
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break


# ── CrystalGraphBuilder (via topology import) ────────────────


class TestTopoGraphBuilder:
    def test_import_from_topology(self):
        """CrystalGraphBuilder should be importable from atlas.topology."""
        from atlas.topology import CrystalGraphBuilder as CGB
        builder = CGB(cutoff=5.0, max_neighbors=12)
        assert builder.cutoff == 5.0
        assert builder.max_neighbors == 12

    def test_node_dim(self):
        builder = CrystalGraphBuilder()
        assert builder.node_dim == 91  # 86 elements + 5 properties

    def test_gaussian_expansion(self):
        builder = CrystalGraphBuilder(cutoff=5.0)
        distances = np.array([0.0, 2.5, 5.0], dtype=np.float32)
        expanded = builder._gaussian_expansion(distances, n_gaussians=20)
        assert expanded.shape == (3, 20)
        assert expanded.dtype == np.float32
