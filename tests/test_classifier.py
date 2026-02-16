"""Tests for atlas.topology.classifier module."""

import pytest
import torch
import numpy as np


@pytest.fixture
def si_structure():
    """Simple Si diamond structure."""
    from pymatgen.core import Structure, Lattice

    lattice = Lattice.cubic(5.43)
    return Structure(
        lattice=lattice,
        species=["Si", "Si"],
        coords=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


@pytest.fixture
def graph_builder():
    """CrystalGraphBuilder instance."""
    from atlas.topology.classifier import CrystalGraphBuilder
    return CrystalGraphBuilder(cutoff=5.0, max_neighbors=12)


def test_element_features_shape(graph_builder):
    """Element features should have correct dimensionality."""
    feats = graph_builder.element_features("Si")
    n_elements = len(graph_builder.ELEMENTS)
    expected_dim = n_elements + 5  # one-hot + 5 properties
    assert feats.shape == (expected_dim,)
    assert feats.dtype == np.float32


def test_element_features_one_hot(graph_builder):
    """One-hot part should have exactly one 1."""
    feats = graph_builder.element_features("Si")
    n_elements = len(graph_builder.ELEMENTS)
    one_hot = feats[:n_elements]
    assert one_hot.sum() == 1.0

    si_idx = graph_builder.ELEMENT_TO_IDX["Si"]
    assert one_hot[si_idx] == 1.0


def test_element_features_unknown(graph_builder):
    """Unknown element should still return valid features."""
    feats = graph_builder.element_features("Og")  # Oganesson, not in list
    assert feats.shape[0] == len(graph_builder.ELEMENTS) + 5


def test_structure_to_graph_keys(graph_builder, si_structure):
    """Graph dict should contain all expected keys."""
    graph = graph_builder.structure_to_graph(si_structure)
    assert "node_features" in graph
    assert "edge_index" in graph
    assert "edge_features" in graph
    assert "num_nodes" in graph


def test_structure_to_graph_shapes(graph_builder, si_structure):
    """Graph tensors should have correct shapes."""
    graph = graph_builder.structure_to_graph(si_structure)

    n_atoms = 2
    expected_node_dim = len(graph_builder.ELEMENTS) + 5

    assert graph["node_features"].shape == (n_atoms, expected_node_dim)
    assert graph["edge_index"].shape[0] == 2  # (2, num_edges)
    assert graph["num_nodes"] == n_atoms

    n_edges = graph["edge_index"].shape[1]
    assert graph["edge_features"].shape == (n_edges, 20)  # 20 Gaussians


def test_structure_to_graph_tensor_types(graph_builder, si_structure):
    """Graph tensors should have correct dtypes."""
    graph = graph_builder.structure_to_graph(si_structure)

    assert graph["node_features"].dtype == torch.float32
    assert graph["edge_index"].dtype == torch.int64
    assert graph["edge_features"].dtype == torch.float32


def test_topo_gnn_forward():
    """TopoGNN forward pass should produce correct output shape."""
    from atlas.topology.classifier import TopoGNN

    model = TopoGNN(node_dim=69, edge_dim=20, hidden_dim=64, n_layers=2)
    model.eval()

    # Create dummy input (single graph)
    n_nodes = 5
    n_edges = 10
    node_feats = torch.randn(n_nodes, 69)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_feats = torch.randn(n_edges, 20)

    with torch.no_grad():
        logits = model(node_feats, edge_index, edge_feats)

    assert logits.shape == (1,)  # single graph → single logit


def test_topo_gnn_predict_proba():
    """predict_proba should return values in [0, 1]."""
    from atlas.topology.classifier import TopoGNN

    model = TopoGNN(node_dim=69, edge_dim=20, hidden_dim=64, n_layers=2)
    model.eval()

    node_feats = torch.randn(3, 69)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_feats = torch.randn(3, 20)

    with torch.no_grad():
        prob = model.predict_proba(node_feats, edge_index, edge_feats)

    assert 0.0 <= prob.item() <= 1.0


def test_message_passing_layer():
    """MessagePassingLayer should produce same-shape output."""
    from atlas.topology.classifier import MessagePassingLayer

    layer = MessagePassingLayer(node_dim=64, edge_dim=64)

    h = torch.randn(5, 64)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    e = torch.randn(4, 64)

    h_out = layer(h, edge_index, e)
    assert h_out.shape == h.shape


def test_end_to_end_graph_to_prediction(graph_builder, si_structure):
    """Full pipeline: structure → graph → GNN → probability."""
    from atlas.topology.classifier import TopoGNN

    graph = graph_builder.structure_to_graph(si_structure)

    node_dim = graph["node_features"].shape[1]
    edge_dim = graph["edge_features"].shape[1]

    model = TopoGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=64, n_layers=2)
    model.eval()

    with torch.no_grad():
        prob = model.predict_proba(
            graph["node_features"],
            graph["edge_index"],
            graph["edge_features"],
        )

    assert 0.0 <= prob.item() <= 1.0
