"""Tests for atlas.models.cgcnn."""

from __future__ import annotations

import pytest
import torch

from atlas.models.cgcnn import CGCNN


def _dummy_graph():
    n_nodes = 18
    n_edges = 48
    x = torch.randn(n_nodes, 91)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, 20)
    batch = torch.tensor([0] * 6 + [1] * 6 + [2] * 6, dtype=torch.long)
    return x, edge_index, edge_attr, batch


@pytest.mark.parametrize("pooling", ["mean", "sum", "max", "mean_max", "attn"])
def test_cgcnn_pooling_modes_forward_shape(pooling: str):
    x, edge_index, edge_attr, batch = _dummy_graph()
    model = CGCNN(
        node_dim=91,
        edge_dim=20,
        hidden_dim=64,
        n_conv=3,
        n_fc=2,
        output_dim=1,
        pooling=pooling,
        jk="concat",
        message_aggr="mean",
        use_edge_gates=True,
    )
    out = model(x, edge_index, edge_attr, batch)
    assert out.shape == (3, 1)


@pytest.mark.parametrize("jk", ["last", "mean", "concat"])
def test_cgcnn_jk_modes_backward(jk: str):
    x, edge_index, edge_attr, batch = _dummy_graph()
    model = CGCNN(
        node_dim=91,
        edge_dim=20,
        hidden_dim=64,
        n_conv=3,
        n_fc=2,
        output_dim=1,
        pooling="attn",
        jk=jk,
        message_aggr="sum",
        use_edge_gates=False,
    )
    out = model(x, edge_index, edge_attr, batch)
    loss = out.mean()
    loss.backward()
    assert out.shape == (3, 1)

