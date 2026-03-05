"""Tests for atlas.models.cgcnn."""

from __future__ import annotations

import pytest
import torch

from atlas.models.cgcnn import CGCNN
from atlas.models.layers import MessagePassingLayer


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


def test_cgcnn_invalid_hyperparameters_raise():
    with pytest.raises(ValueError, match="must be > 0"):
        CGCNN(node_dim=0)
    with pytest.raises(ValueError, match="must be > 0"):
        CGCNN(n_conv=0)
    with pytest.raises(ValueError, match="node_dim must be integer-valued"):
        CGCNN(node_dim=91.5)
    with pytest.raises(ValueError, match="edge_dim must be integer-valued, not boolean"):
        CGCNN(edge_dim=True)
    with pytest.raises(ValueError, match="dropout must be in"):
        CGCNN(dropout=1.2)
    with pytest.raises(ValueError, match="dropout must be in"):
        CGCNN(dropout=False)


def test_cgcnn_encode_validates_input_shapes():
    x, edge_index, edge_attr, batch = _dummy_graph()
    model = CGCNN()
    bad_edge_attr = edge_attr[:-1]
    with pytest.raises(ValueError, match="edge_feats first dim must match edge count"):
        model.encode(x, edge_index, bad_edge_attr, batch)
    with pytest.raises(ValueError, match="batch size must match number of nodes"):
        model.encode(x, edge_index, edge_attr, batch[:-1])


def test_cgcnn_encode_validates_index_dtype_and_bounds():
    x, edge_index, edge_attr, batch = _dummy_graph()
    model = CGCNN()
    with pytest.raises(ValueError, match="edge_index must be integer tensor"):
        model.encode(x, edge_index.to(torch.float32), edge_attr, batch)

    edge_index_bad = edge_index.clone()
    edge_index_bad[0, 0] = x.size(0)
    with pytest.raises(ValueError, match="out-of-range node ids"):
        model.encode(x, edge_index_bad, edge_attr, batch)


def test_cgcnn_encode_validates_finite_features():
    x, edge_index, edge_attr, batch = _dummy_graph()
    model = CGCNN()
    x_bad = x.clone()
    x_bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="node_feats contains NaN or Inf"):
        model.encode(x_bad, edge_index, edge_attr, batch)

    e_bad = edge_attr.clone()
    e_bad[0, 0] = float("inf")
    with pytest.raises(ValueError, match="edge_feats contains NaN or Inf"):
        model.encode(x, edge_index, e_bad, batch)


def test_message_passing_layer_strict_parameter_and_index_validation():
    with pytest.raises(ValueError, match="node_dim must be integer-valued, not boolean"):
        MessagePassingLayer(node_dim=True, edge_dim=20)
    with pytest.raises(ValueError, match="edge_dim must be integer-valued"):
        MessagePassingLayer(node_dim=64, edge_dim=20.5)

    layer = MessagePassingLayer(node_dim=16, edge_dim=4, aggr="mean")
    h = torch.randn(6, 16)
    edge_index = torch.randint(0, 6, (2, 12)).to(torch.float32)
    e = torch.randn(12, 4)
    with pytest.raises(ValueError, match="edge_index must be integer tensor"):
        layer(h, edge_index, e)

