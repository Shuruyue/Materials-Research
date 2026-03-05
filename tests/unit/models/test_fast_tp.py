from __future__ import annotations

import pytest
import torch

from atlas.models.fast_tp import FusedTensorProductScatter

pytest.importorskip("e3nn")


def _build_inputs():
    layer = FusedTensorProductScatter("2x0e", "1x0e", "2x0e")
    x = torch.randn(4, 2)
    edge_attr = torch.randn(5, 1)
    edge_weight = torch.randn(5, layer.weight_numel)
    edge_src = torch.tensor([0, 1, 1, 2, 3], dtype=torch.long)
    edge_dst = torch.tensor([1, 2, 3, 0, 1], dtype=torch.long)
    return layer, x, edge_attr, edge_weight, edge_src, edge_dst


def test_fused_tp_scatter_forward_shape():
    layer, x, edge_attr, edge_weight, edge_src, edge_dst = _build_inputs()
    out = layer(
        x=x,
        edge_attr=edge_attr,
        edge_weight=edge_weight,
        edge_src=edge_src,
        edge_dst=edge_dst,
        num_nodes=x.size(0),
    )
    assert out.shape == (x.size(0), layer.out_dim)
    assert torch.isfinite(out).all()


def test_fused_tp_scatter_rejects_non_integer_edge_indices():
    layer, x, edge_attr, edge_weight, edge_src, edge_dst = _build_inputs()
    with pytest.raises(ValueError, match="edge_src must be integer tensor"):
        layer(
            x=x,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            edge_src=edge_src.to(torch.float32),
            edge_dst=edge_dst,
            num_nodes=x.size(0),
        )
    with pytest.raises(ValueError, match="edge_dst must be integer tensor"):
        layer(
            x=x,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            edge_src=edge_src,
            edge_dst=edge_dst.to(torch.float32),
            num_nodes=x.size(0),
        )


def test_fused_tp_scatter_rejects_non_finite_payloads():
    layer, x, edge_attr, edge_weight, edge_src, edge_dst = _build_inputs()
    edge_weight_bad = edge_weight.clone()
    edge_weight_bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="edge_weight contains NaN or Inf values"):
        layer(
            x=x,
            edge_attr=edge_attr,
            edge_weight=edge_weight_bad,
            edge_src=edge_src,
            edge_dst=edge_dst,
            num_nodes=x.size(0),
        )


def test_fused_tp_scatter_rejects_dtype_mismatch_and_non_floating_payloads():
    layer, x, edge_attr, edge_weight, edge_src, edge_dst = _build_inputs()
    with pytest.raises(ValueError, match="edge_attr dtype must match x dtype"):
        layer(
            x=x.to(torch.float64),
            edge_attr=edge_attr,
            edge_weight=edge_weight.to(torch.float64),
            edge_src=edge_src,
            edge_dst=edge_dst,
            num_nodes=x.size(0),
        )
    with pytest.raises(ValueError, match="x must be a floating-point tensor"):
        layer(
            x=torch.ones_like(x, dtype=torch.int64),
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            edge_src=edge_src,
            edge_dst=edge_dst,
            num_nodes=x.size(0),
        )


@pytest.mark.parametrize("num_nodes", [3.5, True, float("inf"), float("nan")])
def test_fused_tp_scatter_validates_num_nodes_type(num_nodes):
    layer, x, edge_attr, edge_weight, edge_src, edge_dst = _build_inputs()
    with pytest.raises(ValueError, match="num_nodes"):
        layer(
            x=x,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            edge_src=edge_src,
            edge_dst=edge_dst,
            num_nodes=num_nodes,
        )
