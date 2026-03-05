"""Tests for atlas.ops.cpp_ops."""

from __future__ import annotations

import pytest
import torch

import atlas.ops.cpp_ops as cpp_ops


def test_validate_inputs_rejects_non_finite_and_boolean_like():
    pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    batch = torch.tensor([0], dtype=torch.int64)

    with pytest.raises(ValueError, match="pos must be finite"):
        cpp_ops._validate_inputs(
            torch.tensor([[float("nan"), 0.0, 0.0]], dtype=torch.float32),
            batch,
            5.0,
            10,
        )
    with pytest.raises(ValueError, match="r_max must be finite and > 0"):
        cpp_ops._validate_inputs(pos, batch, True, 10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_num_neighbors must be an integer > 0"):
        cpp_ops._validate_inputs(pos, batch, 5.0, 1.2)


def test_radius_graph_fallback_caps_neighbors():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    batch = torch.tensor([0, 0, 0], dtype=torch.int64)
    edge_index, edge_vec, edge_dist = cpp_ops._torch_radius_graph_fallback(
        pos,
        batch,
        1.0,
        1,
    )
    assert edge_index.shape[1] == 3
    assert edge_vec.shape == (3, 3)
    assert edge_dist.shape == (3, 1)
