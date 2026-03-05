"""
Lightweight fused tensor-product + scatter wrapper.

This module provides a compatibility implementation for projects that expect
`FusedTensorProductScatter` (NequIP-style API) while running on plain e3nn.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch import nn


def _is_boolean_like(value: object) -> bool:
    return isinstance(value, bool) or type(value).__name__ == "bool_"


def _coerce_non_negative_int(value: object, name: str) -> int:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be integer-valued, not boolean")
    if isinstance(value, Integral):
        integer = int(value)
    elif isinstance(value, Real):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        rounded = round(scalar)
        if abs(scalar - rounded) > 1e-9:
            raise ValueError(f"{name} must be integer-valued")
        integer = int(rounded)
    else:
        raise ValueError(f"{name} must be integer-valued")
    if integer < 0:
        raise ValueError(f"{name} must be >= 0")
    return integer


class FusedTensorProductScatter(nn.Module):
    """
    e3nn FullyConnectedTensorProduct followed by scatter-add aggregation.

    API contract:
    - `weight_numel` attribute exposed for radial MLP output size.
    - `forward(...)` accepts edge src/dst indices and returns node-aggregated
      messages with shape `(num_nodes, irreps_out.dim)`.
    """

    def __init__(self, irreps_in, irreps_edge, irreps_out):
        super().__init__()
        from e3nn import o3

        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=o3.Irreps(irreps_in),
            irreps_in2=o3.Irreps(irreps_edge),
            irreps_out=o3.Irreps(irreps_out),
            internal_weights=False,
            shared_weights=False,
        )
        self.weight_numel = self.tp.weight_numel
        self.out_dim = self.tp.irreps_out.dim

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        n_nodes = _coerce_non_negative_int(num_nodes, "num_nodes")
        if x.ndim != 2:
            raise ValueError(f"x must be rank-2 tensor, got shape {tuple(x.shape)}")
        if edge_attr.ndim != 2:
            raise ValueError(f"edge_attr must be rank-2 tensor, got shape {tuple(edge_attr.shape)}")
        if edge_weight.ndim != 2:
            raise ValueError(f"edge_weight must be rank-2 tensor, got shape {tuple(edge_weight.shape)}")
        if edge_src.ndim != 1 or edge_dst.ndim != 1:
            raise ValueError("edge_src and edge_dst must be 1D tensors")
        if edge_src.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"edge_src must be integer tensor, got dtype {edge_src.dtype}")
        if edge_dst.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"edge_dst must be integer tensor, got dtype {edge_dst.dtype}")
        if edge_src.numel() != edge_dst.numel():
            raise ValueError("edge_src and edge_dst must have identical lengths")
        if edge_attr.size(0) != edge_src.numel():
            raise ValueError(
                f"edge_attr row count mismatch: got {edge_attr.size(0)}, expected {edge_src.numel()}"
            )
        if edge_weight.size(0) != edge_src.numel():
            raise ValueError(
                f"edge_weight row count mismatch: got {edge_weight.size(0)}, expected {edge_src.numel()}"
            )
        if edge_weight.size(1) != self.weight_numel:
            raise ValueError(
                f"edge_weight must have shape (E, {self.weight_numel}), got {tuple(edge_weight.shape)}"
            )
        if not x.is_floating_point():
            raise ValueError(f"x must be a floating-point tensor, got dtype {x.dtype}")
        if not edge_attr.is_floating_point():
            raise ValueError(f"edge_attr must be a floating-point tensor, got dtype {edge_attr.dtype}")
        if not edge_weight.is_floating_point():
            raise ValueError(f"edge_weight must be a floating-point tensor, got dtype {edge_weight.dtype}")
        if edge_attr.dtype != x.dtype:
            raise ValueError(
                f"edge_attr dtype must match x dtype ({x.dtype}), got {edge_attr.dtype}"
            )
        if edge_weight.dtype != x.dtype:
            raise ValueError(
                f"edge_weight dtype must match x dtype ({x.dtype}), got {edge_weight.dtype}"
            )
        if edge_attr.device != x.device:
            raise ValueError(
                f"edge_attr device must match x device ({x.device}), got {edge_attr.device}"
            )
        if edge_weight.device != x.device:
            raise ValueError(
                f"edge_weight device must match x device ({x.device}), got {edge_weight.device}"
            )
        if edge_src.device != x.device or edge_dst.device != x.device:
            raise ValueError("edge_src and edge_dst must be on the same device as x")
        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN or Inf values")
        if not torch.isfinite(edge_attr).all():
            raise ValueError("edge_attr contains NaN or Inf values")
        if not torch.isfinite(edge_weight).all():
            raise ValueError("edge_weight contains NaN or Inf values")
        if edge_src.numel() == 0:
            return torch.zeros((n_nodes, self.out_dim), dtype=x.dtype, device=x.device)
        if n_nodes == 0:
            raise ValueError("num_nodes must be > 0 when edges are provided")

        edge_src = edge_src.long()
        edge_dst = edge_dst.long()
        if int(edge_src.min()) < 0 or int(edge_src.max()) >= x.size(0):
            raise ValueError("edge_src contains out-of-range node indices")
        if int(edge_dst.min()) < 0 or int(edge_dst.max()) >= n_nodes:
            raise ValueError("edge_dst contains out-of-range node indices")

        messages = self.tp(x[edge_src], edge_attr, edge_weight)
        messages = torch.nan_to_num(messages, nan=0.0, posinf=0.0, neginf=0.0)
        out = torch.zeros(
            (n_nodes, self.out_dim),
            dtype=messages.dtype,
            device=messages.device,
        )
        out.index_add_(0, edge_dst, messages)
        return out

