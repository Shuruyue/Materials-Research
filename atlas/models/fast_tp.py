"""
Lightweight fused tensor-product + scatter wrapper.

This module provides a compatibility implementation for projects that expect
`FusedTensorProductScatter` (NequIP-style API) while running on plain e3nn.
"""

from __future__ import annotations

import torch
from torch import nn


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
        messages = self.tp(x[edge_src], edge_attr, edge_weight)
        out = torch.zeros(
            (num_nodes, self.out_dim),
            dtype=messages.dtype,
            device=messages.device,
        )
        out.index_add_(0, edge_dst, messages)
        return out

