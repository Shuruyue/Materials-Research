from __future__ import annotations

import logging
import math
import warnings
from numbers import Integral, Real

import torch
from torch.utils.cpp_extension import load_inline

logger = logging.getLogger(__name__)

# C++ Source Code for Optimized Radius Graph
cpp_source = """
#include <torch/extension.h>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

// Simple pairwise distance calculation + radius cutoff (Naive O(N^2)).
// For small crystals (N < 200), O(N^2) is often faster than KD-tree overhead.
// Output: (edge_index, edge_vec, edge_dist)
// WARNING: This implementation DOES NOT support Periodic Boundary Conditions (PBC).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> radius_graph_cpp(
    torch::Tensor pos,
    torch::Tensor batch,
    float r_max,
    int max_num_neighbors
) {
    const int64_t n_atoms = pos.size(0);
    const float r2_max = r_max * r_max;

    std::vector<int64_t> row_indices;
    std::vector<int64_t> col_indices;
    std::vector<float> dists;
    std::vector<float> vecs_x;
    std::vector<float> vecs_y;
    std::vector<float> vecs_z;

    auto pos_a = pos.accessor<float, 2>();
    auto batch_a = batch.accessor<int64_t, 1>();

    for (int64_t i = 0; i < n_atoms; ++i) {
        std::vector<std::pair<float, int64_t>> nbrs;  // (d^2, j)
        for (int64_t j = 0; j < n_atoms; ++j) {
            if (i == j) {
                continue;
            }
            if (batch_a[i] != batch_a[j]) {
                continue;
            }

            const float dx = pos_a[j][0] - pos_a[i][0];
            const float dy = pos_a[j][1] - pos_a[i][1];
            const float dz = pos_a[j][2] - pos_a[i][2];
            const float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 <= r2_max) {
                nbrs.emplace_back(d2, j);
            }
        }

        if (max_num_neighbors > 0 && static_cast<int>(nbrs.size()) > max_num_neighbors) {
            std::partial_sort(
                nbrs.begin(),
                nbrs.begin() + max_num_neighbors,
                nbrs.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; }
            );
            nbrs.resize(max_num_neighbors);
        }

        for (const auto& entry : nbrs) {
            const float d2 = entry.first;
            const int64_t j = entry.second;
            const float dx = pos_a[j][0] - pos_a[i][0];
            const float dy = pos_a[j][1] - pos_a[i][1];
            const float dz = pos_a[j][2] - pos_a[i][2];
            row_indices.push_back(i);
            col_indices.push_back(j);
            dists.push_back(std::sqrt(d2));
            vecs_x.push_back(dx);
            vecs_y.push_back(dy);
            vecs_z.push_back(dz);
        }
    }

    const int64_t n_edges = static_cast<int64_t>(row_indices.size());
    auto options_long = torch::TensorOptions().dtype(torch::kLong).device(pos.device());
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(pos.device());

    torch::Tensor edge_index = torch::empty({2, n_edges}, options_long);
    torch::Tensor edge_dist = torch::empty({n_edges, 1}, options_float);
    torch::Tensor edge_vec = torch::empty({n_edges, 3}, options_float);

    auto edge_index_a = edge_index.accessor<int64_t, 2>();
    auto edge_dist_a = edge_dist.accessor<float, 2>();
    auto edge_vec_a = edge_vec.accessor<float, 2>();

    for (int64_t k = 0; k < n_edges; ++k) {
        edge_index_a[0][k] = row_indices[k];
        edge_index_a[1][k] = col_indices[k];
        edge_dist_a[k][0] = dists[k];
        edge_vec_a[k][0] = vecs_x[k];
        edge_vec_a[k][1] = vecs_y[k];
        edge_vec_a[k][2] = vecs_z[k];
    }

    return std::make_tuple(edge_index, edge_vec, edge_dist);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("radius_graph_cpp", &radius_graph_cpp, "Radius Graph (CPP)");
}
"""

_cpp_module = None
_cpp_compile_attempted = False
_pbc_warning_emitted = False


def _is_boolean_like(value: object) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}


def _coerce_positive_int(value: object, *, name: str) -> int:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be an integer > 0, got {value!r}")
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        scalar = float(value)
        if not math.isfinite(scalar) or not scalar.is_integer():
            raise ValueError(f"{name} must be an integer > 0, got {value!r}")
        number = int(scalar)
    else:
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer > 0, got {value!r}") from exc
    if number <= 0:
        raise ValueError(f"{name} must be an integer > 0, got {value!r}")
    return number


def _validate_inputs(
    pos: torch.Tensor,
    batch: torch.Tensor,
    r_max: float,
    max_num_neighbors: int,
) -> int:
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos must have shape (N, 3), got {tuple(pos.shape)}")
    if batch.ndim != 1:
        raise ValueError(f"batch must have shape (N,), got {tuple(batch.shape)}")
    if batch.shape[0] != pos.shape[0]:
        raise ValueError(
            f"batch length must match pos rows, got {batch.shape[0]} and {pos.shape[0]}"
        )
    if pos.device != batch.device:
        raise ValueError("pos and batch must be on the same device")
    if not torch.is_floating_point(pos):
        raise TypeError("pos must be a floating point tensor")
    if not torch.isfinite(pos).all():
        raise ValueError("pos must be finite")
    if batch.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"batch must be int32/int64, got {batch.dtype}")
    if _is_boolean_like(r_max):
        raise ValueError(f"r_max must be finite and > 0, got {r_max!r}")
    if not math.isfinite(float(r_max)) or float(r_max) <= 0:
        raise ValueError(f"r_max must be finite and > 0, got {r_max!r}")
    _coerce_positive_int(max_num_neighbors, name="max_num_neighbors")
    return int(pos.shape[0])


def _empty_outputs(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((2, 0), dtype=torch.long, device=device),
        torch.empty((0, 3), dtype=dtype, device=device),
        torch.empty((0, 1), dtype=dtype, device=device),
    )


def _torch_radius_graph_fallback(
    pos: torch.Tensor,
    batch: torch.Tensor,
    r_max: float,
    max_num_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_neighbors = _coerce_positive_int(max_num_neighbors, name="max_num_neighbors")
    num_atoms = int(pos.shape[0])
    if num_atoms == 0:
        return _empty_outputs(pos.device, pos.dtype)

    pos_work = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
    dist_matrix = torch.cdist(pos_work, pos_work, p=2)

    same_batch = batch[:, None] == batch[None, :]
    within_cutoff = dist_matrix <= float(r_max)
    mask = same_batch & within_cutoff
    mask.fill_diagonal_(False)

    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    for i in range(num_atoms):
        neighbors = torch.nonzero(mask[i], as_tuple=False).flatten()
        if neighbors.numel() == 0:
            continue

        neighbor_d = dist_matrix[i, neighbors]
        if neighbors.numel() > max_neighbors:
            _, keep_idx = torch.topk(
                neighbor_d, k=max_neighbors, largest=False, sorted=False
            )
            neighbors = neighbors[keep_idx]

        rows.append(torch.full_like(neighbors, i, dtype=torch.long))
        cols.append(neighbors.to(torch.long))

    if not rows:
        return _empty_outputs(pos.device, pos.dtype)

    row = torch.cat(rows)
    col = torch.cat(cols)
    edge_index = torch.stack((row, col), dim=0)
    edge_vec = pos[col] - pos[row]
    edge_dist = torch.linalg.norm(edge_vec, dim=-1, keepdim=True)
    edge_vec = torch.nan_to_num(edge_vec, nan=0.0, posinf=0.0, neginf=0.0)
    edge_dist = torch.nan_to_num(edge_dist, nan=0.0, posinf=0.0, neginf=0.0)
    return edge_index, edge_vec, edge_dist


def get_cpp_module():
    """
    JIT compiles the C++ extension on first use.
    Requires: MSVC (Windows) or GCC/Clang (Linux/macOS).
    """
    global _cpp_module, _cpp_compile_attempted
    if _cpp_compile_attempted:
        return _cpp_module

    _cpp_compile_attempted = True
    try:
        _cpp_module = load_inline(
            name="atlas_radius_graph_cpp_v2",
            cpp_sources=cpp_source,
            verbose=False,
            with_cuda=False,
            extra_cflags=["-O3"],
        )
    except Exception as exc:  # pragma: no cover - env dependent
        logger.warning(
            "C++ radius_graph extension compilation failed; using torch fallback (%s)",
            exc,
        )
        _cpp_module = None
    return _cpp_module


def fast_radius_graph(
    pos: torch.Tensor,
    batch: torch.Tensor,
    r_max: float = 5.0,
    max_num_neighbors: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build radius graph edges, vectors, and distances.

    WARNING: This op does not support Periodic Boundary Conditions (PBC).
    """
    global _pbc_warning_emitted
    num_atoms = _validate_inputs(pos, batch, r_max, max_num_neighbors)

    if not _pbc_warning_emitted:
        warnings.warn(
            "fast_radius_graph does not support PBC. Use with caution on crystals.",
            RuntimeWarning,
            stacklevel=2,
        )
        _pbc_warning_emitted = True

    if num_atoms == 0:
        return _empty_outputs(pos.device, pos.dtype)

    module = get_cpp_module() if pos.device.type == "cpu" else None
    if module is not None:
        try:
            pos_cpu = pos.to(device="cpu", dtype=torch.float32).contiguous()
            batch_cpu = batch.to(device="cpu", dtype=torch.int64).contiguous()
            edge_index, edge_vec, edge_dist = module.radius_graph_cpp(
                pos_cpu,
                batch_cpu,
                float(r_max),
                int(max_num_neighbors),
            )
            return (
                edge_index.to(device=pos.device),
                edge_vec.to(device=pos.device, dtype=pos.dtype),
                edge_dist.to(device=pos.device, dtype=pos.dtype),
            )
        except Exception as exc:  # pragma: no cover - extension runtime dependent
            logger.warning(
                "C++ radius_graph runtime failed; using torch fallback (%s)",
                exc,
            )

    return _torch_radius_graph_fallback(pos, batch, float(r_max), int(max_num_neighbors))
