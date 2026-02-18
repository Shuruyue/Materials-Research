import torch
from torch.utils.cpp_extension import load_inline

# C++ Source Code for Optimized Radius Graph
cpp_source = """
#include <torch/extension.h>
#include <cmath>
#include <tuple>

// Simple pairwise distance calculation + radius cutoff (Naive O(N^2))
// For small crystals (N < 200), O(N^2) is faster than KD-Tree overhead.
// Output: (edge_index, edge_vec, edge_dist)
// WARNING: This implementation DOES NOT support Periodic Boundary Conditions (PBC).
// Do NOT use for crystal structures where atoms interact across boundaries.
// TODO(Phase 3): Add PBC support or replace with torch_cluster.radius_graph + shift vectors.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> radius_graph_cpp(
    torch::Tensor pos, 
    torch::Tensor batch, 
    float r_max, 
    int max_num_neighbors
) {
    int n_atoms = pos.size(0);
    
    // Output Vectors
    std::vector<int64_t> row_indices;
    std::vector<int64_t> col_indices;
    std::vector<float> dists;
    std::vector<float> vecs_x;
    std::vector<float> vecs_y;
    std::vector<float> vecs_z;

    // Accessors
    auto pos_a = pos.accessor<float, 2>();
    auto batch_a = batch.accessor<int64_t, 1>();
    
    // Loop over all pairs (Naive)
    // In production, we would use Cell Lists or KD-Tree, but for crystals N is small.
    for (int i = 0; i < n_atoms; i++) {
        for (int j = 0; j < n_atoms; j++) {
            if (i == j) continue; // Self-loop check (optional)
            
            // Optimization: Only check atoms in same batch (crystal)
            if (batch_a[i] != batch_a[j]) continue;

            float dx = pos_a[j][0] - pos_a[i][0];
            float dy = pos_a[j][1] - pos_a[i][1];
            float dz = pos_a[j][2] - pos_a[i][2];
            
            float d2 = dx*dx + dy*dy + dz*dz;
            
            if (d2 <= r_max * r_max) {
                float d = std::sqrt(d2);
                row_indices.push_back(i);
                col_indices.push_back(j);
                dists.push_back(d);
                vecs_x.push_back(dx);
                vecs_y.push_back(dy);
                vecs_z.push_back(dz);
            }
        }
    }
    
    // Convert to Tensors
    int n_edges = row_indices.size();
    auto options_long = torch::TensorOptions().dtype(torch::kLong).device(pos.device());
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(pos.device());

    torch::Tensor edge_index = torch.empty({2, n_edges}, options_long);
    torch::Tensor edge_dist = torch.empty({n_edges, 1}, options_float);
    torch::Tensor edge_vec = torch.empty({n_edges, 3}, options_float);
    
    // Fill tensors (could be optimized with memcpy but this is safe)
    auto edge_index_a = edge_index.accessor<int64_t, 2>();
    auto edge_dist_a = edge_dist.accessor<float, 2>();
    auto edge_vec_a = edge_vec.accessor<float, 2>();
    
    for (int k = 0; k < n_edges; k++) {
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

def get_cpp_module():
    """
    JIT compiles the C++ extension on first use.
    Requires: MSVC (Windows) or GCC (Linux) installed.
    """
    global _cpp_module
    if _cpp_module is None:
        try:
            print("⚙️ Compiling C++ Extension for Graph Building... (Might take a minute)")
            _cpp_module = load_inline(
                name='radius_graph_cpp',
                cpp_sources=cpp_source,
                functions=['radius_graph_cpp'],
                verbose=True,
                with_cuda=False # Keep it CPU-compatible for dataloader workers
            )
            print("✅ C++ Extension Compiled Successfully!")
        except Exception as e:
            print(f"❌ C++ Compilation Failed: {e}")
            print("  (Make sure you have Visual Studio Build Tools C++ installed)")
            return None
    return _cpp_module

def fast_radius_graph(pos, batch, r_max=5.0, max_num_neighbors=100):
    """
    Python wrapper for the C++ function.
    Falls back to PyTorch native implementation if C++ fails.
    WARNING: This op does not support Periodic Boundary Conditions.
    """
    import warnings
    warnings.warn("fast_radius_graph does not support PBC. Use with caution on crystals.", RuntimeWarning)
    
    module = get_cpp_module()
    if module:
        return module.radius_graph_cpp(pos, batch, r_max, max_num_neighbors)
    else:
        # Fallback to pure PyTorch (Vectorized, still fast)
        from torch_cluster import radius_graph
        edge_index = radius_graph(pos, r=r_max, batch=batch, max_num_neighbors=max_num_neighbors)
        # Calculate vectors and distances (Torch vectorized)
        row, col = edge_index
        vec = pos[col] - pos[row]
        dist = vec.norm(dim=-1, keepdim=True)
        return edge_index, vec, dist
