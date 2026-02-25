# ATLAS Performance Analysis Report

## Data Pipeline

### Current Bottlenecks

| Component | Issue | Impact | Recommendation |
|-----------|-------|--------|----------------|
| `CrystalGraphBuilder.structure_to_graph()` | Per-structure neighbor search via `get_all_neighbors()` | ~15ms per structure → ~19min for 76k | ✅ Already parallelized via `ProcessPoolExecutor` |
| `CrystalPropertyDataset.prepare()` | Full graph rebuild on each call without `.pt` cache | Minutes on large datasets | ✅ Has disk cache via `torch.save()` |
| `JARVISClient.load_dft_3d()` | ~500MB FigShare download on first use | One-time cost | ✅ Has resume support |
| Worker count | `n_workers=1` hardcoded (Windows spawn issue) | No parallelism on Windows | 🟡 Consider `loky` backend for Windows |
| 3-body index computation | O(n²) per atom for triplet enumeration | Quadratic growth with `max_neighbors` | 🟡 Cap at `max_neighbors` already helps |

### Graph Construction Cost Estimate

| Dataset Size | Sequential (1 worker) | Parallel (4 workers, Linux) |
|-------------|----------------------|----------------------------|
| 1,000 | ~15s | ~5s |
| 10,000 | ~150s | ~45s |
| 76,000 (full JARVIS) | ~19min | ~5min |

> **Note**: Disk cache prevents repeated construction. First run is slow, subsequent runs load from `.pt` in seconds.

---

## Training Efficiency

| Feature | Status | Notes |
|---------|--------|-------|
| AMP (Automatic Mixed Precision) | ✅ Implemented | Correctly disabled on CPU |
| Gradient Clipping | ✅ `max_norm=1.0` | Fixed threshold, consider adaptive |
| Gradient Accumulation | ⚠️ `theory_tuning.py` defines `acc-steps` | Not implemented in `Trainer.train_epoch()` |
| DataLoader `pin_memory` | ⚠️ Not set | Add `pin_memory=True` for GPU training |
| DataLoader `num_workers` | ⚠️ Fixed at 1 (Windows) | Benchmark with 2-4 on Linux |
| Learning Rate Scheduling | ✅ ReduceLROnPlateau + others | Works correctly |
| Top-K Checkpointing | ⚠️ Only saves best + final | Consider top-3 for ensemble |

---

## Memory Usage

| Risk | Module | Details | Mitigation |
|------|--------|---------|------------|
| 🟡 Medium | `CrystalPropertyDataset` | All PyG `Data` objects held in memory | Fine for JARVIS (~76k × ~2KB = ~150MB) |
| 🟡 Medium | `EnsembleUQ` (5 models) | 5× model parameters | Use shared embeddings or distillation |
| 🟡 Medium | `M3GNet` triplet tensors | `O(N × max_neighbors²)` per graph | Capped at `max_neighbors=12` → max 144 triplets/atom |
| 🟢 Low | `EquivariantGNN` with `e3nn` | Dense tensor products | Typical: 10-50MB per model |

---

## Recommendations (Priority Ordered)

1. **P1**: Add `pin_memory=True` to DataLoader creation for GPU workflows
2. **P1**: Implement gradient accumulation in `Trainer.train_epoch()` to match `theory_tuning.py` profiles
3. **P2**: Benchmark `num_workers=2-4` on Linux CI, keep `=1` as Windows fallback
4. **P2**: Add `--prefetch-factor` option for DataLoader
5. **P3**: Consider lazy loading for very large datasets (>100k)
