# Data Directory

> **Project**: ATLAS — Adaptive Training and Learning for Atomic Structures  
> **Last Updated**: 2026-02-27  
> **Approximate Total Size**: 25.3 GB

---

## Directory Structure

| Path | Description | Size |
|------|-------------|------|
| `raw/datasets/jarvis_dft/` | JARVIS-DFT bulk downloads (dft_3d, dft_2d, cfid_3d) | 1,460 MB |
| `raw/datasets/matbench/` | Matbench v0.1 benchmark tasks (13 tasks, pickle format) | 1,310 MB |
| `raw/datasets/mp_trj/` | MPtrj download instructions (dataset exceeds 100 GB) | — |
| `raw/jarvis_cache/` | Cached JARVIS data for pipeline use | 112 MB |
| `processed/multi_property/` | M3GNet crystal graph splits (train/val/test, seed=42) | ~19.2 GB |
| `processed/phase2_m3gnet_graphs.pt` | Phase 2 consolidated graph dataset | 3,281 MB |
| `processed/topo_materials.pkl` | Topological materials subset | 0.8 MB |
| `processed/trivial_materials.pkl` | Trivial (non-topological) materials subset | 1.0 MB |
| `discovery_results/` | Active learning iteration logs and summary report | <1 MB |
| `topo_db/` | Topological materials CSV reference table | <1 KB |

---

## Raw Dataset Provenance

| Dataset | Source | Entries | DFT Method | Format |
|---------|--------|---------|------------|--------|
| JARVIS dft_3d | NIST | 76,000 materials | VASP / OptB88vdW | JSON |
| JARVIS dft_2d | NIST | ~1,000 materials | VASP / OptB88vdW | JSON |
| JARVIS cfid_3d | NIST | 55,723 entries | Classical descriptors | JSON |
| Matbench mp_e_form | LBNL / MP | 132,752 samples | PBE / PBE+U | Pickle |
| Matbench mp_gap | LBNL / MP | 106,113 samples | PBE / PBE+U | Pickle |
| Matbench mp_is_metal | LBNL / MP | 106,113 samples | PBE / PBE+U | Pickle |
| Matbench log_gvrh | LBNL / MP | 10,987 samples | PBE / PBE+U | Pickle |
| Matbench log_kvrh | LBNL / MP | 10,987 samples | PBE / PBE+U | Pickle |
| Matbench perovskites | LBNL / MP | 18,928 samples | DFT | Pickle |
| Matbench dielectric | LBNL / MP | 4,764 samples | DFT | Pickle |
| Matbench phonons | LBNL / MP | 1,265 samples | DFT | Pickle |
| Matbench expt_gap | LBNL | 4,604 samples | Experimental | Pickle |
| Matbench expt_is_metal | LBNL | 4,921 samples | Experimental | Pickle |
| Matbench glass | LBNL | 5,680 samples | Experimental | Pickle |
| Matbench jdft2d | NIST / JARVIS | 636 samples | VASP / OptB88vdW | Pickle |
| Matbench steels | Literature | 312 samples | Experimental | Pickle |

Each subdirectory under `raw/datasets/` contains a `metadata.json` file recording download timestamp, source URL, and dataset-specific provenance details.

---

## Processed Data

The `processed/multi_property/` directory contains 76 PyTorch `.pt` files representing pre-computed M3GNet crystal graph objects, split into training, validation, and test partitions (seed=42) across 25 target properties. File naming convention:

```
{split}_{seed}_{property_hash}.pt
```

where `split` is one of `train`, `val`, or `test`.

---

## Usage

```python
# JARVIS-DFT
import json
with open("data/raw/datasets/jarvis_dft/dft_3d.json") as f:
    jarvis_data = json.load(f)

# Matbench (pickle format due to pymatgen Structure objects)
import pickle
with open("data/raw/datasets/matbench/matbench_mp_e_form.pkl", "rb") as f:
    df = pickle.load(f)

# Pre-processed crystal graphs
import torch
graphs = torch.load("data/processed/phase2_m3gnet_graphs.pt")
```

---

## Related Documentation

- Full provenance audit: `docs/research_preparation/dataset_tracker.md`
- Download scripts: `scripts/download_datasets.py`, `scripts/download_matbench.py`
