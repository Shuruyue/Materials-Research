# ATLAS — Accelerated Topological Learning And Screening

> AI-driven discovery platform for topological quantum materials

## What is ATLAS?

ATLAS combines three cutting-edge technologies to discover new topological quantum materials:

1. **Equivariant Neural Network Potentials** (MACE) — accelerate molecular dynamics 10³× faster than DFT
2. **Topological Invariant Classification** — automatically classify materials by Z₂, Chern number, etc.
3. **Active Learning Closed-Loop** — Bayesian optimization to efficiently explore chemical space

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install core dependencies
pip install -r requirements.txt

# Install ATLAS in development mode
pip install -e .

# Optional profiles
# pip install -r requirements-dev.txt         # test + jupyter
# pip install -r requirements-benchmark.txt   # matbench + matminer
# pip install -r requirements-full.txt        # all optional extras

# Validate environment
python scripts/dev_tools/check_env.py
```

Run tests:
```bash
# Default: fast/unit suite (integration tests excluded)
pytest

# Integration suite (optional dependencies required)
pytest -m integration
```

### 2. Phase 1 — Baseline Training

```bash
# Download JARVIS-DFT data (~76,000 materials, free)
python scripts/phase1_baseline/01_download_data.py

# Train CGCNN baseline (lite / std / pro tiers)
python scripts/phase1_baseline/10_train_cgcnn_lite.py
python scripts/phase1_baseline/13_inference_demo.py
```

### 3. Phase 2 — Multi-Task Equivariant GNN

```bash
# Pre-compute 3-body graph data
python scripts/phase2_multitask/process_data_phase2.py

# Train multi-task model (lite / std / pro tiers)
python scripts/phase2_multitask/20_train_multitask_lite.py
python scripts/phase2_multitask/23_inference_multitask.py
```

### 4. Phase 3–6 — Potentials, Topology, Discovery, Analysis

```bash
# Phase 3: MACE potential
python scripts/phase3_potentials/03_prepare_mace_data.py
python scripts/phase3_potentials/04_train_mace.py
python scripts/phase3_potentials/05_run_relaxation.py

# Phase 4: Topological classifier
python scripts/phase4_topology/02_init_topo_db.py
python scripts/phase4_topology/05_train_topo_classifier.py

# Phase 5: Active learning discovery
python scripts/phase5_active_learning/06_run_discovery.py
python scripts/phase5_active_learning/07_search_materials.py --help

# Phase 6: Analysis
python scripts/phase6_analysis/08_alloy_properties.py
python scripts/phase6_analysis/09_phase_diagram.py
```

## Pre-trained Models

To keep the GitHub repository lightweight, trained model checkpoints (like MACE or CGCNN weights) are hosted externally on a cloud drive.

1. Download the `models.zip` file from this link: **[Insert Your Google Drive/OneDrive Link Here]**
2. Extract the contents directly into the `models/` directory at the root of this project.

## Project Structure

```
atlas/                              # Main Python package
├── __init__.py                     # Package root, version info
├── config.py                       # Centralized configuration & paths
├── data/                           # Data loading & databases
│   ├── jarvis_client.py            # JARVIS-DFT API (76K materials, no key)
│   ├── crystal_dataset.py          # PyG dataset with multi-property support
│   ├── topo_db.py                  # Topological materials database
│   └── property_estimator.py       # Physics-based property estimation
├── models/                         # Neural network architectures
│   ├── cgcnn.py                    # Crystal Graph Convolutional NN (Phase 1)
│   ├── equivariant.py              # E(3)-Equivariant GNN encoder (Phase 2)
│   ├── multi_task.py               # Multi-task wrapper with per-task heads
│   ├── m3gnet.py                   # M3GNet encoder with 3-body interactions
│   ├── evidential.py               # Evidential uncertainty head
│   ├── graph_builder.py            # Crystal → PyG graph conversion
│   └── layers.py                   # Shared layers (message passing, etc.)
├── training/                       # Training infrastructure
│   ├── trainer.py                  # Generic training loop
│   ├── losses.py                   # Multi-task & evidential losses
│   └── metrics.py                  # MAE, R², classification metrics
├── active_learning/                # Bayesian optimization loop
│   ├── controller.py               # Discovery engine (closed-loop)
│   └── generator.py               # Structure mutation & generation
├── topology/                       # Topological invariant calculators
│   └── classifier.py              # GNN classifier (TopoGNN)
├── potentials/                     # ML interatomic potentials
│   └── mace_relaxer.py            # MACE structure relaxation & stability
├── explain/                        # Model interpretability
│   ├── gnn_explainer.py           # GNNExplainer wrapper
│   └── integrated_gradients.py    # Integrated gradients attribution
├── thermo/                         # Thermodynamic analysis
│   ├── stability.py               # Phase stability analyst
│   └── calphad.py                 # CALPHAD phase diagrams
├── ops/                            # Performance optimizations
│   └── cpp_ops.py                 # C++ JIT-compiled graph ops
└── utils/                          # Utilities
    └── structure.py               # pymatgen ↔ ASE conversion

scripts/                            # Executable pipeline scripts
├── phase1_baseline/                # CGCNN training (lite/std/pro)
├── phase2_multitask/               # E(3)-Equivariant multi-task training
├── phase3_potentials/              # MACE potential training & relaxation
├── phase3_singletask/              # Single-task specialist training
├── phase4_topology/                # Topological classifier
├── phase5_active_learning/         # Discovery loop & search
├── phase6_analysis/                # Alloy properties & phase diagrams
├── phase8_integration/             # Recisic integration (Alchemy, MEPIN, LiFlow)
└── dev_tools/                      # Environment checks & monitoring

tests/                              # Unit tests
data/                               # Runtime data (gitignored)
models/                             # Trained models (gitignored)
```

## Roadmap

- [x] **Phase 0**: Project infrastructure
- [x] **Phase 1**: Data foundation + CGCNN baseline
- [x] **Phase 2**: Multi-task E(3)-Equivariant GNN
- [x] **Phase 3**: MACE potential (Dynamic Relaxation with Foundation Models)
- [x] **Phase 4**: Topological GNN classifier
- [x] **Phase 5**: Closed-loop active learning discovery (Ready)
- [ ] **Phase 6**: Analysis and experimental validation

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1 with CUDA
- GPU: NVIDIA RTX 4060 or better (for training)
- No API key required — JARVIS-DFT data is freely downloadable

## Phase 8: Advanced Discovery (Recisic Integration)
ATLAS now integrates state-of-the-art tools from the Recisic suite:
- **Alchemy**: Continuous composition optimization using Alchemical-MLIP.
- **MEPIN**: Reaction pathway and stability prediction.
- **LiFlow**: Ion transport property prediction.

```bash
# Run the full discovery pipeline (Grand Loop)
python scripts/phase8_integration/run_discovery_pipeline.py
```

## Developer Tools (`scripts/dev_tools/`)
Utility scripts for inspection and verification:
- `inspect_mace_model.py`: Inspect internal weights of MACE models.
- `inspect_liflow_model.py`: Verify LiFlow checkpoint loading.
- `verify_mepin_setup.py`: specific check for MEPIN imports.
- `check_env.py`: Validate python environment dependencies.

## References

- [MACE](https://github.com/ACEsuit/mace) — Equivariant message passing neural networks
- [JARVIS-DFT](https://jarvis.nist.gov) — NIST materials database (~76,000 materials)
- [Z2Pack](https://z2pack.ethz.ch) — Topological invariant computation
- [WannierTools](http://www.wanniertools.com) — Topological property analysis
- [Materials Project](https://materialsproject.org) — Open materials database

## License

MIT License — see [LICENSE](LICENSE) for details.
