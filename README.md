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

# Install dependencies
pip install -r requirements.txt

# Install ATLAS in development mode
pip install -e .
```

### 2. Initialize Database

```bash
# Seed the topological materials database (no API key needed)
python scripts/02_init_topo_db.py

# Download JARVIS-DFT data (~76,000 materials, free)
python scripts/01_download_data.py
```

### 3. Run the Discovery Pipeline

```bash
# Prepare MACE training data
python scripts/03_prepare_mace_data.py

# Train MACE potential (requires GPU)
python scripts/04_train_mace.py

# Train topological GNN classifier
python scripts/05_train_topo_classifier.py

# Run full discovery loop
python scripts/06_run_discovery.py

# Search materials by properties
python scripts/07_search_materials.py --help
```

## Project Structure

```
atlas/                          # Main Python package
├── __init__.py                 # Package root, version info
├── config.py                   # Centralized configuration & paths
├── data/                       # Data loading & databases
│   ├── jarvis_client.py        # JARVIS-DFT API (76K materials, no key)
│   ├── topo_db.py              # Topological materials database
│   └── property_estimator.py   # Physics-based property estimation
├── potentials/                 # ML interatomic potentials
│   └── mace_relaxer.py         # MACE structure relaxation & stability
├── topology/                   # Topological invariant calculators
│   └── classifier.py           # GNN classifier (TopoGNN)
├── active_learning/            # Bayesian optimization loop
│   ├── controller.py           # Discovery engine (closed-loop)
│   └── generator.py            # Structure mutation & generation
└── utils/                      # Utilities
    └── structure.py            # pymatgen ↔ ASE conversion

scripts/                        # Executable pipeline scripts
├── 01_download_data.py         # Download JARVIS-DFT database
├── 02_init_topo_db.py          # Seed known topological materials
├── 03_prepare_mace_data.py     # Convert to MACE training format
├── 04_train_mace.py            # Train MACE neural network potential
├── 05_train_topo_classifier.py # Train topological GNN classifier
├── 06_run_discovery.py         # Run active learning discovery
└── 07_search_materials.py      # Multi-property materials search

tests/                          # Unit tests
data/                           # Runtime data (gitignored)
models/                         # Trained models (gitignored)
```

## Roadmap

- [x] **Phase 0**: Project infrastructure
- [ ] **Phase 1**: Data foundation (DFT database + initial MACE model)
- [ ] **Phase 2**: Topological classifier (Z₂/Chern from band structure)
- [ ] **Phase 3**: Closed-loop active learning integration
- [ ] **Phase 4**: Discovery and experimental validation

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1 with CUDA
- GPU: NVIDIA RTX 4060 or better (for MACE training)
- No API key required — JARVIS-DFT data is freely downloadable

## References

- [MACE](https://github.com/ACEsuit/mace) — Equivariant message passing neural networks
- [JARVIS-DFT](https://jarvis.nist.gov) — NIST materials database (~76,000 materials)
- [Z2Pack](https://z2pack.ethz.ch) — Topological invariant computation
- [WannierTools](http://www.wanniertools.com) — Topological property analysis
- [Materials Project](https://materialsproject.org) — Open materials database

## License

MIT License — see [LICENSE](LICENSE) for details.
