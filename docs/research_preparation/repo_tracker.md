# Open-Source Repository Survey for Crystal GNN and Uncertainty Quantification Research

> **Project**: ATLAS — Adaptive Training and Learning for Atomic Structures  
> **Author**: Zhong  
> **Date**: 2026-02-27  
> **Total Entries**: 90  
> **Status**: Phase 1 Seed Collection Complete

---

## Methodology

### Selection Criteria

Each repository was evaluated across 10 dimensions adapted from software quality assessment frameworks:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Institutional Backing | High | Affiliation with established research groups (MIT, Harvard, Cambridge, NIST, etc.) |
| Publication Impact | High | Citation count and venue of the associated paper |
| Reproducibility | High | Clear installation, runnable examples, successful reproduction |
| Domain Relevance | High | Direct applicability to UQ, OOD detection, crystal GNNs, or Matbench |
| Test Coverage | Medium | Presence of unit tests, CI workflows |
| Documentation Quality | Medium | API docs, docstrings, tutorials |
| Maintenance Activity | Medium | Recent commits, active issue responses |
| Downstream Adoption | Medium | Usage by other reputable repositories |
| Community Governance | Low | Organization membership, contribution guidelines |
| Code Quality | Low | Type hints, architecture clarity |

### Grading System

| Grade | Criteria | Required Action |
|-------|----------|-----------------|
| **A** | >= 3 high-weight dimensions satisfied | Deep code reading, reproduction, detailed notes |
| **B** | >= 2 high-weight or >= 4 medium-weight | README review, core module reading, key takeaways |

All entries are Grade A or B. Low-relevance repositories have been excluded.

### Field Definitions

Each entry contains the following fields:

| Field | Description |
|-------|-------------|
| ID | Category letter + sequential number (e.g., A-01) |
| Repository | GitHub path (owner/repo) |
| Stars | GitHub star count as of 2026-02 |
| Affiliation | Maintaining institution or research group |
| Grade | A or B, based on the criteria above |
| Publication | Associated paper with venue, year, and approximate citation count |
| Description | Technical summary of the repository's purpose and methods |
| ATLAS Relevance | Specific connection to ATLAS project modules, methods, or goals |
| Key Learnings | Concrete items to extract during reading (A-grade only) |
| Reading Status | Not Started / Scanned / In Progress / Complete |
| Notes | Findings recorded during reading |

---

## Category 1: Crystal Property Prediction GNNs (15 entries)

### A-01 | txie-93/cgcnn | 822 stars

- **Affiliation**: MIT (Tian Xie, Jeffrey Grossman)
- **Grade**: A
- **Publication**: Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties. *Physical Review Letters*, 2018. ~3000 citations.
- **Description**: First end-to-end GNN for crystal property prediction directly from structure. Atoms as nodes, bonds as edges. Gaussian distance expansion for edge features. Global mean pooling followed by fully connected layers.
- **ATLAS Relevance**: Direct basis for `atlas/models/cgcnn.py`. Phase 1 baseline architecture.
- **Key Learnings**: (1) Graph construction logic and `GaussianDistance` encoding. (2) `ConvLayer` message-passing mechanism. (3) Global pooling strategy and its effect on property prediction.
- **Reading Status**: Not Started
- **Notes**: —

### A-02 | usnistgov/alignn | 297 stars

- **Affiliation**: NIST (Kamal Choudhary)
- **Grade**: A
- **Publication**: Atomistic Line Graph Neural Network for Improved Materials Property Predictions. *npj Computational Materials*, 2021. ~800 citations.
- **Description**: Augments the standard atom graph with a line graph that encodes bond angle information. Achieves state-of-the-art performance on multiple JARVIS-DFT tasks.
- **ATLAS Relevance**: Primary comparison model. Uses the same JARVIS dataset as ATLAS. Stronger baseline than CGCNN.
- **Key Learnings**: (1) Line graph construction for angular encoding. (2) Architectural differences from CGCNN. (3) Data normalization practices (cf. Issue #54).
- **Reading Status**: Not Started
- **Notes**: —

### A-03 | materialsvirtuallab/matgl | 509 stars

- **Affiliation**: UCSD Materials Virtual Lab (Shyue Ping Ong)
- **Grade**: A
- **Publication**: MatGL: A Framework for Graph Deep Learning in Materials Science. 2025. Integrates M3GNet, CHGNet, MEGNet, TensorNet, SO3Net.
- **Description**: Unified materials GNN framework supporting multiple architectures, pretrained foundation potentials, and PyTorch Lightning training. Integration with ASE and LAMMPS for molecular dynamics.
- **ATLAS Relevance**: Upstream reference for `atlas/models/m3gnet.py`. Potential source of pretrained weights for transfer learning.
- **Key Learnings**: (1) Multi-architecture unified API design. (2) Pretrained model loading and fine-tuning workflow.
- **Reading Status**: Not Started
- **Notes**: —

### A-04 | dhw059/DenseGNN | ~50 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: DenseGNN: Universal, Scalable, Efficient GNN. 2024. JARVIS-DFT state-of-the-art.
- **Description**: Combines dense connectivity networks, hierarchical node-edge-graph residual blocks, and Local structure Order Parameters Embedding (LOPE). Achieves top performance across JARVIS-DFT, Materials Project, and QM9.
- **ATLAS Relevance**: Direct comparison target on JARVIS-DFT benchmarks.
- **Reading Status**: Not Started
- **Notes**: —

### A-05 | hspark1212/crystal-gnn | ~30 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: Crystal GNN Benchmarking Framework.
- **Description**: Unified benchmarking framework integrating SchNet, CGCNN, ALIGNN, and others. Uses JARVIS-Tools and Matbench for standardized evaluation.
- **ATLAS Relevance**: Reference for benchmark pipeline design.
- **Reading Status**: Not Started
- **Notes**: —

### A-06 | modl-uclouvain/modnet | ~80 stars

- **Affiliation**: UCLouvain
- **Grade**: A
- **Publication**: MODNet: Materials Optimal Descriptor Network. ~200 citations. Top performer on 7 of 13 Matbench tasks.
- **Description**: Optimal feature selection combined with a concise neural network architecture. Demonstrates that well-chosen descriptors can outperform complex GNN architectures.
- **ATLAS Relevance**: Direct competitor on Matbench. ATLAS results must exceed MODNet performance to demonstrate value.
- **Key Learnings**: (1) Feature selection strategy that outperforms GNNs. (2) Benchmark numbers for comparison.
- **Reading Status**: Not Started
- **Notes**: —

### A-07 | CompRhys/aviary | ~100 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: Roost / Wren. ~200 citations.
- **Description**: Roost predicts properties from chemical composition alone. Wren uses Wyckoff representation. No full crystal structure required.
- **ATLAS Relevance**: Composition-only baseline for comparison.
- **Reading Status**: Not Started
- **Notes**: —

### A-08 | gasteigerjo/dimenet | ~400 stars

- **Affiliation**: TUM (Johannes Gasteiger)
- **Grade**: B
- **Publication**: DimeNet / DimeNet++. ~1000 citations.
- **Description**: Directional message passing using 2D angular basis functions to capture three-body interactions.
- **ATLAS Relevance**: Reference for angular information encoding in message passing.
- **Reading Status**: Not Started
- **Notes**: —

### A-09 | vgsatorras/egnn | ~500 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: E(n) Equivariant Graph Neural Networks. *ICML* 2021. ~1500 citations.
- **Description**: Minimalist E(n)-equivariant GNN without spherical harmonics. Updates coordinates directly. Simpler than e3nn-based approaches.
- **ATLAS Relevance**: Entry-level reference for understanding equivariant GNN concepts.
- **Reading Status**: Not Started
- **Notes**: —

### A-10 | janosh/matbench-discovery | ~100 stars

- **Affiliation**: —
- **Grade**: A
- **Publication**: Matbench Discovery interactive leaderboard.
- **Description**: Simulates high-throughput materials discovery. Ranks ML models on stable crystal prediction. Current top model: PET-OAM-XL (F1 = 0.924).
- **ATLAS Relevance**: Defines current state-of-the-art performance targets.
- **Key Learnings**: (1) Current best F1/R2/DAF values. (2) Which architectures dominate.
- **Reading Status**: Not Started
- **Notes**: —

### A-11 | AI4Cryst/ASUGNN | ~20 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: ASUGNN: Asymmetric Unit-Based GNN. *Journal of Applied Crystallography*, 2024.
- **Description**: Graph representation based on the crystallographic asymmetric unit. Reduces computational cost while preserving full symmetry information.
- **ATLAS Relevance**: Novel crystal representation approach. Potential efficiency gain.
- **Reading Status**: Not Started
- **Notes**: —

### A-12 | imatge-upc/CartNet | ~20 stars

- **Affiliation**: UPC
- **Grade**: B
- **Publication**: CartNet. *Digital Discovery*, 2025.
- **Description**: Encodes 3D geometry in Cartesian reference frame. Employs SO(3) rotational data augmentation during training for rotation invariance generalization.
- **ATLAS Relevance**: Data augmentation strategy for crystal GNNs.
- **Reading Status**: Not Started
- **Notes**: —

### A-13 | shrimonmuke0202/CrysAtom | ~15 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: CrysAtom. *LOG Conference*, 2024.
- **Description**: Unsupervised pretraining of atom representations from unlabeled crystal data. Plugs into existing GNN architectures to improve accuracy.
- **ATLAS Relevance**: Pretraining paradigm for crystal atoms.
- **Reading Status**: Not Started
- **Notes**: —

### A-14 | mamunm/MatGNN | ~20 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: MatGNN: PyTorch Lightning GNN for materials science.
- **Description**: PyTorch Lightning-based training framework supporting multiple GNN architectures for materials research.
- **ATLAS Relevance**: Training architecture reference.
- **Reading Status**: Not Started
- **Notes**: —

### A-15 | ihalage/Finder | ~30 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: Finder: Formula graph self-attention for materials discovery.
- **Description**: Self-attention on chemical formula graphs. Supports both composition-only and structure-based prediction modes.
- **ATLAS Relevance**: Dual-mode prediction reference.
- **Reading Status**: Not Started
- **Notes**: —

---

## Category 2: Equivariant Models and Machine-Learning Interatomic Potentials (15 entries)

### B-01 | mir-group/nequip | 867 stars

- **Affiliation**: Harvard (Boris Kozinsky, MIR Group)
- **Grade**: A
- **Publication**: E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials. *Nature Communications*, 2022. ~1500 citations.
- **Description**: First E(3)-equivariant interatomic potential. Uses e3nn tensor products for equivariant message passing. Achieves meV-level accuracy with as few as 100 training structures.
- **ATLAS Relevance**: Primary inspiration for `atlas/models/equivariant.py`. Phase 3 reference architecture.
- **Key Learnings**: (1) How equivariant convolutions preserve physical symmetry. (2) Data efficiency mechanisms. (3) e3nn tensor product usage patterns.
- **Reading Status**: Not Started
- **Notes**: —

### B-02 | ACEsuit/mace | 1,100 stars

- **Affiliation**: Cambridge (Gabor Csanyi, ACEsuit)
- **Grade**: A
- **Publication**: MACE: Higher Order Equivariant Message Passing for Fast and Accurate Force Fields. *NeurIPS*, 2022. ~800 citations.
- **Description**: Higher-order equivariant message passing. Incorporates more body-order interactions than NequIP, achieving higher accuracy while maintaining computational efficiency. MACE-MP provides universal foundation models.
- **ATLAS Relevance**: ATLAS MACE integration. Performance comparison target for equivariant models.
- **Key Learnings**: (1) Higher body-order significance. (2) Foundation model fine-tuning methodology.
- **Reading Status**: Not Started
- **Notes**: —

### B-03 | e3nn/e3nn | 1,200 stars

- **Affiliation**: Mario Geiger, Tess Smidt
- **Grade**: A
- **Publication**: e3nn: Euclidean Neural Networks. 2022. ~1000 citations.
- **Description**: Core framework for E(3)-equivariant tensor product operations. Provides spherical harmonics, tensor products, equivariant linear layers. Foundation for NequIP, MACE, Allegro.
- **ATLAS Relevance**: Core dependency of `atlas/models/equivariant.py`. Note: MACE Issue #555 documents version pinning conflicts.
- **Key Learnings**: (1) Spherical harmonics representation. (2) Irreducible representations. (3) Tensor product computation.
- **Reading Status**: Not Started
- **Notes**: —

### B-04 | CederGroupHub/chgnet | ~300 stars

- **Affiliation**: UC Berkeley (Gerbrand Ceder)
- **Grade**: A
- **Publication**: CHGNet: Pretrained Universal Neural Network Potential for Charge-Informed Atomistic Modelling. *Nature Machine Intelligence*, 2023. ~500 citations.
- **Description**: Charge-aware universal potential trained on >1M Materials Project structures with forces, stresses, and magnetic moments. Captures charge transfer effects.
- **ATLAS Relevance**: Universal potential comparison. Pretrained weights available for transfer learning.
- **Reading Status**: Not Started
- **Notes**: —

### B-05 | mir-group/allegro | ~300 stars

- **Affiliation**: Harvard (MIR Group)
- **Grade**: A
- **Publication**: Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics. *Nature Communications*, 2023.
- **Description**: Scalable extension of NequIP using local equivariant representations. Enables efficient parallelization for large-scale molecular dynamics simulations.
- **ATLAS Relevance**: Scaling strategy for equivariant models.
- **Reading Status**: Not Started
- **Notes**: —

### B-06 | mir-group/flare | ~300 stars

- **Affiliation**: Harvard (MIR Group)
- **Grade**: A
- **Publication**: FLARE: Fast Learning of Atomistic Rare Events. ~400 citations.
- **Description**: On-the-fly active learning with Bayesian uncertainty-aware molecular dynamics. During simulation, detects model uncertainty in real-time, triggers DFT calculations for uncertain configurations, and retrains automatically.
- **ATLAS Relevance**: [CRITICAL] Direct reference for combining UQ with active learning. Implements the uncertainty-driven data acquisition loop that ATLAS targets.
- **Key Learnings**: (1) Uncertainty-driven active learning closed loop. (2) Bayesian force field construction. (3) When and how to trigger retraining.
- **Reading Status**: Not Started
- **Notes**: —

### B-07 | deepmodeling/deepmd-kit | ~1,500 stars

- **Affiliation**: DeePMD Community (Peking University et al.)
- **Grade**: B
- **Publication**: DeePMD-kit. ~1500 citations.
- **Description**: Classical MLIP framework. v3 supports TensorFlow, PyTorch, JAX, PaddlePaddle backends. Extensive ecosystem (DPGEN, AIS-Square).
- **ATLAS Relevance**: Alternative MLIP approach for understanding the landscape.
- **Reading Status**: Not Started
- **Notes**: —

### B-08 | FAIR-Chem/fairchem | ~800 stars

- **Affiliation**: Meta FAIR
- **Grade**: A
- **Publication**: Encompasses OC20/OC22/OC25 datasets and UMA universal model.
- **Description**: Meta FAIR Chemistry unified platform. ML methods, data, models, demos. v2 introduces Universal Machine-learning for Atomistic systems (UMA). Multi-node multi-GPU and LAMMPS support.
- **ATLAS Relevance**: Engineering architecture reference. Largest-scale ML chemistry system.
- **Key Learnings**: (1) Multi-GPU training infrastructure. (2) Model deployment patterns.
- **Reading Status**: Not Started
- **Notes**: —

### B-09 | atomicarchitects/equiformer | ~200 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: Equiformer: Equivariant Graph Attention Transformer. 2023; Equiformer v2, 2024.
- **Description**: Combines Transformer attention mechanisms with equivariant representations. State-of-the-art on OC20.
- **ATLAS Relevance**: Frontier architecture trend: Transformer + equivariance.
- **Reading Status**: Not Started
- **Notes**: —

### B-10 | microsoft/mattersim | ~200 stars

- **Affiliation**: Microsoft Research
- **Grade**: B
- **Publication**: MatterSim: Large-scale AI pretrained materials simulation. 2024.
- **Description**: Industry-scale pretrained materials model for property prediction and dynamics simulation.
- **ATLAS Relevance**: Industry perspective on foundation models for materials.
- **Reading Status**: Not Started
- **Notes**: —

### B-11 | torchmd/torchmd-net | ~300 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: TensorNet: Cartesian Tensor Message Passing. 2024.
- **Description**: Cartesian tensor-based message passing that avoids the complexity of spherical harmonics.
- **ATLAS Relevance**: Alternative to spherical harmonics-based equivariant methods.
- **Reading Status**: Not Started
- **Notes**: —

### B-12 | Open-Catalyst-Project/ocp | ~800 stars

- **Affiliation**: Meta FAIR + Carnegie Mellon University
- **Grade**: B
- **Publication**: OC20. ~1000 citations.
- **Description**: Open Catalyst Project. 130M DFT calculations for catalysis. Though catalysis-focused, the scale and training methodology are instructive.
- **ATLAS Relevance**: Training methodology at scale.
- **Reading Status**: Not Started
- **Notes**: —

### B-13 | ACEsuit/mace-foundations | ~50 stars

- **Affiliation**: Cambridge
- **Grade**: A
- **Publication**: MACE foundation model weights.
- **Description**: MACE-MP-0 and other pretrained model weights. Current state-of-the-art comparison baseline.
- **ATLAS Relevance**: Pretrained weights for transfer learning experiments.
- **Reading Status**: Not Started
- **Notes**: —

### B-14 | ACEsuit/mace-tutorials | ~30 stars

- **Affiliation**: Cambridge
- **Grade**: B
- **Publication**: Official MACE tutorial notebooks.
- **Description**: Step-by-step tutorials for MACE training, fine-tuning, and deployment.
- **ATLAS Relevance**: Fastest path to learning MACE usage.
- **Reading Status**: Not Started
- **Notes**: —

### B-15 | ACEsuit/mace-off | ~50 stars

- **Affiliation**: Cambridge
- **Grade**: B
- **Publication**: MACE for organic force fields.
- **Description**: Pretrained models for organic molecular systems. Demonstrates cross-domain transfer.
- **ATLAS Relevance**: Cross-domain transfer learning example.
- **Reading Status**: Not Started
- **Notes**: —

---

## Category 3: Uncertainty Quantification (15 entries)

### C-01 | aamini/evidential-deep-learning | 508 stars

- **Affiliation**: MIT (Alexander Amini)
- **Grade**: A
- **Publication**: Deep Evidential Regression. *NeurIPS*, 2020. ~700 citations.
- **Description**: Single forward pass produces both prediction and uncertainty estimates via Normal-Inverse-Gamma (NIG) prior. Outputs four parameters (mu, nu, alpha, beta) that separate aleatoric and epistemic uncertainty.
- **ATLAS Relevance**: [CRITICAL] Direct source of `atlas/models/uq/EvidentialRegression`. Core UQ method.
- **Key Learnings**: (1) NIG loss function design. (2) Four-parameter interpretation. (3) Aleatoric vs. epistemic separation.
- **Reading Status**: Not Started
- **Notes**: —

### C-02 | google/uncertainty-baselines | ~1,000 stars

- **Affiliation**: Google
- **Grade**: A
- **Publication**: Unified baseline for multiple UQ papers.
- **Description**: Official Google UQ method comparison. Includes Ensemble, MC Dropout, Heteroscedastic, SNGP methods evaluated on ImageNet, CIFAR, UCI datasets.
- **ATLAS Relevance**: UQ method comparison framework reference.
- **Key Learnings**: (1) Standard UQ evaluation protocol. (2) How different methods compare empirically.
- **Reading Status**: Not Started
- **Notes**: —

### C-03 | cornellius-gp/gpytorch | 3,800 stars

- **Affiliation**: Cornell / CMU
- **Grade**: A
- **Publication**: GPyTorch. *NeurIPS*, 2018. ~2000 citations.
- **Description**: High-performance Gaussian Process framework supporting exact and approximate inference.
- **ATLAS Relevance**: ATLAS active learning surrogate model dependency.
- **Reading Status**: Not Started
- **Notes**: —

### C-04 | pytorch/botorch | ~3,200 stars

- **Affiliation**: Meta
- **Grade**: A
- **Publication**: BoTorch. *NeurIPS*, 2020. ~2000 citations.
- **Description**: Bayesian optimization framework built on GPyTorch. Supports multi-objective, multi-fidelity, batch optimization.
- **ATLAS Relevance**: ATLAS dependency for acquisition function computation in active learning.
- **Reading Status**: Not Started
- **Notes**: —

### C-05 | uncertainty-toolbox/uncertainty-toolbox | 2,000 stars

- **Affiliation**: Stanford
- **Grade**: A
- **Publication**: ~500 citations.
- **Description**: Comprehensive UQ evaluation and calibration toolkit. Supports calibration plots, sharpness metrics, miscalibration area, and other standard UQ quality measures.
- **ATLAS Relevance**: [CRITICAL] Tool for generating publication-quality calibration plots and UQ evaluation metrics.
- **Key Learnings**: (1) Standard UQ calibration metrics. (2) How to produce calibration plots for publications.
- **Reading Status**: Not Started
- **Notes**: —

### C-06 | chemprop/chemprop | 128 stars (v2)

- **Affiliation**: MIT (Kevin Yang, Rafael Gomez-Bombarelli)
- **Grade**: A
- **Publication**: Chemprop. *Journal of Chemical Information and Modeling*, 2019. ~1500 citations.
- **Description**: Directed message-passing neural network for molecular property prediction with built-in UQ via ensemble methods and calibration tools. Best-in-class example of GNN + UQ integration (molecular domain).
- **ATLAS Relevance**: [CRITICAL] Reference implementation for integrating UQ into a GNN training pipeline.
- **Key Learnings**: (1) How ensemble UQ is embedded in training. (2) Calibration method implementations.
- **Reading Status**: Not Started
- **Notes**: —

### C-07 | learningmatter-mit/matex | ~30 stars

- **Affiliation**: MIT (Rafael Gomez-Bombarelli)
- **Grade**: A
- **Publication**: Known Unknowns: Out-of-Distribution Property Prediction in Materials and Molecules.
- **Description**: Investigates ML model failure modes during extrapolation in materials and molecular property prediction. Provides methods for detecting out-of-distribution inputs.
- **ATLAS Relevance**: [CRITICAL] Directly targets ATLAS OOD detection objectives.
- **Key Learnings**: (1) How to define OOD in materials contexts. (2) Extrapolation boundary determination. (3) OOD detection metrics.
- **Reading Status**: Not Started
- **Notes**: —

### C-08 | tirtha-v/DPOSE-GNN | ~20 stars

- **Affiliation**: —
- **Grade**: A
- **Publication**: DPOSE: Direct Propagation of Shallow Ensembles for UQ in GNNs. 2024.
- **Description**: Shallow ensemble approach for UQ in GNNs applied to DFT datasets (QM9, OC20, Gold MD). Demonstrates reliable separation of in-domain and out-of-domain samples via uncertainty magnitude.
- **ATLAS Relevance**: [CRITICAL] Direct competitor. UQ + GNN + materials.
- **Reading Status**: Not Started
- **Notes**: —

### C-09 | AutoGNNUQ | ~15 stars

- **Affiliation**: —
- **Grade**: A
- **Publication**: AutoGNNUQ: Automated UQ for Molecular Property Prediction with GNN Architecture Search. *Digital Discovery*, 2024.
- **Description**: Uses neural architecture search to generate an ensemble of high-performing GNNs for uncertainty estimation. Applies variance decomposition to separate aleatoric and epistemic components. Includes recalibration.
- **ATLAS Relevance**: [CRITICAL] Automated approach to UQ architecture selection. Variance decomposition methodology.
- **Key Learnings**: (1) NAS for UQ ensemble construction. (2) Variance decomposition for aleatoric/epistemic separation.
- **Reading Status**: Not Started
- **Notes**: —

### C-10 | snap-stanford/conformalized-gnn | ~100 stars

- **Affiliation**: Stanford
- **Grade**: A
- **Publication**: Uncertainty Quantification over Graph with Conformalized Graph Neural Networks. *NeurIPS*, 2023.
- **Description**: Extends conformal prediction to graph-structured data. Provides distribution-free prediction intervals with guaranteed coverage probability. Includes topology-aware output correction to reduce interval width.
- **ATLAS Relevance**: [CRITICAL] Distribution-free UQ with statistical guarantees. Applicable to ATLAS crystal GNN predictions.
- **Key Learnings**: (1) Conformal prediction on GNNs. (2) Coverage guarantee mechanics. (3) Topology-aware correction.
- **Reading Status**: Not Started
- **Notes**: —

### C-11 | peiyaoli/Conformal-ADMET-Prediction | ~10 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: Conformalized Fusion Regression. 2024.
- **Description**: GNN with joint mean-quantile regression loss and ensemble-based conformal prediction for molecular properties. Demonstrates calibrated uncertainty and high-quality prediction intervals.
- **ATLAS Relevance**: Implementation reference for conformal prediction in molecular GNNs.
- **Reading Status**: Not Started
- **Notes**: —

### C-12 | SeongokRyu/uq_molecule | ~30 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: UQ of Molecular Property Prediction via Bayesian Deep Learning.
- **Description**: MC-Dropout with augmented graph convolutional networks for molecular UQ.
- **ATLAS Relevance**: Bayesian UQ implementation reference for GNNs.
- **Reading Status**: Not Started
- **Notes**: —

### C-13 | ljatynu/CPBayesMPP | ~10 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: Contrastive Prior Enhances Performance of Bayesian Neural Network-Based Molecular Property Prediction.
- **Description**: Combines contrastive learning priors with BNNs to simultaneously improve prediction accuracy and UQ quality. Includes calibration curve generation.
- **ATLAS Relevance**: Contrastive learning + BNN approach. Calibration methodology.
- **Reading Status**: Not Started
- **Notes**: —

### C-14 | y0ast/deterministic-uncertainty-quantification | ~200 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: DUQ. *ICML*, 2020.
- **Description**: Deterministic UQ without multiple forward passes. Uses RBF kernel-based distance estimation.
- **ATLAS Relevance**: Alternative single-pass UQ method for comparison with evidential regression.
- **Reading Status**: Not Started
- **Notes**: —

### C-15 | clqz-trustml/conformal-prediction | ~100 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: Conformal Prediction paper collection (categorized by year and venue).
- **Description**: Systematic index of conformal prediction papers from ICLR, ICML, NeurIPS, and others.
- **ATLAS Relevance**: Entry point for finding state-of-the-art conformal prediction methods.
- **Reading Status**: Not Started
- **Notes**: —

---

## Category 4: Materials Databases and Tools (10 entries)

### D-01 | materialsproject/pymatgen | ~2,000 stars

- **Affiliation**: LBNL / Materials Project
- **Grade**: A
- **Publication**: pymatgen. *Computational Materials Science*, 2013. ~5000 citations.
- **Description**: Core Python library for materials science. Crystal structure analysis, symmetry operations, phase diagrams, electronic structure analysis.
- **ATLAS Relevance**: Direct ATLAS dependency for structure processing and feature extraction.
- **Reading Status**: Not Started
- **Notes**: —

### D-02 | hackingmaterials/matminer | ~500 stars

- **Affiliation**: LBNL (Anubhav Jain)
- **Grade**: A
- **Publication**: matminer. *Computational Materials Science*, 2018. ~1000 citations.
- **Description**: Materials data mining and feature engineering toolkit. Provides 100+ descriptors (compositional, structural, electronic) with automated featurization.
- **ATLAS Relevance**: Feature extraction for RF baseline models.
- **Reading Status**: Not Started
- **Notes**: —

### D-03 | usnistgov/jarvis | ~300 stars

- **Affiliation**: NIST (Kamal Choudhary)
- **Grade**: A
- **Publication**: JARVIS. *npj Computational Materials*, 2020. ~500 citations.
- **Description**: Complete toolkit encompassing the DFT database (~76K materials, ~50 properties), ML training tools, and benchmarking infrastructure.
- **ATLAS Relevance**: Upstream data source for `atlas/data/jarvis_client.py`.
- **Reading Status**: Not Started
- **Notes**: —

### D-04 | materialsproject/matbench | ~100 stars

- **Affiliation**: LBNL / Materials Project
- **Grade**: A
- **Publication**: Matbench. ~300 citations.
- **Description**: 13 standardized materials ML benchmark tasks with 5-fold cross-validation splits. Covers electronic, thermal, mechanical, and thermodynamic properties.
- **ATLAS Relevance**: Dependency of `atlas/benchmarks/MatbenchRunner`. Required for reproducible evaluation.
- **Reading Status**: Not Started
- **Notes**: —

### D-05 | SINGROUP/dscribe | ~400 stars

- **Affiliation**: Aalto University
- **Grade**: B
- **Publication**: DScribe. *Computer Physics Communications*, 2020. ~500 citations.
- **Description**: Atomic structure descriptor library. SOAP, ACSF, Coulomb Matrix, MBTR implementations.
- **ATLAS Relevance**: Descriptor baseline for traditional ML comparison.
- **Reading Status**: Not Started
- **Notes**: —

### D-06 | hackingmaterials/automatminer | ~130 stars

- **Affiliation**: LBNL
- **Grade**: B
- **Description**: Automated ML pipeline for materials. Automatic feature selection, model selection, and hyperparameter tuning.
- **ATLAS Relevance**: AutoML reference for materials property prediction.
- **Reading Status**: Not Started
- **Notes**: —

### D-07 | materialsproject/mp-api | ~300 stars

- **Affiliation**: LBNL
- **Grade**: B
- **Description**: Official Materials Project Python API client. Programmatic access to ~200K materials.
- **ATLAS Relevance**: Required for cross-database OOD experiments (train JARVIS, test MP).
- **Reading Status**: Not Started
- **Notes**: —

### D-08 | WMD-group/SMACT | ~100 stars

- **Affiliation**: Imperial College London
- **Grade**: B
- **Description**: Semiconducting Materials by Analogy and Chemical Theory. Chemical composition feasibility filtering via charge balance, ionic radii, and electronegativity checks.
- **ATLAS Relevance**: Chemistry prior gate in the ATLAS discovery pipeline.
- **Reading Status**: Not Started
- **Notes**: —

### D-09 | hackingmaterials/atomate | ~400 stars

- **Affiliation**: LBNL
- **Grade**: A
- **Publication**: ~700 citations.
- **Description**: Materials computation workflow automation using FireWorks. Manages DFT calculation pipelines.
- **ATLAS Relevance**: Required if generating custom training data via DFT.
- **Reading Status**: Not Started
- **Notes**: —

### D-10 | sedaoturak/data-resources-for-materials-science

- **Affiliation**: —
- **Grade**: B
- **Description**: Curated list of publicly accessible materials science databases and dataset-sharing platforms.
- **ATLAS Relevance**: Quick reference for identifying additional data sources.
- **Reading Status**: Not Started
- **Notes**: —

---

## Category 5: Active Learning and Bayesian Optimization (10 entries)

### E-01 | emdgroup/baybe | ~200 stars

- **Affiliation**: Merck KGaA / Acceleration Consortium
- **Grade**: A
- **Publication**: BayBE. *RSC*, 2024. ~200 citations.
- **Description**: Domain-specific Bayesian optimization for chemistry and materials science. Supports chemical encodings, hybrid search spaces, multi-target optimization. Reduces experimental iterations by 50%+ versus default implementations.
- **ATLAS Relevance**: Best practice reference for domain-specific Bayesian optimization.
- **Key Learnings**: (1) Domain-specific BO configuration. (2) Chemical encoding strategies.
- **Reading Status**: Not Started
- **Notes**: —

### E-02 | deepmodeling/dpgen | ~300 stars

- **Affiliation**: DeePMD Community
- **Grade**: A
- **Publication**: DPGEN. *Computer Physics Communications*, 2020. ~500 citations.
- **Description**: Automatic potential generation pipeline. Active learning loop: select structures, run DFT, retrain potential, repeat. Complete closed-loop example of AL for interatomic potential training.
- **ATLAS Relevance**: Full closed-loop AL reference implementation.
- **Reading Status**: Not Started
- **Notes**: —

### E-03 | facebook/Ax | ~2,400 stars

- **Affiliation**: Meta
- **Grade**: B
- **Description**: General-purpose BO platform (BoTorch upper-layer wrapper). Multi-objective, multi-fidelity, batch optimization.
- **ATLAS Relevance**: BoTorch integration reference.
- **Reading Status**: Not Started
- **Notes**: —

### E-04 | emukit/emukit | ~400 stars

- **Affiliation**: Amazon / Secondmind
- **Grade**: B
- **Description**: Integrated toolkit for UQ, BO, AL, and sensitivity analysis. Modular design.
- **ATLAS Relevance**: UQ + BO integration patterns.
- **Reading Status**: Not Started
- **Notes**: —

### E-05 | modAL-python/modAL | ~2,200 stars

- **Affiliation**: —
- **Grade**: B
- **Description**: Modular active learning framework based on scikit-learn. Flexible query strategies.
- **ATLAS Relevance**: AL baseline implementation.
- **Reading Status**: Not Started
- **Notes**: —

### E-06 | scikit-activeml/scikit-activeml | ~300 stars

- **Affiliation**: —
- **Grade**: B
- **Description**: Modern alternative to modAL with more state-of-the-art strategies and active maintenance.
- **ATLAS Relevance**: Updated AL method implementations.
- **Reading Status**: Not Started
- **Notes**: —

### E-07 | HIPS/dragonfly | ~700 stars

- **Affiliation**: Harvard
- **Grade**: B
- **Description**: Scalable Bayesian optimization. Supports high-dimensional and multi-fidelity settings.
- **ATLAS Relevance**: Scalable BO reference.
- **Reading Status**: Not Started
- **Notes**: —

### E-08 | hackingmaterials/rocketsled | ~50 stars

- **Affiliation**: LBNL
- **Grade**: B
- **Description**: Bayesian optimization embedded within Fireworks materials computation workflows.
- **ATLAS Relevance**: BO integration in materials workflow context.
- **Reading Status**: Not Started
- **Notes**: —

### E-09 | materialsproject/atomate2 | ~200 stars

- **Affiliation**: LBNL
- **Grade**: B
- **Description**: Next-generation materials computation workflow framework.
- **ATLAS Relevance**: DFT workflow automation if generating custom training data.
- **Reading Status**: Not Started
- **Notes**: —

### E-10 | mir-group/FLARE-Tutorials | ~30 stars

- **Affiliation**: Harvard (MIR Group)
- **Grade**: B
- **Description**: Official FLARE tutorial notebooks. Entry point for learning on-the-fly active learning.
- **ATLAS Relevance**: Companion to B-06 (FLARE). Starting point for AL implementation.
- **Reading Status**: Not Started
- **Notes**: —

---

## Category 6: Explainability and Robustness (8 entries)

### F-01 | pyg-team/pytorch_geometric | ~21,000 stars

- **Affiliation**: PyG Team
- **Grade**: A
- **Description**: Most widely used GNN framework. Includes built-in GNNExplainer for model interpretation.
- **ATLAS Relevance**: ATLAS GNN backbone framework. GNNExplainer for prediction interpretation.
- **Reading Status**: Not Started
- **Notes**: —

### F-02 | pytorch/captum | ~1,500 stars

- **Affiliation**: Meta
- **Grade**: B
- **Description**: PyTorch model interpretability library. Supports IntegratedGradients, DeepLift, SHAP, and other attribution methods.
- **ATLAS Relevance**: Attribution analysis for equivariant models.
- **Reading Status**: Not Started
- **Notes**: —

### F-03 | shap/shap | ~23,000 stars

- **Affiliation**: Scott Lundberg
- **Grade**: B
- **Description**: SHapley Additive exPlanations. Standard tool for feature importance analysis.
- **ATLAS Relevance**: Required for RF baseline feature importance plots in publications.
- **Reading Status**: Not Started
- **Notes**: —

### F-04 | divelab/DIG | ~400 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: DIG: Dive into Graphs.
- **Description**: Unified GNN explanation benchmark (GNNExplainer, PGExplainer, SubgraphX) with standardized evaluation.
- **ATLAS Relevance**: GNN explanation method comparison framework.
- **Reading Status**: Not Started
- **Notes**: —

### F-05 | learningmatter-mit/Atomistic-Adversarial-Attacks | ~20 stars

- **Affiliation**: MIT (Rafael Gomez-Bombarelli)
- **Grade**: A
- **Publication**: Adversarial attacks on atomistic systems using neural network potentials.
- **Description**: Tests robustness of neural network potentials by generating adversarial atomic configurations. Directly measures model reliability under perturbation.
- **ATLAS Relevance**: [CRITICAL] Robustness testing methodology for neural potentials. Directly related to OOD and reliability assessment.
- **Key Learnings**: (1) How to construct adversarial examples for atomistic models. (2) Robustness evaluation metrics.
- **Reading Status**: Not Started
- **Notes**: —

### F-06 | interpretml/interpret | ~6,000 stars

- **Affiliation**: Microsoft
- **Grade**: B
- **Description**: Explainable Boosting Machines (EBM), linear models, decision trees. Glassbox interpretable models.
- **ATLAS Relevance**: Interpretable baseline model reference.
- **Reading Status**: Not Started
- **Notes**: —

### F-07 | learningmatter-mit/NeuralForceField | ~50 stars

- **Affiliation**: MIT (Rafael Gomez-Bombarelli)
- **Grade**: B
- **Description**: PyTorch-based neural force field implementation. Educational reference from the Learning Matter group.
- **ATLAS Relevance**: Teaching-quality neural force field implementation.
- **Reading Status**: Not Started
- **Notes**: —

### F-08 | learningmatter-mit/LLM4BO | ~30 stars

- **Affiliation**: MIT
- **Grade**: B
- **Publication**: LLM for Bayesian Optimization benchmark. 2024.
- **Description**: Benchmarks LLM approaches for Bayesian optimization. Emerging trend in AI-driven experiment design.
- **ATLAS Relevance**: Emerging trend awareness: LLM-assisted optimization.
- **Reading Status**: Not Started
- **Notes**: —

---

## Category 7: Benchmarks and Resource Indices (10 entries)

### G-01 | WanyuGroup/AI-for-Crystal-Materials | ~200 stars

- **Affiliation**: —
- **Grade**: A
- **Description**: Systematic collection of AI for Crystalline Materials papers from NeurIPS, ICML, ICLR, AAAI. Regularly updated with code links.
- **ATLAS Relevance**: Primary entry point for discovering new repositories and papers.
- **Reading Status**: Not Started
- **Notes**: —

### G-02 | kdmsit/Awesome-Crystal-GNNs | ~50 stars

- **Affiliation**: —
- **Grade**: B
- **Description**: Crystal GNN paper collection (2024-2025). Includes ComFormer, CrystalFormer, GMTNet, Conformal Crystal Graph Transformer.
- **ATLAS Relevance**: Index of latest crystal GNN architectures.
- **Reading Status**: Not Started
- **Notes**: —

### G-03 | JuDFTteam/best-of-atomistic-machine-learning | ~800 stars

- **Affiliation**: —
- **Grade**: A
- **Description**: 510+ ranked atomistic ML projects across 23 categories. Quality score based on GitHub metrics.
- **ATLAS Relevance**: Comprehensive check for missing important repositories.
- **Reading Status**: Not Started
- **Notes**: —

### G-04 | tilde-lab/awesome-materials-informatics | ~500 stars

- **Affiliation**: —
- **Grade**: B
- **Description**: Curated materials informatics resource list covering libraries, platforms, and datasets.
- **ATLAS Relevance**: Supplementary search entry point.
- **Reading Status**: Not Started
- **Notes**: —

### G-05 | modl-uclouvain/modnet-matbench | ~20 stars

- **Affiliation**: UCLouvain
- **Grade**: B
- **Description**: MODNet Matbench submission results and reproduction code.
- **ATLAS Relevance**: Direct numerical comparison for Matbench results.
- **Reading Status**: Not Started
- **Notes**: —

### G-06 | jakobzeitler/bayesian-deep-learning | ~300 stars

- **Affiliation**: —
- **Grade**: B
- **Description**: Curated Bayesian deep learning paper list.
- **ATLAS Relevance**: Index for BDL-related UQ papers.
- **Reading Status**: Not Started
- **Notes**: —

### G-07 | deepmodeling/AIS-Square | ~50 stars

- **Affiliation**: DeePMD Community
- **Grade**: B
- **Description**: AI for Science model-sharing platform. Hosts DPA-1 attention-based potential model.
- **ATLAS Relevance**: Pretrained model discovery.
- **Reading Status**: Not Started
- **Notes**: —

### G-08 | vertaix/LLM4Mat-Bench | ~50 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: LLM4Mat-Bench. 2024.
- **Description**: Benchmark for evaluating LLMs on crystalline material property prediction.
- **ATLAS Relevance**: Trend awareness: LLM-based materials prediction.
- **Reading Status**: Not Started
- **Notes**: —

### G-09 | vertaix/LLM-Prop | ~30 stars

- **Affiliation**: —
- **Grade**: B
- **Publication**: LLM-Prop: Fine-tuned T5 encoder for crystal properties.
- **Description**: Text-to-property prediction using fine-tuned language models.
- **ATLAS Relevance**: Alternative modality reference.
- **Reading Status**: Not Started
- **Notes**: —

### G-10 | UQ4DD

- **Affiliation**: —
- **Grade**: B
- **Description**: Uncertainty Quantification for Drug Discovery framework. Supports Platt scaling, Venn-ABERS calibration, ensemble and Bayesian methods.
- **ATLAS Relevance**: Calibration methods transferable to materials domain.
- **Reading Status**: Not Started
- **Notes**: —

---

## Category 8: Software Engineering and MLOps (7 entries)

### H-01 | wandb/wandb | ~9,000 stars

- **Affiliation**: Weights and Biases
- **Grade**: B
- **Description**: Experiment tracking. Loss/metrics visualization. Multi-run comparison.
- **ATLAS Relevance**: Training monitoring and visualization.
- **Reading Status**: Not Started
- **Notes**: —

### H-02 | mlflow/mlflow | ~19,000 stars

- **Affiliation**: Databricks
- **Grade**: B
- **Description**: Model versioning, experiment management, deployment.
- **ATLAS Relevance**: Experiment management alternative.
- **Reading Status**: Not Started
- **Notes**: —

### H-03 | iterative/dvc | ~14,000 stars

- **Affiliation**: —
- **Grade**: B
- **Description**: Data version control for large datasets.
- **ATLAS Relevance**: Large dataset tracking for reproducibility.
- **Reading Status**: Not Started
- **Notes**: —

### H-04 | facebookresearch/hydra | ~8,500 stars

- **Affiliation**: Meta
- **Grade**: B
- **Description**: YAML-based configuration composition framework.
- **ATLAS Relevance**: Configuration management reference to replace dataclass configs.
- **Reading Status**: Not Started
- **Notes**: —

### H-05 | Lightning-AI/pytorch-lightning | ~28,000 stars

- **Affiliation**: Lightning AI
- **Grade**: B
- **Description**: Structured PyTorch training framework. Used by MatGL.
- **ATLAS Relevance**: Training framework reference.
- **Reading Status**: Not Started
- **Notes**: —

### H-06 | pyg-team/pytorch_geometric | ~21,000 stars

- **Affiliation**: PyG Team
- **Grade**: A
- **Description**: Cross-reference to F-01. ATLAS GNN backbone framework.
- **Reading Status**: Not Started
- **Notes**: —

### H-07 | mir-group/nequip-tutorial | ~20 stars

- **Affiliation**: Harvard (MIR Group)
- **Grade**: B
- **Description**: NequIP tutorial notebooks. Entry point for equivariant potential training.
- **ATLAS Relevance**: Companion to B-01 (NequIP).
- **Reading Status**: Not Started
- **Notes**: —

---

## Summary Statistics

| Category | Entries | Grade A | Grade B |
|----------|---------|---------|---------|
| 1. Crystal GNN | 15 | 5 | 10 |
| 2. Equivariant / MLIP | 15 | 8 | 7 |
| 3. Uncertainty Quantification | 15 | 10 | 5 |
| 4. Materials Tools | 10 | 5 | 5 |
| 5. Active Learning / BO | 10 | 2 | 8 |
| 6. Explainability / Robustness | 8 | 2 | 6 |
| 7. Benchmarks / Indices | 10 | 2 | 8 |
| 8. Software Engineering | 7 | 1 | 6 |
| **Total** | **90** | **35** | **55** |

All entries are Grade A or B. No filler entries included.

### Priority Reading Order (Top 10)

| Priority | ID | Repository | Rationale |
|----------|----|------------|-----------|
| 1 | C-01 | aamini/evidential-deep-learning | ATLAS UQ core source |
| 2 | C-07 | learningmatter-mit/matex | ATLAS OOD core reference |
| 3 | B-06 | mir-group/flare | UQ + AL closed loop |
| 4 | C-10 | snap-stanford/conformalized-gnn | Distribution-free UQ with guarantees |
| 5 | C-06 | chemprop/chemprop | GNN + UQ integration best practice |
| 6 | A-01 | txie-93/cgcnn | ATLAS baseline architecture |
| 7 | B-01 | mir-group/nequip | ATLAS equivariant source |
| 8 | C-08 | tirtha-v/DPOSE-GNN | Direct UQ + GNN competitor |
| 9 | C-09 | AutoGNNUQ | Automated UQ architecture search |
| 10 | A-06 | modl-uclouvain/modnet | Matbench primary competitor |

---
---

# 中文版：晶體 GNN 與不確定性量化研究之開源倉庫調研

> **專案**：ATLAS — 原子結構自適應訓練與學習  
> **作者**：Zhong  
> **日期**：2026-02-27  
> **總條目**：90  
> **狀態**：第一階段種子收集完成

---

## 評估方法

### 篩選標準

每個倉庫基於以下 10 個維度進行評估：

| 維度 | 權重 | 說明 |
|------|------|------|
| 機構背景 | 高 | 是否隸屬知名研究組（MIT、Harvard、Cambridge、NIST 等） |
| 論文影響力 | 高 | 對應論文的引用數與發表期刊 |
| 可重現性 | 高 | 是否有清晰安裝說明、可運行範例、可成功復現結果 |
| 領域相關度 | 高 | 是否直接涉及 UQ、OOD 偵測、晶體 GNN 或 Matbench |
| 測試覆蓋 | 中 | 是否有單元測試、CI 工作流 |
| 文件品質 | 中 | API 文件、文件字串、教學 |
| 維護活躍度 | 中 | 近期提交頻率、Issue 回應速度 |
| 下游採用 | 中 | 是否被其他知名倉庫引用或依賴 |
| 社群治理 | 低 | 是否屬於組織、是否有貢獻指南 |
| 程式碼品質 | 低 | 型別註解、架構清晰度 |

### 分級體系

| 等級 | 標準 | 要求動作 |
|------|------|----------|
| **A** | 滿足 3 個以上高權重維度 | 深入閱讀程式碼、重現實驗、撰寫詳細筆記 |
| **B** | 滿足 2 個以上高權重或 4 個以上中權重維度 | 閱讀 README、核心模組程式碼、記錄關鍵要點 |

所有條目均為 A 或 B 級。低相關性倉庫已排除。

---

## 第一類：晶體性質預測 GNN（15 個）

### A-01 | txie-93/cgcnn | 822 星

- **機構**：MIT（Tian Xie, Jeffrey Grossman）
- **等級**：A
- **論文**：Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties. *Physical Review Letters*, 2018. 約 3000 次引用。
- **描述**：首個直接從晶體結構端到端預測性質的 GNN。原子為節點、鍵為邊。使用高斯距離展開編碼邊特徵，全域平均池化後接全連接層。
- **與 ATLAS 關聯**：`atlas/models/cgcnn.py` 的直接基礎。Phase 1 基線架構。
- **核心學習點**：(1) 圖構建邏輯與 GaussianDistance 編碼方式。(2) ConvLayer 訊息傳遞機制。(3) 全域池化策略對性質預測的影響。
- **閱讀狀態**：未開始
- **筆記**：—

### A-02 | usnistgov/alignn | 297 星

- **機構**：NIST（Kamal Choudhary）
- **等級**：A
- **論文**：Atomistic Line Graph Neural Network for Improved Materials Property Predictions. *npj Computational Materials*, 2021. 約 800 次引用。
- **描述**：在原子圖基礎上增加線圖（line graph）編碼鍵角資訊。在多個 JARVIS-DFT 任務上取得最優表現。
- **與 ATLAS 關聯**：主要對比模型。使用與 ATLAS 相同的 JARVIS 資料集。比 CGCNN 更強的基線。
- **核心學習點**：(1) 線圖如何編碼角度資訊。(2) 與 CGCNN 的架構差異。(3) 資料歸一化實踐（參見 Issue #54）。
- **閱讀狀態**：未開始
- **筆記**：—

### A-03 | materialsvirtuallab/matgl | 509 星

- **機構**：UCSD 材料虛擬實驗室（Shyue Ping Ong）
- **等級**：A
- **論文**：MatGL: A Framework for Graph Deep Learning in Materials Science, 2025. 整合 M3GNet、CHGNet、MEGNet、TensorNet、SO3Net。
- **描述**：統一材料 GNN 框架，支援多種架構、預訓練基礎勢能和 PyTorch Lightning 訓練。與 ASE 和 LAMMPS 整合。
- **與 ATLAS 關聯**：`atlas/models/m3gnet.py` 的上游參考。可用於遷移學習的預訓練權重來源。
- **核心學習點**：(1) 多架構統一 API 設計。(2) 預訓練模型載入與微調流程。
- **閱讀狀態**：未開始
- **筆記**：—

### A-04 | dhw059/DenseGNN | 約 50 星

- **等級**：B
- **論文**：DenseGNN, 2024. JARVIS-DFT 最優表現。
- **描述**：結合密集連接網路、層次化節點-邊-圖殘差區塊和局部結構序參數嵌入（LOPE）。在 JARVIS-DFT、Materials Project、QM9 上取得頂尖性能。
- **與 ATLAS 關聯**：JARVIS-DFT 上的直接對比目標。
- **閱讀狀態**：未開始

### A-05 | hspark1212/crystal-gnn | 約 30 星

- **等級**：B
- **描述**：統一基準測試框架，整合 SchNet、CGCNN、ALIGNN 等。使用 JARVIS 和 Matbench 做標準化評估。
- **與 ATLAS 關聯**：基準測試流程設計參考。
- **閱讀狀態**：未開始

### A-06 | modl-uclouvain/modnet | 約 80 星

- **機構**：UCLouvain
- **等級**：A
- **論文**：MODNet. 約 200 次引用。Matbench 13 項任務中 7 項最優。
- **描述**：最優特徵選取結合簡潔神經網路架構。證明了精心選取的描述子可以勝過複雜 GNN。
- **與 ATLAS 關聯**：Matbench 上的直接競爭者。ATLAS 結果必須超越 MODNet 方有價值。
- **核心學習點**：(1) 勝過 GNN 的特徵選取策略。(2) 對比用的基準數字。
- **閱讀狀態**：未開始

### A-07 | CompRhys/aviary | 約 100 星

- **等級**：B
- **論文**：Roost / Wren. 約 200 次引用。
- **描述**：Roost 僅用化學組成預測性質；Wren 用 Wyckoff 表示。不需完整晶體結構。
- **與 ATLAS 關聯**：純組成基線對照。
- **閱讀狀態**：未開始

### A-08 | gasteigerjo/dimenet | 約 400 星

- **機構**：TUM（Johannes Gasteiger）
- **等級**：B
- **論文**：DimeNet / DimeNet++. 約 1000 次引用。
- **描述**：方向性訊息傳遞神經網路。使用 2D 角度基函數捕捉三體交互。
- **與 ATLAS 關聯**：訊息傳遞中角度資訊編碼的參考。
- **閱讀狀態**：未開始

### A-09 | vgsatorras/egnn | 約 500 星

- **等級**：B
- **論文**：E(n) Equivariant Graph Neural Networks. *ICML* 2021. 約 1500 次引用。
- **描述**：極簡 E(n)-等變 GNN，無需球諧函數，直接更新座標。比 e3nn 方案簡單得多。
- **與 ATLAS 關聯**：理解等變 GNN 概念的入門參考。
- **閱讀狀態**：未開始

### A-10 | janosh/matbench-discovery | 約 100 星

- **等級**：A
- **描述**：模擬高通量材料發現的互動式排行榜。當前最優模型：PET-OAM-XL (F1 = 0.924)。
- **與 ATLAS 關聯**：定義當前最優性能目標。
- **核心學習點**：(1) 當前最佳 F1/R2/DAF 數值。(2) 哪些架構佔據主導地位。
- **閱讀狀態**：未開始

### A-11 至 A-15（簡要列表）

| 編號 | 倉庫 | 等級 | 核心價值 |
|------|------|------|----------|
| A-11 | AI4Cryst/ASUGNN | B | 不對稱單位圖表示，減少計算量（2024） |
| A-12 | imatge-upc/CartNet | B | 笛卡爾編碼 + SO(3) 旋轉資料增強（2025） |
| A-13 | shrimonmuke0202/CrysAtom | B | 無監督原子表示預訓練（LOG 2024） |
| A-14 | mamunm/MatGNN | B | PyTorch Lightning 材料 GNN 訓練框架 |
| A-15 | ihalage/Finder | B | 化學式圖自注意力，雙模式預測 |

---

## 第二類：等變模型與機器學習原子間勢能（15 個）

### B-01 | mir-group/nequip | 867 星

- **機構**：Harvard（Boris Kozinsky, MIR Group）
- **等級**：A
- **論文**：E(3)-Equivariant GNN for Data-Efficient and Accurate Interatomic Potentials. *Nature Communications*, 2022. 約 1500 次引用。
- **描述**：首個 E(3)-等變原子間勢能。使用 e3nn 張量積進行等變訊息傳遞。僅需 100 個訓練結構即可達到 meV 級精度。
- **與 ATLAS 關聯**：`atlas/models/equivariant.py` 的主要靈感來源。Phase 3 參考架構。
- **核心學習點**：(1) 等變卷積如何保持物理對稱性。(2) 資料效率機制。(3) e3nn 張量積使用模式。
- **閱讀狀態**：未開始

### B-02 | ACEsuit/mace | 1,100 星

- **機構**：Cambridge（Gabor Csanyi）
- **等級**：A
- **論文**：MACE: Higher Order Equivariant MP. *NeurIPS*, 2022. 約 800 次引用。
- **描述**：高階等變訊息傳遞。比 NequIP 使用更多體序交互，精度更高同時保持計算效率。MACE-MP 為通用基礎模型。
- **與 ATLAS 關聯**：ATLAS MACE 整合。等變模型的性能對比目標。
- **核心學習點**：(1) 高階體序的意義。(2) 基礎模型微調方法論。
- **閱讀狀態**：未開始

### B-03 | e3nn/e3nn | 1,200 星

- **等級**：A
- **論文**：e3nn: Euclidean Neural Networks, 2022. 約 1000 次引用。
- **描述**：E(3)-等變張量積運算的核心框架。提供球諧函數、張量積、等變線性層。NequIP、MACE、Allegro 的底層基礎。
- **與 ATLAS 關聯**：`atlas/models/equivariant.py` 的核心依賴。注意 MACE Issue #555 的版本鎖定衝突。
- **核心學習點**：(1) 球諧函數表示。(2) 不可約表示。(3) 張量積計算。
- **閱讀狀態**：未開始

### B-04 至 B-15（簡要列表）

| 編號 | 倉庫 | 星數 | 等級 | 核心價值 |
|------|------|------|------|----------|
| B-04 | CederGroupHub/chgnet | 約 300 | A | 電荷感知通用勢能，MP 百萬結構訓練 |
| B-05 | mir-group/allegro | 約 300 | A | NequIP 大規模擴展，局部等變表示 |
| B-06 | mir-group/flare | 約 300 | A | [關鍵] 線上主動學習 + 貝葉斯不確定性 MD |
| B-07 | deepmodeling/deepmd-kit | 約 1,500 | B | 經典 MLIP 框架，v3 多後端 |
| B-08 | FAIR-Chem/fairchem | 約 800 | A | Meta FAIR 化學統一平台，工程架構參考 |
| B-09 | atomicarchitects/equiformer | 約 200 | B | Transformer + 等變，OC20 最優 |
| B-10 | microsoft/mattersim | 約 200 | B | 微軟大規模預訓練材料模型 |
| B-11 | torchmd/torchmd-net | 約 300 | B | 笛卡爾張量訊息傳遞，避免球諧複雜性 |
| B-12 | Open-Catalyst-Project/ocp | 約 800 | B | 1.3 億 DFT 計算，大規模訓練方法 |
| B-13 | ACEsuit/mace-foundations | 約 50 | A | MACE 基礎模型權重，遷移學習 |
| B-14 | ACEsuit/mace-tutorials | 約 30 | B | MACE 官方教學 |
| B-15 | ACEsuit/mace-off | 約 50 | B | 有機分子力場，跨域遷移範例 |

---

## 第三類：不確定性量化 UQ（15 個）

### C-01 | aamini/evidential-deep-learning | 508 星

- **機構**：MIT（Alexander Amini）
- **等級**：A
- **論文**：Deep Evidential Regression. *NeurIPS*, 2020. 約 700 次引用。
- **描述**：單次前向傳播同時輸出預測值和不確定性估計。透過 Normal-Inverse-Gamma (NIG) 先驗輸出四個參數（mu, nu, alpha, beta），分離隨機不確定性和認知不確定性。
- **與 ATLAS 關聯**：[關鍵] `atlas/models/uq/EvidentialRegression` 的直接來源。核心 UQ 方法。
- **核心學習點**：(1) NIG 損失函數設計。(2) 四參數的物理解讀。(3) 隨機/認知不確定性分離。
- **閱讀狀態**：未開始

### C-02 | google/uncertainty-baselines | 約 1,000 星

- **機構**：Google
- **等級**：A
- **描述**：Google 官方 UQ 方法比較基準。包含 Ensemble、MC Dropout、Heteroscedastic、SNGP 等方法。
- **與 ATLAS 關聯**：UQ 方法比較框架參考。
- **閱讀狀態**：未開始

### C-03 | cornellius-gp/gpytorch | 3,800 星

- **機構**：Cornell / CMU
- **等級**：A
- **論文**：GPyTorch. *NeurIPS*, 2018. 約 2000 次引用。
- **描述**：高效能高斯過程框架，支援精確和近似推斷。
- **與 ATLAS 關聯**：ATLAS 主動學習中代理模型的依賴。
- **閱讀狀態**：未開始

### C-04 | pytorch/botorch | 約 3,200 星

- **機構**：Meta
- **等級**：A
- **論文**：BoTorch. *NeurIPS*, 2020. 約 2000 次引用。
- **描述**：基於 GPyTorch 構建的貝葉斯最佳化框架。支援多目標、多保真度、批量最佳化。
- **與 ATLAS 關聯**：ATLAS 中採集函數計算的依賴。
- **閱讀狀態**：未開始

### C-05 | uncertainty-toolbox/uncertainty-toolbox | 2,000 星

- **機構**：Stanford
- **等級**：A
- **描述**：綜合 UQ 評估與校準工具包。支援校準圖、銳度指標、誤校準面積等標準 UQ 品質度量。
- **與 ATLAS 關聯**：[關鍵] 產生論文品質校準圖和 UQ 評估指標的工具。
- **核心學習點**：(1) 標準 UQ 校準指標定義。(2) 如何產出可發表的校準圖。
- **閱讀狀態**：未開始

### C-06 | chemprop/chemprop | 128 星 (v2)

- **機構**：MIT（Kevin Yang, Rafael Gomez-Bombarelli）
- **等級**：A
- **論文**：Chemprop. *JCIM*, 2019. 約 1500 次引用。
- **描述**：有向訊息傳遞神經網路，內建集成 UQ 和校準工具。GNN + UQ 整合的最佳範例（分子領域）。
- **與 ATLAS 關聯**：[關鍵] 在 GNN 訓練管線中整合 UQ 的參考實作。
- **閱讀狀態**：未開始

### C-07 | learningmatter-mit/matex | 約 30 星

- **機構**：MIT（Rafael Gomez-Bombarelli）
- **等級**：A
- **論文**：Known Unknowns: Out-of-Distribution Property Prediction in Materials and Molecules.
- **描述**：研究 ML 模型在材料和分子外推時的失效模式，提供偵測分佈外輸入的方法。
- **與 ATLAS 關聯**：[關鍵] 直接對應 ATLAS 的 OOD 偵測目標。
- **核心學習點**：(1) 如何在材料情境中定義 OOD。(2) 外推邊界的確定方法。(3) OOD 偵測指標。
- **閱讀狀態**：未開始

### C-08 至 C-15（簡要列表）

| 編號 | 倉庫 | 等級 | 核心價值 |
|------|------|------|----------|
| C-08 | tirtha-v/DPOSE-GNN | A | [關鍵] 淺層集成 GNN UQ，QM9/OC20 上分離域內/域外 |
| C-09 | AutoGNNUQ | A | [關鍵] 神經架構搜尋自動構建 UQ 集成 |
| C-10 | snap-stanford/conformalized-gnn | A | [關鍵] 合形預測 + GNN，無分佈假設 UQ，保證覆蓋率 |
| C-11 | peiyaoli/Conformal-ADMET-Prediction | B | GNN + 分位數迴歸 + 合形預測 |
| C-12 | SeongokRyu/uq_molecule | B | MC-Dropout + 增強圖卷積的貝葉斯 UQ |
| C-13 | ljatynu/CPBayesMPP | B | 對比學習先驗 + BNN，含校準曲線 |
| C-14 | y0ast/DUQ | B | 確定性 UQ，無需多次前向傳播 |
| C-15 | clqz-trustml/conformal-prediction | B | 合形預測論文系統索引 |

---

## 第四類：材料資料庫與工具（10 個）

| 編號 | 倉庫 | 星數 | 等級 | 核心價值 |
|------|------|------|------|----------|
| D-01 | materialsproject/pymatgen | 約 2,000 | A | ATLAS 直接依賴。晶體結構分析、對稱性、相圖。約 5000 次引用 |
| D-02 | hackingmaterials/matminer | 約 500 | A | 100+ 種描述子的特徵工程工具。RF 基線使用 |
| D-03 | usnistgov/jarvis | 約 300 | A | ATLAS 資料上游。76K 材料 + ML 訓練工具 |
| D-04 | materialsproject/matbench | 約 100 | A | 13 個標準 benchmark 任務。ATLAS MatbenchRunner 依賴 |
| D-05 | SINGROUP/dscribe | 約 400 | B | SOAP/ACSF/Coulomb Matrix 原子描述子庫 |
| D-06 | hackingmaterials/automatminer | 約 130 | B | 自動 ML 材料管線 |
| D-07 | materialsproject/mp-api | 約 300 | B | Materials Project API。跨資料庫 OOD 實驗必需 |
| D-08 | WMD-group/SMACT | 約 100 | B | 化學組成可行性過濾 |
| D-09 | hackingmaterials/atomate | 約 400 | A | DFT 計算工作流自動化 |
| D-10 | sedaoturak/data-resources | — | B | 公開材料資料庫彙整 |

---

## 第五類：主動學習與貝葉斯最佳化（10 個）

| 編號 | 倉庫 | 星數 | 等級 | 核心價值 |
|------|------|------|------|----------|
| E-01 | emdgroup/baybe | 約 200 | A | [關鍵] 材料/化學專用 BO。減少 50%+ 實驗次數 |
| E-02 | deepmodeling/dpgen | 約 300 | A | AL 閉環：選結構 - DFT - 重訓練 - 循環 |
| E-03 | facebook/Ax | 約 2,400 | B | 通用 BO 平台（BoTorch 上層封裝） |
| E-04 | emukit/emukit | 約 400 | B | UQ + BO + AL 整合工具包 |
| E-05 | modAL-python/modAL | 約 2,200 | B | 模組化 AL 框架 |
| E-06 | scikit-activeml | 約 300 | B | modAL 的現代替代品 |
| E-07 | HIPS/dragonfly | 約 700 | B | 可擴展 BO，高維 + 多保真度 |
| E-08 | hackingmaterials/rocketsled | 約 50 | B | 材料工作流中嵌入 BO |
| E-09 | materialsproject/atomate2 | 約 200 | B | 新一代 DFT 工作流框架 |
| E-10 | mir-group/FLARE-Tutorials | 約 30 | B | FLARE 線上 AL 教學 |

---

## 第六類：可解釋性與魯棒性（8 個）

| 編號 | 倉庫 | 星數 | 等級 | 核心價值 |
|------|------|------|------|----------|
| F-01 | pyg-team/pytorch_geometric | 約 21,000 | A | ATLAS GNN 骨幹框架。內建 GNNExplainer |
| F-02 | pytorch/captum | 約 1,500 | B | PyTorch 模型歸因分析 |
| F-03 | shap/shap | 約 23,000 | B | SHAP 特徵重要性分析。RF 基線解釋用 |
| F-04 | divelab/DIG | 約 400 | B | GNN 解釋基準評估框架 |
| F-05 | learningmatter-mit/Adversarial-Attacks | 約 20 | A | [關鍵] 原子系統對抗攻擊，直接關聯 OOD 魯棒性 |
| F-06 | interpretml/interpret | 約 6,000 | B | 可解釋增強機器、玻璃盒模型 |
| F-07 | learningmatter-mit/NeuralForceField | 約 50 | B | 教學品質的 PyTorch 神經力場 |
| F-08 | learningmatter-mit/LLM4BO | 約 30 | B | LLM 輔助貝葉斯最佳化基準 |

---

## 第七類：基準與資源索引（10 個）

| 編號 | 倉庫 | 星數 | 等級 | 核心價值 |
|------|------|------|------|----------|
| G-01 | WanyuGroup/AI-for-Crystal-Materials | 約 200 | A | 頂級會議晶體 ML 論文合集，定期更新 |
| G-02 | kdmsit/Awesome-Crystal-GNNs | 約 50 | B | 2024-2025 Crystal GNN 論文集 |
| G-03 | JuDFTteam/best-of-atomistic-ML | 約 800 | A | 510+ 排名 AML 專案，確認無遺漏 |
| G-04 | tilde-lab/awesome-materials-informatics | 約 500 | B | 材料資訊學策展清單 |
| G-05 | modl-uclouvain/modnet-matbench | 約 20 | B | MODNet Matbench 結果，直接數字對照 |
| G-06 | jakobzeitler/bayesian-deep-learning | 約 300 | B | BDL 論文索引 |
| G-07 | deepmodeling/AIS-Square | 約 50 | B | AI for Science 模型共享平台 |
| G-08 | vertaix/LLM4Mat-Bench | 約 50 | B | LLM 材料性質預測基準 |
| G-09 | vertaix/LLM-Prop | 約 30 | B | 微調 T5 編碼器預測晶體性質 |
| G-10 | UQ4DD | — | B | Platt/Venn-ABERS 校準方法，可遷移至材料領域 |

---

## 第八類：軟體工程與 MLOps（7 個）

| 編號 | 倉庫 | 星數 | 等級 | 核心價值 |
|------|------|------|------|----------|
| H-01 | wandb/wandb | 約 9,000 | B | 實驗追蹤與視覺化 |
| H-02 | mlflow/mlflow | 約 19,000 | B | 模型版本控制與管理 |
| H-03 | iterative/dvc | 約 14,000 | B | 大資料版本控制 |
| H-04 | facebookresearch/hydra | 約 8,500 | B | YAML 配置組合框架 |
| H-05 | Lightning-AI/pytorch-lightning | 約 28,000 | B | 結構化訓練框架。MatGL 使用 |
| H-06 | pyg-team/pytorch_geometric | 約 21,000 | A | 交叉引用 F-01。ATLAS GNN 骨幹 |
| H-07 | mir-group/nequip-tutorial | 約 20 | B | NequIP 教學，等變勢能訓練入門 |

---

## 總結統計

| 類別 | 條目數 | A 級 | B 級 |
|------|--------|------|------|
| 1. 晶體 GNN | 15 | 5 | 10 |
| 2. 等變 / MLIP | 15 | 8 | 7 |
| 3. 不確定性量化 | 15 | 10 | 5 |
| 4. 材料工具 | 10 | 5 | 5 |
| 5. 主動學習 / BO | 10 | 2 | 8 |
| 6. 可解釋性 / 魯棒性 | 8 | 2 | 6 |
| 7. 基準 / 索引 | 10 | 2 | 8 |
| 8. 軟體工程 | 7 | 1 | 6 |
| **合計** | **90** | **35** | **55** |

所有條目均為 A 或 B 級。無填充條目。

### 優先閱讀順序（前 10 名）

| 優先級 | 編號 | 倉庫 | 理由 |
|--------|------|------|------|
| 1 | C-01 | aamini/evidential-deep-learning | ATLAS UQ 核心來源 |
| 2 | C-07 | learningmatter-mit/matex | ATLAS OOD 核心參考 |
| 3 | B-06 | mir-group/flare | UQ + 主動學習閉環 |
| 4 | C-10 | snap-stanford/conformalized-gnn | 無分佈假設 UQ，統計保證 |
| 5 | C-06 | chemprop/chemprop | GNN + UQ 整合最佳實踐 |
| 6 | A-01 | txie-93/cgcnn | ATLAS 基線架構 |
| 7 | B-01 | mir-group/nequip | ATLAS 等變模型來源 |
| 8 | C-08 | tirtha-v/DPOSE-GNN | 直接的 UQ + GNN 競爭者 |
| 9 | C-09 | AutoGNNUQ | 自動化 UQ 架構搜尋 |
| 10 | A-06 | modl-uclouvain/modnet | Matbench 主要競爭者 |

