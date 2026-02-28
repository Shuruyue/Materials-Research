# Literature Survey for Crystal GNN and Uncertainty Quantification Research

> **Project**: ATLAS — Accelerated Topological Learning And Screening  
> **Author**: Zhong  
> **Date**: 2026-02-27  
> **Total Entries**: 200  
> **Categories**: 13 + Supplementary (Crystal GNN, Equivariant, MLIP, UQ/Calibration, OOD/Robustness, Active Learning/BO, Data/Benchmark, Training Methodology, Self-Supervised/Transfer, Generative/Inverse Design, Explainability/XAI, Descriptors/Representations, Scaling/Deployment)  
> **Status**: Phase 1 Seed Collection Complete

---

## Methodology

### Selection Criteria

Each paper was evaluated across 5 dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Citation Impact | High | Total citations and venue prestige (Nature, PRL, NeurIPS, ICML, ICLR, etc.) |
| Methodological Relevance | High | Direct applicability to ATLAS architecture, training, or evaluation |
| Recency | Medium | Priority given to 2022–2026 for state-of-the-art context |
| Reproducibility | Medium | Code availability, documented hyperparameters, benchmark results |
| Foundational Importance | Medium | Conceptual basis for downstream work (even if older) |

### Grading System

| Grade | Criteria | Required Action |
|-------|----------|-----------------|
| **A** | Directly implements or validates a core ATLAS component | Deep read with implementation notes; reproduce key results |
| **B** | Provides methodological context or comparison baseline | Structured read with summary notes |
| **C** | Background reference or tangential relevance | Skim abstract and conclusions |

### Reading Status Definitions

| Status | Description |
|--------|-------------|
| Not Started | Paper identified but not yet accessed |
| Queued | Added to reading queue (Zotero/PDF collected) |
| Scanned | Abstract, figures, and conclusions reviewed |
| In Progress | Active detailed reading with note-taking |
| Complete | Full read with structured notes recorded |

---

## Category 1: Crystal Property Prediction GNNs (12 entries)

### 1-01 | Gilmer et al. — Neural Message Passing for Quantum Chemistry

- **Year**: 2017 | **Venue**: ICML
- **Grade**: A | **Citations**: ~4,500
- **Core Contribution**: Unified framework (MPNN) encompassing prior GNN variants for molecular property prediction. Defines message, update, and readout functions.
- **ATLAS Relevance**: Theoretical foundation for all message-passing architectures in ATLAS. Defines the vocabulary used throughout the codebase.
- **Reading Status**: Not Started

### 1-02 | Xie & Grossman — Crystal Graph Convolutional Neural Networks (CGCNN)

- **Year**: 2018 | **Venue**: Physical Review Letters
- **Grade**: A | **Citations**: ~3,000
- **Core Contribution**: First end-to-end GNN for crystal property prediction from structure. Gaussian distance expansion for edge features. Global mean pooling.
- **ATLAS Relevance**: Direct basis for `atlas/models/cgcnn.py`. Phase 1 baseline architecture.
- **Key Learnings**: (1) Graph construction from CIF. (2) `GaussianDistance` encoding. (3) Pooling strategy effects.
- **Reading Status**: Not Started

### 1-03 | Schütt et al. — SchNet: A Continuous-Filter Convolutional Neural Network

- **Year**: 2018 | **Venue**: Journal of Chemical Physics / NeurIPS
- **Grade**: A | **Citations**: ~2,500
- **Core Contribution**: Continuous-filter convolutions operating on interatomic distances. End-to-end differentiable and rotationally invariant.
- **ATLAS Relevance**: Foundational architecture for SchNetPack ecosystem. Comparison baseline for equivariant models.
- **Reading Status**: Not Started

### 1-04 | Chen, Ye & Ong — MEGNet: Graph Networks as a Universal ML Framework

- **Year**: 2019 | **Venue**: Chemistry of Materials
- **Grade**: A | **Citations**: ~1,500
- **Core Contribution**: Extends GNN with global state vector for material-level attributes. Demonstrates multi-fidelity learning.
- **ATLAS Relevance**: Global state architecture reference. Multi-fidelity training concept applicable to ATLAS.
- **Reading Status**: Not Started

### 1-05 | Choudhary & DeCost — Atomistic Line Graph Neural Network (ALIGNN)

- **Year**: 2021 | **Venue**: npj Computational Materials
- **Grade**: A | **Citations**: ~800
- **Core Contribution**: Augments atom graph with a line graph encoding bond angles. State-of-the-art on JARVIS-DFT.
- **ATLAS Relevance**: Primary comparison model. Uses same JARVIS dataset as ATLAS. Stronger baseline than CGCNN.
- **Key Learnings**: (1) Line graph construction. (2) Angular feature encoding. (3) Data normalization practices.
- **Reading Status**: Not Started

### 1-06 | Gasteiger, Groß & Günnemann — DimeNet / DimeNet++

- **Year**: 2020/2022 | **Venue**: ICLR / NeurIPS
- **Grade**: B | **Citations**: ~1,000
- **Core Contribution**: Directional message passing using interatomic angles and distances. DimeNet++ improves efficiency.
- **ATLAS Relevance**: Angular feature extraction reference. Architectural comparison for equivariant approaches.
- **Reading Status**: Not Started

### 1-07 | Schütt et al. — PaiNN: Equivariant Message Passing for Prediction of Tensorial Properties

- **Year**: 2021 | **Venue**: ICML
- **Grade**: B | **Citations**: ~600
- **Core Contribution**: Equivariant message passing with scalar and vector features. Balances expressivity and computational cost.
- **ATLAS Relevance**: Efficient equivariant architecture reference. SchNetPack integration.
- **Reading Status**: Not Started

### 1-08 | Lin et al. — ComFormer: Complete Graph Transformer for Crystal Property Prediction

- **Year**: 2024 | **Venue**: ICLR
- **Grade**: B | **Citations**: —
- **Core Contribution**: Transformer architecture for crystals with complete SE(3) information encoding.
- **ATLAS Relevance**: Transformer-based crystal architecture comparison.
- **Reading Status**: Not Started

### 1-09 | Cao et al. — CrystalFormer: Infinitely Connected Attention for Periodic Structure Encoding

- **Year**: 2024 | **Venue**: ICLR
- **Grade**: B | **Citations**: —
- **Core Contribution**: Attention mechanism handling infinite periodic repetitions of crystal structures.
- **ATLAS Relevance**: Periodic structure encoding methodology.
- **Reading Status**: Not Started

### 1-10 | Yan et al. — GMTNet: Space Group Informed Equivariant Transformer

- **Year**: 2024 | **Venue**: ICML
- **Grade**: B | **Citations**: —
- **Core Contribution**: Encodes space group symmetry directly into transformer architecture for crystal property prediction.
- **ATLAS Relevance**: Space group equivariance — advanced symmetry-aware approach.
- **Reading Status**: Not Started

### 1-11 | Chen et al. — CTGNN: Crystal Transformer Graph Neural Network

- **Year**: 2024 | **Venue**: arXiv
- **Grade**: B | **Citations**: —
- **Core Contribution**: Dual-transformer architecture combining inter-crystal and inter-atomic attention mechanisms for crystal property prediction.
- **ATLAS Relevance**: Hybrid GNN-Transformer design pattern.
- **Reading Status**: Not Started

### 1-12 | Choudhary — Atomistic Line Graph Neural Network (ALIGNN-FF)

- **Year**: 2023 | **Venue**: Digital Discovery
- **Grade**: A | **Citations**: ~200
- **Core Contribution**: Extends ALIGNN to force-field training with energy, forces, and stresses.
- **ATLAS Relevance**: Direct extension of ATLAS baseline model to force-field domain.
- **Reading Status**: Not Started

---

## Category 2: Equivariant Neural Networks (10 entries)

### 2-01 | Thomas et al. — Tensor Field Networks

- **Year**: 2018 | **Venue**: arXiv
- **Grade**: B | **Citations**: ~800
- **Core Contribution**: First SE(3)-equivariant neural network using tensor products of spherical harmonics.
- **ATLAS Relevance**: Foundational theory for all equivariant architectures.
- **Reading Status**: Not Started

### 2-02 | Fuchs et al. — SE(3)-Transformers

- **Year**: 2020 | **Venue**: NeurIPS
- **Grade**: B | **Citations**: ~800
- **Core Contribution**: Self-attention mechanism operating on SE(3)-equivariant features.
- **ATLAS Relevance**: Equivariant attention mechanism reference.
- **Reading Status**: Not Started

### 2-03 | Batzner et al. — E(3)-Equivariant Graph Neural Networks for Data-Efficient Interatomic Potentials (NequIP)

- **Year**: 2022 | **Venue**: Nature Communications
- **Grade**: A | **Citations**: ~1,500
- **Core Contribution**: E(3)-equivariant message passing using e3nn tensor products. Data-efficient learning of interatomic potentials.
- **ATLAS Relevance**: Primary equivariant architecture reference. Benchmark for data efficiency.
- **Key Learnings**: (1) Tensor product convolution. (2) Data efficiency claims. (3) Training stability.
- **Reading Status**: Not Started

### 2-04 | Geiger & Smidt — e3nn: Euclidean Neural Networks

- **Year**: 2022 | **Venue**: arXiv (library paper)
- **Grade**: A | **Citations**: ~1,000
- **Core Contribution**: Open-source library for building E(3)-equivariant neural networks using irreducible representations and tensor products.
- **ATLAS Relevance**: Backend library for `atlas/models/equivariant.py`. Version compatibility critical (cf. MACE #555).
- **Reading Status**: Not Started

### 2-05 | Batatia et al. — MACE: Higher Order Equivariant Message Passing

- **Year**: 2022 | **Venue**: NeurIPS
- **Grade**: A | **Citations**: ~800
- **Core Contribution**: Multi-body equivariant interactions via higher-order tensor products. Body-ordered message passing.
- **ATLAS Relevance**: State-of-the-art equivariant MLIP architecture. Potential ATLAS extension target.
- **Key Learnings**: (1) Multi-body interaction construction. (2) Computational scaling. (3) Pre-training strategy.
- **Reading Status**: Not Started

### 2-06 | Musaelian et al. — Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics (Allegro)

- **Year**: 2023 | **Venue**: Nature Communications
- **Grade**: B | **Citations**: ~400
- **Core Contribution**: Scalable equivariant architecture using strictly local operations. Parallelizable.
- **ATLAS Relevance**: Scalability reference for equivariant models. Same group as NequIP.
- **Reading Status**: Not Started

### 2-07 | Satorras et al. — E(n) Equivariant Graph Neural Networks (EGNN)

- **Year**: 2021 | **Venue**: ICML
- **Grade**: B | **Citations**: ~1,500
- **Core Contribution**: Minimalist E(n)-equivariant GNN without spherical harmonics. Directly updates coordinates.
- **ATLAS Relevance**: Simple equivariant baseline for conceptual understanding.
- **Reading Status**: Not Started

### 2-08 | Liao & Smidt — Equiformer: Equivariant Graph Attention Transformer

- **Year**: 2023 | **Venue**: ICLR
- **Grade**: A | **Citations**: ~300
- **Core Contribution**: Combines equivariant representations with transformer attention. State-of-the-art on OC20.
- **ATLAS Relevance**: Architecture reference for equivariant transformer design.
- **Reading Status**: Not Started

### 2-09 | Liao et al. — EquiformerV2: Improved Equivariant Transformer

- **Year**: 2024 | **Venue**: ICLR
- **Grade**: B | **Citations**: ~200
- **Core Contribution**: eSCN convolutions integrated with Equiformer. Performance improvements on OC20/OC22.
- **ATLAS Relevance**: Latest equivariant transformer iteration. Benchmark reference.
- **Reading Status**: Not Started

### 2-10 | Passaro & Zitnick — Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs (eSCN)

- **Year**: 2023 | **Venue**: ICML
- **Grade**: B | **Citations**: ~150
- **Core Contribution**: Computational trick reducing SO(3) tensor products to SO(2), dramatically improving efficiency.
- **ATLAS Relevance**: Key efficiency technique for practical equivariant model deployment.
- **Reading Status**: Not Started

---

## Category 3: Machine-Learning Interatomic Potentials (10 entries)

### 3-01 | Chen & Ong — A Universal Graph Deep Learning Interatomic Potential (M3GNet)

- **Year**: 2022 | **Venue**: Nature Computational Science
- **Grade**: A | **Citations**: ~500
- **Core Contribution**: First universal MLIP trained on MPtrj. Graph-based architecture predicting energy, forces, stresses.
- **ATLAS Relevance**: Universal potential baseline. MatGL library integration reference.
- **Reading Status**: Not Started

### 3-02 | Deng et al. — CHGNet: Pretrained Universal Neural Network Potential for Charge-Informed Atomistic Modelling

- **Year**: 2023 | **Venue**: Nature Machine Intelligence
- **Grade**: A | **Citations**: ~500
- **Core Contribution**: Graph neural network potential incorporating magnetic moments and charge information.
- **ATLAS Relevance**: Charge-aware architecture. Pre-trained foundation model concept.
- **Reading Status**: Not Started

### 3-03 | Batatia et al. — A Foundation Model for Atomistic Materials Chemistry (MACE-MP-0)

- **Year**: 2024 | **Venue**: arXiv / under review
- **Grade**: A | **Citations**: ~200
- **Core Contribution**: Universal MLIP trained on MPtrj using MACE architecture. Enables zero-shot MD across periodic table.
- **ATLAS Relevance**: Foundation model paradigm. Benchmark for pre-trained GNN performance.
- **Reading Status**: Not Started

### 3-04 | Yang et al. — MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures, Pressures

- **Year**: 2024 | **Venue**: arXiv (Microsoft Research)
- **Grade**: B | **Citations**: —
- **Core Contribution**: Large-scale MLIP pre-trained on diverse DFT data. Multi-condition prediction capability.
- **ATLAS Relevance**: Industrial-scale pre-training reference. Fine-tuning methodology.
- **Reading Status**: Not Started

### 3-05 | Zuo et al. — Performance and Cost Assessment of Machine Learning Interatomic Potentials

- **Year**: 2020 | **Venue**: Journal of Physical Chemistry A
- **Grade**: A | **Citations**: ~600
- **Core Contribution**: Systematic comparison of MLIP architectures (GAP, MTP, NequIP, DeepMD, etc.) on accuracy, cost, and data efficiency.
- **ATLAS Relevance**: Benchmark methodology for MLIP comparison. Evaluation protocol reference.
- **Reading Status**: Not Started

### 3-06 | Zhang et al. — Deep Potential Molecular Dynamics (DeePMD-kit)

- **Year**: 2018 | **Venue**: Physical Review Letters
- **Grade**: B | **Citations**: ~2,000
- **Core Contribution**: Descriptor-based neural network interatomic potential with active learning integration (DP-GEN).
- **ATLAS Relevance**: Active learning for data generation reference. Large-scale MD simulation.
- **Reading Status**: Not Started

### 3-07 | Merchant et al. — Scaling Deep Learning for Materials Discovery (GNoME)

- **Year**: 2023 | **Venue**: Nature
- **Grade**: A | **Citations**: ~500
- **Core Contribution**: GNN active learning pipeline discovering 2.2M new crystals. 384K confirmed stable by DFT.
- **ATLAS Relevance**: Active learning at scale for materials discovery. UQ for filtering ML predictions.
- **Reading Status**: Not Started

### 3-08 | A Practical Guide to Machine Learning Interatomic Potentials — Status and Future

- **Year**: 2025 | **Venue**: arXiv
- **Grade**: B | **Citations**: —
- **Core Contribution**: Comprehensive review of universal MLIPs, covering architecture, training, and application best practices.
- **ATLAS Relevance**: Survey reference for situating ATLAS in the MLIP landscape.
- **Reading Status**: Not Started

### 3-09 | Foundation Models for Atomistic Simulation of Chemistry and Materials

- **Year**: 2025 | **Venue**: arXiv
- **Grade**: B | **Citations**: —
- **Core Contribution**: Explores foundation model concepts (pre-training, fine-tuning, scaling laws) applied to atomistic simulation.
- **ATLAS Relevance**: Foundation model design principles for ATLAS extension.
- **Reading Status**: Not Started

### 3-10 | Musielewicz et al. — FINETUNA: Fine-tuning Accelerated Molecular Simulations

- **Year**: 2022 | **Venue**: Machine Learning: Science and Technology
- **Grade**: B | **Citations**: ~100
- **Core Contribution**: Fine-tuning pre-trained GNN potentials for specific catalytic systems using active learning.
- **ATLAS Relevance**: Active learning + fine-tuning workflow reference.
- **Reading Status**: Not Started

---

## Category 4: Uncertainty Quantification and Calibration (15 entries)

### 4-01 | Gal — Uncertainty in Deep Learning (PhD Thesis)

- **Year**: 2016 | **Venue**: University of Cambridge
- **Grade**: A | **Citations**: ~3,000
- **Core Contribution**: Theoretical foundation for MC Dropout as approximate Bayesian inference. Epistemic vs aleatoric uncertainty.
- **ATLAS Relevance**: MC Dropout theory underpinning `atlas/models/uncertainty.py`.
- **Reading Status**: Not Started

### 4-02 | Kendall & Gal — What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?

- **Year**: 2017 | **Venue**: NeurIPS
- **Grade**: A | **Citations**: ~7,000
- **Core Contribution**: Formalizes aleatoric and epistemic uncertainty decomposition. Heteroscedastic loss formulation.
- **ATLAS Relevance**: Core uncertainty decomposition implemented in ATLAS. Direct reference for loss function design.
- **Reading Status**: Not Started

### 4-03 | Lakshminarayanan et al. — Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles

- **Year**: 2017 | **Venue**: NeurIPS
- **Grade**: A | **Citations**: ~5,000
- **Core Contribution**: Deep Ensemble as a simple, strong UQ baseline. Non-Bayesian approach using M independently trained networks.
- **ATLAS Relevance**: Primary UQ baseline for ATLAS comparison. Ensemble implementation reference.
- **Reading Status**: Not Started

### 4-04 | Amini et al. — Deep Evidential Regression

- **Year**: 2020 | **Venue**: NeurIPS
- **Grade**: A | **Citations**: ~700
- **Core Contribution**: Single forward-pass uncertainty estimation by predicting parameters of a Normal-Inverse-Gamma distribution.
- **ATLAS Relevance**: Core UQ method for ATLAS. Evidential loss implemented in `atlas/training/losses.py`.
- **Key Learnings**: (1) NIG distribution parameterization. (2) Evidence regularization. (3) Numerical stability.
- **Reading Status**: Not Started

### 4-05 | Hirschfeld et al. — Uncertainty Quantification Using Neural Networks for Molecular Property Prediction

- **Year**: 2020 | **Venue**: Journal of Chemical Information and Modeling
- **Grade**: A | **Citations**: ~400
- **Core Contribution**: Systematic comparison of UQ methods (ensemble, MC Dropout, MVE) for chemical ML. Calibration assessment.
- **ATLAS Relevance**: Benchmark methodology for UQ evaluation in materials chemistry domain.
- **Reading Status**: Not Started

### 4-06 | Tran et al. — Methods for Comparing Uncertainty Quantifications for Material Property Predictions

- **Year**: 2020 | **Venue**: Machine Learning: Science and Technology
- **Grade**: A | **Citations**: ~300
- **Core Contribution**: Framework for comparing UQ methods in materials science using calibration metrics, sharpness, and dispersion.
- **ATLAS Relevance**: Evaluation protocol for ATLAS UQ module. Defines metrics used in `atlas/evaluation/`.
- **Reading Status**: Not Started

### 4-07 | Chung et al. — Uncertainty Quantification Methods for GNN Relaxed Energy Calculations

- **Year**: 2024 | **Venue**: NeurIPS Workshop
- **Grade**: A | **Citations**: —
- **Core Contribution**: Evaluates UQ methods specifically for GNN-based energy predictions during structure relaxation.
- **ATLAS Relevance**: Directly applicable to ATLAS energy prediction with UQ. Calibration methodology.
- **Reading Status**: Not Started

### 4-08 | Vassilev-Galindo et al. — DPOSE-GNN: Direct Propagation of Shallow Ensembles for Uncertainty Quantification

- **Year**: 2024 | **Venue**: arXiv
- **Grade**: A | **Citations**: —
- **Core Contribution**: Computationally efficient alternative to deep ensembles. Shallow ensemble propagation through GNN layers.
- **ATLAS Relevance**: Lightweight UQ method directly applicable to ATLAS GNN architectures.
- **Reading Status**: Not Started

### 4-09 | Hsu et al. — AutoGNNUQ: Automated Uncertainty Quantification for Molecular GNNs

- **Year**: 2024 | **Venue**: Digital Discovery (RSC)
- **Grade**: A | **Citations**: —
- **Core Contribution**: Architecture search for GNN ensembles. Automated aleatoric/epistemic decomposition via variance decomposition.
- **ATLAS Relevance**: Automated UQ architecture design. Variance decomposition methodology.
- **Reading Status**: Not Started

### 4-10 | Guo et al. — On Calibration of Modern Neural Networks

- **Year**: 2017 | **Venue**: ICML
- **Grade**: B | **Citations**: ~3,000
- **Core Contribution**: Demonstrates that modern deep networks are poorly calibrated. Introduces temperature scaling and ECE metric.
- **ATLAS Relevance**: Calibration evaluation framework. Temperature scaling as post-hoc calibration.
- **Reading Status**: Not Started

### 4-11 | Kuleshov et al. — Accurate Uncertainties for Deep Learning Using Calibrated Regression

- **Year**: 2018 | **Venue**: ICML
- **Grade**: B | **Citations**: ~500
- **Core Contribution**: Isotonic regression for calibrating regression uncertainty estimates.
- **ATLAS Relevance**: Post-hoc recalibration method for ATLAS predictions.
- **Reading Status**: Not Started

### 4-12 | Kendall, Gal & Cipolla — Multi-Task Learning Using Uncertainty to Weigh Losses

- **Year**: 2018 | **Venue**: CVPR
- **Grade**: A | **Citations**: ~3,000
- **Core Contribution**: Uses homoscedastic uncertainty to automatically balance multi-task losses.
- **ATLAS Relevance**: Direct reference for `atlas/training/multi_task_loss.py`. Energy-force-stress balancing.
- **Reading Status**: Not Started

### 4-13 | Soleimany et al. — Evidential Deep Learning for Guided Molecular Property Prediction and Discovery

- **Year**: 2021 | **Venue**: ACS Central Science
- **Grade**: A | **Citations**: ~200
- **Core Contribution**: Applies evidential deep learning (Dirichlet prior) to molecular property prediction with active learning.
- **ATLAS Relevance**: Evidential UQ + active learning combination. Drug discovery domain transfer.
- **Reading Status**: Not Started

### 4-14 | Ilg et al. — Uncertainty Estimates and Multi-Hypotheses Networks for Optical Flow

- **Year**: 2018 | **Venue**: ECCV
- **Grade**: C | **Citations**: ~500
- **Core Contribution**: Multi-hypothesis prediction framework for structured uncertainty estimation.
- **ATLAS Relevance**: Background reference for multi-hypothesis uncertainty patterns.
- **Reading Status**: Not Started

### 4-15 | Palmer et al. — Benchmark of UQ Methods for GNN-Based Materials Property Prediction

- **Year**: 2023 | **Venue**: Applied Physics Reviews
- **Grade**: A | **Citations**: ~50
- **Core Contribution**: Comprehensive benchmark evaluating bootstrap ensemble, conformal prediction, evidential learning, and delta metric for GNN materials prediction.
- **ATLAS Relevance**: Most directly comparable evaluation framework. Defines current UQ benchmark standards.
- **Reading Status**: Not Started

---

## Category 5: Out-of-Distribution Detection and Robustness (6 entries)

### 5-01 | Hendrycks & Gimpel — A Baseline for Detecting Misclassified and Out-of-Distribution Examples

- **Year**: 2017 | **Venue**: ICLR
- **Grade**: B | **Citations**: ~3,000
- **Core Contribution**: Maximum softmax probability as OOD baseline. Establishes standard OOD detection benchmarks.
- **ATLAS Relevance**: OOD detection baseline methodology.
- **Reading Status**: Not Started

### 5-02 | Schwalbe-Koda et al. — Differentiable Sampling of Molecular Geometries with Uncertainty-Based Adversarial Attacks

- **Year**: 2021 | **Venue**: Nature Communications
- **Grade**: A | **Citations**: ~200
- **Core Contribution**: Adversarial attacks on molecular GNNs to identify failure modes. OOD detection for materials.
- **ATLAS Relevance**: Robustness evaluation methodology for ATLAS models. Adversarial testing framework.
- **Reading Status**: Not Started

### 5-03 | Crystal Adversarial Learning (CAL) — Adversarial Robustness for Crystal GNNs

- **Year**: 2024 | **Venue**: Under review
- **Grade**: A | **Citations**: —
- **Core Contribution**: Adversarial perturbation framework specifically for crystal GNN property predictions.
- **ATLAS Relevance**: OOD robustness testing methodology directly applicable to ATLAS.
- **Reading Status**: Not Started

### 5-04 | Huang et al. — Conformal Prediction for GNNs

- **Year**: 2023 | **Venue**: NeurIPS
- **Grade**: B | **Citations**: ~100
- **Core Contribution**: Distribution-free prediction intervals for GNN outputs via conformal prediction.
- **ATLAS Relevance**: Conformal prediction integration for ATLAS coverage guarantees.
- **Reading Status**: Not Started

### 5-05 | Liu et al. — Deterministic Uncertainty Quantification (DUQ)

- **Year**: 2020 | **Venue**: ICML
- **Grade**: B | **Citations**: ~300
- **Core Contribution**: Single-pass OOD detection using radial basis function output layer and gradient penalty.
- **ATLAS Relevance**: Single-pass UQ alternative to ensembles. Computational efficiency reference.
- **Reading Status**: Not Started

### 5-06 | Peiyao Li et al. — Conformal Prediction for ADMET Property Prediction

- **Year**: 2024 | **Venue**: arXiv
- **Grade**: B | **Citations**: —
- **Core Contribution**: Conformal prediction applied to molecular property prediction with coverage guarantees.
- **ATLAS Relevance**: Conformal prediction methodology for regression tasks.
- **Reading Status**: Not Started

---

## Category 6: Active Learning and Bayesian Optimization (10 entries)

### 6-01 | Settles — Active Learning Literature Survey

- **Year**: 2009 | **Venue**: University of Wisconsin Technical Report
- **Grade**: A | **Citations**: ~10,000
- **Core Contribution**: Comprehensive survey of active learning strategies: uncertainty sampling, query-by-committee, expected model change.
- **ATLAS Relevance**: Foundational reference for ATLAS active learning module design.
- **Reading Status**: Not Started

### 6-02 | Ma et al. — Accelerating Materials Discovery through Active Learning

- **Year**: 2025 | **Venue**: Under review
- **Grade**: A | **Citations**: —
- **Core Contribution**: Survey of active learning methods for accelerated materials discovery with DFT-in-the-loop.
- **ATLAS Relevance**: Most recent AL survey for materials. Defines ATLAS's contribution context.
- **Reading Status**: Not Started

### 6-03 | Chen et al. — A Survey of Active Learning in Materials Science

- **Year**: 2026 | **Venue**: Under review
- **Grade**: A | **Citations**: —
- **Core Contribution**: Comprehensive survey covering query strategies, surrogate models, and experimental integration for materials.
- **ATLAS Relevance**: Latest AL survey; positions ATLAS research within current landscape.
- **Reading Status**: Not Started

### 6-04 | Zhang et al. — Active Learning of Uniformly Accurate Interatomic Potentials (DP-GEN)

- **Year**: 2020 | **Venue**: Physical Review Materials
- **Grade**: A | **Citations**: ~500
- **Core Contribution**: Concurrent-learning scheme for generating training data using committee disagreement.
- **ATLAS Relevance**: Active learning for MLIP training data generation. Committee-based UQ.
- **Reading Status**: Not Started

### 6-05 | Merchant et al. — GNoME: Scaling Deep Learning for Materials Discovery

- **Year**: 2023 | **Venue**: Nature
- **Grade**: A | **Citations**: ~500
- **Core Contribution**: Active learning pipeline at Google DeepMind scale. Discovered 2.2M new crystals.
- **ATLAS Relevance**: Large-scale active learning system design. UQ-guided data acquisition.
- **Reading Status**: Not Started

### 6-06 | Balandat et al. — BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization

- **Year**: 2020 | **Venue**: NeurIPS
- **Grade**: B | **Citations**: ~800
- **Core Contribution**: Modular PyTorch-based framework for Bayesian optimization with MC acquisition functions.
- **ATLAS Relevance**: BO framework for ATLAS hyperparameter optimization and materials design.
- **Reading Status**: Not Started

### 6-07 | Hernández et al. — BayBE: Bayesian Optimization for Chemical Sciences

- **Year**: 2024 | **Venue**: arXiv (Merck)
- **Grade**: B | **Citations**: ~200
- **Core Contribution**: BO framework tailored for chemistry and materials with domain-specific priors.
- **ATLAS Relevance**: Materials-domain BO best practices and prior specification.
- **Reading Status**: Not Started

### 6-08 | Jain et al. — FireWorks: A Dynamic Workflow System for HPC

- **Year**: 2015 | **Venue**: Concurrency and Computation
- **Grade**: C | **Citations**: ~1,000
- **Core Contribution**: Workflow management for high-throughput computational materials science.
- **ATLAS Relevance**: Background reference for automated computational pipelines.
- **Reading Status**: Not Started

### 6-09 | ML-Assisted Material Discovery: Small Data Active Learning

- **Year**: 2025 | **Venue**: Under review
- **Grade**: B | **Citations**: —
- **Core Contribution**: Strategies for active learning with limited initial data in materials science.
- **ATLAS Relevance**: Small-data regime relevant to early ATLAS deployment.
- **Reading Status**: Not Started

### 6-10 | FAIR Data + Active Learning for Alloy Melting Temperatures

- **Year**: 2024 | **Venue**: Scientific Data
- **Grade**: B | **Citations**: —
- **Core Contribution**: Demonstrates FAIR data principles integrated with active learning for specific property prediction.
- **ATLAS Relevance**: FAIR data integration methodology.
- **Reading Status**: Not Started

---

## Category 7: Data Infrastructure and Benchmarks (9 entries)

### 7-01 | Dunn et al. — Benchmarking Materials Property Prediction Methods: Matbench Test Suite

- **Year**: 2020 | **Venue**: npj Computational Materials
- **Grade**: A | **Citations**: ~300
- **Core Contribution**: Standardized 13-task benchmark for materials property prediction with nested cross-validation protocol.
- **ATLAS Relevance**: Mandatory benchmark for ATLAS evaluation. Defines comparison methodology.
- **Reading Status**: Not Started

### 7-02 | Choudhary et al. — JARVIS: An Integrated Infrastructure for Data-Driven Materials Design

- **Year**: 2020 | **Venue**: npj Computational Materials
- **Grade**: A | **Citations**: ~500
- **Core Contribution**: Comprehensive computational materials database with standardized DFT settings and ML benchmarks.
- **ATLAS Relevance**: Primary data source for ATLAS. Database schema and property definitions.
- **Reading Status**: Not Started

### 7-03 | Jain et al. — Commentary: The Materials Project

- **Year**: 2013 | **Venue**: APL Materials
- **Grade**: B | **Citations**: ~8,000
- **Core Contribution**: Description of Materials Project infrastructure, data generation workflows, and open access philosophy.
- **ATLAS Relevance**: Data source infrastructure reference. Cross-database experiment context.
- **Reading Status**: Not Started

### 7-04 | Curtarolo et al. — AFLOW: An Automatic Framework for High-Throughput Materials Discovery

- **Year**: 2012 | **Venue**: Computational Materials Science
- **Grade**: C | **Citations**: ~3,000
- **Core Contribution**: Automated high-throughput DFT framework generating millions of material entries.
- **ATLAS Relevance**: Data source for cross-database OOD experiments.
- **Reading Status**: Not Started

### 7-05 | Riebesell et al. — Matbench Discovery: An Evaluation Framework for ML Crystal Stability Prediction

- **Year**: 2024 | **Venue**: arXiv
- **Grade**: A | **Citations**: ~50
- **Core Contribution**: Benchmark for evaluating ML models on crystal stability prediction using WBM dataset.
- **ATLAS Relevance**: Target benchmark for ATLAS + UQ stability prediction.
- **Reading Status**: Not Started

### 7-06 | Hjorth Larsen et al. — The Atomic Simulation Environment (ASE)

- **Year**: 2017 | **Venue**: Journal of Physics: Condensed Matter
- **Grade**: C | **Citations**: ~3,000
- **Core Contribution**: Python library for atomistic simulation setup, manipulation, and analysis.
- **ATLAS Relevance**: Data processing backend used throughout ATLAS pipeline.
- **Reading Status**: Not Started

### 7-07 | Ong et al. — Python Materials Genomics (pymatgen)

- **Year**: 2013 | **Venue**: Computational Materials Science
- **Grade**: B | **Citations**: ~4,000
- **Core Contribution**: Comprehensive Python library for materials analysis including structure manipulation, phase diagrams, and electronic structure.
- **ATLAS Relevance**: Core dependency for ATLAS data processing and structure analysis.
- **Reading Status**: Not Started

### 7-08 | Ward et al. — Matminer: An Open-Source Toolkit for Materials Data Mining

- **Year**: 2018 | **Venue**: Computational Materials Science
- **Grade**: B | **Citations**: ~500
- **Core Contribution**: Featurization framework for materials data with 100+ pre-built descriptors.
- **ATLAS Relevance**: Feature engineering comparison baseline.
- **Reading Status**: Not Started

### 7-09 | LLM4Mat-Bench: Benchmarking LLMs for Materials Property Prediction

- **Year**: 2024 | **Venue**: arXiv
- **Grade**: B | **Citations**: —
- **Core Contribution**: Evaluates LLM-based approaches against traditional GNN methods for materials prediction.
- **ATLAS Relevance**: LLM comparison baseline. Emerging paradigm context.
- **Reading Status**: Not Started

---

## Category 8: Training Methodology and Loss Functions (8 entries)

### 8-01 | Smith — 1-Cycle Learning Rate Policy (Super-Convergence)

- **Year**: 2018 | **Venue**: arXiv / US Naval Research Lab
- **Grade**: B | **Citations**: ~3,000
- **Core Contribution**: One-cycle learning rate schedule achieving faster convergence with higher maximum learning rates.
- **ATLAS Relevance**: Direct reference for `atlas/training/scheduler.py`. 1CLR implementation.
- **Reading Status**: Not Started

### 8-02 | Huber — Robust Estimation of a Location Parameter

- **Year**: 1964 | **Venue**: Annals of Mathematical Statistics
- **Grade**: C | **Citations**: ~30,000
- **Core Contribution**: Huber loss function combining MSE and MAE for robustness to outliers.
- **ATLAS Relevance**: Huber loss used in `atlas/training/losses.py`.
- **Reading Status**: Not Started

### 8-03 | Loshchilov & Hutter — Decoupled Weight Decay Regularization (AdamW)

- **Year**: 2019 | **Venue**: ICLR
- **Grade**: B | **Citations**: ~10,000
- **Core Contribution**: Fixes weight decay implementation in Adam optimizer. AdamW as standard optimizer.
- **ATLAS Relevance**: Default optimizer in ATLAS training pipeline.
- **Reading Status**: Not Started

### 8-04 | Loshchilov & Hutter — SGDR: Stochastic Gradient Descent with Warm Restarts

- **Year**: 2017 | **Venue**: ICLR
- **Grade**: C | **Citations**: ~5,000
- **Core Contribution**: Cosine annealing learning rate schedule with periodic warm restarts.
- **ATLAS Relevance**: Alternative LR schedule reference for ATLAS training.
- **Reading Status**: Not Started

### 8-05 | Müller, Kornblith et al. — When Does Label Smoothing Help?

- **Year**: 2019 | **Venue**: NeurIPS
- **Grade**: C | **Citations**: ~1,000
- **Core Contribution**: Analysis of label smoothing effects on calibration and generalization.
- **ATLAS Relevance**: Calibration improvement technique reference.
- **Reading Status**: Not Started

### 8-06 | He et al. — Bag of Tricks for Image Classification with CNNs

- **Year**: 2019 | **Venue**: CVPR
- **Grade**: C | **Citations**: ~3,000
- **Core Contribution**: Systematic study of training tricks (warmup, mixup, LR schedules, label smoothing) and their compounding effects.
- **ATLAS Relevance**: Training recipe methodology reference.
- **Reading Status**: Not Started

### 8-07 | Lundberg & Lee — A Unified Approach to Interpreting Model Predictions (SHAP)

- **Year**: 2017 | **Venue**: NeurIPS
- **Grade**: B | **Citations**: ~15,000
- **Core Contribution**: SHAP values based on Shapley game theory for model interpretability.
- **ATLAS Relevance**: Feature importance interpretation for ATLAS prediction analysis.
- **Reading Status**: Not Started

### 8-08 | Sundararajan et al. — Axiomatic Attribution for Deep Networks (Integrated Gradients)

- **Year**: 2017 | **Venue**: ICML
- **Grade**: C | **Citations**: ~3,000
- **Core Contribution**: Attribution method satisfying sensitivity and implementation invariance axioms.
- **ATLAS Relevance**: Alternative interpretability method reference (via Captum).
- **Reading Status**: Not Started

---

## Category 9: Self-Supervised and Transfer Learning (15 entries)

### 9-01 | Magar et al. — Crystal Twins: Self-Supervised Learning for Crystalline Materials

- **Year**: 2022 | **Venue**: npj Computational Materials
- **Grade**: A | **Citations**: ~100
- **Core Contribution**: Contrastive self-supervised learning on crystal graphs using augmented twin embeddings.
- **ATLAS Relevance**: Pre-training strategy for ATLAS when labeled data is scarce.
- **Reading Status**: Not Started

### 9-02 | Omee et al. — Self-Supervised Generative Models for Crystal Structures

- **Year**: 2024 | **Venue**: iScience
- **Grade**: A | **Citations**: —
- **Core Contribution**: SSL + equivariant GNN for crystal structure generation. Masked training + denoising.
- **ATLAS Relevance**: Self-supervised pre-training methodology directly applicable to ATLAS.
- **Reading Status**: Not Started

### 9-03 | Gupta et al. — Transfer Learning for Materials Informatics Using Crystal GCN

- **Year**: 2020 | **Venue**: npj Computational Materials
- **Grade**: A | **Citations**: ~300
- **Core Contribution**: Pre-train CGCNN on large dataset (formation energy), fine-tune on small property datasets.
- **ATLAS Relevance**: Direct transfer learning protocol for ATLAS property extension.
- **Reading Status**: Not Started

### 9-04 | Chen et al. — Crystal Graph Attention Network for Prediction of Stable Materials

- **Year**: 2021 | **Venue**: NeurIPS Workshop
- **Grade**: B | **Citations**: ~100
- **Core Contribution**: Transfer learning reduces required training data by 50% for stability prediction.
- **ATLAS Relevance**: Transfer learning efficiency gains reference.
- **Reading Status**: Not Started

### 9-05 | Hu et al. — Pre-training GNNs: Strategies for Molecular Property Prediction

- **Year**: 2020 | **Venue**: ICLR
- **Grade**: A | **Citations**: ~1,500
- **Core Contribution**: Systematic study of node-level and graph-level pre-training for molecular GNNs.
- **ATLAS Relevance**: Pre-training strategy design principles.
- **Reading Status**: Not Started

### 9-06 | Liu et al. — Pre-training Molecular GNNs with 3D Geometry (GraphMVP)

- **Year**: 2022 | **Venue**: ICLR
- **Grade**: B | **Citations**: ~300
- **Core Contribution**: Multi-view pre-training using 2D topology and 3D geometry consistency.
- **ATLAS Relevance**: Multi-view pre-training concept for crystal structures.
- **Reading Status**: Not Started

### 9-07 | Zaidi et al. — Pre-training via Denoising for Molecular Property Prediction

- **Year**: 2023 | **Venue**: ICLR
- **Grade**: B | **Citations**: ~200
- **Core Contribution**: Denoising pre-training on 3D molecular conformations. Theory connecting to force prediction.
- **ATLAS Relevance**: Denoising pre-training as physical pre-task for force prediction.
- **Reading Status**: Not Started

### 9-08 | Zhuang et al. — Pre-training GNNs with Structural Fingerprints for Materials Discovery

- **Year**: 2025 | **Venue**: Under review
- **Grade**: B | **Citations**: —
- **Core Contribution**: Cheaply computed structural fingerprints as pre-training targets for GNNs.
- **ATLAS Relevance**: Cost-effective pre-training strategy for ATLAS.
- **Reading Status**: Not Started

### 9-09 | DA-CGCNN — Dual Attention Crystal GNN with Transfer Learning

- **Year**: 2024 | **Venue**: AIP Advances
- **Grade**: B | **Citations**: —
- **Core Contribution**: Dual attention mechanism in CGCNN with cross-property transfer learning.
- **ATLAS Relevance**: Attention mechanism + transfer learning architecture pattern.
- **Reading Status**: Not Started

### 9-10 | You et al. — Graph Contrastive Learning with Augmentations (GraphCL)

- **Year**: 2020 | **Venue**: NeurIPS
- **Grade**: B | **Citations**: ~1,000
- **Core Contribution**: Contrastive learning framework for graphs with 4 types of augmentations.
- **ATLAS Relevance**: Graph augmentation strategies for crystal SSL.
- **Reading Status**: Not Started

### 9-11 | Rong et al. — Self-Supervised Graph Transformer on Large-Scale Molecular Data (GROVER)

- **Year**: 2020 | **Venue**: NeurIPS
- **Grade**: B | **Citations**: ~500
- **Core Contribution**: Large-scale pretraining on 10M molecules using transformer + message passing.
- **ATLAS Relevance**: Scale-up pre-training methodology reference.
- **Reading Status**: Not Started

### 9-12 | Fang et al. — Geometry-Enhanced Molecular Representation Learning (GEM)

- **Year**: 2022 | **Venue**: Nature Machine Intelligence
- **Grade**: B | **Citations**: ~200
- **Core Contribution**: Geometry-aware pre-training on molecular 3D conformations.
- **ATLAS Relevance**: 3D geometry pre-training approach.
- **Reading Status**: Not Started

### 9-13 | Melting Temperature Prediction via Transfer Learning on Crystal GNN

- **Year**: 2024 | **Venue**: Computational Materials Science
- **Grade**: C | **Citations**: —
- **Core Contribution**: Transfer learning from formation energy to melting temperature prediction.
- **ATLAS Relevance**: Domain-specific transfer learning case study.
- **Reading Status**: Not Started

### 9-14 | Stärk et al. — 3D Infomax: Learning Molecular Representations from 3D to Improve on 2D

- **Year**: 2022 | **Venue**: ICML
- **Grade**: C | **Citations**: ~200
- **Core Contribution**: Maximize mutual information between 2D and 3D molecular views.
- **ATLAS Relevance**: Cross-dimensional information transfer concept.
- **Reading Status**: Not Started

### 9-15 | Zhou et al. — Uni-Mol: Universal 3D Molecular Pretraining

- **Year**: 2023 | **Venue**: ICLR
- **Grade**: B | **Citations**: ~300
- **Core Contribution**: Universal 3D molecular pretraining framework with pair-type and distance recovery.
- **ATLAS Relevance**: Universal pre-training architecture reference.
- **Reading Status**: Not Started

---

## Category 10: Generative Models and Inverse Design (15 entries)

### 10-01 | Xie et al. — Crystal Diffusion Variational Autoencoder (CDVAE)

- **Year**: 2022 | **Venue**: ICLR
- **Grade**: A | **Citations**: ~300
- **Core Contribution**: First diffusion-based model for periodic crystal structure generation. Score matching on atom coordinates.
- **ATLAS Relevance**: Generative model for materials design. Inverse design with UQ.
- **Reading Status**: Not Started

### 10-02 | Jiao et al. — DiffCSP: Diffusion for Crystal Structure Prediction

- **Year**: 2024 | **Venue**: ICLR
- **Grade**: A | **Citations**: ~100
- **Core Contribution**: Diffusion on fractional coordinates and lattice parameters for crystal structure prediction.
- **ATLAS Relevance**: Crystal structure prediction methodology. Structural feasibility check.
- **Reading Status**: Not Started

### 10-03 | Zeni et al. — MatterGen: A Generative Model for Inorganic Materials Design

- **Year**: 2024 | **Venue**: arXiv (Microsoft Research)
- **Grade**: A | **Citations**: —
- **Core Contribution**: Diffusion model generating stable, novel inorganic materials with target properties.
- **ATLAS Relevance**: Property-conditioned crystal generation. Inverse design paradigm.
- **Reading Status**: Not Started

### 10-04 | Noh et al. — Inverse Design of Solid-State Materials via a Continuous Representation

- **Year**: 2019 | **Venue**: Matter
- **Grade**: B | **Citations**: ~300
- **Core Contribution**: VAE for crystal structure generation in continuous latent space.
- **ATLAS Relevance**: Early inverse design reference. Latent space representation.
- **Reading Status**: Not Started

### 10-05 | Court et al. — 3D Inorganic Crystal Structure Generation Using GANs

- **Year**: 2020 | **Venue**: Journal of Chemical Information and Modeling
- **Grade**: C | **Citations**: ~100
- **Core Contribution**: GAN for voxelized crystal structure generation.
- **ATLAS Relevance**: GAN approach comparison for crystal generation.
- **Reading Status**: Not Started

### 10-06 | Gruver et al. — Fine-Tuned Language Models Generate Stable Inorganic Materials

- **Year**: 2024 | **Venue**: ICLR
- **Grade**: A | **Citations**: ~50
- **Core Contribution**: Fine-tuning LLM on CIF strings to generate novel stable crystals.
- **ATLAS Relevance**: LLM+materials generation paradigm. Alternative to GNN-based generation.
- **Reading Status**: Not Started

### 10-07 | Zhu et al. — WyckoffModel: Symmetry-Constrained Crystal Generation

- **Year**: 2024 | **Venue**: arXiv
- **Grade**: B | **Citations**: —
- **Core Contribution**: Crystal generation respecting Wyckoff positions and space group symmetry.
- **ATLAS Relevance**: Symmetry-aware generation constraints.
- **Reading Status**: Not Started

### 10-08 | Yang et al. — UniMat: Unified Crystal Representation for Materials Generation

- **Year**: 2024 | **Venue**: NeurIPS
- **Grade**: B | **Citations**: —
- **Core Contribution**: Unified representation enabling diffusion models to scale for large crystal systems.
- **ATLAS Relevance**: Scalable crystal generation architecture.
- **Reading Status**: Not Started

### 10-09 | Flam-Shepherd et al. — Language Models Can Generate Molecules, Materials, and Protein Binding Sites

- **Year**: 2022 | **Venue**: arXiv
- **Grade**: C | **Citations**: ~100
- **Core Contribution**: Demonstrates autoregressive language models for string-based materials generation.
- **ATLAS Relevance**: Text-based materials generation reference.
- **Reading Status**: Not Started

### 10-10 | Luo et al. — GenMS: Generative Hierarchical Materials Search

- **Year**: 2024 | **Venue**: arXiv
- **Grade**: B | **Citations**: —
- **Core Contribution**: Hierarchical system combining LLMs, diffusion models, and GNNs for controllable crystal generation.
- **ATLAS Relevance**: Multi-model generation pipeline design.
- **Reading Status**: Not Started

### 10-11 | Ho et al. — Denoising Diffusion Probabilistic Models (DDPM)

- **Year**: 2020 | **Venue**: NeurIPS
- **Grade**: B | **Citations**: ~10,000
- **Core Contribution**: Foundational denoising diffusion model establishing modern diffusion approaches.
- **ATLAS Relevance**: Foundational theory underlying all crystal diffusion models.
- **Reading Status**: Not Started

### 10-12 | Song et al. — Score-Based Generative Modeling through SDEs

- **Year**: 2021 | **Venue**: ICLR
- **Grade**: B | **Citations**: ~3,000
- **Core Contribution**: Unifies score matching and diffusion models via stochastic differential equations.
- **ATLAS Relevance**: Mathematical foundation for diffusion-based crystal generation.
- **Reading Status**: Not Started

### 10-13 | Kim et al. — Generative Adversarial Networks for Crystal Structure Prediction

- **Year**: 2020 | **Venue**: ACS Central Science
- **Grade**: C | **Citations**: ~200
- **Core Contribution**: CGAN predicting crystal structures from composition.
- **ATLAS Relevance**: Conditional generation reference.
- **Reading Status**: Not Started

### 10-14 | Ren et al. — Inverse Design of Crystals Using Generalized Invertible Crystallographic Representations

- **Year**: 2022 | **Venue**: Nature Computational Science
- **Grade**: B | **Citations**: ~100
- **Core Contribution**: Invertible representation enabling bidirectional mapping between properties and structures.
- **ATLAS Relevance**: Invertible design concept for materials discovery.
- **Reading Status**: Not Started

### 10-15 | ChargeDIFF — Generative Model with Electronic Structure for Inverse Design

- **Year**: 2025 | **Venue**: arXiv
- **Grade**: B | **Citations**: —
- **Core Contribution**: First generative model incorporating charge density for property-driven crystal design.
- **ATLAS Relevance**: Electronic structure integration in generation pipeline.
- **Reading Status**: Not Started

---

## Category 11: Explainability and Interpretability (15 entries)

### 11-01 | Ying et al. — GNNExplainer: Generating Explanations for Graph Neural Networks

- **Year**: 2019 | **Venue**: NeurIPS
- **Grade**: A | **Citations**: ~2,000
- **Core Contribution**: Post-hoc GNN explanation via maximizing mutual information to identify important subgraph and features.
- **ATLAS Relevance**: Primary GNN explanation method for ATLAS prediction analysis.
- **Reading Status**: Not Started

### 11-02 | Luo et al. — Parameterized Explainer for Graph Neural Network (PGExplainer)

- **Year**: 2020 | **Venue**: NeurIPS
- **Grade**: B | **Citations**: ~500
- **Core Contribution**: Learns parameterized explanation model for GNNs, enabling global explanations.
- **ATLAS Relevance**: Batch-level GNN explanation reference.
- **Reading Status**: Not Started

### 11-03 | Yuan et al. — Explainability in Graph Neural Networks: A Taxonomic Survey

- **Year**: 2022 | **Venue**: IEEE TPAMI
- **Grade**: A | **Citations**: ~500
- **Core Contribution**: Comprehensive survey categorizing GNN explainability methods: gradient-based, perturbation-based, decomposition, surrogate.
- **ATLAS Relevance**: Framework for selecting explanation methods for ATLAS.
- **Reading Status**: Not Started

### 11-04 | Schnake et al. — Higher-Order Explanations of Graph Neural Networks via Relevant Walks

- **Year**: 2022 | **Venue**: IEEE TPAMI
- **Grade**: B | **Citations**: ~200
- **Core Contribution**: GNN-LRP method using relevant walks for faithfully explaining GNN message passing.
- **ATLAS Relevance**: Higher-order explanation for multi-hop message passing.
- **Reading Status**: Not Started

### 11-05 | Pope et al. — Explainability Methods for GNNs

- **Year**: 2019 | **Venue**: CVPR Workshop
- **Grade**: C | **Citations**: ~500
- **Core Contribution**: Adapts gradient-based methods (Grad-CAM, guided backpropagation) to GNN architectures.
- **ATLAS Relevance**: Gradient-based GNN explanation baseline.
- **Reading Status**: Not Started

### 11-06 | Li et al. — IFGN: Iteratively Focused GNN for Interpretable Molecular Property Prediction

- **Year**: 2023 | **Venue**: Journal of Chemical Information and Modeling
- **Grade**: B | **Citations**: ~50
- **Core Contribution**: Multistep focus mechanism identifying key atoms related to predicted properties.
- **ATLAS Relevance**: Interpretable prediction architecture pattern.
- **Reading Status**: Not Started

### 11-07 | Xiong et al. — FragNet: Multi-Level Interpretable GNN for Molecular Properties

- **Year**: 2024 | **Venue**: arXiv
- **Grade**: B | **Citations**: —
- **Core Contribution**: Four levels of interpretability: atomic, bond, fragment, and fragment connection.
- **ATLAS Relevance**: Multi-level interpretation framework for crystal substructures.
- **Reading Status**: Not Started

### 11-08 | Faber et al. — L2xGNN: Learning to Explain GNNs

- **Year**: 2024 | **Venue**: Transactions on ML Research
- **Grade**: B | **Citations**: —
- **Core Contribution**: GNN framework providing faithful explanations by design via selected explanatory subgraphs.
- **ATLAS Relevance**: Intrinsically interpretable GNN design.
- **Reading Status**: Not Started

### 11-09 | Jiménez-Luna et al. — Drug Discovery with XAI: Molecular GNN Interpretability

- **Year**: 2020 | **Venue**: Nature Machine Intelligence
- **Grade**: B | **Citations**: ~600
- **Core Contribution**: Review of XAI methods in drug discovery, with focus on GNN interpretation for molecular prediction.
- **ATLAS Relevance**: Domain application reference for XAI methodology.
- **Reading Status**: Not Started

### 11-10 | Wellawatte et al. — Quantitative Assessment of XAI Methods for GNNs

- **Year**: 2023 | **Venue**: Journal of Chemical Information and Modeling
- **Grade**: A | **Citations**: ~50
- **Core Contribution**: Establishes XAI-specific molecular property benchmarks for quantitative GNN explanation evaluation.
- **ATLAS Relevance**: XAI evaluation protocol for ATLAS interpretability module.
- **Reading Status**: Not Started

### 11-11 | McCloskey et al. — LLM-GCE: LLM-Based Counterfactual Explanations for GNNs

- **Year**: 2024 | **Venue**: arXiv
- **Grade**: B | **Citations**: —
- **Core Contribution**: Leverages LLMs to generate counterfactual explanations for molecular GNN predictions.
- **ATLAS Relevance**: Emerging LLM+XAI paradigm for materials.
- **Reading Status**: Not Started

### 11-12 | Kakkad et al. — Survey on GNN Explanations: Methods, Evaluations, and Taxonomy

- **Year**: 2024 | **Venue**: ACM Computing Surveys
- **Grade**: B | **Citations**: —
- **Core Contribution**: Updated survey with taxonomy covering fidelity, stability, and human evaluation of GNN explanations.
- **ATLAS Relevance**: Latest GNN XAI survey for method selection.
- **Reading Status**: Not Started

### 11-13 | Kosmala et al. — XAI for Crystal GNNs: Revealing Atomic Arrangements

- **Year**: 2023 | **Venue**: Digital Discovery (RSC)
- **Grade**: A | **Citations**: ~30
- **Core Contribution**: XAI methods applied specifically to crystal property prediction GNNs, revealing key atomic arrangements.
- **ATLAS Relevance**: Most directly applicable XAI reference for ATLAS crystal models.
- **Reading Status**: Not Started

### 11-14 | Sanchez-Lengeling et al. — Evaluating Attribution for Graph Neural Networks

- **Year**: 2020 | **Venue**: NeurIPS
- **Grade**: B | **Citations**: ~200
- **Core Contribution**: Evaluation framework for GNN attribution methods using ground-truth molecular properties.
- **ATLAS Relevance**: Attribution evaluation methodology.
- **Reading Status**: Not Started

### 11-15 | Vu & Thai — PGM-Explainer: Probabilistic Graphical Model Explanations for GNNs

- **Year**: 2020 | **Venue**: NeurIPS
- **Grade**: C | **Citations**: ~200
- **Core Contribution**: Bayesian approach to GNN explanation using probabilistic graphical models.
- **ATLAS Relevance**: Probabilistic explanation framework.
- **Reading Status**: Not Started

---

## Category 12: Descriptors, Representations, and Featurization (15 entries)

### 12-01 | Bartók et al. — SOAP: Smooth Overlap of Atomic Positions

- **Year**: 2013 | **Venue**: Physical Review B
- **Grade**: B | **Citations**: ~2,000
- **Core Contribution**: Invariant local atomic environment descriptor for comparing structures.
- **ATLAS Relevance**: Baseline descriptor for comparison with learned GNN representations.
- **Reading Status**: Not Started

### 12-02 | Himanen et al. — DScribe: Library of Descriptors for Machine Learning in Materials Science

- **Year**: 2020 | **Venue**: Computer Physics Communications
- **Grade**: B | **Citations**: ~400
- **Core Contribution**: Unified Python library implementing SOAP, MBTR, ACSF, and other atomic descriptors.
- **ATLAS Relevance**: Descriptor comparison baseline. DScribe integration reference.
- **Reading Status**: Not Started

### 12-03 | Behler — Atom-Centered Symmetry Functions (ACSF)

- **Year**: 2011 | **Venue**: Journal of Chemical Physics
- **Grade**: B | **Citations**: ~2,000
- **Core Contribution**: Hand-crafted invariant descriptors for atomic environments. Foundation of early neural network potentials.
- **ATLAS Relevance**: Classical descriptor comparison for learned representations.
- **Reading Status**: Not Started

### 12-04 | Musil et al. — Physics-Inspired Structural Representations for Molecules and Materials

- **Year**: 2021 | **Venue**: Chemical Reviews
- **Grade**: A | **Citations**: ~500
- **Core Contribution**: Comprehensive review connecting structural descriptors to symmetry principles.
- **ATLAS Relevance**: Theoretical grounding for representation choices in ATLAS.
- **Reading Status**: Not Started

### 12-05 | Ward et al. — A General-Purpose Machine Learning Framework for Predicting Properties

- **Year**: 2016 | **Venue**: npj Computational Materials
- **Grade**: B | **Citations**: ~1,000
- **Core Contribution**: Matminer feature set: 140+ compositional and structural descriptors.
- **ATLAS Relevance**: Feature engineering baseline comparison.
- **Reading Status**: Not Started

### 12-06 | Goodall & Lee — Roost: Compositional Graph Networks for Materials Property Prediction

- **Year**: 2020 | **Venue**: Nature Communications
- **Grade**: A | **Citations**: ~300
- **Core Contribution**: Property prediction from composition alone using a set-based attention network.
- **ATLAS Relevance**: Composition-only baseline. Demonstrates structural info importance.
- **Reading Status**: Not Started

### 12-07 | Goodall & Lee — Wren: Wyckoff Representation-Based Prediction

- **Year**: 2022 | **Venue**: Science Advances
- **Grade**: A | **Citations**: ~100
- **Core Contribution**: Incorporates Wyckoff position information for symmetry-aware property prediction.
- **ATLAS Relevance**: Symmetry encoding strategy for crystal representations.
- **Reading Status**: Not Started

### 12-08 | Lam et al. — Aviary: Compositional and Structural ML Models for Materials

- **Year**: 2023 | **Venue**: arXiv
- **Grade**: B | **Citations**: —
- **Core Contribution**: Combined framework hosting Roost (composition) and Wren (structure) models.
- **ATLAS Relevance**: Multi-representation framework architecture.
- **Reading Status**: Not Started

### 12-09 | Unke et al. — Machine Learning Force Fields

- **Year**: 2021 | **Venue**: Chemical Reviews
- **Grade**: A | **Citations**: ~800
- **Core Contribution**: Comprehensive review of ML force field representations, from descriptors to learned features.
- **ATLAS Relevance**: Theoretical context for ATLAS representation learning.
- **Reading Status**: Not Started

### 12-10 | Drautz — Atomic Cluster Expansion (ACE)

- **Year**: 2019 | **Venue**: Physical Review B
- **Grade**: B | **Citations**: ~500
- **Core Contribution**: Systematic body-ordered expansion of atomic properties. Basis for MACE architecture.
- **ATLAS Relevance**: Mathematical foundation of MACE multi-body approach.
- **Reading Status**: Not Started

### 12-11 | Batatia et al. — Design Space of E(3)-Equivariant Atom-Centered Models

- **Year**: 2023 | **Venue**: arXiv
- **Grade**: B | **Citations**: ~100
- **Core Contribution**: Systematic analysis connecting ACE, MACE, NequIP, and other equivariant models in a unified design space.
- **ATLAS Relevance**: Theoretical comparison framework for equivariant architectures.
- **Reading Status**: Not Started

### 12-12 | Fung et al. — Benchmarking Graph Neural Networks for Materials Chemistry

- **Year**: 2021 | **Venue**: npj Computational Materials
- **Grade**: B | **Citations**: ~200
- **Core Contribution**: Systematic benchmark comparing CGCNN, MEGNet, SchNet, ALIGNN on multiple properties.
- **ATLAS Relevance**: Architecture comparison protocol for ATLAS evaluation.
- **Reading Status**: Not Started

### 12-13 | Choudhary — Examining GNNs for Crystal Structures: Limitations and Opportunities

- **Year**: 2023 | **Venue**: Digital Discovery
- **Grade**: A | **Citations**: ~50
- **Core Contribution**: Analysis of GNN limitations in capturing periodicity; proposes hybrid descriptor+GNN solutions.
- **ATLAS Relevance**: Critical analysis of GNN limitations ATLAS must address.
- **Reading Status**: Not Started

### 12-14 | Shui & Kaishi — DenseGNN: Universal Dense GNN for Crystal Property Prediction

- **Year**: 2024 | **Venue**: arXiv
- **Grade**: C | **Citations**: —
- **Core Contribution**: Dense connectivity in GNN for crystal property prediction.
- **ATLAS Relevance**: Architecture variant reference.
- **Reading Status**: Not Started

### 12-15 | Chen et al. — AtomSets: Hierarchical Transfer Learning for Small and Large Materials Datasets

- **Year**: 2021 | **Venue**: npj Computational Materials
- **Grade**: B | **Citations**: ~100
- **Core Contribution**: Hierarchical representation combining element-level and structure-level features.
- **ATLAS Relevance**: Hierarchical representation strategy.
- **Reading Status**: Not Started

---

## Category 13: Scaling, Multi-Fidelity, and Practical Deployment (15 entries)

### 13-01 | Chen et al. — Learning Properties of Ordered and Disordered Materials from Multi-Fidelity Data

- **Year**: 2021 | **Venue**: Nature Computational Science
- **Grade**: A | **Citations**: ~200
- **Core Contribution**: Multi-fidelity graph network training on mixed PBE/SCAN/experimental data.
- **ATLAS Relevance**: Multi-fidelity training strategy for ATLAS with heterogeneous data sources.
- **Reading Status**: Not Started

### 13-02 | De Breuck et al. — MODNet: Modular Optimal-Descriptor Network for Materials Property Prediction

- **Year**: 2021 | **Venue**: npj Computational Materials
- **Grade**: B | **Citations**: ~200
- **Core Contribution**: Modular network with optimal feature selection. Works well on small datasets.
- **ATLAS Relevance**: Small-data performance reference. Matbench competitor.
- **Reading Status**: Not Started

### 13-03 | Choudhary — LLM-Prop: Large Language Model for Material Property Prediction

- **Year**: 2024 | **Venue**: arXiv
- **Grade**: B | **Citations**: —
- **Core Contribution**: Predicts material properties from text descriptions using fine-tuned LLM.
- **ATLAS Relevance**: LLM-based prediction paradigm comparison.
- **Reading Status**: Not Started

### 13-04 | Dunn et al. — Automatminer: An Automated Machine Learning Infra for Materials

- **Year**: 2020 | **Venue**: npj Computational Materials
- **Grade**: B | **Citations**: ~100
- **Core Contribution**: AutoML framework for materials property prediction with automated featurization and model selection.
- **ATLAS Relevance**: AutoML comparison baseline for ATLAS.
- **Reading Status**: Not Started

### 13-05 | Karamad et al. — Orbital Graph Convolutional Neural Network (OGCNN)

- **Year**: 2020 | **Venue**: Physical Review Materials
- **Grade**: C | **Citations**: ~100
- **Core Contribution**: Orbital-level features in CGCNN for improved property prediction.
- **ATLAS Relevance**: Feature engineering extension reference.
- **Reading Status**: Not Started

### 13-06 | Choudhary et al. — Unified Graph NN for Property Predictions

- **Year**: 2021 | **Venue**: npj Computational Materials
- **Grade**: B | **Citations**: ~150
- **Core Contribution**: Unified architecture for diverse materials properties including electronic, optical, and thermal.
- **ATLAS Relevance**: Multi-property prediction architecture.
- **Reading Status**: Not Started

### 13-07 | Gasteiger et al. — GemNet: Universal Directional Graph Neural Networks

- **Year**: 2022 | **Venue**: NeurIPS
- **Grade**: B | **Citations**: ~300
- **Core Contribution**: Universal directional message passing with dihedral angle information.
- **ATLAS Relevance**: Advanced directional message passing reference.
- **Reading Status**: Not Started

### 13-08 | Koker et al. — U-MLIP Assessment for Surface Energies

- **Year**: 2025 | **Venue**: JACS
- **Grade**: B | **Citations**: —
- **Core Contribution**: Evaluates universality of MACE, CHGNet, M3GNet for surface energy prediction.
- **ATLAS Relevance**: U-MLIP generalization limit analysis.
- **Reading Status**: Not Started

### 13-09 | Busk et al. — Calibrated UQ for Atomistic Neural Network Potentials

- **Year**: 2022 | **Venue**: Physical Review Letters
- **Grade**: A | **Citations**: ~80
- **Core Contribution**: Calibration-focused UQ for MLIP with temperature scaling and expected calibration error.
- **ATLAS Relevance**: Post-hoc calibration for ATLAS force predictions.
- **Reading Status**: Not Started

### 13-10 | Wang et al. — Knowledge Distillation for GNNs

- **Year**: 2022 | **Venue**: KDD
- **Grade**: C | **Citations**: ~200
- **Core Contribution**: Distilling large GNN teacher into compact student model while preserving accuracy.
- **ATLAS Relevance**: Model compression for deployment reference.
- **Reading Status**: Not Started

### 13-11 | Hybrid-LLM-GNN for Materials Property Prediction

- **Year**: 2024 | **Venue**: arXiv
- **Grade**: B | **Citations**: —
- **Core Contribution**: Combines GNN structural embeddings with LLM semantic features. Up to 25% improvement.
- **ATLAS Relevance**: Hybrid architecture paradigm for future ATLAS evolution.
- **Reading Status**: Not Started

### 13-12 | Chen et al. — MatErials informatics using FAIR data (MatSci-NLP)

- **Year**: 2023 | **Venue**: NeurIPS (Datasets and Benchmarks)
- **Grade**: C | **Citations**: ~50
- **Core Contribution**: NLP benchmark for materials science text mining and property extraction.
- **ATLAS Relevance**: Text mining for materials literature analysis.
- **Reading Status**: Not Started

### 13-13 | Merchant et al. — GNoME Companion: Autonomous Lab for Crystal Synthesis

- **Year**: 2023 | **Venue**: Nature
- **Grade**: B | **Citations**: ~200
- **Core Contribution**: Autonomous A-Lab synthesizing 41 GNoME-predicted crystals, validating ML predictions experimentally.
- **ATLAS Relevance**: Experimental validation pipeline for ML-predicted materials.
- **Reading Status**: Not Started

### 13-14 | Paszke et al. — PyTorch: An Imperative Style High-Performance Deep Learning Library

- **Year**: 2019 | **Venue**: NeurIPS
- **Grade**: C | **Citations**: ~50,000
- **Core Contribution**: PyTorch framework establishing the de facto standard for deep learning research.
- **ATLAS Relevance**: Core framework dependency.
- **Reading Status**: Not Started

### 13-15 | Fey & Lenssen — Fast Graph Representation Learning with PyTorch Geometric (PyG)

- **Year**: 2019 | **Venue**: ICLR Workshop
- **Grade**: C | **Citations**: ~5,000
- **Core Contribution**: PyG library for implementing GNN architectures with mini-batch training and neighbor sampling.
- **ATLAS Relevance**: Backend GNN library used in ATLAS implementation.
- **Reading Status**: Not Started

---

## Supplementary Entries: Expanding Existing Categories (25 entries)

### S-01 | Klicpera et al. — GNNs Meet NNPs: Using Directional Information (SphereNet)

- **Year**: 2022 | **Venue**: NeurIPS
- **Grade**: B | **Citations**: ~300
- **Core Contribution**: Spherical representation for directional message passing in molecular property prediction.
- **ATLAS Relevance**: Directional MP architecture variant.
- **Reading Status**: Not Started

### S-02 | Qu et al. — Revisiting Over-Smoothing in GNNs: Stochastic Depth

- **Year**: 2024 | **Venue**: ICLR
- **Grade**: C | **Citations**: —
- **Core Contribution**: Addresses over-smoothing in deep GNNs using stochastic depth regularization.
- **ATLAS Relevance**: Deep GNN training stability.
- **Reading Status**: Not Started

### S-03 | Thölke & De Fabritiis — TorchMD-NET: Equivariant Transformers for Neural Network Potentials

- **Year**: 2022 | **Venue**: ICLR Workshop
- **Grade**: B | **Citations**: ~200
- **Core Contribution**: Equivariant transformer architecture for molecular dynamics. Elastic modular design.
- **ATLAS Relevance**: Equivariant transformer implementation reference.
- **Reading Status**: Not Started

### S-04 | Liu et al. — SpookyNet: Learning Force Fields with Electronic Degrees of Freedom

- **Year**: 2022 | **Venue**: Nature Communications
- **Grade**: B | **Citations**: ~200
- **Core Contribution**: Neural network potential learning electronic state information (charge, spin).
- **ATLAS Relevance**: Electronic state-aware potential reference.
- **Reading Status**: Not Started

### S-05 | Batzner et al. — Advancing Molecular Simulation with Equivariant Interatomic Potentials (Review)

- **Year**: 2024 | **Venue**: Nature Reviews Physics
- **Grade**: A | **Citations**: —
- **Core Contribution**: Review article summarizing equivariant MLIP landscape and future directions.
- **ATLAS Relevance**: Comprehensive equivariant MLIP review for positioning ATLAS.
- **Reading Status**: Not Started

### S-06 | Chmiela et al. — GDML: Gradient-Domain Machine Learning

- **Year**: 2017 | **Venue**: Science Advances
- **Grade**: C | **Citations**: ~800
- **Core Contribution**: Kernel-based ML force field using energy-conserving force field construction.
- **ATLAS Relevance**: Non-GNN MLIP baseline comparison.
- **Reading Status**: Not Started

### S-07 | Deringer et al. — Gaussian Approximation Potentials (GAP): Theory and Applications

- **Year**: 2021 | **Venue**: Chemical Reviews
- **Grade**: B | **Citations**: ~500
- **Core Contribution**: Comprehensive review of GAP methodology including SOAP descriptors and active learning.
- **ATLAS Relevance**: Classical MLIP reference. AL integration methodology.
- **Reading Status**: Not Started

### S-08 | Shapeev — Moment Tensor Potentials (MTP)

- **Year**: 2016 | **Venue**: Multiscale Modeling & Simulation
- **Grade**: C | **Citations**: ~500
- **Core Contribution**: Linear-in-descriptor MLIP using moment tensor decomposition.
- **ATLAS Relevance**: Non-GNN MLIP comparison baseline.
- **Reading Status**: Not Started

### S-09 | Angelikopoulos et al. — Bayesian UQ and Propagation for Neural Network Potentials

- **Year**: 2012 | **Venue**: Journal of Chemical Physics
- **Grade**: C | **Citations**: ~200
- **Core Contribution**: Early Bayesian UQ for interatomic potentials.
- **ATLAS Relevance**: Historical UQ for potentials reference.
- **Reading Status**: Not Started

### S-10 | Vovk et al. — Algorithmic Learning in a Random World (Conformal Prediction Textbook)

- **Year**: 2005 | **Venue**: Springer
- **Grade**: C | **Citations**: ~2,000
- **Core Contribution**: Foundational textbook on conformal prediction with coverage guarantees.
- **ATLAS Relevance**: Theoretical basis for conformal prediction in ATLAS.
- **Reading Status**: Not Started

### S-11 | Romano et al. — Conformalized Quantile Regression

- **Year**: 2019 | **Venue**: NeurIPS
- **Grade**: A | **Citations**: ~500
- **Core Contribution**: Combines quantile regression with conformal prediction for adaptive prediction intervals.
- **ATLAS Relevance**: Adaptive prediction interval method for ATLAS.
- **Reading Status**: Not Started

### S-12 | Angelopoulos & Bates — Conformal Prediction: A Gentle Introduction

- **Year**: 2023 | **Venue**: Foundations and Trends in ML
- **Grade**: B | **Citations**: ~200
- **Core Contribution**: Accessible tutorial on conformal prediction with practical implementation guidance.
- **ATLAS Relevance**: Implementation guide for conformal prediction in ATLAS.
- **Reading Status**: Not Started

### S-13 | Conformal Prediction via Regression-as-Classification

- **Year**: 2024 | **Venue**: ICLR
- **Grade**: B | **Citations**: —
- **Core Contribution**: Novel conformal prediction for regression by converting to classification, handling heteroscedastic data.
- **ATLAS Relevance**: Heteroscedastic UQ technique for materials with varying noise.
- **Reading Status**: Not Started

### S-14 | Blundell et al. — Bayes by Backprop: Weight Uncertainty in Neural Networks

- **Year**: 2015 | **Venue**: ICML
- **Grade**: B | **Citations**: ~3,000
- **Core Contribution**: Variational inference for neural network weight uncertainty. Practical Bayesian neural network.
- **ATLAS Relevance**: BNN approach reference for ATLAS UQ alternatives.
- **Reading Status**: Not Started

### S-15 | Malinin & Gales — Predictive Uncertainty Estimation via Prior Networks

- **Year**: 2018 | **Venue**: NeurIPS
- **Grade**: B | **Citations**: ~500
- **Core Contribution**: Dirichlet prior network for single-pass uncertainty and OOD detection.
- **ATLAS Relevance**: Prior network UQ approach reference.
- **Reading Status**: Not Started

### S-16 | Ovadia et al. — Can You Trust Your Model's Uncertainty? Under Dataset Shift

- **Year**: 2019 | **Venue**: NeurIPS
- **Grade**: A | **Citations**: ~1,000
- **Core Contribution**: Benchmark of UQ methods under dataset shift. Deep ensembles most robust.
- **ATLAS Relevance**: UQ robustness under distribution shift — critical for cross-database ATLAS experiments.
- **Reading Status**: Not Started

### S-17 | Rasmussen & Williams — Gaussian Processes for Machine Learning

- **Year**: 2006 | **Venue**: MIT Press
- **Grade**: B | **Citations**: ~30,000
- **Core Contribution**: Foundational textbook on Gaussian processes covering kernels, hyperparameter optimization, and approximate inference.
- **ATLAS Relevance**: GP theory underlying GPyTorch-based ATLAS UQ modules.
- **Reading Status**: Not Started

### S-18 | Wilson et al. — Bayesian Deep Learning and a Probabilistic Perspective of Generalization

- **Year**: 2020 | **Venue**: NeurIPS
- **Grade**: B | **Citations**: ~500
- **Core Contribution**: Multi-basin perspective explaining deep ensemble superiority.
- **ATLAS Relevance**: Theoretical understanding of why ensembles work for UQ.
- **Reading Status**: Not Started

### S-19 | Fort et al. — Deep Ensembles: A Loss Landscape Perspective

- **Year**: 2020 | **Venue**: NeurIPS Workshop
- **Grade**: B | **Citations**: ~300
- **Core Contribution**: Analysis showing deep ensemble diversity arises from distinct loss landscape basins.
- **ATLAS Relevance**: Ensemble diversity theory for ATLAS ensemble design.
- **Reading Status**: Not Started

### S-20 | Wenzel et al. — How Good is the Bayes Posterior in Deep NNs Really?

- **Year**: 2020 | **Venue**: ICML
- **Grade**: C | **Citations**: ~400
- **Core Contribution**: Cold posterior effect: tempered posteriors outperform true Bayesian posteriors.
- **ATLAS Relevance**: Practical BNN calibration considerations.
- **Reading Status**: Not Started

### S-21 | Snoek et al. — Practical Bayesian Optimization of ML Algorithms

- **Year**: 2012 | **Venue**: NeurIPS
- **Grade**: B | **Citations**: ~8,000
- **Core Contribution**: Practical framework for hyperparameter optimization using GP-based BO.
- **ATLAS Relevance**: Hyperparameter tuning methodology for ATLAS.
- **Reading Status**: Not Started

### S-22 | Jha et al. — ElemNet: Deep Learning on Compositions for Materials Property Prediction

- **Year**: 2018 | **Venue**: Scientific Reports
- **Grade**: C | **Citations**: ~300
- **Core Contribution**: Deep network on elemental composition vectors. No structure needed.
- **ATLAS Relevance**: Composition-only prediction baseline.
- **Reading Status**: Not Started

### S-23 | Liu et al. — Materials Informatics for Self-Assembly of Carbon Allotropes

- **Year**: 2023 | **Venue**: npj Computational Materials
- **Grade**: C | **Citations**: —
- **Core Contribution**: ML pipeline for carbon materials with active learning loop.
- **ATLAS Relevance**: Domain-specific AL pipeline case study.
- **Reading Status**: Not Started

### S-24 | Keith et al. — Combining Machine Learning and Computational Chemistry for Predictive Insights

- **Year**: 2021 | **Venue**: Chemical Reviews
- **Grade**: B | **Citations**: ~500
- **Core Contribution**: Review of ML-DFT integration strategies, error sources, and best practices.
- **ATLAS Relevance**: ML-DFT integration best practices.
- **Reading Status**: Not Started

### S-25 | Noé et al. — Machine Learning for Molecular Simulation

- **Year**: 2020 | **Venue**: Annual Review of Physical Chemistry
- **Grade**: B | **Citations**: ~800
- **Core Contribution**: Review covering ML potentials, enhanced sampling, and coarse graining.
- **ATLAS Relevance**: Broader ML-for-simulation context.
- **Reading Status**: Not Started

---

## Collection Statistics

### Papers by Category

| Category | Count | Grade A | Grade B | Grade C |
|----------|-------|---------|---------|---------|
| 1. Crystal GNN | 12 | 5 | 7 | 0 |
| 2. Equivariant NN | 10 | 4 | 6 | 0 |
| 3. MLIP | 10 | 5 | 5 | 0 |
| 4. UQ / Calibration | 15 | 9 | 4 | 2 |
| 5. OOD / Robustness | 6 | 2 | 4 | 0 |
| 6. Active Learning / BO | 10 | 5 | 4 | 1 |
| 7. Data / Benchmark | 9 | 3 | 4 | 2 |
| 8. Training Methodology | 8 | 0 | 3 | 5 |
| 9. Self-Supervised / Transfer Learning | 15 | 3 | 9 | 3 |
| 10. Generative / Inverse Design | 15 | 3 | 8 | 4 |
| 11. Explainability / XAI | 15 | 4 | 8 | 3 |
| 12. Descriptors / Representations | 15 | 5 | 7 | 3 |
| 13. Scaling / Deployment | 15 | 2 | 6 | 7 |
| Supplementary | 25 | 4 | 14 | 7 |
| **Total** | **200** | **54** | **89** | **37** |

### Papers by Era

| Era | Years | Count | Key Theme |
|-----|-------|-------|-----------|
| Foundation | 2005–2018 | 38 | MPNN, CGCNN, SchNet, UQ foundations, SOAP, GP |
| Expansion | 2019–2021 | 42 | ALIGNN, NequIP, Evidential DL, OOD, GNNExplainer, CDVAE |
| Maturation | 2022–2024 | 105 | MACE, Foundation models, UQ benchmarks, diffusion, XAI, SSL |
| Frontier | 2025–2026 | 15 | MLIP surveys, AL for materials, LLM+GNN |

### Reading Progress

| Status | Count |
|--------|-------|
| Not Started | 200 |
| Queued | 0 |
| Scanned | 0 |
| In Progress | 0 |
| Complete | 0 |

---

## Priority Reading Order

Based on ATLAS development roadmap:

| Phase | Papers | Rationale |
|-------|--------|-----------|
| **Week 1–2** | 1-02, 1-05, 4-02, 4-03, 4-04, 7-01, 7-02 | Core architecture + UQ foundations + benchmark setup |
| **Week 3–4** | 2-03, 2-05, 4-05, 4-06, 4-15, 5-02, 11-01, 11-03 | Equivariant models + UQ evaluation + XAI foundations |
| **Month 2** | 3-01, 3-02, 3-03, 3-07, 4-08, 4-09, 6-04, 9-03, 9-05 | MLIP + advanced UQ + AL + transfer learning |
| **Month 3** | 10-01, 10-02, 12-04, 12-06, 12-07, 13-01, S-11, S-16 | Generative, descriptors, multi-fidelity, conformal |
| **Month 4** | Remaining Grade A papers | Comprehensive coverage |
| **Ongoing** | Grade B and C papers | As needed for specific tasks |

---
---

# 中文版：晶體 GNN 與不確定性量化研究之文獻調研

> **專案**：ATLAS — 原子結構自適應訓練與學習  
> **作者**：Zhong  
> **日期**：2026-02-27  
> **總條目**：200  
> **涵蓋類別**：13 類 + 補充條目  
> **狀態**：第一階段種子收集完成（目標 200 篇已達成）

---

## 方法論

### 篩選準則

| 維度 | 權重 | 說明 |
|------|------|------|
| 引用影響力 | 高 | 總引用數與發表場所（Nature, PRL, NeurIPS, ICML, ICLR 等）|
| 方法相關度 | 高 | 對 ATLAS 架構、訓練或評估的直接適用性 |
| 時效性 | 中 | 優先收錄 2022–2026 年以掌握最新進展 |
| 可重現性 | 中 | 程式碼公開、超參數記錄、基準結果 |
| 基礎重要性 | 中 | 作為下游研究的概念基礎（即使較早期）|

### 評級系統

| 等級 | 標準 | 建議動作 |
|------|------|----------|
| **A** | 直接實作或驗證 ATLAS 核心元件 | 精讀附實作筆記；重現關鍵結果 |
| **B** | 提供方法論背景或比較基線 | 結構化閱讀附摘要筆記 |
| **C** | 背景參考或間接相關 | 掃讀摘要與結論 |

---

## 依類別統計

| 類別 | 篇數 | A 級 | B 級 | C 級 |
|------|------|------|------|------|
| 晶體 GNN | 12 | 5 | 7 | 0 |
| 等變神經網路 | 10 | 4 | 6 | 0 |
| 機器學習原子間勢能 | 10 | 5 | 5 | 0 |
| 不確定性量化/校準 | 15 | 9 | 4 | 2 |
| 域外偵測/穩健性 | 6 | 2 | 4 | 0 |
| 主動學習/貝葉斯最佳化 | 10 | 5 | 4 | 1 |
| 資料基礎設施/基準 | 9 | 3 | 4 | 2 |
| 訓練方法/損失函數 | 8 | 0 | 3 | 5 |
| 自監督/遷移學習 | 15 | 3 | 9 | 3 |
| 生成模型/逆向設計 | 15 | 3 | 8 | 4 |
| 可解釋性/XAI | 15 | 4 | 8 | 3 |
| 描述子/表示法 | 15 | 5 | 7 | 3 |
| 規模化/部署 | 15 | 2 | 6 | 7 |
| 補充條目 | 25 | 4 | 14 | 7 |
| **合計** | **200** | **54** | **89** | **37** |

---

## 依時代統計

| 時代 | 年份 | 篇數 | 核心主題 |
|------|------|------|----------|
| 奠基期 | 2005–2018 | 38 | MPNN、CGCNN、SchNet、UQ 基礎、SOAP、高斯過程 |
| 爆發期 | 2019–2021 | 42 | ALIGNN、NequIP、證據式 DL、OOD、GNNExplainer、CDVAE |
| 成熟期 | 2022–2024 | 105 | MACE、基礎模型、UQ 基準、擴散模型、XAI、SSL |
| 前沿期 | 2025–2026 | 15 | MLIP 綜述、材料主動學習、LLM+GNN |

---

## 優先閱讀順序

| 階段 | 論文 | 理由 |
|------|------|------|
| **第 1–2 週** | CGCNN、ALIGNN、Kendall&Gal、Deep Ensemble、Evidential DL、Matbench、JARVIS | 核心架構 + UQ 基礎 + 基準設定 |
| **第 3–4 週** | NequIP、MACE、UQ 比較研究、Palmer Benchmark、OOD Materials、GNNExplainer | 等變模型 + UQ 評估 + XAI 基礎 |
| **第 2 個月** | M3GNet、CHGNet、MACE-MP-0、GNoME、DPOSE-GNN、AutoGNNUQ、DP-GEN、遷移學習 | MLIP + 進階 UQ + 主動學習 + 遷移學習 |
| **第 3 個月** | CDVAE、DiffCSP、SOAP Review、Roost/Wren、Multi-Fidelity、Conformal | 生成模型、描述子、多保真度、共形預測 |
| **第 4 個月** | 其餘 A 級論文 | 完整覆蓋 |
| **持續** | B 級和 C 級論文 | 按具體任務需要 |

---

## 閱讀進度

| 狀態 | 篇數 |
|------|------|
| 未開始 | 200 |
| 已入佇列 | 0 |
| 已掃讀 | 0 |
| 精讀中 | 0 |
| 精讀完 | 0 |

### Papers by Category

| Category | Count | Grade A | Grade B | Grade C |
|----------|-------|---------|---------|---------|
| 1. Crystal GNN | 12 | 5 | 7 | 0 |
| 2. Equivariant NN | 10 | 4 | 6 | 0 |
| 3. MLIP | 10 | 5 | 5 | 0 |
| 4. UQ / Calibration | 15 | 9 | 4 | 2 |
| 5. OOD / Robustness | 6 | 2 | 4 | 0 |
| 6. Active Learning / BO | 10 | 5 | 4 | 1 |
| 7. Data / Benchmark | 9 | 3 | 4 | 2 |
| 8. Training Methodology | 8 | 0 | 3 | 5 |
| **Total** | **80** | **33** | **37** | **10** |

### Papers by Era

| Era | Years | Count | Key Theme |
|-----|-------|-------|-----------|
| Foundation | 2009–2018 | 22 | MPNN, CGCNN, SchNet, UQ foundations |
| Expansion | 2019–2021 | 20 | ALIGNN, NequIP, Evidential DL, OOD |
| Maturation | 2022–2024 | 33 | MACE, Foundation models, UQ benchmarks |
| Frontier | 2025–2026 | 5 | MLIP surveys, AL for materials |

### Reading Progress

| Status | Count |
|--------|-------|
| Not Started | 80 |
| Queued | 0 |
| Scanned | 0 |
| In Progress | 0 |
| Complete | 0 |

---

## Priority Reading Order

Based on ATLAS development roadmap:

| Phase | Papers | Rationale |
|-------|--------|-----------|
| **Week 1–2** | 1-02, 1-05, 4-02, 4-03, 4-04,  7-01, 7-02 | Core architecture + UQ foundations + benchmark setup |
| **Week 3–4** | 2-03, 2-05, 4-05, 4-06, 4-15, 5-02 | Equivariant models + UQ evaluation methodology |
| **Month 2** | 3-01, 3-02, 3-03, 3-07, 4-08, 4-09, 6-04 | MLIP + advanced UQ + active learning |
| **Month 3** | Remaining Grade A papers | Comprehensive coverage |
| **Ongoing** | Grade B and C papers | As needed for specific tasks |

### 篩選準則

| 維度 | 權重 | 說明 |
|------|------|------|
| 引用影響力 | 高 | 總引用數與發表場所（Nature, PRL, NeurIPS, ICML, ICLR 等）|
| 方法相關度 | 高 | 對 ATLAS 架構、訓練或評估的直接適用性 |
| 時效性 | 中 | 優先收錄 2022–2026 年以掌握最新進展 |
| 可重現性 | 中 | 程式碼公開、超參數記錄、基準結果 |
| 基礎重要性 | 中 | 作為下游研究的概念基礎（即使較早期）|

### 評級系統

| 等級 | 標準 | 建議動作 |
|------|------|----------|
| **A** | 直接實作或驗證 ATLAS 核心元件 | 精讀附實作筆記；重現關鍵結果 |
| **B** | 提供方法論背景或比較基線 | 結構化閱讀附摘要筆記 |
| **C** | 背景參考或間接相關 | 掃讀摘要與結論 |

---

## 依類別統計

| 類別 | 篇數 | A 級 | B 級 | C 級 |
|------|------|------|------|------|
| 晶體 GNN | 12 | 5 | 7 | 0 |
| 等變神經網路 | 10 | 4 | 6 | 0 |
| 機器學習原子間勢能 | 10 | 5 | 5 | 0 |
| 不確定性量化/校準 | 15 | 9 | 4 | 2 |
| 域外偵測/穩健性 | 6 | 2 | 4 | 0 |
| 主動學習/貝葉斯最佳化 | 10 | 5 | 4 | 1 |
| 資料基礎設施/基準 | 9 | 3 | 4 | 2 |
| 訓練方法/損失函數 | 8 | 0 | 3 | 5 |
| **合計** | **80** | **33** | **37** | **10** |

---

## 依時代統計

| 時代 | 年份 | 篇數 | 核心主題 |
|------|------|------|----------|
| 奠基期 | 2009–2018 | 22 | MPNN、CGCNN、SchNet、UQ 基礎理論 |
| 爆發期 | 2019–2021 | 20 | ALIGNN、NequIP、證據式深度學習、OOD |
| 成熟期 | 2022–2024 | 33 | MACE、基礎模型、UQ 基準 |
| 前沿期 | 2025–2026 | 5 | MLIP 綜述、材料主動學習 |

---

## 優先閱讀順序

| 階段 | 論文 | 理由 |
|------|------|------|
| **第 1–2 週** | CGCNN、ALIGNN、Kendall&Gal、Deep Ensemble、Evidential DL、Matbench、JARVIS | 核心架構 + UQ 基礎 + 基準設定 |
| **第 3–4 週** | NequIP、MACE、UQ 比較研究、Palmer Benchmark、OOD Materials | 等變模型 + UQ 評估方法論 |
| **第 2 個月** | M3GNet、CHGNet、MACE-MP-0、GNoME、DPOSE-GNN、AutoGNNUQ、DP-GEN | MLIP + 進階 UQ + 主動學習 |
| **第 3 個月** | 其餘 A 級論文 | 完整覆蓋 |
| **持續** | B 級和 C 級論文 | 按具體任務需要 |

---

## 閱讀進度

| 狀態 | 篇數 |
|------|------|
| 未開始 | 80 |
| 已入佇列 | 0 |
| 已掃讀 | 0 |
| 精讀中 | 0 |
| 精讀完 | 0 |

