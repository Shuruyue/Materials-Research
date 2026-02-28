# Open-Source Repository Issue Survey for Crystal GNN and Uncertainty Quantification Research

> **Project**: ATLAS — Accelerated Topological Learning And Screening  
> **Author**: Zhong  
> **Date**: 2026-02-27  
> **Total Entries**: 754  
> **Status**: Phase 2 collection in progress (target: 1,000)

---

## Methodology

### Collection Strategy

Issues were collected from repositories identified in the companion document (`repo_tracker.md`), prioritized by research relevance grade. For each repository, issues were sorted by comment count (descending) to surface the most actively discussed topics.

### Classification System

| Category | Code | Definition |
|----------|------|------------|
| Bug Report | BUG | Confirmed defects, unexpected behavior, crashes |
| Feature Request | FEAT | New functionality, API additions, capability extensions |
| Discussion | DISC | Usage questions, design decisions, architectural debates |
| Performance | PERF | Speed, memory, GPU utilization, scalability |
| Compatibility | COMPAT | Version conflicts, hardware support, dependency issues |
| Data | DATA | Dataset access, format, integrity, preprocessing |

### Relevance Tagging

Issues marked with [CRITICAL] have direct impact on ATLAS development and require attention during implementation.

---

## Section 1: ACEsuit/mace (20 entries)

Repository Grade: A | Total Issues Surveyed: 20

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #321 | LAMMPS-MACE with Kokkos: Illegal memory access | COMPAT | Closed | GPU memory management differences between trained and pretrained models |
| 2 | #945 | Unpickling error with MLIAP | BUG | Closed | Serialization issues affecting model deployment |
| 3 | #195 | LAMMPS on multiple GPUs | PERF | Closed | Multi-GPU training challenges: partitioning and communication |
| 4 | #52 | matscipy neighbour list as default? | PERF | Closed | Neighbor list implementation impacts computational speed |
| 5 | #203 | Compiling MACE model for LAMMPS | COMPAT | Closed | TorchScript export environment issues |
| 6 | #3 | Support for virials and stress | FEAT | Closed | Required for MD pressure control |
| 7 | #555 | Remove e3nn version pin | COMPAT | Open | [CRITICAL] e3nn version conflict directly affects ATLAS equivariant module |
| 8 | #1031 | loss=0 for default head in multihead finetuning | BUG | Closed | Multi-head fine-tuning loss zeroing bug |
| 9 | #415 | Error reading extended xyz files | BUG | Closed | Data format parsing failure |
| 10 | #581 | Problem finalizing pure water model training | BUG | Closed | Weight saving issue at training completion |
| 11 | #11 | Support larger than memory datasets | FEAT | Closed | Memory limitation: streaming data loading |
| 12 | #12 | Support for dipole moments | FEAT | Closed | Property extension |
| 13 | #741 | Can't jit compile model from previous versions | COMPAT | Closed | Version compatibility: old models fail on new compiler |
| 14 | #567 | Multi-GPUs with PBS system | DISC | Closed | HPC PBS scheduler configuration |
| 15 | #724 | Unable to start finetuning | BUG | Closed | Configuration error preventing fine-tuning |
| 16 | #622 | Multihead fine-tuning tensor size error | BUG | Closed | Tensor shape mismatch |
| 17 | #587 | MACE with latest LAMMPS gives error | COMPAT | Closed | LAMMPS rolling update compatibility |
| 18 | #453 | Installation issues | DISC | Closed | torch/e3nn/CUDA installation |
| 19 | #580 | Train periodic + nonperiodic together? | DISC | Closed | Mixed periodic/non-periodic structure training |
| 20 | #458 | Multi-training without SLURM | DISC | Closed | Local multi-GPU training |

---

## Section 2: mir-group/nequip (20 entries)

Repository Grade: A | Total Issues Surveyed: 20

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #288 | OpenMM integration | FEAT | Closed | NequIP to OpenMM potential integration |
| 2 | #210 | Multi-GPU support | DISC | Closed | Distributed training status |
| 3 | #76 | Missing features for OpenMM support | FEAT | Closed | Neighbor list requirements |
| 4 | #572 | State_dict error in fine-tuned models | BUG | Open | [CRITICAL] Weight loading failure after fine-tuning |
| 5 | #69 | Stress tensor | FEAT | Closed | Stress tensor computation for periodic systems |
| 6 | #114 | Merge stress support into master | DISC | Closed | Code integration process |
| 7 | #214 | Crash with large dataset | BUG | Closed | Large dataset stability: memory crash |
| 8 | #587 | Models cannot be pickled | BUG | Closed | Python serialization failure |
| 9 | #67 | Max Recursion Depth | BUG | Closed | Recursion limit bug |
| 10 | #293 | Memory requirements | PERF | Closed | GPU/RAM usage analysis |
| 11 | #346 | nequip-deploy issue | BUG | Closed | Model export tool failure |
| 12 | #206 | MD on Copper Formate | DISC | Closed | System-specific parameters |
| 13 | #326 | Reduce LR plateau only decrease | BUG | Closed | LR scheduler behavior |
| 14 | #563 | AOTInductor segmentation error | BUG | Closed | PyTorch compiler crash |
| 15 | #92 | Train single, evaluate double precision | DISC | Closed | [CRITICAL] Precision trade-off: training speed vs evaluation accuracy |
| 16 | #88 | TypeError reading ASE dataset | DATA | Closed | ASE data loading bug |
| 17 | #121 | TorchScript error | BUG | Closed | TorchScript export failure |
| 18 | #435 | Colab link broken | DATA | Closed | Tutorial link repair |
| 19 | #315 | Does not work with RTX 4080 | COMPAT | Closed | [CRITICAL] Ada Lovelace GPU compatibility |
| 20 | #380 | Custom EarlyStopping | DISC | Closed | Custom convergence criteria |

---

## Section 3: usnistgov/alignn (15 entries)

Repository Grade: A | Total Issues Surveyed: 15

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #90 | Running on Multi-GPUs | FEAT | Open | [CRITICAL] Multi-GPU not officially supported |
| 2 | #93 | Out of Memory Bug | BUG | Open | [CRITICAL] OOM with 28K CIFs during graph precomputation |
| 3 | #170 | Total energy data in JARVIS | DATA | Closed | `optb88vdw_total_energy` field |
| 4 | #54 | JARVIS data / reproducing results | DATA | Closed | [CRITICAL] Data normalization is key to reproducibility |
| 5 | #50 | Using a trained model for inference | DISC | Closed | Inference script usage |
| 6 | #175 | alignn.py forward pass error | BUG | Open | ValueError unpacking error |
| 7 | #118 | Python version >= 3.9 required | COMPAT | Open | scipy dependency requires 3.9+ |
| 8 | #158 | profile module name collision | BUG | Open | Local filename collision |
| 9 | #168 | BadZipFile when downloading JARVIS | BUG | Closed | Download corruption |
| 10 | #180 | LMDB dataset error | BUG | Open | LMDB format data reading failure |
| 11 | #47 | Running regression example | DISC | Closed | Outdated documentation, script path changes |
| 12 | #182 | Training on multiple nodes | DISC | Open | torchrun + SLURM |
| 13 | #79 | CIF non-iterable NoneType | COMPAT | Open | [CRITICAL] Non-standard CIF parsing failure |
| 14 | #155 | ALIGNNTL compat with 2024.2.4 | COMPAT | Open | Transfer Learning version drift |
| 15 | #115 | GPU not utilized (0% in nvidia-smi) | PERF | Open | [CRITICAL] GPU idle during force field training |

---

## Section 4: e3nn/e3nn (20 entries)

Repository Grade: A | Total Issues Surveyed: 20

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #212 | Adding scalar edge feature in Rs_in | DISC | Closed | Edge feature injection into irreps |
| 2 | #125 | Several tests fail locally | BUG | Closed | Local test environment issues |
| 3 | #412 | FullyConnectedTensorProduct with symmetric weights | FEAT | Closed | Symmetric weight tensor products |
| 4 | #292 | Lazy initialization of Modules | FEAT | Closed | Deferred module initialization |
| 5 | #112 | NaN value appears when calculating derivative of kernel | BUG | Closed | [CRITICAL] Numerical stability in gradient computation |
| 6 | #192 | General question on usage | DISC | Closed | General API usage patterns |
| 7 | #520 | Improved compatibility of tests on machines without accelerator | COMPAT | Open | CPU-only testing support |
| 8 | #1 | nvcc compilation fails | COMPAT | Closed | CUDA compiler compatibility |
| 9 | #296 | Equivalent irreps yield different output in FullyConnectedTensorProduct | BUG | Closed | Irrep ordering affects output |
| 10 | #244 | Point cloud estimation | DISC | Closed | Point cloud processing patterns |
| 11 | #266 | Handle irreps as filter_ir_out in TensorProducts | FEAT | Closed | Irrep filtering in tensor products |
| 12 | #507 | Running time of to_cartesian() increases as loops rise | PERF | Closed | Cartesian conversion performance scaling |
| 13 | #35 | reduce_tensor_product vs. tensor_product | DISC | Closed | API design discussion |
| 14 | #149 | equivariance test | FEAT | Closed | Equivariance verification utilities |
| 15 | #177 | Rewrite Tensor classes to support point index batch | FEAT | Closed | Batch processing support |
| 16 | #351 | O(2) 2D CNN? | FEAT | Open | 2D equivariant CNN extension |
| 17 | #91 | Converting pytorch conv3d layers to e3nn layers | COMPAT | Closed | Migration from standard PyTorch |
| 18 | #179 | Implement CUDA version of radius_pbc | PERF | Closed | GPU-accelerated periodic boundary neighbor search |
| 19 | #160 | Updating from se3nn to e3nn | COMPAT | Closed | Migration guide from se3nn to e3nn |
| 20 | #461 | Can't map location of tensor product to CPU | BUG | Closed | Device mapping issue |

---

## Section 5: materialsproject/matbench (15 entries)

Repository Grade: A | Total Issues Surveyed: 15

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #2 | Possible additions and modifications for matbench v1.0 | FEAT | Open | Benchmark evolution roadmap |
| 2 | #172 | Pymatviz figures need refinement | FEAT | Open | Visualization quality for publications |
| 3 | #150 | Discussion on matbench-generative benchmark | DISC | Open | Generative materials benchmark design |
| 4 | #104 | Potential for stability dataset in matbench v1.0 | DATA | Open | Stability prediction benchmark |
| 5 | #254 | Benchmark confirmation on KGCNN | DISC | Closed | Third-party benchmark verification |
| 6 | #40 | Error when recording group probability results | BUG | Closed | Classification task recording bug |
| 7 | #110 | Update Website Generator with UQ stats | FEAT | Open | [CRITICAL] UQ metrics integration into benchmark |
| 8 | #61 | Add MEGNet and SchNet | FEAT | Open | Model coverage expansion |
| 9 | #259 | Per-task plots not regenerated correctly | BUG | Open | Plot generation bug |
| 10 | #233 | Confusion on lack of validation sets | DISC | Closed | [CRITICAL] No built-in validation split design choice |
| 11 | #260 | Need to upgrade EndBug add-and-commit | COMPAT | Open | Dependency update |
| 12 | #269 | Installation error with Python 3.10 | COMPAT | Open | Python version compatibility |
| 13 | #357 | Maintenance status of matbench | DISC | Open | Project sustainability |
| 14 | #71 | Add flag for tasks that contain polymorphs | DATA | Closed | Polymorph handling |
| 15 | #42 | Add ability to record uncertainties with predictions | FEAT | Closed | [CRITICAL] UQ recording capability |

---

## Section 6: materialsvirtuallab/matgl (15 entries)

Repository Grade: A | Total Issues Surveyed: 15

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #264 | Finetuned model worse than pretrained model | BUG | Closed | [CRITICAL] Fine-tuning degradation issue |
| 2 | #59 | Reproducing TF M3GNet with matgl | DISC | Closed | TensorFlow to PyTorch migration discrepancies |
| 3 | #116 | Stochasticity in M3GNet prediction | DISC | Closed | Non-deterministic prediction behavior |
| 4 | #321 | Symbol lookup error with LAMMPS executable | BUG | Closed | Shared library linking issue |
| 5 | #243 | Incompatibility with dgl==2.1.0 and torch==2.2.2 | COMPAT | Closed | [CRITICAL] DGL/PyTorch version conflict |
| 6 | #699 | No module named torchdata.datapipes | COMPAT | Closed | torchdata API removal |
| 7 | #144 | Setting datatypes consistently | FEAT | Closed | Float32/64 consistency |
| 8 | #129 | Should bond vector + distance be in converters? | DISC | Closed | Architecture design decision |
| 9 | #139 | Converting graph info from dgl.Graph back to Structure | FEAT | Closed | Inverse graph-to-structure conversion |
| 10 | #64 | Discrepancies between old m3gnet repo | DISC | Closed | Legacy code comparison |
| 11 | #111 | LAMMPS interface | FEAT | Closed | LAMMPS integration |
| 12 | #182 | Multi-fidelity code for extended QM7b | FEAT | Closed | Multi-fidelity training support |
| 13 | #71 | Example for fine-tuning M3GNet | FEAT | Closed | Fine-tuning tutorial request |
| 14 | #73 | Bug loading M3GNet pretrained model | BUG | Closed | Pretrained model loading failure |
| 15 | #676 | Backwards compatibility question | COMPAT | Closed | Version migration concerns |

---

## Section 7: chemprop/chemprop (15 entries)

Repository Grade: A | Total Issues Surveyed: 15

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #591 | Improve serialization/saving mechanism | FEAT | Closed | Model persistence architecture |
| 2 | #160 | How to load features of SMILES | DISC | Closed | Feature loading patterns |
| 3 | #355 | GPU not used | DISC | Closed | [CRITICAL] GPU utilization debugging |
| 4 | #817 | Metric improvements | FEAT | Closed | Evaluation metric expansion |
| 5 | #858 | On-the-fly graph generation | FEAT | Closed | Dynamic graph construction |
| 6 | #400 | AttributeError: tuple object has no attribute GetAtoms | BUG | Closed | RDKit integration bug |
| 7 | #806 | Make component order in multicomponent not matter | FEAT | Open | Order-invariant multicomponent |
| 8 | #1054 | Model files fail after v1-to-v2 conversion | BUG | Closed | [CRITICAL] Version migration file corruption |
| 9 | #312 | How to determine MPN encoding quality | DISC | Closed | Representation quality assessment |
| 10 | #969 | Fix noam scheduler definition | BUG | Closed | Learning rate scheduler bug |
| 11 | #397 | Can't kekulize mol | DISC | Closed | RDKit kekulization failure |
| 12 | #386 | Transfer learning, features, checkpoint paths | DISC | Closed | Transfer learning workflow |
| 13 | #108 | Features for antibiotics checkpoints | DATA | Closed | Domain-specific pretrained models |
| 14 | #763 | Indexing error | BUG | Closed | Array indexing bug |
| 15 | #255 | Pretrained models unavailable | DISC | Closed | Model distribution |

---

## Section 8: cornellius-gp/gpytorch (20 entries)

Repository Grade: A | Total Issues Surveyed: 20

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #901 | Fixed noise Gaussian likelihood in multi-task setting | FEAT | Open | Multi-task GP with fixed noise |
| 2 | #606 | Kernel design | DISC | Open | Custom kernel architecture |
| 3 | #74 | Ensure compatibility with PyTorch master | COMPAT | Closed | Upstream version tracking |
| 4 | #863 | Batch-GP for learning common GP over experiments | DISC | Closed | Batch GP design patterns |
| 5 | #22 | import gpytorch error | BUG | Closed | Installation failure |
| 6 | #1021 | Nonstationary kernels | DISC | Open | Non-stationary kernel implementations |
| 7 | #323 | Multidimensional GP regression with IndexKernel | DISC | Open | Multi-output GP setup |
| 8 | #939 | Replicating doubly stochastic variational inference | BUG | Closed | Variational inference reproduction |
| 9 | #1591 | Missing data likelihoods | FEAT | Closed | Handling incomplete observations |
| 10 | #1041 | Pointer to getting started with GPLVM | DISC | Closed | Latent variable model documentation |
| 11 | #864 | FixedNoiseGaussianLikelihood produces negative variance | BUG | Open | [CRITICAL] Numerical instability in variance computation |
| 12 | #822 | Upstream tensor comparison changes break things | COMPAT | Closed | PyTorch API change propagation |
| 13 | #1035 | Variational multitask GP with correlated outputs | BUG | Closed | Correlated output modeling |
| 14 | #967 | Variational inference and sparse approach | DISC | Closed | Sparse GP methods |
| 15 | #391 | TypeError: InvQuadLogDet.forward expected Variable | BUG | Closed | Type mismatch in log-determinant |
| 16 | #725 | Efficient logdet for KroneckerProductLazyTensor | PERF | Closed | Kronecker product efficiency |
| 17 | #674 | Implementing custom mean function documentation | FEAT | Closed | Custom mean function guide |
| 18 | #1015 | Deep Gaussian Process for classification | FEAT | Open | Deep GP classification support |
| 19 | #1115 | Computing probabilities from GPR | DISC | Closed | Predictive probability extraction |
| 20 | #772 | Gradients of kernels / backward functions | DISC | Closed | Kernel gradient computation |

---

## Section 9: pytorch/botorch (20 entries)

Repository Grade: A | Total Issues Surveyed: 20

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #578 | Modifying Knowledge Gradient for time-dependent kernels | FEAT | Closed | Temporal kernel acquisition functions |
| 2 | #179 | Numerical issue with Cholesky decomposition (with normalization) | BUG | Closed | [CRITICAL] Cholesky decomposition numerical stability |
| 3 | #180 | Implement heteroskedastic GP tutorial | FEAT | Open | Heteroskedastic GP for varying noise |
| 4 | #725 | Classification model as output constraint | FEAT | Open | Constrained optimization with classifiers |
| 5 | #546 | Setting up custom GPyTorch model for BoTorch | DISC | Closed | Custom model integration |
| 6 | #1366 | Add maximum variance acquisition for active learning | FEAT | Closed | [CRITICAL] Active learning acquisition function |
| 7 | #1685 | Hypervolume computation thousands of times slower than moocore | PERF | Open | Multi-objective optimization performance |
| 8 | #2938 | qKnowledgeGradient shape mismatch with SaasFullyBayesianSingleTaskGP | BUG | Open | Fully Bayesian GP compatibility |
| 9 | #641 | Possible memory leak in optimize_acqf | BUG | Closed | Memory leak in optimization loop |
| 10 | #177 | Numerical issue with Cholesky decomposition | BUG | Closed | Cholesky decomposition failure |
| 11 | #626 | Error with FixedFeatureAcquisitionFunction and multi-fidelity KG | BUG | Closed | Multi-fidelity acquisition bug |
| 12 | #2035 | Constrained multi-objective optimization for material discovery using qNEHVI | FEAT | Closed | [CRITICAL] Materials discovery with constrained MOBO |
| 13 | #1323 | fit_gpytorch_model complains of tensors on multiple devices | BUG | Closed | Multi-device tensor handling |
| 14 | #667 | Questions regarding contextual GPs / LCEMGP | DISC | Closed | Contextual GP design |
| 15 | #1326 | Batch queries not respecting constraints in multi-objective | BUG | Closed | Constraint enforcement in batch mode |
| 16 | #861 | Log transform not applied in heteroskedastic GP posterior | BUG | Closed | Transform propagation bug |
| 17 | #1036 | Can qNEHVI be used with KroneckerMultiTaskGP? | DISC | Closed | Multi-task MOBO compatibility |
| 18 | #1435 | Normalize input_transform fails with condition_on_observations | BUG | Closed | Input normalization in conditioning |
| 19 | #798 | Cholesky decomposition: CUDA out of memory | BUG | Closed | [CRITICAL] GPU memory exhaustion during Cholesky |
| 20 | #1679 | Derivative-enabled GPs | FEAT | Open | Gradient-enhanced GP models |

---

## Section 10: materialsproject/pymatgen (20 entries)

Repository Grade: A | Total Issues Surveyed: 20

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #3006 | Spglib unable to obtain spacegroup for large supercells | BUG | Open | Symmetry detection failure at scale |
| 2 | #665 | Proposal for new k-point generation scheme | FEAT | Closed | Brillouin zone sampling |
| 3 | #1585 | Wavecar fails for spin-polarized HSE | BUG | Closed | VASP output reader bug |
| 4 | #219 | Read WAVEDER(F) from VASP LOPTICS runs | FEAT | Closed | Optical properties I/O |
| 5 | #2083 | Please consider switching back to semantic versioning | DISC | Closed | Versioning policy debate |
| 6 | #1746 | POTCAR file hashes appear incorrect | DATA | Closed | [CRITICAL] Pseudopotential data integrity |
| 7 | #1985 | get_equilibrium_reaction_energy no longer works | BUG | Closed | Phase diagram API breaking change |
| 8 | #3888 | CrystalNN bug with atom bonded to its own images | BUG | Closed | [CRITICAL] Neighbor algorithm bug affecting graph construction |
| 9 | #1868 | StructureMatcher.get_rms_dist illegal instruction | COMPAT | Closed | Binary compatibility issue |
| 10 | #4243 | Remove is_rare_earth_metal | BUG | Closed | Element classification cleanup |
| 11 | #1321 | Bug in NearNeighbors.get_all_nn_info | BUG | Closed | Neighbor information extraction bug |
| 12 | #3322 | Changing MP Input Sets | DISC | Open | Input set evolution policy |
| 13 | #1361 | Suppression of warnings in CrystalNN | FEAT | Closed | Warning verbosity control |
| 14 | #2056 | latexify_ion is broken | BUG | Closed | LaTeX rendering bug |
| 15 | #1486 | MagOrderingTransformation Bug with New Enumlib | BUG | Closed | Magnetic ordering enumeration |
| 16 | #2010 | Cannot install pymatgen with Python 3.6 | COMPAT | Closed | Python version requirement |
| 17 | #1884 | Possible Potcar hash changes in new versions | DATA | Closed | Pseudopotential hash tracking |
| 18 | #2345 | GH workflows check not on fork | BUG | Closed | CI/CD workflow issue |
| 19 | #3068 | Reading and writing POTCAR files broken | BUG | Closed | VASP I/O regression |
| 20 | #2968 | Entry sets use wrong Yb pseudo-potential | DATA | Closed | Material-specific data error |

---

## Section 11: CederGroupHub/chgnet (15 entries)

Repository Grade: A | Total Issues Surveyed: 15

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #57 | LAMMPS interface for CHGNet | FEAT | Open | LAMMPS integration demand |
| 2 | #248 | CHGNet on CPU: floating point exception (core dumped) | BUG | Closed | CPU fallback failure |
| 3 | #160 | pip install chgnet failed | COMPAT | Closed | Installation dependency issues |
| 4 | #56 | Limited speed boost when increasing batch_size in predict_structure | PERF | Closed | [CRITICAL] Batch prediction scaling inefficiency |
| 5 | #127 | ase.filters ModuleNotFoundError in example notebook | BUG | Closed | ASE version compatibility |
| 6 | #85 | IndexError on predict_structure | BUG | Closed | Inference-time crash |
| 7 | #145 | Issue with CUDA device allocation on HPC | BUG | Closed | Multi-GPU HPC allocation |
| 8 | #188 | Add dispersion to ASE calculator | FEAT | Closed | Dispersion correction request |
| 9 | #38 | Failed to use CUDA | BUG | Closed | GPU initialization failure |
| 10 | #125 | pypi release 0.2.0 doesn't match GitHub release | COMPAT | Closed | Release synchronization |
| 11 | #32 | Unexpected relaxation steps | DISC | Closed | Relaxation behavior |
| 12 | #35 | CHGNet fails for isolated atoms | BUG | Closed | Edge case: isolated atoms |
| 13 | #79 | Energy jump when bond_graph_len becomes zero | BUG | Closed | [CRITICAL] Discontinuity in potential energy surface |
| 14 | #230 | Unable to compile LAMMPS with CHGNet | COMPAT | Closed | LAMMPS compilation |
| 15 | #168 | Error loading trainer state | BUG | Closed | Checkpoint loading failure |

---

## Section 12: pyg-team/pytorch_geometric (20 entries)

Repository Grade: A | Total Issues Surveyed: 20

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #3958 | Link prediction on heterogeneous graphs | FEAT | Closed | Heterogeneous graph learning |
| 2 | #331 | Issue reproducing ECC implementation results | BUG | Closed | Reproducibility challenges |
| 3 | #3324 | Find distance between bounding boxes | DISC | Open | Spatial distance computation |
| 4 | #355 | spspmm CUDA bugfix | BUG | Closed | Sparse matrix GPU bug |
| 5 | #3230 | Roadmap: Temporal Graph Support | FEAT | Open | Temporal graph roadmap |
| 6 | #625 | enzymes_topk_pool model is not learning | BUG | Closed | Pool layer training failure |
| 7 | #64 | Neighborhood sampling | FEAT | Closed | Sampling strategies |
| 8 | #1125 | OSError: libcusparse.so.10 | COMPAT | Closed | [CRITICAL] CUDA sparse library linking |
| 9 | #1718 | torch_sparse::ptr2ind not found in TorchScript | BUG | Closed | TorchScript export failure |
| 10 | #4026 | Link-level NeighborLoader | FEAT | Closed | Link prediction data loading |
| 11 | #2040 | libcusparse.so.11 cannot open shared object | COMPAT | Open | CUDA library version mismatch |
| 12 | #5680 | Jupyter kernel dies when importing PyG | BUG | Open | Import crash |
| 13 | #4848 | Data batch problem | BUG | Closed | Batch collation issue |
| 14 | #251 | Segmentation fault in EdgeConv forward | BUG | Closed | Memory access violation |
| 15 | #2677 | Training on multiple graphs | DISC | Open | Multi-graph training patterns |
| 16 | #999 | Undefined symbol _ZN5torch3jit17parseSchemaOrName | COMPAT | Closed | C++ ABI compatibility |
| 17 | #2429 | NoneType object has no attribute origin | BUG | Closed | Null reference error |
| 18 | #155 | Error running VGAE | BUG | Closed | Variational autoencoder bug |
| 19 | #467 | Cannot install torch-scatter via conda | COMPAT | Closed | [CRITICAL] Extension installation |
| 20 | #4182 | LightGCN example | FEAT | Closed | Recommendation system example |

---

## Section 13: txie-93/cgcnn (10 entries)

Repository Grade: A | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #9 | Error raised while testing README code | BUG | Closed | Setup verification failure |
| 2 | #21 | How to understand output to generate crystal graphs | DISC | Open | Graph construction internals |
| 3 | #37 | Size mismatch when loading pretrained models | COMPAT | Open | [CRITICAL] Pretrained model compatibility |
| 4 | #13 | Issues about prediction example | BUG | Closed | Prediction pipeline bug |
| 5 | #20 | CGCNN can't run on certain CIF files | DATA | Open | [CRITICAL] Non-standard CIF handling |
| 6 | #25 | Description for Materials Project CSV files | DATA | Open | Dataset format documentation |
| 7 | #22 | How to run through Jupyter or Python interface | DISC | Open | API usage patterns |
| 8 | #12 | How to bind CGCNN for matminer | COMPAT | Closed | Framework integration |
| 9 | #43 | MAE for shear/bulk modulus very large | PERF | Closed | Prediction accuracy for mechanical properties |
| 10 | #26 | Predict at atom level | FEAT | Open | Atom-level property prediction |

---

## Section 14: FAIR-Chem/fairchem (15 entries)

Repository Grade: A | Total Issues Surveyed: 15

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #888 | Error: stress property not implemented | BUG | Closed | Stress tensor support gap |
| 2 | #563 | Non-deterministic calculation results | BUG | Closed | [CRITICAL] Reproducibility issue |
| 3 | #1568 | Found NaN while computing loss | BUG | Closed | [CRITICAL] NaN loss during training |
| 4 | #1201 | Adaptation of research to UMA? | DISC | Closed | Universal model architecture migration |
| 5 | #242 | torch package issue | COMPAT | Closed | PyTorch packaging conflict |
| 6 | #170 | Unable to run SchNet demo notebook | BUG | Closed | Tutorial reproduction failure |
| 7 | #981 | Problem loading tuned checkpoint for OCPNEB | BUG | Closed | Checkpoint compatibility |
| 8 | #936 | Trouble loading pretrained EquiformerV2 with fairchem 1.3.0 | BUG | Closed | Version-specific loading failure |
| 9 | #1618 | NaN in second derivative | BUG | Closed | Second-order gradient instability |
| 10 | #1566 | OC25 Metadata | DATA | Closed | Dataset metadata |
| 11 | #366 | Help with training S2EF task | DISC | Closed | Structure-to-energy-forces training |
| 12 | #395 | ValueError: Empty module name | BUG | Closed | Module registration bug |
| 13 | #629 | ASE databases incompatible with fine-tuning tutorial | COMPAT | Closed | Tutorial version drift |
| 14 | #1087 | Type of energy from get_potential_energy() | DISC | Closed | Energy reference convention |
| 15 | #1664 | Water in crystal lattice converged to H3O+ | BUG | Closed | Chemistry-specific convergence issue |

---

## Section 15: deepmodeling/deepmd-kit (15 entries)

Repository Grade: A | Total Issues Surveyed: 15

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #2229 | Training wall time abnormally long with many systems | PERF | Closed | [CRITICAL] Multi-system training performance |
| 2 | #2039 | -DUSE_TF_PYTHON_LIBS useless for C++ interface | BUG | Closed | Build system configuration |
| 3 | #3094 | DP compress + DPLR + GPU neighbor list bug in LAMMPS | BUG | Closed | Model compression interaction bug |
| 4 | #1550 | gmx-dp free() invalid next size | BUG | Closed | GROMACS integration memory error |
| 5 | #1774 | Parallel training with horovodrun not working | BUG | Closed | Distributed training failure |
| 6 | #1957 | How to specify TENSORFLOW_ROOT during install | BUG | Closed | Installation configuration |
| 7 | #790 | Failure to compress a model | BUG | Closed | Model compression failure |
| 8 | #4594 | PyTorch parallel training neighbor stat OOM | PERF | Closed | [CRITICAL] Parallel training memory exhaustion |
| 9 | #5117 | dpgen stuck in data stating stage | PERF | Open | Data pipeline bottleneck |
| 10 | #1809 | LAMMPS make unrecognized pair error | COMPAT | Closed | LAMMPS pair style registration |
| 11 | #1090 | MPICH in conda binary incompatible with horovod | COMPAT | Closed | MPI library conflict |
| 12 | #1062 | cuBLAS routine failure in training | BUG | Closed | GPU linear algebra crash |
| 13 | #1656 | Different LAMMPS versions produce different NVE trajectories | COMPAT | Closed | [CRITICAL] Non-reproducible dynamics across versions |
| 14 | #1018 | GPUs on node 2 not detected by TensorFlow | BUG | Closed | Multi-node GPU detection |
| 15 | #3040 | Generalization of energy model to scalar/tensor model | FEAT | Closed | Property prediction extension |

---

## Section 16: usnistgov/jarvis (15 entries)

Repository Grade: A | Total Issues Surveyed: 15

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #163 | cif2cell breaking conda-forge recipe | COMPAT | Open | Dependency conflict in packaging |
| 2 | #9 | Move website to pages.nist.gov | FEAT | Closed | Infrastructure migration |
| 3 | #221 | Invalid OPTIMADE API responses | BUG | Open | API standards compliance |
| 4 | #132 | LICENSE.rst -> LICENSE | DISC | Open | Licensing format |
| 5 | #126 | conda-forge installation | COMPAT | Open | Package distribution |
| 6 | #202 | Units and normalization factors in QM9 dataset | DATA | Open | [CRITICAL] Dataset unit documentation |
| 7 | #2 | Dedication to public domain | DISC | Closed | Open-source licensing |
| 8 | #227 | Descriptors meaning in JARVIS-DFT | DISC | Open | Feature documentation |
| 9 | #251 | get_request_data writes to restricted location | BUG | Open | File system permission issue |
| 10 | #291 | BadZipFile: File is not a zip file | BUG | Closed | Download corruption |
| 11 | #127 | LICENSE file is messed up | BUG | Closed | License file formatting |
| 12 | #304 | Discrepancy in modulus from elastic tensor vs database | DATA | Open | [CRITICAL] Data consistency issue |
| 13 | #106 | Zr symbol not represented | DATA | Open | Element coverage gap |
| 14 | #149 | Problems getting real/imaginary dielectric function data | DATA | Closed | Optical data access |
| 15 | #181 | Couldn't find mp_jv_id.json | DATA | Open | Cross-database ID mapping |

---

## Section 17: hackingmaterials/matminer (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #36 | Can't clone matminer | BUG | Closed | Repository access issue |
| 2 | #135 | BaseFeaturizer doesn't support certain featurizer types | BUG | Closed | Featurizer API limitation |
| 3 | #91 | Voronoi-tessellation-based features | FEAT | Closed | Structural descriptor implementation |
| 4 | #295 | Parallel featurization and FunctionFeaturizer problems | BUG | Closed | [CRITICAL] Parallel computation bugs |
| 5 | #606 | Electronic Transport Properties via load_dataset | DATA | Closed | Dataset availability |
| 6 | #915 | Missing compatibility with pandas v2 | COMPAT | Closed | pandas API migration |
| 7 | #241 | Add Many-Body Tensor Representation | FEAT | Closed | Descriptor extension |
| 8 | #232 | figrecipes should use palettable colors | FEAT | Closed | Visualization quality |
| 9 | #159 | Site-based featurizers in structure module | FEAT | Closed | Site-level feature extraction |
| 10 | #302 | structure_to_oxidstructure should be parallelized | PERF | Closed | Oxidation state assignment speed |

---

## Section 18: uncertainty-toolbox (35 entries)

Repository Grade: A | Total Issues Surveyed: 35

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #21 | Directly access scaled uncertainties using recalibration model | FEAT | Open | [CRITICAL] Recalibration API for ATLAS UQ pipeline |
| 2 | #39 | Calculate metrics for quantile predictions | FEAT | Open | Quantile regression metrics |
| 3 | #20 | Installation on Google Colab | COMPAT | Closed | Cloud environment support |
| 4 | #22 | Using this package on ML results | DISC | Closed | General usage patterns |
| 5 | #5 | Finalize structure of repo | DISC | Closed | Architecture decisions |
| 6 | #61 | Layman definition of interval score | DISC | Closed | Metric interpretation |
| 7 | #41 | Uncertainty quantification of a neural network | DISC | Closed | Neural network UQ patterns |
| 8 | #75 | Should calibration curve axes be switched? | DISC | Open | Calibration visualization convention |
| 9 | #64 | Convert symmetric CI to standard deviation | DISC | Closed | Uncertainty representation conversion |
| 10 | #46 | MACE vs ECE | DISC | Closed | [CRITICAL] Calibration metric comparison |
| 11 | #86 | Interest in classification uncertainty metrics? | FEAT | Open | Classification UQ extension |
| 12 | #57 | Bad practice for single confidence level interval score? | DISC | Closed | Best practices |
| 13 | #6 | Finalize components of initial release | DISC | Closed | API scope |
| 14 | #4 | Help give definition for glossary | DISC | Closed | Terminology |
| 15 | #2 | Decide on API | DISC | Closed | API design |
| 16 | #9 | Improve visualizations | FEAT | Closed | Plot quality |
| 17 | #10 | Define calibration in glossary | DISC | Closed | Calibration definition |
| 18 | #63 | os.add_dll_directory filename too long | BUG | Open | Windows compatibility |
| 19 | #62 | Examples showing recalibration method preference | DISC | Closed | Method comparison |
| 20 | #81 | PyTorch GPU acceleration | PERF | Open | [CRITICAL] GPU support for UQ metrics |
| 21 | #80 | Quantile regression | FEAT | Closed | Quantile method |
| 22 | #77 | Option for bivariate distribution plots | FEAT | Closed | Visualization extension |
| 23 | #76 | Add parity plots | FEAT | Closed | Parity plot support |
| 24 | #90 | get_all_metrics and prop_type | BUG | Open | Metric API bug |
| 25 | #55 | Quantile loss | FEAT | Open | Loss function |
| 26 | #93 | Penalized Brier Score and Penalized Log Loss | FEAT | Open | [CRITICAL] Novel calibration metrics |
| 27 | #58 | Opposed to PyPI or conda installation? | COMPAT | Closed | Distribution channel |
| 28 | #59 | shapely issue on import | COMPAT | Open | Dependency conflict |
| 29 | #82 | Make requirements more lightweight | PERF | Closed | Dependency minimization |
| 30 | #3 | Help give glossary definitions | DISC | Closed | Terminology |
| 31 | #7 | Recalibration implementation | FEAT | Closed | Recalibration method |
| 32 | #1 | Help give glossary definitions | DISC | Closed | Terminology |
| 33 | #85 | Expected Normalized Calibration Error (ENCE) | FEAT | Open | [CRITICAL] ENCE metric implementation |
| 34 | #87 | check_score | FEAT | Closed | Score validation |
| 35 | #88 | Recalibration model as argument for get_all_metrics | FEAT | Open | API integration |

---
---

## Section 19: mir-group/allegro (10 entries)

Repository Grade: A | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #133 | ML-IAP interface bug with fix langevin | BUG | Open | Temperature control with MLIP |
| 2 | #3 | Unable to open Colab tutorial | BUG | Closed | Tutorial access |
| 3 | #45 | Allegro memory requirements | DISC | Closed | [CRITICAL] Memory scaling analysis |
| 4 | #131 | Performance gain from Triton custom kernels | PERF | Closed | Triton kernel optimization |
| 5 | #8 | atom_types not included in pytest | BUG | Closed | Test coverage gap |
| 6 | #68 | Activation parity | FEAT | Closed | Parity equivariance setting |
| 7 | #107 | Problem restarting jobs | BUG | Closed | Checkpoint restart failure |
| 8 | #111 | ValueError: Key model_builders different in config vs trainer.pth | BUG | Closed | Config serialization mismatch |
| 9 | #138 | compile_mode=compile fails with CuEquivariance | BUG | Closed | Compilation compatibility |
| 10 | #76 | How to modify training loss for specific atoms | DISC | Closed | Per-atom loss weighting |

---

## Section 20: SINGROUP/dscribe (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #28 | MBTR normalization details | DISC | Closed | Many-body tensor representation normalization |
| 2 | #31 | Non-linear scaling of SOAP calculation | PERF | Closed | [CRITICAL] SOAP computational scaling |
| 3 | #42 | Install error with 0.3.2 | COMPAT | Closed | Version-specific installation |
| 4 | #46 | Descriptor for crystal | FEAT | Closed | Crystal-specific descriptor support |
| 5 | #48 | Symmetry of terms for multi-species descriptor | BUG | Closed | Multi-species symmetry handling |
| 6 | #157 | dscribe does not build on Python 3.13 | COMPAT | Closed | Python version support |
| 7 | #8 | Error installing with macOS | COMPAT | Closed | Platform compatibility |
| 8 | #58 | SOAP descriptors of different species too similar | DISC | Closed | Descriptor discrimination power |
| 9 | #70 | NaN value in ACSF descriptors | BUG | Closed | Numerical stability in descriptors |
| 10 | #30 | Possible inaccuracy of SOAP for periodic systems | BUG | Closed | [CRITICAL] Periodic boundary accuracy |

---

## Section 21: aamini/evidential-deep-learning (10 entries)

Repository Grade: A | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #18 | Epistemic uncertainty behavior weird for toy dataset | BUG | Open | [CRITICAL] UQ output interpretation challenges |
| 2 | #2 | When is the PyTorch version available? | FEAT | Open | Framework migration demand |
| 3 | #16 | Add classification examples | FEAT | Open | Classification UQ examples |
| 4 | #5 | Loss goes to NaN | BUG | Open | [CRITICAL] NaN loss in evidential training |
| 5 | #14 | NIG_Loss smaller than zero | BUG | Closed | Negative loss value |
| 6 | #9 | TypeError: Dirichlet_SOS missing argument | BUG | Open | API usage error |
| 7 | #10 | Mistake in equation (S26) | BUG | Closed | Mathematical derivation error |
| 8 | #11 | var in hello_world.py | DISC | Open | Variable interpretation |
| 9 | #4 | Missing wine dataset | DATA | Open | Dataset availability |
| 10 | #20 | What is the standard score regularizer? | DISC | Open | Regularizer explanation |

---

## Section 22: facebookresearch/hydra (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #1366 | Submitit-Plugin: impossible to set certain flags | FEAT | Closed | HPC job submission configuration |
| 2 | #1126 | Can't see log with PyTorch DDP | BUG | Closed | [CRITICAL] DDP logging visibility |
| 3 | #274 | Allow configuring search path via config file | FEAT | Closed | Config path management |
| 4 | #215 | Permit user-defined arguments | FEAT | Closed | Custom argument support |
| 5 | #1939 | Implement similar functionality to #1389 | FEAT | Open | Feature parity |
| 6 | #468 | Specify schema for optimization result | FEAT | Open | Optimization output format |
| 7 | #988 | Custom resolver only works sometimes | BUG | Closed | Resolver reliability |
| 8 | #386 | Override loaded config via command line flag | FEAT | Closed | Config override mechanism |
| 9 | #2588 | Unexpected ConfigKeyError while composing config | BUG | Closed | Config composition error |
| 10 | #1830 | Cannot use plugin conf dataclasses in structured config | BUG | Open | Plugin configuration |

---

## Section 23: iterative/dvc (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #2325 | Reconsider gc implementation | FEAT | Closed | Garbage collection redesign |
| 2 | #1691 | Support push/pull/metrics across commits | FEAT | Closed | Cross-commit data management |
| 3 | #2831 | Unexpected error adding files | BUG | Closed | File tracking failure |
| 4 | #2799 | ML experiments and hyperparameter tuning | FEAT | Closed | Experiment management |
| 5 | #7093 | Release 3.0 | DISC | Closed | Major version roadmap |
| 6 | #3069 | Support hashing other than MD5 | FEAT | Open | Hash algorithm extension |
| 7 | #3393 | Introduce hyperparameters and config | DISC | Closed | Configuration management |
| 8 | #1871 | Store whole DAG in one DVC file | FEAT | Closed | Pipeline definition format |
| 9 | #2697 | Using DVC only for dataset management | PERF | Closed | Lightweight usage pattern |
| 10 | #755 | Add scheduler for parallelizing execution jobs | FEAT | Open | Parallel pipeline execution |

---

## Section 24: emdgroup/baybe (10 entries)

Repository Grade: A | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #78 | Access to non-persistent data (acqf functions, model) | FEAT | Closed | Acquisition function introspection |
| 2 | #195 | Minor visual issues in documentation | BUG | Closed | Documentation quality |
| 3 | #592 | Issues with reproducibility when recommending experiments | BUG | Closed | [CRITICAL] BO reproducibility |
| 4 | #192 | Recommendations taking a long time | PERF | Closed | BO recommendation speed |
| 5 | #515 | Campaign gets slower with each measurement added | PERF | Closed | [CRITICAL] Scaling with data size |
| 6 | #365 | OOM: required 338 TB memory for search space | PERF | Closed | [CRITICAL] Combinatorial explosion in search space |
| 7 | #612 | Kernel iteration tests fail due to numerics | BUG | Closed | Numerical precision in kernel |
| 8 | #96 | Printing campaign overly verbose with chemical descriptors | DISC | Closed | Output formatting |
| 9 | #218 | Random seed being set somewhere hidden | FEAT | Closed | Reproducibility control |
| 10 | #200 | Simulation bug in ignore mode | BUG | Closed | Simulation configuration |

---

## Section 25: facebook/Ax (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #1557 | Random vs acquisition function selected candidates | DISC | Closed | Exploration-exploitation trade-off |
| 2 | #768 | Initializing experiment with data outside search space | DISC | Closed | Out-of-bounds data handling |
| 3 | #769 | Allow callable for constraint evaluation | FEAT | Closed | Custom constraint functions |
| 4 | #120 | Using Ax as supplier of candidates | DISC | Closed | Black-box evaluation pattern |
| 5 | #77 | Example for online evaluation | DISC | Closed | Online BO pattern |
| 6 | #771 | Calculate expected improvement for candidate list | FEAT | Closed | Batch acquisition computation |
| 7 | #2532 | Error: Try again with more data | BUG | Closed | Insufficient data handling |
| 8 | #727 | Implementing composition-based optimization | DISC | Closed | Materials composition BO |
| 9 | #731 | Parallel multiobjective optimization | DISC | Closed | Parallel MOBO |
| 10 | #228 | Repeated trials: Sobol fallback needed | BUG | Closed | Duplicate suggestion prevention |

---

## Section 26: atomicarchitects/equiformer (10 entries)

Repository Grade: A | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #9 | logger.py import error | BUG | Open | Module path configuration |
| 2 | #1 | eV to meV unit conversion | DISC | Closed | Energy unit convention |
| 3 | #12 | Reduce model size | FEAT | Closed | Model compression |
| 4 | #16 | Updated environment build | COMPAT | Open | Environment reproducibility |
| 5 | #19 | RuntimeError: invalid value for --gpu-architecture | BUG | Open | CUDA architecture mismatch |
| 6 | #10 | How to train on extended xyz format | DATA | Closed | Custom data format support |
| 7 | #4 | Smooth decrease in L1 loss | DISC | Closed | Training dynamics |
| 8 | #2 | Creating standalone general model repository | DISC | Open | Modular architecture proposal |
| 9 | #15 | Can't reproduce MD17 results | BUG | Closed | [CRITICAL] Benchmark reproducibility |
| 10 | #20 | Support for custom dataset | FEAT | Open | Custom dataset integration |

---

## Section 27: shap/shap (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #295 | Save explainer | FEAT | Closed | Explainer serialization |
| 2 | #38 | Spark version planned? | FEAT | Closed | Distributed computing support |
| 3 | #1215 | TreeExplainer utf-8 codec error for XGBoost | BUG | Closed | Model serialization encoding |
| 4 | #29 | Output value outside [0,1] in binary classification | DISC | Closed | SHAP value interpretation |
| 5 | #213 | SHAP for RNN models | FEAT | Open | Sequential model support |
| 6 | #884 | Error with PySpark GBTClassifier | BUG | Closed | Spark integration bug |
| 7 | #580 | Reshape error for SHAP calculation | BUG | Closed | Array shape handling |
| 8 | #3438 | Add support for more nn.Module layers | FEAT | Open | PyTorch layer coverage |
| 9 | #963 | Convert log-odds explanations to probabilities | DISC | Open | Output transformation |
| 10 | #480 | TreeEnsemble has no attribute values for LightGBM | BUG | Closed | Framework compatibility |

---

## Section 28: Lightning-AI/pytorch-lightning (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #839 | Cross validation feature | FEAT | Closed | K-fold cross validation |
| 2 | #13445 | Improve typing coverage | FEAT | Closed | Type annotation quality |
| 3 | #4612 | Code stuck on initializing DDP with multi-GPU | BUG | Closed | [CRITICAL] DDP initialization deadlock |
| 4 | #12521 | Remove deprecated code after 1.6 release | FEAT | Closed | API cleanup |
| 5 | #10389 | Lightning very slow between epochs vs PyTorch | PERF | Closed | [CRITICAL] Framework overhead |
| 6 | #896 | Unify usage of multiple callbacks | DISC | Closed | Callback architecture |
| 7 | #10914 | Add Exponential Moving Average (EMA) | FEAT | Closed | EMA support |
| 8 | #4420 | NCCL error using DDP and PyTorch 1.7 | BUG | Closed | DDP communication failure |
| 9 | #1136 | Performance loss between 0.6.0 and 0.7.1 | PERF | Closed | Performance regression |
| 10 | #4705 | CUDA OOM when initializing DDP | BUG | Closed | [CRITICAL] DDP memory overhead |

---

## Section 29: wandb/wandb (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #9946 | Network error ConnectTimeout, entering retry loop | DISC | Closed | Network resilience |
| 2 | #4409 | wandb saving fails with soft links | BUG | Closed | Symlink handling |
| 3 | #4929 | wandb.finish() stuck after uploading all data | BUG | Closed | Finalization deadlock |
| 4 | #2981 | Table not updating at each log call | FEAT | Closed | Incremental table logging |
| 5 | #5339 | Importing from MLFlow breaks due to permissions | BUG | Closed | MLFlow migration |
| 6 | #3751 | wandb mobile/desktop app? | FEAT | Open | Platform expansion |
| 7 | #1526 | Run state reported as crashed but still running | BUG | Closed | Run state tracking |
| 8 | #6449 | BrokenPipeError: Errno 32 | BUG | Open | Communication failure |
| 9 | #1192 | Create run with previously deleted run name | FEAT | Open | Run name reuse |
| 10 | #1409 | Error communicating with backend | BUG | Closed | Backend connectivity |

---

## Section 30: deepmodeling/dpgen (10 entries)

Repository Grade: A | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #438 | SSH connection fails, change to local context | BUG | Closed | Remote execution failure |
| 2 | #701 | Undefined key load_ckpt in strict mode | BUG | Closed | Configuration validation |
| 3 | #723 | DP-GEN auto_test properties | DISC | Closed | Automated testing workflow |
| 4 | #1460 | RuntimeError with f_devi NaN in MD trajectory | BUG | Open | [CRITICAL] Force deviation NaN during active learning |
| 5 | #1346 | Error in post_fp_cp2k link | BUG | Closed | CP2K integration |
| 6 | #431 | Multi-task submission on server | FEAT | Closed | Parallel task management |
| 7 | #522 | Job failed for more than 3 times | BUG | Closed | Retry limit handling |
| 8 | #1138 | dpgen auto_test post error with DeePMD | BUG | Closed | Post-processing failure |
| 9 | #1108 | init_bulk: invalid format ValueError | BUG | Closed | Input validation |
| 10 | #528 | TensorFlow InternalError | BUG | Closed | TensorFlow crash |

---

## Section 31: pytorch/captum (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #143 | Support for LRP/DeepTaylor | FEAT | Closed | Layer-wise relevance propagation |
| 2 | #150 | Captum for BERT | FEAT | Closed | Transformer interpretability |
| 3 | #311 | Visualization of BERT hard to understand | DISC | Closed | Attention visualization |
| 4 | #373 | How to interpret BERTSequenceClassification | DISC | Closed | Classification attribution |
| 5 | #510 | Integrated Gradient for Intent Classification and NER | FEAT | Open | NLU interpretability |
| 6 | #282 | Integrated gradients for model with embedding | DISC | Closed | Embedding layer attribution |
| 7 | #439 | Problem with inputs using IG and embedding layers | BUG | Closed | [CRITICAL] IG computation failure with embeddings |
| 8 | #723 | Multimodal input binary classifier with Saliency | DISC | Closed | Multimodal attribution |
| 9 | #246 | Extension to graph model | FEAT | Open | [CRITICAL] GNN interpretability |
| 10 | #663 | DLRM tutorial | DISC | Closed | Recommendation model attribution |

---

## Section 32: mir-group/flare (10 entries)

Repository Grade: A | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #289 | Not showing mgp in installed packages | BUG | Closed | Installation verification |
| 2 | #395 | How to accelerate training with large dataset | PERF | Closed | [CRITICAL] Large dataset training efficiency |
| 3 | #291 | Unable to construct GP model from log file | BUG | Closed | Model reconstruction failure |
| 4 | #219 | How to confirm multithreading setup | DISC | Closed | Parallel execution verification |
| 5 | #110 | Error regarding flare-ace implementation | BUG | Closed | ACE integration bug |
| 6 | #222 | MemoryError in GP training | PERF | Closed | [CRITICAL] Memory scaling for GP |
| 7 | #297 | Magnetization command in DFT script not working | BUG | Closed | DFT integration |
| 8 | #292 | MGP from AIMD_GP freeze in LAMMPS | BUG | Closed | LAMMPS integration freeze |
| 9 | #386 | Python API cannot save trained model (segfault) | BUG | Open | [CRITICAL] Model persistence crash |
| 10 | #246 | Cannot map MGP to OTF-trained GP | BUG | Closed | Model mapping failure |

---

## Section 33: atomistic-machine-learning/schnetpack (10 entries)

Repository Grade: A | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #433 | New SchNetPack is very slow on training | PERF | Closed | [CRITICAL] Performance regression in new version |
| 2 | #165 | Stress tensor implementation request | FEAT | Closed | Stress tensor for MD |
| 3 | #407 | Failing to learn forces for periodic system | BUG | Closed | Periodic system training failure |
| 4 | #226 | Materials Project data handling | DISC | Closed | Dataset pipeline design |
| 5 | #546 | Response properties: error in tensor size | BUG | Closed | Tensor shape mismatch |
| 6 | #567 | Config and model representation question | DISC | Closed | Architecture configuration |
| 7 | #184 | TorchEnvironmentProvider | FEAT | Closed | Environment provider design |
| 8 | #672 | How to reproduce PaiNN results on QM9 and MD17 | DISC | Closed | [CRITICAL] Benchmark reproducibility |
| 9 | #104 | Distances not correct for periodically repeated bulk | BUG | Closed | [CRITICAL] Periodic boundary distance calculation |
| 10 | #192 | CSVHook logging small bug | BUG | Closed | Logging utility bug |

---

## Section 34: gasteigerjo/dimenet (15 entries)

Repository Grade: A | Total Issues Surveyed: 15

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #7 | Periodic DimeNet | FEAT | Closed | Periodic system extension |
| 2 | #17 | Questions about MD17 | DATA | Closed | Benchmark dataset details |
| 3 | #25 | How to plot the image | DISC | Closed | Visualization |
| 4 | #9 | Question about angle calculation | DISC | Closed | [CRITICAL] Angular feature computation |
| 5 | #21 | Detail property values in QM9 | DATA | Closed | QM9 property documentation |
| 6 | #3 | Each single target | DISC | Closed | Multi-target training |
| 7 | #6 | Request for pretrained model | FEAT | Closed | Pretrained model distribution |
| 8 | #29 | Coll dataset units | DATA | Closed | Unit convention |
| 9 | #15 | How to extract final layer vector | FEAT | Closed | Feature extraction |
| 10 | #2 | QM9 dataset | DATA | Closed | Dataset usage |
| 11 | #31 | Reason for linear weight twice | DISC | Closed | Architecture detail |
| 12 | #22 | Creation of custom dataset OMDB | DATA | Closed | Custom dataset creation |
| 13 | #32 | Input issue | BUG | Closed | Input processing bug |
| 14 | #19 | Are loss values scaled? | DISC | Closed | Loss scaling convention |
| 15 | #30 | Incorporating DimeNet++ into LAMMPS | FEAT | Closed | LAMMPS integration |

---

## Section 35: atomistic-machine-learning/schnetpack (PaiNN) (10 entries)

Repository Grade: A | Total Issues Surveyed: 10 (PaiNN-specific within SchNetPack)

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #407 | Failing to learn forces for periodic system | BUG | Closed | Periodic force learning |
| 2 | #226 | Materials Project training | DATA | Closed | MP data pipeline |
| 3 | #567 | Config and model representation question | DISC | Closed | PaiNN configuration |
| 4 | #672 | How to reproduce PaiNN results on QM9/MD17 | DISC | Closed | Reproducibility |
| 5 | #401 | Towards stress in dev branch | FEAT | Closed | Stress tensor development |
| 6 | #507 | Predicting analytical Hessian | FEAT | Closed | Second-order properties |
| 7 | #568 | Question about target name | DISC | Closed | Target convention |
| 8 | #563 | Mismatched tensor sizes | BUG | Closed | Tensor dimension error |
| 9 | #416 | How to improve evaluation metrics | DISC | Closed | Metric optimization |
| 10 | #411 | RuntimeError related to PaiNN training | BUG | Closed | Training crash |

---

## Section 36: ppdebreuck/modnet (15 entries)

Repository Grade: A | Total Issues Surveyed: 15

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #180 | TypeError when using matbench_benchmark | BUG | Closed | [CRITICAL] Matbench integration bug |
| 2 | #69 | Facilities for state variables (temperature, pressure)? | FEAT | Open | Multi-condition prediction |
| 3 | #226 | Is GPU acceleration possible? | DISC | Open | GPU training support |
| 4 | #46 | CompositionOnly featurizer not using oxidation states | BUG | Open | Feature engineering gap |
| 5 | #88 | MODNet for precalculated features | FEAT | Closed | Custom feature input |
| 6 | #228 | Non-stratified splitting in classification | BUG | Open | [CRITICAL] Data splitting bias |
| 7 | #202 | Can't featurize twice without kernel restart | BUG | Closed | State management bug |
| 8 | #81 | Problems featurizing with DeBreuck2020Featurizer | BUG | Closed | Featurizer compatibility |
| 9 | #126 | Complex compositions very slow to featurize | PERF | Open | Featurization performance |
| 10 | #73 | Sample from posterior using Bayesian module | DISC | Open | [CRITICAL] UQ via Bayesian inference |
| 11 | #72 | pip install only works up to 0.1.6 on Colab | COMPAT | Closed | Cloud environment limitation |
| 12 | #71 | Better pinning of requirements | COMPAT | Closed | Dependency management |
| 13 | #34 | Missing features for prediction | DATA | Open | Feature availability |
| 14 | #77 | FitGenetic taking 2.5 hrs instead of 5 min | PERF | Closed | Feature selection performance |
| 15 | #235 | FitGenetic.run return value if refit=0 | BUG | Closed | API edge case |

---

## Section 37: emukit/emukit (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #307 | Error installing dependencies with Python 3.8 | COMPAT | Closed | Build toolchain |
| 2 | #342 | External objective function evaluation in BO | DISC | Closed | Black-box evaluation pattern |
| 3 | #143 | LAPACK dpotrs conversion error | BUG | Closed | Linear algebra backend |
| 4 | #149 | Preferential Bayesian optimization | FEAT | Open | Preference-based BO |
| 5 | #18 | Cost sensitive loop | FEAT | Closed | Cost-aware acquisition |
| 6 | #189 | Emukit for engineering optimization | DISC | Closed | Engineering domain application |
| 7 | #330 | Negative LCB acquisition in batch mode | BUG | Open | [CRITICAL] Batch LCB computation error |
| 8 | #140 | Dependency on GPy and GPyOpt | COMPAT | Closed | Backend dependency |
| 9 | #403 | BO gives flat latent function | DISC | Closed | GP model diagnostic |
| 10 | #279 | Gradients wrong when normalizer=True | BUG | Closed | [CRITICAL] Normalization gradient bug |

---

## Section 38: modAL-python/modAL (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #31 | No module named modAL.models | BUG | Closed | Import naming conflict |
| 2 | #39 | Extend modAL to PyTorch models | FEAT | Open | Deep active learning |
| 3 | #2 | Performance on MNIST not great | PERF | Open | Active learning baseline |
| 4 | #170 | Package name changed in pip repository | COMPAT | Open | Distribution naming |
| 5 | #92 | Vote entropy query strategy | DISC | Closed | Ensemble query design |
| 6 | #11 | Support batch-mode queries | FEAT | Closed | Batch active learning |
| 7 | #28 | Cold start in ranked batch sampling | FEAT | Closed | Initial sample strategy |
| 8 | #22 | Refactoring documentation | DATA | Closed | Documentation quality |
| 9 | #41 | Use different query strategies | FEAT | Closed | Strategy flexibility |
| 10 | #85 | Keras regressor integration issues | BUG | Open | Framework integration |

---

## Section 39: dragonfly/dragonfly (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #42 | Installing dragonfly under PyPy fails | COMPAT | Closed | Alternative Python runtime |
| 2 | #39 | Save and load progress | FEAT | Open | Experiment persistence |
| 3 | #33 | Errors encountered using Dragonfly | BUG | Open | General usage errors |
| 4 | #55 | Issue specifying domain constraints | BUG | Open | Constraint specification |
| 5 | #61 | TensorFlow version for NAS | DISC | Closed | NAS dependency |
| 6 | #32 | Passing integers to minimized function | DISC | Closed | Integer variable handling |
| 7 | #73 | ip_mroute kernel error | BUG | Closed | System-level crash |
| 8 | #66 | Complete documentation for Options and Config | DATA | Open | Documentation gap |
| 9 | #77 | Ask-tell mode infinite loop with n_points > 0 | BUG | Open | [CRITICAL] Infinite loop bug |
| 10 | #60 | Batch experiments within ask-tell interface | FEAT | Open | Parallel ask-tell |

---

## Section 40: divelab/DIG (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #54 | GCNConv has no attribute weight in SubgraphX | BUG | Closed | API migration issue |
| 2 | #55 | GradCAM graph visualization example | DISC | Closed | Visualization tutorial |
| 3 | #169 | ModuleNotFoundError in GNN Explainability tutorial | BUG | Open | Module path error |
| 4 | #77 | QM9 and SphereNet example error | BUG | Closed | Example code bug |
| 5 | #24 | Run GNNExplainer | DISC | Closed | GNNExplainer usage |
| 6 | #39 | Optimal parameter set for SphereNet | DISC | Closed | Hyperparameter guidance |
| 7 | #49 | AttributeError in threedgraph notebook | BUG | Closed | Notebook compatibility |
| 8 | #56 | Shapley value calculation mismatch with SubgraphX paper | BUG | Closed | [CRITICAL] Implementation vs paper discrepancy |
| 9 | #97 | Examples need to be updated | BUG | Closed | Documentation freshness |
| 10 | #142 | GATConv has no attribute weight | BUG | Closed | PyG API change |

---

## Section 41: interpretml/interpret (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #184 | Monotone models | FEAT | Closed | Monotonic constraint support |
| 2 | #137 | Manually define link function | FEAT | Closed | Custom link functions |
| 3 | #460 | Feature selection in EBMs via LASSO post-processing | DISC | Closed | Feature selection strategy |
| 4 | #491 | Feature importance visualization question | DISC | Open | Visualization interpretation |
| 5 | #330 | M1 Apple Silicon support | COMPAT | Closed | ARM architecture compatibility |
| 6 | #444 | Segmentation fault | BUG | Closed | Memory access crash |
| 7 | #1 | Use graphs in Jupyter notebook | DISC | Closed | Interactive visualization |
| 8 | #547 | Add Automatic Piecewise Linear Regression | FEAT | Closed | Algorithm extension |
| 9 | #251 | EBM memory issues | PERF | Closed | [CRITICAL] Memory scaling for large datasets |
| 10 | #172 | Query on show() method | BUG | Closed | API usage |

---

## Section 42: mlflow/mlflow (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #9843 | UI not loading in browser | BUG | Closed | Web UI failure |
| 2 | #6181 | Tracking server not working as proxy | BUG | Closed | Server proxy configuration |
| 3 | #6348 | Feedback on experiment tracking | DISC | Closed | UX design discussion |
| 4 | #3154 | IsADirectoryError with S3 artifact | BUG | Closed | Cloud storage bug |
| 5 | #629 | Proxy uploading of artifacts | FEAT | Closed | Artifact management |
| 6 | #925 | Worker timeout when opening UI | PERF | Closed | UI performance |
| 7 | #9008 | Replace % formatting with f-string | FEAT | Closed | Code modernization |
| 8 | #8306 | No module named pip._vendor.six | BUG | Closed | Dependency resolution |
| 9 | #2050 | Autologging for scikit-learn | FEAT | Closed | Auto-instrumentation |
| 10 | #1517 | UI takes long to load experiments | PERF | Closed | UI scalability |

---

## Section 43: materialsproject/atomate2 (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #750 | Advertising atomate2 paper | DISC | Closed | Community outreach |
| 2 | #486 | Move input sets to pymatgen? | DISC | Closed | Architecture decision |
| 3 | #781 | Custom CHGNet model in PhononFlow throws jsanitize error | BUG | Closed | CHGNet integration |
| 4 | #912 | Database not updating after job finished | BUG | Closed | Job completion tracking |
| 5 | #607 | ElasticMaker breaks when fitting tensor | BUG | Closed | Elastic property workflow |
| 6 | #453 | Should prev job INCAR be merged | DISC | Closed | Configuration inheritance |
| 7 | #659 | fit_elastic_tensor hangs | BUG | Closed | Workflow deadlock |
| 8 | #1038 | More transparency for default parameters | DISC | Open | Parameter documentation |
| 9 | #156 | Standardized job checkpoint/restart/continuation | FEAT | Open | Workflow resilience |
| 10 | #488 | Deduplicate doc strings | FEAT | Closed | Documentation quality |

---

## Section 44: snap-stanford/conformalized-gnn (2 entries)

Repository Grade: A | Total Issues Surveyed: 2

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #2 | Error in running demo.ipynb | BUG | Closed | Tutorial reproduction |
| 2 | #1 | Code seems not to match the paper | BUG | Closed | [CRITICAL] Implementation-paper discrepancy |

---

## Section 45: scikit-activeml/scikit-activeml (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #20 | Implement Classification Mixture Model (CMM) | FEAT | Closed | Probabilistic classifier |
| 2 | #297 | Speed-up AL learning cycle with GPU | PERF | Closed | GPU-accelerated active learning |
| 3 | #162 | Include more classifiers in pool tests | FEAT | Open | Classifier coverage |
| 4 | #182 | Versioning the documentation | FEAT | Closed | Documentation management |
| 5 | #84 | Handling sample_weight for SklearnClassifier | BUG | Closed | Weighted sample support |
| 6 | #40 | Fitting before predicting? | DISC | Closed | Workflow order |
| 7 | #186 | Create new developer guide | FEAT | Closed | Developer documentation |
| 8 | #11 | Fix naming of standard parameters | COMPAT | Closed | API naming convention |
| 9 | #208 | X and y as non-optional arguments to query | DISC | Closed | API design |
| 10 | #272 | Toward scikit-activeml for OpenML | FEAT | Open | Platform integration |

---

## Section 46: hackingmaterials/rocketsled (15 entries)

Repository Grade: B | Total Issues Surveyed: 15

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #11 | Disable duplicate checking for parallel optimization | FEAT | Closed | Parallel BO |
| 2 | #55 | Tutorial rework | FEAT | Closed | Documentation |
| 3 | #12 | Multi-objective optimization | FEAT | Closed | MOBO support |
| 4 | #56 | General comment on code and examples | DISC | Closed | Code quality |
| 5 | #48 | Instance attributes only set in __init__ | BUG | Closed | Code pattern |
| 6 | #118 | Batch prediction without retraining | FEAT | Open | Batch efficiency |
| 7 | #16 | Add tags to MongoDB documents | FEAT | Closed | Metadata management |
| 8 | #21 | Write tests for parallel duplicates | BUG | Closed | Test coverage |
| 9 | #25 | Speed improvements | PERF | Closed | Optimization speed |
| 10 | #26 | Add other optimization algorithms | FEAT | Closed | Algorithm diversity |
| 11 | #35 | Rank Pareto solutions in analysis | FEAT | Closed | Pareto front analysis |
| 12 | #29 | Scale data for optimizers | PERF | Closed | Data normalization for BO |
| 13 | #49 | Why is XZ_new one variable? | DISC | Closed | Code design |
| 14 | #44 | Rename space parameter to dimensions_file | FEAT | Closed | API naming |
| 15 | #42 | Add paper citation | FEAT | Closed | Attribution |

---

## Section 47: dhw059/DenseGNN (2 entries)

Repository Grade: B | Total Issues Surveyed: 2

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #3 | Environment question about kgcnn | DISC | Closed | Dependency setup |
| 2 | #4 | ValueError: Property molecule_feature not defined | BUG | Open | Feature property bug |

---

## Section 48: vgsatorras/egnn (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #4 | Training EGNN on QM9 dataset | DISC | Closed | QM9 training |
| 2 | #1 | Question about AE experiment | DISC | Closed | Autoencoder experiment |
| 3 | #2 | EGNN flow | DISC | Closed | Normalizing flow variant |
| 4 | #3 | In-place operation in coord_model | BUG | Closed | Gradient computation bug |
| 5 | #6 | Graph edges in QM9 experiment | DISC | Closed | Edge construction |
| 6 | #5 | Implementation of Eq4 | DISC | Open | Paper equation implementation |
| 7 | #8 | Velocity updates in N-body system | DISC | Closed | Dynamics simulation |
| 8 | #10 | Experiment results on QM9 | DISC | Closed | Benchmark results |
| 9 | #15 | N-body model velocity not updated | BUG | Closed | Velocity bug |
| 10 | #13 | Environment file | DISC | Open | Environment setup |

---

## Section 49: microsoft/mattersim (10 entries)

Repository Grade: A | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #66 | MD acceleration and LAMMPS integration | FEAT | Open | [CRITICAL] LAMMPS integration for production MD |
| 2 | #82 | No PBC structures | BUG | Closed | Non-periodic system handling |
| 3 | #104 | Unable to run on WSL2 | BUG | Closed | WSL compatibility |
| 4 | #102 | Error running minimal test code | BUG | Closed | Installation verification |
| 5 | #49 | Phonon calculation failure | BUG | Closed | Phonon workflow |
| 6 | #60 | moldyn.py calls NPT when NVT selected | BUG | Open | Ensemble selection bug |
| 7 | #63 | Add fine-tune method | FEAT | Closed | [CRITICAL] Fine-tuning support |
| 8 | #116 | Support NumPy>=2 | COMPAT | Open | NumPy version compatibility |
| 9 | #112 | Buffer dtype mismatch | BUG | Open | Data type casting |
| 10 | #10 | Migrate to GitHub inside Microsoft | DISC | Closed | Repository migration |

---

## Section 50: torchmd/torchmd-net (10 entries)

Repository Grade: A | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #26 | Physics-based priors | FEAT | Open | Prior knowledge integration |
| 2 | #19 | Pre-trained model | DISC | Closed | Model availability |
| 3 | #45 | Optimize equivariant transformer | PERF | Open | [CRITICAL] Transformer inference speed |
| 4 | #6 | Creating custom dataset | DISC | Closed | Data pipeline |
| 5 | #92 | Periodic boundary conditions | FEAT | Closed | PBC implementation |
| 6 | #82 | Unable to fit model | BUG | Closed | Training failure |
| 7 | #77 | NaN when fitting with derivative | BUG | Closed | [CRITICAL] NaN in force training |
| 8 | #96 | Ways to reduce memory use | PERF | Open | Memory optimization |
| 9 | #221 | Allow changing box in models | FEAT | Closed | Variable cell support |
| 10 | #372 | Hardcoded PyTorch version | COMPAT | Closed | Version pinning |

---

## Section 51: CompRhys/aviary (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #54 | Wren: averaging augmented Wyckoff positions inside NN | DISC | Closed | Symmetry averaging |
| 2 | #10 | Add models equivalent to Roost | FEAT | Closed | Model equivalence |
| 3 | #63 | How to predict on new materials | DISC | Closed | Inference workflow |
| 4 | #25 | Separate fit and predict | FEAT | Closed | API design |
| 5 | #72 | Matbench results for Wrenformer? | DISC | Closed | Benchmark comparison |
| 6 | #36 | Git surgery plan | FEAT | Closed | Repository restructuring |
| 7 | #105 | Request for PyPI release | FEAT | Closed | Distribution |
| 8 | #23 | Instructions for custom datasets | DISC | Closed | Data loading guide |
| 9 | #21 | Roost Colab CUDA version issue | COMPAT | Closed | Cloud compatibility |
| 10 | #29 | Refactor for consistent docstrings | FEAT | Closed | Code quality |

---

## Section 52: learningmatter-mit/NeuralForceField (15 entries)

Repository Grade: A | Total Issues Surveyed: 15

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #7 | SpookyNet tutorial | DISC | Closed | Model implementation guide |
| 2 | #9 | DimeNet tutorials cannot run | BUG | Closed | Tutorial reproduction |
| 3 | #15 | Meaning of energy_grad | DISC | Closed | Gradient convention |
| 4 | #12 | Energy dataset generation | DISC | Closed | Training data creation |
| 5 | #21 | Material science modules missing | COMPAT | Closed | Feature completeness |
| 6 | #2 | Please add a license | FEAT | Closed | Open-source licensing |
| 7 | #14 | SchNet ethanol dataset energy info | DISC | Closed | Dataset documentation |
| 8 | #16 | Missing c6ab.npy | DATA | Closed | Required data file |
| 9 | #4 | elec_config.json not installed correctly | BUG | Closed | Installation path |
| 10 | #6 | GraphAttention message uses only ij contributions | BUG | Closed | [CRITICAL] Asymmetric message passing |
| 11 | #29 | Train models using ASE extended XYZ format | DISC | Open | Data format support |
| 12 | #5 | Memory leak in PaiNN module | PERF | Closed | [CRITICAL] Memory management |
| 13 | #24 | CHGNet wrapper loads incorrect path | BUG | Open | CHGNet integration |
| 14 | #25 | Handle update from ASE | COMPAT | Open | ASE version compatibility |
| 15 | #22 | Download the full dataset | DATA | Closed | Data access |

---

## Section 53: Open-Catalyst-Project/ocp (10 entries)

Repository Grade: A | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #888 | Stress property not implemented | BUG | Closed | Missing property computation |
| 2 | #563 | Non-deterministic calculation results | BUG | Closed | [CRITICAL] Reproducibility failure |
| 3 | #1568 | NaN while computing loss | BUG | Closed | [CRITICAL] NaN loss in large-scale training |
| 4 | #1201 | Adaptation to UMA | DISC | Closed | Universal model adaptation |
| 5 | #242 | Torch package issue | COMPAT | Closed | PyTorch version |
| 6 | #170 | Unable to run SchNet demo | BUG | Closed | Tutorial setup |
| 7 | #981 | OCPNEB loading checkpoint failure | BUG | Closed | Checkpoint loading |
| 8 | #936 | Trouble loading EquiformerV2 with fairchem 1.3.0 | BUG | Closed | Model-framework version |
| 9 | #1618 | NaN in second derivative | BUG | Closed | Force derivative stability |
| 10 | #1566 | OC25 metadata | DATA | Closed | Dataset documentation |

---

## Section 54: google/uncertainty-baselines (10 entries)

Repository Grade: A | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #91 | Hyperparams for batch ensemble on CIFAR | DISC | Closed | Batch ensemble configuration |
| 2 | #336 | Package tries to install all tf-nightly versions | BUG | Closed | Dependency bloat |
| 3 | #264 | Can't reproduce MIMO accuracy on CIFAR-100 | DISC | Closed | [CRITICAL] Reproducibility gap |
| 4 | #349 | Help with BERT SNGP on CLINC | DISC | Closed | NLP UQ application |
| 5 | #286 | Reproducing OOD scores CIFAR-10 vs SVHN | DISC | Open | OOD detection benchmark |
| 6 | #201 | Reproducing accuracy | DISC | Closed | General reproducibility |
| 7 | #329 | RuntimeError running SNGP | BUG | Open | SNGP execution error |
| 8 | #288 | Questions about SNGP prediction | DISC | Open | SNGP interpretation |
| 9 | #775 | Example code for Mahalanobis score in TF-2 | FEAT | Closed | UQ metric implementation |
| 10 | #258 | SNGP Laplace RF precision update inconsistent | BUG | Closed | [CRITICAL] Theoretical correctness |

---

## Section 55: y0ast/deterministic-uncertainty-quantification (11 entries)

Repository Grade: A | Total Issues Surveyed: 11

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #10 | Error when running train_duq_fm.py | BUG | Closed | Training script bug |
| 2 | #3 | Reproduce results | DISC | Closed | Reproducibility |
| 3 | #7 | Feasibility for object detection | FEAT | Closed | Task extension |
| 4 | #9 | Can DUQ estimate aleatoric uncertainty? | DISC | Open | [CRITICAL] UQ capability scope |
| 5 | #4 | Tensors do not require grad | BUG | Closed | Gradient setup |
| 6 | #8 | Semantic segmentation code | FEAT | Closed | Task extension |
| 7 | #11 | Which variable represents uncertainty in two moons? | DISC | Open | Output interpretation |
| 8 | #6 | Questions about paper and codes | DISC | Closed | Implementation clarification |
| 9 | #2 | notMNIST mat file missing | DATA | Closed | Dataset requirement |
| 10 | #1 | Replication of toy example for deep ensemble | DISC | Closed | Baseline comparison |
| 11 | #5 | A question about the paper | DISC | Closed | Theoretical clarification |

---

## Section 56: hackingmaterials/automatminer (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #253 | Can't pickle RLock when saving pipe | BUG | Closed | Serialization failure |
| 2 | #179 | Featurization takes too long | PERF | Closed | [CRITICAL] Feature extraction bottleneck |
| 3 | #88 | Package structure suggestions | FEAT | Closed | Architecture improvement |
| 4 | #80 | Dataset storage needs improvement | FEAT | Closed | Data management |
| 5 | #82 | TestAllFeaturizers breaks on new featurizer | BUG | Closed | Test fragility |
| 6 | #146 | No module named automatminer.featurize | BUG | Closed | Import error |
| 7 | #49 | Testing takes unacceptably long | PERF | Closed | Test performance |
| 8 | #77 | Using automatminer models as featurizers | FEAT | Open | Feature reuse |
| 9 | #334 | Remove target from autofeaturizer | FEAT | Closed | API simplification |
| 10 | #87 | Rename AutoML segment | FEAT | Closed | Naming convention |

---

## Section 57: WMD-group/SMACT (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #448 | SMACT property prediction module | FEAT | Open | Property prediction extension |
| 2 | #55 | Charge neutrality check function issue | BUG | Closed | Validation logic |
| 3 | #8 | Tests | FEAT | Closed | Test infrastructure |
| 4 | #34 | Advanced Pauling rules for screening? | DISC | Closed | Screening methodology |
| 5 | #7 | Refactoring __init__.py | FEAT | Closed | Code organization |
| 6 | #32 | Oxidation states dataset outdated | DATA | Closed | Data currency |
| 7 | #47 | Gradient information for chemical filters | FEAT | Closed | Differentiable screening |
| 8 | #121 | Oxidation states for nitrides | DATA | Closed | Domain coverage |
| 9 | #378 | smact_validity function slow | PERF | Closed | Validation performance |
| 10 | #9 | Include Faraday oxidation model data | DATA | Closed | Data source expansion |

---

## Section 58: materialsproject/mp-api (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #676 | Parity with legacy MPRester functions | FEAT | Open | API migration |
| 2 | #67 | Entries API | FEAT | Closed | Data access |
| 3 | #820 | MPRester error | BUG | Closed | Client-side bug |
| 4 | #46 | Tasks API request | FEAT | Closed | Task data access |
| 5 | #924 | No way to obtain charge density from task_id | BUG | Closed | [CRITICAL] Data regression |
| 6 | #825 | Circular dependencies between MP packages | COMPAT | Closed | Dependency architecture |
| 7 | #922 | ValidationError for MoleculeSummaryDoc | BUG | Closed | Schema validation |
| 8 | #964 | Inconsistent results of materials query | BUG | Open | Data consistency |
| 9 | #831 | Import of MPRester fails with AttributeError | BUG | Closed | Import error |
| 10 | #363 | get_entries_in_chemsys 6x slower performance | PERF | Closed | API performance regression |

---

## Section 59: janosh/matbench-discovery (10 entries)

Repository Grade: A | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #39 | Benchmark design questions | DISC | Closed | Evaluation methodology |
| 2 | #179 | Error running compiled_wbm_test_set.py | BUG | Closed | Script execution |
| 3 | #40 | Obtain E_above_hull predictions | DISC | Closed | Property prediction |
| 4 | #281 | Include Gruneisen parameter in benchmark | FEAT | Closed | Thermal property evaluation |
| 5 | #22 | Fetching fails with UnicodeDecodeError | BUG | Closed | Data encoding |
| 6 | #53 | PyTorch module and virtual environment | DISC | Closed | Environment setup |
| 7 | #139 | WBM filtering fails on entries_old_corr | BUG | Closed | Data filtering logic |
| 8 | #145 | Unpinned pymatviz causes incompatibility | COMPAT | Closed | Dependency management |
| 9 | #230 | StructureMatcher parameters impact geo-opt metrics | DISC | Closed | [CRITICAL] Metric sensitivity to matching |
| 10 | #259 | Unable to reproduce accuracy of eSEN | DISC | Closed | Reproducibility |

---

## Section 60: hackingmaterials/atomate (10 entries)

Repository Grade: B | Total Issues Surveyed: 10

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #295 | StaticFW doesn't pass VASP input set params correctly | BUG | Closed | VASP parameter passing |
| 2 | #19 | Default procedures for optimizations | FEAT | Open | Optimization defaults |
| 3 | #268 | Duplicate defaults in atomate | BUG | Open | Configuration redundancy |
| 4 | #291 | Atomate v2 workflow organization | FEAT | Open | Architecture planning |
| 5 | #277 | Better presets for DOS calculations | FEAT | Open | Workflow defaults |
| 6 | #289 | Atomate v2 unit testing strategy | FEAT | Open | Testing architecture |
| 7 | #230 | Rethink atomate organization | DISC | Open | Structural redesign |
| 8 | #213 | Optimized NEB workflow | FEAT | Open | Diffusion pathway |
| 9 | #146 | POTCAR modifications ignored in elastic workflow | BUG | Closed | Workflow parameter |
| 10 | #206 | get_settings error with builders | BUG | Closed | Builder configuration |

---

## Section 61: vertaix/LLM-Prop (3 entries)

Repository Grade: B | Total Issues Surveyed: 3

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #1 | Training dataset only 50 samples, poor MAE | DATA | Closed | [CRITICAL] Small dataset limitation |
| 2 | #2 | Trained model availability | FEAT | Open | Model distribution |
| 3 | #3 | Environment setup issue | COMPAT | Open | Environment reproducibility |

---

## Section 62: vertaix/LLM4Mat-Bench (2 entries)

Repository Grade: B | Total Issues Surveyed: 2

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #1 | Incomplete datasets | DATA | Closed | Dataset coverage |
| 2 | #2 | MAD:MAE metric implementation question | BUG | Closed | Metric definition |

---

## Section 63: learningmatter-mit/Atomistic-Adversarial-Attacks (1 entry)

Repository Grade: A | Total Issues Surveyed: 1

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #1 | Running through graph a second time | BUG | Closed | Graph re-execution |

---

## Section 64: learningmatter-mit/matex (1 entry)

Repository Grade: B | Total Issues Surveyed: 1

| # | Issue | Title | Type | Status | Key Learning |
|---|-------|-------|------|--------|--------------|
| 1 | #1 | Trouble reproducing results for ESOL/FreeSolv/BACE | BUG | Closed | [CRITICAL] Benchmark reproducibility |

---

## ATLAS-Critical Issue Summary

Issues with direct impact on ATLAS development, extracted across all surveyed repositories:

| Priority | Source | Issue | Impact on ATLAS |
|----------|--------|-------|-----------------|
| P0 | MACE #555 | e3nn version pin removal | `atlas/models/equivariant.py` depends on e3nn; version conflict blocks integration |
| P0 | ALIGNN #93 | OOM with 28K CIFs | RTX 3060 12GB is more constrained; graph precomputation strategy needed |
| P0 | NequIP #315 | RTX 4080 compatibility | Ada Lovelace GPU family; affects RTX 4060 deployment |
| P0 | Matbench #110 | UQ stats on website | Validates ATLAS UQ contribution path within benchmark ecosystem |
| P0 | Matbench #42 | Record uncertainties with predictions | Confirms UQ integration is a recognized gap in Matbench |
| P0 | UQ-Toolbox #21 | Recalibration API access | Direct integration target for ATLAS UQ calibration pipeline |
| P0 | UQ-Toolbox #93 | Penalized Brier Score / Log Loss | Novel calibration metrics for ATLAS evaluation |
| P1 | ALIGNN #90 | No multi-GPU support | Single GPU training limitation |
| P1 | NequIP #92 | Precision trade-off | train float32 / eval float64 strategy |
| P1 | ALIGNN #115 | GPU 0% utilization | Force field training efficiency |
| P1 | ALIGNN #54 | Data normalization | Key to reproducing baseline results |
| P1 | MatGL #264 | Fine-tuned model degradation | Transfer learning failure mode |
| P1 | MatGL #243 | DGL/PyTorch version conflict | Dependency management for graph libraries |
| P1 | GPyTorch #864 | Negative variance from FixedNoiseGaussianLikelihood | Numerical instability in ATLAS surrogate models |
| P1 | BoTorch #179 | Cholesky decomposition numerical issue | Affects ATLAS acquisition function computation |
| P1 | BoTorch #2035 | Constrained MOBO for material discovery | Direct reference for ATLAS active learning design |
| P1 | Pymatgen #3888 | CrystalNN self-image bonding bug | Affects ATLAS graph construction from pymatgen |
| P1 | CHGNet #79 | Energy jump at zero bond_graph_len | Potential energy surface discontinuity |
| P1 | fairchem #563 | Non-deterministic calculation results | Reproducibility concern for benchmarking |
| P1 | DeePMD #1656 | Different LAMMPS versions produce different trajectories | Cross-version reproducibility |
| P1 | CGCNN #37 | Size mismatch loading pretrained models | ATLAS baseline model loading |
| P1 | JARVIS #304 | Modulus discrepancy: elastic tensor vs database | Training data quality verification |
| P2 | e3nn #112 | NaN in kernel derivative | Gradient stability for equivariant training |
| P2 | Chemprop #355 | GPU not utilized | Common GPU utilization debugging patterns |
| P2 | BoTorch #798 | Cholesky CUDA OOM | GPU memory management for Bayesian optimization |
| P2 | Pymatgen #1746 | POTCAR hash integrity | Data provenance for DFT-derived training sets |
| P2 | UQ-Toolbox #85 | ENCE implementation | Calibration metric for ATLAS evaluation |
| P2 | UQ-Toolbox #81 | GPU acceleration for UQ metrics | Performance for large-scale UQ evaluation |

---

## Collection Statistics

| Repository | Entries | BUG | FEAT | DISC | PERF | COMPAT | DATA |
|------------|---------|-----|------|------|------|--------|------|
| ACEsuit/mace | 20 | 7 | 3 | 4 | 2 | 3 | 1 |
| mir-group/nequip | 20 | 8 | 3 | 3 | 1 | 3 | 2 |
| usnistgov/alignn | 15 | 5 | 1 | 3 | 2 | 2 | 2 |
| e3nn/e3nn | 20 | 4 | 5 | 4 | 2 | 4 | 1 |
| materialsproject/matbench | 15 | 2 | 4 | 3 | 0 | 2 | 2 |
| materialsvirtuallab/matgl | 15 | 3 | 4 | 4 | 0 | 3 | 0 |
| chemprop/chemprop | 15 | 4 | 3 | 5 | 0 | 0 | 1 |
| cornellius-gp/gpytorch | 20 | 5 | 4 | 7 | 1 | 2 | 0 |
| pytorch/botorch | 20 | 10 | 5 | 3 | 1 | 0 | 0 |
| materialsproject/pymatgen | 20 | 10 | 2 | 2 | 0 | 2 | 3 |
| CederGroupHub/chgnet | 15 | 8 | 2 | 1 | 1 | 3 | 0 |
| pyg-team/pytorch_geometric | 20 | 9 | 5 | 2 | 0 | 4 | 0 |
| txie-93/cgcnn | 10 | 2 | 1 | 2 | 1 | 2 | 2 |
| FAIR-Chem/fairchem | 15 | 10 | 0 | 3 | 0 | 2 | 1 |
| deepmodeling/deepmd-kit | 15 | 9 | 1 | 0 | 3 | 3 | 0 |
| usnistgov/jarvis | 15 | 4 | 1 | 3 | 0 | 2 | 5 |
| hackingmaterials/matminer | 10 | 3 | 4 | 0 | 1 | 1 | 1 |
| uncertainty-toolbox | 35 | 2 | 15 | 13 | 2 | 3 | 0 |
| Repository | Entries | BUG | FEAT | DISC | PERF | COMPAT | DATA |
|------------|---------|-----|------|------|------|--------|------|
| ACEsuit/mace | 20 | 7 | 3 | 4 | 2 | 3 | 1 |
| mir-group/nequip | 20 | 8 | 3 | 3 | 1 | 3 | 2 |
| usnistgov/alignn | 15 | 5 | 1 | 3 | 2 | 2 | 2 |
| e3nn/e3nn | 20 | 4 | 5 | 4 | 2 | 4 | 1 |
| materialsproject/matbench | 15 | 2 | 4 | 3 | 0 | 2 | 2 |
| materialsvirtuallab/matgl | 15 | 3 | 4 | 4 | 0 | 3 | 0 |
| chemprop/chemprop | 15 | 4 | 3 | 5 | 0 | 0 | 1 |
| cornellius-gp/gpytorch | 20 | 5 | 4 | 7 | 1 | 2 | 0 |
| pytorch/botorch | 20 | 10 | 5 | 3 | 1 | 0 | 0 |
| materialsproject/pymatgen | 20 | 10 | 2 | 2 | 0 | 2 | 3 |
| CederGroupHub/chgnet | 15 | 8 | 2 | 1 | 1 | 3 | 0 |
| pyg-team/pytorch_geometric | 20 | 9 | 5 | 2 | 0 | 4 | 0 |
| txie-93/cgcnn | 10 | 2 | 1 | 2 | 1 | 2 | 2 |
| FAIR-Chem/fairchem | 15 | 10 | 0 | 3 | 0 | 2 | 1 |
| deepmodeling/deepmd-kit | 15 | 9 | 1 | 0 | 3 | 3 | 0 |
| usnistgov/jarvis | 15 | 4 | 1 | 3 | 0 | 2 | 5 |
| hackingmaterials/matminer | 10 | 3 | 4 | 0 | 1 | 1 | 1 |
| uncertainty-toolbox | 35 | 2 | 15 | 13 | 2 | 3 | 0 |
| mir-group/allegro | 10 | 6 | 1 | 2 | 1 | 0 | 0 |
| SINGROUP/dscribe | 10 | 3 | 1 | 2 | 1 | 3 | 0 |
| aamini/evidential-deep-learning | 10 | 5 | 2 | 2 | 0 | 0 | 1 |
| facebookresearch/hydra | 10 | 4 | 5 | 0 | 0 | 0 | 0 |
| iterative/dvc | 10 | 1 | 6 | 2 | 1 | 0 | 0 |
| emdgroup/baybe | 10 | 4 | 2 | 1 | 3 | 0 | 0 |
| facebook/Ax | 10 | 2 | 2 | 6 | 0 | 0 | 0 |
| atomicarchitects/equiformer | 10 | 3 | 2 | 3 | 0 | 1 | 1 |
| shap/shap | 10 | 4 | 4 | 2 | 0 | 0 | 0 |
| Lightning-AI/pytorch-lightning | 10 | 3 | 4 | 1 | 2 | 0 | 0 |
| wandb/wandb | 10 | 5 | 3 | 1 | 0 | 0 | 0 |
| deepmodeling/dpgen | 10 | 8 | 1 | 1 | 0 | 0 | 0 |
| pytorch/captum | 10 | 1 | 4 | 4 | 0 | 0 | 0 |
| **Total** | **445** | **153** | **100** | **88** | **23** | **45** | **23** |

---

## Next Collection Targets

| Priority | Repository | Target Issues | Status |
|----------|------------|---------------|--------|
| P0 | CederGroupHub/chgnet | 15 | Done |
| P0 | pyg-team/pytorch_geometric | 20 | Done |
| P0 | txie-93/cgcnn | 10 | Done |
| P1 | FAIR-Chem/fairchem | 15 | Done |
| P1 | deepmodeling/deepmd-kit | 15 | Done |
| P1 | uncertainty-toolbox/uncertainty-toolbox | 35 | Done |
| P1 | hackingmaterials/matminer | 10 | Done |
| P1 | usnistgov/jarvis | 15 | Done |
| P2 | mir-group/allegro | 10 | Pending |
| P2 | microsoft/mattersim | 10 | Pending |
| P2 | SINGROUP/dscribe | 10 | Pending |
| P2 | Remaining 60+ repos | 5-10 each | Pending |
| | **Projected Total** | **~700+** | |

---

## Search Syntax Reference

```
# Per-repository search
repo:ACEsuit/mace is:issue sort:comments-desc
repo:mir-group/nequip is:issue sort:comments-desc
repo:usnistgov/alignn is:issue sort:comments-desc
repo:e3nn/e3nn is:issue sort:comments-desc

# Cross-repository topic searches
"out of distribution" is:issue language:Python
"uncertainty quantification" is:issue
"crystal property" is:issue
"memory" OR "OOM" is:issue language:Python topic:materials
"e3nn" "compatibility" is:issue
"matbench" is:issue
"active learning" is:issue language:Python topic:materials
```

---
---

# 中文版：晶體 GNN 與不確定性量化研究之開源議題（Issue）調研

> **專案**：ATLAS — 原子結構自適應訓練與學習  
> **作者**：Zhong  
> **日期**：2026-02-27  
> **總條目**：754  
> **涵蓋倉庫**：59 個（64 個調研段落）  
> **狀態**：所有追蹤倉庫已完成掃描

---

## 方法論

### 收集策略

議題來源為配套文件（`repo_tracker.md`）中已鑑定的 90 個倉庫，依研究相關度分級排序。每個倉庫的議題按評論數（降序）排列，以篩選出社群最活躍討論的主題。少數倉庫（教學 repo、awesome-list 等）因無 issue 而未納入。

### 分類體系

| 類別 | 代碼 | 定義 |
|------|------|------|
| 缺陷報告 | BUG | 已確認的瑕疵、非預期行為、程式崩潰 |
| 功能需求 | FEAT | 新功能、API 擴充、能力延伸 |
| 討論 | DISC | 使用問題、設計決策、架構辯論 |
| 效能 | PERF | 速度、記憶體、GPU 使用率、可擴展性 |
| 相容性 | COMPAT | 版本衝突、硬體支援、依賴問題 |
| 資料 | DATA | 資料集存取、格式、完整性、預處理 |

### 相關度標註

標註 [關鍵] 的議題對 ATLAS 開發有直接影響，實作階段需特別關注。

---

## 與 ATLAS 直接相關的關鍵議題

跨所有調研倉庫，對 ATLAS 開發有直接影響的議題摘要：

| 優先級 | 來源 | 議題 | 對 ATLAS 的影響 |
|--------|------|------|--------------------|
| P0 | MACE #555 | 解除 e3nn 版本鎖定 | `atlas/models/equivariant.py` 依賴 e3nn；版本衝突阻礙整合 |
| P0 | ALIGNN #93 | 28K CIF 記憶體溢位 | RTX 3060 12GB 更受限；需圖預計算策略 |
| P0 | NequIP #315 | RTX 4080 相容性 | Ada Lovelace GPU 系列；影響 RTX 4060 部署 |
| P0 | Matbench #110 | 網站加入 UQ 統計 | 驗證 ATLAS UQ 貢獻路徑在基準生態中的定位 |
| P0 | Matbench #42 | 預測附帶不確定性記錄 | 確認 UQ 整合是 Matbench 公認的缺口 |
| P0 | UQ-Toolbox #38 | 可重新校準 API | ATLAS 預測校準管線的直接參考 |
| P0 | UQ-Toolbox #33 | 懲罰 Brier 分數 | ATLAS 評估框架的候選指標 |
| P1 | ALIGNN #90 | 無多 GPU 支援 | 單 GPU 訓練限制 |
| P1 | NequIP #92 | 精度權衡 | 訓練 float32 / 評估 float64 策略 |
| P1 | ALIGNN #115 | GPU 使用率 0% | 力場訓練效率問題 |
| P1 | MatGL #264 | 微調後模型退化 | 遷移學習失效模式 |
| P1 | GPyTorch #864 | 固定噪音似然產生負方差 | ATLAS 代理模型的數值穩定性 |
| P1 | BoTorch #2035 | 材料發現的約束多目標最佳化 | ATLAS 主動學習設計的直接參考 |
| P1 | Pymatgen #3888 | CrystalNN 自映像鍵結 bug | 影響 ATLAS 從 pymatgen 建構圖 |
| P1 | CHGNet #115 | 能量不連續 | 影響 ATLAS 力場預測穩定性 |
| P1 | fairchem #726 | 結果不可重現 | 訓練可重現性影響實驗設計 |
| P1 | DeePMD #3584 | 跨版本模型差異 | 模型格式向後相容性 |
| P1 | JARVIS #304 | 彈性模量差異 | 訓練資料品質驗證 |
| P1 | MatterSim #63 | 微調方法 | MLIP 微調參考架構 |
| P1 | OCP #563 | 非確定性計算結果 | 大規模可重現性 |
| P2 | e3nn #112 | 核函數導數出現 NaN | 等變模型訓練的梯度穩定性 |
| P2 | BoTorch #798 | Cholesky 分解 CUDA 記憶體溢位 | 貝葉斯最佳化的 GPU 記憶體管理 |
| P2 | Pymatgen #1746 | POTCAR 雜湊值完整性 | DFT 衍生訓練集的資料出處 |
| P2 | UQ-Toolbox #85 | ENCE 指標實作 | ATLAS 評估的校準指標 |
| P2 | UQ-Toolbox #81 | UQ 指標 GPU 加速 | 大規模 UQ 評估效能 |
| P2 | Evidential-DL #18 | 認知不確定性行為異常 | UQ 輸出解讀挑戰 |
| P2 | Evidential-DL #5 | 損失值變 NaN | 證據式訓練的數值穩定性 |
| P2 | SchNetPack #104 | 周期性結構距離計算錯誤 | 周期邊界距離的正確性 |

---

## 收集統計

### 依倉庫分類

| 倉庫 | 條目數 | BUG | FEAT | DISC | PERF | COMPAT | DATA |
|------|--------|-----|------|------|------|--------|------|
| ACEsuit/mace | 20 | 7 | 3 | 4 | 2 | 3 | 1 |
| mir-group/nequip | 20 | 8 | 3 | 3 | 1 | 3 | 2 |
| usnistgov/alignn | 15 | 5 | 1 | 3 | 2 | 2 | 2 |
| e3nn/e3nn | 20 | 4 | 5 | 4 | 2 | 4 | 1 |
| materialsproject/matbench | 15 | 2 | 4 | 3 | 0 | 2 | 2 |
| materialsvirtuallab/matgl | 15 | 3 | 4 | 4 | 0 | 3 | 0 |
| chemprop/chemprop | 15 | 4 | 3 | 5 | 0 | 0 | 1 |
| cornellius-gp/gpytorch | 20 | 5 | 4 | 7 | 1 | 2 | 0 |
| pytorch/botorch | 20 | 10 | 5 | 3 | 1 | 0 | 0 |
| materialsproject/pymatgen | 20 | 10 | 2 | 2 | 0 | 2 | 3 |
| CederGroupHub/chgnet | 15 | 8 | 2 | 1 | 1 | 3 | 0 |
| pyg-team/pytorch_geometric | 20 | 9 | 5 | 2 | 0 | 4 | 0 |
| txie-93/cgcnn | 10 | 2 | 1 | 2 | 1 | 2 | 2 |
| FAIR-Chem/fairchem | 15 | 10 | 0 | 3 | 0 | 2 | 1 |
| deepmodeling/deepmd-kit | 15 | 9 | 1 | 0 | 3 | 3 | 0 |
| usnistgov/jarvis | 15 | 4 | 1 | 3 | 0 | 2 | 5 |
| hackingmaterials/matminer | 10 | 3 | 4 | 0 | 1 | 1 | 1 |
| uncertainty-toolbox | 35 | 2 | 15 | 13 | 2 | 3 | 0 |
| mir-group/allegro | 10 | 6 | 1 | 2 | 1 | 0 | 0 |
| SINGROUP/dscribe | 10 | 3 | 1 | 2 | 1 | 3 | 0 |
| aamini/evidential-deep-learning | 10 | 5 | 2 | 2 | 0 | 0 | 1 |
| facebookresearch/hydra | 10 | 4 | 5 | 0 | 0 | 0 | 0 |
| iterative/dvc | 10 | 1 | 6 | 2 | 1 | 0 | 0 |
| emdgroup/baybe | 10 | 4 | 2 | 1 | 3 | 0 | 0 |
| facebook/Ax | 10 | 2 | 2 | 6 | 0 | 0 | 0 |
| atomicarchitects/equiformer | 10 | 3 | 2 | 3 | 0 | 1 | 1 |
| shap/shap | 10 | 4 | 4 | 2 | 0 | 0 | 0 |
| Lightning-AI/pytorch-lightning | 10 | 3 | 4 | 1 | 2 | 0 | 0 |
| wandb/wandb | 10 | 5 | 3 | 1 | 0 | 0 | 0 |
| deepmodeling/dpgen | 10 | 8 | 1 | 1 | 0 | 0 | 0 |
| pytorch/captum | 10 | 1 | 4 | 4 | 0 | 0 | 0 |
| mir-group/flare | 10 | 7 | 0 | 1 | 2 | 0 | 0 |
| schnetpack | 10 | 4 | 2 | 3 | 1 | 0 | 0 |
| gasteigerjo/dimenet | 15 | 1 | 4 | 5 | 0 | 0 | 5 |
| schnetpack (PaiNN) | 10 | 3 | 2 | 4 | 0 | 0 | 1 |
| ppdebreuck/modnet | 15 | 6 | 2 | 2 | 2 | 2 | 1 |
| emukit/emukit | 10 | 3 | 2 | 3 | 0 | 2 | 0 |
| modAL-python/modAL | 10 | 2 | 4 | 1 | 1 | 1 | 1 |
| dragonfly/dragonfly | 10 | 4 | 2 | 2 | 0 | 1 | 1 |
| divelab/DIG | 10 | 7 | 0 | 3 | 0 | 0 | 0 |
| interpretml/interpret | 10 | 2 | 3 | 3 | 1 | 1 | 0 |
| mlflow/mlflow | 10 | 4 | 3 | 1 | 2 | 0 | 0 |
| materialsproject/atomate2 | 10 | 4 | 2 | 4 | 0 | 0 | 0 |
| conformalized-gnn | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| scikit-activeml | 10 | 1 | 5 | 2 | 1 | 1 | 0 |
| hackingmaterials/rocketsled | 15 | 2 | 10 | 2 | 1 | 0 | 0 |
| dhw059/DenseGNN | 2 | 1 | 0 | 1 | 0 | 0 | 0 |
| vgsatorras/egnn | 10 | 2 | 0 | 8 | 0 | 0 | 0 |
| microsoft/mattersim | 10 | 5 | 2 | 1 | 0 | 1 | 0 |
| torchmd/torchmd-net | 10 | 2 | 3 | 2 | 2 | 1 | 0 |
| CompRhys/aviary | 10 | 0 | 5 | 4 | 0 | 1 | 0 |
| NeuralForceField | 15 | 4 | 2 | 6 | 1 | 2 | 3 |
| Open-Catalyst-Project/ocp | 10 | 7 | 0 | 1 | 0 | 1 | 1 |
| google/uncertainty-baselines | 10 | 3 | 1 | 6 | 0 | 0 | 0 |
| DUQ | 11 | 2 | 2 | 6 | 0 | 0 | 1 |
| hackingmaterials/automatminer | 10 | 3 | 5 | 0 | 2 | 0 | 0 |
| WMD-group/SMACT | 10 | 1 | 4 | 1 | 1 | 0 | 3 |
| materialsproject/mp-api | 10 | 5 | 3 | 0 | 1 | 1 | 0 |
| janosh/matbench-discovery | 10 | 3 | 1 | 4 | 0 | 1 | 0 |
| hackingmaterials/atomate | 10 | 4 | 6 | 1 | 0 | 0 | 0 |
| vertaix/LLM-Prop | 3 | 0 | 1 | 0 | 0 | 1 | 1 |
| vertaix/LLM4Mat-Bench | 2 | 1 | 0 | 0 | 0 | 0 | 1 |
| Atomistic-Adversarial-Attacks | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| learningmatter-mit/matex | 1 | 1 | 0 | 0 | 0 | 0 | 0 |

### 總計摘要

| 指標 | 數值 |
|------|------|
| 已調研倉庫數 | 59 |
| 調研段落數 | 64 |
| 總議題條目 | 754 |
| 標註 [關鍵] 的議題 | 28 (P0: 7, P1: 13, P2: 8) |

### 依類別統計

| 類別 | 數量 | 佔比 |
|------|------|------|
| BUG（缺陷報告）| ~230 | 30.5% |
| FEAT（功能需求）| ~160 | 21.2% |
| DISC（討論）| ~170 | 22.5% |
| PERF（效能）| ~40 | 5.3% |
| COMPAT（相容性）| ~55 | 7.3% |
| DATA（資料）| ~35 | 4.6% |

---

## 收集完成狀態

| 優先級 | 倉庫 | 議題數 | 狀態 |
|--------|------|--------|------|
| P0 | 所有 P0 核心倉庫（15 個） | ~220 | 已完成 |
| P1 | 所有 P1 重要倉庫（20 個） | ~280 | 已完成 |
| P2 | 所有 P2 輔助倉庫（24 個） | ~254 | 已完成 |
| — | 無 issue 的倉庫（教學/索引類） | 0 | 不適用 |
| | **總計** | **754** | **全部完成** |

