# Open Dataset Survey for Crystal GNN and Uncertainty Quantification Research

> **Project**: ATLAS — Accelerated Topological Learning And Screening  
> **Author**: Zhong  
> **Date**: 2026-02-27  
> **Total Entries**: 25  
> **Status**: Phase 1 Inventory Complete

---

## Methodology

### Selection Criteria

Each dataset was evaluated across 6 dimensions relevant to the ATLAS research programme:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Scale | High | Number of materials, structures, or data points available |
| Property Coverage | High | Range and diversity of computed or measured properties |
| Data Quality | High | Provenance (DFT functional, convergence settings), validation status |
| Accessibility | High | Open access, API availability, download ease |
| Community Adoption | Medium | Usage in published benchmarks, citation frequency |
| ATLAS Relevance | Medium | Direct applicability to crystal GNN training, UQ evaluation, or OOD testing |

### Grading System

| Grade | Criteria | Recommended Action |
|-------|----------|--------------------|
| **A** | >= 3 high-weight dimensions satisfied | Prioritize for integration; detailed schema review |
| **B** | >= 2 high-weight or >= 4 medium-weight | Evaluate for secondary experiments or cross-validation |

### Field Definitions

| Field | Description |
|-------|-------------|
| ID | Category letter + sequential number (e.g., A-01) |
| Name | Official dataset name |
| Maintainer | Hosting institution or research group |
| Scale | Approximate number of entries, structures, or calculations |
| Cost | Free / Free with registration / Paid |
| Provenance | DFT functional, code, and quality level |
| Access Method | API, bulk download, Python package, or web interface |
| ATLAS Relevance | Specific connection to ATLAS project modules, methods, or goals |
| Integration Status | Not Started / Evaluated / Integrated / Excluded |
| Notes | Observations and caveats recorded during evaluation |

---

## Category A: Primary DFT-Computed Crystal Property Databases (8 entries)

### A-01 | JARVIS-DFT

- **Maintainer**: NIST (Kamal Choudhary)
- **Grade**: A
- **Scale**: ~76,000 3D materials, ~50 computed properties; ~1,000 2D materials
- **Cost**: Free (public domain, US government)
- **Provenance**: VASP + OptB88vdW functional; well-documented convergence criteria
- **Access Method**: `jarvis-tools` Python package; REST API; bulk JSON/CSV download
- **Properties**: Formation energy, bandgap (OPT/MBJ), elastic constants, dielectric function, solar efficiency, phonon DOS, exfoliation energy (2D)
- **ATLAS Relevance**: Primary training dataset. Already integrated as the default data source in `atlas/data/`. Shares data lineage with ALIGNN benchmark results.
- **Integration Status**: Integrated
- **Notes**: Version pinning required; property coverage varies by material.

### A-02 | Materials Project (MP)

- **Maintainer**: Lawrence Berkeley National Laboratory (LBNL) / DOE
- **Grade**: A
- **Scale**: ~200,000 inorganic materials
- **Cost**: Free (requires API key registration)
- **Provenance**: VASP + PBE/PBE+U; GGA-level with Hubbard U corrections for transition metals
- **Access Method**: `mp-api` Python package; REST API v3; bulk download via MPContribs
- **Properties**: Formation energy, bandgap, density, elastic tensor, piezoelectric tensor, magnetic ordering, electronic DOS/band structure
- **ATLAS Relevance**: Cross-database OOD experiments (train on JARVIS, test on MP). Largest curated DFT database for inorganic crystals.
- **Integration Status**: Not Started
- **Notes**: Requires free account at materialsproject.org. Data format differs from JARVIS; conversion utility needed.

### A-03 | OQMD (Open Quantum Materials Database)

- **Maintainer**: Northwestern University (Chris Wolverton group)
- **Grade**: A
- **Scale**: ~1,000,000 entries
- **Cost**: Free
- **Provenance**: VASP + PBE/PBE+U; standardized calculation parameters
- **Access Method**: `qmpy` Python package; REST API; bulk SQL dump download
- **Properties**: Formation energy, stability (hull distance), band structure, total energy
- **ATLAS Relevance**: Largest single-functional DFT database. Scale experiments and cross-database generalization testing.
- **Integration Status**: Not Started
- **Notes**: Property coverage narrower than JARVIS/MP but much larger scale.

### A-04 | AFLOW (Automatic Flow for Materials Discovery)

- **Maintainer**: Duke University (Stefano Curtarolo group)
- **Grade**: A
- **Scale**: ~3,500,000 materials entries; 817M+ calculated properties
- **Cost**: Free
- **Provenance**: VASP; standardized high-throughput settings (AFLOW Standard)
- **Access Method**: REST API (`aflowlib`); bulk download; AFLOW Python SDK
- **Properties**: Formation energy, bandgap, elastic properties, magnetic properties, Bader charges, thermal properties
- **ATLAS Relevance**: Cross-database domain-shift experiments. Largest freely available DFT repository.
- **Integration Status**: Not Started
- **Notes**: Data schema differs significantly from JARVIS/MP; requires dedicated parser.

### A-05 | Alexandria

- **Maintainer**: KU Leuven (Geoffrey Hautier group)
- **Grade**: B
- **Scale**: ~5,000,000 entries
- **Cost**: Free
- **Provenance**: VASP + PBE; PBEsol for selected subsets
- **Access Method**: Bulk download (Materials Cloud Archive); API in development
- **Properties**: Total energy, formation energy, stability, electronic structure
- **ATLAS Relevance**: Newest and potentially largest open DFT database. Good for evaluating model generalization to novel structures.
- **Integration Status**: Not Started
- **Notes**: Relatively recent; community validation ongoing.

### A-06 | GNoME (Graph Networks for Materials Exploration)

- **Maintainer**: Google DeepMind
- **Grade**: A
- **Scale**: ~2,200,000 materials; ~384,000 confirmed stable
- **Cost**: Free
- **Provenance**: ML-predicted stability verified by DFT (r2SCAN functional for subset)
- **Access Method**: Bulk download via Google Cloud; integrated with Materials Project
- **Properties**: Stability, formation energy, crystal structure
- **ATLAS Relevance**: Largest set of predicted stable crystals. Excellent for evaluating UQ on ML-generated data vs DFT-verified data.
- **Integration Status**: Not Started
- **Notes**: Many entries are ML-predicted, not DFT-verified. Useful for UQ calibration experiments.

### A-07 | NOMAD (Novel Materials Discovery)

- **Maintainer**: NOMAD Centre of Excellence / Max Planck Society
- **Grade**: B
- **Scale**: ~100,000,000+ individual calculations; ~12M unique systems
- **Cost**: Free
- **Provenance**: Mixed (multiple DFT codes: VASP, Quantum ESPRESSO, FHI-aims, etc.); FAIR-compliant metadata
- **Access Method**: REST API; NOMAD Archive download; NOMAD Oasis (local deployment)
- **Properties**: Heterogeneous — depends on uploaded calculation; total energy, forces, electronic structure
- **ATLAS Relevance**: FAIR data reference. Mixed-provenance data for robustness testing.
- **Integration Status**: Not Started
- **Notes**: Data heterogeneity requires careful filtering. Raw calculation files rather than curated property tables.

### A-08 | Materials Cloud

- **Maintainer**: EPFL / Swiss NCCR MARVEL
- **Grade**: B
- **Scale**: Variable (hosts many individual datasets); ~680,000 curated 3D structures
- **Cost**: Free
- **Provenance**: Varies by dataset; many use Quantum ESPRESSO + PBE
- **Access Method**: AiiDA Python framework; Materials Cloud Archive; REST API
- **Properties**: Dataset-dependent; includes band structure, phonon, and thermodynamic properties
- **ATLAS Relevance**: Curated datasets for specific applications. AiiDA workflow reference.
- **Integration Status**: Not Started
- **Notes**: Acts as a data platform hosting multiple independent datasets.

---

## Category B: Benchmark and Evaluation Datasets (5 entries)

### B-01 | Matbench

- **Maintainer**: Materials Project / LBNL (Alex Dunn, Anubhav Jain)
- **Grade**: A
- **Scale**: 13 standardized tasks; sample sizes from 312 to 132,000
- **Cost**: Free
- **Provenance**: Curated subsets from MP, JARVIS, and other sources
- **Access Method**: `matbench` Python package (installable via pip)
- **Properties**: Formation energy, bandgap, dielectric constant, elastic properties (per-task)
- **ATLAS Relevance**: Mandatory benchmark for ATLAS. Provides standardized train/test splits and leaderboard comparison.
- **Integration Status**: Not Started
- **Notes**: Must use official `matbench` API for fair comparison. Nested cross-validation protocol.

### B-02 | Matbench Discovery

- **Maintainer**: Janosh Riebesell (Cambridge / Materials Project)
- **Grade**: A
- **Scale**: WBM dataset (~257,000 structures for stability prediction)
- **Cost**: Free
- **Provenance**: DFT-relaxed structures from WBM (Wang-Bocarsly-Meschel) dataset
- **Access Method**: `matbench-discovery` Python package; HuggingFace
- **Properties**: E_above_hull (thermodynamic stability), crystal structure
- **ATLAS Relevance**: State-of-the-art materials discovery benchmark. Tests whether ATLAS + UQ can identify stable crystals.
- **Integration Status**: Not Started
- **Notes**: Interactive leaderboard at matbench-discovery.materialsproject.org.

### B-03 | QM9

- **Maintainer**: University of Basel (Raghunathan Ramakrishnan, O. Anatole von Lilienfeld)
- **Grade**: B
- **Scale**: ~134,000 molecules (not crystals) — C, H, O, N, F with up to 9 heavy atoms
- **Cost**: Free
- **Provenance**: B3LYP/6-31G(2df,p) (DFT hybrid functional)
- **Access Method**: Bulk download (various mirrors); PyG built-in; SchNetPack built-in
- **Properties**: 12 quantum-chemical properties (HOMO, LUMO, dipole, etc.)
- **ATLAS Relevance**: Standard molecular GNN benchmark. Not directly applicable to crystals but useful for architecture validation.
- **Integration Status**: Not Started
- **Notes**: Molecular, not crystalline. Use for model architecture sanity checks only.

### B-04 | MD17 / rMD17 / MD22

- **Maintainer**: TU Berlin (Klaus-Robert Muller group) / University of Stuttgart
- **Grade**: B
- **Scale**: MD17: ~1M conformations for 10 molecules; MD22: 7 larger systems
- **Cost**: Free
- **Provenance**: DFT-computed energies and forces along MD trajectories (PBE / CCSD(T) for rMD17)
- **Access Method**: Bulk download; built-in loaders in SchNetPack, TorchMD-NET
- **Properties**: Energy, atomic forces per conformation
- **ATLAS Relevance**: Force-field training benchmark. Evaluating energy-force consistency in ATLAS MLIP module.
- **Integration Status**: Not Started
- **Notes**: Molecular systems, not periodic crystals. Useful for force prediction benchmarks.

### B-05 | ANI-1x / ANI-1ccx

- **Maintainer**: University of Florida (Olexandr Isayev group)
- **Grade**: B
- **Scale**: ANI-1x: ~5M conformations; ANI-1ccx: ~500K conformations
- **Cost**: Free
- **Provenance**: ANI-1x: wB97X/6-31G*; ANI-1ccx: CCSD(T)/CBS extrapolation
- **Access Method**: Bulk download (figshare / GitHub)
- **Properties**: Energy, atomic forces
- **ATLAS Relevance**: Large-scale force-field benchmark with coupled-cluster reference data.
- **Integration Status**: Not Started
- **Notes**: Organic molecules only (C, H, N, O). Not periodic.

---

## Category C: Trajectory and Force-Field Datasets (4 entries)

### C-01 | MPtrj (Materials Project Trajectories)

- **Maintainer**: Materials Project / Microsoft Research
- **Grade**: A
- **Scale**: ~1,600,000 structures with DFT energies, forces, and stresses
- **Cost**: Free
- **Provenance**: VASP relaxation trajectories from Materials Project
- **Access Method**: HuggingFace dataset; MP API
- **Properties**: Energy, forces, stress tensors per ionic step
- **ATLAS Relevance**: Primary training data for MLIP. Direct compatibility with CHGNet and MACE pre-training pipelines.
- **Integration Status**: Not Started
- **Notes**: CHGNet and M3GNet were both trained on this dataset.

### C-02 | Open Catalyst 2020/2022 (OC20/OC22)

- **Maintainer**: Meta FAIR + Carnegie Mellon University
- **Grade**: B
- **Scale**: OC20: ~1,300,000 DFT relaxations; OC22: oxide systems
- **Cost**: Free
- **Provenance**: VASP + RPBE for OC20; VASP + PBE+U for OC22
- **Access Method**: `ocpmodels` / `fairchem` Python; bulk download
- **Properties**: Energy, forces, relaxed structures for surface adsorption
- **ATLAS Relevance**: Low priority — catalysis-focused, not bulk crystal properties. However, architectures (EquiformerV2, GemNet) from OCP are relevant.
- **Integration Status**: Excluded (domain mismatch)
- **Notes**: Valuable for pre-training equivariant architectures, not for crystal property prediction.

### C-03 | SPICE

- **Maintainer**: Microsoft Research (Peter Eastman)
- **Grade**: B
- **Scale**: ~1,100,000 conformations for drug-like molecules
- **Cost**: Free
- **Provenance**: wB97M-D3(BJ)/def2-TZVPPD (high-accuracy DFT)
- **Access Method**: Bulk download (Zenodo); OpenMM integration
- **Properties**: Energy, forces, dipole moments
- **ATLAS Relevance**: Low — pharmaceutical molecules, not crystals. Reference for high-accuracy DFT benchmarking methodology.
- **Integration Status**: Excluded (domain mismatch)
- **Notes**: Useful as methodological reference for data quality standards.

### C-04 | AIS-Square Datasets

- **Maintainer**: DeePModeling / AI for Science Institute
- **Grade**: B
- **Scale**: Multiple datasets; includes alloy, semiconductor, and electrolyte data
- **Cost**: Free
- **Provenance**: Varies; typically VASP
- **Access Method**: AIS-Square platform download
- **Properties**: Energy, forces, virial tensors for MD simulation
- **ATLAS Relevance**: Deep Potential training data. Reference for active learning data generation workflows.
- **Integration Status**: Not Started
- **Notes**: Growing platform; datasets curated for DeePMD ecosystem.

---

## Category D: Experimental and Crystallographic Databases (4 entries)

### D-01 | Crystallography Open Database (COD)

- **Maintainer**: Vilnius University / international consortium
- **Grade**: B
- **Scale**: ~500,000 crystal structure entries
- **Cost**: Free (open access)
- **Provenance**: Experimentally determined crystal structures (X-ray, neutron diffraction)
- **Access Method**: Web search; REST API; bulk CIF download; MySQL dump
- **Properties**: Crystal structure (unit cell, space group, atomic positions) — no computed properties
- **ATLAS Relevance**: Source of experimental crystal structures for graph construction validation. OOD testing (DFT-relaxed vs experimental structures).
- **Integration Status**: Not Started
- **Notes**: Structures only, no target properties. Must cross-reference with DFT databases for labels.

### D-02 | ICSD (Inorganic Crystal Structure Database)

- **Maintainer**: FIZ Karlsruhe
- **Grade**: B
- **Scale**: ~280,000 crystal structure entries
- **Cost**: Paid (institutional subscription required)
- **Provenance**: Experimentally determined; considered gold standard for inorganic crystal structures
- **Access Method**: Web interface; bulk export for subscribers
- **Properties**: Crystal structure, bibliographic data
- **ATLAS Relevance**: Gold-standard experimental structures. MP uses ICSD as starting points for DFT relaxations.
- **Integration Status**: Excluded (paid)
- **Notes**: Not free. Listed here for reference only — many free databases derive from ICSD.

### D-03 | Materials Data Facility (MDF)

- **Maintainer**: Argonne National Laboratory / University of Chicago
- **Grade**: B
- **Scale**: ~150 published datasets (variable sizes)
- **Cost**: Free
- **Provenance**: Mixed (experimental and computational)
- **Access Method**: Globus data publication platform; REST API; Foundry Python package
- **Properties**: Dataset-dependent; includes mechanical, thermal, and electronic properties
- **ATLAS Relevance**: Data publication platform. Useful for discovering niche datasets (e.g., experimental validation sets).
- **Integration Status**: Not Started
- **Notes**: Acts as a data publication platform rather than a unified database.

### D-04 | JARVIS-FF / JARVIS-ML / JARVIS-STM / JARVIS-DFT (2D)

- **Maintainer**: NIST
- **Grade**: A
- **Scale**: ~1,000 2D materials; ~2,000 force-field entries; 30K+ ML predictions
- **Cost**: Free
- **Provenance**: OptB88vdW (DFT-2D); classical force fields (FF); ML predictions
- **Access Method**: `jarvis-tools` Python; REST API
- **Properties**: Exfoliation energy, bandgap, elastic constants (2D-specific); force-field parameters
- **ATLAS Relevance**: 2D materials sub-domain. Complementary to 3D JARVIS-DFT for cross-dimensional transfer experiments.
- **Integration Status**: Not Started
- **Notes**: Smaller datasets but unique 2D focus.

---

## Category E: Specialized and Emerging Datasets (4 entries)

### E-01 | C2DB (Computational 2D Materials Database)

- **Maintainer**: Technical University of Denmark (Kristian Thygesen group)
- **Grade**: B
- **Scale**: ~4,000 2D materials; ~15,000 monolayers screened
- **Cost**: Free
- **Provenance**: GPAW + PBE/HSE06
- **Access Method**: Web interface; bulk download via Atomic Simulation Environment (ASE)
- **Properties**: Bandgap, magnetic properties, stability, phonon spectrum, optical properties
- **ATLAS Relevance**: 2D materials UQ experiments. Can test whether ATLAS UQ generalizes to dimensionally different structures.
- **Integration Status**: Not Started
- **Notes**: High-quality curated dataset for 2D materials discovery.

### E-02 | Quantum-ESPRESSO Benchmarks (SSSP / PseudoDojo)

- **Maintainer**: EPFL (Nicola Marzari group) / Universite catholique de Louvain
- **Grade**: B
- **Scale**: SSSP: curated pseudopotentials for all elements; PseudoDojo: comprehensive verification datasets
- **Cost**: Free
- **Provenance**: Quantum ESPRESSO / ABINIT DFT calculations
- **Access Method**: Materials Cloud; PseudoDojo website
- **Properties**: Convergence data, Delta values, phonon frequencies
- **ATLAS Relevance**: Reference for DFT data quality assessment. Useful for understanding uncertainty in DFT labels themselves.
- **Integration Status**: Not Started
- **Notes**: Methodology reference rather than training dataset.

### E-03 | OPTIMADE-Accessible Databases

- **Maintainer**: OPTIMADE Consortium (cross-institutional)
- **Grade**: B
- **Scale**: Federated access to ~30 databases (MP, AFLOW, OQMD, NOMAD, COD, etc.)
- **Cost**: Free
- **Provenance**: Varies by provider
- **Access Method**: Unified REST API (OPTIMADE specification); `optimade-python-tools` package
- **Properties**: Crystal structure (standardized format); properties vary by provider
- **ATLAS Relevance**: Unified query interface for cross-database experiments without writing per-database parsers.
- **Integration Status**: Not Started
- **Notes**: API specification, not a database itself. Provides federated access.

### E-04 | WBM (Wang-Bocarsly-Meschel) Dataset

- **Maintainer**: Princeton University (compiled by Matbench Discovery)
- **Grade**: A
- **Scale**: ~257,000 crystal structures with DFT-computed stability
- **Cost**: Free
- **Provenance**: VASP + PBE; enumerated from known stable ternary/quaternary systems
- **Access Method**: Via `matbench-discovery` package; HuggingFace
- **Properties**: E_above_hull, crystal structure (CIF), composition
- **ATLAS Relevance**: Primary test set for Matbench Discovery benchmark. Critical for evaluating ATLAS stability prediction with UQ.
- **Integration Status**: Not Started
- **Notes**: Used as the official test set in Matbench Discovery leaderboard.

---

## ATLAS Integration Strategy

### Priority Roadmap

| Priority | Dataset | Purpose | Timeline |
|----------|---------|---------|----------|
| **P0** | JARVIS-DFT | Primary training set (already integrated) | Integrated |
| **P0** | Matbench | Standardized benchmark evaluation | Month 1 |
| **P0** | Matbench Discovery + WBM | Stability prediction benchmark | Month 1 |
| **P1** | Materials Project | Cross-database transfer / OOD experiments | Month 2 |
| **P1** | MPtrj | MLIP force-field training | Month 2 |
| **P1** | GNoME | UQ calibration on ML-generated vs DFT-verified data | Month 3 |
| **P2** | OQMD | Scale experiments; cross-DB generalization | Month 3 |
| **P2** | AFLOW | Cross-database OOD testing (train JARVIS, test AFLOW) | Month 3 |
| **P2** | COD | Experimental vs DFT structure comparison | Month 4 |
| **P3** | NOMAD, Alexandria, C2DB | Specialized experiments; 2D transfer | As needed |
| **P3** | OPTIMADE | Federated cross-DB queries | As needed |

### Data Acquisition Notes

| Dataset | Registration Required | Download Size | Special Requirements |
|---------|----------------------|---------------|---------------------|
| JARVIS-DFT | No | ~2 GB | `pip install jarvis-tools` |
| Matbench | No | ~500 MB | `pip install matbench` |
| Materials Project | Yes (free API key) | ~5 GB (full) | Register at materialsproject.org |
| MPtrj | No | ~15 GB | HuggingFace or MP API |
| OQMD | No | ~3 GB (SQL dump) | MySQL or `qmpy` |
| AFLOW | No | Variable (API-based) | REST API queries |
| GNoME | No | ~10 GB | Google Cloud download |
| COD | No | ~8 GB (full CIF dump) | MySQL dump or rsync |
| Matbench Discovery | No | ~2 GB | `pip install matbench-discovery` |

---
---

# 中文版：晶體 GNN 與不確定性量化研究之開放資料集調研

> **專案**：ATLAS — 原子結構自適應訓練與學習  
> **作者**：Zhong  
> **日期**：2026-02-27  
> **總條目**：25  
> **涵蓋類別**：5 類（主要 DFT 資料庫、基準資料集、軌跡/力場、實驗晶體學、新興專用）  
> **狀態**：第一階段盤點完成

---

## 方法論

### 篩選準則

每個資料集依 6 個維度進行評估：

| 維度 | 權重 | 說明 |
|------|------|------|
| 規模 | 高 | 可用的材料、結構或資料點數量 |
| 性質覆蓋度 | 高 | 計算或量測性質的範圍與多樣性 |
| 資料品質 | 高 | 來源（DFT 泛函、收斂設定）、驗證狀態 |
| 可存取性 | 高 | 開放存取、API 可用性、下載便利性 |
| 社群採用度 | 中 | 在已發表基準中的使用頻率、引用次數 |
| ATLAS 相關度 | 中 | 對晶體 GNN 訓練、UQ 評估或 OOD 測試的直接適用性 |

### 評級系統

| 等級 | 標準 | 建議動作 |
|------|------|----------|
| **A** | 滿足 >= 3 個高權重維度 | 優先整合；詳細結構審查 |
| **B** | 滿足 >= 2 個高權重或 >= 4 個中權重 | 評估用於次要實驗或交叉驗證 |

---

## 與 ATLAS 直接相關的關鍵資料集

| 優先級 | 資料集 | 規模 | 用途 | 時程 |
|--------|--------|------|------|------|
| P0 | JARVIS-DFT | ~76K 材料 | 主訓練集（已整合） | 已完成 |
| P0 | Matbench | 13 個標準任務 | 標準基準評估 | 第 1 個月 |
| P0 | Matbench Discovery + WBM | ~257K 結構 | 穩定性預測基準 | 第 1 個月 |
| P1 | Materials Project | ~200K 材料 | 跨資料庫遷移 / OOD 實驗 | 第 2 個月 |
| P1 | MPtrj | ~1.6M 結構+力 | MLIP 力場訓練 | 第 2 個月 |
| P1 | GNoME | ~2.2M 材料 | ML 生成 vs DFT 驗證的 UQ 校準 | 第 3 個月 |
| P2 | OQMD | ~1M 條目 | 規模實驗；跨 DB 泛化 | 第 3 個月 |
| P2 | AFLOW | ~3.5M 條目 | 跨資料庫 OOD 測試 | 第 3 個月 |
| P2 | COD | ~500K 結構 | 實驗 vs DFT 結構比較 | 第 4 個月 |
| P3 | NOMAD, Alexandria, C2DB | 各異 | 專用實驗；2D 遷移 | 按需 |

---

## 收集統計

### 依類別

| 類別 | 條目數 | 免費 | 付費 |
|------|--------|------|------|
| A. 主要 DFT 計算資料庫 | 8 | 8 | 0 |
| B. 基準與評估資料集 | 5 | 5 | 0 |
| C. 軌跡與力場資料集 | 4 | 4 | 0 |
| D. 實驗與晶體學資料庫 | 4 | 3 | 1 (ICSD) |
| E. 新興專用資料集 | 4 | 4 | 0 |
| **合計** | **25** | **24** | **1** |

### 資料取得摘要

| 資料集 | 需註冊 | 下載大小 | 特殊需求 |
|--------|--------|----------|----------|
| JARVIS-DFT | 否 | ~2 GB | `pip install jarvis-tools` |
| Matbench | 否 | ~500 MB | `pip install matbench` |
| Materials Project | 是（免費 API key） | ~5 GB | 至 materialsproject.org 註冊 |
| MPtrj | 否 | ~15 GB | HuggingFace 或 MP API |
| OQMD | 否 | ~3 GB | MySQL 或 `qmpy` |
| AFLOW | 否 | 依查詢而定 | REST API |
| GNoME | 否 | ~10 GB | Google Cloud 下載 |
| COD | 否 | ~8 GB | MySQL dump 或 rsync |
| Matbench Discovery | 否 | ~2 GB | `pip install matbench-discovery` |

---

## Provenance and Credibility Audit

> This section documents the data generation method, environmental constraints, and methodological limitations for each dataset. These factors directly affect training label quality and must be accounted for in ATLAS UQ design.

### Summary of Data Generation Methods

| Category | Primary Method | Key Limitation |
|----------|---------------|----------------|
| A. DFT Crystal DBs | High-throughput DFT (VASP/QE) | ~0 K, ideal crystals, functional-dependent |
| B. Benchmarks | Curated DFT + experimental subsets | Benchmark abstraction, not raw physics |
| C. Trajectory/Force | DFT relaxation trajectories | Off-equilibrium frames, mixed protocol |
| D. Experimental | X-ray/neutron diffraction | Structure only, no unified supervised labels |
| E. Specialized | Mixed (DFT, ML, FF) | Narrower scope, domain-specific assumptions |

### Common Constraints for HT-DFT Crystal Datasets

All high-throughput DFT crystal property datasets share these systematic limitations:

| Constraint | Description | Impact on ATLAS |
|-----------|-------------|-----------------|
| **Temperature** | Most properties computed at ~0 K (ground state / static calculations). No finite-temperature entropy, phase transitions, or defect equilibria. | Predictions may not transfer to operational temperatures |
| **Pressure** | Typically 0 GPa or specific cell relaxation protocol. No residual stress, thin-film constraint, or high-pressure phases | Phase stability rankings may differ under real conditions |
| **Structural Idealization** | Perfect periodic crystals. Defects, grain boundaries, amorphous regions, surfaces, and impurities are absent | Real materials behaviour governed by defects is not captured |
| **XC Functional** | DFT labels depend on exchange-correlation functional (PBE, PBE+U, SCAN, r2SCAN, OptB88vdW, etc.). Bandgap systematically underestimated by PBE | Label noise floor = functional choice difference |
| **Magnetic States** | +U correction and initial magnetic configurations affect energetics. Convergence to local minima can change stability rankings | Must treat functional/+U as mandatory metadata |
| **Versioning** | Database updates change numerical values for same materials. Must pin version/DOI | Reproducibility requires explicit version tracking |

### Per-Dataset Provenance Audit

#### Category A: Primary DFT Databases

| ID | Method | Settings | Verified | Key Constraints |
|----|--------|----------|----------|-----------------|
| A-01 JARVIS-DFT | DFT (VASP) | OptB88vdW; PREC=Accurate; forces < 0.001 eV/Å; E_tol = 10⁻⁷ eV | ✓ NIST docs | vdW functional → good for layered/2D but not standard for all systems; property coverage uneven across subsets |
| A-02 Materials Project | DFT (VASP) | PBE / PBE+U / r2SCAN (since v2022.10.28); PAW pseudopotentials; mixing scheme for thermodynamics | ✓ MP docs | +U choices vary by element; older pseudopotentials still used for some elements; r2SCAN added recently |
| A-03 OQMD | DFT (VASP via qmpy) | PBE + PAW; standardized workflow | ✓ qmpy docs | Narrower property coverage than MP; protocol differs from MP/AFLOW → cross-DB domain shift |
| A-04 AFLOW | DFT (VASP) | PBE + AFLOW Standard | ✓ AFLOW docs | Highly automated; different protocol from MP/OQMD → cross-DB training learns workflow fingerprints |
| A-05 Alexandria | DFT | Mixed functionals (PBE, PBEsol, SCAN) across datasets | ✓ Materials Cloud | Must treat functional as mandatory metadata; version expansion ongoing |
| A-06 GNoME | ML generation + DFT verification | GNN proposals → DFT stability checks for subsets | ✓ DeepMind/Nature 2023 | **Selection bias**: active learning preferentially samples certain structural spaces; not all entries have equal DFT verification strength; compositional distribution differs from MP |
| A-07 NOMAD | Multi-code archive | Heterogeneous uploads (VASP, QE, FHI-aims, etc.) | ✓ FAIRmat | **Extreme heterogeneity**: must filter by code/functional/convergence before training |
| A-08 Materials Cloud | Platform-hosted | Per-dataset; commonly with AiiDA provenance graphs | ✓ EPFL docs | Not a single-protocol DB; credibility depends on selected sub-dataset |

#### Category B: Benchmarks

| ID | Method | Verified | Key Constraints |
|----|--------|----------|-----------------|
| B-01 Matbench | Mixed DFT + experimental | ✓ | Benchmark abstraction — must follow its split protocol; mixing DFT and experimental labels |
| B-02 Matbench Discovery + WBM | DFT stability screening | ✓ | Tests prospective utility, not just MAE; domain shift and ranking metrics matter |
| B-03 QM9 | QC B3LYP/6-31G(2df,p) | ✓ | Molecular, not crystal; elements CHONF only; architecture sanity check only |
| B-04 MD17/rMD17/MD22 | MD trajectories + DFT/CCSD(T) | ✓ | Gas-phase molecules; rMD17 re-computed at higher accuracy; sampling temperature shapes distribution |
| B-05 ANI-1x/ANI-1ccx | DFT / CCSD(T) molecular | ✓ | Organic molecules only; good for UQ calibration prototyping, not crystal training |

#### Category C: Trajectory / Force-Field

| ID | Method | Verified | Key Constraints |
|----|--------|----------|-----------------|
| C-01 MPtrj | DFT relaxation frames (PBE/GGA+U) | ✓ | Contains many **off-equilibrium structures** (not ground states only); great for MLIP, different distribution from static property datasets |
| C-02 OC20/OC22 | DFT surface+adsorbate calculations | ✓ | Surface/catalysis domain — **severe domain mismatch** with bulk crystal property prediction |
| C-03 SPICE | QC ωB97M-D3(BJ)/def2-TZVPPD | ✓ | Molecular/biochemistry; specific sampling protocol (500 K MD) shapes conformer distribution |
| C-04 AIS-Square / DeePMD | Per-dataset DFT + DP-GEN | ✓ | Must verify per-dataset: system, DFT settings, generation workflow; "same format" ≠ "can mix" |

#### Category D: Experimental / Crystallographic

| ID | Method | Verified | Key Constraints |
|----|--------|----------|-----------------|
| D-01 COD | Experimental diffraction | ✓ | Structure only, no standardized target properties; variable quality/occupancy handling |
| D-02 ICSD | Experimental diffraction (gold standard) | ✓ | **Paid — institutional subscription required**; not usable without license |
| D-03 MDF | Platform (mixed experimental/computational) | ✓ | Heterogeneous; metadata inconsistency is common |
| D-04 JARVIS-FF/ML/STM | Classical FF + ML predictions | ✓ | FF data is empirical potential-based (narrow applicability); ML predictions are not first-principles |

#### Category E: Specialized / Emerging

| ID | Method | Verified | Key Constraints |
|----|--------|----------|-----------------|
| E-01 C2DB | DFT GPAW (PBE/HSE06) | ✓ | 2D-specific; different distribution from 3D bulk → useful for dimension-shift OOD testing |
| E-02 SSSP / PseudoDojo | Pseudopotential benchmarks | ✓ | **Not training data** — quantifies DFT label uncertainty sources (Δ-factor, convergence) |
| E-03 OPTIMADE | Federated API standard | ✓ | Solves access unification, **not** physics/protocol unification |
| E-04 WBM | Mixed (part of Matbench Discovery) | ✓ | Benchmark-defined collection, not a natural distribution |

### Data Governance Recommendations for ATLAS

#### 1. Mandatory Provenance Metadata

Every dataset used in ATLAS training must record:

| Field | Example |
|-------|---------|
| DFT Code | VASP 6.3.2 |
| XC Functional | PBE+U (U_Ti = 4.0 eV) |
| Pseudopotential | PAW PBE 54 |
| k-mesh / ENCUT | Γ-centered 8×8×8 / 520 eV |
| Convergence | E_tol = 10⁻⁶ eV, F < 0.01 eV/Å |
| Magnetic Treatment | Spin-polarized, ferromagnetic init |
| SOC | No |
| Database Version + DOI | MP v2024.12.18 / DOI:10.xxxx |

#### 2. Three-Layer Training/Evaluation Protocol

| Layer | Purpose | Datasets |
|-------|---------|----------|
| Layer-1: Homogeneous HT-DFT | Main training | JARVIS-DFT or MP single-functional subset |
| Layer-2: Cross-DB OOD | Domain shift testing | MP ↔ AFLOW ↔ OQMD ↔ Alexandria |
| Layer-3: Experimental | Distribution stress-test | COD structures with re-computed DFT labels |

#### 3. Protocol Shift as UQ Lower Bound

> The formation energy difference for the same material across MP, OQMD, and AFLOW (due to different DFT settings) defines the **label noise floor** — this is the minimum uncertainty your model should report.

---
---

## 來源可信度與方法限制審核

> 本節記錄每個資料集的資料生成方法、環境限制與方法論假設。這些因素直接影響訓練標籤品質，必須納入 ATLAS UQ 設計考量。

### 資料生成方法總結

| 類別 | 主要方法 | 核心限制 |
|------|----------|----------|
| A. DFT 晶體資料庫 | 高通量 DFT (VASP/QE) | ~0 K、理想晶體、依賴泛函選擇 |
| B. 基準資料集 | 整理後的 DFT + 實驗子集 | 基準抽象化，非原始物理資料 |
| C. 軌跡/力場 | DFT 弛豫軌跡 | 含非平衡態結構、混合 protocol |
| D. 實驗資料 | X 射線/中子繞射 | 僅結構，無統一監督式標籤 |
| E. 專用資料 | 混合（DFT、ML、力場） | 範圍窄、特定領域假設 |

### HT-DFT 晶體資料集的共通系統性限制

| 限制 | 說明 | 對 ATLAS 的影響 |
|------|------|-----------------|
| **溫度** | 多數性質在 ~0 K 基態/靜態計算。缺少有限溫度熵、相變、缺陷平衡 | 預測可能無法轉移至操作溫度 |
| **壓力** | 通常 0 GPa 或特定 cell relaxation protocol | 真實條件下相穩定性排序可能不同 |
| **結構理想化** | 完美周期晶體。缺陷、晶界、非晶區域、表面、雜質均缺席 | 無法捕捉由缺陷主導的真實材料行為 |
| **交換關聯泛函** | DFT 標籤取決於 XC 泛函（PBE、PBE+U、SCAN、r2SCAN、OptB88vdW 等）。PBE 系統性低估帶隙 | 標籤噪聲下限 = 泛函選擇差異 |
| **磁性態** | +U 修正與初始磁性配置影響能量。局域極小收斂可改變穩定性排序 | 必須將泛函/+U 當作必要的 metadata |
| **版本控制** | 資料庫更新會改變同一材料的數值。必須釘死版本/DOI | 可重現性需要顯式版本追蹤 |

### 逐資料集審核摘要

| 資料集 | 方法 | 核心限制 | 驗證 |
|--------|------|----------|------|
| JARVIS-DFT | VASP DFT (OptB88vdW) | vdW 泛函適合層狀/2D 但非所有體系標準；性質覆蓋不均 | ✓ |
| Materials Project | VASP DFT (PBE/PBE+U/r2SCAN) | +U 依元素變化；r2SCAN 為 v2022.10.28 後新增 | ✓ |
| OQMD | VASP DFT (PBE) | 性質覆蓋較窄；protocol 與 MP/AFLOW 不同 → 跨庫偏移 | ✓ |
| AFLOW | VASP DFT (PBE + AFLOW Standard) | 高度自動化但 protocol 不同 → 跨庫訓練學到工作流指紋 | ✓ |
| Alexandria | DFT (混合泛函) | 必須將泛函當 metadata；版本持續擴張 | ✓ |
| GNoME | **ML 生成 + DFT 驗證** | 選擇偏差（活性學習偏好特定結構空間）；非每筆都有等強度 DFT 驗證；元素頻率分佈與 MP 不同 | ✓ |
| NOMAD | 多碼異質歸檔 | **極端異質性**：不做嚴格過濾（碼/泛函/收斂）會把噪聲灌進模型 | ✓ |
| Matbench | DFT + 實驗混合 | 基準抽象化 — 必須遵守切分協議 | ✓ |
| MPtrj | DFT 弛豫軌跡 (PBE/GGA+U) | 含大量非平衡結構（非僅基態） | ✓ |
| COD | 實驗繞射 | 僅結構、品質不一、無統一標籤 | ✓ |
| ICSD | 實驗繞射（金標準） | **付費** — 需機構訂閱 | ✓ |

### ATLAS 資料治理建議

#### 三層訓練/評估協議

| 層級 | 用途 | 使用資料集 |
|------|------|-----------|
| 第一層：同質 HT-DFT | 主訓練 | JARVIS-DFT 或 MP 單泛函子集 |
| 第二層：跨庫 OOD | 領域偏移測試 | MP ↔ AFLOW ↔ OQMD ↔ Alexandria |
| 第三層：實驗結構 | 分佈壓力測試 | COD 結構 + 重新計算 DFT 標籤 |

#### Protocol Shift 是 UQ 的下限

> 同一材料在 MP、OQMD、AFLOW 之間的 formation energy 差異（因不同 DFT 設定）定義了**標籤噪聲下限** — 這是你的模型應報告的最低不確定性。

