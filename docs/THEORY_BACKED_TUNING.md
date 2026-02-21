# Theory-Backed Hyperparameter Tuning (Phase 1-4)

This document defines the non-heuristic references used by:

- `atlas/training/theory_tuning.py`
- `scripts/training/run_adaptive_rounds.py`

The goal is to make tuning choices auditable and reproducible.

## 1. Practical Rules Used in Code

1. Round-1 starts from literature-backed priors per model family (CGCNN, E(3) GNN, MACE, RF).
2. Rounds 2/3 use metric-driven adaptation:
- regression tasks: minimize `best_val_mae` (or available MAE proxy)
- topology/RF tasks: maximize `best_val_acc` or `validation_f1`
3. If improvement is below threshold, automatically:
- lower LR
- increase epochs/resource budget
- increase sample budget when supported
4. If run fails or metric is missing, use a conservative recovery move:
- stronger LR decay + more epochs + safer batch

This is aligned with multi-fidelity resource allocation and iterative tuning, rather than one-shot fixed settings.

## 2. Reference Mapping

### Optimization / Scheduling

- `smith_2018_disciplined`  
  Leslie N. Smith. *A Disciplined Approach to Neural Network Hyper-Parameters.*  
  https://arxiv.org/abs/1803.09820

- `smith_2017_superconvergence`  
  Leslie N. Smith, Nicholay Topin. *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates.*  
  https://arxiv.org/abs/1708.07120

- `loshchilov_2019_adamw`  
  Ilya Loshchilov, Frank Hutter. *Decoupled Weight Decay Regularization (AdamW).*  
  https://openreview.net/forum?id=Bkg6RiCqY7

- `li_2018_hyperband`  
  Lisha Li et al. *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization.*  
  https://jmlr.org/papers/v18/16-558.html

- `bergstra_bengio_2012_random_search`  
  James Bergstra, Yoshua Bengio. *Random Search for Hyper-Parameter Optimization.*  
  https://jmlr.org/beta/papers/v13/bergstra12a.html

### Model-Specific

- `cgcnn_2018`  
  Tian Xie, Jeffrey C. Grossman. *Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties.*  
  https://arxiv.org/abs/1710.10324

- `e3nn_2022`  
  Mario Geiger, Tess Smidt. *e3nn: Euclidean Neural Networks.*  
  https://arxiv.org/abs/2207.09453

- `kendall_2018_uncertainty_weighting`  
  Alex Kendall et al. *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.*  
  https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html

- `mace_2022`  
  Ilyes Batatia et al. *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields.*  
  https://arxiv.org/abs/2206.07697

- `mace_mp_2023`  
  Ilyes Batatia et al. *A Foundation Model for Atomistic Materials Chemistry (MACE-MP).*  
  https://arxiv.org/abs/2401.00096

### Robust Regression / Outliers

- `huber_1964`  
  Peter J. Huber. *Robust Estimation of a Location Parameter.*  
  https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full

### Random Forest

- `breiman_2001_rf`  
  Leo Breiman. *Random Forests.*  
  https://link.springer.com/article/10.1023/A:1010933404324

- `probst_2019_rf_tuning`  
  Philipp Probst, Marvin Wright, Anne-Laure Boulesteix. *Hyperparameters and tuning strategies for random forest.*  
  https://arxiv.org/abs/1804.03515

## 3. Why This Was Added

The project previously mixed:
- static level profiles,
- hardware-friendly ad-hoc overrides,
- manual round scripts.

The new adaptive pipeline keeps profiles explicit and traceable, then tunes between rounds with deterministic rules tied to published practices.
