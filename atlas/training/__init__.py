"""
ATLAS Training Module

Training infrastructure for crystal property prediction:
- Trainer: Robust single/multi-task training loop with AMP and Gradient Clipping
- MultiTaskLoss: Uncertainty-weighted or fixed-weight multi-task loss
- PhysicsConstraintLoss: Physics-informed loss functions (Positivity, Bounds)
- Metrics: Comprehensive evaluation metrics (MAE, RMSE, R², AUC, F1)
- Normalizers: Target normalization for single and multi-property training
- Filters: Statistical outlier detection and removal
- CheckpointManager: Top-k best + rotating checkpoint persistence
"""

from atlas.training.checkpoint import CheckpointManager
from atlas.training.filters import filter_outliers
from atlas.training.losses import MultiTaskLoss, PropertyLoss
from atlas.training.metrics import classification_metrics, scalar_metrics, tensor_metrics
from atlas.training.normalizers import MultiTargetNormalizer, TargetNormalizer
from atlas.training.physics_losses import PhysicsConstraintLoss, VoigtReussBoundsLoss
from atlas.training.trainer import Trainer

__all__ = [
    "Trainer",
    "MultiTaskLoss",
    "PropertyLoss",
    "PhysicsConstraintLoss",
    "VoigtReussBoundsLoss",
    "scalar_metrics",
    "tensor_metrics",
    "classification_metrics",
    "TargetNormalizer",
    "MultiTargetNormalizer",
    "filter_outliers",
    "CheckpointManager",
]
