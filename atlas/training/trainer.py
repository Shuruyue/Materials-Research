"""
Training Loop

General-purpose trainer for crystal GNN models.
Handles training, validation, early stopping, robust checkpointing, and logging.
"""

import json
import logging
import time
from inspect import signature
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from atlas.config import get_config

logger = logging.getLogger(__name__)


class Trainer:
    """
    General-purpose GNN trainer (Robust).

    Features:
    - Automatic Mixed Precision (AMP) for speed
    - Gradient Clipping
    - Top-K Checkpointing
    - Learning Rate Scheduling
    - Early Stopping

    Args:
        model: GNN model
        optimizer: torch optimizer
        loss_fn: loss function
        device: 'cuda' or 'cpu'
        save_dir: directory for checkpoints
        use_amp: enable Automatic Mixed Precision
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        device: str | torch.device | None = None,
        save_dir: Path | None = None,
        use_amp: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.device_type = self.device.type
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.use_amp = use_amp and (self.device_type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.save_dir = save_dir or get_config().paths.models_dir / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "epoch_time": [],
        }
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _forward_batch(self, batch):
        """
        Forward wrapper that adapts to common ATLAS model signatures.

        This keeps Trainer usable across CGCNN/TopoGNN/Equivariant/MultiTask
        without requiring per-script forward glue.
        """
        if not hasattr(batch, "x") or not hasattr(batch, "edge_index"):
            return self.model(batch)

        params = signature(self.model.forward).parameters

        def has(name: str) -> bool:
            return name in params

        edge_attr = getattr(batch, "edge_attr", None)
        edge_vec = getattr(batch, "edge_vec", None)
        batch_index = getattr(batch, "batch", None)

        kwargs = {}
        if has("node_feats"):
            kwargs["node_feats"] = batch.x
        elif has("x"):
            kwargs["x"] = batch.x

        if has("edge_index"):
            kwargs["edge_index"] = batch.edge_index

        if has("edge_feats"):
            kwargs["edge_feats"] = edge_attr if edge_attr is not None else edge_vec
        if has("edge_attr"):
            kwargs["edge_attr"] = edge_attr if edge_attr is not None else edge_vec
        if has("edge_vectors"):
            kwargs["edge_vectors"] = edge_vec if edge_vec is not None else edge_attr
        if has("edge_vec"):
            kwargs["edge_vec"] = edge_vec if edge_vec is not None else edge_attr
        if has("batch"):
            kwargs["batch"] = batch_index

        if has("edge_index_3body") and hasattr(batch, "edge_index_3body"):
            kwargs["edge_index_3body"] = batch.edge_index_3body

        try:
            return self.model(**kwargs)
        except Exception:
            # Positional fallback for legacy/simple models.
            if edge_attr is not None:
                return self.model(batch.x, batch.edge_index, edge_attr, batch_index)
            if edge_vec is not None:
                return self.model(batch.x, batch.edge_index, edge_vec, batch_index)
            return self.model(batch)

    def _resolve_targets(self, pred, batch):
        """
        Build training targets compatible with task-wise or single-task losses.
        """
        if isinstance(pred, dict):
            if hasattr(batch, "y_dict") and batch.y_dict is not None:
                return batch.y_dict

            target_dict = {}
            for key in pred:
                if hasattr(batch, key):
                    target_dict[key] = getattr(batch, key)

            if not target_dict and hasattr(batch, "y") and batch.y is not None and len(pred) == 1:
                target_key = next(iter(pred.keys()))
                target_dict[target_key] = batch.y

            return target_dict

        if hasattr(batch, "y") and batch.y is not None:
            return batch.y
        return batch

    def _compute_loss(self, pred, batch):
        targets = self._resolve_targets(pred, batch)
        pred, targets = self._align_prediction_target(pred, targets, batch)
        loss_out = self.loss_fn(pred, targets)
        if isinstance(loss_out, dict):
            return loss_out["total"] if "total" in loss_out else next(iter(loss_out.values()))
        return loss_out

    @staticmethod
    def _align_prediction_target(pred, targets, batch):
        """
        Align prediction/target shapes for common graph-level edge cases.

        If the model returns node-level outputs but labels are graph-level scalars,
        aggregate predictions by `batch` index to avoid silent broadcasting.
        """
        if isinstance(pred, dict) or isinstance(targets, dict):
            return pred, targets

        if not isinstance(pred, torch.Tensor) or not isinstance(targets, torch.Tensor):
            return pred, targets

        if pred.shape == targets.shape:
            return pred, targets

        batch_index = getattr(batch, "batch", None)
        if (
            batch_index is not None
            and isinstance(batch_index, torch.Tensor)
            and pred.dim() in {1, 2}
            and targets.dim() in {1, 2}
        ):
            try:
                from torch_geometric.nn import global_mean_pool

                pred_graph = global_mean_pool(pred.reshape(pred.size(0), -1), batch_index)
                pred_graph = pred_graph.squeeze(-1) if pred_graph.size(-1) == 1 else pred_graph
                target_graph = targets.squeeze(-1) if targets.dim() == 2 and targets.size(-1) == 1 else targets
                if pred_graph.shape == target_graph.shape:
                    return pred_graph, target_graph
            except Exception:
                pass

        if targets.numel() == 1 and pred.numel() > 1:
            return pred.mean().reshape_as(targets), targets

        return pred, targets

    def train_epoch(self, loader: DataLoader) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred = self._forward_batch(batch)
                loss = self._compute_loss(pred, batch)

            # Backward pass with scaler
            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Validate model. Returns dict of metrics."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred = self._forward_batch(batch)
                loss = self._compute_loss(pred, batch)

            total_loss += loss.item()
            n_batches += 1

        metrics = {
            "val_loss": total_loss / max(n_batches, 1),
        }
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 300,
        patience: int = 50,
        min_delta: float = 1e-2,
        verbose: bool = True,
        checkpoint_name: str = "model"
    ) -> dict:
        """
        Full training loop.
        """
        if verbose:
            logger.info(f"Training on {self.device} (AMP={self.use_amp}) for {n_epochs} epochs")
            logger.info(f"Checkpoints: {self.save_dir}")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics["val_loss"]

            if self.scheduler is not None:
                # Handle ReduceLROnPlateau vs others
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            dt = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)
            self.history["epoch_time"].append(dt)

            # checkpointing
            improved = val_loss < (self.best_val_loss - min_delta)
            if improved:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(f"{checkpoint_name}_best.pt", epoch, val_loss)
            else:
                self.patience_counter += 1

            if verbose and (epoch % 5 == 0 or epoch == 1):
                logger.info(
                    f"Epoch {epoch:4d} | "
                    f"L_trn: {train_loss:.4f} | "
                    f"L_val: {val_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Pat: {self.patience_counter}/{patience}"
                )

            if self.patience_counter >= patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch}")
                break

        self._save_checkpoint(f"{checkpoint_name}_final.pt", epoch, val_loss)
        self._save_history(f"{checkpoint_name}_history.json")

        return self.history

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        path = self.save_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": get_config() # Save config just in case
        }, path)

    def load_checkpoint(self, filename: str = "model_best.pt"):
        """Load model from checkpoint."""
        path = self.save_dir / filename
        if not path.exists():
            # Try looking in parent if simple name given
            path = self.save_dir.parent / filename

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint

    def _save_history(self, filename: str = "history.json"):
        """Save training history."""
        path = self.save_dir / filename
        hist = {k: [float(v) for v in vs] for k, vs in self.history.items()}
        with open(path, "w") as f:
            json.dump(hist, f, indent=2)
