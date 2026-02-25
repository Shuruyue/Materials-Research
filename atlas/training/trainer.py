"""
Training Loop

General-purpose trainer for crystal GNN models.
Handles training, validation, early stopping, robust checkpointing, and logging.
"""

import json
import logging
import time
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: Path | None = None,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and (device == "cuda")
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

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

    def train_epoch(self, loader: DataLoader) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                # Flexible model call (some graph models take different args)
                if hasattr(getattr(self.model, "forward", None), "__code__"):
                     # Standard PyG call
                     pred = self.model(
                        batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.batch,
                     )
                else:
                     # Fallback
                     pred = self.model(batch)

                # Loss computation
                # Handle Multi-Task (Dict output)
                if isinstance(pred, dict):
                    # For multi-task, we usually need keys from the batch, so pass the whole batch
                    # explicitly as targets if 'y_dict' isn't available.
                    # Also, carefully verify if we really should use batch.y
                    targets = batch
                    if hasattr(batch, 'y_dict'):
                        targets = batch.y_dict

                    loss_dict = self.loss_fn(pred, targets)
                    # Extract total loss
                    loss = loss_dict["total"] if isinstance(loss_dict, dict) else loss_dict
                else:
                    # Single task
                    # Prefer batch.y if it exists and is not None
                    targets = batch.y if (hasattr(batch, 'y') and batch.y is not None) else batch
                    loss = self.loss_fn(pred, targets)

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

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                pred = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                )

                if isinstance(pred, dict):
                     targets = batch
                     if hasattr(batch, 'y_dict'):
                        targets = batch.y_dict
                     loss_dict = self.loss_fn(pred, targets)
                     loss = loss_dict["total"] if isinstance(loss_dict, dict) else loss_dict
                else:
                    targets = batch.y if (hasattr(batch, 'y') and batch.y is not None) else batch
                    loss = self.loss_fn(pred, targets)

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
            if val_loss < self.best_val_loss:
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
