"""
Training Loop

General-purpose trainer for crystal GNN models.
Handles training, validation, early stopping, checkpointing, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Callable
import time
import json

from atlas.config import get_config


class Trainer:
    """
    General-purpose GNN trainer.

    Args:
        model: GNN model
        optimizer: torch optimizer
        loss_fn: loss function
        device: 'cuda' or 'cpu'
        save_dir: directory for checkpoints
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
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

            pred = self.model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )

            loss = self.loss_fn(pred, batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validate model. Returns dict of metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)

            pred = self.model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )

            loss = self.loss_fn(pred, batch)
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
    ) -> Dict:
        """
        Full training loop with early stopping.

        Args:
            train_loader: training data
            val_loader: validation data
            n_epochs: max epochs
            patience: early stopping patience
            verbose: print progress

        Returns:
            Training history dict
        """
        if verbose:
            print(f"Training on {self.device} for up to {n_epochs} epochs")
            print(f"Early stopping patience: {patience}")
            print(f"Checkpoints: {self.save_dir}")
            print("-" * 60)

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics["val_loss"]

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            dt = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)
            self.history["epoch_time"].append(dt)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best.pt", epoch, val_loss)
            else:
                self.patience_counter += 1

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(
                    f"  Epoch {epoch:4d} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"val_loss: {val_loss:.4f} | "
                    f"lr: {lr:.2e} | "
                    f"time: {dt:.1f}s | "
                    f"patience: {self.patience_counter}/{patience}"
                )

            if self.patience_counter >= patience:
                if verbose:
                    print(f"\n  Early stopping at epoch {epoch}")
                break

        # Save final and history
        self._save_checkpoint("final.pt", epoch, val_loss)
        self._save_history()

        if verbose:
            print(f"\n  Best val_loss: {self.best_val_loss:.4f}")
            print(f"  Total epochs: {epoch}")

        return self.history

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        path = self.save_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }, path)

    def load_checkpoint(self, filename: str = "best.pt"):
        """Load model from checkpoint."""
        path = self.save_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def _save_history(self):
        """Save training history to JSON."""
        path = self.save_dir / "history.json"
        # Convert to serializable
        hist = {k: [float(v) for v in vs] for k, vs in self.history.items()}
        with open(path, "w") as f:
            json.dump(hist, f, indent=2)
