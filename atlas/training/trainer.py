"""
Training Loop

General-purpose trainer for crystal GNN models.
Handles training, validation, early stopping, robust checkpointing, and logging.
"""

import json
import logging
import math
import os
import re
import tempfile
import time
from collections.abc import Mapping
from contextlib import suppress
from inspect import signature
from numbers import Integral, Real
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from atlas.config import get_config

logger = logging.getLogger(__name__)
_CHECKPOINT_STEM_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_CHECKPOINT_SCHEMA_VERSION = 1


def _is_boolean_like(value: object) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool_", "bool"}


def _coerce_non_negative_int(value: object, name: str) -> int:
    """Convert scalar knobs to non-negative integers without silent truncation."""
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be integer-valued, not boolean")

    if isinstance(value, Integral):
        integer = int(value)
    elif isinstance(value, Real):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        rounded = round(scalar)
        if abs(scalar - rounded) > 1e-9:
            raise ValueError(f"{name} must be integer-valued")
        integer = int(rounded)
    else:
        raise ValueError(f"{name} must be integer-valued")

    if integer < 0:
        raise ValueError(f"{name} must be >= 0")
    return integer


def _coerce_non_negative_float(value: float, name: str) -> float:
    """Convert runtime scalar knobs to finite non-negative floats."""
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be finite non-negative float, not boolean")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if scalar < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return scalar


def _coerce_finite_float(value: float, name: str) -> float:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be finite float, not boolean")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _validate_checkpoint_stem(value: str, field_name: str) -> str:
    stem = str(value).strip()
    if not stem:
        raise ValueError(f"{field_name} must be a non-empty string")
    if "/" in stem or "\\" in stem:
        raise ValueError(f"{field_name} must not contain path separators")
    if Path(stem).name != stem:
        raise ValueError(f"{field_name} must be a simple file stem")
    if ".." in stem:
        raise ValueError(f"{field_name} must not contain path traversal segments")
    if not _CHECKPOINT_STEM_PATTERN.fullmatch(stem):
        raise ValueError(
            f"{field_name} may only contain letters, numbers, '.', '_' and '-' characters"
        )
    return stem


def _atomic_json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
            tmp_path = Path(handle.name)
        if tmp_path is None:
            raise RuntimeError("failed to create temporary JSON file")
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            with suppress(OSError):
                tmp_path.unlink()


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
        if tmp_path is None:
            raise RuntimeError("failed to create temporary checkpoint file")
        torch.save(dict(payload), tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            with suppress(OSError):
                tmp_path.unlink()


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
        grad_clip_norm: float = 1.0,
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
        self.grad_clip_norm = _coerce_non_negative_float(grad_clip_norm, "grad_clip_norm")
        try:
            self._forward_params = set(signature(self.model.forward).parameters.keys())
        except (TypeError, ValueError):
            self._forward_params = set()

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

        def has(name: str) -> bool:
            return name in self._forward_params

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
        except TypeError:
            # Positional fallback for legacy/simple models.
            if edge_attr is not None:
                return self.model(batch.x, batch.edge_index, edge_attr, batch_index)
            if edge_vec is not None:
                return self.model(batch.x, batch.edge_index, edge_vec, batch_index)
            return self.model(batch)

    def _autocast(self):
        return torch.amp.autocast(self.device_type, enabled=self.use_amp)

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
        if isinstance(pred, dict) and isinstance(targets, dict) and not targets:
            raise ValueError(
                "No targets resolved for prediction dictionary. "
                "Provide batch.y_dict or matching target attributes."
            )
        pred, targets = self._align_prediction_target(pred, targets, batch)
        loss_out = self.loss_fn(pred, targets)
        if isinstance(loss_out, dict):
            if "total" in loss_out:
                return loss_out["total"]
            if not loss_out:
                raise ValueError("Loss function returned an empty dict. Expected 'total' or at least one task loss.")
            return next(iter(loss_out.values()))
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
            except (ImportError, RuntimeError, TypeError, ValueError):
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
            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast():
                pred = self._forward_batch(batch)
                loss = self._compute_loss(pred, batch)
            if not torch.isfinite(loss).all():
                raise ValueError("Non-finite loss encountered during training")

            # Backward pass with scaler
            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first)
            if self.grad_clip_norm > 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            raise ValueError("train loader is empty; expected at least one batch")
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Validate model. Returns dict of metrics."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)

            with self._autocast():
                pred = self._forward_batch(batch)
                loss = self._compute_loss(pred, batch)
            if not torch.isfinite(loss).all():
                raise ValueError("Non-finite loss encountered during validation")

            total_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            raise ValueError("validation loader is empty; expected at least one batch")
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
        n_epochs_int = _coerce_non_negative_int(n_epochs, "n_epochs")
        if n_epochs_int <= 0:
            raise ValueError("n_epochs must be > 0")
        patience_int = _coerce_non_negative_int(patience, "patience")
        min_delta = _coerce_non_negative_float(min_delta, "min_delta")
        checkpoint_name = _validate_checkpoint_stem(checkpoint_name, "checkpoint_name")

        if verbose:
            logger.info(f"Training on {self.device} (AMP={self.use_amp}) for {n_epochs_int} epochs")
            logger.info(f"Checkpoints: {self.save_dir}")

        last_epoch = 0
        last_val_loss = float("nan")
        for epoch in range(1, n_epochs_int + 1):
            t0 = time.time()

            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            val_loss = _coerce_finite_float(val_metrics["val_loss"], "val_loss")
            last_epoch = epoch
            last_val_loss = val_loss

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
                    f"Pat: {self.patience_counter}/{patience_int}"
                )

            should_stop = False
            if patience_int == 0:
                should_stop = not improved
            elif self.patience_counter >= patience_int:
                should_stop = True
            if should_stop:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch}")
                break

        self._save_checkpoint(f"{checkpoint_name}_final.pt", last_epoch, last_val_loss)
        self._save_history(f"{checkpoint_name}_history.json")

        return self.history

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string")
        if "/" in filename or "\\" in filename or Path(filename).name != filename or ".." in filename:
            raise ValueError("filename must be a simple file name without path separators")
        epoch_int = _coerce_non_negative_int(epoch, "epoch")
        val_loss_scalar = _coerce_finite_float(val_loss, "val_loss")
        path = self.save_dir / filename
        cfg = get_config()
        payload = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "epoch": epoch_int,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler_state_dict": (
                self.scaler.state_dict() if self.scaler is not None else None
            ),
            "trainer_state": {
                "best_val_loss": (
                    float(self.best_val_loss) if math.isfinite(float(self.best_val_loss)) else None
                ),
                "patience_counter": int(self.patience_counter),
                "history": {k: list(v) for k, v in self.history.items()},
            },
            "val_loss": val_loss_scalar,
            "config": {
                "summary": cfg.summary(),
                "device": str(self.device),
                "use_amp": bool(self.use_amp),
                "grad_clip_norm": float(self.grad_clip_norm),
            },
        }
        _atomic_torch_save(path, payload)

    @staticmethod
    def _validate_checkpoint_payload(checkpoint: Mapping[str, object]) -> None:
        if "model_state_dict" not in checkpoint:
            raise KeyError("checkpoint missing required key 'model_state_dict'")
        model_state = checkpoint["model_state_dict"]
        if not isinstance(model_state, Mapping):
            raise TypeError("checkpoint['model_state_dict'] must be a mapping")

        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None and not isinstance(
            checkpoint["optimizer_state_dict"], Mapping
        ):
            raise TypeError("checkpoint['optimizer_state_dict'] must be a mapping")
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None and not isinstance(
            checkpoint["scheduler_state_dict"], Mapping
        ):
            raise TypeError("checkpoint['scheduler_state_dict'] must be a mapping")
        if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None and not isinstance(
            checkpoint["scaler_state_dict"], Mapping
        ):
            raise TypeError("checkpoint['scaler_state_dict'] must be a mapping")
        if "trainer_state" in checkpoint and checkpoint["trainer_state"] is not None and not isinstance(
            checkpoint["trainer_state"], Mapping
        ):
            raise TypeError("checkpoint['trainer_state'] must be a mapping")

        if "epoch" in checkpoint and checkpoint["epoch"] is not None:
            _coerce_non_negative_int(checkpoint["epoch"], "checkpoint epoch")
        if "val_loss" in checkpoint and checkpoint["val_loss"] is not None:
            _coerce_finite_float(checkpoint["val_loss"], "checkpoint val_loss")

    def load_checkpoint(
        self,
        filename: str = "model_best.pt",
        *,
        restore_optimizer: bool = False,
        restore_scheduler: bool = False,
        restore_scaler: bool = False,
        restore_trainer_state: bool = False,
        strict: bool = True,
    ):
        """Load model from checkpoint."""
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string")
        if "/" in filename or "\\" in filename or Path(filename).name != filename or ".." in filename:
            raise ValueError("filename must be a simple file name without path separators")
        path = self.save_dir / filename
        if not path.exists():
            # Try looking in parent if simple name given
            path = self.save_dir.parent / filename
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filename}")

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except Exception as exc:  # pragma: no cover - fallback path for legacy checkpoints
            logger.warning(
                "weights_only=True checkpoint load failed (%s); retrying with weights_only=False",
                exc,
            )
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError("checkpoint payload must be a mapping")
        self._validate_checkpoint_payload(checkpoint)

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        if restore_optimizer:
            if "optimizer_state_dict" not in checkpoint or checkpoint["optimizer_state_dict"] is None:
                raise KeyError("checkpoint missing required key 'optimizer_state_dict'")
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if restore_scheduler:
            if self.scheduler is None:
                raise ValueError("restore_scheduler=True requires trainer.scheduler to be configured")
            if "scheduler_state_dict" not in checkpoint or checkpoint["scheduler_state_dict"] is None:
                raise KeyError("checkpoint missing required key 'scheduler_state_dict'")
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if restore_scaler:
            if "scaler_state_dict" not in checkpoint or checkpoint["scaler_state_dict"] is None:
                raise KeyError("checkpoint missing required key 'scaler_state_dict'")
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if restore_trainer_state:
            trainer_state = checkpoint.get("trainer_state")
            if not isinstance(trainer_state, Mapping):
                raise KeyError("checkpoint missing required key 'trainer_state'")
            best_val_loss = trainer_state.get("best_val_loss")
            if best_val_loss is not None:
                self.best_val_loss = _coerce_finite_float(best_val_loss, "trainer_state.best_val_loss")
            patience_counter = trainer_state.get("patience_counter")
            if patience_counter is not None:
                self.patience_counter = _coerce_non_negative_int(
                    patience_counter, "trainer_state.patience_counter"
                )
            history = trainer_state.get("history")
            if isinstance(history, Mapping):
                for key in self.history:
                    values = history.get(key)
                    if isinstance(values, list):
                        self.history[key] = [
                            _coerce_finite_float(v, f"trainer_state.history[{key}]")
                            for v in values
                        ]

        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint

    def _save_history(self, filename: str = "history.json"):
        """Save training history."""
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string")
        if "/" in filename or "\\" in filename or Path(filename).name != filename or ".." in filename:
            raise ValueError("filename must be a simple file name without path separators")
        path = self.save_dir / filename
        hist = {
            k: [float(v) if math.isfinite(float(v)) else None for v in vs]
            for k, vs in self.history.items()
        }
        _atomic_json_dump(path, hist)
