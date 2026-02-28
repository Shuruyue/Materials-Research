"""
Unit tests for atlas.training.trainer

Tests the Trainer class core functionality:
- Initialization and device handling
- Single epoch training
- Validation loop
- Checkpointing (save/load)
- Early stopping logic
- History tracking
"""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from atlas.training.trainer import Trainer

# ── Fixtures ──────────────────────────────────────────────────


class TinyModel(nn.Module):
    """Minimal model that accepts PyG-style batch inputs."""

    def __init__(self, in_dim=4, out_dim=1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        # Simple mean pool over nodes
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            return global_mean_pool(self.fc(x), batch).squeeze(-1)
        return self.fc(x).mean(dim=0)


class SimpleBatchModel(nn.Module):
    """Even simpler model for testing without PyG dependency."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        return self.fc(x).squeeze(-1)


@pytest.fixture
def simple_model():
    return SimpleBatchModel()


@pytest.fixture
def trainer_components(simple_model, tmp_path):
    """Create a minimal trainer setup."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    return {
        "model": simple_model,
        "optimizer": optimizer,
        "loss_fn": loss_fn,
        "device": "cpu",
        "save_dir": tmp_path / "checkpoints",
    }


def _make_fake_batch(n_nodes=8, feat_dim=4):
    """Create a fake PyG-like batch object."""
    batch = MagicMock()
    batch.x = torch.randn(n_nodes, feat_dim)
    batch.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    batch.edge_attr = torch.randn(4, 8)
    batch.batch = torch.zeros(n_nodes, dtype=torch.long)
    batch.y = torch.randn(1)
    batch.to = MagicMock(return_value=batch)
    return batch


def _make_fake_loader(n_batches=5, n_nodes=8, feat_dim=4):
    """Create a list of fake batches that acts as a DataLoader."""
    return [_make_fake_batch(n_nodes, feat_dim) for _ in range(n_batches)]


# ── Initialization ────────────────────────────────────────────


class TestTrainerInit:
    def test_creates_save_dir(self, trainer_components):
        trainer = Trainer(**trainer_components)
        assert trainer.save_dir.exists()

    def test_amp_disabled_on_cpu(self, trainer_components):
        trainer_components["use_amp"] = True
        trainer = Trainer(**trainer_components)
        assert not trainer.use_amp  # AMP should be False on CPU

    def test_history_initialized(self, trainer_components):
        trainer = Trainer(**trainer_components)
        for key in ["train_loss", "val_loss", "lr", "epoch_time"]:
            assert key in trainer.history
            assert trainer.history[key] == []

    def test_best_val_loss_initialized(self, trainer_components):
        trainer = Trainer(**trainer_components)
        assert trainer.best_val_loss == float("inf")


# ── Training ──────────────────────────────────────────────────


class TestTrainEpoch:
    def test_train_epoch_returns_float(self, trainer_components):
        trainer = Trainer(**trainer_components)
        loader = _make_fake_loader(n_batches=3)
        loss = trainer.train_epoch(loader)
        assert isinstance(loss, float)
        assert not np.isnan(loss)

    def test_train_epoch_model_in_train_mode(self, trainer_components):
        trainer = Trainer(**trainer_components)
        loader = _make_fake_loader(n_batches=2)
        trainer.train_epoch(loader)
        # Model should be in train mode after training
        assert trainer.model.training

    def test_train_epoch_updates_params(self, trainer_components):
        trainer = Trainer(**trainer_components)
        loader = _make_fake_loader(n_batches=3)
        params_before = [p.clone() for p in trainer.model.parameters()]
        trainer.train_epoch(loader)
        params_after = list(trainer.model.parameters())
        # At least one parameter should have changed
        any_changed = any(
            not torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        )
        assert any_changed


# ── Validation ────────────────────────────────────────────────


class TestValidation:
    def test_validate_returns_dict(self, trainer_components):
        trainer = Trainer(**trainer_components)
        loader = _make_fake_loader(n_batches=3)
        metrics = trainer.validate(loader)
        assert isinstance(metrics, dict)
        assert "val_loss" in metrics

    def test_validate_loss_is_finite(self, trainer_components):
        trainer = Trainer(**trainer_components)
        loader = _make_fake_loader(n_batches=3)
        metrics = trainer.validate(loader)
        assert np.isfinite(metrics["val_loss"])


# ── Checkpointing ─────────────────────────────────────────────


class TestCheckpointing:
    def test_save_checkpoint_creates_file(self, trainer_components):
        trainer = Trainer(**trainer_components)
        trainer._save_checkpoint("test_ckpt.pt", epoch=1, val_loss=0.5)
        assert (trainer.save_dir / "test_ckpt.pt").exists()

    def test_save_and_load_checkpoint(self, trainer_components):
        trainer = Trainer(**trainer_components)
        # Save
        trainer._save_checkpoint("roundtrip.pt", epoch=5, val_loss=0.42)
        # Load
        ckpt = trainer.load_checkpoint("roundtrip.pt")
        assert ckpt["epoch"] == 5
        assert ckpt["val_loss"] == 0.42
        assert "model_state_dict" in ckpt

    def test_save_history(self, trainer_components):
        trainer = Trainer(**trainer_components)
        trainer.history["train_loss"] = [1.0, 0.5, 0.3]
        trainer.history["val_loss"] = [1.1, 0.6, 0.4]
        trainer._save_history("test_history.json")

        path = trainer.save_dir / "test_history.json"
        assert path.exists()
        with open(path) as f:
            saved = json.load(f)
        assert saved["train_loss"] == [1.0, 0.5, 0.3]


# ── Full Fit ──────────────────────────────────────────────────


class TestFit:
    def test_fit_runs_without_error(self, trainer_components):
        trainer = Trainer(**trainer_components)
        train_loader = _make_fake_loader(n_batches=3)
        val_loader = _make_fake_loader(n_batches=2)
        history = trainer.fit(
            train_loader, val_loader,
            n_epochs=3, patience=10, verbose=False
        )
        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3

    def test_early_stopping(self, trainer_components):
        trainer = Trainer(**trainer_components)
        train_loader = _make_fake_loader(n_batches=2)
        val_loader = _make_fake_loader(n_batches=2)
        # patience=1 should trigger early stop quickly
        history = trainer.fit(
            train_loader, val_loader,
            n_epochs=100, patience=1, verbose=False
        )
        # Should stop well before 100 epochs
        assert len(history["train_loss"]) < 100

    def test_fit_creates_best_checkpoint(self, trainer_components):
        trainer = Trainer(**trainer_components)
        train_loader = _make_fake_loader(n_batches=2)
        val_loader = _make_fake_loader(n_batches=2)
        trainer.fit(
            train_loader, val_loader,
            n_epochs=3, patience=10, verbose=False,
            checkpoint_name="mymodel"
        )
        assert (trainer.save_dir / "mymodel_best.pt").exists()
        assert (trainer.save_dir / "mymodel_final.pt").exists()
        assert (trainer.save_dir / "mymodel_history.json").exists()
