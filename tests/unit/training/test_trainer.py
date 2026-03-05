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
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

import atlas.training.trainer as trainer_module
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

    def test_invalid_grad_clip_norm_raises(self, trainer_components):
        trainer_components["grad_clip_norm"] = float("inf")
        with pytest.raises(ValueError, match="grad_clip_norm"):
            Trainer(**trainer_components)

    def test_signature_failure_falls_back_to_empty_params(self, trainer_components, monkeypatch):
        def _raise_signature(*_args, **_kwargs):
            raise TypeError("signature not supported")

        monkeypatch.setattr(trainer_module, "signature", _raise_signature)
        trainer = Trainer(**trainer_components)
        assert trainer._forward_params == set()


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

    def test_train_epoch_rejects_empty_loader(self, trainer_components):
        trainer = Trainer(**trainer_components)
        with pytest.raises(ValueError, match="train loader is empty"):
            trainer.train_epoch([])


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

    def test_validate_rejects_empty_loader(self, trainer_components):
        trainer = Trainer(**trainer_components)
        with pytest.raises(ValueError, match="validation loader is empty"):
            trainer.validate([])


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

    def test_checkpoint_includes_resume_state(self, trainer_components):
        trainer = Trainer(**trainer_components)
        trainer.history["train_loss"] = [1.0, 0.9]
        trainer.best_val_loss = 0.5
        trainer.patience_counter = 2
        trainer._save_checkpoint("resume_state.pt", epoch=2, val_loss=0.5)
        ckpt = trainer.load_checkpoint("resume_state.pt")
        assert ckpt["schema_version"] == 1
        assert "trainer_state" in ckpt
        assert ckpt["trainer_state"]["best_val_loss"] == 0.5
        assert ckpt["trainer_state"]["patience_counter"] == 2

    def test_load_checkpoint_restores_optimizer_and_trainer_state(self, trainer_components):
        trainer = Trainer(**trainer_components)
        trainer.history["train_loss"] = [1.1, 0.8]
        trainer.best_val_loss = 0.4
        trainer.patience_counter = 1
        trainer._save_checkpoint("restore_state.pt", epoch=3, val_loss=0.4)

        trainer.optimizer.param_groups[0]["lr"] = 0.123
        trainer.best_val_loss = 9.9
        trainer.patience_counter = 0
        trainer.history = {"train_loss": [], "val_loss": [], "lr": [], "epoch_time": []}

        trainer.load_checkpoint(
            "restore_state.pt",
            restore_optimizer=True,
            restore_trainer_state=True,
        )
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
        assert trainer.best_val_loss == pytest.approx(0.4)
        assert trainer.patience_counter == 1
        assert trainer.history["train_loss"] == [1.1, 0.8]

    def test_load_checkpoint_rejects_non_mapping_payload(self, trainer_components, monkeypatch):
        trainer = Trainer(**trainer_components)
        trainer._save_checkpoint("bad_payload.pt", epoch=1, val_loss=0.1)
        monkeypatch.setattr(trainer_module.torch, "load", lambda *args, **kwargs: ["bad"])
        with pytest.raises(TypeError, match="mapping"):
            trainer.load_checkpoint("bad_payload.pt")

    def test_load_checkpoint_requires_optional_states_when_requested(self, trainer_components):
        trainer = Trainer(**trainer_components)
        payload = {"model_state_dict": trainer.model.state_dict()}
        torch.save(payload, trainer.save_dir / "minimal.pt")
        with pytest.raises(KeyError, match="optimizer_state_dict"):
            trainer.load_checkpoint("minimal.pt", restore_optimizer=True)
        with pytest.raises(KeyError, match="trainer_state"):
            trainer.load_checkpoint("minimal.pt", restore_trainer_state=True)

    def test_load_checkpoint_rejects_non_integral_epoch_and_patience_counter(self, trainer_components):
        trainer = Trainer(**trainer_components)
        torch.save(
            {"model_state_dict": trainer.model.state_dict(), "epoch": 1.25},
            trainer.save_dir / "invalid_epoch.pt",
        )
        with pytest.raises(ValueError, match="checkpoint epoch"):
            trainer.load_checkpoint("invalid_epoch.pt")

        torch.save(
            {
                "model_state_dict": trainer.model.state_dict(),
                "trainer_state": {"best_val_loss": 0.5, "patience_counter": 2.4, "history": {}},
            },
            trainer.save_dir / "invalid_patience_counter.pt",
        )
        with pytest.raises(ValueError, match="trainer_state.patience_counter"):
            trainer.load_checkpoint("invalid_patience_counter.pt", restore_trainer_state=True)

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

    def test_fit_rejects_invalid_args(self, trainer_components):
        trainer = Trainer(**trainer_components)
        train_loader = _make_fake_loader(n_batches=1)
        val_loader = _make_fake_loader(n_batches=1)
        with pytest.raises(ValueError, match="n_epochs"):
            trainer.fit(train_loader, val_loader, n_epochs=0, verbose=False)
        with pytest.raises(ValueError, match="n_epochs"):
            trainer.fit(train_loader, val_loader, n_epochs=1.5, verbose=False)
        with pytest.raises(ValueError, match="patience"):
            trainer.fit(train_loader, val_loader, n_epochs=1, patience=-1, verbose=False)
        with pytest.raises(ValueError, match="patience"):
            trainer.fit(train_loader, val_loader, n_epochs=1, patience=True, verbose=False)
        with pytest.raises(ValueError, match="min_delta"):
            trainer.fit(train_loader, val_loader, n_epochs=1, min_delta=float("nan"), verbose=False)
        with pytest.raises(ValueError, match="checkpoint_name"):
            trainer.fit(train_loader, val_loader, n_epochs=1, checkpoint_name="  ", verbose=False)


class TestLossResolution:
    class _EmptyLossDict(nn.Module):
        def forward(self, _pred, _target):
            return {}

    def test_compute_loss_rejects_empty_loss_dict(self, trainer_components):
        trainer_components["loss_fn"] = self._EmptyLossDict()
        trainer = Trainer(**trainer_components)
        batch = _make_fake_batch(n_nodes=4, feat_dim=4)
        pred = torch.randn(4)
        with pytest.raises(ValueError, match="empty dict"):
            trainer._compute_loss(pred, batch)

    def test_compute_loss_rejects_missing_targets_for_dict_predictions(self, trainer_components):
        trainer = Trainer(**trainer_components)
        batch = SimpleNamespace(y=None)
        pred = {"band_gap": torch.tensor([1.0])}
        with pytest.raises(ValueError, match="No targets resolved"):
            trainer._compute_loss(pred, batch)

    def test_align_prediction_target_recovers_from_pooling_runtime_error(self, monkeypatch):
        import torch_geometric.nn as pyg_nn

        def _boom(*_args, **_kwargs):
            raise RuntimeError("pooling failed")

        monkeypatch.setattr(pyg_nn, "global_mean_pool", _boom)
        pred = torch.randn(3, 1)
        target = torch.randn(2)
        batch = SimpleNamespace(batch=torch.tensor([0, 1, 1], dtype=torch.long))
        aligned_pred, aligned_target = Trainer._align_prediction_target(pred, target, batch)
        assert aligned_pred.shape == pred.shape
        assert aligned_target.shape == target.shape


class TestTrainerStabilityGuards:
    class _NaNLoss(nn.Module):
        def forward(self, pred, _target):
            return pred.mean() * torch.tensor(float("nan"), device=pred.device)

    def test_train_epoch_raises_on_non_finite_loss(self, trainer_components):
        trainer_components["loss_fn"] = self._NaNLoss()
        trainer = Trainer(**trainer_components)
        loader = _make_fake_loader(n_batches=1)
        with pytest.raises(ValueError, match="Non-finite loss"):
            trainer.train_epoch(loader)

    def test_fit_raises_on_non_finite_val_loss(self, trainer_components):
        trainer = Trainer(**trainer_components)
        trainer.train_epoch = MagicMock(return_value=0.1)
        trainer.validate = MagicMock(return_value={"val_loss": float("nan")})
        with pytest.raises(ValueError, match="val_loss"):
            trainer.fit([], [], n_epochs=2, verbose=False)

    def test_validate_raises_on_non_finite_loss(self, trainer_components):
        trainer_components["loss_fn"] = self._NaNLoss()
        trainer = Trainer(**trainer_components)
        loader = _make_fake_loader(n_batches=1)
        with pytest.raises(ValueError, match="validation"):
            trainer.validate(loader)

    def test_save_checkpoint_rejects_invalid_epoch_and_val_loss(self, trainer_components):
        trainer = Trainer(**trainer_components)
        with pytest.raises(ValueError, match="epoch"):
            trainer._save_checkpoint("bad.pt", epoch=-1, val_loss=0.1)
        with pytest.raises(ValueError, match="epoch"):
            trainer._save_checkpoint("bad.pt", epoch=1.5, val_loss=0.1)
        with pytest.raises(ValueError, match="epoch"):
            trainer._save_checkpoint("bad.pt", epoch=True, val_loss=0.1)
        with pytest.raises(ValueError, match="val_loss"):
            trainer._save_checkpoint("bad.pt", epoch=1, val_loss=float("nan"))
        with pytest.raises(ValueError, match="simple file name"):
            trainer._save_checkpoint("../bad.pt", epoch=1, val_loss=0.1)


class TestPatienceSemantics:
    def test_patience_zero_continues_while_improving(self, trainer_components):
        trainer = Trainer(**trainer_components)
        trainer.train_epoch = MagicMock(return_value=0.1)
        trainer.validate = MagicMock(
            side_effect=[
                {"val_loss": 1.0},
                {"val_loss": 0.9},
                {"val_loss": 0.8},
            ]
        )
        history = trainer.fit([], [], n_epochs=3, patience=0, verbose=False)
        assert len(history["val_loss"]) == 3

    def test_patience_zero_stops_on_first_non_improvement(self, trainer_components):
        trainer = Trainer(**trainer_components)
        trainer.train_epoch = MagicMock(return_value=0.1)
        trainer.validate = MagicMock(
            side_effect=[
                {"val_loss": 1.0},
                {"val_loss": 1.1},
                {"val_loss": 0.9},
            ]
        )
        history = trainer.fit([], [], n_epochs=3, patience=0, verbose=False)
        assert len(history["val_loss"]) == 2

    def test_fit_rejects_path_like_checkpoint_name(self, trainer_components):
        trainer = Trainer(**trainer_components)
        with pytest.raises(ValueError, match="checkpoint_name"):
            trainer.fit([], [], n_epochs=1, checkpoint_name="../escape", verbose=False)
