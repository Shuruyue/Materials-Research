"""Unit tests for training checkpoint manager."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from atlas.training.checkpoint import CheckpointManager


def _load_epoch(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["epoch"])


def test_checkpoint_manager_validates_constructor_and_inputs(tmp_path: Path):
    with pytest.raises(ValueError, match="top_k"):
        CheckpointManager(tmp_path, top_k=0, keep_last_k=2)
    with pytest.raises(ValueError, match="keep_last_k"):
        CheckpointManager(tmp_path, top_k=1, keep_last_k=0)

    manager = CheckpointManager(tmp_path, top_k=2, keep_last_k=2)
    with pytest.raises(ValueError, match="mae"):
        manager.save_best({"epoch": 1}, mae=float("nan"), epoch=1)
    with pytest.raises(ValueError, match="epoch"):
        manager.save_best({"epoch": 1}, mae=0.1, epoch=-1)
    with pytest.raises(ValueError, match="epoch"):
        manager.save_checkpoint({"state": 1}, epoch=-1)
    with pytest.raises(TypeError, match="mapping"):
        manager.save_checkpoint(["bad"], epoch=1)  # type: ignore[arg-type]


def test_checkpoint_manager_rejects_non_integral_or_bool_int_controls(tmp_path: Path):
    manager = CheckpointManager(tmp_path, top_k=2, keep_last_k=2)
    with pytest.raises(ValueError, match="integer"):
        manager.save_best({"epoch": 1}, mae=0.1, epoch=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer"):
        manager.save_best({"epoch": 1}, mae=0.1, epoch=1.5)
    with pytest.raises(ValueError, match="integer"):
        manager.save_checkpoint({"state": 1}, epoch=2.2)
    with pytest.raises(ValueError, match="finite float"):
        manager.save_best({"epoch": 1}, mae=False, epoch=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer"):
        manager.save_checkpoint({"state": 1}, epoch=np.bool_(True))  # type: ignore[arg-type]


def test_save_best_keeps_top_k_and_updates_best_pointer(tmp_path: Path):
    manager = CheckpointManager(tmp_path, top_k=2, keep_last_k=2)
    manager.save_best({"epoch": 1}, mae=0.5, epoch=1)
    manager.save_best({"epoch": 2}, mae=0.4, epoch=2)
    manager.save_best({"epoch": 3}, mae=0.6, epoch=3)  # should not qualify into top-2

    files = sorted(tmp_path.glob("best_epoch_*.pt"))
    assert len(files) == 2
    assert all("epoch_3_" not in file.name for file in files)
    assert (tmp_path / "best.pt").exists()
    assert _load_epoch(tmp_path / "best.pt") == 2


def test_save_checkpoint_rotates_history(tmp_path: Path):
    manager = CheckpointManager(tmp_path, top_k=2, keep_last_k=3)
    for epoch in range(1, 5):
        manager.save_checkpoint({"epoch": epoch}, epoch=epoch)

    assert _load_epoch(tmp_path / "checkpoint.pt") == 4
    assert _load_epoch(tmp_path / "checkpoint_prev_1.pt") == 3
    assert _load_epoch(tmp_path / "checkpoint_prev_2.pt") == 2
    assert not (tmp_path / "checkpoint_prev_3.pt").exists()


def test_save_checkpoint_keep_last_one_does_not_create_history(tmp_path: Path):
    manager = CheckpointManager(tmp_path, top_k=1, keep_last_k=1)
    manager.save_checkpoint({"state": 1}, epoch=1)
    manager.save_checkpoint({"state": 2}, epoch=2)
    assert (tmp_path / "checkpoint.pt").exists()
    assert _load_epoch(tmp_path / "checkpoint.pt") == 2
    assert not (tmp_path / "checkpoint_prev_1.pt").exists()


def test_checkpoint_manager_rehydrates_best_models_from_disk(tmp_path: Path):
    torch.save({"epoch": 1, "val_mae": 0.4}, tmp_path / "best_epoch_1_mae_0.4000.pt")
    torch.save({"epoch": 2, "val_mae": 0.3}, tmp_path / "best_epoch_2_mae_0.3000.pt")
    torch.save({"epoch": 3, "val_mae": 0.5}, tmp_path / "best_epoch_3_mae_0.5000.pt")

    manager = CheckpointManager(tmp_path, top_k=2, keep_last_k=2)

    assert len(manager.best_models) == 2
    assert [entry[1] for entry in manager.best_models] == [2, 1]
    assert not (tmp_path / "best_epoch_3_mae_0.5000.pt").exists()
    assert (tmp_path / "best.pt").exists()
    assert _load_epoch(tmp_path / "best.pt") == 2
