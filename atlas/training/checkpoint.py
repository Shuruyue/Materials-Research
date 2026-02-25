"""
Checkpoint Manager

Manages model checkpoint persistence with top-k best models
and rotating last-k checkpoints for training recovery.
"""

import logging
import shutil
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages top-k best models and last-k rotating checkpoints.

    Strategy:
    1. Keeps 'best.pt' as the absolute best model.
    2. Keeps 'best_2.pt', 'best_3.pt' as runners-up.
    3. Keeps 'checkpoint.pt' as the latest state.
    4. Keeps 'checkpoint_prev_1.pt', 'checkpoint_prev_2.pt' as history.
    """

    def __init__(self, save_dir, top_k: int = 3, keep_last_k: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.keep_last_k = keep_last_k
        self.best_models: list[tuple[float, int, str]] = []  # (mae, epoch, filename)

    def save_best(self, state: dict, mae: float, epoch: int):
        """Save model if it qualifies as a top-k best."""
        filename = f"best_epoch_{epoch}_mae_{mae:.4f}.pt"
        path = self.save_dir / filename

        torch.save(state, path)
        self.best_models.append((mae, epoch, filename))
        self.best_models.sort(key=lambda x: x[0])  # Sort by MAE ascending

        # Prune beyond top-k
        if len(self.best_models) > self.top_k:
            worst = self.best_models.pop()
            worst_path = self.save_dir / worst[2]
            if worst_path.exists():
                worst_path.unlink()

        # Update 'best.pt' symlink/copy if this is the new #1
        if self.best_models[0][1] == epoch:
            shutil.copy(path, self.save_dir / "best.pt")
            logger.info(f"New best model: epoch={epoch}, MAE={mae:.4f}")

    def save_checkpoint(self, state: dict, epoch: int):
        """Save rotating checkpoint with history."""
        # Delete oldest
        oldest = self.save_dir / f"checkpoint_prev_{self.keep_last_k - 1}.pt"
        if oldest.exists():
            oldest.unlink()

        # Shift others
        for i in range(self.keep_last_k - 2, 0, -1):
            src = self.save_dir / f"checkpoint_prev_{i}.pt"
            dst = self.save_dir / f"checkpoint_prev_{i + 1}.pt"
            if src.exists():
                shutil.move(str(src), str(dst))

        # Move current 'checkpoint.pt' to 'prev_1'
        current = self.save_dir / "checkpoint.pt"
        if current.exists():
            shutil.move(str(current), str(self.save_dir / "checkpoint_prev_1.pt"))

        # Save new
        torch.save(state, current)
