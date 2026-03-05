"""
Checkpoint Manager

Manages model checkpoint persistence with top-k best models
and rotating last-k checkpoints for training recovery.
"""

import logging
import math
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)
_EPS = 1e-12
_BEST_FILENAME_RE = re.compile(
    r"^best_epoch_(?P<epoch>\d+)_mae_(?P<mae>[-+]?\d+(?:\.\d+)?)\.pt$"
)


def _is_boolean_like(value: Any) -> bool:
    return isinstance(value, bool) or type(value).__name__ == "bool_"


def _is_integral_like_float(value: float, *, tol: float = 1e-9) -> bool:
    return math.isfinite(value) and abs(value - round(value)) <= tol


def _coerce_int(value: Any, name: str) -> int:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be an integer, got bool")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not _is_integral_like_float(value):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return int(round(value))
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be an integer, got empty value")
    try:
        return int(text, 10)
    except ValueError:
        try:
            parsed = float(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc
        if not _is_integral_like_float(parsed):
            raise ValueError(f"{name} must be an integer, got {value!r}") from None
        return int(round(parsed))


def _coerce_finite_float(value: Any, name: str) -> float:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be a finite float, got bool")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _coerce_positive_int(value: int, name: str) -> int:
    scalar = _coerce_int(value, name)
    if scalar <= 0:
        raise ValueError(f"{name} must be > 0")
    return scalar


def _coerce_non_negative_int(value: Any, name: str) -> int:
    scalar = _coerce_int(value, name)
    if scalar < 0:
        raise ValueError(f"{name} must be >= 0")
    return scalar


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """Atomically persist a checkpoint payload to avoid partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".pt",
        prefix=f".{path.stem}.",
        dir=path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(dict(payload), tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_copy(source: Path, target: Path) -> None:
    """Atomically copy an existing file into target location."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=target.suffix or ".pt",
        prefix=f".{target.stem}.",
        dir=target.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        shutil.copy2(source, tmp_path)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _parse_best_filename(path: Path) -> tuple[float, int, str] | None:
    match = _BEST_FILENAME_RE.fullmatch(path.name)
    if match is None:
        return None
    try:
        epoch_idx = int(match.group("epoch"))
        mae_value = float(match.group("mae"))
    except (TypeError, ValueError):
        return None
    if epoch_idx < 0 or not math.isfinite(mae_value):
        return None
    return mae_value, epoch_idx, path.name


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
        self.top_k = _coerce_positive_int(top_k, "top_k")
        self.keep_last_k = _coerce_positive_int(keep_last_k, "keep_last_k")
        self.best_models: list[tuple[float, int, str]] = []  # (mae, epoch, filename)
        self._sync_best_models_from_disk()

    @staticmethod
    def _coerce_state(state: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(state, Mapping):
            raise TypeError(f"state must be a mapping, got {type(state)!r}")
        return dict(state)

    def _sync_best_models_from_disk(self) -> None:
        discovered: list[tuple[float, int, str]] = []
        for path in self.save_dir.glob("best_epoch_*_mae_*.pt"):
            parsed = _parse_best_filename(path)
            if parsed is None:
                continue
            discovered.append(parsed)

        discovered.sort(key=lambda x: (x[0], x[1]))
        for _, _, filename in discovered[self.top_k:]:
            stale_path = self.save_dir / filename
            if stale_path.exists():
                stale_path.unlink()
        self.best_models = discovered[: self.top_k]

        best_pointer = self.save_dir / "best.pt"
        if not self.best_models:
            if best_pointer.exists():
                best_pointer.unlink()
            return
        source = self.save_dir / self.best_models[0][2]
        if source.exists():
            _atomic_copy(source, best_pointer)

    def save_best(self, state: dict, mae: float, epoch: int):
        """Save model if it qualifies as a top-k best."""
        payload = self._coerce_state(state)
        mae_value = _coerce_finite_float(mae, "mae")
        epoch_idx = _coerce_non_negative_int(epoch, "epoch")
        payload.setdefault("epoch", epoch_idx)
        payload.setdefault("val_mae", mae_value)

        if (
            len(self.best_models) >= self.top_k
            and self.best_models
            and mae_value >= self.best_models[-1][0] - _EPS
        ):
            return

        filename = f"best_epoch_{epoch_idx}_mae_{mae_value:.4f}.pt"
        path = self.save_dir / filename

        _atomic_torch_save(payload, path)
        self.best_models.append((mae_value, epoch_idx, filename))
        self.best_models.sort(key=lambda x: (x[0], x[1]))  # Sort by MAE ascending, then epoch

        # Prune beyond top-k
        if len(self.best_models) > self.top_k:
            worst = self.best_models.pop()
            worst_path = self.save_dir / worst[2]
            if worst_path.exists():
                worst_path.unlink()

        # Always refresh best pointer from current ranking.
        if self.best_models:
            best_source = self.save_dir / self.best_models[0][2]
            if best_source.exists():
                _atomic_copy(best_source, self.save_dir / "best.pt")
            if self.best_models[0][1] == epoch_idx:
                logger.info(f"New best model: epoch={epoch_idx}, MAE={mae_value:.4f}")

    def save_checkpoint(self, state: dict, epoch: int):
        """Save rotating checkpoint with history."""
        payload = self._coerce_state(state)
        epoch_idx = _coerce_non_negative_int(epoch, "epoch")
        payload.setdefault("epoch", epoch_idx)

        # Delete oldest
        if self.keep_last_k > 1:
            oldest = self.save_dir / f"checkpoint_prev_{self.keep_last_k - 1}.pt"
            if oldest.exists():
                oldest.unlink()

        # Shift others
        if self.keep_last_k > 2:
            for i in range(self.keep_last_k - 2, 0, -1):
                src = self.save_dir / f"checkpoint_prev_{i}.pt"
                dst = self.save_dir / f"checkpoint_prev_{i + 1}.pt"
                if src.exists():
                    shutil.move(str(src), str(dst))

        # Move current 'checkpoint.pt' to 'prev_1'
        current = self.save_dir / "checkpoint.pt"
        if self.keep_last_k > 1 and current.exists():
            shutil.move(str(current), str(self.save_dir / "checkpoint_prev_1.pt"))

        # Save new
        _atomic_torch_save(payload, current)
