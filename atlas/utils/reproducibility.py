"""Reproducibility and runtime metadata helpers."""

from __future__ import annotations

import math
import os
import platform
import random
import sys
from contextlib import suppress
from typing import Any

import numpy as np

_UINT32_MOD = 2**32
_DETERMINISTIC_CUBLAS_CONFIGS = {":16:8", ":4096:8"}
_DEFAULT_DETERMINISTIC_CUBLAS_CONFIG = ":4096:8"


def _is_boolean_like(value: Any) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}


def _is_integral_float(value: float, *, tol: float = 1e-9) -> bool:
    """Return True when a finite float is effectively integral."""
    if not math.isfinite(value):
        return False
    return abs(value - round(value)) <= tol


def _coerce_bool(value: Any, default: bool = True) -> bool:
    """Parse bool-like inputs robustly (CLI/env friendly)."""
    if _is_boolean_like(value):
        return bool(value)
    if value is None:
        return bool(default)
    if isinstance(value, (int, np.integer)):
        integer = int(value)
        if integer in (0, 1):
            return bool(integer)
        return bool(default)
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        if not _is_integral_float(scalar):
            return bool(default)
        integer = int(round(scalar))
        if integer in (0, 1):
            return bool(integer)
        return bool(default)
    text = str(value).strip().lower()
    if not text:
        return bool(default)
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    # Accept numeric string booleans ("0.0", "1.0") only if integral-like.
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return bool(default)
    if not _is_integral_float(numeric):
        return bool(default)
    integer = int(round(numeric))
    if integer in (0, 1):
        return bool(integer)
    return bool(default)


def _coerce_seed(seed: Any, default: int = 42) -> int:
    """Normalize arbitrary seed input into uint32 range."""
    if _is_boolean_like(seed):
        raw = int(default)
    elif isinstance(seed, (int, np.integer)):
        raw = int(seed)
    elif isinstance(seed, (float, np.floating)):
        scalar = float(seed)
        if not _is_integral_float(scalar):
            raw = int(default)
        else:
            raw = int(round(scalar))
    else:
        text = str(seed).strip()
        if not text:
            raw = int(default)
        else:
            try:
                raw = int(text, 0)
            except (TypeError, ValueError):
                try:
                    f = float(text)
                except (TypeError, ValueError, OverflowError):
                    raw = int(default)
                else:
                    raw = int(round(f)) if _is_integral_float(f) else int(default)
    return int(raw % _UINT32_MOD)


def _enable_torch_determinism(torch_module: Any, deterministic_requested: bool) -> bool | None:
    """Best-effort deterministic configuration across torch versions."""
    if hasattr(torch_module, "use_deterministic_algorithms"):
        try:
            torch_module.use_deterministic_algorithms(deterministic_requested, warn_only=True)
        except TypeError:
            # Backward compatibility for older torch signatures.
            torch_module.use_deterministic_algorithms(deterministic_requested)
        except Exception:
            pass

    if hasattr(torch_module.backends, "cudnn"):
        try:
            torch_module.backends.cudnn.benchmark = not deterministic_requested
            torch_module.backends.cudnn.deterministic = deterministic_requested
            if hasattr(torch_module.backends.cudnn, "allow_tf32"):
                torch_module.backends.cudnn.allow_tf32 = not deterministic_requested
        except Exception:
            pass
    if hasattr(torch_module.backends, "cuda") and hasattr(torch_module.backends.cuda, "matmul"):
        allow_tf32 = getattr(torch_module.backends.cuda.matmul, "allow_tf32", None)
        if allow_tf32 is not None:
            with suppress(Exception):
                torch_module.backends.cuda.matmul.allow_tf32 = not deterministic_requested

    if hasattr(torch_module, "are_deterministic_algorithms_enabled"):
        try:
            return bool(torch_module.are_deterministic_algorithms_enabled())
        except Exception:
            return None
    return None


def _configure_cublas_workspace(deterministic_requested: bool) -> str | None:
    """
    Keep CUBLAS workspace config aligned with deterministic preference.

    When deterministic mode is disabled, clear only known deterministic defaults
    to avoid mutating arbitrary user-provided custom values.
    """
    if deterministic_requested:
        existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if existing not in _DETERMINISTIC_CUBLAS_CONFIGS:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = _DEFAULT_DETERMINISTIC_CUBLAS_CONFIG
        return os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing in _DETERMINISTIC_CUBLAS_CONFIGS:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    return os.environ.get("CUBLAS_WORKSPACE_CONFIG")


def set_global_seed(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """
    Set process-level random seeds.

    Returns a compact metadata dictionary for logging/manifests.
    """
    seed_i = _coerce_seed(seed, default=42)
    os.environ["PYTHONHASHSEED"] = str(seed_i)
    random.seed(seed_i)
    np.random.seed(seed_i)
    deterministic_requested = _coerce_bool(deterministic, default=True)
    cublas_workspace_config = _configure_cublas_workspace(deterministic_requested)

    meta: dict[str, Any] = {
        "seed": int(seed_i),
        "seed_input": str(seed),
        "deterministic_requested": deterministic_requested,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": cublas_workspace_config,
    }

    try:
        import torch

        torch.manual_seed(seed_i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_i)

        deterministic_enabled = _enable_torch_determinism(torch, deterministic_requested)

        meta.update(
            {
                "torch_available": True,
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "deterministic_enabled": deterministic_enabled,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            }
        )
    except Exception:
        meta.update(
            {
                "torch_available": False,
                "cuda_available": False,
                "cuda_device_count": 0,
                "deterministic_enabled": None,
            }
        )

    return meta


def collect_runtime_metadata() -> dict[str, Any]:
    """Collect lightweight environment metadata for run manifests."""
    meta: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    try:
        import torch

        meta["torch_version"] = torch.__version__
        meta["cuda_available"] = bool(torch.cuda.is_available())
        meta["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if hasattr(torch, "are_deterministic_algorithms_enabled"):
            meta["deterministic_enabled"] = bool(torch.are_deterministic_algorithms_enabled())
        else:
            meta["deterministic_enabled"] = None
    except Exception:
        meta["torch_version"] = None
        meta["cuda_available"] = False
        meta["cuda_device_count"] = 0
        meta["deterministic_enabled"] = None
    return meta

