"""Reproducibility and runtime metadata helpers."""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any, Dict

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> Dict[str, Any]:
    """
    Set process-level random seeds.

    Returns a compact metadata dictionary for logging/manifests.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    meta: Dict[str, Any] = {
        "seed": int(seed),
        "deterministic_requested": bool(deterministic),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

        meta.update(
            {
                "torch_available": True,
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
    except Exception:
        meta.update({"torch_available": False})

    return meta


def collect_runtime_metadata() -> Dict[str, Any]:
    """Collect lightweight environment metadata for run manifests."""
    meta: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    try:
        import torch

        meta["torch_version"] = torch.__version__
        meta["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        meta["torch_version"] = None
        meta["cuda_available"] = False
    return meta

