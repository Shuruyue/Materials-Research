"""Tests for reproducibility helpers."""

import os
import random

import numpy as np

from atlas.utils.reproducibility import _coerce_bool, _coerce_seed, collect_runtime_metadata, set_global_seed


def test_set_global_seed_reproducible_random_streams():
    set_global_seed(123, deterministic=True)
    a_py = random.random()
    a_np = np.random.rand()

    set_global_seed(123, deterministic=True)
    b_py = random.random()
    b_np = np.random.rand()

    assert a_py == b_py
    assert a_np == b_np


def test_runtime_metadata_has_core_fields():
    meta = collect_runtime_metadata()
    assert "python_version" in meta
    assert "platform" in meta
    assert "numpy_version" in meta
    assert "deterministic_enabled" in meta


def test_set_global_seed_normalizes_non_finite_input():
    meta = set_global_seed(float("nan"), deterministic=True)
    assert meta["seed"] == 42
    assert meta["python_hash_seed"] == "42"


def test_set_global_seed_normalizes_negative_seed_to_uint32_range():
    meta = set_global_seed(-1, deterministic=False)
    assert meta["seed"] == (2**32 - 1)
    assert meta["python_hash_seed"] == str(2**32 - 1)


def test_set_global_seed_parses_hex_and_scientific_seed_strings():
    assert _coerce_seed("0x10") == 16
    assert _coerce_seed("1e3") == 1000


def test_coerce_seed_rejects_bool_and_non_integral_float():
    assert _coerce_seed(True, default=42) == 42
    assert _coerce_seed(np.bool_(True), default=42) == 42
    assert _coerce_seed(12.7, default=42) == 42
    assert _coerce_seed("12.3", default=42) == 42


def test_coerce_bool_rejects_non_integral_numeric_floats():
    assert _coerce_bool(np.bool_(False), default=True) is False
    assert _coerce_bool(0.0, default=True) is False
    assert _coerce_bool(1.0, default=False) is True
    assert _coerce_bool(0.7, default=True) is True
    assert _coerce_bool("0.2", default=False) is False


def test_coerce_bool_accepts_only_binary_numeric_values():
    assert _coerce_bool(2, default=False) is False
    assert _coerce_bool(-1, default=True) is True
    assert _coerce_bool("2", default=False) is False
    assert _coerce_bool("2.0", default=True) is True


def test_set_global_seed_configures_cublas_workspace(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    meta = set_global_seed(77, deterministic=True)
    if meta["torch_available"]:
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_set_global_seed_parses_bool_like_deterministic_flag():
    meta = set_global_seed(99, deterministic="false")
    assert meta["deterministic_requested"] is False


def test_set_global_seed_clears_known_deterministic_cublas_config_when_disabled(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    meta = set_global_seed(7, deterministic=False)
    assert meta["cublas_workspace_config"] is None
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None


def test_set_global_seed_preserves_custom_cublas_config_when_disabled(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "custom-config")
    meta = set_global_seed(8, deterministic=False)
    assert meta["cublas_workspace_config"] == "custom-config"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == "custom-config"


def test_set_global_seed_enforces_deterministic_cublas_config_when_enabled(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "custom-config")
    meta = set_global_seed(8, deterministic=True)
    assert meta["cublas_workspace_config"] == ":4096:8"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_set_global_seed_preserves_known_deterministic_cublas_config_when_enabled(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    meta = set_global_seed(9, deterministic=True)
    assert meta["cublas_workspace_config"] == ":16:8"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":16:8"


def test_set_global_seed_can_disable_deterministic_algorithms():
    try:
        import torch
    except Exception:
        return

    before = (
        bool(torch.are_deterministic_algorithms_enabled())
        if hasattr(torch, "are_deterministic_algorithms_enabled")
        else None
    )
    try:
        meta_on = set_global_seed(123, deterministic=True)
        if meta_on["torch_available"] and before is not None:
            assert meta_on["deterministic_enabled"] is True

        meta_off = set_global_seed(123, deterministic=False)
        if meta_off["torch_available"] and before is not None:
            assert meta_off["deterministic_enabled"] is False
    finally:
        if before is not None:
            torch.use_deterministic_algorithms(before, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = before
                torch.backends.cudnn.benchmark = not before


def test_runtime_metadata_reports_reproducibility_runtime_fields():
    meta = collect_runtime_metadata()
    assert "cublas_workspace_config" in meta
    assert "cuda_device_count" in meta
