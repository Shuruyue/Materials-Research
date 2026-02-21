"""Tests for reproducibility helpers."""

import random

import numpy as np

from atlas.utils.reproducibility import collect_runtime_metadata, set_global_seed


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
