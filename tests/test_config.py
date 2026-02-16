"""Tests for atlas.config module."""

import pytest
from pathlib import Path


def test_config_instantiation():
    """Config should instantiate without errors."""
    from atlas.config import Config
    cfg = Config()
    assert cfg is not None
    assert isinstance(cfg.paths.project_root, Path)


def test_config_singleton():
    """get_config should return the same instance."""
    from atlas.config import get_config
    c1 = get_config()
    c2 = get_config()
    assert c1 is c2


def test_config_paths_exist():
    """ensure_dirs should create data and models directories."""
    from atlas.config import Config
    cfg = Config()
    assert cfg.paths.data_dir.exists()
    assert cfg.paths.models_dir.exists()


def test_config_summary():
    """summary() should return a non-empty string."""
    from atlas.config import get_config
    cfg = get_config()
    s = cfg.summary()
    assert isinstance(s, str)
    assert "ATLAS" in s
    assert "JARVIS" in s


def test_dft_config_defaults():
    """DFT config should have sensible defaults."""
    from atlas.config import DFTConfig
    dft = DFTConfig()
    assert dft.encut > 0
    assert dft.kpoints_density > 0
    assert dft.ediff > 0


def test_mace_config_defaults():
    """MACE config should have sensible defaults."""
    from atlas.config import MACEConfig
    mace = MACEConfig()
    assert mace.r_max > 0
    assert mace.max_epochs > 0
    assert mace.lr > 0
    assert mace.batch_size > 0
