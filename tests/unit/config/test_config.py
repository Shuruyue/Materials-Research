"""Tests for atlas.config module."""

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
    assert "workflow_reproducible_graph" in s
    assert "Seed" in s
    assert "Deterministic" in s


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


def test_path_config_preserves_explicit_subdirs(tmp_path):
    from atlas.config import PathConfig

    cfg = PathConfig(
        project_root=tmp_path,
        data_dir=tmp_path / "data_root",
        raw_dir=tmp_path / "custom_raw",
        processed_dir=tmp_path / "custom_processed",
        artifacts_dir=tmp_path / "custom_artifacts",
    )
    cfg.ensure_dirs()

    assert cfg.raw_dir == tmp_path / "custom_raw"
    assert cfg.processed_dir == tmp_path / "custom_processed"
    assert cfg.artifacts_dir == tmp_path / "custom_artifacts"
    assert cfg.raw_dir.exists()
    assert cfg.processed_dir.exists()
    assert cfg.artifacts_dir.exists()


def test_path_config_env_relative_resolves_under_project_root(monkeypatch, tmp_path):
    from atlas.config import PathConfig

    monkeypatch.setenv("ATLAS_DATA_DIR", "relative_data")
    cfg = PathConfig(project_root=tmp_path)
    assert cfg.data_dir == tmp_path / "relative_data"


def test_config_invalid_device_falls_back_to_cpu():
    from atlas.config import Config, TrainConfig

    cfg = Config(train=TrainConfig(device="not-a-real-device"))
    assert str(cfg.device) == "cpu"


def test_path_config_rejects_bool_candidate(tmp_path):
    from atlas.config import PathConfig

    try:
        PathConfig(project_root=tmp_path, data_dir=True)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "path-like" in str(exc)
    else:
        raise AssertionError("Expected ValueError for bool path candidate")


def test_train_config_sanitizes_invalid_payloads():
    from atlas.config import TrainConfig

    cfg = TrainConfig(
        device="  ",
        seed=-11,
        deterministic="yes",  # type: ignore[arg-type]
        num_workers=-3,
        pin_memory="off",  # type: ignore[arg-type]
    )
    assert cfg.device == "auto"
    assert cfg.seed == 0
    assert cfg.deterministic is True
    assert cfg.num_workers == 0
    assert cfg.pin_memory is False


def test_profile_config_normalizes_fallback_methods():
    from atlas.config import ProfileConfig

    profile = ProfileConfig(
        model_name=" ",
        fallback_methods=(" ", "method_a", "", "method_b"),  # type: ignore[arg-type]
    )
    assert profile.model_name == "mace_default"
    assert profile.fallback_methods == ("method_a", "method_b")
