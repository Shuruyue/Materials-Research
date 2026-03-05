"""Unit tests for training preflight gates."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import atlas.training.preflight as preflight


def _patch_config(monkeypatch: pytest.MonkeyPatch, artifacts_dir: Path) -> None:
    cfg = SimpleNamespace(paths=SimpleNamespace(artifacts_dir=artifacts_dir))
    monkeypatch.setattr(preflight, "get_config", lambda: cfg)


def test_run_preflight_dry_run_creates_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_config(monkeypatch, tmp_path / "artifacts")
    result = preflight.run_preflight(project_root=tmp_path, dry_run=True)
    assert result.return_code == 0
    assert result.validation_report.parent.exists()
    assert result.split_dir.exists()


def test_run_preflight_validates_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_config(monkeypatch, tmp_path / "artifacts")
    with pytest.raises(ValueError, match="property_group"):
        preflight.run_preflight(project_root=tmp_path, property_group="", dry_run=True)
    with pytest.raises(ValueError, match="max_samples"):
        preflight.run_preflight(project_root=tmp_path, max_samples=-1, dry_run=True)
    with pytest.raises(ValueError, match="timeout_sec"):
        preflight.run_preflight(project_root=tmp_path, timeout_sec=0, dry_run=True)
    with pytest.raises(ValueError, match="timeout_sec"):
        preflight.run_preflight(project_root=tmp_path, timeout_sec=0.7, dry_run=True)
    with pytest.raises(ValueError, match="max_samples"):
        preflight.run_preflight(project_root=tmp_path, max_samples=True, dry_run=True)
    with pytest.raises(ValueError, match="split_seed"):
        preflight.run_preflight(project_root=tmp_path, split_seed=np.bool_(False), dry_run=True)


def test_run_preflight_returns_timeout_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_config(monkeypatch, tmp_path / "artifacts")

    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="atlas.data.data_validation", timeout=1)

    monkeypatch.setattr(preflight.subprocess, "run", _timeout)
    result = preflight.run_preflight(project_root=tmp_path, timeout_sec=1, dry_run=False)
    assert result.return_code == 124
    assert result.error_message == "validate-data failed: timeout"


def test_run_preflight_reports_missing_manifests(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_config(monkeypatch, tmp_path / "artifacts")
    artifacts_dir = tmp_path / "artifacts"

    def _fake_run(cmd, **_kwargs):
        if "atlas.data.data_validation" in cmd:
            report = artifacts_dir / "preflight" / "validation_report_preflight.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(preflight.subprocess, "run", _fake_run)
    result = preflight.run_preflight(project_root=tmp_path, dry_run=False)
    assert result.return_code == 2
    assert result.error_message == "missing split manifests"


def test_run_preflight_success_when_split_manifests_emitted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_config(monkeypatch, tmp_path / "artifacts")

    def _fake_run(cmd, **_kwargs):
        if "atlas.data.data_validation" in cmd:
            output = Path(cmd[cmd.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("{}", encoding="utf-8")
        if "atlas.data.split_governance" in cmd:
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in preflight._REQUIRED_SPLIT_MANIFESTS:
                (output_dir / name).write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(preflight.subprocess, "run", _fake_run)
    result = preflight.run_preflight(project_root=tmp_path, dry_run=False)
    assert result.return_code == 0
    assert result.error_message is None


def test_run_preflight_reports_missing_validation_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_config(monkeypatch, tmp_path / "artifacts")
    monkeypatch.setattr(preflight.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    result = preflight.run_preflight(project_root=tmp_path, dry_run=False)
    assert result.return_code == 2
    assert result.error_message == "missing validation report"


def test_run_preflight_returns_127_on_oserror(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_config(monkeypatch, tmp_path / "artifacts")

    def _missing(*_args, **_kwargs):
        raise FileNotFoundError("python not found")

    monkeypatch.setattr(preflight.subprocess, "run", _missing)
    result = preflight.run_preflight(project_root=tmp_path, dry_run=False)
    assert result.return_code == 127
    assert result.error_message == "validate-data failed: oserror:FileNotFoundError"


def test_run_preflight_reports_split_stage_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_config(monkeypatch, tmp_path / "artifacts")
    artifacts_dir = tmp_path / "artifacts"
    call_count = {"value": 0}

    def _fake_run(cmd, **_kwargs):
        call_count["value"] += 1
        if "atlas.data.data_validation" in cmd:
            output = artifacts_dir / "preflight" / "validation_report_preflight.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("{}", encoding="utf-8")
            return SimpleNamespace(returncode=0)
        raise subprocess.TimeoutExpired(cmd="atlas.data.split_governance", timeout=1)

    monkeypatch.setattr(preflight.subprocess, "run", _fake_run)
    result = preflight.run_preflight(project_root=tmp_path, timeout_sec=1, dry_run=False)
    assert call_count["value"] >= 2
    assert result.return_code == 124
    assert result.error_message == "make-splits failed: timeout"
