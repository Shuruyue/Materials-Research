"""Unit tests for run manifest v2 helpers."""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pytest

import atlas.training.run_utils as run_utils
from atlas.training.run_utils import resolve_run_dir, write_run_manifest


def test_write_run_manifest_v2_required_blocks(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text("numpy==1.26.0\n", encoding="utf-8")
    run_dir = tmp_path / "run_001"
    manifest_path = write_run_manifest(
        run_dir,
        args={"seed": 42},
        project_root=tmp_path,
        dataset_block={"snapshot_id": "ds_v1"},
        split_block={"split_id": "iid_s42"},
        artifacts_block={"model_best": str(run_dir / "model_best.pt")},
        metrics_block={"mae": 0.123},
        seeds_block={"global_seed": 42},
        configs_block={"config_name": "smoke"},
    )
    assert manifest_path.exists()
    yaml_path = run_dir / "run_manifest.yaml"
    assert yaml_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "schema_version",
        "visibility",
        "runtime",
        "args",
        "dataset",
        "split",
        "environment_lock",
        "artifacts",
        "metrics",
        "seeds",
        "configs",
    ):
        assert key in payload
    assert payload["schema_version"] == "2.0"
    assert payload["split"]["split_id"] == "iid_s42"


def test_write_run_manifest_public_redaction(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text("numpy==1.26.0\n", encoding="utf-8")
    run_dir = tmp_path / "run_public"
    manifest_path = write_run_manifest(
        run_dir,
        args={"data_path": str(tmp_path / "data" / "raw")},
        project_root=tmp_path,
        visibility="public",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = payload["runtime"]
    assert payload["visibility"] == "public"
    assert runtime.get("hostname") == "<redacted>"
    assert runtime.get("cwd") == "<redacted>"
    assert runtime.get("pid") == "<redacted>"
    assert payload["environment_lock"]["lock_file"] == "<redacted-path>"
    assert payload["artifacts"]["run_manifest_json"] == "run_manifest.json"
    assert payload["artifacts"]["run_manifest_yaml"] == "run_manifest.yaml"


def test_write_run_manifest_sanitizes_non_finite_numbers_and_set_order(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text("numpy==1.26.0\n", encoding="utf-8")
    run_dir = tmp_path / "run_nan"
    manifest_path = write_run_manifest(
        run_dir,
        args={"tags": {"b", "a"}, "alpha": float("nan")},
        project_root=tmp_path,
        metrics_block={"mae": float("inf")},
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["args"]["alpha"] is None
    assert payload["metrics"]["mae"] is None
    assert payload["args"]["tags"] == ["a", "b"]


def test_write_run_manifest_repairs_corrupted_sections_when_merging(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text("numpy==1.26.0\n", encoding="utf-8")
    run_dir = tmp_path / "run_corrupt"
    run_dir.mkdir(parents=True, exist_ok=True)
    bad_payload = {"schema_version": "2.0", "metrics": [], "artifacts": "oops"}
    (run_dir / "run_manifest.json").write_text(json.dumps(bad_payload), encoding="utf-8")

    manifest_path = write_run_manifest(
        run_dir,
        project_root=tmp_path,
        metrics_block={"rmse": 0.5},
        artifacts_block={"model": "model.pt"},
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(payload["metrics"], dict)
    assert isinstance(payload["artifacts"], dict)
    assert math.isclose(payload["metrics"]["rmse"], 0.5, rel_tol=0.0, abs_tol=1e-12)


def test_write_run_manifest_extra_rejects_non_mapping_reserved_section(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text("numpy==1.26.0\n", encoding="utf-8")
    run_dir = tmp_path / "run_bad_extra"
    with pytest.raises(ValueError, match="manifest schema integrity"):
        write_run_manifest(
            run_dir,
            project_root=tmp_path,
            extra={"metrics": "invalid"},
        )


def test_write_run_manifest_repairs_invalid_created_at(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text("numpy==1.26.0\n", encoding="utf-8")
    run_dir = tmp_path / "run_bad_created_at"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"created_at": {"oops": 1}, "schema_version": "2.0"}),
        encoding="utf-8",
    )
    manifest_path = write_run_manifest(run_dir, project_root=tmp_path, merge_existing=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(payload["created_at"], str)
    assert payload["created_at"]


def test_write_run_manifest_parses_strict_lock_string(tmp_path: Path):
    req_lock = tmp_path / "requirements-lock.txt"
    req_lock.write_text("numpy==1.26.0\n", encoding="utf-8")
    run_dir = tmp_path / "run_lock"
    manifest_path = write_run_manifest(
        run_dir,
        project_root=tmp_path,
        strict_lock="true",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["environment_lock"]["strict_lock"] is True
    assert payload["environment_lock"]["lock_file"] == str(req_lock)


def test_write_run_manifest_parses_strict_lock_numpy_bool(tmp_path: Path):
    req_lock = tmp_path / "requirements-lock.txt"
    req_lock.write_text("numpy==1.26.0\n", encoding="utf-8")
    run_dir = tmp_path / "run_lock_npbool"
    manifest_path = write_run_manifest(
        run_dir,
        project_root=tmp_path,
        strict_lock=np.bool_(True),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["environment_lock"]["strict_lock"] is True
    assert payload["environment_lock"]["lock_file"] == str(req_lock)


def test_write_run_manifest_preserves_explicit_seed_block(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text("numpy==1.26.0\n", encoding="utf-8")
    run_dir = tmp_path / "run_seed"
    manifest_path = write_run_manifest(
        run_dir,
        args={},
        project_root=tmp_path,
        seeds_block={"global_seed": 123, "split_seed": 7},
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["seeds"]["global_seed"] == 123
    assert payload["seeds"]["split_seed"] == 7


def test_resolve_run_dir_rejects_path_traversal_run_id(tmp_path: Path):
    with pytest.raises(ValueError, match="path separators|path traversal"):
        resolve_run_dir(tmp_path, resume=False, run_id="../escape")
    with pytest.raises(ValueError, match="path separators|path traversal"):
        resolve_run_dir(tmp_path, resume=False, run_id="..\\escape")


def test_resolve_run_dir_rejects_invalid_prefix(tmp_path: Path):
    with pytest.raises(ValueError, match="prefix"):
        resolve_run_dir(tmp_path, resume=False, prefix="../run_")


def test_resolve_run_dir_handles_timestamp_collision(tmp_path: Path, monkeypatch):
    class _FixedDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ARG003
            return cls(2026, 3, 4, 12, 0, 0)

    monkeypatch.setattr(run_utils._dt, "datetime", _FixedDateTime)
    (tmp_path / "run_20260304_120000").mkdir(parents=True, exist_ok=True)
    run_dir, created = resolve_run_dir(tmp_path, resume=False, run_id=None)
    assert created is True
    assert run_dir.name == "run_20260304_120000_01"


def test_write_run_manifest_rejects_non_mapping_runtime_context(tmp_path: Path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("numpy==1.26.0\n", encoding="utf-8")
    run_dir = tmp_path / "run_runtime_bad"
    monkeypatch.setattr(run_utils, "collect_runtime_context", lambda project_root=None: "bad")
    with pytest.raises(TypeError, match="runtime"):
        write_run_manifest(run_dir, project_root=tmp_path)
