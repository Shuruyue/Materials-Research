"""Unit tests for run manifest v2 helpers."""

from __future__ import annotations

import json
from pathlib import Path

from atlas.training.run_utils import write_run_manifest


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
