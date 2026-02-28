"""Preflight gates for training and benchmark launchers."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from atlas.config import get_config


@dataclass
class PreflightResult:
    return_code: int
    validation_report: Path
    split_dir: Path


def run_preflight(
    *,
    project_root: Path,
    property_group: str = "priority7",
    max_samples: int = 0,
    split_seed: int = 42,
    dry_run: bool = False,
) -> PreflightResult:
    """Run mandatory data validation and split generation gates."""
    cfg = get_config()
    preflight_dir = cfg.paths.artifacts_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    validation_report = preflight_dir / "validation_report_preflight.json"
    validation_md = preflight_dir / "validation_report_preflight.md"
    split_dir = cfg.paths.artifacts_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    validate_cmd = [
        sys.executable,
        "-m",
        "atlas.data.data_validation",
        "--property-group",
        property_group,
        "--max-samples",
        str(max_samples),
        "--split-seed",
        str(split_seed),
        "--schema-version",
        "2.0",
        "--output",
        str(validation_report),
        "--markdown",
        str(validation_md),
    ]
    split_cmd = [
        sys.executable,
        "-m",
        "atlas.data.split_governance",
        "--strategy",
        "all",
        "--seed",
        str(split_seed),
        "--property-group",
        property_group,
        "--max-samples",
        str(max_samples),
        "--output-dir",
        str(split_dir),
        "--emit-assignment",
        "--group-definition-version",
        "1",
    ]

    print("[Preflight] validate-data command:")
    print("  " + " ".join(validate_cmd))
    print("[Preflight] make-splits command:")
    print("  " + " ".join(split_cmd))

    if dry_run:
        return PreflightResult(
            return_code=0,
            validation_report=validation_report,
            split_dir=split_dir,
        )

    rc = subprocess.run(validate_cmd, cwd=str(project_root)).returncode
    if rc != 0:
        return PreflightResult(
            return_code=rc,
            validation_report=validation_report,
            split_dir=split_dir,
        )

    rc = subprocess.run(split_cmd, cwd=str(project_root)).returncode
    if rc != 0:
        return PreflightResult(
            return_code=rc,
            validation_report=validation_report,
            split_dir=split_dir,
        )

    required_manifests = [
        split_dir / "split_manifest_iid.json",
        split_dir / "split_manifest_compositional.json",
        split_dir / "split_manifest_prototype.json",
    ]
    missing = [p for p in required_manifests if not p.exists()]
    if missing:
        print("[Preflight] Missing split manifests:")
        for path in missing:
            print(f"  - {path}")
        return PreflightResult(
            return_code=2,
            validation_report=validation_report,
            split_dir=split_dir,
        )

    return PreflightResult(
        return_code=0,
        validation_report=validation_report,
        split_dir=split_dir,
    )
