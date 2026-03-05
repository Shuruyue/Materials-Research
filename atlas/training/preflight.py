"""Preflight gates for training and benchmark launchers."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atlas.config import get_config

_DEFAULT_PREFLIGHT_TIMEOUT_SEC = 1800
_SAFE_PROPERTY_GROUP = re.compile(r"^[A-Za-z0-9._-]+$")
_REQUIRED_SPLIT_MANIFESTS = (
    "split_manifest_iid.json",
    "split_manifest_compositional.json",
    "split_manifest_prototype.json",
)


@dataclass
class PreflightResult:
    return_code: int
    validation_report: Path
    split_dir: Path
    error_message: str | None = None


@dataclass
class _CommandResult:
    return_code: int
    error_reason: str | None = None


def _is_boolean_like(value: Any) -> bool:
    return isinstance(value, bool) or type(value).__name__ == "bool_"


def _coerce_non_negative_int(value: int, name: str) -> int:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be an integer >= 0, got bool")
    if isinstance(value, int):
        scalar = int(value)
    elif isinstance(value, float):
        if not (value.is_integer() and value >= 0):
            raise ValueError(f"{name} must be an integer >= 0")
        scalar = int(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{name} must be an integer >= 0")
        try:
            scalar = int(text, 10)
        except ValueError:
            try:
                parsed = float(text)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer >= 0") from exc
            if not (parsed.is_integer() and parsed >= 0):
                raise ValueError(f"{name} must be an integer >= 0") from None
            scalar = int(parsed)
    if scalar < 0:
        raise ValueError(f"{name} must be >= 0")
    return scalar


def _coerce_positive_int(value: Any, name: str) -> int:
    scalar = _coerce_non_negative_int(value, name)
    if scalar <= 0:
        raise ValueError(f"{name} must be > 0")
    return scalar


def _validate_property_group(value: str) -> str:
    group = str(value).strip()
    if not group:
        raise ValueError("property_group must not be empty")
    if not _SAFE_PROPERTY_GROUP.fullmatch(group):
        raise ValueError("property_group contains unsupported characters")
    return group


def _run_command(cmd: list[str], *, project_root: Path, timeout_sec: int) -> _CommandResult:
    try:
        return _CommandResult(
            return_code=subprocess.run(
                cmd,
                cwd=str(project_root),
                check=False,
                timeout=timeout_sec,
            ).returncode
        )
    except subprocess.TimeoutExpired:
        print(f"[Preflight] Command timed out after {timeout_sec}s:")
        print("  " + " ".join(cmd))
        return _CommandResult(return_code=124, error_reason="timeout")
    except OSError as exc:
        print(f"[Preflight] Failed to execute command ({exc.__class__.__name__}):")
        print("  " + " ".join(cmd))
        return _CommandResult(
            return_code=127,
            error_reason=f"oserror:{exc.__class__.__name__}",
        )


def _format_stage_error(stage: str, detail: str | None = None) -> str:
    if detail:
        return f"{stage} failed: {detail}"
    return f"{stage} failed"


def _run_stage(
    *,
    stage_name: str,
    cmd: list[str],
    project_root: Path,
    timeout_sec: int,
    validation_report: Path,
    split_dir: Path,
) -> PreflightResult | None:
    cmd_result = _run_command(cmd, project_root=project_root, timeout_sec=timeout_sec)
    if cmd_result.return_code == 0:
        return None
    return PreflightResult(
        return_code=cmd_result.return_code,
        validation_report=validation_report,
        split_dir=split_dir,
        error_message=_format_stage_error(stage_name, cmd_result.error_reason),
    )


def _is_non_empty_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def run_preflight(
    *,
    project_root: Path,
    property_group: str = "priority7",
    max_samples: int = 0,
    split_seed: int = 42,
    dry_run: bool = False,
    timeout_sec: int = _DEFAULT_PREFLIGHT_TIMEOUT_SEC,
) -> PreflightResult:
    """Run mandatory data validation and split generation gates."""
    project_root = Path(project_root).expanduser()
    if not project_root.exists():
        raise FileNotFoundError(f"project_root does not exist: {project_root}")
    if not project_root.is_dir():
        raise NotADirectoryError(f"project_root must be a directory: {project_root}")

    property_group = _validate_property_group(property_group)
    max_samples = _coerce_non_negative_int(max_samples, "max_samples")
    split_seed = _coerce_non_negative_int(split_seed, "split_seed")
    timeout_sec = _coerce_positive_int(timeout_sec, "timeout_sec")

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

    validate_failure = _run_stage(
        stage_name="validate-data",
        cmd=validate_cmd,
        project_root=project_root,
        timeout_sec=timeout_sec,
        validation_report=validation_report,
        split_dir=split_dir,
    )
    if validate_failure is not None:
        return validate_failure
    if not _is_non_empty_file(validation_report):
        return PreflightResult(
            return_code=2,
            validation_report=validation_report,
            split_dir=split_dir,
            error_message="missing validation report",
        )

    split_failure = _run_stage(
        stage_name="make-splits",
        cmd=split_cmd,
        project_root=project_root,
        timeout_sec=timeout_sec,
        validation_report=validation_report,
        split_dir=split_dir,
    )
    if split_failure is not None:
        return split_failure

    required_manifests = [split_dir / filename for filename in _REQUIRED_SPLIT_MANIFESTS]
    missing = [p for p in required_manifests if not _is_non_empty_file(p)]
    if missing:
        print("[Preflight] Missing split manifests:")
        for path in missing:
            print(f"  - {path}")
        return PreflightResult(
            return_code=2,
            validation_report=validation_report,
            split_dir=split_dir,
            error_message="missing split manifests",
        )

    return PreflightResult(
        return_code=0,
        validation_report=validation_report,
        split_dir=split_dir,
    )
