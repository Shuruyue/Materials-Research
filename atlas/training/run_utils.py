"""
Utilities for experiment run directory management.

Provides consistent behavior for:
- creating isolated run directories
- resuming from the latest run or a specific run id
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any


def list_run_dirs(base_dir: Path, prefix: str = "run_") -> list[Path]:
    """Return sorted run directories under a base path."""
    if not base_dir.exists():
        return []
    runs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    runs.sort(key=lambda p: p.name)
    return runs


def latest_run_dir(base_dir: Path, prefix: str = "run_") -> Path | None:
    """Return the latest run directory or None."""
    runs = list_run_dirs(base_dir, prefix=prefix)
    return runs[-1] if runs else None


def resolve_run_dir(
    base_dir: Path,
    *,
    resume: bool,
    run_id: str | None = None,
    prefix: str = "run_",
) -> tuple[Path, bool]:
    """
    Resolve a run directory.

    Returns (run_dir, created_new).
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    if run_id:
        run_name = run_id if run_id.startswith(prefix) else f"{prefix}{run_id}"
        run_dir = base_dir / run_name
        if resume:
            if not run_dir.exists():
                raise FileNotFoundError(f"Requested resume run does not exist: {run_dir}")
            return run_dir, False
        if run_dir.exists():
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. Use --resume or another --run-id."
            )
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir, True

    if resume:
        latest = latest_run_dir(base_dir, prefix=prefix)
        if latest is not None:
            return latest, False

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{prefix}{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, True


def _json_safe(value: Any) -> Any:
    """Convert objects into JSON-serializable structures."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def _run_git(args: list[str], project_root: Path | None) -> str | None:
    cwd = str(project_root) if project_root is not None else None
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def collect_runtime_context(project_root: Path | None = None) -> dict[str, Any]:
    """Collect reproducibility metadata for a training run."""
    context: dict[str, Any] = {
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    if project_root is not None:
        context["project_root"] = str(project_root)

    commit = _run_git(["rev-parse", "HEAD"], project_root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], project_root)
    status = _run_git(["status", "--porcelain"], project_root)
    if commit is not None or branch is not None or status is not None:
        context["git"] = {
            "commit": commit,
            "branch": branch,
            "dirty": bool(status) if status is not None else None,
        }
    return context


def write_run_manifest(
    save_dir: Path,
    *,
    args: Any | None = None,
    extra: dict[str, Any] | None = None,
    project_root: Path | None = None,
    merge_existing: bool = True,
) -> Path:
    """
    Write/update run_manifest.json under a run directory.

    The file is overwritten with merged content when merge_existing=True.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = save_dir / "run_manifest.json"
    manifest: dict[str, Any] = {}
    if merge_existing and manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict):
                manifest = prev
        except Exception:
            manifest = {}

    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    manifest.setdefault("created_at", now)
    manifest["updated_at"] = now
    manifest["run_id"] = save_dir.name
    manifest["runtime"] = collect_runtime_context(project_root=project_root)
    if args is not None:
        manifest["args"] = _json_safe(args)
    if extra:
        manifest.update(_json_safe(extra))

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path
