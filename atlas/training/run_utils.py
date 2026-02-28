"""
Utilities for experiment run directory management and reproducibility manifests.

Provides consistent behavior for:
- creating isolated run directories
- resuming from the latest run or a specific run id
- emitting run_manifest v2 (JSON canonical + YAML mirror)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_MANIFEST_SCHEMA_VERSION = "2.0"
RUN_MANIFEST_VISIBILITY = ("internal", "public")
PrivacyRedactionMap = dict[str, str]


@dataclass
class RunManifestV2:
    """Typed shape helper for run_manifest v2."""

    schema_version: str = RUN_MANIFEST_SCHEMA_VERSION
    visibility: str = "internal"
    created_at: str = ""
    updated_at: str = ""
    run_id: str = ""
    runtime: dict[str, Any] | None = None
    args: dict[str, Any] | None = None
    dataset: dict[str, Any] | None = None
    split: dict[str, Any] | None = None
    environment_lock: dict[str, Any] | None = None
    artifacts: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    seeds: dict[str, Any] | None = None
    configs: dict[str, Any] | None = None


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


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _env_manifest_defaults(project_root: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build default dataset/split blocks from environment contracts."""
    root = project_root or Path.cwd()
    dataset_block: dict[str, Any] = {}
    split_block: dict[str, Any] = {}

    source_key = os.environ.get("ATLAS_DATASET_SOURCE_KEY", "").strip()
    snapshot_id = os.environ.get("ATLAS_DATASET_SNAPSHOT_ID", "").strip()
    if source_key:
        dataset_block["source_key"] = source_key
    if snapshot_id:
        dataset_block["snapshot_id"] = snapshot_id

    split_manifest_env = os.environ.get("ATLAS_SPLIT_MANIFEST", "").strip()
    if split_manifest_env:
        manifest_path = Path(split_manifest_env)
        if not manifest_path.is_absolute():
            manifest_path = (root / manifest_path).resolve()
        split_block["manifest_path"] = str(manifest_path)
        split_block["manifest_hash"] = _file_sha256(manifest_path)
        if manifest_path.exists():
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    split_id = payload.get("split_id")
                    split_hash = payload.get("split_hash")
                    strategy = payload.get("split_strategy")
                    if split_id:
                        split_block["split_id"] = str(split_id)
                    if split_hash:
                        split_block["split_hash"] = str(split_hash)
                    if strategy:
                        split_block["split_strategy"] = str(strategy)
            except Exception:
                pass
    return dataset_block, split_block


def _resolve_environment_lock(
    *,
    project_root: Path | None,
    strict_lock: bool | None = None,
) -> dict[str, Any]:
    root = project_root or Path.cwd()
    if strict_lock is not None:
        wants_strict = bool(strict_lock)
    else:
        raw = str(os.environ.get("ATLAS_STRICT_LOCK", "0") or "0").strip().lower()
        wants_strict = raw in {"1", "true", "yes", "on"}
    strict_path = root / "requirements-lock.txt"
    light_path = root / "requirements.txt"
    chosen = strict_path if wants_strict and strict_path.exists() else light_path
    return {
        "strict_lock": bool(wants_strict and strict_path.exists()),
        "lock_file": str(chosen),
        "lock_hash": _file_sha256(chosen),
    }


def _redact_path_string(raw: str) -> str:
    text = raw.replace("\\", "/")
    if ":" in text:
        _, tail = text.split(":", 1)
    else:
        tail = text
    parts = [p for p in tail.split("/") if p]
    if not parts:
        return "<redacted-path>"
    return f".../{parts[-1]}"


def _redact_public_payload(payload: Any) -> Any:
    """Recursively redact path/device-identifying fields for public visibility."""
    if isinstance(payload, dict):
        redacted: dict[str, Any] = {}
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered in {"hostname", "pid", "user", "username"}:
                redacted[key] = "<redacted>"
                continue
            if lowered in {"cwd", "project_root", "python_executable"}:
                redacted[key] = "<redacted>"
                continue
            if lowered.endswith("_path") or lowered.endswith("_dir") or lowered.endswith("_file"):
                if isinstance(value, (str, Path)):
                    redacted[key] = "<redacted-path>"
                else:
                    redacted[key] = _redact_public_payload(value)
                continue
            redacted[key] = _redact_public_payload(value)
        return redacted
    if isinstance(payload, list):
        return [_redact_public_payload(v) for v in payload]
    if isinstance(payload, str):
        if ("\\" in payload or "/" in payload) and (":" in payload or payload.startswith("/")):
            return _redact_path_string(payload)
        return payload
    return payload


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
    schema_version: str = RUN_MANIFEST_SCHEMA_VERSION,
    visibility: str | None = None,
    dataset_block: dict[str, Any] | None = None,
    split_block: dict[str, Any] | None = None,
    environment_lock_block: dict[str, Any] | None = None,
    artifacts_block: dict[str, Any] | None = None,
    metrics_block: dict[str, Any] | None = None,
    seeds_block: dict[str, Any] | None = None,
    configs_block: dict[str, Any] | None = None,
    strict_lock: bool | None = None,
    emit_yaml: bool = True,
) -> Path:
    """
    Write/update run_manifest.json under a run directory.

    JSON is canonical. YAML mirror is emitted for human readability.
    """
    effective_visibility = (
        visibility
        or os.environ.get("ATLAS_MANIFEST_VISIBILITY", "internal").strip().lower()
    )
    if effective_visibility not in RUN_MANIFEST_VISIBILITY:
        raise ValueError(
            f"Unsupported visibility '{effective_visibility}'. Expected one of {RUN_MANIFEST_VISIBILITY}."
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = save_dir / "run_manifest.json"
    manifest: dict[str, Any] = {}
    if merge_existing and manifest_path.exists():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict):
                manifest = prev
        except Exception:
            manifest = {}

    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    manifest.setdefault("created_at", now)
    manifest["updated_at"] = now
    manifest["schema_version"] = schema_version
    manifest["visibility"] = effective_visibility
    manifest["run_id"] = save_dir.name
    runtime = collect_runtime_context(project_root=project_root)
    if effective_visibility == "public":
        runtime = _redact_public_payload(runtime)
    manifest["runtime"] = runtime
    if args is not None:
        args_payload = _json_safe(args)
        if effective_visibility == "public":
            args_payload = _redact_public_payload(args_payload)
        manifest["args"] = args_payload
    elif "args" not in manifest:
        manifest["args"] = {}

    manifest.setdefault("dataset", {})
    manifest.setdefault("split", {})
    manifest.setdefault("environment_lock", {})
    manifest.setdefault("artifacts", {})
    manifest.setdefault("metrics", {})
    manifest.setdefault("seeds", {})
    manifest.setdefault("configs", {})

    env_dataset_block, env_split_block = _env_manifest_defaults(project_root=project_root)
    if env_dataset_block:
        manifest["dataset"].update(_json_safe(env_dataset_block))
    if env_split_block:
        manifest["split"].update(_json_safe(env_split_block))
    if dataset_block:
        manifest["dataset"].update(_json_safe(dataset_block))
    if split_block:
        manifest["split"].update(_json_safe(split_block))
    if artifacts_block:
        artifacts_payload = _json_safe(artifacts_block)
        if effective_visibility == "public":
            artifacts_payload = _redact_public_payload(artifacts_payload)
        manifest["artifacts"].update(artifacts_payload)
    if metrics_block:
        manifest["metrics"].update(_json_safe(metrics_block))
    if seeds_block:
        manifest["seeds"].update(_json_safe(seeds_block))
    if configs_block:
        manifest["configs"].update(_json_safe(configs_block))

    env_lock = _resolve_environment_lock(project_root=project_root, strict_lock=strict_lock)
    manifest["environment_lock"].update(env_lock)
    if environment_lock_block:
        manifest["environment_lock"].update(_json_safe(environment_lock_block))

    if extra:
        extra_payload = _json_safe(extra)
        if effective_visibility == "public":
            extra_payload = _redact_public_payload(extra_payload)
        manifest.update(extra_payload)

    # Fill minimum reproducibility blocks even when callers do not provide them.
    seed_payload: dict[str, Any] = {}
    args_obj = manifest.get("args", {})
    if isinstance(args_obj, dict):
        for key in ("seed", "split_seed", "preflight_split_seed"):
            if key in args_obj and args_obj.get(key) is not None:
                seed_payload[key] = args_obj.get(key)
    if not seed_payload:
        env_seed = os.environ.get("ATLAS_SEED", "").strip()
        if env_seed:
            seed_payload["global_seed"] = env_seed
    if not seed_payload:
        seed_payload["global_seed"] = 42
    manifest["seeds"].update(_json_safe(seed_payload))

    config_payload = {
        "entrypoint": str(sys.argv[0]) if sys.argv else "",
        "python_version_short": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    manifest["configs"].update(_json_safe(config_payload))

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if emit_yaml:
        # JSON is valid YAML 1.2; writing mirror without adding a hard dependency.
        yaml_path = save_dir / "run_manifest.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    manifest["artifacts"].setdefault("run_manifest_json", str(manifest_path))
    if emit_yaml:
        manifest["artifacts"].setdefault("run_manifest_yaml", str(save_dir / "run_manifest.yaml"))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    if emit_yaml:
        with open(save_dir / "run_manifest.yaml", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest_path
