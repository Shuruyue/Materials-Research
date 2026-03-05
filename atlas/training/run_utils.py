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
import math
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_MANIFEST_SCHEMA_VERSION = "2.0"
RUN_MANIFEST_VISIBILITY = ("internal", "public")
PrivacyRedactionMap = dict[str, str]
_GIT_SUBPROCESS_TIMEOUT_SECONDS = 3.0
_MANIFEST_JSON_NAME = "run_manifest.json"
_MANIFEST_YAML_NAME = "run_manifest.yaml"
_RUN_DIR_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_DICT_MANIFEST_ROOT_KEYS = {
    "runtime",
    "args",
    "dataset",
    "split",
    "environment_lock",
    "artifacts",
    "metrics",
    "seeds",
    "configs",
}


def _is_boolean_like(value: Any) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool_", "bool"}


def _coerce_bool_like(value: Any, *, default: bool = False) -> bool:
    if _is_boolean_like(value):
        return bool(value)
    if value is None:
        return bool(default)
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return bool(default)
    if isinstance(value, float):
        if not math.isfinite(value):
            return bool(default)
        if abs(value - round(value)) > 1e-9:
            return bool(default)
        rounded = int(round(value))
        if rounded in (0, 1):
            return bool(rounded)
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


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


def _validate_run_prefix(prefix: str) -> str:
    candidate = str(prefix).strip()
    if not candidate:
        raise ValueError("prefix must be a non-empty string")
    if "/" in candidate or "\\" in candidate:
        raise ValueError(f"prefix must not contain path separators: {prefix!r}")
    if ".." in candidate:
        raise ValueError(f"prefix must not contain path traversal segments: {prefix!r}")
    if not _RUN_DIR_NAME_PATTERN.fullmatch(candidate):
        raise ValueError(
            "prefix may only contain letters, numbers, '.', '_' and '-' characters"
        )
    return candidate


def _validate_manifest_payload(manifest: dict[str, Any]) -> None:
    required_str_fields = (
        "schema_version",
        "visibility",
        "created_at",
        "updated_at",
        "run_id",
    )
    for key in required_str_fields:
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"manifest[{key!r}] must be a non-empty string")

    visibility = manifest.get("visibility")
    if visibility not in RUN_MANIFEST_VISIBILITY:
        raise ValueError(
            f"Unsupported visibility '{visibility}'. Expected one of {RUN_MANIFEST_VISIBILITY}."
        )

    for key in _DICT_MANIFEST_ROOT_KEYS:
        section = manifest.get(key)
        if not isinstance(section, dict):
            raise TypeError(f"manifest[{key!r}] must be a mapping")


def _ensure_manifest_section(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    section = manifest.get(key)
    if not isinstance(section, dict):
        manifest[key] = {}
    return manifest[key]


def _dump_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
            tmp_path = Path(handle.name)
        if tmp_path is None:
            raise RuntimeError("failed to create temporary manifest file")
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            with suppress(OSError):
                tmp_path.unlink()


def _merge_extra_payload(manifest: dict[str, Any], extra_payload: dict[str, Any]) -> None:
    for key, value in extra_payload.items():
        if key in _DICT_MANIFEST_ROOT_KEYS:
            if not isinstance(value, dict):
                raise ValueError(
                    f"extra['{key}'] must be a mapping to preserve run-manifest schema integrity"
                )
            _ensure_manifest_section(manifest, key).update(value)
            continue
        manifest[key] = value


def _ensure_seed_and_config_sections(manifest: dict[str, Any]) -> None:
    seeds_section = _ensure_manifest_section(manifest, "seeds")
    configs_section = _ensure_manifest_section(manifest, "configs")

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
    for key, value in _json_safe(seed_payload).items():
        seeds_section.setdefault(key, value)

    config_payload = {
        "entrypoint": str(sys.argv[0]) if sys.argv else "",
        "python_version_short": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
    }
    configs_section.update(_json_safe(config_payload))


def _redact_manifest_sections(manifest: dict[str, Any]) -> None:
    for key in (
        "dataset",
        "split",
        "environment_lock",
        "artifacts",
        "metrics",
        "seeds",
        "configs",
    ):
        section_value = manifest.get(key, {})
        manifest[key] = _redact_public_payload(_json_safe(section_value))


def _normalize_run_name(run_id: str, prefix: str) -> str:
    prefix = _validate_run_prefix(prefix)
    candidate = run_id.strip()
    if not candidate:
        raise ValueError("run_id must be a non-empty string")
    run_name = candidate if candidate.startswith(prefix) else f"{prefix}{candidate}"
    if "/" in run_name or "\\" in run_name:
        raise ValueError(f"run_id must not contain path separators: {run_id!r}")
    if Path(run_name).name != run_name:
        raise ValueError(f"run_id must be a simple directory name: {run_id!r}")
    if ".." in run_name:
        raise ValueError(f"run_id must not contain path traversal segments: {run_id!r}")
    if not _RUN_DIR_NAME_PATTERN.fullmatch(run_name):
        raise ValueError(
            "run_id may only contain letters, numbers, '.', '_' and '-' characters"
        )
    return run_name


def _create_timestamped_run_dir(base_dir: Path, prefix: str) -> Path:
    prefix = _validate_run_prefix(prefix)
    for attempt in range(100):
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "" if attempt == 0 else f"_{attempt:02d}"
        run_dir = base_dir / f"{prefix}{ts}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise FileExistsError(
        f"Unable to create unique run directory under {base_dir} after repeated timestamp collisions."
    )


def list_run_dirs(base_dir: Path, prefix: str = "run_") -> list[Path]:
    """Return sorted run directories under a base path."""
    prefix = _validate_run_prefix(prefix)
    if not base_dir.exists():
        return []
    runs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    runs.sort(key=lambda p: p.name)
    return runs


def latest_run_dir(base_dir: Path, prefix: str = "run_") -> Path | None:
    """Return the latest run directory or None."""
    prefix = _validate_run_prefix(prefix)
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
    prefix = _validate_run_prefix(prefix)
    base_dir.mkdir(parents=True, exist_ok=True)

    if run_id:
        run_name = _normalize_run_name(str(run_id), prefix=prefix)
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

    run_dir = _create_timestamped_run_dir(base_dir, prefix=prefix)
    return run_dir, True


def _json_safe(value: Any) -> Any:
    """Convert objects into JSON-serializable structures."""
    if _is_boolean_like(value) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        return {str(k): _json_safe(v) for k, v in items}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        cleaned = [_json_safe(v) for v in value]
        return sorted(
            cleaned,
            key=lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False, default=str),
        )
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
            timeout=_GIT_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
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
            except (OSError, json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
                pass
    return dataset_block, split_block


def _resolve_environment_lock(
    *,
    project_root: Path | None,
    strict_lock: bool | None = None,
) -> dict[str, Any]:
    root = project_root or Path.cwd()
    if strict_lock is not None:
        wants_strict = _coerce_bool_like(strict_lock, default=False)
    else:
        raw = os.environ.get("ATLAS_STRICT_LOCK", "0")
        wants_strict = _coerce_bool_like(raw, default=False)
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
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise ValueError("schema_version must be a non-empty string")

    effective_visibility = (
        visibility
        or os.environ.get("ATLAS_MANIFEST_VISIBILITY", "internal").strip().lower()
    )
    if effective_visibility not in RUN_MANIFEST_VISIBILITY:
        raise ValueError(
            f"Unsupported visibility '{effective_visibility}'. Expected one of {RUN_MANIFEST_VISIBILITY}."
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = save_dir / _MANIFEST_JSON_NAME
    manifest: dict[str, Any] = {}
    if merge_existing and manifest_path.exists():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict):
                manifest = prev
        except (OSError, json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
            manifest = {}

    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    created_at = manifest.get("created_at")
    if not isinstance(created_at, str) or not created_at.strip():
        manifest["created_at"] = now
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

    dataset_section = _ensure_manifest_section(manifest, "dataset")
    split_section = _ensure_manifest_section(manifest, "split")
    environment_lock_section = _ensure_manifest_section(manifest, "environment_lock")
    artifacts_section = _ensure_manifest_section(manifest, "artifacts")
    metrics_section = _ensure_manifest_section(manifest, "metrics")
    seeds_section = _ensure_manifest_section(manifest, "seeds")
    configs_section = _ensure_manifest_section(manifest, "configs")

    env_dataset_block, env_split_block = _env_manifest_defaults(project_root=project_root)
    if env_dataset_block:
        dataset_section.update(_json_safe(env_dataset_block))
    if env_split_block:
        split_section.update(_json_safe(env_split_block))
    if dataset_block:
        dataset_section.update(_json_safe(dataset_block))
    if split_block:
        split_section.update(_json_safe(split_block))
    if artifacts_block:
        artifacts_payload = _json_safe(artifacts_block)
        if effective_visibility == "public":
            artifacts_payload = _redact_public_payload(artifacts_payload)
        artifacts_section.update(artifacts_payload)
    if metrics_block:
        metrics_section.update(_json_safe(metrics_block))
    if seeds_block:
        seeds_section.update(_json_safe(seeds_block))
    if configs_block:
        configs_section.update(_json_safe(configs_block))

    env_lock = _resolve_environment_lock(project_root=project_root, strict_lock=strict_lock)
    environment_lock_payload = _json_safe(env_lock)
    if effective_visibility == "public":
        environment_lock_payload = _redact_public_payload(environment_lock_payload)
    environment_lock_section.update(environment_lock_payload)
    if environment_lock_block:
        custom_lock_payload = _json_safe(environment_lock_block)
        if effective_visibility == "public":
            custom_lock_payload = _redact_public_payload(custom_lock_payload)
        environment_lock_section.update(custom_lock_payload)

    if extra:
        extra_payload = _json_safe(extra)
        if effective_visibility == "public":
            extra_payload = _redact_public_payload(extra_payload)
        if not isinstance(extra_payload, dict):
            raise ValueError("extra payload must serialize to a mapping")
        _merge_extra_payload(manifest, extra_payload)

    _ensure_seed_and_config_sections(manifest)

    artifacts_section.setdefault("run_manifest_json", _MANIFEST_JSON_NAME)
    yaml_path = save_dir / _MANIFEST_YAML_NAME
    if emit_yaml:
        artifacts_section.setdefault("run_manifest_yaml", _MANIFEST_YAML_NAME)

    if effective_visibility == "public":
        _redact_manifest_sections(manifest)

    manifest_payload = _json_safe(manifest)
    if not isinstance(manifest_payload, dict):
        raise TypeError("manifest payload must serialize to a mapping")
    _validate_manifest_payload(manifest_payload)
    _dump_json_file(manifest_path, manifest_payload)
    if emit_yaml:
        # JSON is valid YAML 1.2; emit a deterministic YAML-compatible mirror
        # without introducing an extra dependency.
        _dump_json_file(yaml_path, manifest_payload)

    return manifest_path
