"""
Reproducible workflow scaffold for graph-first discovery runs.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from atlas.config import get_config
from atlas.utils.reproducibility import collect_runtime_metadata

_RUN_ID_TOKEN_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_run_token(value: str) -> str:
    text = _RUN_ID_TOKEN_PATTERN.sub("-", value.strip())
    text = text.strip("-._")
    return text or "unknown"


def _is_boolean_like(value: object) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}


def _normalize_non_empty_string(value: object, *, name: str) -> str:
    if _is_boolean_like(value) or not isinstance(value, str):
        raise ValueError(f"{name} must be a non-empty string")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


def _coerce_bool(value: object, name: str) -> bool:
    if _is_boolean_like(value):
        return bool(value)
    if isinstance(value, Integral):
        integer = int(value)
        if integer in (0, 1):
            return bool(integer)
        raise ValueError(f"{name} integer payload must be 0/1")
    if isinstance(value, Real):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        rounded = round(scalar)
        if abs(scalar - rounded) > 1e-9:
            raise ValueError(f"{name} must be bool-like")
        integer = int(rounded)
        if integer in (0, 1):
            return bool(integer)
        raise ValueError(f"{name} numeric payload must be 0/1")
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be bool-like")


def _ensure_non_negative_int(value: object, name: str) -> int:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be integer-valued, not boolean")
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        rounded = round(scalar)
        if abs(scalar - rounded) > 1e-9:
            raise ValueError(f"{name} must be integer-valued")
        number = int(rounded)
    else:
        raise ValueError(f"{name} must be integer-valued")
    if number < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return number


def _ensure_non_negative_float(value: object, name: str) -> float:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be finite and >= 0, not boolean")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return number


def _normalize_stage_plan(stages: Sequence[str]) -> list[str]:
    if isinstance(stages, str):
        raise ValueError("stage_plan must be a sequence of stage names, not a raw string")
    if not stages:
        raise ValueError("stage_plan must not be empty")
    cleaned_stage_plan: list[str] = []
    for stage in stages:
        stage_name = _normalize_non_empty_string(stage, name="stage_plan entry")
        if stage_name not in cleaned_stage_plan:
            cleaned_stage_plan.append(stage_name)
    return cleaned_stage_plan


def _normalize_fallback_methods(methods: Sequence[str]) -> tuple[str, ...]:
    if isinstance(methods, str):
        # Preserve backward-compatibility with string payloads but avoid char-wise splitting.
        methods = [methods]
    deduped: list[str] = []
    seen: set[str] = set()
    for method in methods:
        if _is_boolean_like(method) or not isinstance(method, str):
            raise ValueError("fallback method names must be non-empty strings")
        normalized = _sanitize_run_token(method)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return tuple(deduped)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        cleaned = [_json_safe(v) for v in value]
        return sorted(cleaned, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False, default=str))
    return str(value)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_fp:
            tmp_path = Path(tmp_fp.name)
            json.dump(payload, tmp_fp, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            tmp_fp.flush()
            os.fsync(tmp_fp.fileno())
        if tmp_path is None:
            raise RuntimeError("failed to create temporary manifest file")
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            with suppress(OSError):
                tmp_path.unlink()


@dataclass
class IterationSnapshot:
    """Compact per-iteration execution summary."""

    iteration: int
    generated: int = 0
    unique: int = 0
    relaxed: int = 0
    selected: int = 0
    duration_sec: float = 0.0
    stage_timings_sec: dict[str, float] = field(default_factory=dict)
    seed_pool_size: int = 0
    status: str = "ok"
    notes: str = ""

    def __post_init__(self) -> None:
        self.iteration = _ensure_non_negative_int(self.iteration, "iteration")
        self.generated = _ensure_non_negative_int(self.generated, "generated")
        self.unique = _ensure_non_negative_int(self.unique, "unique")
        self.relaxed = _ensure_non_negative_int(self.relaxed, "relaxed")
        self.selected = _ensure_non_negative_int(self.selected, "selected")
        self.seed_pool_size = _ensure_non_negative_int(
            self.seed_pool_size,
            "seed_pool_size",
        )
        self.duration_sec = _ensure_non_negative_float(self.duration_sec, "duration_sec")
        if self.unique > self.generated:
            raise ValueError("unique must be <= generated")
        if self.relaxed > self.unique:
            raise ValueError("relaxed must be <= unique")
        if self.selected > self.relaxed:
            raise ValueError("selected must be <= relaxed")

        cleaned_timings: dict[str, float] = {}
        for key, value in self.stage_timings_sec.items():
            stage_key = str(key).strip()
            if not stage_key:
                raise ValueError("stage_timings_sec keys must be non-empty")
            cleaned_timings[stage_key] = _ensure_non_negative_float(
                value,
                f"stage_timings_sec[{stage_key!r}]",
            )
        self.stage_timings_sec = cleaned_timings
        self.status = str(self.status).strip() or "ok"


@dataclass
class RunManifest:
    """Run-level metadata for reproducibility."""

    run_id: str
    method_key: str
    data_source_key: str
    fallback_methods: Sequence[str]
    model_name: str
    relaxer_name: str
    evaluator_name: str
    seed: int
    deterministic: bool
    stage_plan: list[str]
    started_at: float
    runtime_metadata: dict[str, Any] = field(default_factory=dict)
    ended_at: float | None = None
    status: str = "running"
    metrics: dict[str, Any] = field(default_factory=dict)
    iterations: list[IterationSnapshot] = field(default_factory=list)
    schema_version: str = "1.1"

    def __post_init__(self) -> None:
        self.run_id = _sanitize_run_token(
            _normalize_non_empty_string(self.run_id, name="run_id")
        )
        self.method_key = _sanitize_run_token(
            _normalize_non_empty_string(self.method_key, name="method_key")
        )
        self.data_source_key = _sanitize_run_token(
            _normalize_non_empty_string(self.data_source_key, name="data_source_key")
        )
        self.model_name = _normalize_non_empty_string(self.model_name, name="model_name")
        self.relaxer_name = _normalize_non_empty_string(self.relaxer_name, name="relaxer_name")
        self.evaluator_name = _normalize_non_empty_string(
            self.evaluator_name,
            name="evaluator_name",
        )
        self.schema_version = _normalize_non_empty_string(
            self.schema_version,
            name="schema_version",
        )
        self.stage_plan = _normalize_stage_plan(self.stage_plan)
        self.fallback_methods = _normalize_fallback_methods(self.fallback_methods)
        self.seed = _ensure_non_negative_int(self.seed, "seed")
        self.deterministic = _coerce_bool(self.deterministic, "deterministic")
        self.started_at = _ensure_non_negative_float(self.started_at, "started_at")
        if self.ended_at is not None:
            self.ended_at = _ensure_non_negative_float(self.ended_at, "ended_at")
            if self.ended_at < self.started_at:
                raise ValueError("ended_at must be >= started_at")
        self.status = _normalize_non_empty_string(self.status, name="status")


class WorkflowReproducibleGraph:
    """Minimal reproducibility layer inspired by workflow/benchmark-first repos."""

    DEFAULT_STAGE_PLAN = [
        "ingest",
        "generate",
        "relax",
        "classify",
        "acquire",
        "report",
    ]

    def __init__(self, output_dir: Path | None = None):
        cfg = get_config()
        self.cfg = cfg
        self.profile = cfg.profile
        self.output_dir = output_dir or (cfg.paths.data_dir / "discovery_results" / "workflow_runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest: RunManifest | None = None
        self._manifest_path: Path | None = None

    @staticmethod
    def make_run_id(method_key: str, data_source_key: str, timestamp: float | None = None) -> str:
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(timestamp or time.time()))
        method_token = _sanitize_run_token(
            _normalize_non_empty_string(method_key, name="method_key")
        )
        source_token = _sanitize_run_token(
            _normalize_non_empty_string(data_source_key, name="data_source_key")
        )
        return f"{ts}-{source_token}-{method_token}"

    def _ensure_manifest(self) -> RunManifest:
        if self.manifest is None:
            self.start()
        assert self.manifest is not None
        return self.manifest

    def _resolve_manifest_path(self, run_id: str) -> Path:
        run_token = _sanitize_run_token(run_id)
        for attempt in range(100):
            suffix = "" if attempt == 0 else f"_{attempt:02d}"
            candidate = self.output_dir / f"{run_token}{suffix}.json"
            if not candidate.exists():
                return candidate
        raise FileExistsError(
            f"Unable to allocate unique manifest file for run id {run_token!r} under {self.output_dir}"
        )

    def start(
        self,
        stage_plan: list[str] | None = None,
        extra_metrics: dict[str, Any] | None = None,
    ) -> RunManifest:
        started_at = time.time()
        run_id = self.make_run_id(self.profile.method_key, self.profile.data_source_key, started_at)
        self.manifest = RunManifest(
            run_id=run_id,
            method_key=self.profile.method_key,
            data_source_key=self.profile.data_source_key,
            fallback_methods=tuple(self.profile.fallback_methods),
            model_name=self.profile.model_name,
            relaxer_name=self.profile.relaxer_name,
            evaluator_name=self.profile.evaluator_name,
            seed=self.cfg.train.seed,
            deterministic=self.cfg.train.deterministic,
            runtime_metadata=collect_runtime_metadata(),
            stage_plan=list(self.DEFAULT_STAGE_PLAN) if stage_plan is None else stage_plan,
            started_at=started_at,
        )
        if extra_metrics:
            self.manifest.metrics.update(extra_metrics)
        self._manifest_path = self._resolve_manifest_path(run_id)
        self._persist()
        return self.manifest

    def record_iteration(self, snapshot: IterationSnapshot) -> None:
        manifest = self._ensure_manifest()
        if not isinstance(snapshot, IterationSnapshot):
            raise TypeError(f"snapshot must be IterationSnapshot, got {type(snapshot)!r}")

        if manifest.iterations:
            last_iter = manifest.iterations[-1].iteration
            if snapshot.iteration <= last_iter:
                raise ValueError(
                    f"iteration must be strictly increasing (last={last_iter}, new={snapshot.iteration})"
                )

        manifest.iterations.append(snapshot)
        self._persist()

    def set_metric(self, key: str, value: Any) -> None:
        metric_key = _normalize_non_empty_string(key, name="metric key")
        manifest = self._ensure_manifest()
        manifest.metrics[metric_key] = value
        self._persist()

    def finalize(self, status: str = "completed", extra_metrics: dict[str, Any] | None = None) -> None:
        manifest = self._ensure_manifest()
        status_value = _normalize_non_empty_string(status, name="status")
        manifest.status = status_value
        manifest.ended_at = time.time()
        if extra_metrics:
            manifest.metrics.update(extra_metrics)
        self._persist()

    def _persist(self) -> None:
        if self.manifest is None or self._manifest_path is None:
            return

        payload = asdict(self.manifest)
        safe_payload = _json_safe(payload)
        if not isinstance(safe_payload, dict):
            raise TypeError("manifest payload must serialize to a mapping")
        _atomic_json_write(self._manifest_path, safe_payload)
