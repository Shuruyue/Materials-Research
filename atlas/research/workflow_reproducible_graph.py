"""
Reproducible workflow scaffold for graph-first discovery runs.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from atlas.config import get_config
from atlas.utils.reproducibility import collect_runtime_metadata


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


class WorkflowReproducibleGraph:
    """
    Minimal reproducibility layer inspired by workflow/benchmark-first repos.
    """

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
        return f"{ts}-{data_source_key}-{method_key}"

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
            stage_plan=stage_plan or list(self.DEFAULT_STAGE_PLAN),
            started_at=started_at,
        )
        if extra_metrics:
            self.manifest.metrics.update(extra_metrics)
        self._manifest_path = self.output_dir / f"{run_id}.json"
        self._persist()
        return self.manifest

    def record_iteration(self, snapshot: IterationSnapshot):
        if self.manifest is None:
            self.start()
        self.manifest.iterations.append(snapshot)  # type: ignore[union-attr]
        self._persist()

    def set_metric(self, key: str, value: Any):
        if self.manifest is None:
            self.start()
        self.manifest.metrics[key] = value  # type: ignore[union-attr]
        self._persist()

    def finalize(self, status: str = "completed", extra_metrics: dict[str, Any] | None = None):
        if self.manifest is None:
            self.start()
        self.manifest.status = status  # type: ignore[union-attr]
        self.manifest.ended_at = time.time()  # type: ignore[union-attr]
        if extra_metrics:
            self.manifest.metrics.update(extra_metrics)  # type: ignore[union-attr]
        self._persist()

    def _persist(self):
        if self.manifest is None or self._manifest_path is None:
            return
        payload = asdict(self.manifest)
        with open(self._manifest_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, default=str)
