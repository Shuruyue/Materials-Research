#!/usr/bin/env python3
"""
Phase 1 multi-round training orchestrator.

Order per round (5 stages):
  smoke -> lite -> std -> competition -> max

Default setup is tuned for a single desktop so 3 rounds can complete in a
reasonable time while still keeping model complexity progression.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_PHASE1 = PROJECT_ROOT / "scripts" / "phase1_baseline" / "run_phase1.py"


@dataclass
class StageConfig:
    name: str
    level: str | None = None
    competition: bool = False
    epochs_by_round: list[int] | None = None
    max_samples_by_round: list[int] | None = None
    batch_size_by_round: list[int] | None = None


STAGES = [
    StageConfig(name="smoke", level="smoke"),
    StageConfig(name="lite", level="lite"),
    StageConfig(name="std", level="std", epochs_by_round=[5, 6, 7], max_samples_by_round=[1000, 1500, 2000], batch_size_by_round=[32, 32, 32]),
    StageConfig(name="competition", competition=True, epochs_by_round=[8, 10, 12], max_samples_by_round=[1500, 2000, 2500], batch_size_by_round=[32, 32, 32]),
    StageConfig(name="max", level="max", epochs_by_round=[10, 12, 14], max_samples_by_round=[2000, 2500, 3000], batch_size_by_round=[16, 16, 16]),
]


def _round_value(values: list[int] | None, round_idx_1_based: int) -> int | None:
    if not values:
        return None
    idx = min(round_idx_1_based - 1, len(values) - 1)
    return values[idx]


def _latest_pro_run(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    runs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")], key=lambda p: p.name)
    return runs[-1] if runs else None


def _resolve_model_dir(stage_name: str, property_name: str) -> Path | None:
    models_dir = PROJECT_ROOT / "models"
    if stage_name in {"smoke", "lite"}:
        p = models_dir / f"cgcnn_lite_{property_name}"
        return p if p.exists() else None
    if stage_name in {"std", "competition"}:
        p = models_dir / f"cgcnn_std_{property_name}"
        return p if p.exists() else None
    if stage_name == "max":
        return _latest_pro_run(models_dir / f"cgcnn_pro_{property_name}")
    return None


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def _snapshot_stage(model_dir: Path | None, dst_dir: Path) -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)
    info: dict = {"model_dir": str(model_dir) if model_dir else None}

    if model_dir is None or not model_dir.exists():
        info["snapshot"] = "missing_model_dir"
        return info

    for name in ["results.json", "history.json", "outliers.csv", "best.pt", "checkpoint.pt"]:
        _copy_if_exists(model_dir / name, dst_dir / name)

    results_path = model_dir / "results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            info["results"] = json.load(f)
    else:
        info["results"] = None

    info["snapshot"] = "ok"
    return info


def _build_cmd(algorithm: str, property_name: str, stage: StageConfig, round_idx: int) -> list[str]:
    cmd = [sys.executable, str(RUN_PHASE1), "--algorithm", algorithm, "--property", property_name]
    if stage.competition:
        cmd.append("--competition")
    else:
        cmd.extend(["--level", stage.level or "std"])

    epochs = _round_value(stage.epochs_by_round, round_idx)
    max_samples = _round_value(stage.max_samples_by_round, round_idx)
    batch_size = _round_value(stage.batch_size_by_round, round_idx)

    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 curriculum rounds")
    parser.add_argument("--property", default="formation_energy")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--algorithms", nargs="+", default=["cgcnn"], help="Currently phase1 supports cgcnn")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    unsupported = [a for a in args.algorithms if a != "cgcnn"]
    if unsupported:
        print(f"[WARN] Unsupported phase1 algorithms skipped: {unsupported}")
    algorithms = [a for a in args.algorithms if a == "cgcnn"]
    if not algorithms:
        print("[ERROR] No supported algorithms to run.")
        return 2

    out_root = PROJECT_ROOT / "artifacts" / "phase1_rounds"
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    print("=" * 72)
    print("Phase 1 Round Training")
    print(f"Property   : {args.property}")
    print(f"Rounds     : {args.rounds}")
    print(f"Algorithms : {algorithms}")
    print("Order      : smoke -> lite -> std -> competition -> max")
    print("=" * 72)

    for algorithm in algorithms:
        for r in range(1, args.rounds + 1):
            print(f"\n[ROUND {r}] algorithm={algorithm}")
            for stage in STAGES:
                cmd = _build_cmd(algorithm, args.property, stage, r)
                cmd_str = " ".join(cmd)
                print(f"\n[RUN] round={r} stage={stage.name}")
                print(f"      {cmd_str}")

                stage_out = out_root / algorithm / f"round_{r:02d}" / stage.name
                stage_out.mkdir(parents=True, exist_ok=True)
                (stage_out / "command.txt").write_text(cmd_str + "\n", encoding="utf-8")

                t0 = time.time()
                if args.dry_run:
                    rc = 0
                else:
                    rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode
                elapsed = time.time() - t0

                model_dir = _resolve_model_dir(stage.name, args.property)
                snap = _snapshot_stage(model_dir, stage_out)

                meta = {
                    "algorithm": algorithm,
                    "round": r,
                    "stage": stage.name,
                    "command": cmd,
                    "return_code": rc,
                    "duration_sec": elapsed,
                    **snap,
                }
                with open(stage_out / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                results = snap.get("results") if isinstance(snap, dict) else None
                metrics = (results or {}).get("test_metrics", {}) if isinstance(results, dict) else {}
                mae_key = f"{args.property}_MAE"
                summary_rows.append({
                    "algorithm": algorithm,
                    "round": r,
                    "stage": stage.name,
                    "return_code": rc,
                    "duration_sec": round(elapsed, 2),
                    "model_dir": snap.get("model_dir"),
                    "test_mae": metrics.get(mae_key),
                    "passed": (results or {}).get("passed"),
                })

                if rc != 0 and not args.continue_on_error:
                    print(f"[ERROR] Failed at round={r}, stage={stage.name}, return_code={rc}")
                    _write_summary(out_root, summary_rows)
                    return rc

    _write_summary(out_root, summary_rows)
    print("\n[OK] All requested rounds completed.")
    print(f"[OK] Summary saved under: {out_root}")
    return 0


def _write_summary(out_root: Path, rows: list[dict]) -> None:
    json_path = out_root / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    csv_path = out_root / "summary.csv"
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
