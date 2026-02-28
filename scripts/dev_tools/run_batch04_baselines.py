#!/usr/bin/env python3
"""
Run Batch-04 baseline commands (A-11/A-12/A-13) and collect evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _run_with_timeout(cmd: list[str], timeout_sec: int) -> tuple[int, float, str | None]:
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False, timeout=timeout_sec).returncode
        return rc, time.time() - t0, None
    except subprocess.TimeoutExpired:
        return 124, time.time() - t0, "timeout"
    except Exception as exc:  # pragma: no cover - defensive
        return 1, time.time() - t0, str(exc)


def _build_run_name(run_id: str) -> str:
    return run_id if run_id.startswith("run_") else f"run_{run_id}"


def _collect_artifacts(row: dict[str, Any]) -> dict[str, Any]:
    run_name = _build_run_name(row["run_id"])
    model_family = row["model_family"]
    run_dir = PROJECT_ROOT / "models" / model_family / run_name

    manifest = _load_json(run_dir / "run_manifest.json") or {}
    results = _load_json(run_dir / "results.json") or {}
    training_info = _load_json(run_dir / "training_info.json") or {}

    payload: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_dir_exists": run_dir.exists(),
    }
    if manifest:
        result = manifest.get("result")
        payload["manifest_status"] = manifest.get("status")
        if isinstance(result, dict):
            payload["manifest_result"] = result
    if results:
        payload["results_keys"] = sorted(results.keys())
        if "best_val_mae" in results:
            payload["best_val_mae"] = results.get("best_val_mae")
        if "avg_test_mae" in results:
            payload["avg_test_mae"] = results.get("avg_test_mae")
        if "test_metrics" in results:
            payload["test_metrics"] = results.get("test_metrics")
    if training_info:
        payload["training_info"] = training_info
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Batch-04 baselines and export a report.")
    parser.add_argument("--run-prefix", type=str, default="audit4_batch04")
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    jobs = [
        {
            "stage": "A-11",
            "name": "phase1_cgcnn_std",
            "model_family": "cgcnn_std_formation_energy",
            "run_id": f"{args.run_prefix}_a11_phase1_std",
            "cmd": [
                sys.executable,
                "scripts/phase1_baseline/run_phase1.py",
                "--algorithm",
                "cgcnn",
                "--level",
                "std",
                "--property",
                "formation_energy",
                "--epochs",
                "4",
                "--max-samples",
                "800",
                "--batch-size",
                "32",
                "--run-id",
                f"{args.run_prefix}_a11_phase1_std",
            ],
        },
        {
            "stage": "A-12",
            "name": "phase2_e3nn_smoke",
            "model_family": "multitask_lite_e3nn",
            "run_id": f"{args.run_prefix}_a12_phase2_e3nn",
            "cmd": [
                sys.executable,
                "scripts/phase2_multitask/train_multitask_lite.py",
                "--epochs",
                "5",
                "--batch-size",
                "8",
                "--lr",
                "0.002",
                "--property-group",
                "core4",
                "--train-samples",
                "800",
                "--eval-samples",
                "100",
                "--run-id",
                f"{args.run_prefix}_a12_phase2_e3nn",
            ],
        },
        {
            "stage": "A-12",
            "name": "phase2_cgcnn_smoke",
            "model_family": "multitask_cgcnn",
            "run_id": f"{args.run_prefix}_a12_phase2_cgcnn",
            "cmd": [
                sys.executable,
                "scripts/phase2_multitask/run_phase2.py",
                "--algorithm",
                "cgcnn",
                "--level",
                "smoke",
                "--property-group",
                "core4",
                "--max-samples",
                "800",
                "--run-id",
                f"{args.run_prefix}_a12_phase2_cgcnn",
            ],
        },
        {
            "stage": "A-12",
            "name": "phase2_m3gnet_smoke",
            "model_family": "multitask_m3gnet",
            "run_id": f"{args.run_prefix}_a12_phase2_m3gnet",
            "cmd": [
                sys.executable,
                "scripts/phase2_multitask/run_phase2.py",
                "--algorithm",
                "m3gnet",
                "--level",
                "smoke",
                "--property-group",
                "core4",
                "--max-samples",
                "800",
                "--run-id",
                f"{args.run_prefix}_a12_phase2_m3gnet",
            ],
        },
        {
            "stage": "A-13",
            "name": "phase3_equivariant_smoke",
            "model_family": "specialist_formation_energy",
            "run_id": f"{args.run_prefix}_a13_phase3_equiv",
            "cmd": [
                sys.executable,
                "scripts/phase3_potentials/run_phase3.py",
                "--algorithm",
                "equivariant",
                "--level",
                "smoke",
                "--property",
                "formation_energy",
                "--epochs",
                "3",
                "--max-samples",
                "400",
                "--acc-steps",
                "1",
                "--run-id",
                f"{args.run_prefix}_a13_phase3_equiv",
            ],
        },
        {
            "stage": "A-13",
            "name": "phase3_mace_smoke",
            "model_family": "mace",
            "run_id": f"{args.run_prefix}_a13_phase3_mace",
            "cmd": [
                sys.executable,
                "scripts/phase3_potentials/run_phase3.py",
                "--algorithm",
                "mace",
                "--level",
                "smoke",
                "--epochs",
                "1",
                "--run-id",
                f"{args.run_prefix}_a13_phase3_mace",
            ],
        },
    ]

    rows: list[dict[str, Any]] = []
    has_error = False

    for job in jobs:
        cmd = job["cmd"]
        cmd_str = " ".join(cmd)
        print(f"[RUN] {job['stage']} {job['name']}")
        print(f"  {cmd_str}")

        if args.dry_run:
            rc, duration, error = 0, 0.0, None
        else:
            rc, duration, error = _run_with_timeout(cmd, timeout_sec=args.timeout_sec)

        row = {
            "stage": job["stage"],
            "name": job["name"],
            "command": cmd_str,
            "return_code": rc,
            "duration_sec": round(duration, 2),
        }
        if error:
            row["error"] = error
        row["artifacts"] = _collect_artifacts(job)
        rows.append(row)

        if rc != 0:
            has_error = True
            if not args.continue_on_error:
                break

    report = {
        "timestamp": time.time(),
        "config": {
            "run_prefix": args.run_prefix,
            "timeout_sec": args.timeout_sec,
            "dry_run": args.dry_run,
        },
        "runs": rows,
        "pass": not has_error,
    }

    if args.output is None:
        out_dir = PROJECT_ROOT / "artifacts" / "program_plan"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        args.output = out_dir / f"batch04_baseline_runs_{stamp}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Batch-04 baseline report saved: {args.output}")
    print(f"  pass={report['pass']}")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
