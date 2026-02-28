#!/usr/bin/env python3
"""
Replay core baseline runs for reproducibility checks.

This script supports Track A / A-09 baseline lock execution.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay baseline runs and export execution report.")
    parser.add_argument("--run-id-prefix", type=str, default="audit3_baseline")
    parser.add_argument("--phase1-level", choices=["smoke", "lite", "std", "pro", "max"], default="smoke")
    parser.add_argument("--phase4-level", choices=["smoke", "lite", "std", "pro", "max"], default="smoke")
    parser.add_argument("--phase4-algorithm", choices=["topognn", "rf"], default="rf")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _run(cmd: list[str], dry_run: bool) -> tuple[int, float]:
    t0 = time.time()
    if dry_run:
        return 0, 0.0
    rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False).returncode
    return rc, time.time() - t0


def main() -> int:
    args = parse_args()

    phase_commands = {
        "phase1": [
            sys.executable,
            str(PROJECT_ROOT / "scripts/phase1_baseline/run_phase1.py"),
            "--level",
            args.phase1_level,
            "--algorithm",
            "cgcnn",
            "--property",
            "formation_energy",
            "--run-id",
            f"{args.run_id_prefix}_phase1",
        ],
        "phase4": [
            sys.executable,
            str(PROJECT_ROOT / "scripts/phase4_topology/run_phase4.py"),
            "--algorithm",
            args.phase4_algorithm,
            "--level",
            args.phase4_level,
            "--run-id",
            f"{args.run_id_prefix}_phase4",
        ],
    }

    rows = []
    has_error = False
    for phase_name, cmd in phase_commands.items():
        print(f"[RUN] {phase_name}")
        print("  " + " ".join(cmd))
        rc, duration = _run(cmd, dry_run=args.dry_run)
        rows.append(
            {
                "phase": phase_name,
                "command": " ".join(cmd),
                "return_code": rc,
                "duration_sec": round(duration, 2),
            }
        )
        if rc != 0:
            has_error = True

    report = {
        "timestamp": time.time(),
        "config": {
            "run_id_prefix": args.run_id_prefix,
            "phase1_level": args.phase1_level,
            "phase4_level": args.phase4_level,
            "phase4_algorithm": args.phase4_algorithm,
            "dry_run": args.dry_run,
        },
        "runs": rows,
        "pass": not has_error,
    }

    if args.output is None:
        out_dir = PROJECT_ROOT / "artifacts" / "program_plan"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        args.output = out_dir / f"baseline_replay_{stamp}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Baseline replay report saved: {args.output}")
    print(f"  pass={report['pass']}")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
