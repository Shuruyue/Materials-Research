#!/usr/bin/env python3
"""
Check phase-runner CLI interface contracts.

This script supports Track B / B-05 phase interface governance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


CONTRACTS = {
    "phase1": {
        "script": "scripts/phase1_baseline/run_phase1.py",
        "required_flags": ["--algorithm", "--level", "--property", "--run-id", "--resume", "--dry-run"],
    },
    "phase2": {
        "script": "scripts/phase2_multitask/run_phase2.py",
        "required_flags": [
            "--algorithm",
            "--level",
            "--property-group",
            "--run-id",
            "--resume",
            "--dry-run",
        ],
    },
    "phase3": {
        "script": "scripts/phase3_potentials/run_phase3.py",
        "required_flags": ["--algorithm", "--level", "--run-id", "--resume", "--dry-run"],
    },
    "phase4": {
        "script": "scripts/phase4_topology/run_phase4.py",
        "required_flags": ["--algorithm", "--level", "--run-id", "--resume", "--dry-run"],
    },
    "phase5": {
        "script": "scripts/phase5_active_learning/run_phase5.py",
        "required_flags": ["--level", "--run-id", "--resume", "--dry-run"],
    },
    "phase6": {
        "script": "scripts/phase6_analysis/run_phase6.py",
        "required_flags": ["--level", "--alloy", "--compare", "--dry-run"],
    },
    "phase8": {
        "script": "scripts/phase8_integration/run_phase8.py",
        "required_flags": ["--level", "--run-id", "--output-dir", "--dry-run"],
    },
    "full_project": {
        "script": "scripts/training/run_full_project.py",
        "required_flags": ["--phase", "--level", "--dry-run", "--session-id", "--continue-on-error"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate phase CLI interface contracts.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _run_help(script_path: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, text


def main() -> int:
    args = parse_args()
    t0 = time.time()
    checks = []
    has_error = False

    for name, spec in CONTRACTS.items():
        script = PROJECT_ROOT / spec["script"]
        row = {
            "name": name,
            "script": str(script),
            "exists": script.exists(),
            "help_return_code": None,
            "missing_flags": [],
            "pass": False,
        }
        if not script.exists():
            has_error = True
            row["error"] = "script_not_found"
            checks.append(row)
            continue

        rc, help_text = _run_help(script)
        row["help_return_code"] = rc
        if rc != 0:
            has_error = True
            row["error"] = "help_command_failed"
            checks.append(row)
            continue

        missing = [flag for flag in spec["required_flags"] if flag not in help_text]
        row["missing_flags"] = missing
        row["pass"] = len(missing) == 0
        if missing:
            has_error = True
        checks.append(row)

    report = {
        "timestamp": time.time(),
        "checks": checks,
        "summary": {
            "total": len(checks),
            "failed": sum(1 for c in checks if not c["pass"]),
            "pass": not has_error,
        },
        "duration_sec": round(time.time() - t0, 2),
    }

    if args.output is None:
        out_dir = PROJECT_ROOT / "artifacts" / "program_plan"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        args.output = out_dir / f"phase_interface_contract_{stamp}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Phase interface contract report saved: {args.output}")
    print(
        f"  total={report['summary']['total']}, "
        f"failed={report['summary']['failed']}, "
        f"pass={report['summary']['pass']}"
    )

    if args.strict and has_error:
        print("[ERROR] Strict mode failed due to interface contract violations.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
