#!/usr/bin/env python3
"""
Phase 2 launcher: unified entry for algorithm switching + level profiles.

Usage examples:
  python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level std
  python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level pro --all-properties
  python scripts/phase2_multitask/run_phase2.py --algorithm cgcnn --level lite
  python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level max --resume
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE2_PROFILES = {
    "e3nn": {
        "smoke": {
            "script": "scripts/phase2_multitask/train_multitask_std.py",
            "args": ["--epochs", "5", "--batch-size", "8", "--lr", "0.002"],
            "supports_resume": True,
            "supports_all_properties": False,
            "supports_max_samples": False,
        },
        "lite": {
            "script": "scripts/phase2_multitask/train_multitask_lite.py",
            "args": [],
            "supports_resume": False,
            "supports_all_properties": False,
            "supports_max_samples": False,
        },
        "std": {
            "script": "scripts/phase2_multitask/train_multitask_std.py",
            "args": [],
            "supports_resume": True,
            "supports_all_properties": False,
            "supports_max_samples": False,
        },
        "pro": {
            "script": "scripts/phase2_multitask/train_multitask_pro.py",
            "args": [],
            "supports_resume": True,
            "supports_all_properties": True,
            "supports_max_samples": False,
        },
        "max": {
            "script": "scripts/phase2_multitask/train_multitask_pro.py",
            "args": ["--epochs", "800", "--batch-size", "4", "--lr", "0.0003"],
            "supports_resume": True,
            "supports_all_properties": True,
            "supports_max_samples": False,
        },
    },
    "cgcnn": {
        "smoke": {
            "script": "scripts/phase2_multitask/train_multitask_cgcnn.py",
            "args": ["--preset", "small", "--epochs", "5", "--batch-size", "64", "--max-samples", "800"],
            "supports_resume": False,
            "supports_all_properties": False,
            "supports_max_samples": True,
        },
        "lite": {
            "script": "scripts/phase2_multitask/train_multitask_cgcnn.py",
            "args": ["--preset", "small", "--epochs", "40", "--batch-size", "96", "--max-samples", "3000"],
            "supports_resume": False,
            "supports_all_properties": False,
            "supports_max_samples": True,
        },
        "std": {
            "script": "scripts/phase2_multitask/train_multitask_cgcnn.py",
            "args": ["--preset", "medium", "--epochs", "200", "--batch-size", "128"],
            "supports_resume": False,
            "supports_all_properties": False,
            "supports_max_samples": True,
        },
        "pro": {
            "script": "scripts/phase2_multitask/train_multitask_cgcnn.py",
            "args": ["--preset", "large", "--epochs", "300", "--batch-size", "128"],
            "supports_resume": False,
            "supports_all_properties": False,
            "supports_max_samples": True,
        },
        "max": {
            "script": "scripts/phase2_multitask/train_multitask_cgcnn.py",
            "args": ["--preset", "large", "--epochs", "500", "--batch-size", "160", "--lr", "0.0007"],
            "supports_resume": False,
            "supports_all_properties": False,
            "supports_max_samples": True,
        },
    },
}


def build_command(args: argparse.Namespace) -> list[str]:
    profile = PHASE2_PROFILES[args.algorithm][args.level]
    cmd = [sys.executable, str(PROJECT_ROOT / profile["script"])]
    cmd.extend(profile["args"])

    if args.resume:
        if profile["supports_resume"]:
            cmd.append("--resume")
        else:
            print(f"[WARN] --resume ignored for algorithm={args.algorithm}, level={args.level}")

    if args.all_properties:
        if profile["supports_all_properties"]:
            cmd.append("--all-properties")
        else:
            print(f"[WARN] --all-properties ignored for algorithm={args.algorithm}, level={args.level}")

    overrides = {
        "--epochs": args.epochs,
        "--batch-size": args.batch_size,
        "--lr": args.lr,
    }
    for key, value in overrides.items():
        if value is not None:
            cmd.extend([key, str(value)])

    if args.algorithm == "cgcnn":
        if args.preset is not None:
            cmd.extend(["--preset", args.preset])
        if args.max_samples is not None:
            cmd.extend(["--max-samples", str(args.max_samples)])
    elif args.max_samples is not None:
        print("[WARN] --max-samples is only supported by Phase 2 CGCNN baseline")

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Phase 2 training launcher")
    parser.add_argument("--algorithm", default="e3nn", choices=["e3nn", "cgcnn"])
    parser.add_argument(
        "--level",
        default="std",
        choices=["smoke", "lite", "std", "pro", "max"],
        help="Hyperparameter level",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--all-properties", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--preset", choices=["small", "medium", "large"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd = build_command(args)
    print("[Phase2] Command:")
    print("  " + " ".join(cmd))

    if args.dry_run:
        print("[Phase2] Dry run only, not executing.")
        return 0

    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())

