#!/usr/bin/env python3
"""
Phase 1 launcher: unified entry for algorithm + training level selection.

Usage examples:
  python scripts/phase1_baseline/run_phase1.py --level smoke
  python scripts/phase1_baseline/run_phase1.py --level std --property band_gap
  python scripts/phase1_baseline/run_phase1.py --level pro --resume
  python scripts/phase1_baseline/run_phase1.py --level max --property formation_energy
  python scripts/phase1_baseline/run_phase1.py --competition --property formation_energy
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE1_LEVELS = {
    "smoke": {
        "script": "scripts/phase1_baseline/train_cgcnn_lite.py",
        "args": ["--epochs", "2", "--max-samples", "256", "--batch-size", "16"],
        "supports_resume": True,
        "supports_no_filter": False,
    },
    "lite": {
        "script": "scripts/phase1_baseline/train_cgcnn_lite.py",
        "args": [],
        "supports_resume": True,
        "supports_no_filter": False,
    },
    "std": {
        "script": "scripts/phase1_baseline/train_cgcnn_std.py",
        "args": [],
        "supports_resume": True,
        "supports_no_filter": True,
    },
    "pro": {
        "script": "scripts/phase1_baseline/train_cgcnn_pro.py",
        "args": [],
        "supports_resume": True,
        "supports_no_filter": True,
    },
    "max": {
        "script": "scripts/phase1_baseline/train_cgcnn_pro.py",
        "args": [
            "--epochs", "3000",
            "--batch-size", "48",
            "--lr", "0.0007",
            "--patience", "300",
            "--hidden-dim", "768",
            "--n-conv", "6",
        ],
        "supports_resume": True,
        "supports_no_filter": True,
    },
}

PHASE1_COMPETITION = {
    "script": "scripts/phase1_baseline/train_cgcnn_std.py",
    "args": [
        "--epochs", "450",
        "--batch-size", "48",
        "--lr", "0.0008",
        "--patience", "80",
        "--hidden-dim", "192",
        "--n-conv", "4",
        "--max-samples", "40000",
    ],
    "supports_resume": True,
    "supports_no_filter": True,
}


def build_command(args: argparse.Namespace) -> list[str]:
    if args.algorithm != "cgcnn":
        raise ValueError(f"Unsupported Phase 1 algorithm: {args.algorithm}")

    profile = PHASE1_COMPETITION if args.competition else PHASE1_LEVELS[args.level]
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / profile["script"]),
        "--property",
        args.property,
    ]
    cmd.extend(profile["args"])

    if args.resume:
        if profile["supports_resume"]:
            cmd.append("--resume")
        else:
            print(f"[WARN] --resume ignored for level='{args.level}'")

    if args.no_filter:
        if profile["supports_no_filter"]:
            cmd.append("--no-filter")
        else:
            print(f"[WARN] --no-filter ignored for level='{args.level}'")

    overrides = {
        "--epochs": args.epochs,
        "--batch-size": args.batch_size,
        "--lr": args.lr,
        "--max-samples": args.max_samples,
        "--hidden-dim": args.hidden_dim,
        "--n-conv": args.n_conv,
    }
    for key, value in overrides.items():
        if value is not None:
            cmd.extend([key, str(value)])

    if args.run_id:
        cmd.extend(["--run-id", args.run_id])
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.keep_last_k is not None:
        cmd.extend(["--keep-last-k", str(args.keep_last_k)])

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Phase 1 training launcher")
    parser.add_argument("--algorithm", default="cgcnn", choices=["cgcnn"])
    parser.add_argument(
        "--level",
        default="std",
        choices=["smoke", "lite", "std", "pro", "max"],
        help="Hyperparameter level",
    )
    parser.add_argument(
        "--competition",
        action="store_true",
        help="Use competition-optimized profile (independent from --level)",
    )
    parser.add_argument("--property", default="formation_energy")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--n-conv", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--keep-last-k", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd = build_command(args)
    print("[Phase1] Command:")
    if args.competition:
        print("  [Mode] competition profile enabled")
    print("  " + " ".join(cmd))

    if args.dry_run:
        print("[Phase1] Dry run only, not executing.")
        return 0

    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
