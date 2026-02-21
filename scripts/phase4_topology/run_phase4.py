#!/usr/bin/env python3
"""
Phase 4 launcher: topology algorithm switching + level profiles.

Usage examples:
  python scripts/phase4_topology/run_phase4.py --algorithm topognn --level std
  python scripts/phase4_topology/run_phase4.py --algorithm rf --level pro
  python scripts/phase4_topology/run_phase4.py --algorithm topognn --level max --max-samples 10000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE4_PROFILES = {
    "topognn": {
        "smoke": ["--epochs", "10", "--max", "1000", "--batch-size", "16", "--hidden", "64", "--lr", "0.001"],
        "lite": ["--epochs", "30", "--max", "2500", "--batch-size", "24", "--hidden", "96", "--lr", "0.001"],
        "std": ["--epochs", "100", "--max", "5000", "--batch-size", "32", "--hidden", "128", "--lr", "0.001"],
        "pro": ["--epochs", "180", "--max", "8000", "--batch-size", "48", "--hidden", "160", "--lr", "0.0008"],
        "max": ["--epochs", "260", "--max", "12000", "--batch-size", "64", "--hidden", "192", "--lr", "0.0005"],
    },
    "rf": {
        "smoke": ["--max-samples", "1000", "--n-estimators", "120", "--max-depth", "12", "--min-samples-leaf", "3"],
        "lite": ["--max-samples", "2500", "--n-estimators", "300", "--max-depth", "16", "--min-samples-leaf", "2"],
        "std": ["--max-samples", "5000", "--n-estimators", "600", "--max-depth", "24", "--min-samples-leaf", "2"],
        "pro": ["--max-samples", "8000", "--n-estimators", "900", "--max-depth", "28", "--min-samples-leaf", "1"],
        "max": ["--max-samples", "12000", "--n-estimators", "1200", "--max-depth", "32", "--min-samples-leaf", "1"],
    },
}


def build_command(args: argparse.Namespace) -> list[str]:
    if args.algorithm == "topognn":
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts/phase4_topology/train_topo_classifier.py")]
        cmd.extend(PHASE4_PROFILES["topognn"][args.level])

        overrides = {
            "--epochs": args.epochs,
            "--max": args.max_samples,
            "--batch-size": args.batch_size,
            "--lr": args.lr,
            "--hidden": args.hidden,
        }
        for key, value in overrides.items():
            if value is not None:
                cmd.extend([key, str(value)])

        if args.n_estimators is not None or args.max_depth is not None or args.min_samples_leaf is not None:
            print("[WARN] RF-only args ignored for algorithm='topognn'")
        return cmd

    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/phase4_topology/train_topo_classifier_rf.py")]
    cmd.extend(PHASE4_PROFILES["rf"][args.level])

    overrides = {
        "--max-samples": args.max_samples,
        "--n-estimators": args.n_estimators,
        "--max-depth": args.max_depth,
        "--min-samples-leaf": args.min_samples_leaf,
    }
    for key, value in overrides.items():
        if value is not None:
            cmd.extend([key, str(value)])

    if args.epochs is not None or args.batch_size is not None or args.lr is not None or args.hidden is not None:
        print("[WARN] GNN-only args ignored for algorithm='rf'")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Phase 4 training launcher")
    parser.add_argument("--algorithm", default="topognn", choices=["topognn", "rf"])
    parser.add_argument(
        "--level",
        default="std",
        choices=["smoke", "lite", "std", "pro", "max"],
        help="Hyperparameter level",
    )

    parser.add_argument("--max-samples", type=int, default=None)

    # TopoGNN overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden", type=int, default=None)

    # RF overrides
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=None)

    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd = build_command(args)
    print("[Phase4] Command:")
    print("  " + " ".join(cmd))

    if args.dry_run:
        print("[Phase4] Dry run only, not executing.")
        return 0

    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())

