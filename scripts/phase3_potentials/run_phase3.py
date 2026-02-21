#!/usr/bin/env python3
"""
Phase 3 launcher: switch between potential learning (MACE) and specialist model.

Usage examples:
  python scripts/phase3_potentials/run_phase3.py --algorithm mace --level std --prepare-mace-data
  python scripts/phase3_potentials/run_phase3.py --algorithm mace --level max --with-forces
  python scripts/phase3_potentials/run_phase3.py --algorithm equivariant --level pro --property band_gap
  python scripts/phase3_potentials/run_phase3.py --algorithm equivariant --level max --all-properties
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE3_PROFILES = {
    "mace": {
        "smoke": ["--epochs", "20", "--batch-size", "16", "--lr", "0.001", "--r-max", "4.5"],
        "lite": ["--epochs", "100", "--batch-size", "16", "--lr", "0.0005", "--r-max", "5.0"],
        "std": ["--epochs", "300", "--batch-size", "32", "--lr", "0.0003", "--r-max", "5.0"],
        "pro": ["--epochs", "600", "--batch-size", "32", "--lr", "0.0002", "--r-max", "5.5"],
        "max": ["--epochs", "1000", "--batch-size", "48", "--lr", "0.0001", "--r-max", "6.0"],
    },
    "equivariant": {
        "smoke": [
            "--epochs", "50",
            "--batch-size", "8",
            "--acc-steps", "1",
            "--lr", "0.0005",
            "--max-samples", "2000",
            "--ema-decay", "0.0",
            "--outlier-sigma", "6.0",
        ],
        "lite": [
            "--epochs", "300",
            "--batch-size", "12",
            "--acc-steps", "2",
            "--lr", "0.0003",
            "--max-samples", "12000",
            "--outlier-sigma", "7.0",
        ],
        "std": [
            "--epochs", "800",
            "--batch-size", "16",
            "--acc-steps", "4",
            "--lr", "0.0002",
            "--outlier-sigma", "8.0",
        ],
        "pro": [],
        "max": [
            "--epochs", "2200",
            "--batch-size", "16",
            "--acc-steps", "6",
            "--lr", "0.00015",
            "--use-swa",
            "--ema-decay", "0.9995",
        ],
    },
}


def build_prepare_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/phase3_potentials/prepare_mace_data.py"),
        "--elements",
        *args.elements,
        "--max",
        str(args.max_structures),
        "--ehull",
        str(args.ehull),
    ]
    return cmd


def build_train_command(args: argparse.Namespace) -> list[str]:
    if args.algorithm == "mace":
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts/phase3_potentials/train_mace.py")]
        cmd.extend(PHASE3_PROFILES["mace"][args.level])
        if args.with_forces:
            cmd.append("--with-forces")

        overrides = {
            "--epochs": args.epochs,
            "--batch-size": args.batch_size,
            "--lr": args.lr,
            "--r-max": args.r_max,
        }
        for key, value in overrides.items():
            if value is not None:
                cmd.extend([key, str(value)])

        if args.property != "formation_energy":
            print("[WARN] --property is not used by MACE training")

        return cmd

    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/phase3_singletask/train_singletask_pro.py")]
    cmd.extend(PHASE3_PROFILES["equivariant"][args.level])

    if args.all_properties:
        cmd.append("--all-properties")
    else:
        cmd.extend(["--property", args.property])

    if args.finetune_from:
        cmd.extend(["--finetune-from", args.finetune_from])
    if args.freeze_encoder:
        cmd.append("--freeze-encoder")

    overrides = {
        "--epochs": args.epochs,
        "--batch-size": args.batch_size,
        "--lr": args.lr,
        "--max-samples": args.max_samples,
        "--acc-steps": args.acc_steps,
    }
    for key, value in overrides.items():
        if value is not None:
            cmd.extend([key, str(value)])

    if args.r_max is not None:
        print("[WARN] --r-max is only supported by algorithm='mace'")
    if args.with_forces:
        print("[WARN] --with-forces is only supported by algorithm='mace'")

    return cmd


def run_command(cmd: list[str], dry_run: bool) -> int:
    print("  " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Phase 3 training launcher")
    parser.add_argument("--algorithm", default="equivariant", choices=["mace", "equivariant"])
    parser.add_argument(
        "--level",
        default="std",
        choices=["smoke", "lite", "std", "pro", "max"],
        help="Hyperparameter level",
    )

    # MACE options
    parser.add_argument("--prepare-mace-data", action="store_true")
    parser.add_argument("--elements", nargs="+", default=["Si", "Ge", "Sn"])
    parser.add_argument("--max-structures", type=int, default=2000)
    parser.add_argument("--ehull", type=float, default=0.3)
    parser.add_argument("--with-forces", action="store_true")
    parser.add_argument("--r-max", type=float, default=None)

    # Specialist options
    parser.add_argument("--property", default="formation_energy")
    parser.add_argument("--all-properties", action="store_true")
    parser.add_argument("--finetune-from", default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--acc-steps", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    # Common overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.algorithm == "mace" and args.prepare_mace_data:
        print("[Phase3] Prepare data command:")
        prepare_cmd = build_prepare_command(args)
        status = run_command(prepare_cmd, args.dry_run)
        if status != 0:
            return status

    train_cmd = build_train_command(args)
    print("[Phase3] Training command:")
    return run_command(train_cmd, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

