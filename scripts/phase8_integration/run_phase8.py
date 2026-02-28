#!/usr/bin/env python3
"""
Phase 8 launcher: integrated discovery pipeline profile switching.

Usage examples:
  python scripts/phase8_integration/run_phase8.py --level smoke
  python scripts/phase8_integration/run_phase8.py --level std --run-id intg_demo_01
  python scripts/phase8_integration/run_phase8.py --competition --seed 123
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atlas.console_style import install_console_style

install_console_style()

PHASE8_PROFILES = {
    "smoke": [
        "--composition-steps", "2",
        "--liflow-steps", "8",
        "--liflow-flow-steps", "2",
        "--skip-mepin",
        "--skip-liflow",
    ],
    "lite": [
        "--composition-steps", "3",
        "--liflow-steps", "12",
        "--liflow-flow-steps", "3",
        "--skip-mepin",
    ],
    "std": [],
    "pro": [
        "--composition-steps", "8",
        "--mepin-images", "7",
        "--liflow-steps", "30",
        "--liflow-flow-steps", "8",
    ],
    "max": [
        "--composition-steps", "12",
        "--mepin-images", "9",
        "--liflow-steps", "50",
        "--liflow-flow-steps", "12",
    ],
}

PHASE8_COMPETITION = [
    "--composition-steps", "6",
    "--mepin-images", "5",
    "--liflow-steps", "24",
    "--liflow-flow-steps", "6",
]


def build_command(args: argparse.Namespace) -> list[str]:
    profile_args = PHASE8_COMPETITION if args.competition else PHASE8_PROFILES[args.level]
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/phase8_integration/run_discovery_pipeline.py")]
    cmd.extend(profile_args)

    overrides = {
        "--composition-steps": args.composition_steps,
        "--mepin-images": args.mepin_images,
        "--liflow-steps": args.liflow_steps,
        "--liflow-flow-steps": args.liflow_flow_steps,
        "--seed": args.seed,
    }
    for key, value in overrides.items():
        if value is not None:
            cmd.extend([key, str(value)])

    if args.run_id:
        cmd.extend(["--run-id", args.run_id])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.skip_alchemy:
        cmd.append("--skip-alchemy")
    if args.skip_mepin:
        cmd.append("--skip-mepin")
    if args.skip_liflow:
        cmd.append("--skip-liflow")

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Phase 8 integration launcher")
    parser.add_argument(
        "--level",
        default="std",
        choices=["smoke", "lite", "std", "pro", "max"],
        help="Pipeline profile level",
    )
    parser.add_argument(
        "--competition",
        action="store_true",
        help="Use competition profile (independent from --level)",
    )
    parser.add_argument("--composition-steps", type=int, default=None)
    parser.add_argument("--mepin-images", type=int, default=None)
    parser.add_argument("--liflow-steps", type=int, default=None)
    parser.add_argument("--liflow-flow-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-alchemy", action="store_true")
    parser.add_argument("--skip-mepin", action="store_true")
    parser.add_argument("--skip-liflow", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd = build_command(args)
    print("[Phase8] Command:")
    if args.competition:
        print("  [Mode] competition profile enabled")
    print("  " + " ".join(cmd))

    if args.dry_run:
        print("[Phase8] Dry run only, not executing.")
        return 0

    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())

