#!/usr/bin/env python3
"""
Phase 6 launcher: analysis profile switching for alloy-property workflows.

Usage examples:
  python scripts/phase6_analysis/run_phase6.py --level smoke
  python scripts/phase6_analysis/run_phase6.py --level std --alloy SnPb63
  python scripts/phase6_analysis/run_phase6.py --level pro
  python scripts/phase6_analysis/run_phase6.py --competition
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

PHASE6_PROFILES = {
    "smoke": ["--alloy", "pure_Sn"],
    "lite": ["--alloy", "SAC305"],
    "std": ["--alloy", "SAC305"],
    "pro": ["--compare"],
    "max": ["--compare"],
}

PHASE6_COMPETITION = ["--compare"]


def build_command(args: argparse.Namespace) -> list[str]:
    profile_args = PHASE6_COMPETITION if args.competition else PHASE6_PROFILES[args.level]
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/phase6_analysis/alloy_properties.py")]
    cmd.extend(profile_args)

    if args.list:
        cmd.append("--list")
    elif args.compare:
        if "--compare" not in cmd:
            cmd.append("--compare")
    elif args.alloy is not None:
        cmd.extend(["--alloy", args.alloy])

    if args.save_json:
        cmd.extend(["--save-json", args.save_json])

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Phase 6 analysis launcher")
    parser.add_argument(
        "--level",
        default="std",
        choices=["smoke", "lite", "std", "pro", "max"],
        help="Preset analysis level",
    )
    parser.add_argument(
        "--competition",
        action="store_true",
        help="Use competition profile (compare mode)",
    )
    parser.add_argument("--alloy", type=str, default=None, help="Preset alloy name for single-alloy analysis")
    parser.add_argument("--compare", action="store_true", help="Force compare mode")
    parser.add_argument("--list", action="store_true", help="List available presets")
    parser.add_argument("--save-json", type=str, default=None, help="Optional output JSON path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd = build_command(args)
    print("[Phase6] Command:")
    if args.competition:
        print("  [Mode] competition profile enabled")
    print("  " + " ".join(cmd))

    if args.dry_run:
        print("[Phase6] Dry run only, not executing.")
        return 0

    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())

