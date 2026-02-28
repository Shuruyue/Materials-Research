#!/usr/bin/env python3
"""
Phase 5 launcher: active-learning discovery profile switching.

Usage examples:
  python scripts/phase5_active_learning/run_phase5.py --level smoke
  python scripts/phase5_active_learning/run_phase5.py --level std --run-id demo01
  python scripts/phase5_active_learning/run_phase5.py --level pro --no-mace
  python scripts/phase5_active_learning/run_phase5.py --competition --resume
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atlas.console_style import install_console_style
from atlas.training.preflight import run_preflight

install_console_style()

PHASE5_PROFILES = {
    "smoke": {
        "args": ["--iterations", "1", "--candidates", "12", "--top", "4", "--seeds", "8"],
        "default_no_mace": True,
    },
    "lite": {
        "args": ["--iterations", "2", "--candidates", "20", "--top", "6", "--seeds", "10"],
        "default_no_mace": True,
    },
    "std": {
        "args": ["--iterations", "5", "--candidates", "50", "--top", "10", "--seeds", "15"],
        "default_no_mace": False,
    },
    "pro": {
        "args": ["--iterations", "10", "--candidates", "80", "--top", "20", "--seeds", "20"],
        "default_no_mace": False,
    },
    "max": {
        "args": ["--iterations", "20", "--candidates", "120", "--top", "30", "--seeds", "30"],
        "default_no_mace": False,
    },
}

PHASE5_COMPETITION = {
    "args": ["--iterations", "8", "--candidates", "100", "--top", "25", "--seeds", "20"],
    "default_no_mace": False,
}


def build_command(args: argparse.Namespace) -> list[str]:
    profile = PHASE5_COMPETITION if args.competition else PHASE5_PROFILES[args.level]
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/phase5_active_learning/run_discovery.py")]
    cmd.extend(profile["args"])

    overrides = {
        "--iterations": args.iterations,
        "--candidates": args.candidates,
        "--top": args.top,
        "--seeds": args.seeds,
    }
    for key, value in overrides.items():
        if value is not None:
            cmd.extend([key, str(value)])

    if args.no_mace or profile["default_no_mace"]:
        cmd.append("--no-mace")
    if args.resume:
        cmd.append("--resume")
    if args.run_id:
        cmd.extend(["--run-id", args.run_id])
    if args.results_dir:
        cmd.extend(["--results-dir", args.results_dir])
    if args.acq_strategy:
        cmd.extend(["--acq-strategy", args.acq_strategy])
    if args.acq_kappa is not None:
        cmd.extend(["--acq-kappa", str(args.acq_kappa)])
    if args.acq_best_f is not None:
        cmd.extend(["--acq-best-f", str(args.acq_best_f)])
    if args.acq_jitter is not None:
        cmd.extend(["--acq-jitter", str(args.acq_jitter)])

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Phase 5 active-learning launcher")
    parser.add_argument(
        "--level",
        default="std",
        choices=["smoke", "lite", "std", "pro", "max"],
        help="Hyperparameter level",
    )
    parser.add_argument(
        "--competition",
        action="store_true",
        help="Use competition profile (independent from --level)",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--no-mace", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument(
        "--acq-strategy",
        type=str,
        choices=["hybrid", "stability", "ei", "pi", "ucb", "lcb", "thompson", "mean", "uncertainty"],
        default=None,
        help="Discovery acquisition strategy",
    )
    parser.add_argument("--acq-kappa", type=float, default=None)
    parser.add_argument("--acq-best-f", type=float, default=None)
    parser.add_argument("--acq-jitter", type=float, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-property-group", type=str, default="priority7")
    parser.add_argument("--preflight-max-samples", type=int, default=0)
    parser.add_argument("--preflight-split-seed", type=int, default=42)
    parser.add_argument("--manifest-visibility", choices=["internal", "public"], default="internal")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.skip_preflight and not args.dry_run:
        print("[ERROR] --skip-preflight is only allowed together with --dry-run.")
        return 2

    if not args.skip_preflight:
        result = run_preflight(
            project_root=PROJECT_ROOT,
            property_group=args.preflight_property_group,
            max_samples=args.preflight_max_samples,
            split_seed=args.preflight_split_seed,
            dry_run=args.dry_run,
        )
        if result.return_code != 0:
            print(f"[ERROR] Preflight failed with return code {result.return_code}")
            return result.return_code

    if args.preflight_only:
        print("[Phase5] Preflight-only mode completed.")
        return 0

    cmd = build_command(args)
    print("[Phase5] Command:")
    if args.competition:
        print("  [Mode] competition profile enabled")
    print("  " + " ".join(cmd))

    if args.dry_run:
        print("[Phase5] Dry run only, not executing.")
        return 0
    env = dict(os.environ)
    split_manifest = PROJECT_ROOT / "artifacts" / "splits" / "split_manifest_iid.json"
    if split_manifest.exists():
        env["ATLAS_SPLIT_MANIFEST"] = str(split_manifest)
    env["ATLAS_MANIFEST_VISIBILITY"] = args.manifest_visibility
    env.setdefault("ATLAS_DATASET_SOURCE_KEY", "jarvis_dft")
    env.setdefault("ATLAS_DATASET_SNAPSHOT_ID", "jarvis_dft_primary")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
