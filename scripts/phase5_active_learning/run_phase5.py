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
import math
import os
import re
import subprocess
import sys
from numbers import Integral, Real
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atlas.active_learning.acquisition import DISCOVERY_ACQUISITION_STRATEGIES
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

_SAFE_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_PROPERTY_GROUP = re.compile(r"^[A-Za-z0-9._-]+$")


def _is_safe_run_id(run_id: str | None) -> bool:
    if not run_id:
        return True
    if run_id in {".", ".."}:
        return False
    if "/" in run_id or "\\" in run_id:
        return False
    return bool(_SAFE_RUN_ID_PATTERN.fullmatch(run_id))


def _is_finite_non_negative(value: float | None) -> bool:
    if value is None:
        return True
    numeric = float(value)
    return math.isfinite(numeric) and numeric >= 0.0


def _coerce_int_with_bounds(
    value: object,
    *,
    arg_name: str,
    min_value: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{arg_name} must be an integer")
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        number_f = float(value)
        if not math.isfinite(number_f) or not number_f.is_integer():
            raise ValueError(f"{arg_name} must be an integer")
        number = int(number_f)
    else:
        try:
            number = int(value)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(f"{arg_name} must be an integer") from exc

    if min_value is not None and number < min_value:
        comparator = "> 0" if min_value == 1 else f">= {min_value}"
        raise ValueError(f"{arg_name} must be {comparator}")
    return number


def _set_or_replace_flag(cmd: list[str], flag: str, value: int | float | str) -> None:
    """Replace an existing CLI flag value or append a new flag/value pair."""
    sval = str(value)
    for idx in range(len(cmd) - 1):
        if cmd[idx] == flag:
            cmd[idx + 1] = sval
            return
    cmd.extend([flag, sval])


def _validate_args(args: argparse.Namespace) -> tuple[bool, str]:
    positive_int_fields = ("iterations", "candidates", "top", "seeds", "calibration_window")
    for field in positive_int_fields:
        value = getattr(args, field, None)
        if value is None:
            continue
        try:
            normalized = _coerce_int_with_bounds(value, arg_name=f"--{field.replace('_', '-')}", min_value=1)
        except ValueError as exc:
            return False, str(exc)
        setattr(args, field, normalized)

    if args.preflight_max_samples is not None:
        try:
            args.preflight_max_samples = _coerce_int_with_bounds(
                args.preflight_max_samples,
                arg_name="--preflight-max-samples",
                min_value=0,
            )
        except ValueError as exc:
            return False, str(exc)
    property_group = str(args.preflight_property_group or "").strip()
    if not property_group:
        return False, "--preflight-property-group must not be empty"
    if not _SAFE_PROPERTY_GROUP.fullmatch(property_group):
        return False, "--preflight-property-group contains unsupported characters"
    if not _is_finite_non_negative(args.acq_kappa):
        return False, "--acq-kappa must be finite and >= 0"
    if not _is_finite_non_negative(args.acq_jitter):
        return False, "--acq-jitter must be finite and >= 0"
    if args.acq_best_f is not None and not math.isfinite(float(args.acq_best_f)):
        return False, "--acq-best-f must be finite"
    if args.preflight_split_seed is not None:
        try:
            args.preflight_split_seed = _coerce_int_with_bounds(
                args.preflight_split_seed,
                arg_name="--preflight-split-seed",
                min_value=0,
            )
        except ValueError as exc:
            return False, str(exc)
    if args.preflight_timeout_sec is not None:
        try:
            args.preflight_timeout_sec = _coerce_int_with_bounds(
                args.preflight_timeout_sec,
                arg_name="--preflight-timeout-sec",
                min_value=1,
            )
        except ValueError as exc:
            return False, str(exc)
    if bool(args.run_id) and bool(args.results_dir):
        return False, "--run-id and --results-dir cannot be used together"
    if bool(args.resume) and bool(args.results_dir):
        return False, "--resume and --results-dir cannot be used together"
    if bool(args.preflight_only) and bool(args.skip_preflight):
        return False, "--preflight-only cannot be used with --skip-preflight"
    if not _is_safe_run_id(args.run_id):
        return False, "--run-id contains unsafe characters"
    if args.results_dir is not None and not str(args.results_dir).strip():
        return False, "--results-dir must not be empty"
    if args.candidates is not None and args.top is not None and int(args.top) > int(args.candidates):
        return False, "--top cannot be greater than --candidates"
    return True, ""


def build_command(args: argparse.Namespace) -> list[str]:
    profile = PHASE5_COMPETITION if args.competition else PHASE5_PROFILES[args.level]
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/phase5_active_learning/run_discovery.py")]
    cmd.extend(profile["args"])

    overrides = (
        ("--iterations", args.iterations),
        ("--candidates", args.candidates),
        ("--top", args.top),
        ("--seeds", args.seeds),
    )
    for key, value in overrides:
        if value is not None:
            _set_or_replace_flag(cmd, key, value)

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
    if args.policy:
        cmd.extend(["--policy", args.policy])
    if args.risk_mode:
        cmd.extend(["--risk-mode", args.risk_mode])
    if args.cost_aware:
        cmd.append("--cost-aware")
    if args.calibration_window is not None:
        cmd.extend(["--calibration-window", str(args.calibration_window)])

    return cmd


def _build_parser() -> argparse.ArgumentParser:
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
        choices=sorted(DISCOVERY_ACQUISITION_STRATEGIES),
        default=None,
        help="Discovery acquisition strategy",
    )
    parser.add_argument("--acq-kappa", type=float, default=None)
    parser.add_argument("--acq-best-f", type=float, default=None)
    parser.add_argument("--acq-jitter", type=float, default=None)
    parser.add_argument("--policy", choices=["legacy", "cmoeic"], default="legacy")
    parser.add_argument("--risk-mode", choices=["soft", "hard", "hybrid"], default="soft")
    parser.add_argument("--cost-aware", action="store_true")
    parser.add_argument("--calibration-window", type=int, default=128)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-property-group", type=str, default="priority7")
    parser.add_argument("--preflight-max-samples", type=int, default=0)
    parser.add_argument("--preflight-split-seed", type=int, default=42)
    parser.add_argument("--preflight-timeout-sec", type=int, default=1800)
    parser.add_argument("--manifest-visibility", choices=["internal", "public"], default="internal")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    ok, message = _validate_args(args)
    if not ok:
        print(f"[ERROR] {message}", file=sys.stderr)
        return 2

    if args.skip_preflight and not args.dry_run:
        print("[ERROR] --skip-preflight is only allowed together with --dry-run.", file=sys.stderr)
        return 2

    if not args.skip_preflight:
        result = run_preflight(
            project_root=PROJECT_ROOT,
            property_group=args.preflight_property_group,
            max_samples=args.preflight_max_samples,
            split_seed=args.preflight_split_seed,
            dry_run=args.dry_run,
            timeout_sec=args.preflight_timeout_sec,
        )
        if result.return_code != 0:
            detail = f" ({result.error_message})" if result.error_message else ""
            print(f"[ERROR] Preflight failed with return code {result.return_code}{detail}", file=sys.stderr)
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
