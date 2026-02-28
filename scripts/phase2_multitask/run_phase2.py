#!/usr/bin/env python3
"""
Phase 2 launcher: unified entry for algorithm switching + level profiles.

Usage examples:
  python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level std
  python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --level pro --all-properties
  python scripts/phase2_multitask/run_phase2.py --algorithm cgcnn --level lite
  python scripts/phase2_multitask/run_phase2.py --algorithm m3gnet --level std
  python scripts/phase2_multitask/run_phase2.py --algorithm e3nn --competition
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
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

PROPERTY_GROUP_CHOICES = ("core4", "priority7", "secondary2", "all9")

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
            "supports_resume": True,
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
            "args": ["--encoder", "cgcnn", "--preset", "small", "--epochs", "5", "--batch-size", "64", "--max-samples", "800"],
            "supports_resume": True,
            "supports_all_properties": False,
            "supports_max_samples": True,
        },
        "lite": {
            "script": "scripts/phase2_multitask/train_multitask_cgcnn.py",
            "args": ["--encoder", "cgcnn", "--preset", "small", "--epochs", "40", "--batch-size", "96", "--max-samples", "3000"],
            "supports_resume": True,
            "supports_all_properties": False,
            "supports_max_samples": True,
        },
        "std": {
            "script": "scripts/phase2_multitask/train_multitask_cgcnn.py",
            "args": ["--encoder", "cgcnn", "--preset", "medium", "--epochs", "200", "--batch-size", "128"],
            "supports_resume": True,
            "supports_all_properties": False,
            "supports_max_samples": True,
        },
        "pro": {
            "script": "scripts/phase2_multitask/train_multitask_cgcnn.py",
            "args": ["--encoder", "cgcnn", "--preset", "large", "--epochs", "300", "--batch-size", "128"],
            "supports_resume": True,
            "supports_all_properties": False,
            "supports_max_samples": True,
        },
        "max": {
            "script": "scripts/phase2_multitask/train_multitask_cgcnn.py",
            "args": ["--encoder", "cgcnn", "--preset", "large", "--epochs", "500", "--batch-size", "160", "--lr", "0.0007"],
            "supports_resume": True,
            "supports_all_properties": False,
            "supports_max_samples": True,
        },
    },
    "m3gnet": {
        "smoke": {
            "script": "scripts/phase2_multitask/train_multitask_cgcnn.py",
            "args": ["--encoder", "m3gnet", "--preset", "small", "--epochs", "5", "--batch-size", "48", "--max-samples", "800"],
            "supports_resume": True,
            "supports_all_properties": False,
            "supports_max_samples": True,
        },
        "lite": {
            "script": "scripts/phase2_multitask/train_multitask_cgcnn.py",
            "args": ["--encoder", "m3gnet", "--preset", "small", "--epochs", "40", "--batch-size", "64", "--max-samples", "3000"],
            "supports_resume": True,
            "supports_all_properties": False,
            "supports_max_samples": True,
        },
        "std": {
            "script": "scripts/phase2_multitask/train_multitask_cgcnn.py",
            "args": ["--encoder", "m3gnet", "--preset", "medium", "--epochs", "200", "--batch-size", "80"],
            "supports_resume": True,
            "supports_all_properties": False,
            "supports_max_samples": True,
        },
        "pro": {
            "script": "scripts/phase2_multitask/train_multitask_cgcnn.py",
            "args": ["--encoder", "m3gnet", "--preset", "large", "--epochs", "300", "--batch-size", "96"],
            "supports_resume": True,
            "supports_all_properties": False,
            "supports_max_samples": True,
        },
        "max": {
            "script": "scripts/phase2_multitask/train_multitask_cgcnn.py",
            "args": ["--encoder", "m3gnet", "--preset", "large", "--epochs", "500", "--batch-size", "112", "--lr", "0.0007"],
            "supports_resume": True,
            "supports_all_properties": False,
            "supports_max_samples": True,
        },
    },
}

PHASE2_COMPETITION = {
    "e3nn": {
        "script": "scripts/phase2_multitask/train_multitask_pro.py",
        "args": ["--epochs", "260", "--batch-size", "6", "--lr", "0.00045"],
        "supports_resume": True,
        "supports_all_properties": True,
        "supports_max_samples": False,
    },
    "cgcnn": {
        "script": "scripts/phase2_multitask/train_multitask_cgcnn.py",
        "args": ["--encoder", "cgcnn", "--preset", "medium", "--epochs", "280", "--batch-size", "128", "--lr", "0.0009"],
        "supports_resume": True,
        "supports_all_properties": False,
        "supports_max_samples": True,
    },
    "m3gnet": {
        "script": "scripts/phase2_multitask/train_multitask_cgcnn.py",
        "args": ["--encoder", "m3gnet", "--preset", "medium", "--epochs", "280", "--batch-size", "96", "--lr", "0.0009"],
        "supports_resume": True,
        "supports_all_properties": False,
        "supports_max_samples": True,
    },
}


def _default_property_group(algorithm: str, level: str, competition: bool) -> str:
    if competition:
        return "priority7"
    if level in {"smoke", "lite"}:
        return "core4"
    return "priority7"


def build_command(args: argparse.Namespace) -> list[str]:
    profile = PHASE2_COMPETITION[args.algorithm] if args.competition else PHASE2_PROFILES[args.algorithm][args.level]
    cmd = [sys.executable, "-u", str(PROJECT_ROOT / profile["script"])]
    cmd.extend(profile["args"])

    if args.resume:
        if profile["supports_resume"]:
            cmd.append("--resume")
        else:
            print(f"[WARN] --resume ignored for algorithm={args.algorithm}, level={args.level}")

    selected_group = args.property_group
    if args.all_properties:
        selected_group = "all9"
    if selected_group is None:
        selected_group = _default_property_group(args.algorithm, args.level, args.competition)
    cmd.extend(["--property-group", selected_group])

    overrides = {
        "--epochs": args.epochs,
        "--batch-size": args.batch_size,
        "--lr": args.lr,
    }
    for key, value in overrides.items():
        if value is not None:
            cmd.extend([key, str(value)])

    if args.algorithm in {"cgcnn", "m3gnet"}:
        if args.preset is not None:
            cmd.extend(["--preset", args.preset])
        if args.max_samples is not None:
            cmd.extend(["--max-samples", str(args.max_samples)])

        if args.algorithm == "cgcnn":
            if args.pooling is not None:
                cmd.extend(["--pooling", args.pooling])
            if args.jk is not None:
                cmd.extend(["--jk", args.jk])
            if args.message_aggr is not None:
                cmd.extend(["--message-aggr", args.message_aggr])
            if args.no_edge_gates:
                cmd.append("--no-edge-gates")
        elif args.pooling is not None or args.jk is not None or args.message_aggr is not None or args.no_edge_gates:
            print("[WARN] CGCNN-only graph-encoder args ignored for algorithm=m3gnet")
    elif args.max_samples is not None:
        print("[WARN] --max-samples is only supported by Phase 2 CGCNN/M3GNet baselines")
    elif args.pooling is not None or args.jk is not None or args.message_aggr is not None or args.no_edge_gates:
        print("[WARN] CGCNN-only graph-encoder args ignored (pooling/jk/message-aggr/no-edge-gates)")

    if args.run_id:
        cmd.extend(["--run-id", args.run_id])
    if args.init_from:
        if args.algorithm == "e3nn" and str(profile["script"]).endswith(
            ("train_multitask_std.py", "train_multitask_pro.py")
        ):
            cmd.extend(["--init-from", args.init_from])
        else:
            print("[WARN] --init-from is only supported by e3nn std/pro/competition runs")
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.keep_last_k is not None:
        cmd.extend(["--keep-last-k", str(args.keep_last_k)])

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Phase 2 training launcher")
    parser.add_argument("--algorithm", default="e3nn", choices=["e3nn", "cgcnn", "m3gnet"])
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--property-group",
        choices=PROPERTY_GROUP_CHOICES,
        default=None,
        help="Override property group (defaults: lite/smoke=core4, std+=priority7)",
    )
    parser.add_argument("--all-properties", action="store_true", help="Alias for --property-group all9")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--preset", choices=["small", "medium", "large"], default=None)
    parser.add_argument("--pooling", choices=["mean", "sum", "max", "mean_max", "attn"], default=None)
    parser.add_argument("--jk", choices=["last", "mean", "concat"], default=None)
    parser.add_argument("--message-aggr", choices=["sum", "mean"], default=None)
    parser.add_argument("--no-edge-gates", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--keep-last-k", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-property-group", type=str, default="priority7")
    parser.add_argument("--preflight-max-samples", type=int, default=0)
    parser.add_argument("--preflight-split-seed", type=int, default=42)
    parser.add_argument("--split-manifest", type=str, default=None)
    parser.add_argument("--manifest-visibility", choices=["internal", "public"], default="internal")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.skip_preflight and not args.dry_run:
        print("[ERROR] --skip-preflight is only allowed together with --dry-run.")
        return 2

    if not args.skip_preflight:
        preflight_group = args.preflight_property_group
        if preflight_group is None:
            preflight_group = _default_property_group(
                args.algorithm, args.level, args.competition
            )
        result = run_preflight(
            project_root=PROJECT_ROOT,
            property_group=preflight_group,
            max_samples=args.preflight_max_samples,
            split_seed=args.preflight_split_seed,
            dry_run=args.dry_run,
        )
        if result.return_code != 0:
            print(f"[ERROR] Preflight failed with return code {result.return_code}")
            return result.return_code

    if args.preflight_only:
        print("[Phase2] Preflight-only mode completed.")
        return 0

    cmd = build_command(args)
    print("[Phase2] Command:")
    if args.competition:
        print(f"  [Mode] competition profile enabled ({args.algorithm})")
    print("  " + " ".join(cmd))

    if args.dry_run:
        print("[Phase2] Dry run only, not executing.")
        return 0

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    split_manifest = Path(args.split_manifest) if args.split_manifest else (
        PROJECT_ROOT / "artifacts" / "splits" / "split_manifest_iid.json"
    )
    if split_manifest.exists():
        env["ATLAS_SPLIT_MANIFEST"] = str(split_manifest)
    else:
        print(f"[WARN] Split manifest not found, fallback to random split: {split_manifest}")
        env.pop("ATLAS_SPLIT_MANIFEST", None)
    env["ATLAS_MANIFEST_VISIBILITY"] = args.manifest_visibility
    env.setdefault("ATLAS_DATASET_SOURCE_KEY", "jarvis_dft")
    env.setdefault("ATLAS_DATASET_SNAPSHOT_ID", "jarvis_dft_primary")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
