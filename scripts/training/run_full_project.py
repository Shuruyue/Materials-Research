#!/usr/bin/env python3
"""
Unified full-project orchestrator across Phase 1/2/3/4/5/6/8.

This script provides a single entrypoint to launch the whole ATLAS workflow
with consistent run ids, summaries, and failure handling.

Phase 7 is intentionally reserved and is not part of the public orchestrated path.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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


RUNNERS = {
    "phase1": PROJECT_ROOT / "scripts/phase1_baseline/run_phase1.py",
    "phase2": PROJECT_ROOT / "scripts/phase2_multitask/run_phase2.py",
    "phase3": PROJECT_ROOT / "scripts/phase3_potentials/run_phase3.py",
    "phase4": PROJECT_ROOT / "scripts/phase4_topology/run_phase4.py",
    "phase5": PROJECT_ROOT / "scripts/phase5_active_learning/run_phase5.py",
    "phase6": PROJECT_ROOT / "scripts/phase6_analysis/run_phase6.py",
    "phase8": PROJECT_ROOT / "scripts/phase8_integration/run_phase8.py",
}

PHASE_ORDER = ["phase1", "phase2", "phase3", "phase4", "phase5", "phase6", "phase8"]
RESERVED_PHASE_NOTE = "phase7 is intentionally reserved and not exposed as a runnable public phase."


def _subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["ATLAS_MANIFEST_VISIBILITY"] = args.manifest_visibility
    env.setdefault("ATLAS_DATASET_SOURCE_KEY", "jarvis_dft")
    env.setdefault("ATLAS_DATASET_SNAPSHOT_ID", "jarvis_dft_primary")
    split_manifest = args.split_manifest
    if split_manifest:
        env["ATLAS_SPLIT_MANIFEST"] = str(Path(split_manifest).resolve())
    else:
        default_manifest = PROJECT_ROOT / "artifacts" / "splits" / "split_manifest_iid.json"
        if default_manifest.exists():
            env["ATLAS_SPLIT_MANIFEST"] = str(default_manifest)
    return env


def _parse_phases(raw_phase: str) -> list[str]:
    if raw_phase == "all":
        return list(PHASE_ORDER)
    return [raw_phase]


def _append_if(cmd: list[str], flag: str, value: Any | None) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def build_command(
    *,
    phase: str,
    args: argparse.Namespace,
    run_id: str,
    out_dir: Path,
) -> list[str]:
    cmd = [sys.executable, "-u", str(RUNNERS[phase])]

    if phase == "phase1":
        cmd.extend(
            [
                "--algorithm",
                "cgcnn",
                "--level",
                args.level,
                "--property",
                args.property,
                "--run-id",
                run_id,
                "--top-k",
                str(args.top_k),
                "--keep-last-k",
                str(args.keep_last_k),
                "--manifest-visibility",
                args.manifest_visibility,
            ]
        )
        if args.split_manifest:
            cmd.extend(["--split-manifest", args.split_manifest])
        if args.resume:
            cmd.append("--resume")
        return cmd

    if phase == "phase2":
        cmd.extend(
            [
                "--algorithm",
                args.phase2_algorithm,
                "--level",
                args.level,
                "--run-id",
                run_id,
                "--top-k",
                str(args.top_k),
                "--keep-last-k",
                str(args.keep_last_k),
                "--manifest-visibility",
                args.manifest_visibility,
            ]
        )
        if args.split_manifest:
            cmd.extend(["--split-manifest", args.split_manifest])
        if args.resume:
            cmd.append("--resume")
        if args.phase2_all_properties:
            cmd.append("--all-properties")
        elif args.phase2_property_group:
            cmd.extend(["--property-group", args.phase2_property_group])
        return cmd

    if phase == "phase3":
        cmd.extend(
            [
                "--algorithm",
                args.phase3_algorithm,
                "--level",
                args.level,
                "--run-id",
                run_id,
                "--top-k",
                str(args.top_k),
                "--keep-last-k",
                str(args.keep_last_k),
            ]
        )
        if args.resume:
            cmd.append("--resume")
        if args.phase3_algorithm == "equivariant":
            if args.phase3_all_properties:
                cmd.append("--all-properties")
            else:
                cmd.extend(["--property", args.property])
        if args.phase3_algorithm == "mace":
            if args.phase3_with_forces:
                cmd.append("--with-forces")
            if args.phase3_prepare_mace_data:
                cmd.append("--prepare-mace-data")
        return cmd

    if phase == "phase4":
        cmd.extend(
            [
                "--algorithm",
                args.phase4_algorithm,
                "--level",
                args.level,
                "--run-id",
                run_id,
            ]
        )
        _append_if(cmd, "--top-k", args.top_k)
        _append_if(cmd, "--keep-last-k", args.keep_last_k)
        if args.resume:
            cmd.append("--resume")
        return cmd

    if phase == "phase5":
        cmd.extend(
            [
                "--level",
                args.level,
                "--run-id",
                run_id,
                "--manifest-visibility",
                args.manifest_visibility,
            ]
        )
        if args.resume:
            cmd.append("--resume")
        if args.phase5_no_mace:
            cmd.append("--no-mace")
        _append_if(cmd, "--iterations", args.phase5_iterations)
        _append_if(cmd, "--candidates", args.phase5_candidates)
        _append_if(cmd, "--top", args.phase5_top)
        _append_if(cmd, "--seeds", args.phase5_seeds)
        return cmd

    if phase == "phase6":
        cmd.extend(["--level", args.level])
        if args.phase6_compare:
            cmd.append("--compare")
        else:
            cmd.extend(["--alloy", args.phase6_alloy])
        phase6_json = out_dir / f"{run_id}_phase6.json"
        cmd.extend(["--save-json", str(phase6_json)])
        return cmd

    if phase == "phase8":
        cmd.extend(
            [
                "--level",
                args.level,
                "--run-id",
                run_id,
                "--output-dir",
                str(out_dir / run_id),
            ]
        )
        _append_if(cmd, "--composition-steps", args.phase8_composition_steps)
        _append_if(cmd, "--mepin-images", args.phase8_mepin_images)
        _append_if(cmd, "--liflow-steps", args.phase8_liflow_steps)
        _append_if(cmd, "--liflow-flow-steps", args.phase8_liflow_flow_steps)
        _append_if(cmd, "--seed", args.phase8_seed)
        if args.phase8_skip_alchemy:
            cmd.append("--skip-alchemy")
        if args.phase8_skip_mepin:
            cmd.append("--skip-mepin")
        if args.phase8_skip_liflow:
            cmd.append("--skip-liflow")
        return cmd

    raise ValueError(f"Unsupported phase: {phase}")


def _write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full ATLAS project pipeline",
        epilog=RESERVED_PHASE_NOTE,
    )
    parser.add_argument("--phase", choices=["all", *PHASE_ORDER], default="all")
    parser.add_argument("--level", choices=["smoke", "lite", "std", "pro", "max"], default="std")
    parser.add_argument("--property", default="formation_energy")

    parser.add_argument("--phase2-algorithm", choices=["e3nn", "cgcnn"], default="e3nn")
    parser.add_argument(
        "--phase2-property-group",
        choices=["core4", "priority7", "secondary2", "all9"],
        default=None,
    )
    parser.add_argument("--phase2-all-properties", action="store_true")

    parser.add_argument("--phase3-algorithm", choices=["equivariant", "mace"], default="equivariant")
    parser.add_argument("--phase3-all-properties", action="store_true")
    parser.add_argument("--phase3-with-forces", action="store_true")
    parser.add_argument("--phase3-prepare-mace-data", action="store_true")

    parser.add_argument("--phase4-algorithm", choices=["topognn", "rf"], default="topognn")

    parser.add_argument("--phase5-no-mace", action="store_true")
    parser.add_argument("--phase5-iterations", type=int, default=None)
    parser.add_argument("--phase5-candidates", type=int, default=None)
    parser.add_argument("--phase5-top", type=int, default=None)
    parser.add_argument("--phase5-seeds", type=int, default=None)

    parser.add_argument("--phase6-alloy", type=str, default="SAC305")
    parser.add_argument("--phase6-compare", action="store_true")

    parser.add_argument("--phase8-composition-steps", type=int, default=None)
    parser.add_argument("--phase8-mepin-images", type=int, default=None)
    parser.add_argument("--phase8-liflow-steps", type=int, default=None)
    parser.add_argument("--phase8-liflow-flow-steps", type=int, default=None)
    parser.add_argument("--phase8-seed", type=int, default=None)
    parser.add_argument("--phase8-skip-alchemy", action="store_true")
    parser.add_argument("--phase8-skip-mepin", action="store_true")
    parser.add_argument("--phase8-skip-liflow", action="store_true")

    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--keep-last-k", type=int, default=3)
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--run-id-prefix", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-property-group", type=str, default="priority7")
    parser.add_argument("--preflight-max-samples", type=int, default=0)
    parser.add_argument("--preflight-split-seed", type=int, default=42)
    parser.add_argument("--split-manifest", type=str, default=None)
    parser.add_argument("--manifest-visibility", choices=["internal", "public"], default="internal")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.skip_preflight and not args.dry_run:
        print("[ERROR] --skip-preflight is only allowed together with --dry-run.")
        return 2

    if not args.skip_preflight:
        preflight = run_preflight(
            project_root=PROJECT_ROOT,
            property_group=args.preflight_property_group,
            max_samples=args.preflight_max_samples,
            split_seed=args.preflight_split_seed,
            dry_run=args.dry_run,
        )
        if preflight.return_code != 0:
            print(f"[ERROR] Preflight failed with return code {preflight.return_code}")
            return preflight.return_code
    if args.preflight_only:
        print("[FullProject] Preflight-only mode completed.")
        return 0

    phases = _parse_phases(args.phase)
    session_id = args.session_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = args.run_id_prefix or session_id
    out_dir = PROJECT_ROOT / "artifacts" / "full_project_runs" / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ATLAS Full Project Pipeline")
    print(f"Session ID : {session_id}")
    print(f"Phases     : {phases}")
    print(f"Level      : {args.level}")
    print(f"Dry Run    : {args.dry_run}")
    print("=" * 80)

    summary_rows: list[dict[str, Any]] = []
    error_count = 0

    for phase in phases:
        run_id = f"{run_prefix}_{phase}"
        cmd = build_command(phase=phase, args=args, run_id=run_id, out_dir=out_dir)
        cmd_str = " ".join(cmd)

        print(f"\n[RUN] {phase}")
        print(f"      {cmd_str}")
        t0 = time.time()
        if args.dry_run:
            rc = 0
        else:
            rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=_subprocess_env(args)).returncode
        duration = time.time() - t0

        summary_rows.append(
            {
                "session_id": session_id,
                "phase": phase,
                "run_id": run_id,
                "return_code": rc,
                "duration_sec": round(duration, 2),
                "command": cmd_str,
            }
        )
        _write_summary(out_dir, summary_rows)

        if rc != 0:
            error_count += 1
            print(f"[ERROR] {phase} failed with return code: {rc}")
            if not args.continue_on_error:
                print(f"[ERROR] Aborted. Summary: {out_dir}")
                return rc

    print(f"\n[OK] Completed. Summary directory: {out_dir}")
    if error_count > 0:
        print(f"[WARN] Runs with non-zero return code: {error_count}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
