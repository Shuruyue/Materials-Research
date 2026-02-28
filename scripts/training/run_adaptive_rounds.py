#!/usr/bin/env python3
"""
Theory-backed adaptive multi-round tuning for Phase 1-4 models.

Key behaviors:
- round 1: literature-backed baseline profile
- round 2/3: auto-adjust based on previous metric quality
- per run: deterministic run-id + manifest lookup for traceability
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

from atlas.console_style import install_console_style
from atlas.training.theory_tuning import (
    DEFAULT_PHASE_ALGORITHMS,
    DEFAULT_STAGE_ORDER,
    adapt_params_for_next_round,
    extract_score_from_manifest,
    get_profile,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = PROJECT_ROOT / "models"

install_console_style()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env

RUNNERS = {
    "phase1": PROJECT_ROOT / "scripts/phase1_baseline/run_phase1.py",
    "phase2": PROJECT_ROOT / "scripts/phase2_multitask/run_phase2.py",
    "phase3": PROJECT_ROOT / "scripts/phase3_potentials/run_phase3.py",
    "phase4": PROJECT_ROOT / "scripts/phase4_topology/run_phase4.py",
}


def _parse_phases(raw_phase: str) -> list[str]:
    if raw_phase == "all":
        return ["phase1", "phase2", "phase3", "phase4"]
    return [raw_phase]


def _run_name(run_id: str) -> str:
    return run_id if run_id.startswith("run_") else f"run_{run_id}"


def _manifest_candidate_dirs(
    *,
    phase: str,
    algorithm: str,
    stage: str,
    property_name: str,
) -> list[Path]:
    if phase == "phase1":
        if stage in {"lite", "smoke"}:
            return [MODELS_ROOT / f"cgcnn_lite_{property_name}"]
        if stage in {"std", "competition"}:
            return [MODELS_ROOT / f"cgcnn_std_{property_name}"]
        return [MODELS_ROOT / f"cgcnn_pro_{property_name}"]
    if phase == "phase2":
        if algorithm == "cgcnn":
            return [MODELS_ROOT / "multitask_cgcnn"]
        if stage == "lite":
            return [MODELS_ROOT / "multitask_lite_e3nn"]
        if stage == "std":
            return [MODELS_ROOT / "multitask_std_e3nn"]
        return [MODELS_ROOT / "multitask_pro_e3nn"]
    if phase == "phase3":
        if algorithm == "mace":
            return [MODELS_ROOT / "mace"]
        return [MODELS_ROOT / f"specialist_{property_name}"]
    if phase == "phase4":
        if algorithm == "rf":
            return [MODELS_ROOT / "topo_classifier_rf"]
        return [MODELS_ROOT / "topo_classifier"]
    return [MODELS_ROOT]


def _find_manifest_path(
    run_id: str,
    *,
    phase: str,
    algorithm: str,
    stage: str,
    property_name: str,
) -> Path | None:
    run_name = _run_name(run_id)
    candidates = _manifest_candidate_dirs(
        phase=phase,
        algorithm=algorithm,
        stage=stage,
        property_name=property_name,
    )
    for base_dir in candidates:
        path = base_dir / run_name / "run_manifest.json"
        if path.exists():
            return path

    if MODELS_ROOT.exists():
        matches = list(MODELS_ROOT.glob(f"**/{run_name}/run_manifest.json"))
        if matches:
            matches.sort(key=lambda p: p.stat().st_mtime)
            return matches[-1]
    return None


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _append_if_present(cmd: list[str], params: dict[str, Any], param: str, flag: str) -> None:
    if param in params and params[param] is not None:
        cmd.extend([flag, str(params[param])])


def build_command(
    *,
    phase: str,
    algorithm: str,
    stage: str,
    params: dict[str, Any],
    run_id: str,
    property_name: str,
    phase2_all_properties: bool,
    phase3_all_properties: bool,
    phase3_with_forces: bool,
    prepare_mace_data: bool,
    top_k: int,
    keep_last_k: int,
) -> list[str]:
    if phase == "phase1":
        cmd = [
            sys.executable,
            "-u",
            str(RUNNERS[phase]),
            "--algorithm",
            algorithm,
            "--property",
            property_name,
        ]
        if params.get("competition") or stage == "competition":
            cmd.append("--competition")
        else:
            cmd.extend(["--level", str(params.get("level", stage))])
        _append_if_present(cmd, params, "epochs", "--epochs")
        _append_if_present(cmd, params, "batch-size", "--batch-size")
        _append_if_present(cmd, params, "lr", "--lr")
        _append_if_present(cmd, params, "max-samples", "--max-samples")
        _append_if_present(cmd, params, "hidden-dim", "--hidden-dim")
        _append_if_present(cmd, params, "n-conv", "--n-conv")
        cmd.extend(["--top-k", str(top_k), "--keep-last-k", str(keep_last_k), "--run-id", run_id])
        return cmd

    if phase == "phase2":
        cmd = [sys.executable, "-u", str(RUNNERS[phase]), "--algorithm", algorithm]
        if params.get("competition") or stage == "competition":
            cmd.append("--competition")
        else:
            cmd.extend(["--level", str(params.get("level", stage))])
        if algorithm == "e3nn" and phase2_all_properties:
            cmd.append("--all-properties")
        _append_if_present(cmd, params, "epochs", "--epochs")
        _append_if_present(cmd, params, "batch-size", "--batch-size")
        _append_if_present(cmd, params, "lr", "--lr")
        if algorithm == "cgcnn":
            _append_if_present(cmd, params, "preset", "--preset")
            _append_if_present(cmd, params, "max-samples", "--max-samples")
        cmd.extend(["--top-k", str(top_k), "--keep-last-k", str(keep_last_k), "--run-id", run_id])
        return cmd

    if phase == "phase3":
        cmd = [sys.executable, "-u", str(RUNNERS[phase]), "--algorithm", algorithm]
        if params.get("competition") or stage == "competition":
            cmd.append("--competition")
        else:
            cmd.extend(["--level", str(params.get("level", stage))])
        if algorithm == "equivariant":
            if phase3_all_properties:
                cmd.append("--all-properties")
            else:
                cmd.extend(["--property", property_name])
        if algorithm == "mace":
            if prepare_mace_data:
                cmd.append("--prepare-mace-data")
            if phase3_with_forces:
                cmd.append("--with-forces")
        _append_if_present(cmd, params, "epochs", "--epochs")
        _append_if_present(cmd, params, "batch-size", "--batch-size")
        _append_if_present(cmd, params, "lr", "--lr")
        _append_if_present(cmd, params, "max-samples", "--max-samples")
        _append_if_present(cmd, params, "acc-steps", "--acc-steps")
        _append_if_present(cmd, params, "r-max", "--r-max")
        cmd.extend(["--top-k", str(top_k), "--keep-last-k", str(keep_last_k), "--run-id", run_id])
        return cmd

    if phase == "phase4":
        cmd = [sys.executable, "-u", str(RUNNERS[phase]), "--algorithm", algorithm]
        if params.get("competition") or stage == "competition":
            cmd.append("--competition")
        else:
            cmd.extend(["--level", str(params.get("level", stage))])

        _append_if_present(cmd, params, "max-samples", "--max-samples")
        if algorithm == "topognn":
            _append_if_present(cmd, params, "epochs", "--epochs")
            _append_if_present(cmd, params, "batch-size", "--batch-size")
            _append_if_present(cmd, params, "lr", "--lr")
            _append_if_present(cmd, params, "hidden", "--hidden")
            cmd.extend(["--top-k", str(top_k), "--keep-last-k", str(keep_last_k)])
        else:
            _append_if_present(cmd, params, "n-estimators", "--n-estimators")
            _append_if_present(cmd, params, "max-depth", "--max-depth")
            _append_if_present(cmd, params, "min-samples-leaf", "--min-samples-leaf")
            cmd.extend(["--top-k", str(top_k)])
        cmd.extend(["--run-id", run_id])
        return cmd

    raise ValueError(f"Unsupported phase: {phase}")


def _phase_algorithms_from_args(args: argparse.Namespace, phase: str) -> list[str]:
    if phase == "phase1":
        return args.phase1_algorithms
    if phase == "phase2":
        return args.phase2_algorithms
    if phase == "phase3":
        return args.phase3_algorithms
    if phase == "phase4":
        return args.phase4_algorithms
    raise ValueError(phase)


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run theory-backed adaptive rounds across Phase 1-4")
    parser.add_argument("--phase", choices=["all", "phase1", "phase2", "phase3", "phase4"], default="all")
    parser.add_argument("--stages", nargs="+", default=DEFAULT_STAGE_ORDER)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--property", default="formation_energy")

    parser.add_argument("--phase1-algorithms", nargs="+", default=DEFAULT_PHASE_ALGORITHMS["phase1"])
    parser.add_argument("--phase2-algorithms", nargs="+", default=DEFAULT_PHASE_ALGORITHMS["phase2"])
    parser.add_argument("--phase3-algorithms", nargs="+", default=DEFAULT_PHASE_ALGORITHMS["phase3"])
    parser.add_argument("--phase4-algorithms", nargs="+", default=DEFAULT_PHASE_ALGORITHMS["phase4"])

    parser.add_argument("--phase2-all-properties", action="store_true")
    parser.add_argument("--phase3-all-properties", action="store_true")
    parser.add_argument("--phase3-with-forces", action="store_true")
    parser.add_argument("--phase3-prepare-mace-data", action="store_true")

    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--keep-last-k", type=int, default=3)
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--disable-auto-tune", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    phases = _parse_phases(args.phase)
    session_id = args.session_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "artifacts" / "adaptive_tuning" / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Adaptive Theory-Backed Tuning")
    print(f"Session ID : {session_id}")
    print(f"Phases     : {phases}")
    print(f"Stages     : {args.stages}")
    print(f"Rounds     : {args.rounds}")
    print(f"Property   : {args.property}")
    print(f"Dry Run    : {args.dry_run}")
    print("=" * 80)

    summary_rows: list[dict[str, Any]] = []
    error_count = 0
    mace_prepare_consumed = False

    for phase in phases:
        algorithms = _phase_algorithms_from_args(args, phase)
        for algorithm in algorithms:
            for stage in args.stages:
                try:
                    profile = get_profile(phase, algorithm, stage)
                except KeyError:
                    print(f"[WARN] No profile: phase={phase}, algorithm={algorithm}, stage={stage} (skip)")
                    continue

                print(f"\n[PROFILE] phase={phase} algorithm={algorithm} stage={stage}")
                params = dict(profile.params)
                prev_score: float | None = None

                for round_idx in range(1, args.rounds + 1):
                    run_id = f"{session_id}_{phase}_{algorithm}_{stage}_r{round_idx}"
                    prepare_mace_data = (
                        args.phase3_prepare_mace_data
                        and phase == "phase3"
                        and algorithm == "mace"
                        and not mace_prepare_consumed
                    )
                    cmd = build_command(
                        phase=phase,
                        algorithm=algorithm,
                        stage=stage,
                        params=params,
                        run_id=run_id,
                        property_name=args.property,
                        phase2_all_properties=args.phase2_all_properties,
                        phase3_all_properties=args.phase3_all_properties,
                        phase3_with_forces=args.phase3_with_forces,
                        prepare_mace_data=prepare_mace_data,
                        top_k=args.top_k,
                        keep_last_k=args.keep_last_k,
                    )
                    if prepare_mace_data:
                        mace_prepare_consumed = True
                    cmd_str = " ".join(cmd)
                    print(f"\n[RUN] {phase}/{algorithm}/{stage} round={round_idx}")
                    print(f"      {cmd_str}")

                    t0 = time.time()
                    if args.dry_run:
                        rc = 0
                    else:
                        rc = subprocess.run(
                            cmd,
                            cwd=str(PROJECT_ROOT),
                            env=_subprocess_env(),
                        ).returncode
                    duration = time.time() - t0

                    manifest_path = _find_manifest_path(
                        run_id,
                        phase=phase,
                        algorithm=algorithm,
                        stage=stage,
                        property_name=args.property,
                    )
                    manifest = _load_json(manifest_path)
                    score = extract_score_from_manifest(manifest, profile.objective)
                    failed = (rc != 0)

                    if args.disable_auto_tune or args.dry_run or round_idx >= args.rounds:
                        next_params = dict(params)
                        adapt_reason = "auto_tune_disabled_or_last_round"
                        improvement = None
                    else:
                        next_params, adapt_reason, improvement = adapt_params_for_next_round(
                            profile=profile,
                            current_params=params,
                            previous_score=prev_score,
                            current_score=score,
                            failed=failed or (score is None),
                        )

                    summary_rows.append(
                        {
                            "session_id": session_id,
                            "phase": phase,
                            "algorithm": algorithm,
                            "stage": stage,
                            "round": round_idx,
                            "run_id": run_id,
                            "return_code": rc,
                            "duration_sec": round(duration, 2),
                            "metric_mode": profile.objective.mode,
                            "metric_keys": ",".join(profile.objective.keys),
                            "score": score,
                            "prev_score": prev_score,
                            "relative_improvement": improvement,
                            "auto_tune_reason": adapt_reason,
                            "params_in": json.dumps(params, ensure_ascii=False, sort_keys=True),
                            "params_out": json.dumps(next_params, ensure_ascii=False, sort_keys=True),
                            "manifest_path": str(manifest_path) if manifest_path else "",
                            "references": ",".join(profile.references),
                            "command": cmd_str,
                        }
                    )

                    if rc != 0:
                        error_count += 1
                        print(f"[ERROR] return_code={rc}")
                        if not args.continue_on_error:
                            _write_summary(out_dir, summary_rows)
                            print(f"[ERROR] Aborted. Summary: {out_dir}")
                            return rc

                    prev_score = score if score is not None else prev_score
                    params = next_params

                    _write_summary(out_dir, summary_rows)

    print(f"\n[OK] Completed. Summary directory: {out_dir}")
    if error_count > 0:
        print(f"[WARN] Runs with non-zero return code: {error_count}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

