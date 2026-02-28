#!/usr/bin/env python3
"""
Create a compact code/data/model version snapshot.

This script supports Track B / B-06 version policy execution.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atlas import __version__ as atlas_version  # noqa: E402


def _run_git(args: list[str]) -> str | None:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _latest_file(glob_pattern: str) -> Path | None:
    matches = sorted((PROJECT_ROOT / "artifacts" / "program_plan").glob(glob_pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create version snapshot for code/data/model governance.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    commit = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(["status", "--porcelain"]) or ""
    short_commit = commit[:12] if commit else "unknown"

    latest_dataset_manifest = _latest_file("dataset_manifest_*.json")
    latest_split_stability = _latest_file("split_stability_*.json")
    latest_interface_contract = _latest_file("phase_interface_contract_*.json")
    latest_run_manifest_contract = _latest_file("run_manifest_contract_*.json")

    run_manifests = list((PROJECT_ROOT / "models").glob("**/run_manifest.json"))
    run_manifests.extend((PROJECT_ROOT / "data" / "discovery_results").glob("**/run_manifest.json"))
    run_manifests.extend((PROJECT_ROOT / "artifacts" / "full_project_runs").glob("**/run_manifest.json"))

    snapshot = {
        "timestamp": now,
        "code": {
            "atlas_version": atlas_version,
            "git_commit": commit,
            "git_branch": branch,
            "git_dirty": bool(status.strip()),
            "semantic_code_version": f"{atlas_version}+{short_commit}",
        },
        "data": {
            "latest_dataset_manifest": str(latest_dataset_manifest) if latest_dataset_manifest else None,
            "latest_split_stability_report": str(latest_split_stability) if latest_split_stability else None,
        },
        "model": {
            "run_manifest_count": len(run_manifests),
            "latest_run_manifest_contract_report": (
                str(latest_run_manifest_contract) if latest_run_manifest_contract else None
            ),
        },
        "interface": {
            "latest_phase_interface_contract_report": (
                str(latest_interface_contract) if latest_interface_contract else None
            ),
        },
    }

    if args.output is None:
        out_dir = PROJECT_ROOT / "artifacts" / "program_plan"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = out_dir / f"version_snapshot_{stamp}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print(f"[OK] Version snapshot saved: {args.output}")
    print(f"  semantic_code_version={snapshot['code']['semantic_code_version']}")
    print(f"  run_manifest_count={snapshot['model']['run_manifest_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
