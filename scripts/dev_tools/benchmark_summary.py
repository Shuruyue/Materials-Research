#!/usr/bin/env python3
"""
Summarize Phase 1/Phase 2 run artifacts into a flat table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_ROOT = PROJECT_ROOT / "models"

PHASE2_FAMILIES = (
    "multitask_lite_e3nn",
    "multitask_std_e3nn",
    "multitask_pro_e3nn",
    "multitask_cgcnn",
)

PHASE1_PREFIXES = (
    "cgcnn_lite_",
    "cgcnn_std_",
    "cgcnn_pro_",
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _run_dirs(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    runs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    runs.sort(key=lambda p: p.name)
    return runs


def _extract_row(phase: str, family: str, run_dir: Path) -> dict[str, Any]:
    results = _load_json(run_dir / "results.json") or {}
    manifest = _load_json(run_dir / "run_manifest.json") or {}
    result_manifest = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    hyper = results.get("hyperparameters") if isinstance(results.get("hyperparameters"), dict) else {}

    row = {
        "phase": phase,
        "family": family,
        "run_id": run_dir.name,
        "algorithm": results.get("algorithm", ""),
        "best_epoch": results.get("best_epoch", result_manifest.get("best_epoch", "")),
        "total_epochs": results.get("total_epochs", result_manifest.get("total_epochs", "")),
        "best_val_mae": results.get("best_val_mae", result_manifest.get("best_val_mae", "")),
        "avg_test_mae": results.get("avg_test_mae", result_manifest.get("avg_test_mae", "")),
        "n_train": results.get("n_train", ""),
        "n_val": results.get("n_val", ""),
        "n_test": results.get("n_test", ""),
        "epochs_hp": hyper.get("epochs", ""),
        "batch_size_hp": hyper.get("batch_size", ""),
        "lr_hp": hyper.get("lr", ""),
        "updated_at": manifest.get("updated_at", ""),
        "path": str(run_dir),
    }

    if family.startswith("cgcnn_") and phase == "phase1":
        for prefix in PHASE1_PREFIXES:
            if family.startswith(prefix):
                row["property"] = family[len(prefix):]
                break
        else:
            row["property"] = ""
    else:
        row["property"] = ""

    return row


def collect_rows(scope: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if scope in {"all", "phase1"} and MODELS_ROOT.exists():
        for base in MODELS_ROOT.iterdir():
            if not base.is_dir():
                continue
            if not any(base.name.startswith(prefix) for prefix in PHASE1_PREFIXES):
                continue
            for run_dir in _run_dirs(base):
                rows.append(_extract_row("phase1", base.name, run_dir))

    if scope in {"all", "phase2"}:
        for family in PHASE2_FAMILIES:
            base = MODELS_ROOT / family
            for run_dir in _run_dirs(base):
                rows.append(_extract_row("phase2", family, run_dir))

    rows.sort(key=lambda r: (r["phase"], r["family"], r["run_id"]))
    return rows


def _print_preview(rows: list[dict[str, Any]], limit: int = 20) -> None:
    print("=" * 120)
    print("Benchmark Summary")
    print("=" * 120)
    if not rows:
        print("No runs found.")
        return

    header = (
        f"{'phase':<7} {'family':<24} {'run_id':<22} "
        f"{'best_val_mae':>12} {'avg_test_mae':>12} {'epochs':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows[:limit]:
        print(
            f"{str(row['phase']):<7} {str(row['family']):<24} {str(row['run_id']):<22} "
            f"{str(row['best_val_mae']):>12} {str(row['avg_test_mae']):>12} {str(row['total_epochs']):>8}"
        )
    if len(rows) > limit:
        print(f"... ({len(rows) - limit} more rows)")


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Phase 1/2 benchmark runs")
    parser.add_argument("--scope", choices=("all", "phase1", "phase2"), default="all")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "reports" / "benchmark_summary.csv",
    )
    parser.add_argument("--preview", type=int, default=20, help="Number of rows to print")
    args = parser.parse_args()

    rows = collect_rows(args.scope)
    _print_preview(rows, limit=max(args.preview, 1))
    _write_csv(rows, args.output)
    print(f"CSV saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
