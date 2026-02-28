#!/usr/bin/env python3
"""
Summarize Phase 1 training outputs into a single, comparable table.

Usage:
  python scripts/phase1_baseline/report_phase1_results.py
  python scripts/phase1_baseline/report_phase1_results.py --property formation_energy --top 15
  python scripts/phase1_baseline/report_phase1_results.py --csv artifacts/phase1_results_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ResultRow:
    tier: str
    run_id: str
    property_name: str
    mae: float
    rmse: float
    r2: float
    maxae: float
    target_mae: float
    passed: bool
    n_train: int
    n_val: int
    n_test: int
    n_params: int
    best_epoch: int
    total_epochs: int
    train_min: float
    result_path: str
    updated_at: str


def _safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v, default=-1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _collect_one(results_path: Path, tier: str, run_id: str) -> ResultRow | None:
    if not results_path.exists():
        return None

    with open(results_path, encoding="utf-8") as f:
        payload = json.load(f)

    property_name = str(payload.get("property", "unknown"))
    metrics = payload.get("test_metrics", {}) or {}

    row = ResultRow(
        tier=tier,
        run_id=run_id,
        property_name=property_name,
        mae=_safe_float(metrics.get(f"{property_name}_MAE")),
        rmse=_safe_float(metrics.get(f"{property_name}_RMSE")),
        r2=_safe_float(metrics.get(f"{property_name}_R2")),
        maxae=_safe_float(metrics.get(f"{property_name}_MaxAE")),
        target_mae=_safe_float(payload.get("target_mae")),
        passed=bool(payload.get("passed", False)),
        n_train=_safe_int(payload.get("n_train", payload.get("n_train_used", -1))),
        n_val=_safe_int(payload.get("n_val", payload.get("n_val_used", -1))),
        n_test=_safe_int(payload.get("n_test", payload.get("n_test_used", -1))),
        n_params=_safe_int(payload.get("n_params")),
        best_epoch=_safe_int(payload.get("best_epoch")),
        total_epochs=_safe_int(payload.get("total_epochs")),
        train_min=_safe_float(payload.get("training_time_minutes")),
        result_path=str(results_path),
        updated_at=datetime.fromtimestamp(results_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    )
    return row


def collect_results(property_name: str, latest_pro_only: bool = False) -> list[ResultRow]:
    rows: list[ResultRow] = []
    models_dir = PROJECT_ROOT / "models"

    for tier, base_name in [("lite", f"cgcnn_lite_{property_name}"), ("std", f"cgcnn_std_{property_name}")]:
        base_dir = models_dir / base_name
        direct_row = _collect_one(base_dir / "results.json", tier=tier, run_id="-")
        if direct_row:
            rows.append(direct_row)
        if base_dir.exists():
            runs = sorted(
                [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
                key=lambda p: p.name,
            )
            for run_dir in runs:
                row = _collect_one(run_dir / "results.json", tier=tier, run_id=run_dir.name)
                if row:
                    rows.append(row)

    pro_base = models_dir / f"cgcnn_pro_{property_name}"
    if pro_base.exists():
        runs = sorted([d for d in pro_base.iterdir() if d.is_dir() and d.name.startswith("run_")], key=lambda p: p.name)
        if latest_pro_only and runs:
            runs = [runs[-1]]
        for run_dir in runs:
            row = _collect_one(run_dir / "results.json", tier="pro", run_id=run_dir.name)
            if row:
                rows.append(row)

    return rows


def _fmt_float(v: float, digits: int = 4) -> str:
    if v != v:  # NaN check
        return "nan"
    return f"{v:.{digits}f}"


def _to_table(rows: list[ResultRow]) -> str:
    headers = [
        "rank",
        "tier",
        "run_id",
        "property",
        "MAE",
        "RMSE",
        "R2",
        "MaxAE",
        "target_MAE",
        "gap(MAE-target)",
        "pass",
        "n_train",
        "n_val",
        "n_test",
        "params",
        "best_ep",
        "epochs",
        "train_min",
        "updated_at",
    ]

    body: list[list[str]] = []
    for i, r in enumerate(rows, start=1):
        gap = r.mae - r.target_mae if (r.mae == r.mae and r.target_mae == r.target_mae) else float("nan")
        body.append([
            str(i),
            r.tier,
            r.run_id,
            r.property_name,
            _fmt_float(r.mae),
            _fmt_float(r.rmse),
            _fmt_float(r.r2),
            _fmt_float(r.maxae),
            _fmt_float(r.target_mae),
            _fmt_float(gap),
            "PASS" if r.passed else "FAIL",
            str(r.n_train),
            str(r.n_val),
            str(r.n_test),
            str(r.n_params),
            str(r.best_epoch),
            str(r.total_epochs),
            _fmt_float(r.train_min, digits=2),
            r.updated_at,
        ])

    widths = [len(h) for h in headers]
    for row in body:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(sep: str = "-") -> str:
        return "+-" + "-+-".join(sep * w for w in widths) + "-+"

    out = []
    out.append(_line("-"))
    out.append("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    out.append(_line("="))
    for row in body:
        out.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    out.append(_line("-"))
    return "\n".join(out)


def _write_csv(path: Path, rows: list[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tier",
            "run_id",
            "property",
            "mae",
            "rmse",
            "r2",
            "maxae",
            "target_mae",
            "passed",
            "n_train",
            "n_val",
            "n_test",
            "n_params",
            "best_epoch",
            "total_epochs",
            "training_time_minutes",
            "result_path",
            "updated_at",
        ])
        for r in rows:
            writer.writerow([
                r.tier,
                r.run_id,
                r.property_name,
                r.mae,
                r.rmse,
                r.r2,
                r.maxae,
                r.target_mae,
                r.passed,
                r.n_train,
                r.n_val,
                r.n_test,
                r.n_params,
                r.best_epoch,
                r.total_epochs,
                r.train_min,
                r.result_path,
                r.updated_at,
            ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Phase 1 results")
    parser.add_argument("--property", default="formation_energy")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--latest-pro-only", action="store_true")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    rows = collect_results(args.property, latest_pro_only=args.latest_pro_only)
    if not rows:
        print(f"[WARN] No Phase 1 results found for property='{args.property}'.")
        return 1

    rows.sort(key=lambda r: (r.mae if r.mae == r.mae else float("inf")))
    rows = rows[: args.top]

    print("=" * 88)
    print("Phase 1 Result Summary")
    print(f"Property: {args.property}")
    print("Metric definitions:")
    print("  MAE   : Mean Absolute Error (lower is better)")
    print("  RMSE  : Root Mean Squared Error (lower is better)")
    print("  R2    : Coefficient of Determination (higher is better)")
    print("  MaxAE : Maximum Absolute Error (lower is better)")
    print("=" * 88)
    print(_to_table(rows))

    if args.csv:
        _write_csv(args.csv, rows)
        print(f"[OK] CSV written to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
