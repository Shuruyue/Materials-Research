#!/usr/bin/env python3
"""
Validate split stability across multiple random seeds.

This script supports Track A / A-08 split protocol governance.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atlas.data.crystal_dataset import (  # noqa: E402
    DEFAULT_PROPERTIES,
    PHASE2_PROPERTY_GROUP_CHOICES,
    PROPERTY_MAP,
    CrystalPropertyDataset,
    resolve_phase2_property_group,
)


def _resolve_properties(property_group: str | None, properties_csv: str | None) -> list[str]:
    if properties_csv:
        values = [x.strip() for x in properties_csv.split(",") if x.strip()]
        if not values:
            raise ValueError("Empty --properties argument.")
        return values
    if property_group:
        return resolve_phase2_property_group(property_group)
    return list(DEFAULT_PROPERTIES)


def _parse_seeds(raw: str) -> list[int]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one seed is required.")
    return [int(v) for v in values]


def _split_jids(ds: CrystalPropertyDataset) -> set[str]:
    df = ds._df  # noqa: SLF001 - governance tool only
    if df is None or "jid" not in df.columns:
        return set()
    return {str(v) for v in df["jid"].dropna().tolist()}


def _compute_property_summary(ds: CrystalPropertyDataset, properties: list[str]) -> dict[str, dict]:
    df = ds._df  # noqa: SLF001 - governance tool only
    rev_map = {v: k for k, v in PROPERTY_MAP.items()}
    out: dict[str, dict] = {}
    for prop in properties:
        stat = {
            "non_nan_count": 0,
            "nan_count": len(ds),
            "mean": None,
            "std": None,
        }
        if df is not None:
            col = rev_map.get(prop)
            if col and col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                stat["non_nan_count"] = int(series.notna().sum())
                stat["nan_count"] = int(series.isna().sum())
                if stat["non_nan_count"] > 0:
                    stat["mean"] = float(series.mean(skipna=True))
                    stat["std"] = float(series.std(skipna=True))
        out[prop] = stat
    return out


def _pair_overlap(a: set[str], b: set[str]) -> int:
    return len(a.intersection(b))


def _pair_jaccard(a: set[str], b: set[str]) -> float:
    union = len(a.union(b))
    if union == 0:
        return 1.0
    return len(a.intersection(b)) / union


def _relative_range(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    vmin = min(values)
    vmax = max(values)
    center = abs(sum(values) / len(values))
    scale = max(center, 1e-12)
    return (vmax - vmin) / scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate split stability across multiple seeds.")
    parser.add_argument("--property-group", choices=PHASE2_PROPERTY_GROUP_CHOICES, default="core4")
    parser.add_argument("--properties", type=str, default=None, help="Comma-separated property names")
    parser.add_argument("--seeds", type=str, default="42,52,62", help="Comma-separated split seeds")
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--stability-filter", type=float, default=0.1)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--min-non-nan", type=int, default=20)
    parser.add_argument("--max-count-cv", type=float, default=0.03)
    parser.add_argument("--max-rel-drift", type=float, default=0.50)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    properties = _resolve_properties(args.property_group, args.properties)
    seeds = _parse_seeds(args.seeds)
    t0 = time.time()

    per_seed: dict[int, dict] = {}
    jid_pool: dict[int, dict[str, set[str]]] = {}

    for seed in seeds:
        seed_payload: dict[str, dict] = {}
        jid_pool[seed] = {}
        for split in ("train", "val", "test"):
            ds = CrystalPropertyDataset(
                properties=properties,
                max_samples=args.max_samples,
                split=split,
                split_seed=seed,
                stability_filter=args.stability_filter,
            ).prepare(force_reload=args.force_reload)
            jids = _split_jids(ds)
            jid_pool[seed][split] = jids
            seed_payload[split] = {
                "n_samples": len(ds),
                "jid_count": len(jids),
                "property_summary": _compute_property_summary(ds, properties),
            }
        per_seed[seed] = seed_payload

    leakage_by_seed: dict[str, dict[str, int]] = {}
    leakage_fail = False
    for seed in seeds:
        train = jid_pool[seed]["train"]
        val = jid_pool[seed]["val"]
        test = jid_pool[seed]["test"]
        leakage = {
            "train__val": _pair_overlap(train, val),
            "train__test": _pair_overlap(train, test),
            "val__test": _pair_overlap(val, test),
        }
        leakage_by_seed[str(seed)] = leakage
        if any(v > 0 for v in leakage.values()):
            leakage_fail = True

    split_stability: dict[str, dict] = {}
    count_cv_fail = False
    drift_fail = False

    for split in ("train", "val", "test"):
        counts = [per_seed[seed][split]["n_samples"] for seed in seeds]
        mean_count = sum(counts) / len(counts)
        count_std = statistics.pstdev(counts) if len(counts) > 1 else 0.0
        count_cv = (count_std / mean_count) if mean_count > 0 else 0.0
        if count_cv > args.max_count_cv:
            count_cv_fail = True

        property_drift: dict[str, dict] = {}
        for prop in properties:
            means: list[float] = []
            contributing_seeds: list[int] = []
            for seed in seeds:
                stat = per_seed[seed][split]["property_summary"][prop]
                if stat["non_nan_count"] >= args.min_non_nan and stat["mean"] is not None:
                    means.append(float(stat["mean"]))
                    contributing_seeds.append(seed)

            rel_drift = _relative_range(means) if means else None
            if rel_drift is not None and rel_drift > args.max_rel_drift:
                drift_fail = True

            property_drift[prop] = {
                "seeds": contributing_seeds,
                "means": means,
                "relative_range": rel_drift,
            }

        pairwise_jaccard = {}
        for s1, s2 in itertools.combinations(seeds, 2):
            key = f"{s1}__{s2}"
            pairwise_jaccard[key] = _pair_jaccard(jid_pool[s1][split], jid_pool[s2][split])

        split_stability[split] = {
            "counts": counts,
            "count_mean": mean_count,
            "count_std": count_std,
            "count_cv": count_cv,
            "pairwise_jaccard": pairwise_jaccard,
            "property_drift": property_drift,
        }

    status = {
        "leakage_ok": not leakage_fail,
        "count_cv_ok": not count_cv_fail,
        "property_drift_ok": not drift_fail,
    }
    status["overall_pass"] = all(status.values())

    payload = {
        "timestamp": time.time(),
        "config": {
            "property_group": args.property_group,
            "properties": properties,
            "seeds": seeds,
            "max_samples": args.max_samples,
            "stability_filter": args.stability_filter,
            "min_non_nan": args.min_non_nan,
            "max_count_cv": args.max_count_cv,
            "max_rel_drift": args.max_rel_drift,
        },
        "per_seed": per_seed,
        "leakage_by_seed": leakage_by_seed,
        "split_stability": split_stability,
        "status": status,
        "duration_sec": round(time.time() - t0, 2),
    }

    if args.output is None:
        out_dir = PROJECT_ROOT / "artifacts" / "program_plan"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        args.output = out_dir / f"split_stability_{stamp}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] Split stability report saved: {args.output}")
    print(
        "  status: "
        f"leakage_ok={status['leakage_ok']}, "
        f"count_cv_ok={status['count_cv_ok']}, "
        f"property_drift_ok={status['property_drift_ok']}, "
        f"overall_pass={status['overall_pass']}"
    )
    for split in ("train", "val", "test"):
        cv = split_stability[split]["count_cv"]
        print(f"  - {split}: count_cv={cv:.5f}")

    if args.strict and not status["overall_pass"]:
        print("[ERROR] Strict mode failed due to split stability checks.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
