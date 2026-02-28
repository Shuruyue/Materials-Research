#!/usr/bin/env python3
"""
Check data leakage across train/val/test splits for CrystalPropertyDataset.

Leakage keys:
1) jid overlap (strict)
2) atoms_hash overlap (strict)
3) formula overlap (warning by default; strict with --strict-formula)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atlas.data.crystal_dataset import (  # noqa: E402
    DEFAULT_PROPERTIES,
    PHASE2_PROPERTY_GROUP_CHOICES,
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


def _stable_atoms_hash(atoms_obj) -> str:
    try:
        payload = json.dumps(atoms_obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        payload = str(atoms_obj)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _split_keys(ds: CrystalPropertyDataset) -> dict[str, set[str]]:
    keys = {
        "jid": set(),
        "atoms_hash": set(),
        "formula": set(),
    }
    if ds._df is None:  # noqa: SLF001 - dev-only leakage tool
        return keys

    for row in ds._df.to_dict("records"):  # noqa: SLF001
        jid = row.get("jid")
        if jid is not None:
            keys["jid"].add(str(jid))

        atoms = row.get("atoms")
        if atoms is not None:
            keys["atoms_hash"].add(_stable_atoms_hash(atoms))

        formula = row.get("formula")
        if formula is not None:
            keys["formula"].add(str(formula))

    return keys


def _pair_overlap(a: set[str], b: set[str]) -> int:
    return len(a.intersection(b))


def _compute_overlap_report(split_keys: dict[str, dict[str, set[str]]]) -> dict:
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    report: dict[str, dict[str, int]] = {}
    for left, right in pairs:
        key = f"{left}__{right}"
        report[key] = {}
        for field in ("jid", "atoms_hash", "formula"):
            report[key][field] = _pair_overlap(split_keys[left][field], split_keys[right][field])
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check split leakage across train/val/test.")
    parser.add_argument("--property-group", choices=PHASE2_PROPERTY_GROUP_CHOICES, default="priority7")
    parser.add_argument("--properties", type=str, default=None, help="Comma-separated property names")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--stability-filter", type=float, default=0.1)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--strict-formula", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    properties = _resolve_properties(args.property_group, args.properties)

    split_keys: dict[str, dict[str, set[str]]] = {}
    for split in ("train", "val", "test"):
        ds = CrystalPropertyDataset(
            properties=properties,
            max_samples=args.max_samples,
            split=split,
            split_seed=args.split_seed,
            stability_filter=args.stability_filter,
        ).prepare(force_reload=args.force_reload)
        split_keys[split] = _split_keys(ds)

    overlap = _compute_overlap_report(split_keys)
    payload = {
        "timestamp": time.time(),
        "property_group": args.property_group,
        "properties": properties,
        "max_samples": args.max_samples,
        "split_seed": args.split_seed,
        "stability_filter": args.stability_filter,
        "overlap": overlap,
    }

    if args.output is None:
        out_dir = PROJECT_ROOT / "artifacts" / "program_plan"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        args.output = out_dir / f"split_leakage_check_{stamp}.json"

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] Leakage report saved: {args.output}")
    hard_fail = False
    for pair, metrics in overlap.items():
        print(
            f"  - {pair}: jid={metrics['jid']}, "
            f"atoms_hash={metrics['atoms_hash']}, formula={metrics['formula']}"
        )
        if metrics["jid"] > 0 or metrics["atoms_hash"] > 0:
            hard_fail = True
        if args.strict_formula and metrics["formula"] > 0:
            hard_fail = True

    if hard_fail:
        print("[ERROR] Leakage detected under current strictness.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
