#!/usr/bin/env python3
"""
Validate ATLAS crystal dataset contract for train/val/test splits.

Checks:
1) Graph schema integrity (x/edge_index/edge_attr shape consistency)
2) Required property field availability
3) Finite/NaN statistics per property
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atlas.data.crystal_dataset import (  # noqa: E402
    CrystalPropertyDataset,
    DEFAULT_PROPERTIES,
    PHASE2_PROPERTY_GROUP_CHOICES,
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


def _is_finite_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x)


def _validate_data_item(item, required_properties: list[str]) -> dict:
    errors: list[str] = []
    out = {
        "graph_valid": True,
        "errors": errors,
        "property_stats": {},
    }

    if not hasattr(item, "x") or item.x.ndim != 2 or item.x.size(0) <= 0:
        errors.append("invalid_x")
    if not hasattr(item, "edge_index") or item.edge_index.ndim != 2 or item.edge_index.size(0) != 2:
        errors.append("invalid_edge_index")
    if not hasattr(item, "edge_attr") or item.edge_attr.ndim != 2:
        errors.append("invalid_edge_attr")

    if hasattr(item, "edge_index") and hasattr(item, "edge_attr"):
        if item.edge_index.ndim == 2 and item.edge_attr.ndim == 2:
            if item.edge_index.size(1) != item.edge_attr.size(0):
                errors.append("edge_count_mismatch")

    if hasattr(item, "x") and item.x.ndim == 2 and item.x.size(0) > 0:
        if not bool(_is_finite_tensor(item.x).all()):
            errors.append("x_non_finite")
    if hasattr(item, "edge_attr") and item.edge_attr.ndim == 2:
        if not bool(_is_finite_tensor(item.edge_attr).all()):
            errors.append("edge_attr_non_finite")

    for prop in required_properties:
        stat = {"present": False, "finite": False, "is_nan": False}
        if hasattr(item, prop):
            stat["present"] = True
            raw = getattr(item, prop)
            value = raw if torch.is_tensor(raw) else torch.as_tensor(raw)
            flat = value.reshape(-1)
            if flat.numel() > 0:
                finite_mask = torch.isfinite(flat)
                stat["finite"] = bool(finite_mask.any().item())
                stat["is_nan"] = bool(torch.isnan(flat).any().item())
        out["property_stats"][prop] = stat

    if errors:
        out["graph_valid"] = False
    return out


def _validate_split(
    split: str,
    properties: list[str],
    max_samples: int | None,
    split_seed: int,
    stability_filter: float | None,
    force_reload: bool,
) -> dict:
    ds = CrystalPropertyDataset(
        properties=properties,
        max_samples=max_samples,
        split=split,
        split_seed=split_seed,
        stability_filter=stability_filter,
    ).prepare(force_reload=force_reload)

    summary = {
        "split": split,
        "n_samples": len(ds),
        "graph_invalid_count": 0,
        "graph_error_counts": {},
        "property_presence": {p: 0 for p in properties},
        "property_finite": {p: 0 for p in properties},
        "property_nan": {p: 0 for p in properties},
    }

    for i in range(len(ds)):
        item = ds[i]
        check = _validate_data_item(item, properties)
        if not check["graph_valid"]:
            summary["graph_invalid_count"] += 1
            for err in check["errors"]:
                summary["graph_error_counts"][err] = summary["graph_error_counts"].get(err, 0) + 1

        for prop in properties:
            p = check["property_stats"][prop]
            if p["present"]:
                summary["property_presence"][prop] += 1
            if p["finite"]:
                summary["property_finite"][prop] += 1
            if p["is_nan"]:
                summary["property_nan"][prop] += 1

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ATLAS phase datasets against schema contract.")
    parser.add_argument("--property-group", choices=PHASE2_PROPERTY_GROUP_CHOICES, default="priority7")
    parser.add_argument("--properties", type=str, default=None, help="Comma-separated property names")
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--stability-filter", type=float, default=0.1)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Return non-zero if invalid graphs found")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    properties = _resolve_properties(args.property_group, args.properties)

    t0 = time.time()
    payload = {
        "timestamp": time.time(),
        "property_group": args.property_group,
        "properties": properties,
        "max_samples": args.max_samples,
        "split_seed": args.split_seed,
        "stability_filter": args.stability_filter,
        "splits": [],
    }

    for split in ("train", "val", "test"):
        summary = _validate_split(
            split=split,
            properties=properties,
            max_samples=args.max_samples,
            split_seed=args.split_seed,
            stability_filter=args.stability_filter,
            force_reload=args.force_reload,
        )
        payload["splits"].append(summary)

    payload["duration_sec"] = round(time.time() - t0, 2)

    if args.output is None:
        out_dir = PROJECT_ROOT / "artifacts" / "program_plan"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        args.output = out_dir / f"dataset_contract_validation_{stamp}.json"

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] Dataset validation report saved: {args.output}")
    for split in payload["splits"]:
        print(
            f"  - {split['split']}: n={split['n_samples']}, "
            f"graph_invalid={split['graph_invalid_count']}"
        )

    if args.strict and any(s["graph_invalid_count"] > 0 for s in payload["splits"]):
        print("[ERROR] Strict mode failed due to invalid graph samples.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
