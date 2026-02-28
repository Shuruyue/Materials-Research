#!/usr/bin/env python3
"""
Generate a reproducible dataset manifest for CrystalPropertyDataset.

The manifest is intended for Track A / A-07 data version governance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atlas.config import get_config  # noqa: E402
from atlas.data.crystal_dataset import (  # noqa: E402
    DEFAULT_PROPERTIES,
    PHASE2_PROPERTY_GROUP_CHOICES,
    PROPERTY_MAP,
    CrystalPropertyDataset,
    resolve_phase2_property_group,
)
from atlas.data.source_registry import DATA_SOURCES  # noqa: E402


def _resolve_properties(property_group: str | None, properties_csv: str | None) -> list[str]:
    if properties_csv:
        values = [x.strip() for x in properties_csv.split(",") if x.strip()]
        if not values:
            raise ValueError("Empty --properties argument.")
        return values
    if property_group:
        return resolve_phase2_property_group(property_group)
    return list(DEFAULT_PROPERTIES)


def _cache_file_for_split(
    *,
    split: str,
    properties: list[str],
    max_samples: int | None,
    split_seed: int,
    stability_filter: float | None,
) -> Path:
    cfg = get_config()
    props_key = "_".join(sorted(properties))
    filter_key = f"stab{stability_filter}" if stability_filter is not None else "nofilter"
    max_key = f"max{max_samples}" if max_samples else "full"
    cache_hash = hashlib.md5(f"{props_key}_{filter_key}_{max_key}".encode("utf-8")).hexdigest()[:8]
    return cfg.paths.processed_dir / "multi_property" / f"{split}_{split_seed}_{cache_hash}.pt"


def _hash_values(values: list[str]) -> str:
    text = "\n".join(sorted(values))
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _split_summary(ds: CrystalPropertyDataset, properties: list[str]) -> dict:
    df = ds._df  # noqa: SLF001 - governance tool only
    rev_map = {v: k for k, v in PROPERTY_MAP.items()}

    jids: list[str] = []
    formulas: list[str] = []
    if df is not None:
        if "jid" in df.columns:
            jids = sorted({str(v) for v in df["jid"].dropna().tolist()})
        if "formula" in df.columns:
            formulas = sorted({str(v) for v in df["formula"].dropna().tolist()})

    property_stats: dict[str, dict[str, float | int | None]] = {}
    for prop in properties:
        stat: dict[str, float | int | None] = {
            "count_total": len(ds),
            "count_non_nan": 0,
            "count_nan": len(ds),
            "mean": None,
            "std": None,
        }

        if df is not None:
            source_col = rev_map.get(prop)
            if source_col and source_col in df.columns:
                series = pd.to_numeric(df[source_col], errors="coerce")
                non_nan = int(series.notna().sum())
                nan = int(series.isna().sum())
                stat["count_non_nan"] = non_nan
                stat["count_nan"] = nan
                if non_nan > 0:
                    stat["mean"] = float(series.mean(skipna=True))
                    stat["std"] = float(series.std(skipna=True))

        property_stats[prop] = stat

    return {
        "n_samples": len(ds),
        "jid_count": len(jids),
        "jid_hash": _hash_values(jids) if jids else None,
        "formula_count": len(formulas),
        "formula_hash": _hash_values(formulas) if formulas else None,
        "property_stats": property_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset manifest for CrystalPropertyDataset.")
    parser.add_argument("--property-group", choices=PHASE2_PROPERTY_GROUP_CHOICES, default="priority7")
    parser.add_argument("--properties", type=str, default=None, help="Comma-separated property names")
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--stability-filter", type=float, default=0.1)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    properties = _resolve_properties(args.property_group, args.properties)

    t0 = time.time()
    cfg = get_config()
    source_spec = DATA_SOURCES.get("jarvis_dft")
    raw_cache = cfg.paths.raw_dir / "jarvis_cache" / "dft_3d.pkl"
    raw_json = cfg.paths.raw_dir / "jarvis_cache" / "dft_3d.json"

    manifest = {
        "timestamp": time.time(),
        "dataset_source": {
            "key": source_spec.key,
            "name": source_spec.name,
            "domain": source_spec.domain,
            "url": source_spec.url,
            "citation": source_spec.citation,
        },
        "dataset_config": {
            "property_group": args.property_group,
            "properties": properties,
            "max_samples": args.max_samples,
            "split_seed": args.split_seed,
            "stability_filter": args.stability_filter,
        },
        "paths": {
            "raw_cache_pkl": str(raw_cache),
            "raw_cache_json": str(raw_json),
            "raw_cache_pkl_exists": raw_cache.exists(),
            "raw_cache_json_exists": raw_json.exists(),
        },
        "splits": {},
    }

    split_hash_inputs: list[str] = []
    for split in ("train", "val", "test"):
        ds = CrystalPropertyDataset(
            properties=properties,
            max_samples=args.max_samples,
            split=split,
            split_seed=args.split_seed,
            stability_filter=args.stability_filter,
        ).prepare(force_reload=args.force_reload)

        summary = _split_summary(ds, properties)
        cache_file = _cache_file_for_split(
            split=split,
            properties=properties,
            max_samples=args.max_samples,
            split_seed=args.split_seed,
            stability_filter=args.stability_filter,
        )
        summary["cache_file"] = str(cache_file)
        summary["cache_exists"] = cache_file.exists()
        summary["cache_size_bytes"] = cache_file.stat().st_size if cache_file.exists() else None
        manifest["splits"][split] = summary

        if summary["jid_hash"] is not None:
            split_hash_inputs.append(f"{split}:{summary['jid_hash']}")

    manifest["dataset_fingerprint"] = {
        "method": "md5(join(split:jid_hash))",
        "value": hashlib.md5("\n".join(sorted(split_hash_inputs)).encode("utf-8")).hexdigest(),
    }
    manifest["duration_sec"] = round(time.time() - t0, 2)

    if args.output is None:
        out_dir = PROJECT_ROOT / "artifacts" / "program_plan"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        args.output = out_dir / f"dataset_manifest_{stamp}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[OK] Dataset manifest saved: {args.output}")
    for split, summary in manifest["splits"].items():
        print(
            f"  - {split}: n={summary['n_samples']}, "
            f"jid={summary['jid_count']}, cache={summary['cache_exists']}"
        )
    print(f"  fingerprint={manifest['dataset_fingerprint']['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
