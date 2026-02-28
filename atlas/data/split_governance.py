"""
Split Governance for ATLAS.

Provides deterministic IID, compositional, and prototype splits
with SHA-256 manifest generation.

CLI entrypoint: ``make-splits``

Extends existing split logic in :class:`CrystalPropertyDataset` with
OOD-aware splitting strategies.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SPLIT_STRATEGIES = ("iid", "compositional", "prototype", "all")
SPLIT_SCHEMA_VERSION = "2.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ids_hash(ids: list[str]) -> str:
    """Deterministic hash of a sorted ID list."""
    return _sha256_hex("\n".join(sorted(ids)))


def _extract_elements(formula: str) -> frozenset[str]:
    """Extract element symbols from a chemical formula string.

    Simple parser: splits on uppercase letters.
    'Li2Fe3O4' -> {'Li', 'Fe', 'O'}
    """
    import re
    elems = re.findall(r"[A-Z][a-z]?", formula)
    return frozenset(elems)


def _chemical_system(formula: str) -> str:
    """Return sorted chemical system string, e.g. 'Fe-Li-O'."""
    return "-".join(sorted(_extract_elements(formula)))


# ---------------------------------------------------------------------------
# Split manifest
# ---------------------------------------------------------------------------


@dataclass
class SplitManifestV2:
    """Deterministic split manifest with auditable split assignment metadata."""

    schema_version: str = SPLIT_SCHEMA_VERSION
    strategy: str = "iid"
    split_id: str = ""
    seed: int = 42
    timestamp: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )
    split_hash: str = ""
    assignment_hash: str = ""
    dataset_fingerprint: str = ""
    group_definition_version: str = "1"
    splits: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split_strategy": self.strategy,
            "split_id": self.split_id,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "split_hash": self.split_hash,
            "assignment_hash": self.assignment_hash,
            "dataset_fingerprint": self.dataset_fingerprint,
            "group_definition_version": self.group_definition_version,
            "splits": self.splits,
            "metadata": self.metadata,
        }

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    def compute_hash(self) -> str:
        """Compute overall split hash from per-split ID hashes."""
        parts = [f"schema:{self.schema_version}", f"strategy:{self.strategy}", f"seed:{self.seed}"]
        for name in sorted(self.splits.keys()):
            h = self.splits[name].get("sample_ids_hash", "")
            parts.append(f"{name}:{h}")
        self.split_hash = f"sha256:{_sha256_hex('|'.join(parts))}"
        return self.split_hash


# Backward-compatible alias
SplitManifest = SplitManifestV2


# ---------------------------------------------------------------------------
# Splitting functions
# ---------------------------------------------------------------------------


def iid_split(
    sample_ids: list[str],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[str]]:
    """Deterministic random IID split.

    Parameters
    ----------
    sample_ids : list[str]
        Unique sample identifiers.
    seed : int
        Random seed for reproducibility.
    ratios : tuple
        (train, val, test) ratios.

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}``
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(sample_ids))
    rng.shuffle(indices)

    n = len(sample_ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return {
        "train": [sample_ids[i] for i in train_idx],
        "val": [sample_ids[i] for i in val_idx],
        "test": [sample_ids[i] for i in test_idx],
    }


def compositional_split(
    sample_ids: list[str],
    formulas: list[str],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[str]]:
    """Compositional OOD split: groups by chemical system.

    Entire chemical systems (e.g., all Li-Fe-O compounds) are held out
    for validation/test so that test compositions are never seen during training.

    Parameters
    ----------
    sample_ids : list[str]
        Unique sample identifiers (parallel with ``formulas``).
    formulas : list[str]
        Chemical formula per sample.
    seed : int
        Random seed.
    ratios : tuple
        (train, val, test) ratios (by group count, approximately).

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}``
    """
    # Group samples by chemical system
    system_map: dict[str, list[str]] = defaultdict(list)
    for sid, formula in zip(sample_ids, formulas):
        system = _chemical_system(formula) if formula else "unknown"
        system_map[system].append(sid)

    systems = sorted(system_map.keys())
    rng = np.random.RandomState(seed)
    indices = np.arange(len(systems))
    rng.shuffle(indices)

    n = len(systems)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_systems = set(systems[i] for i in indices[:n_train])
    val_systems = set(systems[i] for i in indices[n_train : n_train + n_val])
    # Remaining systems go to test (no need to compute explicitly)

    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for system, sids in system_map.items():
        if system in train_systems:
            result["train"].extend(sids)
        elif system in val_systems:
            result["val"].extend(sids)
        else:
            result["test"].extend(sids)

    return result


def prototype_split(
    sample_ids: list[str],
    spacegroups: list[int | str],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[str]]:
    """Prototype OOD split: groups by crystal structure prototype (spacegroup).

    Entire spacegroups are held out for test so that test structures
    are never seen during training.

    Parameters
    ----------
    sample_ids : list[str]
        Unique sample identifiers (parallel with ``spacegroups``).
    spacegroups : list
        Spacegroup number or label per sample.
    seed : int
        Random seed.
    ratios : tuple
        (train, val, test) ratios (by group count, approximately).

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}``
    """
    # Group by spacegroup
    sg_map: dict[str, list[str]] = defaultdict(list)
    for sid, sg in zip(sample_ids, spacegroups):
        sg_key = str(sg) if sg else "unknown"
        sg_map[sg_key].append(sid)

    groups = sorted(sg_map.keys())
    rng = np.random.RandomState(seed)
    indices = np.arange(len(groups))
    rng.shuffle(indices)

    n = len(groups)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_groups = set(groups[i] for i in indices[:n_train])
    val_groups = set(groups[i] for i in indices[n_train : n_train + n_val])
    # Remaining groups go to test (no need to compute explicitly)

    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for sg_key, sids in sg_map.items():
        if sg_key in train_groups:
            result["train"].extend(sids)
        elif sg_key in val_groups:
            result["val"].extend(sids)
        else:
            result["test"].extend(sids)

    return result


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------


def build_assignment_records(
    splits: dict[str, list[str]],
    *,
    group_by_id: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Build a deterministic row-wise split assignment list."""
    rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        for sample_id in sorted(splits.get(split, [])):
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "split": split,
                    "group": str(group_by_id.get(sample_id, "")) if group_by_id else "",
                }
            )
    return rows


def compute_split_overlap_counts(splits: dict[str, list[str]]) -> dict[str, int]:
    """Compute pairwise overlap counts for split integrity checks."""
    train = set(splits.get("train", []))
    val = set(splits.get("val", []))
    test = set(splits.get("test", []))
    return {
        "train__val": len(train & val),
        "train__test": len(train & test),
        "val__test": len(val & test),
    }


def _assignment_hash(assignment_rows: list[dict[str, str]]) -> str:
    payload = json.dumps(assignment_rows, sort_keys=True, separators=(",", ":"))
    return f"sha256:{_sha256_hex(payload)}"


def generate_manifest(
    strategy: str,
    splits: dict[str, list[str]],
    *,
    seed: int = 42,
    split_id: str = "",
    dataset_fingerprint: str = "",
    group_definition_version: str = "1",
    assignment_rows: list[dict[str, str]] | None = None,
    group_by_id: dict[str, str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> SplitManifestV2:
    """Build a split manifest from computed splits."""
    if assignment_rows is None:
        assignment_rows = build_assignment_records(splits, group_by_id=group_by_id)

    manifest = SplitManifestV2(
        strategy=strategy,
        seed=seed,
        split_id=split_id,
        dataset_fingerprint=dataset_fingerprint,
        group_definition_version=group_definition_version,
        assignment_hash=_assignment_hash(assignment_rows),
    )

    for name, ids in splits.items():
        group_count = 0
        if group_by_id:
            group_count = len({group_by_id.get(sample_id, "") for sample_id in ids})
        manifest.splits[name] = {
            "n_samples": len(ids),
            "sample_ids_hash": f"sha256:{_ids_hash(ids)}",
            "n_groups": group_count,
        }
    if extra_metadata:
        manifest.metadata = extra_metadata
    manifest.compute_hash()
    if not manifest.split_id:
        short_hash = manifest.split_hash.split(":", 1)[-1][:12]
        manifest.split_id = f"{strategy}_s{seed}_{short_hash}"
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make-splits",
        description="ATLAS split governance: deterministic IID, compositional, and prototype splits.",
    )
    parser.add_argument(
        "--strategy",
        choices=SPLIT_STRATEGIES,
        default="iid",
        help="Split strategy (default: iid)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits (default: 42)",
    )
    parser.add_argument(
        "--property-group",
        type=str,
        default="priority7",
        help="Property group for dataset loading",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000,
        help="Max samples (default: 3000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for split manifests (default: artifacts/splits/)",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.8,0.1,0.1",
        help="Train,val,test ratios (default: 0.8,0.1,0.1)",
    )
    parser.add_argument(
        "--emit-assignment",
        action="store_true",
        help="Emit per-sample assignment files (JSON + CSV)",
    )
    parser.add_argument(
        "--split-id",
        type=str,
        default="",
        help="Optional explicit split identifier (default: auto-generated)",
    )
    parser.add_argument(
        "--group-definition-version",
        type=str,
        default="1",
        help="Version string for grouping logic definition",
    )
    return parser


def _write_assignment_json(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return path


def _write_assignment_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "split", "group"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for make-splits."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Parse ratios
    try:
        ratios = tuple(float(x) for x in args.ratios.split(","))
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 0.01:
            raise ValueError
    except (ValueError, TypeError):
        print("[ERROR] --ratios must be 3 comma-separated floats summing to 1.0")
        return 1

    # Determine strategies to run
    strategies = (
        ["iid", "compositional", "prototype"]
        if args.strategy == "all"
        else [args.strategy]
    )

    try:
        from atlas.data.crystal_dataset import (
            PROPERTY_MAP,
            resolve_phase2_property_group,
        )
        from atlas.data.jarvis_client import JARVISClient
    except ImportError as exc:
        print(f"[ERROR] Cannot import dataset module: {exc}")
        return 1

    # Resolve properties
    try:
        properties = resolve_phase2_property_group(args.property_group)
    except (KeyError, ValueError):
        from atlas.data.crystal_dataset import DEFAULT_PROPERTIES
        properties = list(DEFAULT_PROPERTIES)

    # Load raw DataFrame (no graph building needed for splits)
    try:
        import pandas as pd

        client = JARVISClient()
        df = client.load_dft_3d()

        # Filter valid rows (same logic as CrystalPropertyDataset)
        rev_map = {v: k for k, v in PROPERTY_MAP.items()}
        jarvis_cols = [rev_map[p] for p in properties if p in rev_map]
        valid_mask = pd.Series(False, index=df.index)
        for col in jarvis_cols:
            if col in df.columns:
                valid_mask |= df[col].notna() & (df[col] != "na")
        df = df[valid_mask].copy()

        # Stability filter
        if "ehull" in df.columns:
            df = df[df["ehull"].notna() & (df["ehull"] <= 0.1)]

        df = df[df["atoms"].notna()].reset_index(drop=True)

        if args.max_samples and len(df) > args.max_samples:
            df = df.sample(args.max_samples, random_state=args.seed).reset_index(drop=True)

    except Exception as exc:
        print(f"[ERROR] Could not load dataset: {exc}")
        return 1

    if len(df) == 0:
        print("[ERROR] Dataset is empty after filtering")
        return 1

    print(f"[make-splits] Loaded {len(df)} samples for splitting")

    sample_ids = [str(x) for x in df["jid"].tolist()] if "jid" in df.columns else [str(i) for i in range(len(df))]
    formulas = [str(x) for x in df["formula"].tolist()] if "formula" in df.columns else [""] * len(df)
    spacegroups = (
        [str(x) for x in df["spg_number"].tolist()]
        if "spg_number" in df.columns
        else ["unknown"] * len(df)
    )
    dataset_fingerprint = f"sha256:{_ids_hash(sample_ids)}"

    # Resolve output dir
    from atlas.config import get_config
    cfg = get_config()
    output_dir = args.output_dir or (cfg.paths.artifacts_dir / "splits")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run each strategy
    for strategy in strategies:
        group_by_id: dict[str, str] = {}
        if strategy == "iid":
            splits = iid_split(sample_ids, seed=args.seed, ratios=ratios)
            group_by_id = {sid: "iid" for sid in sample_ids}
        elif strategy == "compositional":
            splits = compositional_split(
                sample_ids, formulas, seed=args.seed, ratios=ratios
            )
            group_by_id = {
                sid: (_chemical_system(formula) if formula else "unknown")
                for sid, formula in zip(sample_ids, formulas)
            }
        elif strategy == "prototype":
            splits = prototype_split(
                sample_ids, spacegroups, seed=args.seed, ratios=ratios
            )
            group_by_id = {
                sid: (str(spacegroup) if spacegroup else "unknown")
                for sid, spacegroup in zip(sample_ids, spacegroups)
            }
        else:
            print(f"[WARN] Unknown strategy: {strategy}")
            continue

        assignment_rows = build_assignment_records(splits, group_by_id=group_by_id)
        overlap_counts = compute_split_overlap_counts(splits)
        if sum(overlap_counts.values()) > 0:
            print(f"[ERROR] Split overlap detected for strategy='{strategy}': {overlap_counts}")
            return 2
        split_id = args.split_id
        if split_id and args.strategy == "all":
            split_id = f"{split_id}_{strategy}"
        manifest = generate_manifest(
            strategy,
            splits,
            seed=args.seed,
            split_id=split_id,
            dataset_fingerprint=dataset_fingerprint,
            group_definition_version=args.group_definition_version,
            assignment_rows=assignment_rows,
            group_by_id=group_by_id,
            extra_metadata={
                "property_group": args.property_group,
                "max_samples": args.max_samples,
                "ratios": list(ratios),
                "n_total": len(sample_ids),
                "assignment_json": f"split_assignment_{strategy}.json",
                "assignment_csv": f"split_assignment_{strategy}.csv",
                "overlap_counts": overlap_counts,
            },
        )

        out_path = output_dir / f"split_manifest_{strategy}.json"
        manifest.to_json(out_path)
        if args.emit_assignment:
            assignment_json = output_dir / f"split_assignment_{strategy}.json"
            assignment_csv = output_dir / f"split_assignment_{strategy}.csv"
            _write_assignment_json(assignment_json, assignment_rows)
            _write_assignment_csv(assignment_csv, assignment_rows)

        print(f"[make-splits] {strategy} manifest saved: {out_path}")
        for name, info in manifest.splits.items():
            print(f"  {name}: n={info['n_samples']}")
        print(
            "  overlap: "
            f"train__val={overlap_counts['train__val']}, "
            f"train__test={overlap_counts['train__test']}, "
            f"val__test={overlap_counts['val__test']}"
        )
        print(f"  hash: {manifest.split_hash}")
        if args.emit_assignment:
            print(
                f"  assignment: json={output_dir / f'split_assignment_{strategy}.json'}, "
                f"csv={output_dir / f'split_assignment_{strategy}.csv'}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
