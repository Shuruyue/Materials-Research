#!/usr/bin/env python3
"""
Validate run_manifest.json contract across experiment outputs.

This validator enforces run_manifest v2 completeness for reproducibility audits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_REQUIRED_TOP = [
    "schema_version",
    "visibility",
    "created_at",
    "updated_at",
    "run_id",
    "runtime",
    "args",
    "dataset",
    "split",
    "environment_lock",
    "artifacts",
    "metrics",
    "seeds",
    "configs",
]
DEFAULT_REQUIRED_RUNTIME = [
    "argv",
    "python_version",
    "platform",
    "git",
]
VALID_VISIBILITY = {"internal", "public"}


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate run_manifest.json contract.")
    parser.add_argument(
        "--roots",
        type=str,
        default="models,data/discovery_results,artifacts/full_project_runs",
        help="Comma-separated root directories to scan",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of manifests to scan")
    parser.add_argument(
        "--required-top",
        type=str,
        default=",".join(DEFAULT_REQUIRED_TOP),
        help="Comma-separated required top-level keys",
    )
    parser.add_argument(
        "--required-runtime",
        type=str,
        default=",".join(DEFAULT_REQUIRED_RUNTIME),
        help="Comma-separated required runtime.* keys",
    )
    parser.add_argument("--schema-version", type=str, default="2.0")
    parser.add_argument(
        "--strict-legacy",
        action="store_true",
        help="Treat legacy manifests without schema_version as hard failures.",
    )
    parser.add_argument(
        "--strict-completeness",
        action="store_true",
        help="Require non-empty critical fields in dataset/split/environment/artifacts/seeds/configs blocks.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _collect_manifest_paths(roots: list[Path], limit: int | None) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        matches = list(root.rglob("run_manifest.json"))
        matches.sort(key=lambda p: str(p))
        paths.extend(matches)
    if limit is not None:
        return paths[:limit]
    return paths


def _is_public_runtime_redacted(runtime_obj: dict) -> bool:
    hostname = runtime_obj.get("hostname")
    cwd = runtime_obj.get("cwd")
    pid = runtime_obj.get("pid")
    return hostname == "<redacted>" and cwd == "<redacted>" and pid == "<redacted>"


def main() -> int:
    args = parse_args()
    required_top = _parse_csv(args.required_top)
    required_runtime = _parse_csv(args.required_runtime)
    roots = [(PROJECT_ROOT / p.strip()).resolve() for p in _parse_csv(args.roots)]
    t0 = time.time()

    manifest_paths = _collect_manifest_paths(roots, args.limit)
    invalid_count = 0
    top_missing_total: dict[str, int] = {}
    runtime_missing_total: dict[str, int] = {}
    records: list[dict] = []
    legacy_count = 0
    legacy_skipped = 0

    for path in manifest_paths:
        record = {
            "path": str(path),
            "parse_ok": False,
            "missing_top_keys": [],
            "missing_runtime_keys": [],
            "errors": [],
        }
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise ValueError("manifest is not a JSON object")
            record["parse_ok"] = True
            schema_version = str(payload.get("schema_version", "")).strip()
            if not schema_version:
                legacy_count += 1
                record["legacy_manifest"] = True
                if args.strict_legacy:
                    record["errors"].append(
                        "legacy manifest missing schema_version (strict-legacy enabled)"
                    )
                    invalid_count += 1
                else:
                    record["legacy_skipped"] = True
                    legacy_skipped += 1
                    record["complete"] = False
                records.append(record)
                continue

            missing_top = [k for k in required_top if k not in payload]
            runtime_obj = payload.get("runtime", {})
            if not isinstance(runtime_obj, dict):
                runtime_obj = {}
            missing_runtime = [k for k in required_runtime if k not in runtime_obj]

            record["missing_top_keys"] = missing_top
            record["missing_runtime_keys"] = missing_runtime

            for key in missing_top:
                top_missing_total[key] = top_missing_total.get(key, 0) + 1
            for key in missing_runtime:
                runtime_missing_total[key] = runtime_missing_total.get(key, 0) + 1

            if schema_version != args.schema_version:
                record["errors"].append(
                    f"schema_version mismatch: got '{schema_version}', expected '{args.schema_version}'"
                )

            visibility = payload.get("visibility")
            if visibility not in VALID_VISIBILITY:
                record["errors"].append(
                    f"invalid visibility: {visibility!r} (allowed: {sorted(VALID_VISIBILITY)})"
                )
            if visibility == "public" and not _is_public_runtime_redacted(runtime_obj):
                record["errors"].append(
                    "public manifest runtime fields are not fully redacted "
                    "(hostname/cwd/pid must be '<redacted>')"
                )

            if args.strict_completeness:
                phase_name = str(payload.get("phase", "")).strip().lower()
                requires_dataset_split = phase_name in {
                    "phase1",
                    "phase2",
                    "phase3",
                    "phase4",
                    "benchmark",
                    "full_project",
                }
                required_blocks = {
                    "environment_lock": ["lock_file", "lock_hash"],
                    "artifacts": ["run_manifest_json"],
                    "configs": ["entrypoint"],
                }
                if requires_dataset_split:
                    required_blocks["dataset"] = ["source_key", "snapshot_id"]
                    required_blocks["split"] = ["manifest_path", "split_id"]
                for block_name, fields in required_blocks.items():
                    block_obj = payload.get(block_name, {})
                    if not isinstance(block_obj, dict):
                        record["errors"].append(
                            f"{block_name} is not a JSON object"
                        )
                        continue
                    for field in fields:
                        value = block_obj.get(field)
                        if value is None or value == "":
                            record["errors"].append(
                                f"{block_name}.{field} missing or empty"
                            )
                seeds_obj = payload.get("seeds", {})
                if not isinstance(seeds_obj, dict):
                    record["errors"].append("seeds is not a JSON object")
                else:
                    if all(seeds_obj.get(key) in (None, "") for key in ("global_seed", "seed")):
                        record["errors"].append(
                            "seeds requires at least one of: global_seed, seed"
                        )

            completeness_ok = (
                len(missing_top) == 0
                and len(missing_runtime) == 0
                and len(record["errors"]) == 0
            )
            record["complete"] = completeness_ok
            if not completeness_ok:
                invalid_count += 1
        except Exception as exc:
            invalid_count += 1
            record["errors"].append(str(exc))
        records.append(record)

    complete_count = sum(1 for r in records if r.get("complete", False))
    completeness_ratio = (
        (complete_count / len(records)) * 100.0 if records else 0.0
    )

    summary = {
        "timestamp": time.time(),
        "scan": {
            "roots": [str(p) for p in roots],
            "manifest_count": len(manifest_paths),
            "required_top": required_top,
            "required_runtime": required_runtime,
            "schema_version": args.schema_version,
        },
        "result": {
            "invalid_count": invalid_count,
            "complete_count": complete_count,
            "completeness_ratio": round(completeness_ratio, 2),
            "legacy_count": legacy_count,
            "legacy_skipped": legacy_skipped,
            "top_missing_total": top_missing_total,
            "runtime_missing_total": runtime_missing_total,
            "pass": invalid_count == 0,
        },
        "records": records,
        "duration_sec": round(time.time() - t0, 2),
    }

    if args.output is None:
        out_dir = PROJECT_ROOT / "artifacts" / "program_plan"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        args.output = out_dir / f"run_manifest_contract_{stamp}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Run-manifest contract report saved: {args.output}")
    print(
        f"  scanned={len(manifest_paths)}, invalid={invalid_count}, "
        f"complete={complete_count}, completeness={completeness_ratio:.1f}%"
    )

    if args.strict and invalid_count > 0:
        print("[ERROR] Strict mode failed due to manifest contract violations.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
