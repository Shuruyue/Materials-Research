#!/usr/bin/env python3
"""
Read research tracker entries in fixed-size batches (default: 5).

This script does NOT rewrite tracker markdown files.
It keeps batch progress in a separate state JSON and emits a markdown report
for each batch so reading/integration can proceed incrementally.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _sort_key(entry: dict[str, Any]) -> tuple[str, str]:
    return str(entry.get("id", "")), str(entry.get("title", ""))


def _module_action(tag: str) -> str:
    actions = {
        "phase1": "Compare against CGCNN/ALIGNN baseline and extract architecture deltas.",
        "phase2": "Extract multi-task training/loss balancing tactics and map to phase2 profiles.",
        "phase3": "Map equivariant/MLIP design choices to MACE/NequIP implementation gaps.",
        "phase4": "Extract topology classification features and evaluation metrics.",
        "phase5": "Map active learning/acquisition strategy to discovery controller policy.",
        "phase6": "Map analysis/reporting methods to stable reporting outputs.",
        "phase8": "Map integration/deployment patterns to end-to-end pipeline hardening.",
        "uq": "Extract uncertainty calibration/diagnostic methods and logging metrics.",
        "benchmark": "Extract benchmark protocol to normalize cross-model comparison.",
        "data": "Extract data pipeline constraints and dataset compatibility notes.",
        "xai": "Extract explainability workflow and integration hooks.",
        "ops": "Extract reproducibility/workflow automation practices.",
    }
    return actions.get(tag, "No specific action mapping yet.")


def _entry_actions(entry: dict[str, Any]) -> list[str]:
    tags = entry.get("module_tags", [])
    if not isinstance(tags, list):
        return ["No module tags inferred; manual mapping required."]
    if not tags:
        return ["No module tags inferred; manual mapping required."]
    return [_module_action(tag) for tag in tags]


def _entry_summary(entry: dict[str, Any]) -> str:
    for key in ("atlas_relevance", "core_contribution", "description", "publication"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "No structured summary available in tracker fields."


def _select_entries(
    entries: list[dict[str, Any]],
    *,
    done_ids: set[str],
    batch_size: int,
) -> list[dict[str, Any]]:
    pending = [
        e for e in entries
        if str(e.get("id", "")) not in done_ids
        and str(e.get("reading_status_normalized", "")) != "complete"
    ]
    pending.sort(key=_sort_key)
    return pending[:batch_size]


def _build_batch_markdown(
    *,
    source: str,
    batch_number: int,
    picked: list[dict[str, Any]],
) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append(f"# Batch {batch_number:03d} - {source.capitalize()} Reading")
    lines.append("")
    lines.append(f"- Generated at: {now}")
    lines.append(f"- Source: `{source}`")
    lines.append(f"- Batch size: {len(picked)}")
    lines.append("")

    for idx, item in enumerate(picked, start=1):
        lines.append(f"## {idx}. {item.get('id', '')} | {item.get('title', '')}")
        lines.append("")
        lines.append(f"- Grade: `{item.get('grade', 'UNKNOWN')}`")
        lines.append(f"- Current status: `{item.get('reading_status_normalized', 'unknown')}`")
        tags = item.get("module_tags", [])
        tags_text = ", ".join(tags) if isinstance(tags, list) and tags else "(none)"
        lines.append(f"- Module tags: {tags_text}")
        lines.append(f"- Tracker summary: { _entry_summary(item) }")
        lines.append("- Integration actions:")
        for action in _entry_actions(item):
            lines.append(f"  - {action}")
        lines.append("")

    lines.append("## Completion Notes")
    lines.append("")
    lines.append("- Mark this batch as reviewed after manual validation.")
    lines.append("- Then run the script again to fetch the next batch.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read research entries in batches")
    parser.add_argument(
        "--index-json",
        type=Path,
        default=Path("artifacts/research_index/latest/research_index.json"),
    )
    parser.add_argument("--source", choices=("repo", "paper"), default="repo")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--state-json",
        type=Path,
        default=Path("artifacts/research_index/latest/batch_state.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/research_preparation/batch_reading"),
    )
    parser.add_argument(
        "--mark-reviewed",
        action="store_true",
        help="Mark picked ids as reviewed in state file after generating the batch report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.index_json.exists():
        raise FileNotFoundError(
            f"Missing index file: {args.index_json}. Run build_research_index.py first."
        )

    index_data = _load_json(args.index_json)
    key = f"{args.source}_entries"
    entries = index_data.get(key, [])
    if not isinstance(entries, list):
        raise ValueError(f"Invalid index structure: missing list `{key}`")

    state = _load_json(args.state_json)
    done_key = f"{args.source}_reviewed_ids"
    done_ids = set(state.get(done_key, []))
    batch_no_key = f"{args.source}_batch_number"
    batch_number = int(state.get(batch_no_key, 0)) + 1

    picked = _select_entries(entries, done_ids=done_ids, batch_size=args.batch_size)
    if not picked:
        print(f"[OK] No pending {args.source} entries left.")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"batch_{args.source}_{batch_number:03d}.md"
    out_path.write_text(
        _build_batch_markdown(source=args.source, batch_number=batch_number, picked=picked),
        encoding="utf-8",
    )

    if args.mark_reviewed:
        new_done = sorted(done_ids.union({str(x.get('id', '')) for x in picked}))
        state[done_key] = new_done
        state[batch_no_key] = batch_number
        _save_json(args.state_json, state)
        print(f"[OK] Updated state: {args.state_json}")

    print(f"[OK] Batch report: {out_path}")
    print("[PICKED] " + ", ".join(str(x.get("id", "")) for x in picked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

