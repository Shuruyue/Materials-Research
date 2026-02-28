#!/usr/bin/env python3
"""
Build a machine-readable research index from repo/paper tracker markdown files.

Outputs:
- JSON index for downstream tooling
- Markdown status report for human review
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

REPO_HEADING_RE = re.compile(r"^###\s+([A-Z]-\d{2})\s+\|\s+(.+?)(?:\s+\|\s+.+)?$")
PAPER_HEADING_RE = re.compile(r"^###\s+((?:\d+|S)-\d{2})\s+\|\s+(.+)$")
CATEGORY_RE = re.compile(r"^##\s+(.+)$")
FIELD_RE = re.compile(r"^- \*\*(.+?)\*\*[:：]\s*(.*)$")


FIELD_ALIASES = {
    "affiliation": "affiliation",
    "機構": "affiliation",
    "机构": "affiliation",
    "grade": "grade",
    "等級": "grade",
    "等级": "grade",
    "publication": "publication",
    "論文": "publication",
    "论文": "publication",
    "description": "description",
    "描述": "description",
    "atlas relevance": "atlas_relevance",
    "atals relevance": "atlas_relevance",
    "與 atlas 關聯": "atlas_relevance",
    "与 atlas 关联": "atlas_relevance",
    "key learnings": "key_learnings",
    "核心學習點": "key_learnings",
    "核心学习点": "key_learnings",
    "core contribution": "core_contribution",
    "核心貢獻": "core_contribution",
    "核心贡献": "core_contribution",
    "reading status": "reading_status",
    "閱讀狀態": "reading_status",
    "阅读状态": "reading_status",
    "notes": "notes",
    "筆記": "notes",
    "笔记": "notes",
    "year": "year",
    "年份": "year",
    "venue": "venue",
    "會議": "venue",
    "会议": "venue",
    "citations": "citations",
    "引用": "citations",
}


STATUS_MAP = {
    "not started": "not_started",
    "queued": "queued",
    "scanned": "scanned",
    "in progress": "in_progress",
    "complete": "complete",
    "completed": "complete",
    "done": "complete",
    "未開始": "not_started",
    "未开始": "not_started",
    "排隊中": "queued",
    "排队中": "queued",
    "已排程": "queued",
    "已排程待讀": "queued",
    "已扫描": "scanned",
    "已掃描": "scanned",
    "進行中": "in_progress",
    "进行中": "in_progress",
    "完成": "complete",
    "已完成": "complete",
}


def _normalize_field_name(raw: str) -> str:
    key = raw.strip().lower()
    return FIELD_ALIASES.get(key, FIELD_ALIASES.get(raw.strip(), raw.strip()))


def _normalize_status(raw: str | None) -> str:
    if not raw:
        return "unknown"
    s = raw.strip()
    if s in STATUS_MAP:
        return STATUS_MAP[s]
    s_lower = s.lower()
    return STATUS_MAP.get(s_lower, "unknown")


def _infer_module_tags(text: str | None) -> list[str]:
    if not text:
        return []
    t = text.lower()
    tags: list[str] = []
    rules = [
        ("phase1", ["cgcnn", "phase 1", "phase1"]),
        ("phase2", ["multitask", "phase 2", "phase2", "e3nn"]),
        ("phase3", ["mace", "equivariant", "phase 3", "phase3", "interatomic potential", "mlip"]),
        ("phase4", ["topology", "topognn", "phase 4", "phase4"]),
        ("phase5", ["active learning", "bayesian optimization", "bo", "gp", "phase 5", "phase5"]),
        ("phase6", ["analysis", "alloy", "phase 6", "phase6"]),
        ("phase8", ["integration", "phase 8", "phase8", "pipeline"]),
        ("uq", ["uncertainty", "uq", "calibration", "evidential", "ood"]),
        ("benchmark", ["matbench", "leaderboard", "benchmark"]),
        ("data", ["jarvis", "materials project", "pymatgen", "matminer", "dataset"]),
        ("xai", ["explain", "xai", "interpret"]),
        ("ops", ["workflow", "mlops", "dvc", "tracking"]),
    ]
    for tag, keys in rules:
        if any(k in t for k in keys):
            tags.append(tag)
    return sorted(set(tags))


def _finalize_entry(entry: dict[str, Any], tracker_type: str) -> dict[str, Any]:
    data = dict(entry)
    data["reading_status_normalized"] = _normalize_status(data.get("reading_status"))
    data["grade"] = (data.get("grade") or "").strip().upper() or "UNKNOWN"
    relevance = data.get("atlas_relevance")
    data["module_tags"] = _infer_module_tags(relevance)
    data["tracker_type"] = tracker_type
    return data


def _parse_tracker(
    path: Path,
    *,
    heading_re: re.Pattern[str],
    tracker_type: str,
) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_category = ""

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line:
            continue

        cat_match = CATEGORY_RE.match(line)
        if cat_match:
            current_category = cat_match.group(1).strip()
            continue

        head_match = heading_re.match(line)
        if head_match:
            if current is not None:
                entries.append(_finalize_entry(current, tracker_type))
            current = {
                "id": head_match.group(1).strip(),
                "title": head_match.group(2).strip(),
                "category": current_category,
            }
            continue

        if current is None:
            continue

        year_venue = re.match(r"^- \*\*Year\*\*[:：]\s*(.*?)\s+\|\s+\*\*Venue\*\*[:：]\s*(.*)$", line)
        if year_venue:
            current["year"] = year_venue.group(1).strip()
            current["venue"] = year_venue.group(2).strip()
            continue

        year_venue_zh = re.match(r"^- \*\*年份\*\*[:：]\s*(.*?)\s+\|\s+\*\*會議\*\*[:：]\s*(.*)$", line)
        if year_venue_zh:
            current["year"] = year_venue_zh.group(1).strip()
            current["venue"] = year_venue_zh.group(2).strip()
            continue

        grade_cite = re.match(r"^- \*\*Grade\*\*[:：]\s*(.*?)\s+\|\s+\*\*Citations\*\*[:：]\s*(.*)$", line)
        if grade_cite:
            current["grade"] = grade_cite.group(1).strip()
            current["citations"] = grade_cite.group(2).strip()
            continue

        grade_cite_zh = re.match(r"^- \*\*等級\*\*[:：]\s*(.*?)\s+\|\s+\*\*引用\*\*[:：]\s*(.*)$", line)
        if grade_cite_zh:
            current["grade"] = grade_cite_zh.group(1).strip()
            current["citations"] = grade_cite_zh.group(2).strip()
            continue

        field = FIELD_RE.match(line)
        if field:
            key = _normalize_field_name(field.group(1))
            value = field.group(2).strip()
            if key:
                current[key] = value

    if current is not None:
        entries.append(_finalize_entry(current, tracker_type))

    unique: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for item in entries:
        entry_id = item["id"]
        if entry_id in unique:
            duplicates += 1
            continue
        unique[entry_id] = item

    return {
        "entries": list(unique.values()),
        "raw_entries": len(entries),
        "duplicates_dropped": duplicates,
    }


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        val = str(item.get(key, "unknown"))
        out[val] = out.get(val, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _top_priority(items: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    ranked = [
        x for x in items
        if x.get("grade") == "A"
        and x.get("reading_status_normalized") in {"not_started", "queued", "scanned", "in_progress", "unknown"}
    ]
    status_rank = {
        "not_started": 0,
        "queued": 1,
        "scanned": 2,
        "in_progress": 3,
        "unknown": 4,
        "complete": 5,
    }
    ranked.sort(
        key=lambda x: (
            status_rank.get(x.get("reading_status_normalized", "unknown"), 9),
            x.get("id", ""),
        )
    )
    return ranked[:top_n]


def _to_markdown(index_data: dict[str, Any], top_n: int) -> str:
    repo_entries = index_data["repo_entries"]
    paper_entries = index_data["paper_entries"]
    summary = index_data["summary"]

    top_repo = _top_priority(repo_entries, top_n)
    top_paper = _top_priority(paper_entries, top_n)

    lines: list[str] = []
    lines.append("# Research Integration Status")
    lines.append("")
    lines.append(f"- Generated at: {index_data['generated_at']}")
    lines.append(f"- Repo tracker: `{index_data['sources']['repo_tracker']}`")
    lines.append(f"- Paper tracker: `{index_data['sources']['paper_tracker']}`")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- Repos indexed: {summary['repo_total']}")
    lines.append(f"- Papers indexed: {summary['paper_total']}")
    lines.append(f"- Repo duplicates dropped: {summary['repo_duplicates_dropped']}")
    lines.append(f"- Paper duplicates dropped: {summary['paper_duplicates_dropped']}")
    lines.append("")

    lines.append("### Repo Status")
    lines.append("")
    for k, v in summary["repo_status_counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines.append("### Paper Status")
    lines.append("")
    for k, v in summary["paper_status_counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines.append("## Top Priority Repos (A-grade, not complete)")
    lines.append("")
    lines.append("| ID | Repo | Status | Module Tags |")
    lines.append("|---|---|---|---|")
    for item in top_repo:
        tags = ", ".join(item.get("module_tags", []))
        lines.append(f"| {item.get('id','')} | {item.get('title','')} | {item.get('reading_status_normalized','')} | {tags} |")
    if not top_repo:
        lines.append("| - | - | - | - |")
    lines.append("")

    lines.append("## Top Priority Papers (A-grade, not complete)")
    lines.append("")
    lines.append("| ID | Paper | Status | Module Tags |")
    lines.append("|---|---|---|---|")
    for item in top_paper:
        tags = ", ".join(item.get("module_tags", []))
        lines.append(f"| {item.get('id','')} | {item.get('title','')} | {item.get('reading_status_normalized','')} | {tags} |")
    if not top_paper:
        lines.append("| - | - | - | - |")
    lines.append("")

    lines.append("## Next Step")
    lines.append("")
    lines.append("Use this index as input to map evidence -> module-level optimization tasks.")
    lines.append("")
    return "\n".join(lines)


def build_index(repo_tracker: Path, paper_tracker: Path, top_n: int) -> dict[str, Any]:
    repo_data = _parse_tracker(repo_tracker, heading_re=REPO_HEADING_RE, tracker_type="repo")
    paper_data = _parse_tracker(paper_tracker, heading_re=PAPER_HEADING_RE, tracker_type="paper")

    repo_entries = repo_data["entries"]
    paper_entries = paper_data["entries"]

    summary = {
        "repo_total": len(repo_entries),
        "paper_total": len(paper_entries),
        "repo_duplicates_dropped": repo_data["duplicates_dropped"],
        "paper_duplicates_dropped": paper_data["duplicates_dropped"],
        "repo_status_counts": _count_by(repo_entries, "reading_status_normalized"),
        "paper_status_counts": _count_by(paper_entries, "reading_status_normalized"),
        "repo_grade_counts": _count_by(repo_entries, "grade"),
        "paper_grade_counts": _count_by(paper_entries, "grade"),
    }

    module_counts: dict[str, int] = {}
    for item in repo_entries + paper_entries:
        for tag in item.get("module_tags", []):
            module_counts[tag] = module_counts.get(tag, 0) + 1
    summary["module_tag_counts"] = dict(sorted(module_counts.items(), key=lambda kv: (-kv[1], kv[0])))

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sources": {
            "repo_tracker": str(repo_tracker),
            "paper_tracker": str(paper_tracker),
        },
        "repo_entries": repo_entries,
        "paper_entries": paper_entries,
        "summary": summary,
        "top_n": top_n,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build machine-readable index from research trackers")
    parser.add_argument(
        "--repo-tracker",
        type=Path,
        default=Path("docs/research_preparation/repo_tracker.md"),
    )
    parser.add_argument(
        "--paper-tracker",
        type=Path,
        default=Path("docs/research_preparation/paper_tracker.md"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/research_index/latest/research_index.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/research_preparation/research_integration_status.md"),
    )
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for p in (args.repo_tracker, args.paper_tracker):
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    index_data = build_index(args.repo_tracker, args.paper_tracker, args.top_n)
    markdown = _to_markdown(index_data, args.top_n)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown, encoding="utf-8")

    print(f"[OK] JSON index: {args.out_json}")
    print(f"[OK] Markdown report: {args.out_md}")
    print(
        "[SUMMARY] "
        f"repos={index_data['summary']['repo_total']}, "
        f"papers={index_data['summary']['paper_total']}, "
        f"repo_not_started={index_data['summary']['repo_status_counts'].get('not_started', 0)}, "
        f"paper_not_started={index_data['summary']['paper_status_counts'].get('not_started', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
