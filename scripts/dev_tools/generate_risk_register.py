#!/usr/bin/env python3
"""
Generate an actionable risk register from current evidence artifacts.

Supports Track B / B-07.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _risk(
    *,
    risk_id: str,
    title: str,
    category: str,
    severity: str,
    likelihood: str,
    impact: str,
    evidence: str,
    mitigation: str,
) -> dict[str, Any]:
    return {
        "risk_id": risk_id,
        "title": title,
        "category": category,
        "severity": severity,
        "likelihood": likelihood,
        "impact": impact,
        "status": "open",
        "owner": "unassigned",
        "evidence": evidence,
        "mitigation": mitigation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate risk register from Batch evidence.")
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "program_plan" / "dataset_manifest_audit3.json",
    )
    parser.add_argument(
        "--split-stability",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "program_plan" / "split_stability_audit3.json",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "program_plan" / "batch04_baseline_runs_audit4.json",
    )
    parser.add_argument(
        "--security-report",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "program_plan" / "security_baseline_audit4.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    risks: list[dict[str, Any]] = []

    dataset_manifest = _load_json(args.dataset_manifest)
    split_stability = _load_json(args.split_stability)
    baseline_report = _load_json(args.baseline_report)
    security_report = _load_json(args.security_report)

    # Data missingness risk from dataset manifest.
    if dataset_manifest and isinstance(dataset_manifest.get("splits"), dict):
        train = dataset_manifest["splits"].get("train", {})
        property_stats = train.get("property_stats", {})
        high_missing_props = []
        for prop, stat in property_stats.items():
            total = float(stat.get("count_total", 0) or 0)
            nan_count = float(stat.get("count_nan", 0) or 0)
            if total <= 0:
                continue
            nan_ratio = nan_count / total
            if nan_ratio >= 0.5:
                high_missing_props.append(f"{prop}:{nan_ratio:.2f}")
        if high_missing_props:
            risks.append(
                _risk(
                    risk_id="R-DATA-001",
                    title="High missingness in key target properties",
                    category="data",
                    severity="high",
                    likelihood="high",
                    impact="model bias / unstable benchmark claims",
                    evidence=f"{args.dataset_manifest} -> train property_stats ({', '.join(high_missing_props)})",
                    mitigation="Implement targeted data curation, missing-value strategy, and per-property eligibility protocol before publication lock.",
                )
            )

    # Split stability risk.
    if split_stability:
        status = split_stability.get("status", {})
        if not bool(status.get("overall_pass", False)):
            risks.append(
                _risk(
                    risk_id="R-REPRO-001",
                    title="Split stability protocol does not pass",
                    category="reproducibility",
                    severity="high",
                    likelihood="medium",
                    impact="results not reproducible across seeds",
                    evidence=f"{args.split_stability}",
                    mitigation="Block progression to paper figures until leakage/count_cv/property_drift checks pass in strict mode.",
                )
            )

    # Baseline execution risk.
    if baseline_report:
        runs = baseline_report.get("runs", [])
        failed = [r for r in runs if int(r.get("return_code", 1)) != 0]
        if failed:
            risks.append(
                _risk(
                    risk_id="R-ML-001",
                    title="Baseline suite has failed runs",
                    category="ml_pipeline",
                    severity="high",
                    likelihood="medium",
                    impact="incomplete evidence for benchmark comparison",
                    evidence=f"{args.baseline_report} -> failed_runs={len(failed)}",
                    mitigation="Resolve failed algorithms/dependencies and rerun baseline suite until all required tracks pass.",
                )
            )
        elif runs:
            risks.append(
                _risk(
                    risk_id="R-ML-002",
                    title="Baseline suite executed but may be underpowered",
                    category="ml_pipeline",
                    severity="medium",
                    likelihood="high",
                    impact="insufficient confidence for SOTA claims",
                    evidence=f"{args.baseline_report} (fast/smoke-like settings)",
                    mitigation="Promote top candidates to std/pro settings and require repeated-seed statistical confirmation.",
                )
            )

    # Security baseline risk.
    if security_report:
        summary = security_report.get("summary", {})
        if not bool(summary.get("pass", False)):
            risks.append(
                _risk(
                    risk_id="R-SEC-001",
                    title="Security baseline check failed",
                    category="security",
                    severity="high",
                    likelihood="medium",
                    impact="potential credential leakage or compliance issue",
                    evidence=f"{args.security_report}",
                    mitigation="Fix high-severity secret findings and remove tracked sensitive files before release.",
                )
            )

    # Governance and people risks should always exist.
    risks.append(
        _risk(
            risk_id="R-GOV-001",
            title="Stage owners are not assigned",
            category="governance",
            severity="medium",
            likelihood="high",
            impact="execution delay and accountability gaps",
            evidence="execution board owner fields are currently unassigned",
            mitigation="Assign owner/deputy per stage and enforce weekly review cadence.",
        )
    )
    risks.append(
        _risk(
            risk_id="R-LEGAL-001",
            title="External dataset/license obligations not yet checklisted",
            category="legal_compliance",
            severity="medium",
            likelihood="medium",
            impact="publication/open-source release risk",
            evidence="Need explicit license matrix for JARVIS/MP/OQMD usage pathways",
            mitigation="Create a data license matrix and approval checklist before public release.",
        )
    )

    severity_order = {"high": 0, "medium": 1, "low": 2}
    risks.sort(key=lambda r: (severity_order.get(r["severity"], 9), r["risk_id"]))

    payload = {
        "timestamp": time.time(),
        "inputs": {
            "dataset_manifest": str(args.dataset_manifest),
            "split_stability": str(args.split_stability),
            "baseline_report": str(args.baseline_report),
            "security_report": str(args.security_report),
        },
        "summary": {
            "risk_count": len(risks),
            "high_count": sum(1 for r in risks if r["severity"] == "high"),
            "medium_count": sum(1 for r in risks if r["severity"] == "medium"),
        },
        "risks": risks,
    }

    if args.output is None:
        out_dir = PROJECT_ROOT / "artifacts" / "program_plan"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        args.output = out_dir / f"risk_register_{stamp}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] Risk register saved: {args.output}")
    print(
        f"  risks={payload['summary']['risk_count']}, "
        f"high={payload['summary']['high_count']}, "
        f"medium={payload['summary']['medium_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
