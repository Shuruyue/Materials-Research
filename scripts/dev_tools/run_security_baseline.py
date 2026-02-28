#!/usr/bin/env python3
"""
Run a lightweight security/compliance baseline scan for the repository.

Supports Track B / B-08.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


SECRET_RULES: list[dict[str, Any]] = [
    {"name": "private_key_block", "severity": "high", "regex": re.compile(r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----")},
    {"name": "aws_access_key", "severity": "high", "regex": re.compile(r"\bAKIA[0-9A-Z]{16}\b")},
    {"name": "openai_api_key", "severity": "high", "regex": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")},
    {"name": "hf_token", "severity": "high", "regex": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")},
    {"name": "generic_password_assign", "severity": "medium", "regex": re.compile(r"(?i)\b(password|passwd|pwd)\s*[:=]\s*[\"'][^\"']{4,}[\"']")},
    {"name": "generic_api_key_assign", "severity": "medium", "regex": re.compile(r"(?i)\b(api[_-]?key|token)\s*[:=]\s*[\"'][^\"']{12,}[\"']")},
]

TEXT_EXT_ALLOW = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".env",
    ".ps1",
    ".sh",
    ".bat",
    ".csv",
}

SKIP_PREFIXES = (
    "atlas/third_party/",
    "models/",
    "data/",
    "artifacts/",
    ".git/",
)


def _run_git(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip()


def _tracked_files() -> list[Path]:
    rc, out = _run_git(["ls-files"])
    if rc != 0:
        return []
    paths: list[Path] = []
    for line in out.splitlines():
        rel = line.strip().replace("\\", "/")
        if not rel:
            continue
        if rel.startswith(SKIP_PREFIXES):
            continue
        p = PROJECT_ROOT / rel
        if not p.exists() or not p.is_file():
            continue
        if p.suffix.lower() not in TEXT_EXT_ALLOW:
            continue
        paths.append(p)
    return paths


def _scan_file(path: Path, max_bytes: int) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return findings

    if len(text.encode("utf-8", errors="ignore")) > max_bytes:
        text = text[:max_bytes]

    lines = text.splitlines()
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    for idx, line in enumerate(lines, start=1):
        normalized = line.strip().lower()
        # Avoid obvious placeholders/examples.
        if "example" in normalized or "placeholder" in normalized:
            continue
        for rule in SECRET_RULES:
            if rule["regex"].search(line):
                findings.append(
                    {
                        "file": rel,
                        "line": idx,
                        "rule": rule["name"],
                        "severity": rule["severity"],
                        "snippet": line[:220],
                    }
                )
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run security/compliance baseline scan.")
    parser.add_argument("--max-bytes-per-file", type=int, default=2_000_000)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    tracked = _tracked_files()
    findings: list[dict[str, Any]] = []
    for path in tracked:
        findings.extend(_scan_file(path, max_bytes=args.max_bytes_per_file))

    high_count = sum(1 for f in findings if f["severity"] == "high")
    medium_count = sum(1 for f in findings if f["severity"] == "medium")

    rc_env, tracked_env = _run_git(["ls-files", ".env", ".env.*"])
    tracked_env_files = [x for x in tracked_env.splitlines() if x.strip()] if rc_env == 0 else []
    risky_tracked_env_files = [
        x
        for x in tracked_env_files
        if Path(x).name not in {".env.example", ".env.template", ".env.sample"}
    ]

    license_exists = (PROJECT_ROOT / "LICENSE").exists()

    payload = {
        "timestamp": time.time(),
        "scan": {
            "tracked_file_count": len(tracked),
            "max_bytes_per_file": args.max_bytes_per_file,
        },
        "checks": {
            "license_exists": license_exists,
            "tracked_env_files": tracked_env_files,
            "risky_tracked_env_files": risky_tracked_env_files,
        },
        "findings": findings,
        "summary": {
            "high_count": high_count,
            "medium_count": medium_count,
            "pass": high_count == 0 and len(risky_tracked_env_files) == 0 and license_exists,
        },
        "duration_sec": round(time.time() - t0, 2),
    }

    if args.output is None:
        out_dir = PROJECT_ROOT / "artifacts" / "program_plan"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        args.output = out_dir / f"security_baseline_{stamp}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] Security baseline report saved: {args.output}")
    print(
        f"  high={high_count}, medium={medium_count}, "
        f"risky_tracked_env={len(risky_tracked_env_files)}, license_exists={license_exists}"
    )
    print(f"  pass={payload['summary']['pass']}")

    if args.strict and not payload["summary"]["pass"]:
        print("[ERROR] Strict mode failed due to security baseline findings.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
