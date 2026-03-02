"""
Data Validation Pipeline for ATLAS.

Provides schema validation, unit checks, duplicate detection, leakage detection,
outlier detection, drift analysis, and trust scoring.

CLI entrypoint: ``validate-data``

Builds on logic from existing dev tools:
- scripts/dev_tools/validate_phase_datasets.py  (schema/graph checks)
- scripts/dev_tools/check_split_leakage.py      (jid/hash/formula overlap)
- scripts/dev_tools/generate_dataset_manifest.py (dataset fingerprints)
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import json
import logging
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provenance constants
# ---------------------------------------------------------------------------

VALID_PROVENANCE_TYPES = frozenset(
    {"dft_primary", "experimental", "db_import", "literature", "synthetic"}
)

PROVENANCE_TRUST_POINTS: dict[str, int] = {
    "dft_primary": 30,
    "experimental": 25,
    "db_import": 20,
    "literature": 12,
    "synthetic": 8,
}

PROVENANCE_METADATA_FIELDS = [
    "provenance_type",
    "source_key",
    "source_version",
    "source_id",
]

DEFAULT_UNITS_SPEC: dict[str, dict[str, Any]] = {
    # Values are documented units for common ATLAS properties.
    # Unit validation is opportunistic unless a record has explicit unit fields.
    "formation_energy": {"expected": ["eV/atom", "ev/atom"]},
    "band_gap": {"expected": ["eV", "ev"]},
    "band_gap_mbj": {"expected": ["eV", "ev"]},
    "bulk_modulus": {"expected": ["GPa", "gpa"]},
    "shear_modulus": {"expected": ["GPa", "gpa"]},
    "dielectric": {"expected": ["dimensionless", "1"]},
    "piezoelectric": {"expected": ["C/m^2", "c/m^2"]},
    "spillage": {"expected": ["dimensionless", "1"]},
    "ehull": {"expected": ["eV/atom", "ev/atom"]},
}

DEFAULT_JARVIS_PROVENANCE = {
    "provenance_type": "dft_primary",
    "source_key": "jarvis_dft",
    "source_version": "figshare_40357663",
}

# ---------------------------------------------------------------------------
# Trust scoring
# ---------------------------------------------------------------------------


@dataclass
class TrustScore:
    """Computed trust score for a single sample."""

    sample_id: str
    total: float = 0.0
    provenance_points: float = 0.0
    completeness_points: float = 0.0
    schema_points: float = 0.0
    plausibility_points: float = 0.0
    tier: str = "raw"  # raw | curated | benchmark

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "total": round(self.total, 2),
            "provenance_points": self.provenance_points,
            "completeness_points": self.completeness_points,
            "schema_points": self.schema_points,
            "plausibility_points": round(self.plausibility_points, 2),
            "tier": self.tier,
        }


def compute_trust_score(
    sample: dict[str, Any],
    *,
    property_stats: dict[str, dict[str, float]] | None = None,
    properties: list[str] | None = None,
) -> TrustScore:
    """Compute trust score (0-100) for a single sample.

    Parameters
    ----------
    sample : dict
        Sample record with at least ``provenance_type`` and metadata fields.
    property_stats : dict, optional
        Per-property ``{"mean": ..., "std": ...}`` for plausibility scoring.
    properties : list[str], optional
        Property names to check for plausibility.
    """
    sid = str(sample.get("jid", sample.get("source_id", "unknown")))
    ts = TrustScore(sample_id=sid)

    # 1) Provenance type (max 30)
    prov = sample.get("provenance_type", "")
    ts.provenance_points = PROVENANCE_TRUST_POINTS.get(prov, 0)

    # 2) Source completeness (max 20)
    present = sum(1 for f in PROVENANCE_METADATA_FIELDS if sample.get(f))
    ts.completeness_points = min(20, (present / len(PROVENANCE_METADATA_FIELDS)) * 20)

    # 3) Schema validity (max 20) – caller deducts for violations
    ts.schema_points = 20.0

    # 4) Plausibility (max 15)
    if property_stats and properties:
        n_props = len(properties)
        prop_score = 0.0
        for prop in properties:
            stats = property_stats.get(prop)
            val = sample.get(prop)
            if not stats or val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue

            # Prefer robust location/scale when available.
            # Reference: Rousseeuw & Croux (1993), robust scale estimation.
            median = stats.get("median")
            mad = stats.get("mad")
            if (
                median is not None
                and mad is not None
                and float(mad) > 1e-12
                and math.isfinite(float(median))
            ):
                try:
                    robust_sigma = 1.4826 * float(mad)
                    z = abs(fval - float(median)) / robust_sigma
                except (TypeError, ValueError):
                    z = 999
            elif stats.get("std", 0) and float(stats["std"]) > 1e-12:
                try:
                    z = abs(fval - float(stats["mean"])) / float(stats["std"])
                except (TypeError, ValueError):
                    z = 999
            else:
                continue

            if z <= 3:
                prop_score += 15.0 / n_props
            elif z <= 5:
                prop_score += 8.0 / n_props
            # beyond 5σ -> 0
        ts.plausibility_points = min(15.0, prop_score)
    else:
        ts.plausibility_points = 8.0  # neutral when no stats available

    # 5) Cross-validation (max 15) — heuristic baseline (single-source = 8)
    cross_val = 8.0
    ts.total = (
        ts.provenance_points
        + ts.completeness_points
        + ts.schema_points
        + ts.plausibility_points
        + cross_val
    )
    ts.total = min(100.0, max(0.0, ts.total))

    if ts.total >= 70:
        ts.tier = "benchmark"
    elif ts.total >= 40:
        ts.tier = "curated"
    else:
        ts.tier = "raw"

    return ts


# Explicit type aliases for governance/reporting layers.
TrustScoreBreakdown = TrustScore
ProvenanceRecord = dict[str, Any]


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Aggregated validation report."""

    schema_version: str = "2.0"
    timestamp: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )
    schema_violations: int = 0
    provenance_missing: int = 0
    leakage_count: int = 0
    unit_violations: int = 0
    duplicate_count: int = 0
    outlier_count: int = 0
    drift_summary: dict[str, float] = field(default_factory=dict)
    drift_alert_count: int = 0
    gate_pass: bool = True
    gate_failures: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    n_samples: int = 0
    trust_tier_counts: dict[str, int] = field(default_factory=dict)
    duration_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "n_samples": self.n_samples,
            "schema_violations": self.schema_violations,
            "provenance_missing": self.provenance_missing,
            "leakage_count": self.leakage_count,
            "unit_violations": self.unit_violations,
            "duplicate_count": self.duplicate_count,
            "outlier_count": self.outlier_count,
            "drift_summary": self.drift_summary,
            "drift_alert_count": self.drift_alert_count,
            "gate_pass": self.gate_pass,
            "gate_failures": self.gate_failures,
            "trust_tier_counts": self.trust_tier_counts,
            "duration_sec": round(self.duration_sec, 2),
            "details": self.details,
        }

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    def to_markdown(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Data Validation Report",
            "",
            f"**Schema Version**: {self.schema_version}",
            f"**Timestamp**: {self.timestamp}",
            f"**Samples**: {self.n_samples}",
            f"**Duration**: {self.duration_sec:.1f}s",
            "",
            "## Gate Results",
            "",
            "| Gate | Count | Pass |",
            "|---|---|---|",
            f"| Schema violations | {self.schema_violations} | {'✅' if self.schema_violations == 0 else '❌'} |",
            f"| Provenance missing | {self.provenance_missing} | {'✅' if self.provenance_missing == 0 else '❌'} |",
            f"| Leakage | {self.leakage_count} | {'✅' if self.leakage_count == 0 else '❌'} |",
            f"| Unit violations | {self.unit_violations} | {'✅' if self.unit_violations == 0 else '❌'} |",
            f"| Duplicates | {self.duplicate_count} | ⚠️ (warning) |",
            f"| Outliers | {self.outlier_count} | ⚠️ (warning) |",
            f"| Drift alerts | {self.drift_alert_count} | {'⚠️' if self.drift_alert_count > 0 else '✅'} |",
            "",
            f"**Overall**: {'✅ PASS' if self.gate_pass else '❌ FAIL'}",
            "",
        ]
        if self.gate_failures:
            lines.append("### Failures")
            for gf in self.gate_failures:
                lines.append(f"- {gf}")
            lines.append("")

        if self.trust_tier_counts:
            lines.append("## Trust Score Distribution")
            lines.append("")
            lines.append("| Tier | Count |")
            lines.append("|---|---|")
            for tier, count in sorted(self.trust_tier_counts.items()):
                lines.append(f"| {tier} | {count} |")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path

    def apply_gates(self, *, strict: bool = False, drift_hard_gate: bool = False) -> None:
        """Apply gate rules and set pass/fail."""
        self.gate_failures = []
        if self.schema_violations > 0:
            self.gate_failures.append(
                f"schema_violations={self.schema_violations} (must be 0)"
            )
        if self.provenance_missing > 0:
            self.gate_failures.append(
                f"provenance_missing={self.provenance_missing} (must be 0)"
            )
        if self.leakage_count > 0:
            self.gate_failures.append(
                f"leakage_count={self.leakage_count} (must be 0)"
            )
        if self.unit_violations > 0:
            self.gate_failures.append(
                f"unit_violations={self.unit_violations} (must be 0)"
            )
        if drift_hard_gate and self.drift_alert_count > 0:
            self.gate_failures.append(
                f"drift_alert_count={self.drift_alert_count} (hard gate: must be 0)"
            )
        if strict:
            if self.duplicate_count > 0:
                self.gate_failures.append(
                    f"duplicate_count={self.duplicate_count} (strict: must be 0)"
                )
            if self.outlier_count > 0:
                self.gate_failures.append(
                    f"outlier_count={self.outlier_count} (strict: must be 0)"
                )
            if self.drift_alert_count > 0:
                self.gate_failures.append(
                    f"drift_alert_count={self.drift_alert_count} (strict: must be 0)"
                )
        self.gate_pass = len(self.gate_failures) == 0


def _stable_hash(obj: Any) -> str:
    """Deterministic hash for a JSON-serializable object."""
    try:
        payload = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    except (TypeError, ValueError):
        payload = str(obj)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _sample_id(rec: dict[str, Any], index: int) -> str:
    value = rec.get("jid", rec.get("source_id", f"idx:{index}"))
    return str(value) if value is not None else f"idx:{index}"


def _infer_jarvis_source_version() -> str:
    try:
        from atlas.data.jarvis_client import JARVIS_DFT_3D_URL
    except Exception:
        return DEFAULT_JARVIS_PROVENANCE["source_version"]

    match = re.search(r"/files/(\d+)", str(JARVIS_DFT_3D_URL))
    if not match:
        return DEFAULT_JARVIS_PROVENANCE["source_version"]
    return f"figshare_{match.group(1)}"


def _attach_known_source_provenance(record: dict[str, Any]) -> bool:
    """
    Attach explicit provenance metadata for known JARVIS source records.

    Returns True if any provenance field was added.
    """
    updated = False
    source_version = _infer_jarvis_source_version()
    if not record.get("provenance_type"):
        record["provenance_type"] = DEFAULT_JARVIS_PROVENANCE["provenance_type"]
        updated = True
    if not record.get("source_key"):
        record["source_key"] = DEFAULT_JARVIS_PROVENANCE["source_key"]
        updated = True
    if not record.get("source_version"):
        record["source_version"] = source_version
        updated = True
    if not record.get("source_id"):
        sid = record.get("jid")
        if sid is not None:
            record["source_id"] = str(sid)
            updated = True
    return updated


def check_schema(
    records: list[dict[str, Any]],
    required_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Check each record has required fields with non-None values.

    Returns list of violation dicts: ``{"index": i, "missing": [...]}``.
    """
    if required_fields is None:
        required_fields = ["jid", "atoms"]
    violations: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        missing = [f for f in required_fields if rec.get(f) is None]
        prov = rec.get("provenance_type")
        if prov is not None and prov not in VALID_PROVENANCE_TYPES:
            missing.append("provenance_type(valid)")
        if missing:
            violations.append(
                {"index": i, "sample_id": _sample_id(rec, i), "missing": missing}
            )
    return violations


def check_provenance(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return records missing or invalid ``provenance_type``."""
    missing: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        prov = rec.get("provenance_type")
        if not prov or prov not in VALID_PROVENANCE_TYPES:
            missing.append(
                {
                    "index": i,
                    "sample_id": _sample_id(rec, i),
                    "provenance_type": prov,
                }
            )
    return missing


def check_duplicates(
    records: list[dict[str, Any]],
    key_field: str = "jid",
) -> dict[str, list[int]]:
    """Return mapping of duplicate key -> list of indices."""
    seen: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        key = str(r.get(key_field, ""))
        if not key:
            continue
        seen.setdefault(key, []).append(i)
    return {k: v for k, v in seen.items() if len(v) > 1}


def check_leakage(
    split_ids: dict[str, set[str]],
) -> dict[str, int]:
    """Check pairwise overlap between split ID sets.

    Parameters
    ----------
    split_ids : dict
        ``{"train": {id1, id2, ...}, "val": {...}, "test": {...}}``

    Returns
    -------
    dict
        ``{"train__val": n, "train__test": n, "val__test": n}``
    """
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    result: dict[str, int] = {}
    for a, b in pairs:
        ids_a = split_ids.get(a, set())
        ids_b = split_ids.get(b, set())
        result[f"{a}__{b}"] = len(ids_a & ids_b)
    return result


def check_outliers(
    values: list[float],
    sigma_threshold: float = 5.0,
) -> list[int]:
    """Return indices of values beyond ``sigma_threshold`` from median.

    Uses median + MAD (median absolute deviation) for robustness.

    Reference:
    Rousseeuw & Croux (1993), alternatives to classical variance scale estimators.
    """
    if not values:
        return []
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if len(clean) < 3:
        return []
    clean_sorted = sorted(clean)
    n = len(clean_sorted)
    median = clean_sorted[n // 2]
    mad = sorted(abs(v - median) for v in clean_sorted)[n // 2]
    if mad == 0:
        mad = 1e-10  # avoid division by zero
    # MAD-based sigma ≈ 1.4826 * MAD
    sigma_est = 1.4826 * mad
    threshold = sigma_threshold * sigma_est
    outliers: list[int] = []
    for i, v in enumerate(values):
        if v is None or not math.isfinite(v):
            continue
        if abs(v - median) > threshold:
            outliers.append(i)
    return outliers


def check_units(
    records: list[dict[str, Any]],
    *,
    properties: list[str] | None = None,
    units_spec: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Validate explicit unit fields against expected units.

    Notes
    -----
    This check is opportunistic: if a record has no explicit unit field for a
    property, no violation is recorded by default.
    """
    if not properties:
        return []
    spec = units_spec or DEFAULT_UNITS_SPEC
    violations: list[dict[str, Any]] = []
    unit_keys = ("unit", "units")
    for idx, rec in enumerate(records):
        sid = _sample_id(rec, idx)
        for prop in properties:
            prop_cfg = spec.get(prop, {})
            expected = [str(x).strip().lower() for x in prop_cfg.get("expected", [])]
            if not expected:
                continue
            found_unit = None
            for key in (f"{prop}_unit", f"{prop}_units", *unit_keys):
                raw = rec.get(key)
                if raw is None:
                    continue
                found_unit = str(raw).strip().lower()
                break
            if found_unit is None:
                continue
            if found_unit not in expected:
                violations.append(
                    {
                        "index": idx,
                        "sample_id": sid,
                        "property": prop,
                        "found_unit": found_unit,
                        "expected": expected,
                    }
                )
    return violations


def summarize_properties(
    records: list[dict[str, Any]],
    properties: list[str] | None,
) -> dict[str, dict[str, float]]:
    """Compute count/mean/std/min/max summary for canonical property names."""
    if not properties:
        return {}
    out: dict[str, dict[str, float]] = {}
    for prop in properties:
        vals: list[float] = []
        for rec in records:
            raw = rec.get(prop)
            if raw is None:
                continue
            with contextlib.suppress(TypeError, ValueError):
                val = float(raw)
                if math.isfinite(val):
                    vals.append(val)
        if not vals:
            out[prop] = {"count": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
            continue
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        out[prop] = {
            "count": float(n),
            "mean": mean,
            "std": math.sqrt(var),
            "min": min(vals),
            "max": max(vals),
        }
    return out


def collect_property_values(
    records: list[dict[str, Any]],
    properties: list[str] | None,
    *,
    allow_source_fallback: bool = True,
) -> dict[str, list[float]]:
    """Collect finite float values for each property.

    When ``allow_source_fallback`` is enabled, canonical property names
    are mapped back to source columns via ``PROPERTY_MAP``.
    """
    if not properties:
        return {}
    rev_map: dict[str, str] = {}
    if allow_source_fallback:
        with contextlib.suppress(Exception):
            from atlas.data.crystal_dataset import PROPERTY_MAP

            rev_map = {v: k for k, v in PROPERTY_MAP.items()}
    out = {prop: [] for prop in properties}
    for rec in records:
        for prop in properties:
            raw = rec.get(prop)
            if raw is None:
                source_col = rev_map.get(prop)
                if source_col:
                    raw = rec.get(source_col)
            if raw is None:
                continue
            with contextlib.suppress(TypeError, ValueError):
                val = float(raw)
                if math.isfinite(val):
                    out[prop].append(val)
    return out


def compute_mean_shift_drift(
    baseline_summary: dict[str, dict[str, float]] | None,
    current_summary: dict[str, dict[str, float]],
) -> dict[str, float]:
    """
    Compute lightweight drift metric from baseline and current property summaries.

    Metric: absolute z-shift of means using baseline std as scale.
    """
    if not baseline_summary:
        return {}
    drift: dict[str, float] = {}
    for prop, cur in current_summary.items():
        base = baseline_summary.get(prop)
        if not base:
            continue
        b_mean = float(base.get("mean", math.nan))
        b_std = float(base.get("std", math.nan))
        c_mean = float(cur.get("mean", math.nan))
        if not (math.isfinite(b_mean) and math.isfinite(c_mean)):
            continue
        denom = abs(b_std) if math.isfinite(b_std) and abs(b_std) > 1e-12 else 1.0
        drift[prop] = abs(c_mean - b_mean) / denom
    return drift


def _to_clean_array(values: list[float]) -> np.ndarray:
    clean: list[float] = []
    for v in values:
        if v is None:
            continue
        with contextlib.suppress(TypeError, ValueError):
            fv = float(v)
            if math.isfinite(fv):
                clean.append(fv)
    return np.asarray(clean, dtype=float)


def _median_heuristic_bandwidth(values: np.ndarray) -> float:
    """Median-heuristic kernel bandwidth for 1D RBF-MMD."""
    if values.size < 2:
        return 1.0
    sample = values
    if sample.size > 256:
        rng = np.random.RandomState(0)
        sample = sample[rng.choice(sample.size, size=256, replace=False)]
    diffs = sample[:, None] - sample[None, :]
    d2 = (diffs * diffs).astype(float)
    tri = d2[np.triu_indices_from(d2, k=1)]
    tri = tri[tri > 0]
    if tri.size == 0:
        return 1.0
    return max(1e-6, float(np.sqrt(np.median(tri))))


def compute_wasserstein_1d(
    values_old: list[float],
    values_new: list[float],
) -> float:
    """Compute empirical 1-Wasserstein distance for one-dimensional samples.

    Reference:
    Ramdas et al. (2015), Wasserstein two-sample testing.
    """
    old = np.sort(_to_clean_array(values_old))
    new = np.sort(_to_clean_array(values_new))
    if old.size < 2 or new.size < 2:
        return 0.0

    grid = np.unique(np.concatenate([old, new]))
    if grid.size < 2:
        return 0.0

    cdf_old = np.searchsorted(old, grid[:-1], side="right") / float(old.size)
    cdf_new = np.searchsorted(new, grid[:-1], side="right") / float(new.size)
    widths = np.diff(grid)
    w1 = float(np.sum(np.abs(cdf_old - cdf_new) * widths))
    return max(0.0, w1)


def compute_ks_2sample(
    values_old: list[float],
    values_new: list[float],
    *,
    method: str = "auto",
    n_permutations: int = 256,
    random_state: int = 0,
) -> tuple[float, float]:
    """Kolmogorov-Smirnov two-sample test.

    Method selection:
    - ``asymptotic``: classical KS asymptotic p-value.
    - ``permutation``: permutation-calibrated p-value.
    - ``auto``: chooses permutation when sample is small or heavily tied.

    References:
    Kolmogorov (1933), Smirnov (1948), and permutation testing practice.
    """

    def _ks_statistic(sorted_old: np.ndarray, sorted_new: np.ndarray) -> float:
        grid = np.unique(np.concatenate([sorted_old, sorted_new]))
        if grid.size == 0:
            return 0.0
        cdf_old = np.searchsorted(sorted_old, grid, side="right") / float(
            sorted_old.size
        )
        cdf_new = np.searchsorted(sorted_new, grid, side="right") / float(
            sorted_new.size
        )
        return float(np.max(np.abs(cdf_old - cdf_new)))

    def _asymptotic_pvalue(stat: float, n: int, m: int) -> float:
        en = math.sqrt((float(n) * float(m)) / float(n + m))
        if en <= 0:
            return 1.0
        lam = (en + 0.12 + 0.11 / en) * stat
        if lam <= 1e-12:
            return 1.0
        series = 0.0
        for j in range(1, 101):
            term = ((-1) ** (j - 1)) * math.exp(-2.0 * (lam * lam) * (j * j))
            series += term
            if abs(term) < 1e-10:
                break
        return max(0.0, min(1.0, 2.0 * series))

    def _permutation_pvalue(
        old_arr: np.ndarray,
        new_arr: np.ndarray,
        observed_stat: float,
        *,
        n_perm: int,
        seed: int,
    ) -> float:
        combined = np.concatenate([old_arr, new_arr])
        n_old = old_arr.size
        rng = np.random.RandomState(seed)
        exceed = 0
        n_perm = max(64, int(n_perm))
        for _ in range(n_perm):
            perm = rng.permutation(combined.size)
            perm_old = np.sort(combined[perm[:n_old]])
            perm_new = np.sort(combined[perm[n_old:]])
            stat = _ks_statistic(perm_old, perm_new)
            if stat >= observed_stat - 1e-12:
                exceed += 1
        return (exceed + 1.0) / (n_perm + 1.0)

    old = np.sort(_to_clean_array(values_old))
    new = np.sort(_to_clean_array(values_new))
    if old.size < 5 or new.size < 5:
        return 0.0, 1.0
    stat = _ks_statistic(old, new)

    tie_frac_old = 1.0 - (np.unique(old).size / float(old.size))
    tie_frac_new = 1.0 - (np.unique(new).size / float(new.size))
    auto_needs_perm = (
        min(old.size, new.size) < 40
        or tie_frac_old > 0.1
        or tie_frac_new > 0.1
    )
    chosen = method
    if chosen not in {"auto", "asymptotic", "permutation"}:
        chosen = "auto"
    if chosen == "auto":
        chosen = "permutation" if auto_needs_perm else "asymptotic"

    if chosen == "permutation":
        pvalue = _permutation_pvalue(
            old,
            new,
            stat,
            n_perm=n_permutations,
            seed=random_state,
        )
    else:
        pvalue = _asymptotic_pvalue(stat, int(old.size), int(new.size))
    return stat, pvalue


def compute_mmd_rbf(
    values_old: list[float],
    values_new: list[float],
    *,
    max_points: int = 512,
) -> float:
    """Compute unbiased RBF-MMD^2 between two one-dimensional samples.

    Reference:
    Gretton et al. (2012), kernel two-sample tests (MMD).
    """
    old = _to_clean_array(values_old)
    new = _to_clean_array(values_new)
    if old.size < 2 or new.size < 2:
        return 0.0

    rng = np.random.RandomState(0)
    if old.size > max_points:
        old = old[rng.choice(old.size, size=max_points, replace=False)]
    if new.size > max_points:
        new = new[rng.choice(new.size, size=max_points, replace=False)]

    sigma = _median_heuristic_bandwidth(np.concatenate([old, new]))
    gamma = 1.0 / (2.0 * sigma * sigma)

    diff_xx = old[:, None] - old[None, :]
    diff_yy = new[:, None] - new[None, :]
    diff_xy = old[:, None] - new[None, :]
    k_xx = np.exp(-gamma * (diff_xx * diff_xx))
    k_yy = np.exp(-gamma * (diff_yy * diff_yy))
    k_xy = np.exp(-gamma * (diff_xy * diff_xy))

    m = float(old.size)
    n = float(new.size)
    if m < 2 or n < 2:
        return 0.0
    term_xx = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1.0))
    term_yy = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1.0))
    term_xy = k_xy.mean()
    mmd2 = float(term_xx + term_yy - 2.0 * term_xy)
    return max(0.0, mmd2)


def benjamini_hochberg_qvalues(pvalues: dict[str, float]) -> dict[str, float]:
    """Benjamini-Hochberg FDR adjustment.

    Reference:
    Benjamini & Hochberg (1995), controlling false discovery rate.
    """
    if not pvalues:
        return {}
    items = sorted(
        ((k, min(1.0, max(0.0, float(v)))) for k, v in pvalues.items()),
        key=lambda x: x[1],
    )
    m = len(items)
    qvals: list[float] = [1.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        _, p = items[rank - 1]
        raw = (p * m) / float(rank)
        running = min(running, raw)
        qvals[rank - 1] = min(1.0, running)
    return {items[i][0]: qvals[i] for i in range(m)}


def compute_distribution_drift(
    baseline_values: dict[str, list[float]] | None,
    current_values: dict[str, list[float]],
    *,
    alpha: float = 0.05,
    effect_threshold: float = 0.1,
    mmd_effect_threshold: float = 0.01,
    ks_method: str = "auto",
    ks_permutations: int = 256,
    max_points: int = 512,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Compute distribution-level drift stats and drift alerts per property.

    Alert rule:
    (KS q-value <= alpha) and (scaled Wasserstein >= effect_threshold)
    and (MMD^2 >= mmd_effect_threshold)
    where scaled Wasserstein is normalized by baseline IQR.
    """
    if not baseline_values:
        return {}, []

    details: dict[str, dict[str, Any]] = {}
    pvalues: dict[str, float] = {}
    for prop, new_vals in current_values.items():
        old_vals = baseline_values.get(prop, [])
        if len(old_vals) < 10 or len(new_vals) < 10:
            continue
        ks_stat, ks_p = compute_ks_2sample(
            old_vals,
            new_vals,
            method=ks_method,
            n_permutations=ks_permutations,
            random_state=0,
        )
        w1 = compute_wasserstein_1d(old_vals, new_vals)
        mmd2 = compute_mmd_rbf(old_vals, new_vals, max_points=max_points)

        old_arr = _to_clean_array(old_vals)
        if old_arr.size >= 2:
            q1, q3 = np.quantile(old_arr, [0.25, 0.75])
            iqr = float(q3 - q1)
        else:
            iqr = 0.0
        denom = iqr if iqr > 1e-12 else 1.0
        scaled_w1 = w1 / denom

        details[prop] = {
            "n_old": float(len(old_vals)),
            "n_new": float(len(new_vals)),
            "ks_stat": ks_stat,
            "ks_pvalue": ks_p,
            "wasserstein": w1,
            "scaled_wasserstein": scaled_w1,
            "mmd2_rbf": mmd2,
        }
        pvalues[prop] = ks_p

    qvals = benjamini_hochberg_qvalues(pvalues)
    alerts: list[str] = []
    for prop, stats in details.items():
        qv = qvals.get(prop, 1.0)
        w1_ok = float(stats["scaled_wasserstein"]) >= effect_threshold
        mmd_ok = float(stats["mmd2_rbf"]) >= mmd_effect_threshold
        is_alert = bool(
            qv <= alpha and w1_ok and mmd_ok
        )
        stats["ks_qvalue"] = qv
        stats["alert_rule"] = "ks_q & scaled_w1 & mmd2"
        stats["w1_effect_threshold"] = effect_threshold
        stats["mmd_effect_threshold"] = mmd_effect_threshold
        stats["alert"] = is_alert
        if is_alert:
            alerts.append(prop)
    return details, alerts


def compute_drift(
    values_old: list[float],
    values_new: list[float],
    n_bins: int = 50,
) -> float:
    """Approximate KL divergence between two value distributions.

    Uses histogram-based estimation. Returns KL(new || old).
    """
    clean_old = [v for v in values_old if v is not None and math.isfinite(v)]
    clean_new = [v for v in values_new if v is not None and math.isfinite(v)]
    if len(clean_old) < 10 or len(clean_new) < 10:
        return 0.0

    lo = min(min(clean_old), min(clean_new))
    hi = max(max(clean_old), max(clean_new))
    if hi <= lo:
        return 0.0

    bin_width = (hi - lo) / n_bins
    eps = 1e-8

    def _hist(vals: list[float]) -> list[float]:
        counts = [0.0] * n_bins
        for v in vals:
            idx = min(int((v - lo) / bin_width), n_bins - 1)
            counts[idx] += 1
        total = sum(counts)
        return [(c / total) + eps for c in counts]

    p = _hist(clean_new)
    q = _hist(clean_old)
    kl = sum(p[i] * math.log(p[i] / q[i]) for i in range(n_bins))
    return max(0.0, kl)


def validate_dataset(
    records: list[dict[str, Any]],
    *,
    split_ids: dict[str, set[str]] | None = None,
    properties: list[str] | None = None,
    property_stats: dict[str, dict[str, float]] | None = None,
    required_fields: list[str] | None = None,
    units_spec: dict[str, dict[str, Any]] | None = None,
    baseline_property_summary: dict[str, dict[str, float]] | None = None,
    baseline_property_values: dict[str, list[float]] | None = None,
    drift_alpha: float = 0.05,
    drift_effect_threshold: float = 0.1,
    drift_mmd_effect_threshold: float = 0.01,
    drift_ks_method: str = "auto",
    drift_ks_permutations: int = 256,
    drift_max_points: int = 512,
    drift_hard_gate: bool = False,
    strict: bool = False,
) -> ValidationReport:
    """Run full validation pipeline on a list of sample records.

    Parameters
    ----------
    records : list[dict]
        Flat list of sample dicts (typically from DataFrame rows).
    split_ids : dict, optional
        ``{"train": set(...), "val": set(...), "test": set(...)}`` for leakage check.
    properties : list[str], optional
        Property names for outlier/plausibility checks.
    property_stats : dict, optional
        Pre-computed per-property mean/std for trust scoring.
    required_fields : list[str], optional
        Fields required in schema (default: ``["jid", "atoms"]``).
    baseline_property_values : dict, optional
        Optional per-property baseline value lists for two-sample drift tests.
        If provided, KS/Wasserstein/MMD drift metrics are computed.
    drift_alpha : float
        BH-FDR significance threshold used for drift alerts.
    drift_effect_threshold : float
        Minimum scaled Wasserstein effect size for a drift alert.
    drift_mmd_effect_threshold : float
        Minimum RBF-MMD^2 effect size for a drift alert.
    drift_ks_method : str
        KS p-value method: auto, asymptotic, permutation.
    drift_ks_permutations : int
        Permutation count when KS permutation calibration is used.
    drift_max_points : int
        Maximum sample size per split for MMD computation.
    drift_hard_gate : bool
        If True, non-zero drift alerts fail hard gates.
    strict : bool
        If True, warnings (duplicates, outliers) also cause gate failure.
    """
    t0 = time.time()
    report = ValidationReport(n_samples=len(records))

    # Schema
    schema_viol = check_schema(records, required_fields)
    report.schema_violations = len(schema_viol)
    report.details["schema_violations"] = schema_viol[:20]  # cap detail log

    # Provenance
    prov_missing = check_provenance(records)
    report.provenance_missing = len(prov_missing)
    report.details["provenance_missing"] = prov_missing[:20]

    # Duplicates
    dupes = check_duplicates(records)
    report.duplicate_count = sum(len(v) - 1 for v in dupes.values())
    report.details["duplicate_keys"] = list(dupes.keys())[:20]

    # Leakage
    if split_ids:
        overlap = check_leakage(split_ids)
        report.leakage_count = sum(overlap.values())
        report.details["leakage_overlap"] = overlap

    # Units
    unit_violations = check_units(records, properties=properties, units_spec=units_spec)
    report.unit_violations = len(unit_violations)
    report.details["unit_violations"] = unit_violations[:20]

    # Outliers
    if properties:
        outlier_detail: dict[str, dict[str, Any]] = {}
        for prop in properties:
            vals = []
            sample_ids: list[str] = []
            for r in records:
                v = r.get(prop)
                if v is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        vals.append(float(v))
                        sample_ids.append(str(r.get("jid", r.get("source_id", "unknown"))))
            out_idx = check_outliers(vals)
            outlier_ids = [sample_ids[i] for i in out_idx if i < len(sample_ids)]
            outlier_detail[prop] = {"count": len(out_idx), "sample_ids": outlier_ids[:20]}
            report.outlier_count += len(out_idx)
        report.details["outliers_per_property"] = outlier_detail

    # Property summary + drift
    property_summary = summarize_properties(records, properties)
    report.details["property_summary"] = property_summary
    report.drift_summary = compute_mean_shift_drift(
        baseline_property_summary, property_summary
    )
    current_property_values = collect_property_values(records, properties)
    if baseline_property_values:
        drift_details, drift_alerts = compute_distribution_drift(
            baseline_property_values,
            current_property_values,
            alpha=drift_alpha,
            effect_threshold=drift_effect_threshold,
            mmd_effect_threshold=drift_mmd_effect_threshold,
            ks_method=drift_ks_method,
            ks_permutations=drift_ks_permutations,
            max_points=drift_max_points,
        )
        report.details["distribution_drift"] = drift_details
        report.details["distribution_drift_alerts"] = drift_alerts
        report.drift_alert_count = len(drift_alerts)

    # Trust scoring
    tier_counts: dict[str, int] = {"raw": 0, "curated": 0, "benchmark": 0}
    for rec in records:
        ts = compute_trust_score(
            rec, property_stats=property_stats, properties=properties
        )
        tier_counts[ts.tier] = tier_counts.get(ts.tier, 0) + 1
    report.trust_tier_counts = tier_counts

    report.duration_sec = time.time() - t0
    report.apply_gates(strict=strict, drift_hard_gate=drift_hard_gate)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validate-data",
        description="ATLAS data validation pipeline: schema, provenance, leakage, outliers, trust scoring.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report output path (default: artifacts/validation_report.json)",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Markdown summary output path (default: alongside JSON)",
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
        help="Max samples to validate (default: 3000)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for split generation",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings (duplicates, outliers) as failures",
    )
    parser.add_argument(
        "--strict-gates",
        action="store_true",
        help="Alias flag to emphasize hard gates (schema/provenance/leakage/units).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: schema + provenance + leakage only",
    )
    parser.add_argument(
        "--units-spec",
        type=Path,
        default=None,
        help="Optional JSON file overriding expected units per property",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=None,
        help="Optional previous validation JSON report used for drift summary",
    )
    parser.add_argument(
        "--baseline-input",
        type=Path,
        default=None,
        help="Optional baseline dataset (.json/.jsonl/.csv) for two-sample drift tests.",
    )
    parser.add_argument(
        "--drift-alpha",
        type=float,
        default=0.05,
        help="FDR threshold alpha for drift alerts (default: 0.05).",
    )
    parser.add_argument(
        "--drift-effect-threshold",
        type=float,
        default=0.1,
        help="Minimum scaled Wasserstein effect size for a drift alert.",
    )
    parser.add_argument(
        "--drift-mmd-effect-threshold",
        type=float,
        default=0.01,
        help="Minimum RBF-MMD^2 effect size for a drift alert.",
    )
    parser.add_argument(
        "--drift-ks-method",
        type=str,
        default="auto",
        choices=["auto", "asymptotic", "permutation"],
        help="KS p-value method used in drift testing.",
    )
    parser.add_argument(
        "--drift-ks-permutations",
        type=int,
        default=256,
        help="Permutation count for KS permutation mode.",
    )
    parser.add_argument(
        "--drift-max-points",
        type=int,
        default=512,
        help="Maximum sample count per side for MMD computation.",
    )
    parser.add_argument(
        "--drift-hard-gate",
        action="store_true",
        help="Fail validation if drift alerts are detected.",
    )
    parser.add_argument(
        "--schema-version",
        type=str,
        default="2.0",
        help="Validation report schema version",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional input dataset file (.json/.jsonl/.csv). If provided, validate this file instead of JARVIS source.",
    )
    return parser


def _load_split_records(
    properties: list[str],
    max_samples: int,
    split_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, set[str]], int]:
    """Load dataset splits and collect records + IDs for validation."""
    import pandas as pd

    from atlas.data.crystal_dataset import PROPERTY_MAP
    from atlas.data.jarvis_client import JARVISClient

    all_records: list[dict[str, Any]] = []
    split_ids: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    provenance_injected = 0
    rev_map = {v: k for k, v in PROPERTY_MAP.items()}

    client = JARVISClient()
    df = client.load_dft_3d()

    jarvis_cols = [rev_map[p] for p in properties if p in rev_map]
    valid_mask = pd.Series(False, index=df.index)
    for col in jarvis_cols:
        if col in df.columns:
            valid_mask |= df[col].notna() & (df[col] != "na")
    df = df[valid_mask].copy()

    if "ehull" in df.columns:
        df = df[df["ehull"].notna() & (df["ehull"] <= 0.1)]

    df = df[df["atoms"].notna()].reset_index(drop=True)
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=split_seed).reset_index(drop=True)

    n = len(df)
    if n == 0:
        return [], split_ids, provenance_injected

    rng = np.random.RandomState(split_seed)
    indices = rng.permutation(n)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    index_map = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }

    for split, split_idx in index_map.items():
        df_split = df.iloc[split_idx].reset_index(drop=True)
        rows = df_split.to_dict("records")
        for row in rows:
            for prop in properties:
                source_col = rev_map.get(prop)
                if source_col:
                    row[prop] = row.get(source_col)
            if _attach_known_source_provenance(row):
                provenance_injected += 1
            jid = row.get("jid")
            if jid is not None:
                split_ids[split].add(str(jid))
            all_records.append(row)

    return all_records, split_ids, provenance_injected


def _load_units_spec(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return DEFAULT_UNITS_SPEC
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("--units-spec must be a JSON object")
    merged = dict(DEFAULT_UNITS_SPEC)
    for key, value in payload.items():
        if isinstance(value, dict):
            merged[str(key)] = value
    return merged


def _load_baseline_property_summary(path: Path | None) -> dict[str, dict[str, float]] | None:
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None
    details = payload.get("details", {})
    if not isinstance(details, dict):
        return None
    summary = details.get("property_summary")
    return summary if isinstance(summary, dict) else None


def _load_records_from_input(path: Path) -> list[dict[str, Any]]:
    """Load records from a user-provided JSON/JSONL/CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"--input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [r for r in payload if isinstance(r, dict)]
        if isinstance(payload, dict):
            maybe_records = payload.get("records")
            if isinstance(maybe_records, list):
                return [r for r in maybe_records if isinstance(r, dict)]
        raise ValueError("--input .json must be a list of objects or an object with 'records' list")
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows
    if suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        return df.to_dict("records")
    raise ValueError("--input must be one of: .json, .jsonl, .csv")


def _compute_property_stats(
    properties: list[str],
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute per-property robust and classical stats from records."""
    from atlas.data.crystal_dataset import PROPERTY_MAP

    rev_map = {v: k for k, v in PROPERTY_MAP.items()}
    stats: dict[str, dict[str, float]] = {}
    for prop in properties:
        vals: list[float] = []
        source_col = rev_map.get(prop)
        for r in records:
            v = r.get(prop)
            if v is None and source_col:
                v = r.get(source_col)
            if v is not None:
                with contextlib.suppress(TypeError, ValueError):
                    fv = float(v)
                    if math.isfinite(fv):
                        vals.append(fv)
        if vals:
            arr = np.asarray(vals, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std(ddof=0))
            median = float(np.median(arr))
            mad = float(np.median(np.abs(arr - median)))
            stats[prop] = {
                "mean": mean,
                "std": std,
                "median": median,
                "mad": mad,
            }
    return stats


def _select_records_for_property_stats(
    records: list[dict[str, Any]],
    split_ids: dict[str, set[str]] | None,
) -> tuple[list[dict[str, Any]], str]:
    """Choose records used to estimate trust-score property stats.

    Prefers train split records to avoid information leakage.
    """
    if not records:
        return [], "empty"
    if not split_ids:
        return records, "all_records"
    train_ids = {str(x) for x in split_ids.get("train", set())}
    if not train_ids:
        return records, "all_records"
    selected: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        if _sample_id(rec, i) in train_ids:
            selected.append(rec)
    if not selected:
        return records, "all_records_fallback"
    return selected, "train_split"


def _collect_property_values_from_records(
    properties: list[str],
    records: list[dict[str, Any]],
) -> dict[str, list[float]]:
    """Collect canonical property values with source-column fallback."""
    return collect_property_values(records, properties, allow_source_fallback=True)


def _summarize_property_values(
    property_values: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for prop, vals in property_values.items():
        if not vals:
            summary[prop] = {
                "count": 0.0,
                "mean": math.nan,
                "std": math.nan,
                "min": math.nan,
                "max": math.nan,
            }
            continue
        arr = np.asarray(vals, dtype=float)
        summary[prop] = {
            "count": float(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for validate-data."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        from atlas.data.crystal_dataset import resolve_phase2_property_group
    except ImportError as exc:
        print(f"[ERROR] Cannot import dataset module: {exc}")
        return 1

    # Resolve properties
    try:
        properties = resolve_phase2_property_group(args.property_group)
    except (KeyError, ValueError):
        from atlas.data.crystal_dataset import DEFAULT_PROPERTIES
        properties = list(DEFAULT_PROPERTIES)

    # Load data and compute stats
    provenance_injected = 0
    if args.input is not None:
        all_records = _load_records_from_input(args.input)
        split_ids = {}
    else:
        all_records, split_ids, provenance_injected = _load_split_records(
            properties, args.max_samples, args.split_seed
        )
    if not all_records:
        print("[ERROR] No records loaded for validation after filtering.")
        return 2
    units_spec = _load_units_spec(args.units_spec)
    baseline_summary = _load_baseline_property_summary(args.baseline_report)
    baseline_property_values: dict[str, list[float]] | None = None
    if args.baseline_input is not None:
        baseline_records = _load_records_from_input(args.baseline_input)
        baseline_property_values = _collect_property_values_from_records(
            properties, baseline_records
        )
        if baseline_summary is None:
            baseline_summary = _summarize_property_values(baseline_property_values)
    stats_records, stats_source = _select_records_for_property_stats(
        all_records, split_ids
    )
    property_stats = (
        _compute_property_stats(properties, stats_records) if not args.quick else {}
    )

    # Run validation
    report = validate_dataset(
        all_records,
        split_ids=split_ids,
        properties=None if args.quick else properties,
        property_stats=property_stats or None,
        units_spec=units_spec,
        baseline_property_summary=baseline_summary,
        baseline_property_values=None if args.quick else baseline_property_values,
        drift_alpha=args.drift_alpha,
        drift_effect_threshold=args.drift_effect_threshold,
        drift_mmd_effect_threshold=args.drift_mmd_effect_threshold,
        drift_ks_method=args.drift_ks_method,
        drift_ks_permutations=max(32, int(args.drift_ks_permutations)),
        drift_max_points=max(32, int(args.drift_max_points)),
        drift_hard_gate=args.drift_hard_gate,
        strict=args.strict,
    )
    report.schema_version = args.schema_version
    report.details["provenance_injected_from_source"] = provenance_injected
    report.details["property_stats_source"] = stats_source
    report.details["property_stats_n_records"] = len(stats_records)

    # Resolve output paths
    from atlas.config import get_config
    cfg = get_config()
    if args.output is None:
        args.output = cfg.paths.artifacts_dir / "validation_report.json"
    if args.markdown is None:
        args.markdown = args.output.with_suffix(".md")

    report.to_json(args.output)
    report.to_markdown(args.markdown)

    print(f"[validate-data] Report saved: {args.output}")
    print(f"[validate-data] Summary saved: {args.markdown}")
    print(f"  samples={report.n_samples}, gate={'PASS' if report.gate_pass else 'FAIL'}")
    print(f"  schema_violations={report.schema_violations}")
    print(f"  provenance_missing={report.provenance_missing}")
    print(f"  leakage={report.leakage_count}")
    print(f"  duplicates={report.duplicate_count}")
    print(f"  outliers={report.outlier_count}")
    print(f"  drift_alerts={report.drift_alert_count}")
    print(f"  trust tiers: {report.trust_tier_counts}")

    if not report.gate_pass:
        print("[ERROR] Validation gates FAILED:")
        for gf in report.gate_failures:
            print(f"  - {gf}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
