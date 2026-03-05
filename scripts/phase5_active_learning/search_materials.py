#!/usr/bin/env python3
"""
Script 07: Multi-Property Materials Search

Search the full 75,993 materials database using any combination of
properties — both direct (from DFT) and derived (physics estimates).

Available properties for search:
    ─── Direct DFT Properties ───
    bandgap_best          : Band gap (eV)
    formation_energy_peratom : Formation energy (eV/atom)
    ehull                 : Energy above hull (eV/atom)
    bulk_modulus_kv       : Bulk modulus (GPa)
    shear_modulus_gv      : Shear modulus (GPa)
    density               : Density (g/cm³)
    seebeck_best          : Seebeck coefficient (μV/K)
    elec_cond_best        : Electrical conductivity (S/m)
    thermal_cond_best     : Thermal conductivity (W/mK)
    spillage              : Topological spillage
    slme                  : Solar cell efficiency (%)
    Tc_supercon           : Superconducting Tc estimate (K)

    ─── Derived Physics Estimates ───
    melting_point_est     : Melting point (K)
    debye_temperature     : Debye temperature (K)
    kappa_slack           : Thermal conductivity - Slack model (W/mK)
    hardness_chen         : Vickers hardness (GPa)
    pugh_ratio            : K/G ductility ratio (>1.75 = ductile)
    youngs_modulus        : Young's modulus (GPa)
    electromigration_resistance : EM resistance score (0-1)
    thermal_expansion_est : Thermal expansion (×10⁻⁶/K)
    conductivity_class    : metal / semimetal / semiconductor / insulator

Examples:
    # Find low-melting-point metals with good conductivity
    python scripts/phase5_active_learning/search_materials.py --metal --melting-max 600 --sort melting_point_est

    # Find hard, stiff ceramics
    python scripts/phase5_active_learning/search_materials.py --hardness-min 10 --youngs-min 200

    # Find thermoelectric materials (high Seebeck, semiconductor)
    python scripts/phase5_active_learning/search_materials.py --semiconductor --seebeck-min 200 --sort seebeck_best -desc

    # Find EM-resistant interconnect metals
    python scripts/phase5_active_learning/search_materials.py --metal --em-min 0.5 --sort electromigration_resistance -desc

    # Full database summary
    python scripts/phase5_active_learning/search_materials.py --summary

    # Custom query: any column with min/max
    python scripts/phase5_active_learning/search_materials.py --filter "melting_point_est<600" --filter "bandgap_best<0.1"
"""

import argparse
import math
import re
import sys
from numbers import Integral, Real
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_FILTER_PATTERN = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*(<=|>=|<|>)\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
_MAX_RESULTS_LIMIT = 5000


def _is_finite(value: float | None) -> bool:
    return value is None or math.isfinite(float(value))


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        number_f = float(value)
        if not math.isfinite(number_f) or not number_f.is_integer():
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        number = int(number_f)
    else:
        try:
            number = int(value)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return number


def _coerce_optional_finite_float(
    value: object,
    *,
    name: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if min_value is not None and out < float(min_value):
        raise ValueError(f"{name} must be >= {min_value}")
    if max_value is not None and out > float(max_value):
        raise ValueError(f"{name} must be <= {max_value}")
    return out


def _merge_bounds(
    existing: tuple[float | None, float | None],
    incoming: tuple[float | None, float | None],
) -> tuple[float | None, float | None]:
    lo_a, hi_a = existing
    lo_b, hi_b = incoming
    lo = lo_a if lo_b is None else (lo_b if lo_a is None else max(lo_a, lo_b))
    hi = hi_a if hi_b is None else (hi_b if hi_a is None else min(hi_a, hi_b))
    return lo, hi


def _parse_custom_filter(expr: str) -> tuple[str, tuple[float | None, float | None]]:
    match = _FILTER_PATTERN.match(expr or "")
    if match is None:
        raise ValueError(f"Invalid custom filter '{expr}'. Use forms like 'column<1.0' or 'column>=2'.")

    column, operator, raw_value = match.groups()
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError(f"Custom filter value must be finite: {expr}")
    if operator in {"<", "<="}:
        return column, (None, value)
    return column, (value, None)


def _validate_args(args: argparse.Namespace) -> tuple[bool, str]:
    presets = [bool(args.metal), bool(args.semiconductor), bool(args.insulator)]
    if sum(presets) > 1:
        return False, "Only one of --metal/--semiconductor/--insulator can be set."

    try:
        args.max = _coerce_positive_int(getattr(args, "max"), name="--max")
    except ValueError as exc:
        return False, str(exc)
    if args.max > _MAX_RESULTS_LIMIT:
        return False, f"--max cannot exceed {_MAX_RESULTS_LIMIT}"
    try:
        args.ehull_max = _coerce_optional_finite_float(args.ehull_max, name="--ehull-max", min_value=0.0)
    except ValueError as exc:
        return False, str(exc)
    if args.sort is not None and not str(args.sort).strip():
        return False, "--sort must not be empty"
    if args.save is not None and not str(args.save).strip():
        return False, "--save must not be empty"
    if args.columns:
        normalized = []
        for col in args.columns:
            token = str(col).strip()
            if not token:
                return False, "--columns entries must be non-empty"
            normalized.append(token)
        args.columns = normalized

    numeric_fields = (
        "bandgap_min",
        "bandgap_max",
        "melting_min",
        "melting_max",
        "hardness_min",
        "hardness_max",
        "youngs_min",
        "youngs_max",
        "seebeck_min",
        "em_min",
        "em_max",
        "kappa_min",
        "kappa_max",
        "density_min",
        "density_max",
    )
    for field in numeric_fields:
        cli_name = f"--{field.replace('_', '-')}"
        try:
            setattr(args, field, _coerce_optional_finite_float(getattr(args, field, None), name=cli_name))
        except ValueError as exc:
            return False, str(exc)

    paired_ranges = (
        ("bandgap_min", "bandgap_max"),
        ("melting_min", "melting_max"),
        ("hardness_min", "hardness_max"),
        ("youngs_min", "youngs_max"),
        ("em_min", "em_max"),
        ("kappa_min", "kappa_max"),
        ("density_min", "density_max"),
    )
    for low_key, high_key in paired_ranges:
        low = getattr(args, low_key, None)
        high = getattr(args, high_key, None)
        if low is not None and high is not None and float(low) > float(high):
            return False, f"--{low_key.replace('_', '-')} cannot be greater than --{high_key.replace('_', '-')}"

    for field in ("em_min", "em_max"):
        value = getattr(args, field, None)
        if value is not None and not (0.0 <= float(value) <= 1.0):
            return False, f"--{field.replace('_', '-')} must be within [0, 1]"

    if args.filter:
        for expr in args.filter:
            try:
                _parse_custom_filter(expr)
            except ValueError as exc:
                return False, str(exc)

    return True, ""


def _validate_criteria_bounds(criteria: dict[str, object]) -> None:
    for column, rule in criteria.items():
        if not (isinstance(rule, tuple) and len(rule) == 2):
            continue
        lo, hi = rule
        if lo is not None and not math.isfinite(float(lo)):
            raise ValueError(f"Lower bound for {column} must be finite")
        if hi is not None and not math.isfinite(float(hi)):
            raise ValueError(f"Upper bound for {column} must be finite")
        if lo is not None and hi is not None and float(lo) > float(hi):
            raise ValueError(f"Inconsistent bounds for {column}: lower {lo} > upper {hi}")


def _build_criteria(args: argparse.Namespace) -> dict[str, object]:
    criteria: dict[str, object] = {}
    criteria["ehull"] = (None, float(args.ehull_max))

    if args.metal:
        criteria["conductivity_class"] = "metal"
    elif args.semiconductor:
        criteria["bandgap_best"] = (0.5, 3.0)
    elif args.insulator:
        criteria["bandgap_best"] = (3.0, None)

    range_filters = [
        ("bandgap_best", args.bandgap_min, args.bandgap_max),
        ("melting_point_est", args.melting_min, args.melting_max),
        ("hardness_chen", args.hardness_min, args.hardness_max),
        ("youngs_modulus", args.youngs_min, args.youngs_max),
        ("seebeck_best", args.seebeck_min, None),
        ("electromigration_resistance", args.em_min, args.em_max),
        ("kappa_slack", args.kappa_min, args.kappa_max),
        ("density", args.density_min, args.density_max),
    ]
    for column, low, high in range_filters:
        if low is not None or high is not None:
            criteria[column] = (low, high)

    if args.ductile:
        criteria["is_ductile"] = True

    if args.filter:
        for expr in args.filter:
            column, bounds = _parse_custom_filter(expr)
            existing = criteria.get(column)
            if isinstance(existing, tuple) and len(existing) == 2:
                criteria[column] = _merge_bounds(existing, bounds)
            else:
                criteria[column] = bounds

    _validate_criteria_bounds(criteria)
    return criteria


def _resolve_display_columns(results: pd.DataFrame, extra_columns: list[str] | None = None) -> list[str]:
    ordered: list[str] = []

    def add_column(name: str) -> None:
        if name not in ordered and name in results.columns:
            ordered.append(name)

    for base in ("jid", "formula"):
        add_column(base)

    auto_cols = [
        "bandgap_best",
        "conductivity_class",
        "melting_point_est",
        "hardness_chen",
        "youngs_modulus",
        "electromigration_resistance",
        "kappa_slack",
        "pugh_ratio",
        "density",
    ]
    for col in auto_cols:
        if col in results.columns and results[col].notna().any():
            add_column(col)

    if extra_columns:
        for col in extra_columns:
            add_column(col)

    return ordered


def _validate_query_columns(
    df: pd.DataFrame,
    *,
    criteria: dict[str, object],
    sort_by: str | None,
    requested_columns: list[str] | None = None,
) -> tuple[bool, str]:
    missing = sorted(col for col in criteria if col not in df.columns)
    if missing:
        return False, f"Unknown criteria columns: {', '.join(missing)}"

    for column, rule in criteria.items():
        if isinstance(rule, tuple) and len(rule) == 2:
            if not pd.api.types.is_numeric_dtype(df[column]):
                return False, f"Range filter requires numeric column: {column}"

    if sort_by and sort_by not in df.columns:
        return False, f"Unknown sort column: {sort_by}"
    if requested_columns:
        missing_requested = sorted(col for col in requested_columns if col not in df.columns)
        if missing_requested:
            return False, f"Unknown requested output columns: {', '.join(missing_requested)}"
    return True, ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-Property Materials Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Preset conductivity filters
    parser.add_argument("--metal", action="store_true", help="Only metals (bandgap ≈ 0)")
    parser.add_argument("--semiconductor", action="store_true", help="Only semiconductors (0.5-3 eV)")
    parser.add_argument("--insulator", action="store_true", help="Only insulators (>3 eV)")

    # Property range filters
    parser.add_argument("--bandgap-min", type=float, help="Min band gap (eV)")
    parser.add_argument("--bandgap-max", type=float, help="Max band gap (eV)")
    parser.add_argument("--melting-min", type=float, help="Min melting point (K)")
    parser.add_argument("--melting-max", type=float, help="Max melting point (K)")
    parser.add_argument("--hardness-min", type=float, help="Min Vickers hardness (GPa)")
    parser.add_argument("--hardness-max", type=float, help="Max Vickers hardness (GPa)")
    parser.add_argument("--youngs-min", type=float, help="Min Young's modulus (GPa)")
    parser.add_argument("--youngs-max", type=float, help="Max Young's modulus (GPa)")
    parser.add_argument("--seebeck-min", type=float, help="Min Seebeck coeff (μV/K)")
    parser.add_argument("--em-min", type=float, help="Min EM resistance (0-1)")
    parser.add_argument("--em-max", type=float, help="Max EM resistance (0-1)")
    parser.add_argument("--kappa-min", type=float, help="Min thermal conductivity (W/mK)")
    parser.add_argument("--kappa-max", type=float, help="Max thermal conductivity (W/mK)")
    parser.add_argument("--density-min", type=float, help="Min density (g/cm³)")
    parser.add_argument("--density-max", type=float, help="Max density (g/cm³)")
    parser.add_argument("--ehull-max", type=float, default=0.1, help="Max energy above hull (eV/atom)")
    parser.add_argument("--ductile", action="store_true", help="Only ductile materials")

    # Custom filters
    parser.add_argument("--filter", action="append", help="Custom filter: 'column<value' or 'column>value'")

    # Output
    parser.add_argument("--sort", type=str, help="Column to sort by")
    parser.add_argument("--desc", "-desc", dest="desc", action="store_true", help="Sort descending")
    parser.add_argument(
        "--max",
        type=int,
        default=30,
        help=f"Max results (default: 30, cap: {_MAX_RESULTS_LIMIT})",
    )
    parser.add_argument("--summary", action="store_true", help="Show database summary")
    parser.add_argument("--columns", nargs="+", help="Extra columns to display")
    parser.add_argument("--save", type=str, help="Save results to CSV")
    return parser


def main(argv: list[str] | None = None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    ok, message = _validate_args(args)
    if not ok:
        print(f"\n[ERROR] {message}", file=sys.stderr)
        return 2

    from atlas.data.jarvis_client import JARVISClient
    from atlas.data.property_estimator import PropertyEstimator

    client = JARVISClient()
    estimator = PropertyEstimator()

    # Load and process
    print("\n=== Loading JARVIS-DFT Database ===\n")
    raw_df = client.load_dft_3d()

    print("  Computing derived properties...")
    df = estimator.extract_all_properties(raw_df)
    print(f"  Total materials: {len(df)}")

    # Summary mode
    if args.summary:
        print("\n=== Full Property Summary ===\n")
        summary = estimator.property_summary(df)
        for prop, stats in summary.items():
            if isinstance(stats, dict) and "count" in stats:
                print(f"  {prop:40s}: {stats['count']:6d} values | "
                      f"mean={stats['mean']:10.3f} | "
                      f"range=[{stats['min']:.3f}, {stats['max']:.3f}]")
            else:
                print(f"  {prop}: {stats}")
        return 0

    try:
        criteria = _build_criteria(args)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 2
    ok, message = _validate_query_columns(
        df,
        criteria=criteria,
        sort_by=args.sort,
        requested_columns=args.columns,
    )
    if not ok:
        print(f"\n[ERROR] {message}", file=sys.stderr)
        return 2

    # Search
    print(f"\n=== Searching with {len(criteria)} criteria ===\n")
    for col, crit in criteria.items():
        print(f"  {col}: {crit}")

    results = estimator.search(
        df, criteria,
        sort_by=args.sort,
        ascending=not args.desc,
        max_results=args.max,
    )
    print(f"\n  Found {len(results)} matching materials\n")

    if len(results) == 0:
        print("  No materials match all criteria. Try relaxing some filters.")
        return 0

    # Display columns
    display_cols = _resolve_display_columns(results, extra_columns=args.columns)
    display_df = results[display_cols].copy()

    # Format for display
    with pd.option_context(
        "display.max_columns",
        20,
        "display.width",
        200,
        "display.float_format",
        lambda x: f"{x:.3f}" if abs(x) < 1000 else f"{x:.0f}",
    ):
        print(display_df.to_string(index=False))

    # Save
    if args.save:
        save_path = Path(args.save).expanduser()
        if save_path.parent and not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(save_path, index=False)
        print(f"\n  Saved {len(results)} results to {save_path}")

    print("\n[OK] Search complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
