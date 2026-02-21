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
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
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
    parser.add_argument("-desc", action="store_true", help="Sort descending")
    parser.add_argument("--max", type=int, default=30, help="Max results (default: 30)")
    parser.add_argument("--summary", action="store_true", help="Show database summary")
    parser.add_argument("--columns", nargs="+", help="Extra columns to display")
    parser.add_argument("--save", type=str, help="Save results to CSV")

    args = parser.parse_args()

    from atlas.config import get_config
    from atlas.data.jarvis_client import JARVISClient
    from atlas.data.property_estimator import PropertyEstimator

    cfg = get_config()
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
        return

    # Build criteria
    criteria = {}

    # Stability filter
    criteria["ehull"] = (None, args.ehull_max)

    # Conductivity preset
    if args.metal:
        criteria["conductivity_class"] = "metal"
    elif args.semiconductor:
        criteria["bandgap_best"] = (0.5, 3.0)
    elif args.insulator:
        criteria["bandgap_best"] = (3.0, None)

    # Range filters
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

    for col, lo, hi in range_filters:
        if lo is not None or hi is not None:
            criteria[col] = (lo, hi)

    if args.ductile:
        criteria["is_ductile"] = True

    # Custom filters
    if args.filter:
        for f in args.filter:
            if "<" in f:
                col, val = f.split("<", 1)
                criteria[col.strip()] = (None, float(val))
            elif ">" in f:
                col, val = f.split(">", 1)
                criteria[col.strip()] = (float(val), None)

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
        return

    # Display columns
    display_cols = ["jid", "formula"]
    auto_cols = [
        "bandgap_best", "conductivity_class",
        "melting_point_est", "hardness_chen", "youngs_modulus",
        "electromigration_resistance", "kappa_slack",
        "pugh_ratio", "density",
    ]
    # Only show columns that were part of filter or have data
    for col in auto_cols:
        if col in results.columns and results[col].notna().any():
            display_cols.append(col)
    if args.columns:
        display_cols.extend(args.columns)

    # Keep only columns that exist
    display_cols = [c for c in display_cols if c in results.columns]
    display_df = results[display_cols].copy()

    # Format for display
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}" if abs(x) < 1000 else f"{x:.0f}")
    print(display_df.to_string(index=False))

    # Save
    if args.save:
        results.to_csv(args.save, index=False)
        print(f"\n  Saved {len(results)} results to {args.save}")

    print(f"\n✓ Search complete!")


if __name__ == "__main__":
    main()
