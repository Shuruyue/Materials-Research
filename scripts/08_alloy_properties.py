#!/usr/bin/env python3
"""
Script 08: Multi-Phase Alloy Property Estimator

Estimates effective properties of multi-phase alloys from constituent
phase DFT/literature data using Voigt-Reuss-Hill mixing rules.

Built-in presets: SAC305, SAC405, SnPb63, pure_Sn, pure_Cu

Usage:
    python scripts/08_alloy_properties.py                    # SAC305 (default)
    python scripts/08_alloy_properties.py --alloy SAC405     # SAC405
    python scripts/08_alloy_properties.py --alloy SnPb63     # Leaded solder
    python scripts/08_alloy_properties.py --compare          # Compare all presets
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas.data.alloy_estimator import AlloyEstimator


# Published experimental data for validation
EXPERIMENTAL = {
    "SAC305": {
        "density_g_cm3": 7.38,
        "youngs_modulus_GPa": 51.0,
        "thermal_conductivity_W_mK": 58.7,
        "thermal_expansion_1e6_K": 21.7,
        "melting_point_K": 490.0,  # solidus ~217°C = 490K
    },
    "SNPB63": {
        "density_g_cm3": 8.40,
        "youngs_modulus_GPa": 30.0,
        "thermal_conductivity_W_mK": 50.0,
        "melting_point_K": 456.0,  # eutectic 183°C = 456K
    },
}


def main():
    parser = argparse.ArgumentParser(description="Alloy Property Estimator")
    parser.add_argument("--alloy", type=str, default="SAC305",
                        help="Alloy preset name (default: SAC305)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all built-in presets")
    parser.add_argument("--list", action="store_true",
                        help="List available presets")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable alloy presets:")
        for key in ["SAC305", "SAC405", "SnPb63", "pure_Sn", "pure_Cu"]:
            try:
                est = AlloyEstimator.from_preset(key)
                phases = ", ".join(p.name for p in est.phases)
                print(f"  {key:12s} — {est.name} [{phases}]")
            except Exception:
                pass
        return

    if args.compare:
        compare_all()
        return

    # Single alloy report
    try:
        est = AlloyEstimator.from_preset(args.alloy)
    except ValueError as e:
        print(f"\n  ✗ {e}")
        return

    key = args.alloy.upper().replace("-", "").replace(" ", "")
    exp = EXPERIMENTAL.get(key)
    est.print_report(experimental=exp)


def compare_all():
    """Compare properties across all preset alloys."""
    presets = ["SAC305", "SAC405", "SnPb63", "PURESN", "PURECU"]
    estimators = {}
    for name in presets:
        try:
            estimators[name] = AlloyEstimator.from_preset(name)
        except Exception:
            pass

    # Compute all
    all_props = {}
    for name, est in estimators.items():
        all_props[name] = est.estimate_properties()

    properties = [
        ("density_g_cm3", "Density (g/cm³)", ".2f"),
        ("bulk_modulus_GPa", "K (GPa)", ".1f"),
        ("shear_modulus_GPa", "G (GPa)", ".1f"),
        ("youngs_modulus_GPa", "E (GPa)", ".1f"),
        ("poisson_ratio", "ν", ".3f"),
        ("thermal_conductivity_W_mK", "κ (W/m·K)", ".1f"),
        ("thermal_expansion_1e6_K", "CTE (×10⁻⁶/K)", ".1f"),
        ("melting_point_K", "Tm (K)", ".0f"),
        ("hardness_GPa", "Hv (GPa)", ".2f"),
        ("pugh_ratio", "K/G", ".2f"),
    ]

    # Header
    print(f"\n{'═' * 90}")
    print(f"  Alloy Properties Comparison")
    print(f"{'═' * 90}")
    print(f"\n  {'Property':25s}", end="")
    for name in estimators:
        print(f" {name:>12s}", end="")
    print()
    print(f"  {'─' * 25}", end="")
    for _ in estimators:
        print(f" {'─' * 12}", end="")
    print()

    import math
    for key, label, fmt in properties:
        print(f"  {label:25s}", end="")
        for name in estimators:
            val = all_props[name].get(key, float("nan"))
            if math.isnan(val):
                print(f" {'N/A':>12s}", end="")
            else:
                print(f" {val:>12{fmt}}", end="")
        print()

    print(f"\n{'═' * 90}")


if __name__ == "__main__":
    main()
