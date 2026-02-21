#!/usr/bin/env python3
"""
Script 09: CALPHAD Phase Diagram Calculator

Computes precise equilibrium phase diagrams, solidification paths,
and thermodynamic properties for solder alloys using CALPHAD.

Uses pycalphad with the Sn-Ag-Cu thermodynamic database.

Available alloys: SAC305, SAC405, SAC105, SN100C, SNAG36

Usage:
    python scripts/phase6_analysis/phase_diagram.py                       # SAC305 equilibrium table
    python scripts/phase6_analysis/phase_diagram.py --alloy SAC405        # Different alloy
    python scripts/phase6_analysis/phase_diagram.py --solidify             # Solidification curve
    python scripts/phase6_analysis/phase_diagram.py --plot                 # Plot solidification
    python scripts/phase6_analysis/phase_diagram.py --binary SN AG         # Binary phase diagram
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(description="CALPHAD Phase Diagram Calculator")
    parser.add_argument("--alloy", type=str, default="SAC305",
                        help="Alloy name (default: SAC305)")
    parser.add_argument("--solidify", action="store_true",
                        help="Compute solidification path")
    parser.add_argument("--plot", action="store_true",
                        help="Plot solidification curve")
    parser.add_argument("--binary", nargs=2, metavar=("ELEM1", "ELEM2"),
                        help="Plot binary phase diagram")
    parser.add_argument("--T-start", type=float, default=550,
                        help="Start temperature K (default: 550)")
    parser.add_argument("--T-end", type=float, default=400,
                        help="End temperature K (default: 400)")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of temperature steps (default: 30)")
    args = parser.parse_args()

    from atlas.thermo.calphad import CalphadCalculator

    print("  Loading Sn-Ag-Cu thermodynamic database...")
    calc = CalphadCalculator.sn_ag_cu()
    print(f"  Database loaded: {len(calc.all_phases)} phases")
    print(f"  Available phases: {', '.join(calc.all_phases)}")

    if args.binary:
        # Binary phase diagram
        e1, e2 = args.binary
        print(f"\n  Computing {e1}-{e2} binary phase diagram...")
        calc.plot_binary(e1, e2, T_range=(373, 773))
        return

    if args.solidify or args.plot:
        # Solidification path
        print(f"\n  Computing solidification path for {args.alloy}...")
        result = calc.solidification_path(
            args.alloy,
            T_start=args.T_start,
            T_end=args.T_end,
            n_steps=args.steps,
        )

        print(f"\n  ┌─────────────────────────────────────┐")
        print(f"  │ Solidification Results: {args.alloy:12s} │")
        print(f"  ├─────────────────────────────────────┤")
        print(f"  │ Liquidus: {result.liquidus_K:.1f} K ({result.liquidus_K - 273.15:.1f}°C)     │")
        print(f"  │ Solidus:  {result.solidus_K:.1f} K ({result.solidus_K - 273.15:.1f}°C)     │")
        print(f"  │ ΔT:       {result.liquidus_K - result.solidus_K:.1f} K                   │")
        print(f"  └─────────────────────────────────────┘")

        # Print phase fraction table
        print(f"\n  {'T(K)':>6} {'T(°C)':>6}", end="")
        for phase in result.solid_phases:
            print(f" {phase:>10}", end="")
        print()
        print(f"  {'─' * 6} {'─' * 6}", end="")
        for _ in result.solid_phases:
            print(f" {'─' * 10}", end="")
        print()

        for i, T in enumerate(result.temperatures):
            T_C = T - 273.15
            print(f"  {T:6.0f} {T_C:6.1f}", end="")
            for phase in result.solid_phases:
                f = result.solid_phases[phase][i]
                if f > 0.001:
                    print(f" {f:10.3f}", end="")
                else:
                    print(f" {'—':>10}", end="")
            print()

        if args.plot:
            save_dir = Path("data/calphad_results")
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{args.alloy}_solidification.png"
            calc.plot_solidification(result, save_path=save_path)

        return

    # Default: equilibrium table
    calc.print_equilibrium_table(
        args.alloy,
        T_range=(args.T_end, args.T_start),
        step=5,
    )


if __name__ == "__main__":
    main()
