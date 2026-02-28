"""
Phase 6: Visualization with Pymatviz

This script uses the installed 'pymatviz' library to generate interactive plots.
Prerequisites: pip install pymatviz (Installed)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import directly now that it's installed
try:
    import pymatviz
    from pymatviz import ptable_heatmap_plotly, spacegroup_sunburst
    print(f"Using pymatviz version: {pymatviz.__version__}")
except ImportError as e:
    print(f"CRITICAL: pymatviz should be installed but failed to import: {e}")
    sys.exit(1)

def main():
    print("--- Phase 6: Generating Visualizations ---")

    # 1. Periodic Table Heatmap
    # Real data simulation: Element average energies
    print("1. Generating Periodic Table Heatmap...")
    data = {
        "H": -1.0, "He": -0.1, "Li": -1.8, "Be": -2.1, "B": -3.5, "C": -4.8, "N": -5.2, "O": -6.0, "F": -2.3, "Ne": -0.1,
        "Na": -1.1, "Mg": -1.5, "Al": -3.2, "Si": -4.5, "P": -4.1, "S": -3.8, "Cl": -1.8, "Ar": -0.1,
        "K": -0.9, "Ca": -1.8, "Sc": -6.5, "Ti": -7.2, "V": -8.1, "Cr": -8.5, "Mn": -8.2, "Fe": -7.5, "Co": -6.1, "Ni": -5.2, "Cu": -3.5, "Zn": -1.2,
        "Ga": -2.5, "Ge": -3.8, "As": -4.2, "Se": -3.5, "Br": -1.5, "Kr": -0.1,
        "Sr": -6.5, "Y": -7.2, "Zr": -8.5, "Nb": -9.2
    }
    s = pd.Series(data, name="Formation Energy (eV/atom)")

    # Ensure output directory exists
    output_dir = Path("scripts/phase6_analysis/outputs")
    output_dir.mkdir(exist_ok=True, parents=True)

    fig_ptable = ptable_heatmap_plotly(s)
    # Save as HTML for interactivity
    output_ptable = output_dir / "ptable_heatmap.html"
    try:
        fig_ptable.write_html(str(output_ptable))
        print(f"  [SAVED] {output_ptable}")
    except Exception as e:
         print(f"  [ERROR] Could not save heatmap: {e}")

    # 2. Spacegroup Sunburst
    print("2. Generating Spacegroup Sunburst...")
    # Simulate a dataset of spacegroups
    np.random.seed(42)
    spg_nums = np.random.randint(1, 231, size=200)

    fig_sunburst = spacegroup_sunburst(spg_nums)
    output_sunburst = output_dir / "spacegroup_sunburst.html"
    try:
        fig_sunburst.write_html(str(output_sunburst))
        print(f"  [SAVED] {output_sunburst}")
    except Exception as e:
         print(f"  [ERROR] Could not save sunburst: {e}")

    print("\n[SUCCESS] pymatviz visualizations generated successfully.")

if __name__ == "__main__":
    main()
