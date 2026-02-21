#!/usr/bin/env python3
"""
Script 01: Download JARVIS-DFT Data

Downloads the JARVIS-DFT database (~76,000 materials) and prepares
training data for the ATLAS platform.

NO API KEY REQUIRED — data is freely downloadable from NIST/Figshare.

Usage:
    python scripts/phase1_baseline/download_data.py              # Full download + stats
    python scripts/phase1_baseline/download_data.py --stats       # Show database statistics only
    python scripts/phase1_baseline/download_data.py --topo        # Show topological materials
"""

import argparse
import sys
from pathlib import Path

# Enhance module discovery
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from atlas.config import get_config
    from atlas.data.jarvis_client import JARVISClient
except ImportError as e:
    print(f"Error: Could not import atlas package. ({e})")
    print("Please install the package in editable mode: pip install -e .")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download JARVIS-DFT data")
    parser.add_argument("--stats", action="store_true", help="Show stats only")
    parser.add_argument("--topo", action="store_true", help="Show topological materials")
    parser.add_argument("--heavy", action="store_true", help="Show heavy-element materials")
    args = parser.parse_args()

    cfg = get_config()
    print(cfg.summary())

    client = JARVISClient()

    # Download (or load cached) database
    print("\n=== Loading JARVIS-DFT Database ===\n")
    df = client.load_dft_3d()
    print(f"\n  Total materials loaded: {len(df)}")

    # Show statistics
    print("\n=== Database Statistics ===\n")
    stats = client.stats()
    for key, val in stats.items():
        print(f"  {key:30s}: {val}")

    if args.topo:
        print("\n=== Topological Candidates ===\n")
        topo = client.get_topological_materials()
        print(f"\n  Sample topological materials:")
        cols = ["jid", "optb88vdw_bandgap"]
        if "spillage" in topo.columns:
            cols.append("spillage")
        print(topo[cols].head(20).to_string(index=False))

    if args.heavy:
        print("\n=== Heavy Element Materials ===\n")
        heavy = client.get_heavy_element_materials()
        print(f"\n  Sample heavy-element materials:")
        print(heavy[["jid", "optb88vdw_bandgap"]].head(20).to_string(index=False))

    if not args.stats and not args.topo and not args.heavy:
        # Default: prepare training data
        print("\n=== Preparing Training Data ===\n")
        data = client.get_training_data(n_topo=500, n_trivial=500)

        output_dir = cfg.paths.processed_dir
        data["topo"].to_pickle(output_dir / "topo_materials.pkl")
        data["trivial"].to_pickle(output_dir / "trivial_materials.pkl")
        print(f"\n  Saved to: {output_dir}")

    print("\n[OK] Done!")


if __name__ == "__main__":
    main()
