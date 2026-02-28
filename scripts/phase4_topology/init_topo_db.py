#!/usr/bin/env python3
"""
Initialize Topological Materials Database

Creates and populates the local topological materials database
with known/confirmed topological materials as seed data.

Usage:
    python scripts/phase4_topology/init_topo_db.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atlas.config import get_config
from atlas.data.topo_db import TopoDB


def main():
    cfg = get_config()
    print(cfg.summary())

    print("\n=== Initializing Topological Materials Database ===\n")

    db = TopoDB()

    # Load experimentally confirmed topological materials
    db.load_seed_data()

    # Print statistics
    stats = db.stats()
    print("\nDatabase Statistics:")
    print(f"  Total materials: {stats['total']}")
    print("\n  By topological class:")
    for cls, count in stats["by_class"].items():
        print(f"    {cls:8s}: {count}")

    # Show some queries
    print("\n=== Sample Queries ===")

    ti = db.query(topo_class="TI")
    print(f"\n  Topological Insulators ({len(ti)}):")
    for _, row in ti.iterrows():
        print(f"    {row['jid']:12s}  {row['formula']:15s}  "
              f"Eg={row['band_gap']:.3f} eV")

    tsm = db.query(topo_class="TSM")
    print(f"\n  Topological Semimetals ({len(tsm)}):")
    for _, row in tsm.iterrows():
        print(f"    {row['jid']:12s}  {row['formula']:15s}")

    print(f"\n[OK] Database initialized at: {db.db_file}")


if __name__ == "__main__":
    main()
