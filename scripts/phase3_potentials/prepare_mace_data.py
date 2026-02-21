#!/usr/bin/env python3
"""
Prepare MACE Training Data

Extracts crystal structures from JARVIS-DFT and converts them
to the format required by MACE training (extended XYZ format).

MACE needs: atomic positions, lattice, energies, forces, stresses.
JARVIS provides pre-computed DFT energies and structures.

Usage:
    python scripts/phase3_potentials/prepare_mace_data.py                  # Default: Si,Ge,Sn
    python scripts/phase3_potentials/prepare_mace_data.py --elements Bi Se Te
    python scripts/phase3_potentials/prepare_mace_data.py --max 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atlas.config import get_config
from atlas.data.jarvis_client import JARVISClient


def jarvis_to_ase_atoms(atoms_dict: dict, energy: float = None):
    """Convert a JARVIS atoms dict to an ASE Atoms object with energy info."""
    from jarvis.core.atoms import Atoms as JAtoms
    from ase import Atoms as AseAtoms

    jatoms = JAtoms.from_dict(atoms_dict)

    ase_atoms = AseAtoms(
        symbols=jatoms.elements,
        positions=jatoms.cart_coords,
        cell=jatoms.lattice_mat,
        pbc=True,
    )

    # Store energy as info (MACE reads from info dict)
    if energy is not None and not np.isnan(energy):
        n = len(ase_atoms)
        ase_atoms.info["REF_energy"] = energy * n  # total energy (JARVIS gives per-atom)
        ase_atoms.info["config_type"] = "Default"

        # Zero forces placeholder — JARVIS doesn't provide forces directly,
        # but MACE can train on energy-only data
        from ase.calculators.singlepoint import SinglePointCalculator
        forces = np.zeros((n, 3))
        calc = SinglePointCalculator(ase_atoms, energy=energy * n, forces=forces)
        ase_atoms.calc = calc

    return ase_atoms


def filter_by_elements(df, target_elements: list[str]):
    """Filter DataFrame to materials containing ONLY the target elements."""
    from jarvis.core.atoms import Atoms as JAtoms

    def check(row):
        try:
            atoms = JAtoms.from_dict(row["atoms"])
            mat_elements = set(atoms.elements)
            return mat_elements.issubset(set(target_elements))
        except Exception:
            return False

    mask = df.apply(check, axis=1)
    return df[mask].copy()


def main():
    parser = argparse.ArgumentParser(description="Prepare MACE training data")
    parser.add_argument(
        "--elements", nargs="+", default=["Si", "Ge", "Sn"],
        help="Target element set (default: Si Ge Sn)"
    )
    parser.add_argument(
        "--max", type=int, default=2000,
        help="Max number of structures (default: 2000)"
    )
    parser.add_argument(
        "--ehull", type=float, default=0.3,
        help="Max energy above hull in eV/atom (default: 0.3)"
    )
    args = parser.parse_args()

    cfg = get_config()
    print(cfg.summary())

    client = JARVISClient()

    # Load all materials
    print(f"\n=== Preparing MACE Training Data ===")
    print(f"  Target elements: {args.elements}")
    print(f"  Max structures:  {args.max}")

    df = client.get_stable_materials(ehull_max=args.ehull)

    # Filter by elements
    print(f"\n  Filtering for element set: {args.elements}")
    filtered = filter_by_elements(df, args.elements)
    print(f"  Found {len(filtered)} structures with only {args.elements}")

    if len(filtered) == 0:
        print("  ⚠ No structures found! Try a broader element set or higher ehull.")
        return

    # Limit
    if len(filtered) > args.max:
        filtered = filtered.sample(n=args.max, random_state=42)
        print(f"  Sampled {args.max} structures")

    # Convert to ASE Atoms and write extended XYZ
    from ase.io import write as ase_write

    output_dir = cfg.paths.processed_dir / "mace_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_atoms = []
    skipped = 0

    for idx, (_, row) in enumerate(filtered.iterrows()):
        try:
            energy = row.get("optb88vdw_total_energy", None)
            if energy is None or (isinstance(energy, float) and np.isnan(energy)):
                energy = row.get("formation_energy_peratom", None)

            atoms = jarvis_to_ase_atoms(row["atoms"], energy=energy)
            all_atoms.append(atoms)
        except Exception as e:
            skipped += 1

    # Split: 80% train, 10% val, 10% test
    n = len(all_atoms)
    np.random.seed(42)
    indices = np.random.permutation(n)

    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_atoms = [all_atoms[i] for i in indices[:n_train]]
    val_atoms = [all_atoms[i] for i in indices[n_train:n_train + n_val]]
    test_atoms = [all_atoms[i] for i in indices[n_train + n_val:]]

    # Write extended XYZ files
    train_file = output_dir / "train.xyz"
    val_file = output_dir / "val.xyz"
    test_file = output_dir / "test.xyz"

    ase_write(str(train_file), train_atoms, format="extxyz")
    ase_write(str(val_file), val_atoms, format="extxyz")
    ase_write(str(test_file), test_atoms, format="extxyz")

    elem_str = "_".join(sorted(args.elements))

    print(f"\n=== Results ===")
    print(f"  Elements:     {args.elements}")
    print(f"  Total valid:  {n} (skipped {skipped})")
    print(f"  Train:        {len(train_atoms)} → {train_file}")
    print(f"  Validation:   {len(val_atoms)} → {val_file}")
    print(f"  Test:         {len(test_atoms)} → {test_file}")
    print(f"\n✓ MACE training data ready!")


if __name__ == "__main__":
    main()
