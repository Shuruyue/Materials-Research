"""
Test Structure Enumerator (Pymatgen Fallback)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Fallback: import directly from structure_enumerator.py in current dir
try:
    from scripts.phase5_active_learning.structure_enumerator import StructureEnumerator
except ImportError:
    from structure_enumerator import StructureEnumerator

from pymatgen.core import DummySpecies, Lattice, Structure


def get_perovskite_structure():
    lattice = Lattice.cubic(3.945)
    species = ["Sr", "Ti", "O", "O", "O"]
    coords = [
        [0, 0, 0],
        [0.5, 0.5, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]
    return Structure(lattice, species, coords)

def main():
    print("Initializing Base Structure (SrTiO3)...")
    base = get_perovskite_structure()
    print(base)

    enumerator = StructureEnumerator(base)

    print("\n--- Test 1: Simple Substitution (Ti -> Ti, Zr) ---")
    subs = {"Ti": ["Ti", "Zr"]}
    structures = enumerator.generate(subs)

    print(f"Generated {len(structures)} unique structures.")
    for i, s in enumerate(structures):
        print(f"  {i+1}: {s.composition.reduced_formula}")

    print("\n--- Test 2: Oxygen Vacancy (O -> O, Vacancy) ---")
    # For vacancy, use DummySpecies "X" or explicit removal?
    # Pymatgen SubstitutionTransformation works with DummySpecies.
    subs_vac = {"O": ["O", DummySpecies("X")]}

    structures_vac = enumerator.generate(subs_vac)
    print(f"Generated {len(structures_vac)} unique structures (with vacancies/dummies).")
    for i, s in enumerate(structures_vac):
        print(f"  {i+1}: {s.composition.reduced_formula}")

if __name__ == "__main__":
    main()
