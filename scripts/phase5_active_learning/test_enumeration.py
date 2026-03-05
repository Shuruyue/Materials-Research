"""Demo script for validating the StructureEnumerator fallback path."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from pymatgen.core import DummySpecies, Lattice, Structure


def _load_local_enumerator_class() -> type:
    """Load sibling `structure_enumerator.py` without mutating sys.path."""
    module_path = Path(__file__).resolve().with_name("structure_enumerator.py")
    spec = importlib.util.spec_from_file_location("atlas_phase5_structure_enumerator", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load StructureEnumerator fallback from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    enumerator_cls = getattr(module, "StructureEnumerator", None)
    if not callable(enumerator_cls):
        raise TypeError("fallback module must define callable StructureEnumerator")
    return enumerator_cls


def _resolve_enumerator_class() -> type:
    """Import StructureEnumerator from project package, with local fallback."""
    try:
        from scripts.phase5_active_learning.structure_enumerator import StructureEnumerator as Enumerator

        return Enumerator
    except ModuleNotFoundError as exc:  # pragma: no cover - local execution fallback path
        if exc.name not in {
            "scripts",
            "scripts.phase5_active_learning",
            "scripts.phase5_active_learning.structure_enumerator",
        }:
            raise
        return _load_local_enumerator_class()


def _coerce_non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        number_f = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if not number_f.is_integer():
        raise ValueError(f"{name} must be a non-negative integer")
    number = int(number_f)
    if number < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return number


def get_perovskite_structure() -> Structure:
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


def _validate_generated_structures(payload: Any, *, label: str) -> list[Structure]:
    if not isinstance(payload, (list, tuple)):
        raise TypeError(f"{label} generate() result must be a list/tuple of Structure")
    validated: list[Structure] = []
    for idx, structure in enumerate(payload):
        if not isinstance(structure, Structure):
            raise TypeError(f"{label} generate() result[{idx}] must be a pymatgen Structure")
        validated.append(structure)
    return validated


def run_demo(
    *,
    enumerator_cls: type | None = None,
    include_vacancies: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run substitution demo and return summary counts for tests/automation."""
    cls = enumerator_cls or _resolve_enumerator_class()
    if not callable(cls):
        raise TypeError("enumerator_cls must be a class-like callable")
    base = get_perovskite_structure()
    enumerator = cls(base)
    if not callable(getattr(enumerator, "generate", None)):
        raise TypeError("enumerator_cls(base) must provide a callable generate(...) method")

    if verbose:
        print("Initializing Base Structure (SrTiO3)...")
        print(base)
        print("\n--- Test 1: Simple Substitution (Ti -> Ti, Zr) ---")
    subs = {"Ti": ["Ti", "Zr"]}
    structures = _validate_generated_structures(
        enumerator.generate(subs),
        label="simple_substitution",
    )

    if verbose:
        print(f"Generated {len(structures)} unique structures.")
        for i, structure in enumerate(structures):
            print(f"  {i + 1}: {structure.composition.reduced_formula}")

    structures_vac = []
    if include_vacancies:
        if verbose:
            print("\n--- Test 2: Oxygen Vacancy (O -> O, Vacancy) ---")
        subs_vac = {"O": ["O", DummySpecies("X")]}
        structures_vac = _validate_generated_structures(
            enumerator.generate(subs_vac, remove_incomplete=False),
            label="vacancy_substitution",
        )
        if verbose:
            print(f"Generated {len(structures_vac)} unique structures (with vacancies/dummies).")
            for i, structure in enumerate(structures_vac):
                print(f"  {i + 1}: {structure.composition.reduced_formula}")

    summary = {
        "base_formula": base.composition.reduced_formula,
        "simple_substitution_count": len(structures),
        "vacancy_substitution_count": len(structures_vac),
    }
    if not isinstance(summary["base_formula"], str) or not summary["base_formula"].strip():
        raise ValueError("base formula must be a non-empty string")
    summary["simple_substitution_count"] = _coerce_non_negative_int(
        summary["simple_substitution_count"],
        name="simple_substitution_count",
    )
    summary["vacancy_substitution_count"] = _coerce_non_negative_int(
        summary["vacancy_substitution_count"],
        name="vacancy_substitution_count",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run StructureEnumerator fallback demo.")
    parser.add_argument("--skip-vacancy", action="store_true", help="Skip the vacancy-substitution demonstration.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose structure prints.")
    parser.add_argument("--json", action="store_true", help="Print summary as JSON object.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run_demo(
            include_vacancies=not args.skip_vacancy,
            # JSON mode must remain machine-parseable without extra text.
            verbose=not (args.quiet or args.json),
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if args.quiet or args.json:
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        else:
            print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
