"""
Structure Enumerator (Phase 5)

Wraps pymatgen base classes to provide derivative structure generation logic.
This implementation serves as a robust fallback for 'dsenum', which requires
C++ compilation not available in all Windows environments.

Features:
- Combinatorial substitution of species
- Automatic deduplication of symmetrically equivalent structures
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from numbers import Integral, Real
from typing import TypeAlias

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import DummySpecies, Structure

logger = logging.getLogger(__name__)
_MAX_VARIANTS = 5000
_TRUNCATED_VARIANTS = 500

SpeciesOption: TypeAlias = str | DummySpecies


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


class StructureEnumerator:
    def __init__(self, base_structure: Structure):
        """
        Initialize with a base structure.
        """
        if not isinstance(base_structure, Structure):
            raise TypeError("base_structure must be a pymatgen Structure")
        self.base_structure = base_structure

    @staticmethod
    def _normalize_substitutions(
        substitutions: dict[str, list[str | DummySpecies]]
    ) -> dict[str, list[SpeciesOption]]:
        if not isinstance(substitutions, dict):
            raise TypeError("substitutions must be a dictionary")
        normalized: dict[str, list[SpeciesOption]] = {}
        seen_by_key: dict[str, set[str]] = {}
        for key, values in substitutions.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                raise ValueError("substitution keys must be non-empty")
            if not isinstance(values, (list, tuple)) or len(values) == 0:
                raise ValueError(f"substitutions[{key!r}] must be a non-empty list/tuple")

            deduped = normalized.setdefault(normalized_key, [])
            seen = seen_by_key.setdefault(normalized_key, set())
            for option in values:
                if isinstance(option, DummySpecies):
                    token = str(option)
                    candidate = option
                else:
                    token = str(option).strip()
                    candidate = token
                if not token:
                    continue
                if token in seen:
                    continue
                seen.add(token)
                deduped.append(candidate)
        for key, deduped in normalized.items():
            if not deduped:
                raise ValueError(f"substitutions[{key!r}] must contain at least one non-empty replacement")
        return normalized

    @staticmethod
    def _select_variant_ordinals(total_variants: int, limit: int) -> list[int]:
        total = int(total_variants)
        bound = int(limit)
        if total <= 0 or bound <= 0:
            return []
        if total <= bound:
            return list(range(total))
        if bound == 1:
            return [0]
        # Deterministic near-uniform coverage over [0, total-1] that includes both ends.
        ordinals: list[int] = []
        for idx in range(bound):
            target = int(round((idx * (total - 1)) / (bound - 1)))
            if ordinals and target <= ordinals[-1]:
                target = ordinals[-1] + 1
            remaining_slots = bound - idx - 1
            max_allowed = total - remaining_slots - 1
            target = min(target, max_allowed)
            ordinals.append(target)
        return ordinals

    @staticmethod
    def _variant_space(option_lengths: list[int]) -> int:
        if not option_lengths:
            return 0
        space = 1
        for base in option_lengths:
            base_i = int(base)
            if base_i <= 0:
                raise ValueError("radices must be positive integers")
            space *= base_i
        return space

    @staticmethod
    def _decode_variant_ordinal(ordinal: int, radices: list[int]) -> list[int]:
        value = int(ordinal)
        max_ordinal = StructureEnumerator._variant_space(radices)
        if value < 0 or value >= max_ordinal:
            raise ValueError(f"ordinal out of range [0, {max_ordinal - 1}]: {ordinal!r}")

        digits = [0] * len(radices)
        for idx in range(len(radices) - 1, -1, -1):
            base = int(radices[idx])
            if base <= 0:
                raise ValueError("radices must be positive integers")
            digits[idx] = value % base
            value //= base
        return digits

    def generate(
        self,
        substitutions: dict[str, list[str | DummySpecies]],
        max_index: int = 1,  # Not fully used in fallback
        remove_superperiodic: bool = True,  # Handled by unique check
        remove_incomplete: bool = True,
    ) -> list[Structure]:
        """
        Generate derivative structures using combinatorial substitution.

        Args:
            substitutions: Map of species to allowed replacements.
                e.g. {"Ti": ["Ti", "Zr"], "O": ["O", "F"]}
            max_index: Ignored in fallback (supports 1x1 only effectively).

        Returns:
            List of unique pymatgen Structures.
        """
        max_index_i = _coerce_positive_int(max_index, name="max_index")
        substitutions = self._normalize_substitutions(substitutions)
        if max_index_i > 1:
            logger.warning(
                "fallback enumerator currently approximates only index-1 enumeration (got max_index=%s)",
                max_index_i,
            )
        logger.info("Generating derivatives for %s...", self.base_structure.composition.reduced_formula)

        # 1. Identify sites to vary
        sites_to_vary = []  # List of (site_index, [specie_options])

        for i, site in enumerate(self.base_structure):
            s_str = str(site.specie)
            if s_str in substitutions:
                sites_to_vary.append((i, substitutions[s_str]))

        if not sites_to_vary:
            return [self.base_structure.copy()]

        # 2. Check Combinatorial Space
        option_lengths = [len(opts) for _, opts in sites_to_vary]
        n_variants = self._variant_space(option_lengths)

        if n_variants > _MAX_VARIANTS:
            logger.warning("Combinatorial space too large (%s). Capping at %s.", n_variants, _TRUNCATED_VARIANTS)
            limit = _TRUNCATED_VARIANTS
        else:
            limit = n_variants

        # 3. Generate variants
        generated: list[Structure] = []

        # Prepare iterators
        all_options = [opts for _, opts in sites_to_vary]
        indices = [i for i, _ in sites_to_vary]
        sampled_ordinals = self._select_variant_ordinals(n_variants, limit)
        for ordinal in sampled_ordinals:
            digit_indices = self._decode_variant_ordinal(ordinal, option_lengths)

            s = self.base_structure.copy()
            for site_idx, option_idx, options in zip(indices, digit_indices, all_options, strict=True):
                s.replace(site_idx, options[option_idx])
            generated.append(s)

        if remove_incomplete:
            generated = [
                s for s in generated
                if all(not isinstance(site.specie, DummySpecies) for site in s)
            ]

        if not generated:
            return []
        if not remove_superperiodic:
            logger.info(
                "Skipping superperiodic duplicate filtering because remove_superperiodic=False"
            )
            return generated
        logger.info("Generated %s candidate structures.", len(generated))

        # 4. Filter Unique (StructureMatcher)
        # This removes rotationally/translationally equivalent structures
        matcher = StructureMatcher()

        # Group structures
        # This is O(N^2) generally.
        if len(generated) > 200:
            logger.info("Filtering duplicates (this may take a moment)...")

        by_formula: dict[str, list[Structure]] = defaultdict(list)
        for structure in generated:
            by_formula[structure.composition.reduced_formula].append(structure)

        unique: list[Structure] = []
        for formula in sorted(by_formula):
            structures = by_formula[formula]
            if len(structures) == 1:
                unique.append(structures[0])
                continue
            groups = matcher.group_structures(structures)
            unique.extend(g[0] for g in groups)

        logger.info("Found %s unique structures.", len(unique))
        return unique

    def _build_constraints(
        self, substitutions: dict[str, list[str | DummySpecies]]
    ) -> dict[str, int]:
        normalized = self._normalize_substitutions(substitutions)
        sites_to_vary = sum(1 for site in self.base_structure if str(site.specie) in normalized)
        option_lengths = [len(normalized[str(site.specie)]) for site in self.base_structure if str(site.specie) in normalized]
        n_variants = self._variant_space(option_lengths)
        return {
            "sites_to_vary": sites_to_vary,
            "variant_space": int(n_variants),
            "max_variants": _MAX_VARIANTS,
        }
