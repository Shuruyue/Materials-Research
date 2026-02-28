"""
Structure Enumerator (Phase 5)

Wraps pymatgen base classes to provide derivative structure generation logic.
This implementation serves as a robust fallback for 'dsenum', which requires
C++ compilation not available in all Windows environments.

Features:
- Combinatorial substitution of species
- Automatic deduplication of symmetrically equivalent structures
"""

import itertools
import logging

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

logger = logging.getLogger(__name__)

class StructureEnumerator:
    def __init__(self, base_structure: Structure):
        """
        Initialize with a base structure.
        """
        self.base_structure = base_structure

    def generate(
        self,
        substitutions: dict[str, list[str]],
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
        logger.info(f"Generating derivatives for {self.base_structure.composition.reduced_formula}...")

        # 1. Identify sites to vary
        sites_to_vary = [] # List of (site_index, [specie_options])

        for i, site in enumerate(self.base_structure):
            s_str = str(site.specie)
            if s_str in substitutions:
                # User provided options.
                # Ensure options are valid Species/Elements matches
                sites_to_vary.append((i, substitutions[s_str]))
            else:
                # Keep original (fixed)
                pass

        if not sites_to_vary:
            return [self.base_structure.copy()]

        # 2. Check Combinatorial Space
        n_variants = 1
        for _, opts in sites_to_vary:
            n_variants *= len(opts)

        if n_variants > 5000:
            logger.warning(f"Combinatorial space too large ({n_variants}). Capping at 500.")
            limit = 500
        else:
            limit = 5000

        # 3. Generate variants
        generated = []

        # Prepare iterators
        all_options = [opts for _, opts in sites_to_vary]
        indices = [i for i, _ in sites_to_vary]

        count = 0
        for combination in itertools.product(*all_options):
            if count >= limit: break

            s = self.base_structure.copy()
            for idx, new_sp in zip(indices, combination):
                s.replace(idx, new_sp)
            generated.append(s)
            count += 1

        logger.info(f"Generated {len(generated)} candidate structures.")

        # 4. Filter Unique (StructureMatcher)
        # This removes rotationally/translationally equivalent structures
        matcher = StructureMatcher()

        # Group structures
        # This is O(N^2) generally.
        if len(generated) > 200:
             logger.info("Filtering duplicates (this may take a moment)...")

        groups = matcher.group_structures(generated)
        unique = [g[0] for g in groups]

        logger.info(f"Found {len(unique)} unique structures.")
        return unique

    def _build_constraints(self, substitutions):
        pass
