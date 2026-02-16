"""
Structure Generator for Materials Discovery

Generates novel crystal structure candidates by:
1. Element substitution — replace atoms with chemically similar ones
2. Composition perturbation — vary stoichiometry
3. Lattice strain — explore phase stability under deformation
4. Prototype decoration — fill known structure prototypes with new elements

The generator takes known topological materials as seeds and mutates
them to explore nearby chemical space, guided by domain knowledge
about which element combinations are likely to produce topological behavior.
"""

import numpy as np
from typing import Optional
from copy import deepcopy

from pymatgen.core import Structure, Lattice, Element


# Element substitution rules based on chemical similarity
# Key: original element, Values: substitution candidates
SUBSTITUTION_MAP = {
    # Chalcogenides (Group 16)
    "S":  ["Se", "Te"],
    "Se": ["S", "Te"],
    "Te": ["Se", "S"],
    # Pnictogens (Group 15) — key for TIs
    "Bi": ["Sb", "As"],
    "Sb": ["Bi", "As"],
    "As": ["Sb", "P"],
    # Post-transition metals
    "Pb": ["Sn", "Ge"],
    "Sn": ["Pb", "Ge", "Si"],
    "Ge": ["Sn", "Si"],
    # Transition metals (Weyl semimetal relevant)
    "Ta": ["Nb", "V"],
    "Nb": ["Ta", "V"],
    "W":  ["Mo", "Cr"],
    "Mo": ["W", "Cr"],
    # Alkali / alkaline earth
    "Na": ["K", "Li"],
    "K":  ["Na", "Rb", "Cs"],
    "Ca": ["Sr", "Ba"],
    "Sr": ["Ca", "Ba"],
    "Ba": ["Sr", "Ca"],
    # Rare earth
    "La": ["Ce", "Y"],
    "Ce": ["La", "Pr"],
    "Y":  ["La", "Sc"],
    # Halogens
    "Cl": ["Br", "I"],
    "Br": ["Cl", "I"],
    "I":  ["Br", "Cl"],
    # Group 14
    "Si": ["Ge", "Sn"],
    "C":  ["Si"],
}

# Heavy elements with strong SOC — these are topological-friendly
TOPO_FRIENDLY_ELEMENTS = {
    "Bi", "Sb", "Pb", "Sn", "Te", "Se", "Hg", "Tl",
    "Ta", "Nb", "W", "Mo", "Ir", "Pt", "Au",
    "In", "Cd", "Hf", "Zr",
}

# Known topological structure prototypes (space group → typical formula)
TOPO_PROTOTYPES = {
    166: ["A2B3"],       # Bi2Se3-type (rhombohedral, R-3m)
    225: ["AB"],         # NaCl-type (Fm-3m), e.g. SnTe
    109: ["AB"],         # TaAs-type (I4_1md)
    194: ["A3B"],        # Na3Bi-type (P6_3/mmc)
    137: ["A3B2"],       # Cd3As2-type (P4_2/nmc)
    129: ["ABC"],        # ZrSiS-type (P4/nmm)
    216: ["AB"],         # Zincblende (F-43m)
    31:  ["AB2"],        # WTe2-type (Pmn2_1)
    187: ["ABC2"],       # PbTaSe2-type (P-6m2)
}


class StructureGenerator:
    """
    Generates novel crystal structure candidates from seed structures.
    
    Strategy:
        1. Start from known topological materials
        2. Apply mutations (substitution, strain, mixing)
        3. Filter for chemical validity
        4. Score by "topological likelihood" heuristic
    """

    def __init__(self, seed_structures: list[Structure] = None, rng_seed: int = 42):
        self.seeds = seed_structures or []
        self.rng = np.random.RandomState(rng_seed)
        self.generated = []
        self.history = []  # track what mutations were applied

    def add_seeds(self, structures: list[Structure]):
        """Add seed structures for mutation."""
        self.seeds.extend(structures)

    def generate_batch(
        self,
        n_candidates: int = 50,
        methods: list[str] = None,
    ) -> list[dict]:
        """
        Generate a batch of candidate structures.

        Args:
            n_candidates: number of candidates to generate
            methods: list of methods to use, from:
                ["substitute", "strain", "mix", "decorate"]

        Returns:
            List of dicts with keys: structure, method, parent, mutations
        """
        if not self.seeds:
            raise ValueError("No seed structures! Call add_seeds() first.")

        methods = methods or ["substitute", "strain", "mix"]
        candidates = []

        per_method = n_candidates // len(methods)

        for method in methods:
            for _ in range(per_method):
                try:
                    if method == "substitute":
                        cand = self._substitute(self.rng.choice(self.seeds))
                    elif method == "strain":
                        cand = self._strain(self.rng.choice(self.seeds))
                    elif method == "mix":
                        if len(self.seeds) >= 2:
                            idx = self.rng.choice(len(self.seeds), 2, replace=False)
                            cand = self._mix(self.seeds[idx[0]], self.seeds[idx[1]])
                        else:
                            cand = self._substitute(self.seeds[0])
                    elif method == "decorate":
                        cand = self._decorate_prototype()
                    else:
                        continue

                    if cand is not None:
                        candidates.append(cand)
                except Exception:
                    continue

        # Fill remaining with substitutions
        while len(candidates) < n_candidates:
            try:
                cand = self._substitute(self.rng.choice(self.seeds))
                if cand is not None:
                    candidates.append(cand)
            except Exception:
                break

        self.generated.extend(candidates)
        return candidates[:n_candidates]

    def _substitute(self, parent: Structure) -> Optional[dict]:
        """
        Element substitution: replace one element with a chemically similar one.
        Preserves crystal symmetry and lattice parameters.
        """
        struct = parent.copy()
        elements = list(set(str(s.specie) for s in struct))

        # Pick a random element to substitute
        self.rng.shuffle(elements)
        for elem in elements:
            if elem in SUBSTITUTION_MAP:
                subs = SUBSTITUTION_MAP[elem]
                new_elem = subs[self.rng.randint(len(subs))]

                # Replace all sites of this element
                new_struct = struct.copy()
                for i, site in enumerate(new_struct):
                    if str(site.specie) == elem:
                        new_struct.replace(i, new_elem)

                return {
                    "structure": new_struct,
                    "method": "substitute",
                    "parent": parent.formula,
                    "mutations": f"{elem} → {new_elem}",
                    "topo_score": self._heuristic_topo_score(new_struct),
                }

        return None

    def _strain(self, parent: Structure, max_strain: float = 0.05) -> dict:
        """
        Lattice strain: apply random strain tensor.
        Explores how lattice deformation affects topology.
        """
        struct = parent.copy()
        lattice = struct.lattice

        # Random strain tensor (symmetric, small)
        strain = self.rng.uniform(-max_strain, max_strain, (3, 3))
        strain = (strain + strain.T) / 2  # symmetrize
        np.fill_diagonal(strain, strain.diagonal() + 1.0)

        new_matrix = lattice.matrix @ strain
        new_lattice = Lattice(new_matrix)

        new_struct = Structure(
            new_lattice,
            [str(s.specie) for s in struct],
            struct.frac_coords,
        )

        return {
            "structure": new_struct,
            "method": "strain",
            "parent": parent.formula,
            "mutations": f"strain={max_strain:.3f}",
            "topo_score": self._heuristic_topo_score(new_struct),
        }

    def _mix(self, parent_a: Structure, parent_b: Structure) -> Optional[dict]:
        """
        Alchemical mixing: take structure from A, elements from B.
        Only works if A and B have the same number of unique elements.
        """
        elems_a = sorted(set(str(s.specie) for s in parent_a))
        elems_b = sorted(set(str(s.specie) for s in parent_b))

        if len(elems_a) != len(elems_b):
            return None

        # Create mapping: elements of A → elements of B
        mapping = dict(zip(elems_a, self.rng.permutation(elems_b)))

        new_struct = parent_a.copy()
        for i, site in enumerate(new_struct):
            old_elem = str(site.specie)
            if old_elem in mapping:
                new_struct.replace(i, mapping[old_elem])

        return {
            "structure": new_struct,
            "method": "mix",
            "parent": f"{parent_a.formula} × {parent_b.formula}",
            "mutations": str(mapping),
            "topo_score": self._heuristic_topo_score(new_struct),
        }

    def _decorate_prototype(self) -> Optional[dict]:
        """
        Prototype decoration: pick a known topological structure prototype
        and fill it with random topo-friendly elements.
        """
        # Pick a random seed as the structural template
        parent = self.rng.choice(self.seeds)
        struct = parent.copy()

        elements = list(set(str(s.specie) for s in struct))

        # Replace each element with a random topo-friendly one
        topo_elems = list(TOPO_FRIENDLY_ELEMENTS)
        self.rng.shuffle(topo_elems)

        mapping = {}
        used = set()
        for elem in elements:
            for te in topo_elems:
                if te not in used and te != elem:
                    mapping[elem] = te
                    used.add(te)
                    break

        if len(mapping) < len(elements):
            return None

        new_struct = struct.copy()
        for i, site in enumerate(new_struct):
            old_elem = str(site.specie)
            if old_elem in mapping:
                new_struct.replace(i, mapping[old_elem])

        return {
            "structure": new_struct,
            "method": "decorate",
            "parent": parent.formula,
            "mutations": str(mapping),
            "topo_score": self._heuristic_topo_score(new_struct),
        }

    def _heuristic_topo_score(self, structure: Structure) -> float:
        """
        Heuristic score for topological likelihood.
        Higher = more likely to be topological.

        Based on:
        - Heavy element content (strong SOC)
        - Known topological space groups
        - Element combinations seen in known TIs/TSMs
        """
        score = 0.0
        elements = set(str(s.specie) for s in structure)

        # Heavy element bonus
        for elem in elements:
            try:
                z = Element(elem).Z
                if z >= 50:
                    score += 0.3 * (z / 83.0)  # normalize by Bi
                if z >= 70:
                    score += 0.2
            except Exception:
                pass

        # Topo-friendly element bonus
        n_topo_elem = len(elements & TOPO_FRIENDLY_ELEMENTS)
        score += 0.2 * n_topo_elem

        # Known topological space group bonus
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(structure)
            sg = sga.get_space_group_number()
            if sg in TOPO_PROTOTYPES:
                score += 0.3
        except Exception:
            pass

        # Small band gap heuristic: narrow-gap → more likely topological
        # (can't compute here, but structures with Bi/Te/Pb tend to be narrow-gap)
        if {"Bi", "Te"} <= elements or {"Bi", "Se"} <= elements:
            score += 0.3
        if {"Pb", "Sn"} <= elements or {"Sn", "Te"} <= elements:
            score += 0.2
        if {"Ta", "As"} <= elements or {"Nb", "As"} <= elements:
            score += 0.25

        return min(score, 1.0)

    def get_top_candidates(self, n: int = 10) -> list[dict]:
        """Get the top-N candidates by heuristic topological score."""
        sorted_cands = sorted(
            self.generated, key=lambda x: x["topo_score"], reverse=True
        )
        return sorted_cands[:n]
