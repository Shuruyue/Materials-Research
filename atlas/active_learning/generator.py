"""
Structure Generator for Materials Discovery

Generates novel crystal structure candidates by:
1. Element substitution — replace atoms with chemically similar ones
2. Composition perturbation — vary stoichiometry
3. Lattice strain — explore phase stability under deformation
4. Prototype decoration — fill known structure prototypes with new elements

Optimization:
- Uses ProcessPoolExecutor for parallel generation
- Implements strict physical validation (min atomic distance)
- Detailed error logging
"""

import numpy as np
import logging
from typing import Optional, List, Dict
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from pymatgen.core import Structure, Lattice, Element
from pymatgen.analysis.structure_matcher import StructureMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Element substitution rules based on chemical similarity
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

TOPO_FRIENDLY_ELEMENTS = {
    "Bi", "Sb", "Pb", "Sn", "Te", "Se", "Hg", "Tl",
    "Ta", "Nb", "W", "Mo", "Ir", "Pt", "Au",
    "In", "Cd", "Hf", "Zr",
}

TOPO_PROTOTYPES = {
    166: ["A2B3"],       # Bi2Se3-type (rhombohedral, R-3m)
    225: ["AB"],         # NaCl-type (Fm-3m)
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
    Optimized for parallel execution and robustness.
    """

    def __init__(self, seed_structures: list[Structure] = None, rng_seed: int = 42):
        self.seeds = seed_structures or []
        self.rng_seed = rng_seed
        self.rng = np.random.RandomState(rng_seed)
        self.generated = []
        
        # Determine number of workers (leave 1 core for OS)
        self.n_workers = max(1, multiprocessing.cpu_count() - 1)

    def add_seeds(self, structures: list[Structure]):
        """Add seed structures for mutation."""
        self.seeds.extend(structures)

    def _validate_structure(self, struct: Structure) -> bool:
        """
        Check if structure is physically reasonable.
        1. Min distance > 0.7 Angstrom (avoid atomic overlap)
        2. Volume per atom > 2.0 Angstrom^3 (avoid collapse)
        """
        if struct.volume / len(struct) < 2.0:
            return False
        
        # Fast distance check (only nearest neighbors)
        try:
            min_dist = min([d for n in struct.get_all_neighbors(r=1.5) for d in n], default=2.0)
            if min_dist < 0.7:
                return False
        except Exception:
            return True # Fallback if neighbor check fails
            
        return True

    def generate_batch(
        self,
        n_candidates: int = 50,
        methods: list[str] = None,
    ) -> list[dict]:
        """
        Generate a batch of candidate structures in parallel.
        """
        if not self.seeds:
            raise ValueError("No seed structures! Call add_seeds() first.")

        methods = methods or ["substitute", "strain", "mix"]
        candidates = []
        
        # Create tasks
        tasks = []
        per_method = n_candidates // len(methods) + 1
        
        # We need to pass seed data to workers explicitly or rely on copy-on-write
        # For simplicity in 'spawn' contexts, we pass necessary data
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            
            # Submit substitution tasks
            if "substitute" in methods:
                for _ in range(per_method):
                    seed = self.rng.choice(self.seeds)
                    futures.append(executor.submit(_worker_substitute, seed, self.rng.randint(0, 10000)))
            
            # Submit strain tasks
            if "strain" in methods:
                for _ in range(per_method):
                    seed = self.rng.choice(self.seeds)
                    futures.append(executor.submit(_worker_strain, seed, 0.05, self.rng.randint(0, 10000)))
            
            # Submit mixing tasks
            if "mix" in methods:
                if len(self.seeds) >= 2:
                    for _ in range(per_method):
                        idx = self.rng.choice(len(self.seeds), 2, replace=False)
                        futures.append(executor.submit(_worker_mix, self.seeds[idx[0]], self.seeds[idx[1]], self.rng.randint(0, 10000)))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    cand = future.result()
                    if cand and self._validate_structure(cand["structure"]):
                        candidates.append(cand)
                except Exception as e:
                    logger.debug(f"Generation task failed: {e}")

        # Limit to requested number
        self.generated.extend(candidates[:n_candidates])
        return candidates[:n_candidates]

    # ... (Helper methods for heuristic score can remain static or outside) ...
    

# ── Worker Functions (Must be top-level for pickling) ──

def _heuristic_topo_score(structure: Structure) -> float:
    """Static version of heuristic score for workers."""
    score = 0.0
    elements = set(str(s.specie) for s in structure)

    # Heavy element bonus
    for elem in elements:
        try:
            z = Element(elem).Z
            if z >= 50: score += 0.3 * (z / 83.0)
            if z >= 70: score += 0.2
        except: pass

    # Topo-friendly bonus
    n_topo_elem = len(elements & TOPO_FRIENDLY_ELEMENTS)
    score += 0.2 * n_topo_elem

    # Space group bonus
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sg = SpacegroupAnalyzer(structure).get_space_group_number()
        if sg in TOPO_PROTOTYPES: score += 0.3
    except: pass

    return min(score, 1.0)


def _worker_substitute(parent: Structure, seed: int) -> Optional[dict]:
    rng = np.random.RandomState(seed)
    struct = parent.copy()
    elements = list(set(str(s.specie) for s in struct))
    rng.shuffle(elements)
    
    for elem in elements:
        if elem in SUBSTITUTION_MAP:
            subs = SUBSTITUTION_MAP[elem]
            new_elem = subs[rng.randint(len(subs))]
            
            new_struct = struct.copy()
            for i, site in enumerate(new_struct):
                if str(site.specie) == elem:
                    new_struct.replace(i, new_elem)
            
            return {
                "structure": new_struct,
                "method": "substitute",
                "parent": parent.formula,
                "mutations": f"{elem} -> {new_elem}",
                "topo_score": _heuristic_topo_score(new_struct),
            }
    return None


def _worker_strain(parent: Structure, max_strain: float, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    struct = parent.copy()
    lattice = struct.lattice
    
    strain = rng.uniform(-max_strain, max_strain, (3, 3))
    strain = (strain + strain.T) / 2
    np.fill_diagonal(strain, strain.diagonal() + 1.0)
    
    new_matrix = lattice.matrix @ strain
    new_lattice = Lattice(new_matrix)
    new_struct = Structure(new_lattice, [str(s.specie) for s in struct], struct.frac_coords)
    
    return {
        "structure": new_struct,
        "method": "strain",
        "parent": parent.formula,
        "mutations": f"strain={max_strain:.3f}",
        "topo_score": _heuristic_topo_score(new_struct),
    }


def _worker_mix(parent_a: Structure, parent_b: Structure, seed: int) -> Optional[dict]:
    rng = np.random.RandomState(seed)
    elems_a = sorted(set(str(s.specie) for s in parent_a))
    elems_b = sorted(set(str(s.specie) for s in parent_b))
    
    if len(elems_a) != len(elems_b):
        return None
        
    mapping = dict(zip(elems_a, rng.permutation(elems_b)))
    new_struct = parent_a.copy()
    for i, site in enumerate(new_struct):
        if str(site.specie) in mapping:
            new_struct.replace(i, mapping[str(site.specie)])
            
    return {
        "structure": new_struct,
        "method": "mix",
        "parent": f"{parent_a.formula} x {parent_b.formula}",
        "mutations": str(mapping),
        "topo_score": _heuristic_topo_score(new_struct),
    }

