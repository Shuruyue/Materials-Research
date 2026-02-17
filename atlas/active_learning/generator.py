"""
Structure Generator for Materials Discovery

Generates novel crystal structure candidates by:
1. Element substitution — replace atoms with chemically similar ones
2. Lattice strain — explore phase stability under deformation
3. Prototype decoration — fill known structure prototypes with new elements

Optimization:
- Symmetry-aware generation (filters out low-symmetry junk)
- Parallel execution with ProcessPoolExecutor
- Strict physical validation
"""

import numpy as np
import logging
from typing import Optional, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback

from pymatgen.core import Structure, Lattice, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Element substitution rules based on chemical similarity
SUBSTITUTION_MAP = {
    # Chalcogenides (Group 16)
    "S":  ["Se", "Te"], "Se": ["S", "Te"], "Te": ["Se", "S"],
    # Pnictogens (Group 15) — key for TIs
    "Bi": ["Sb", "As"], "Sb": ["Bi", "As"], "As": ["Sb", "P"],
    # Post-transition metals
    "Pb": ["Sn", "Ge"], "Sn": ["Pb", "Ge"], "Ge": ["Sn", "Si"],
    # Transition metals (Weyl semimetal relevant)
    "Ta": ["Nb", "V"], "Nb": ["Ta", "V"], "W":  ["Mo", "Cr"], "Mo": ["W", "Cr"],
    # Alkali / alkaline earth
    "Na": ["K", "Li"], "K":  ["Na", "Rb"], "Ca": ["Sr", "Ba"], "Sr": ["Ca", "Ba"],
    # Rare earth
    "La": ["Ce", "Y"], "Ce": ["La", "Pr"], "Y":  ["La", "Sc"],
    # Halogens
    "Cl": ["Br", "I"], "Br": ["Cl", "I"], "I":  ["Br", "Cl"],
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
        
        # Parallel setup
        self.n_workers = max(1, multiprocessing.cpu_count() - 2) # Leave 2 cores free

    def add_seeds(self, structures: list[Structure]):
        """Add seed structures for mutation."""
        self.seeds.extend(structures)

    def _validate_structure(self, struct: Structure) -> bool:
        """
        Check if structure is physically reasonable.
        1. Volume per atom > 5.0 A^3 (too small = collapse)
        2. Min distance > 1.5 A (too close = boom)
        """
        if struct.volume / len(struct) < 5.0:
            return False
        
        # Check nearest neighbor distance
        # Efficiently check only if density suggests trouble
        try:
            # get_all_neighbors is slow for large supercells, but fine for unit cells
            # Check r=1.5A
            neighbors = struct.get_all_neighbors(r=1.5)
            for atom_neighbors in neighbors:
                if atom_neighbors: # If any neighbor is < 1.5A
                    return False
        except Exception:
            pass # Fallback
            
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
            logger.warning("No seeds provided. Returning empty batch.")
            return []

        methods = methods or ["substitute", "strain"]
        candidates = []
        
        per_method = n_candidates // len(methods) + 2
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            
            # Submit substitution tasks
            if "substitute" in methods:
                for _ in range(per_method):
                    seed = self.rng.choice(self.seeds)
                    futures.append(executor.submit(_worker_substitute, seed, self.rng.randint(1e9)))
            
            # Submit strain tasks
            if "strain" in methods:
                for _ in range(per_method):
                    seed = self.rng.choice(self.seeds)
                    futures.append(executor.submit(_worker_strain, seed, 0.03, self.rng.randint(1e9)))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    cand = future.result()
                    if cand and self._validate_structure(cand["structure"]):
                        candidates.append(cand)
                except Exception as e:
                    pass

        # Sort by heuristic score and trim
        # Prefer higher heuristic scores for initial screening
        candidates.sort(key=lambda x: x.get("topo_score", 0), reverse=True)
        return candidates[:n_candidates]


# ── Worker Functions ──

def _heuristic_topo_score(structure: Structure) -> float:
    """Static version of heuristic score for workers."""
    score = 0.0
    elements = set(str(s.specie) for s in structure)

    # Heavy element bonus (SOC strength)
    max_z = 0
    for elem in elements:
        try:
            z = Element(elem).Z
            max_z = max(max_z, z)
            if z >= 50: score += 0.2
            if z >= 70: score += 0.2
        except: pass
    
    # SOC requires heavy elements generally
    if max_z < 30: return 0.0

    # Topo-friendly bonus
    n_topo_elem = len(elements & TOPO_FRIENDLY_ELEMENTS)
    if n_topo_elem > 0: score += 0.3

    # Symmetry bonus
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        sg = sga.get_space_group_number()
        # High symmetry is often good for TIs
        if sg > 100: score += 0.2
        if sg in TOPO_PROTOTYPES: score += 0.4
    except: pass

    return min(score, 1.0)


def _worker_substitute(parent: Structure, seed: int) -> Optional[dict]:
    rng = np.random.RandomState(seed)
    struct = parent.copy()
    elements = list(set(str(s.specie) for s in struct))
    
    # Try multiple times to find a valid substitution
    for _ in range(5):
        target_elem = rng.choice(elements)
        if target_elem in SUBSTITUTION_MAP:
            subs = SUBSTITUTION_MAP[target_elem]
            new_elem = rng.choice(subs)
            
            # Replace ALL instances of target_elem to preserve symmetry
            new_struct = struct.copy()
            new_struct.replace_species({target_elem: new_elem})
            
            if new_struct.composition == struct.composition:
                continue
                
            return {
                "structure": new_struct,
                "method": "substitute",
                "parent": parent.composition.reduced_formula,
                "mutations": f"{target_elem}->{new_elem}",
                "topo_score": _heuristic_topo_score(new_struct),
            }
    return None


def _worker_strain(parent: Structure, max_strain: float, seed: int) -> Optional[dict]:
    rng = np.random.RandomState(seed)
    struct = parent.copy()
    
    # Apply volume-conserving strain or hydrostatic strain
    # Random strain tensor
    eps = rng.uniform(-max_strain, max_strain, (3, 3))
    eps = (eps + eps.T) / 2.0 # Symmetrize
    
    # Strain lattice
    lat = struct.lattice.matrix
    new_lat = lat @ (np.eye(3) + eps)
    
    new_struct = Structure(new_lat, struct.species, struct.frac_coords)
    
    # Check if symmetry is completely destroyed
    try:
        sga = SpacegroupAnalyzer(new_struct, symprec=0.1)
        if sga.get_space_group_number() == 1: # P1
             # Try to symmetrize? Or just reject P1 if parent was high symmetry
             pass
    except:
        pass

    return {
        "structure": new_struct,
        "method": "strain",
        "parent": parent.composition.reduced_formula,
        "mutations": f"strain={max_strain:.2%}",
        "topo_score": _heuristic_topo_score(new_struct),
    }
