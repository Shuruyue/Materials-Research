"""
Structure Generator for Materials Discovery.

Algorithmic upgrades in this implementation:
1) Probabilistic substitution kernel using element-feature geometry
   instead of uniform lookup-table replacement.
2) Log-Euclidean lattice perturbation for strain proposals:
   a symmetric log-strain tensor S is mapped by exp(S), yielding
   an SPD deformation with controlled volume behavior.
3) Multi-objective candidate scoring and diversity-aware greedy selection:
   utility = weighted_score - lambda * local_penalty.
4) Polymorph-aware novelty via composition + symmetry + lattice fingerprints.
5) Adaptive objective weighting and online substitution-kernel learning.

References:
- Ionic substitution statistics:
  Hautier et al., Inorg. Chem. (2011), https://doi.org/10.1021/ic102031h
- Composition constraints / SMACT:
  Davies et al., Digital Discovery (2022), https://doi.org/10.1039/D2DD00028H
- Log-Euclidean calculus on SPD manifolds:
  Arsigny et al. (2007), https://doi.org/10.1002/mrm.20965
- Local penalization for batch BO:
  Gonzalez et al. (2016), https://arxiv.org/abs/1505.08052
- Bandit/UCB style exploration:
  Auer et al. (2002), https://link.springer.com/article/10.1023/A:1013689704352
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from numbers import Integral, Real

import numpy as np
from pymatgen.core import Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

logger = logging.getLogger(__name__)


# Element substitution rules based on chemical similarity.
SUBSTITUTION_MAP = {
    # Chalcogenides (Group 16)
    "S": ["Se", "Te"],
    "Se": ["S", "Te"],
    "Te": ["Se", "S"],
    # Pnictogens (Group 15)
    "Bi": ["Sb", "As"],
    "Sb": ["Bi", "As"],
    "As": ["Sb", "P"],
    # Post-transition metals
    "Pb": ["Sn", "Ge"],
    "Sn": ["Pb", "Ge"],
    "Ge": ["Sn", "Si"],
    # Transition metals
    "Ta": ["Nb", "V"],
    "Nb": ["Ta", "V"],
    "W": ["Mo", "Cr"],
    "Mo": ["W", "Cr"],
    # Alkali / alkaline earth
    "Na": ["K", "Li"],
    "K": ["Na", "Rb"],
    "Ca": ["Sr", "Ba"],
    "Sr": ["Ca", "Ba"],
    # Rare earth
    "La": ["Ce", "Y"],
    "Ce": ["La", "Pr"],
    "Y": ["La", "Sc"],
    # Halogens
    "Cl": ["Br", "I"],
    "Br": ["Cl", "I"],
    "I": ["Br", "Cl"],
}

TOPO_FRIENDLY_ELEMENTS = {
    "Bi",
    "Sb",
    "Pb",
    "Sn",
    "Te",
    "Se",
    "Hg",
    "Tl",
    "Ta",
    "Nb",
    "W",
    "Mo",
    "Ir",
    "Pt",
    "Au",
    "In",
    "Cd",
    "Hf",
    "Zr",
}

TOPO_PROTOTYPES = {
    166: ["A2B3"],  # Bi2Se3-type (R-3m)
    225: ["AB"],  # NaCl-type (Fm-3m)
    109: ["AB"],  # TaAs-type (I41md)
    194: ["A3B"],  # Na3Bi-type (P63/mmc)
    137: ["A3B2"],  # Cd3As2-type (P42/nmc)
    129: ["ABC"],  # ZrSiS-type (P4/nmm)
    216: ["AB"],  # Zincblende (F-43m)
    31: ["AB2"],  # WTe2-type (Pmn21)
    187: ["ABC2"],  # PbTaSe2-type (P-6m2)
}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return np.zeros(0, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.ones_like(x, dtype=float) / float(x.size)
    floor = np.min(x[finite]) if np.any(finite) else 0.0
    x = np.where(finite, x, floor)
    x = x - np.max(x)
    # Guard exp overflow/underflow in pathological logits.
    ex = np.exp(np.clip(x, -700.0, 700.0))
    ex = np.nan_to_num(ex, nan=0.0, posinf=0.0, neginf=0.0)
    denom = float(np.sum(ex))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.ones_like(ex) / max(len(ex), 1)
    return ex / denom


def _stable_seed_from_text(text: str) -> int:
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _species_symbol(specie) -> str:
    if hasattr(specie, "symbol"):
        return str(specie.symbol)
    text = str(specie)
    letters = "".join(ch for ch in text if ch.isalpha())
    if not letters:
        return text
    if len(letters) >= 2 and letters[1].islower():
        return letters[0].upper() + letters[1].lower()
    return letters[0].upper()


def _structure_symbols(structure: Structure) -> list[str]:
    symbols = []
    for site in structure:
        sym = _species_symbol(site.specie)
        if sym not in symbols:
            symbols.append(sym)
    return symbols


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _coerce_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(default)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        out_f = float(value)
        if not np.isfinite(out_f) or not out_f.is_integer():
            return int(default)
        return int(out_f)
    try:
        return int(value)
    except Exception:
        return int(default)


def _coerce_float(value: object, default: float) -> float:
    return _safe_float(value, default)


def _element_feature_vector(symbol: str) -> np.ndarray:
    elem = Element(symbol)
    z = float(elem.Z)
    group = _safe_float(getattr(elem, "group", 0.0), 0.0)
    row = _safe_float(getattr(elem, "row", 0.0), 0.0)
    x = _safe_float(getattr(elem, "X", 0.0), 0.0)
    radius = _safe_float(getattr(elem, "atomic_radius", 0.0), 0.0)
    if radius <= 0.0:
        radius = _safe_float(getattr(elem, "atomic_radius_calculated", 1.5), 1.5)
    ox_states = list(getattr(elem, "common_oxidation_states", ()) or ())
    if not ox_states:
        ox_states = list(getattr(elem, "oxidation_states", ()) or ())
    max_ox = float(max((abs(_safe_float(v, 0.0)) for v in ox_states), default=0.0))
    topo_flag = 1.0 if symbol in TOPO_FRIENDLY_ELEMENTS else 0.0
    # Normalized feature vector for distance-based substitution kernels.
    return np.array(
        [
            z / 100.0,
            group / 18.0,
            row / 7.0,
            min(max(x, 0.0), 4.0) / 4.0,
            min(max(radius, 0.0), 3.5) / 3.5,
            min(max(max_ox, 0.0), 7.0) / 7.0,
            topo_flag,
        ],
        dtype=float,
    )


def _periodic_neighbor_substitutions(symbol: str) -> list[str]:
    try:
        e = Element(symbol)
    except Exception:
        return []

    candidates = []
    target_group = _safe_float(getattr(e, "group", 0.0), 0.0)
    target_row = _safe_float(getattr(e, "row", 0.0), 0.0)
    for z in range(1, 119):
        other = Element.from_Z(z)
        if other.symbol == symbol:
            continue
        group = _safe_float(getattr(other, "group", 0.0), 0.0)
        row = _safe_float(getattr(other, "row", 0.0), 0.0)
        if target_group > 0 and group == target_group and abs(row - target_row) <= 1:
            candidates.append(other.symbol)

    if candidates:
        return candidates

    z = int(e.Z)
    fallback = []
    for dz in (-2, -1, 1, 2):
        zz = z + dz
        if 1 <= zz <= 118:
            fallback.append(Element.from_Z(zz).symbol)
    return fallback


def _candidate_substitutions(symbol: str) -> list[str]:
    cands = list(SUBSTITUTION_MAP.get(symbol, []))
    if not cands:
        cands = _periodic_neighbor_substitutions(symbol)
    # Uniquify while preserving order.
    out = []
    seen = set()
    for c in cands:
        if c == symbol:
            continue
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _weighted_substitution_choice(target: str, candidates: list[str], rng: np.random.RandomState, temperature: float) -> str:
    logits = np.asarray([_substitution_prior_logit(target, c, temperature) for c in candidates], dtype=float)
    probs = _softmax(np.asarray(logits, dtype=float))
    idx = int(rng.choice(len(candidates), p=probs))
    return candidates[idx]


def _substitution_prior_logit(target: str, candidate: str, temperature: float) -> float:
    target_vec = _element_feature_vector(target)
    c_vec = _element_feature_vector(candidate)
    weights = np.array([1.2, 1.0, 0.8, 0.9, 0.9, 0.7, 0.4], dtype=float)
    dist = float(np.linalg.norm(weights * (c_vec - target_vec)))
    topo_bonus = 0.35 if candidate in TOPO_FRIENDLY_ELEMENTS else 0.0
    temp = max(float(temperature), 1e-6)
    return float(-dist / temp + topo_bonus)


def _composition_fingerprint(structure: Structure) -> np.ndarray:
    vec = np.zeros(120, dtype=float)
    frac_comp = structure.composition.fractional_composition
    for sym, frac in frac_comp.get_el_amt_dict().items():
        try:
            z = int(Element(sym).Z)
        except Exception:
            continue
        if 0 <= z < len(vec):
            vec[z] = float(frac)
    return vec


def _lattice_feature_vector(structure: Structure) -> np.ndarray:
    lat = structure.lattice
    vpa = float(structure.volume / max(len(structure), 1))
    return np.array(
        [
            min(max(float(lat.a), 0.0), 25.0) / 25.0,
            min(max(float(lat.b), 0.0), 25.0) / 25.0,
            min(max(float(lat.c), 0.0), 35.0) / 35.0,
            float(lat.alpha) / 180.0,
            float(lat.beta) / 180.0,
            float(lat.gamma) / 180.0,
            min(max(vpa, 0.0), 60.0) / 60.0,
        ],
        dtype=float,
    )


def _symmetry_feature_vector(structure: Structure) -> np.ndarray:
    # Include SG number and crystal-system one-hot to separate polymorphs with same composition.
    out = np.zeros(8, dtype=float)
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        sg = int(sga.get_space_group_number())
        out[0] = min(max(float(sg), 0.0), 230.0) / 230.0
        crystal = str(sga.get_crystal_system()).strip().lower()
        systems = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]
        if crystal in systems:
            out[1 + systems.index(crystal)] = 1.0
    except Exception:
        pass
    return out


def _structure_fingerprint(structure: Structure) -> np.ndarray:
    comp = _composition_fingerprint(structure)
    lat = _lattice_feature_vector(structure)
    sym = _symmetry_feature_vector(structure)
    return np.concatenate([comp, lat, sym], axis=0)


def _charge_neutrality_score(
    structure: Structure,
    max_states_per_element: int = 3,
    max_combinations: int = 256,
) -> float:
    """
    Soft electrostatic feasibility prior using oxidation-state combinations.

    Returns exp(-|charge residual| / scale) in [0, 1].
    """
    comp = structure.composition.get_el_amt_dict()
    symbols = list(comp.keys())
    if not symbols:
        return 0.0

    amounts = np.asarray([float(comp[s]) for s in symbols], dtype=float)
    state_options = []
    for s in symbols:
        try:
            e = Element(s)
        except Exception:
            state_options.append([0])
            continue
        states = list(e.common_oxidation_states or ())
        if not states:
            states = list(e.oxidation_states or ())
        if not states:
            states = [0]
        uniq = sorted({int(round(_safe_float(v, 0.0))) for v in states}, key=lambda v: (abs(v), v))
        state_options.append(uniq[: max(1, int(max_states_per_element))])

    combo_count = int(np.prod([len(opts) for opts in state_options]))
    best_residual = float("inf")
    if combo_count <= max_combinations:
        iterator = itertools.product(*state_options)
    else:
        # Bounded Monte Carlo on oxidation-state product space.
        signature = "|".join(f"{sym}:{amount:.8f}" for sym, amount in sorted(comp.items()))
        seed = _stable_seed_from_text(signature) % (2**32)
        rng = np.random.RandomState(int(seed))

        def _sample_iter():
            for _ in range(max_combinations):
                yield [opts[int(rng.randint(0, len(opts)))] for opts in state_options]

        iterator = _sample_iter()

    for charge_tuple in iterator:
        residual = abs(float(np.dot(amounts, np.asarray(charge_tuple, dtype=float))))
        if residual < best_residual:
            best_residual = residual
            if residual <= 1e-10:
                break

    scale = max(float(np.sum(np.abs(amounts))), 1e-8)
    return float(np.exp(-best_residual / scale))


def _electronegativity_compatibility_score(structure: Structure) -> float:
    comp = structure.composition.fractional_composition.get_el_amt_dict()
    if not comp:
        return 0.0
    elems = list(comp.keys())
    fracs = np.asarray([float(comp[e]) for e in elems], dtype=float)
    if len(elems) == 1:
        return 0.65

    xs = []
    has_pos = False
    has_neg = False
    for e in elems:
        try:
            el = Element(e)
        except Exception:
            xs.append(0.0)
            continue
        xs.append(_safe_float(getattr(el, "X", 0.0), 0.0))
        ox = list(el.common_oxidation_states or ()) or list(el.oxidation_states or ())
        if any(_safe_float(v, 0.0) > 0 for v in ox):
            has_pos = True
        if any(_safe_float(v, 0.0) < 0 for v in ox):
            has_neg = True

    # Pairwise electronegativity contrast proxy.
    contrast = 0.0
    weight_sum = 0.0
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            w = fracs[i] * fracs[j]
            contrast += w * abs(xs[i] - xs[j])
            weight_sum += w
    mean_contrast = contrast / max(weight_sum, 1e-12)

    # Broad bell around moderate ionic/covalent balance.
    center = 1.2
    spread = 0.9
    contrast_score = float(np.exp(-((mean_contrast - center) ** 2) / (2.0 * spread * spread)))
    sign_score = 1.0 if (has_pos and has_neg) else 0.35
    return float(np.clip(0.55 * contrast_score + 0.45 * sign_score, 0.0, 1.0))


def _smact_like_feasibility_score(structure: Structure) -> float:
    charge = _charge_neutrality_score(structure)
    en = _electronegativity_compatibility_score(structure)
    return float(np.clip(0.7 * charge + 0.3 * en, 0.0, 1.0))


def _novelty_against_archive(fp: np.ndarray, archive: list[np.ndarray], sigma: float) -> float:
    if not archive:
        return 1.0
    sigma = max(float(sigma), 1e-6)
    best_sim = 0.0
    for ref in archive:
        d2 = float(np.sum((fp - ref) ** 2))
        sim = float(np.exp(-d2 / (2.0 * sigma * sigma)))
        if sim > best_sim:
            best_sim = sim
    return float(np.clip(1.0 - best_sim, 0.0, 1.0))


def _heuristic_topo_score(structure: Structure) -> float:
    """Static topological prior used inside workers."""
    score = 0.0
    elements = set(_structure_symbols(structure))

    max_z = 0
    for elem in elements:
        try:
            z = Element(elem).Z
            max_z = max(max_z, int(z))
            if z >= 50:
                score += 0.2
            if z >= 70:
                score += 0.2
        except Exception:
            continue

    if max_z < 30:
        return 0.0

    n_topo_elem = len(elements & TOPO_FRIENDLY_ELEMENTS)
    if n_topo_elem > 0:
        score += 0.3

    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        sg = sga.get_space_group_number()
        if sg > 100:
            score += 0.2
        if sg in TOPO_PROTOTYPES:
            score += 0.4
    except Exception:
        pass

    return float(min(score, 1.0))


def _worker_substitute(parent: Structure, seed: int, substitution_temperature: float) -> dict | None:
    rng = np.random.RandomState(seed)
    struct = parent.copy()
    elements = _structure_symbols(struct)
    if not elements:
        return None

    for _ in range(6):
        target_elem = str(rng.choice(elements))
        candidates = _candidate_substitutions(target_elem)
        if not candidates:
            continue

        new_elem = _weighted_substitution_choice(target_elem, candidates, rng, substitution_temperature)
        if new_elem == target_elem:
            continue

        new_struct = struct.copy()
        new_struct.replace_species({target_elem: new_elem})
        if new_struct.composition.reduced_formula == struct.composition.reduced_formula:
            continue

        return {
            "structure": new_struct,
            "method": "substitute",
            "parent": parent.composition.reduced_formula,
            "mutations": f"{target_elem}->{new_elem}",
            "topo_score": _heuristic_topo_score(new_struct),
        }
    return None


def _worker_apply_substitution(parent: Structure, target_elem: str, new_elem: str) -> dict | None:
    struct = parent.copy()
    if target_elem == new_elem:
        return None
    try:
        new_struct = struct.copy()
        new_struct.replace_species({target_elem: new_elem})
    except Exception:
        return None

    if new_struct.composition.reduced_formula == struct.composition.reduced_formula:
        return None

    return {
        "structure": new_struct,
        "method": "substitute",
        "parent": parent.composition.reduced_formula,
        "mutations": f"{target_elem}->{new_elem}",
        "topo_score": _heuristic_topo_score(new_struct),
    }


def _worker_strain(parent: Structure, max_strain: float, seed: int) -> dict | None:
    """
    Generate strain via log-Euclidean SPD map.

    Construct symmetric traceless S, then F = exp(S).
    New lattice is L' = L @ F scaled by mild hydrostatic factor.
    """
    rng = np.random.RandomState(seed)
    struct = parent.copy()

    a = rng.normal(loc=0.0, scale=max_strain / 1.5, size=(3, 3))
    s = 0.5 * (a + a.T)
    s = s - np.trace(s) / 3.0 * np.eye(3)

    eigvals, eigvecs = np.linalg.eigh(s)
    stretch = eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T
    hydro = float(rng.uniform(-0.5 * max_strain, 0.5 * max_strain))
    stretch = stretch * np.exp(hydro)

    lat = struct.lattice.matrix
    new_lat = lat @ stretch
    if not np.isfinite(new_lat).all():
        return None
    if abs(np.linalg.det(new_lat)) < 1e-8:
        return None

    new_struct = Structure(new_lat, struct.species, struct.frac_coords)
    strain_norm = float(np.linalg.norm(s, ord="fro"))

    return {
        "structure": new_struct,
        "method": "strain",
        "parent": parent.composition.reduced_formula,
        "mutations": f"logstrain_fro={strain_norm:.4f};hydro={hydro:.4f}",
        "topo_score": _heuristic_topo_score(new_struct),
        "strain_norm_fro": strain_norm,
    }


class StructureGenerator:
    """
    Generates novel crystal structure candidates from seed structures.
    """

    def __init__(
        self,
        seed_structures: list[Structure] | None = None,
        rng_seed: int = 42,
        *,
        substitution_temperature: float = 0.35,
        max_strain: float = 0.03,
        w_topo: float = 0.50,
        w_novelty: float = 0.30,
        w_feasibility: float = 0.20,
        w_strain: float = 0.10,
        novelty_sigma: float = 0.20,
        diversity_lambda: float = 0.20,
        diversity_sigma: float = 0.22,
        strain_regularity_sigma: float = 0.08,
        archive_limit: int = 1024,
        adaptive_weight_power: float = 1.0,
        substitution_ucb_beta: float = 0.25,
        substitution_reward_weight: float = 1.0,
        substitution_stat_decay: float = 0.995,
    ):
        self.seeds = seed_structures or []
        self.rng_seed = _coerce_int(rng_seed, 42)
        self.rng = np.random.RandomState(self.rng_seed)
        self.generated: list[dict] = []

        self.substitution_temperature = float(max(_coerce_float(substitution_temperature, 0.35), 1e-6))
        self.max_strain = float(max(_coerce_float(max_strain, 0.03), 1e-6))
        self.weights = {
            "topo": float(max(_coerce_float(w_topo, 0.50), 0.0)),
            "novelty": float(max(_coerce_float(w_novelty, 0.30), 0.0)),
            "feasibility": float(max(_coerce_float(w_feasibility, 0.20), 0.0)),
            "strain": float(max(_coerce_float(w_strain, 0.10), 0.0)),
        }
        self.novelty_sigma = float(max(_coerce_float(novelty_sigma, 0.20), 1e-6))
        self.diversity_lambda = float(max(_coerce_float(diversity_lambda, 0.20), 0.0))
        self.diversity_sigma = float(max(_coerce_float(diversity_sigma, 0.22), 1e-6))
        self.strain_regularity_sigma = float(max(_coerce_float(strain_regularity_sigma, 0.08), 1e-6))
        self.archive_limit = max(32, _coerce_int(archive_limit, 1024))
        self.adaptive_weight_power = float(max(_coerce_float(adaptive_weight_power, 1.0), 0.0))
        self.substitution_ucb_beta = float(max(_coerce_float(substitution_ucb_beta, 0.25), 0.0))
        self.substitution_reward_weight = float(max(_coerce_float(substitution_reward_weight, 1.0), 0.0))
        decay = _coerce_float(substitution_stat_decay, 0.995)
        self.substitution_stat_decay = float(np.clip(decay, 0.90, 1.0))

        # Online substitution kernel learning cache.
        # key: (target_symbol, new_symbol) -> {"count": float, "reward_sum": float}
        self.substitution_stats: dict[tuple[str, str], dict[str, float]] = {}
        self.last_adaptive_weights: dict[str, float] = dict(self.weights)
        self._seed_fingerprints: list[np.ndarray] = []
        if self.seeds:
            for seed in self.seeds:
                try:
                    self._seed_fingerprints.append(_structure_fingerprint(seed))
                except Exception:
                    continue

        if os.name == "nt":
            self.n_workers = 1
        else:
            self.n_workers = max(1, multiprocessing.cpu_count() - 2)

    def add_seeds(self, structures: list[Structure]):
        """Add seed structures for mutation."""
        if not structures:
            return
        self.seeds.extend(structures)
        for struct in structures:
            try:
                self._seed_fingerprints.append(_structure_fingerprint(struct))
            except Exception:
                continue

    def _validate_structure(self, struct: Structure) -> bool:
        """Check if structure is physically reasonable."""
        if len(struct) <= 0:
            return False
        if struct.volume / len(struct) < 5.0:
            return False

        try:
            neighbors = struct.get_all_neighbors(r=1.5)
            for atom_neighbors in neighbors:
                if atom_neighbors:
                    return False
        except Exception:
            pass

        return True

    def _archive_fingerprints(self) -> list[np.ndarray]:
        fps = list(self._seed_fingerprints)
        generated_structs = [c.get("structure") for c in self.generated if c.get("structure") is not None]
        if generated_structs:
            for s in generated_structs[-self.archive_limit :]:
                fps.append(_structure_fingerprint(s))
        return fps[-self.archive_limit :]

    def _normalize_weights(self, d: dict[str, float]) -> dict[str, float]:
        cleaned: dict[str, float] = {}
        for key, value in d.items():
            v = _coerce_float(value, 0.0)
            cleaned[key] = max(v, 0.0)
        total = sum(cleaned.values())
        if total <= 0.0:
            k = max(len(d), 1)
            return {kk: 1.0 / k for kk in d}
        return {kk: cleaned.get(kk, 0.0) / total for kk in d}

    def _adaptive_weights(self, objective_matrix: dict[str, np.ndarray]) -> dict[str, float]:
        base = self._normalize_weights(self.weights)
        if self.adaptive_weight_power <= 0.0:
            self.last_adaptive_weights = dict(base)
            return base

        adjusted = {}
        for key, base_w in base.items():
            vals = np.asarray(objective_matrix.get(key, np.asarray([0.0], dtype=float)), dtype=float)
            dispersion = float(np.std(vals))
            info = max(dispersion, 1e-6) ** self.adaptive_weight_power
            adjusted[key] = base_w * info
        normalized = self._normalize_weights(adjusted)
        self.last_adaptive_weights = dict(normalized)
        return normalized

    def _sample_substitution_action(self, structure: Structure) -> tuple[str, str] | None:
        elements = _structure_symbols(structure)
        actions: list[tuple[str, str]] = []
        logits: list[float] = []

        total_count = sum(_coerce_float(v.get("count", 0.0), 0.0) for v in self.substitution_stats.values())
        total_count = max(float(total_count), 0.0)

        for target in elements:
            candidates = _candidate_substitutions(target)
            for new_elem in candidates:
                key = (target, new_elem)
                stat = self.substitution_stats.get(key, {"count": 0.0, "reward_sum": 0.0})
                count = float(max(_coerce_float(stat.get("count", 0.0), 0.0), 0.0))
                reward_sum = float(max(_coerce_float(stat.get("reward_sum", 0.0), 0.0), 0.0))
                mean_reward = (reward_sum / count) if count > 0 else 0.5

                prior = _substitution_prior_logit(target, new_elem, self.substitution_temperature)
                bonus = self.substitution_ucb_beta * np.sqrt(np.log(total_count + 2.0) / (count + 1.0))
                logit = prior + self.substitution_reward_weight * mean_reward + bonus
                actions.append((target, new_elem))
                logits.append(float(logit))

        if not actions:
            return None
        probs = _softmax(np.asarray(logits, dtype=float))
        idx = int(self.rng.choice(len(actions), p=probs))
        return actions[idx]

    def _update_substitution_stats(self, selected: list[dict]):
        decay = self.substitution_stat_decay
        for key in list(self.substitution_stats.keys()):
            base_count = _coerce_float(self.substitution_stats[key].get("count", 0.0), 0.0)
            base_reward = _coerce_float(self.substitution_stats[key].get("reward_sum", 0.0), 0.0)
            self.substitution_stats[key]["count"] = base_count * decay
            self.substitution_stats[key]["reward_sum"] = base_reward * decay

        for cand in selected:
            method = str(cand.get("method", ""))
            if method not in {"substitute", "mix"}:
                continue
            mutation = str(cand.get("mutations", ""))
            if "->" not in mutation:
                continue
            pair = mutation.split("->", 1)
            if len(pair) != 2:
                continue
            target = pair[0].strip()
            new_elem = pair[1].strip().split(";")[0].strip()
            if not target or not new_elem:
                continue
            key = (target, new_elem)
            stat = self.substitution_stats.setdefault(key, {"count": 0.0, "reward_sum": 0.0})
            raw_reward = cand.get("generator_score", cand.get("topo_score", 0.0))
            reward = float(np.clip(_safe_float(raw_reward, 0.0), 0.0, 1.0))
            stat["count"] += 1.0
            stat["reward_sum"] += reward

    def _rank_candidates(self, candidates: list[dict], n_select: int) -> list[dict]:
        if not candidates:
            return []

        archive = self._archive_fingerprints()
        objective = {"topo": [], "novelty": [], "feasibility": [], "strain": []}
        raw_items = []
        for cand in candidates:
            struct = cand["structure"]
            fp = _structure_fingerprint(struct)
            topo = float(cand.get("topo_score", 0.0))
            novelty = _novelty_against_archive(fp, archive, self.novelty_sigma)
            feasibility = _smact_like_feasibility_score(struct)
            strain_norm = float(cand.get("strain_norm_fro", 0.0))
            strain_reg = float(np.exp(-(strain_norm**2) / (2.0 * self.strain_regularity_sigma**2)))

            raw_items.append((cand, fp, topo, novelty, feasibility, strain_reg))
            objective["topo"].append(topo)
            objective["novelty"].append(novelty)
            objective["feasibility"].append(feasibility)
            objective["strain"].append(strain_reg)

        objective_np = {k: np.asarray(v, dtype=float) for k, v in objective.items()}
        adapted = self._adaptive_weights(objective_np)
        for cand, fp, topo, novelty, feasibility, strain_reg in raw_items:
            score = (
                adapted["topo"] * topo
                + adapted["novelty"] * novelty
                + adapted["feasibility"] * feasibility
                + adapted["strain"] * strain_reg
            )
            cand["novelty_score"] = novelty
            cand["feasibility_score"] = feasibility
            cand["strain_regular_score"] = strain_reg
            cand["generator_score"] = float(score)
            cand["adaptive_weights"] = dict(adapted)
            cand["_fp"] = fp

        selected = []
        selected_fps: list[np.ndarray] = []
        remaining = list(candidates)
        sigma = self.diversity_sigma

        while remaining and len(selected) < int(n_select):
            best_idx = 0
            best_utility = -float("inf")
            best_penalty = 0.0
            best_base = -float("inf")
            for i, cand in enumerate(remaining):
                fp = cand["_fp"]
                base = float(cand["generator_score"])
                if selected_fps:
                    min_d2 = min(float(np.sum((fp - sfp) ** 2)) for sfp in selected_fps)
                    penalty = float(np.exp(-min_d2 / (2.0 * sigma * sigma)))
                else:
                    penalty = 0.0

                utility = base - self.diversity_lambda * penalty
                if utility > best_utility or (utility == best_utility and base > best_base):
                    best_idx = i
                    best_utility = utility
                    best_penalty = penalty
                    best_base = base

            chosen = remaining.pop(best_idx)
            chosen["diversity_penalty"] = float(best_penalty)
            chosen["selection_utility"] = float(best_utility)
            selected_fps.append(chosen["_fp"])
            selected.append(chosen)

        for cand in selected:
            cand.pop("_fp", None)

        self._update_substitution_stats(selected)
        return selected

    def generate_batch(
        self,
        n_candidates: int = 50,
        methods: list[str] | None = None,
    ) -> list[dict]:
        """Generate a batch of candidate structures in parallel."""
        if not self.seeds:
            raise ValueError("No seed structures provided.")

        methods = methods or ["substitute", "strain"]
        candidates = []
        per_method = n_candidates // max(len(methods), 1) + 2

        tasks: list[tuple[Callable[..., dict | None], tuple]] = []

        if "substitute" in methods:
            for _ in range(per_method):
                seed = self.seeds[int(self.rng.randint(0, len(self.seeds)))]
                action = self._sample_substitution_action(seed)
                if action is None:
                    tasks.append(
                        (
                            _worker_substitute,
                            (seed, int(self.rng.randint(1e9)), self.substitution_temperature),
                        )
                    )
                else:
                    target, new_elem = action
                    tasks.append((_worker_apply_substitution, (seed, target, new_elem)))

        if "strain" in methods:
            for _ in range(per_method):
                seed = self.seeds[int(self.rng.randint(0, len(self.seeds)))]
                tasks.append((_worker_strain, (seed, self.max_strain, int(self.rng.randint(1e9)))))

        # Keep legacy "mix" method, mapped to stochastic substitution kernels.
        if "mix" in methods:
            for _ in range(per_method):
                seed = self.seeds[int(self.rng.randint(0, len(self.seeds)))]
                action = self._sample_substitution_action(seed)
                if action is None:
                    tasks.append(
                        (
                            _worker_substitute,
                            (seed, int(self.rng.randint(1e9)), self.substitution_temperature),
                        )
                    )
                else:
                    target, new_elem = action
                    tasks.append((_worker_apply_substitution, (seed, target, new_elem)))

        def _consume_candidate(cand: dict | None) -> None:
            if not cand:
                return
            if not self._validate_structure(cand["structure"]):
                return
            if "mix" in methods and cand.get("method") == "substitute":
                cand = {**cand, "method": "mix"}
            candidates.append(cand)

        if self.n_workers <= 1:
            # Single-process fallback avoids ProcessPool spawn/pickle overhead.
            for fn, args in tasks:
                try:
                    cand = fn(*args)
                except Exception:
                    continue
                _consume_candidate(cand)
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(fn, *args) for fn, args in tasks]
                for future in as_completed(futures):
                    try:
                        cand = future.result()
                    except Exception:
                        continue
                    _consume_candidate(cand)

        ranked = self._rank_candidates(candidates, n_candidates)
        self.generated.extend(ranked)
        return ranked

    def get_top_candidates(self, n: int = 10) -> list[dict]:
        """Return top generated candidates ranked by selection utility."""
        key = "selection_utility"
        if self.generated and key not in self.generated[0]:
            key = "generator_score" if "generator_score" in self.generated[0] else "topo_score"
        ranked = sorted(self.generated, key=lambda x: x.get(key, 0.0), reverse=True)
        return ranked[:n]
