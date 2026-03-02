"""
Multi-Property Crystal Dataset

PyTorch Geometric dataset for multi-task property prediction.
Loads JARVIS-DFT data and converts crystal structures to graphs
with multiple property labels.

Optimization:
- Parallel graph construction using ProcessPoolExecutor (significant speedup)
- Improved error handling and progress reporting
"""

# Workaround for WeightsUnpickler error with slice
import contextlib
import hashlib
import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

with contextlib.suppress(AttributeError):
    torch.serialization.add_safe_globals([slice])

from atlas.config import get_config
from atlas.models.graph_builder import CrystalGraphBuilder

# Configure logging
logger = logging.getLogger(__name__)


# JARVIS column name → our property name mapping
PROPERTY_MAP = {
    "formation_energy_peratom": "formation_energy",
    "optb88vdw_bandgap": "band_gap",
    "mbj_bandgap": "band_gap_mbj",
    "bulk_modulus_kv": "bulk_modulus",
    "shear_modulus_gv": "shear_modulus",
    "dfpt_piezo_max_dielectric": "dielectric",
    "dfpt_piezo_max_eij": "piezoelectric",
    "spillage": "spillage",
    "ehull": "ehull",
}

DEFAULT_PROPERTIES = [
    "formation_energy",
    "band_gap",
    "bulk_modulus",
    "shear_modulus",
]

PHASE2_PRIORITY_PROPERTIES = [
    "formation_energy",
    "ehull",
    "band_gap",
    "bulk_modulus",
    "shear_modulus",
    "band_gap_mbj",
    "spillage",
]

PHASE2_SECONDARY_PROPERTIES = [
    "dielectric",
    "piezoelectric",
]

ALL_DISCOVERABLE_PROPERTIES = [
    "formation_energy",
    "band_gap",
    "band_gap_mbj",
    "bulk_modulus",
    "shear_modulus",
    "dielectric",
    "piezoelectric",
    "spillage",
    "ehull",
]

PHASE2_PROPERTY_GROUPS = {
    "core4": DEFAULT_PROPERTIES,
    "priority7": PHASE2_PRIORITY_PROPERTIES,
    "secondary2": PHASE2_SECONDARY_PROPERTIES,
    "all9": ALL_DISCOVERABLE_PROPERTIES,
}

PHASE2_PROPERTY_GROUP_ALIASES = {
    "core": "core4",
    "default": "core4",
    "priority": "priority7",
    "primary": "priority7",
    "secondary": "secondary2",
    "all": "all9",
}

PHASE2_PROPERTY_GROUP_CHOICES = tuple(PHASE2_PROPERTY_GROUPS.keys())

SAMPLING_STRATEGIES = ("random", "kcenter_formula", "kcenter_hybrid")
FALLBACK_SPLIT_STRATEGIES = ("iid", "compositional", "prototype")

_FORMULA_TOKEN_RE = re.compile(r"[A-Z][a-z]?")


def resolve_phase2_property_group(group: str) -> list[str]:
    """
    Resolve a named Phase 2 property group to a concrete property list.
    """
    key = group.strip().lower()
    key = PHASE2_PROPERTY_GROUP_ALIASES.get(key, key)
    if key not in PHASE2_PROPERTY_GROUPS:
        choices = ", ".join(sorted(PHASE2_PROPERTY_GROUP_CHOICES))
        raise ValueError(f"Unknown Phase 2 property group '{group}'. Choices: {choices}")
    return list(PHASE2_PROPERTY_GROUPS[key])


def _normalize_sampling_strategy(strategy: str) -> str:
    canon = strategy.strip().lower()
    if canon not in SAMPLING_STRATEGIES:
        choices = ", ".join(SAMPLING_STRATEGIES)
        raise ValueError(f"Unknown sampling strategy '{strategy}'. Choices: {choices}")
    return canon


def _normalize_fallback_split_strategy(strategy: str) -> str:
    canon = strategy.strip().lower()
    if canon not in FALLBACK_SPLIT_STRATEGIES:
        choices = ", ".join(FALLBACK_SPLIT_STRATEGIES)
        raise ValueError(f"Unknown fallback split strategy '{strategy}'. Choices: {choices}")
    return canon


def _formula_fraction_vector(formula: str) -> np.ndarray:
    """
    Convert chemical formula to a normalized 118-d composition vector.

    This representation enables metric-space subset selection via k-center.
    """
    vec = np.zeros(118, dtype=np.float32)
    if not isinstance(formula, str) or not formula.strip():
        return vec
    try:
        from pymatgen.core import Composition, Element

        comp = Composition(formula)
        total = 0.0
        for symbol, amount in comp.get_el_amt_dict().items():
            try:
                z = Element(symbol).Z
            except Exception:
                continue
            amount_f = float(amount)
            if amount_f <= 0:
                continue
            vec[z - 1] += amount_f
            total += amount_f
        if total > 0:
            vec /= total
        return vec
    except Exception:
        # Fallback parser if Composition fails on malformed formulas.
        symbols = _FORMULA_TOKEN_RE.findall(formula)
        if not symbols:
            return vec
        try:
            from pymatgen.core import Element
        except Exception:
            return vec
        for sym in symbols:
            try:
                z = Element(sym).Z
            except Exception:
                continue
            vec[z - 1] += 1.0
        norm = float(vec.sum())
        if norm > 0:
            vec /= norm
        return vec


def _kcenter_coreset_indices(features: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    Greedy k-center subset (farthest-point traversal).

    Reference:
    - Gonzalez (1985), "Clustering to minimize the maximum intercluster distance"
      2-approximation for metric k-center.
    """
    n = int(features.shape[0])
    if k >= n:
        return np.arange(n, dtype=np.int64)
    if k <= 0:
        return np.zeros(0, dtype=np.int64)

    rng = np.random.RandomState(seed)
    first = int(rng.randint(0, n))
    selected = np.empty(k, dtype=np.int64)
    selected[0] = first

    diff0 = features - features[first]
    min_d2 = np.einsum("ij,ij->i", diff0, diff0, optimize=True)
    min_d2[first] = -1.0

    for t in range(1, k):
        nxt = int(np.argmax(min_d2))
        selected[t] = nxt
        diff = features - features[nxt]
        d2 = np.einsum("ij,ij->i", diff, diff, optimize=True)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[selected[: t + 1]] = -1.0
    return selected


def _property_presence_vector(
    row_data: dict[str, Any],
    properties: list[str],
    rev_map: dict[str, str],
) -> np.ndarray:
    """
    Binary mask for which target properties are present for a sample.

    This encourages coreset subsets to preserve label-availability diversity.
    """
    out = np.zeros(len(properties), dtype=np.float32)
    for i, prop_name in enumerate(properties):
        col = rev_map.get(prop_name)
        if not col:
            continue
        value = row_data.get(col)
        if value is not None and value != "na" and pd.notna(value):
            out[i] = 1.0
    return out


def _worker_process_row(
    row_data: dict,
    properties: list[str],
    rev_map: dict[str, str],
    graph_cutoff: float = 5.0,
    graph_max_neighbors: int = 12,
    graph_compute_3body: bool = True,
) -> Any | None:
    """
    Worker function to process a single DataFrame row into a PyG Data object.
    Must be top-level for pickling.
    """
    try:
        # Convert JARVIS atoms dict to pymatgen Structure
        from jarvis.core.atoms import Atoms as JarvisAtoms
        atoms = JarvisAtoms.from_dict(row_data["atoms"])
        structure = atoms.pymatgen_converter()

        # Extract properties
        props = {}
        for prop_name in properties:
            jarvis_col = rev_map.get(prop_name)
            if jarvis_col and jarvis_col in row_data:
                val = row_data.get(jarvis_col)
                if val is not None and val != "na":
                    try:
                        props[prop_name] = float(val)
                    except (ValueError, TypeError):
                        props[prop_name] = float('nan')
                else:
                    props[prop_name] = float('nan')
            else:
                props[prop_name] = float('nan')

        if not props:
            return None

        # Build graph
        # Note: We instantiate a fresh builder here because it's cheap and safe
        builder = CrystalGraphBuilder(
            cutoff=graph_cutoff,
            max_neighbors=graph_max_neighbors,
            compute_3body=graph_compute_3body,
        )
        data = builder.structure_to_pyg(structure, **props)
        data.jid = row_data.get("jid", "unknown")

        return data

    except Exception:
        # Silently fail in worker, or return None
        return None


class CrystalPropertyDataset:
    """
    Multi-property crystal dataset backed by JARVIS-DFT.
    Optimized for parallel processing.
    """

    def __init__(
        self,
        properties: list[str] | None = None,
        max_samples: int | None = None,
        stability_filter: float | None = 0.1,
        split: str = "train",
        split_seed: int = 42,
        split_ratio: tuple = (0.8, 0.1, 0.1),
        split_manifest_path: str | Path | None = None,
        assignment_col: str = "split",
        enforce_manifest_split: bool = True,
        min_labeled_properties: int = 1,
        sampling_strategy: str = "random",
        fallback_split_strategy: str = "iid",
        enforce_nonempty_split: bool = False,
        graph_cutoff: float = 5.0,
        graph_max_neighbors: int = 12,
        graph_compute_3body: bool = True,
    ):
        self.properties = list(dict.fromkeys(properties or DEFAULT_PROPERTIES))
        self.max_samples = max_samples
        self.stability_filter = stability_filter
        self.split = split
        self.split_seed = split_seed
        self.split_ratio = split_ratio
        env_manifest = os.environ.get("ATLAS_SPLIT_MANIFEST", "").strip()
        manifest_candidate = split_manifest_path
        if manifest_candidate is None and env_manifest:
            manifest_candidate = env_manifest
        if manifest_candidate is None:
            default_manifest = get_config().paths.artifacts_dir / "splits" / "split_manifest_iid.json"
            if default_manifest.exists():
                manifest_candidate = default_manifest
        self.split_manifest_path = Path(manifest_candidate) if manifest_candidate else None
        self.assignment_col = assignment_col
        self.enforce_manifest_split = enforce_manifest_split
        self.min_labeled_properties = max(1, int(min_labeled_properties))
        self.sampling_strategy = _normalize_sampling_strategy(sampling_strategy)
        self.fallback_split_strategy = _normalize_fallback_split_strategy(fallback_split_strategy)
        self.enforce_nonempty_split = bool(enforce_nonempty_split)
        self.graph_cutoff = float(graph_cutoff)
        self.graph_max_neighbors = int(graph_max_neighbors)
        self.graph_compute_3body = bool(graph_compute_3body)

        known_properties = set(PROPERTY_MAP.values())
        unknown = sorted(p for p in self.properties if p not in known_properties)
        if unknown:
            choices = ", ".join(sorted(known_properties))
            raise ValueError(
                f"Unknown properties: {', '.join(unknown)}. Supported properties: {choices}"
            )

        if self.graph_cutoff <= 0:
            raise ValueError("graph_cutoff must be > 0.")
        if self.graph_max_neighbors <= 0:
            raise ValueError("graph_max_neighbors must be > 0.")

        self._data_list = None
        self._df = None

        # Parallel workers: leave 1 core free
        # self.n_workers = max(1, multiprocessing.cpu_count() - 1)
        self.n_workers = 1 # Force sequential on Windows to avoid spawn errors

    def prepare(self, force_reload: bool = False) -> "CrystalPropertyDataset":
        """
        Load and prepare the dataset with parallel processing.
        """
        from atlas.data import JARVISClient

        config = get_config()
        cache_dir = config.paths.processed_dir / "multi_property"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache key
        props_key = "_".join(sorted(self.properties))
        filter_key = f"stab{self.stability_filter}" if self.stability_filter else "nofilter"
        max_key = f"max{self.max_samples}" if self.max_samples else "full"
        manifest_key = (
            str(self.split_manifest_path.resolve()) if self.split_manifest_path else "random_split"
        )
        cache_hash = hashlib.md5(
            (
                f"{props_key}_{filter_key}_{max_key}_{manifest_key}_{self.assignment_col}_"
                f"{self.enforce_manifest_split}_{self.min_labeled_properties}_{self.sampling_strategy}_"
                f"{self.fallback_split_strategy}_{self.enforce_nonempty_split}_{self.graph_cutoff}_{self.graph_max_neighbors}_"
                f"{self.graph_compute_3body}"
            ).encode()
        ).hexdigest()[:8]
        cache_file = cache_dir / f"{self.split}_{self.split_seed}_{cache_hash}.pt"

        if not force_reload and cache_file.exists():
            print(f"  Loading cached {self.split} dataset from {cache_file}")
            self._data_list = torch.load(cache_file, weights_only=False)
            print(f"  Loaded {len(self._data_list)} samples")
            return self

        # Load raw data
        client = JARVISClient()
        df = client.load_dft_3d()

        # Reverse property map
        rev_map = {v: k for k, v in PROPERTY_MAP.items()}

        # Filter valid rows
        jarvis_cols = [rev_map[p] for p in self.properties if p in rev_map]
        valid_count = pd.Series(0, index=df.index, dtype=np.int32)
        for col in jarvis_cols:
            if col in df.columns:
                valid_count += (df[col].notna() & (df[col] != "na")).astype(np.int32)

        valid_mask = valid_count >= self.min_labeled_properties

        df = df[valid_mask].copy()

        if self.stability_filter is not None and "ehull" in df.columns:
            df = df[df["ehull"].notna() & (df["ehull"] <= self.stability_filter)]

        df = df[df["atoms"].notna()].reset_index(drop=True)

        if self.max_samples and len(df) > self.max_samples:
            df = self._sample_subset(df).reset_index(drop=True)

        print(f"  Filtered dataset: {len(df)} materials with valid properties")

        # Split
        df_split = self._apply_split(df)
        self._df = df_split
        print(f"  {self.split} split: {len(df_split)} materials")

        # Parallel Graph Construction
        data_list = []
        rows = df_split.to_dict("records")

        print(f"  Building {self.split} graphs with {self.n_workers} workers...")

        if self.n_workers > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit all tasks
                futures = [
                    executor.submit(
                        _worker_process_row,
                        row,
                        self.properties,
                        rev_map,
                        self.graph_cutoff,
                        self.graph_max_neighbors,
                        self.graph_compute_3body,
                    )
                    for row in rows
                ]

                # Progress bar
                for future in tqdm(as_completed(futures), total=len(futures), desc="  Converting"):
                    try:
                        data = future.result()
                        if data is not None:
                            data_list.append(data)
                    except Exception:
                        pass
        else:
            # Sequential fallback
            for row in tqdm(rows, desc="  Converting (Sequential)"):
                data = _worker_process_row(
                    row,
                    self.properties,
                    rev_map,
                    self.graph_cutoff,
                    self.graph_max_neighbors,
                    self.graph_compute_3body,
                )
                if data is not None:
                    data_list.append(data)

        print(f"  Successfully built {len(data_list)} graphs")

        # Cache
        torch.save(data_list, cache_file)
        print(f"  Cached to {cache_file}")

        self._data_list = data_list
        return self

    def _apply_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply split assignment from manifest, otherwise use configured fallback split."""
        if self.split_manifest_path is not None:
            assignment = self._load_manifest_assignment()
            if "jid" not in df.columns:
                if self.enforce_manifest_split:
                    raise ValueError("Manifest split requested but dataset has no 'jid' column.")
            else:
                jid_series = df["jid"].astype(str)
                mapped = jid_series.map(assignment)
                missing_count = int(mapped.isna().sum())
                if self.enforce_manifest_split and missing_count > 0:
                    print(
                        f"  [WARN] Manifest split active: dropping {missing_count} samples "
                        "without assignment mapping."
                    )
                matched = df[mapped == self.split].copy()
                if self.enforce_manifest_split and matched.empty:
                    raise ValueError(
                        f"Manifest split enforcement failed: split='{self.split}' has no matched samples."
                    )
                if not matched.empty:
                    return matched.reset_index(drop=True)

        if self.fallback_split_strategy == "iid":
            return self._iid_split(df)

        if "jid" in df.columns:
            sample_ids = [str(x) for x in df["jid"].tolist()]
        else:
            sample_ids = [str(i) for i in range(len(df))]
        id_series = pd.Series(sample_ids, index=df.index, dtype="string")

        # OOD-style split families (composition/prototype) are standard in
        # materials benchmarks such as Matbench.
        # Ref: Dunn et al., npj Comput Mater 2020 (Matbench).
        from atlas.data.split_governance import compositional_split, prototype_split

        if self.fallback_split_strategy == "compositional":
            if "formula" not in df.columns:
                print("  [WARN] compositional split requested but 'formula' missing; fallback to IID.")
                return self._iid_split(df)
            formulas = [str(x) if pd.notna(x) else "" for x in df["formula"]]
            split_map = compositional_split(
                sample_ids,
                formulas,
                seed=self.split_seed,
                ratios=self.split_ratio,
            )
        elif self.fallback_split_strategy == "prototype":
            if "spg_number" not in df.columns:
                print("  [WARN] prototype split requested but 'spg_number' missing; fallback to IID.")
                return self._iid_split(df)
            spacegroups = [str(x) if pd.notna(x) else "unknown" for x in df["spg_number"]]
            split_map = prototype_split(
                sample_ids,
                spacegroups,
                seed=self.split_seed,
                ratios=self.split_ratio,
            )
        else:
            # Safety net: should not be reachable due normalization.
            return self._iid_split(df)

        selected_ids = set(split_map.get(self.split, []))
        matched = df[id_series.isin(selected_ids)].copy()
        if matched.empty and self._target_split_ratio() > 0 and len(df) > 0:
            msg = (
                f"Fallback split strategy '{self.fallback_split_strategy}' produced empty "
                f"split='{self.split}'. Falling back to IID split."
            )
            if self.enforce_nonempty_split:
                raise ValueError(msg)
            print(f"  [WARN] {msg}")
            return self._iid_split(df)
        return matched.reset_index(drop=True)

    def _iid_split(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        rng = np.random.RandomState(self.split_seed)
        indices = rng.permutation(n)
        n_train, n_val, _ = self._split_counts(n)

        if self.split == "train":
            split_idx = indices[:n_train]
        elif self.split == "val":
            split_idx = indices[n_train:n_train + n_val]
        else:
            split_idx = indices[n_train + n_val:]
        out = df.iloc[split_idx].reset_index(drop=True)
        if out.empty and self._target_split_ratio() > 0 and n > 0:
            msg = (
                f"IID split produced empty split='{self.split}' for n={n} "
                f"and split_ratio={self.split_ratio}."
            )
            if self.enforce_nonempty_split:
                raise ValueError(msg)
            print(f"  [WARN] {msg}")
        return out

    def _target_split_ratio(self) -> float:
        if self.split == "train":
            return float(self.split_ratio[0])
        if self.split == "val":
            return float(self.split_ratio[1])
        return float(self.split_ratio[2])

    def _split_counts(self, n: int) -> tuple[int, int, int]:
        """
        Deterministic split counts from ratios with residual balancing.

        Ensures each positive-ratio split gets at least one sample when feasible.
        """
        ratios = np.array(self.split_ratio, dtype=float)
        if ratios.shape != (3,):
            raise ValueError(f"split_ratio must have 3 values, got {self.split_ratio}")
        if np.any(ratios < 0):
            raise ValueError(f"split_ratio cannot contain negatives: {self.split_ratio}")
        s = float(ratios.sum())
        if s <= 0:
            raise ValueError(f"split_ratio sum must be positive: {self.split_ratio}")
        ratios = ratios / s

        raw = ratios * float(n)
        counts = np.floor(raw).astype(int)
        remainder = int(n - counts.sum())
        if remainder > 0:
            frac = raw - counts
            order = np.argsort(-frac)
            for idx in order[:remainder]:
                counts[idx] += 1

        positive = [i for i, r in enumerate(ratios) if r > 0]
        if n >= len(positive):
            for idx in positive:
                if counts[idx] > 0:
                    continue
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[idx] += 1
        return int(counts[0]), int(counts[1]), int(counts[2])

    def _sample_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.max_samples is None or len(df) <= self.max_samples:
            return df

        if self.sampling_strategy == "random":
            return df.sample(self.max_samples, random_state=self.split_seed)

        if self.sampling_strategy == "kcenter_formula":
            if "formula" not in df.columns:
                print("  [WARN] kcenter_formula requires 'formula'; fallback to random sample.")
                return df.sample(self.max_samples, random_state=self.split_seed)

            # k-center coreset on composition simplex.
            # References:
            # - Gonzalez (1985): greedy farthest-point 2-approx for metric k-center.
            # - Sener & Savarese (2018), arXiv:1708.00489:
            #   "Active Learning for Convolutional Neural Networks: A Core-Set Approach".
            formulas = [str(x) if pd.notna(x) else "" for x in df["formula"]]
            features = np.vstack([_formula_fraction_vector(formula) for formula in formulas])
            if np.allclose(features, 0.0):
                print("  [WARN] Formula features degenerate; fallback to random sample.")
                return df.sample(self.max_samples, random_state=self.split_seed)
            selected = _kcenter_coreset_indices(features, self.max_samples, self.split_seed)
            return df.iloc[selected]

        if self.sampling_strategy == "kcenter_hybrid":
            if "formula" not in df.columns:
                print("  [WARN] kcenter_hybrid requires 'formula'; fallback to random sample.")
                return df.sample(self.max_samples, random_state=self.split_seed)

            # Hybrid coreset:
            # - composition vector (chemical coverage)
            # - target-presence mask (multi-task label coverage)
            # - optional normalized spacegroup scalar (coarse structure diversity)
            # References:
            # - Gonzalez (1985): metric k-center approximation.
            # - Sener & Savarese (2018), arXiv:1708.00489 (core-set active selection).
            formulas = [str(x) if pd.notna(x) else "" for x in df["formula"]]
            comp = np.vstack([_formula_fraction_vector(formula) for formula in formulas])
            if np.allclose(comp, 0.0):
                print("  [WARN] Formula features degenerate; fallback to random sample.")
                return df.sample(self.max_samples, random_state=self.split_seed)

            rev_map = {v: k for k, v in PROPERTY_MAP.items()}
            rows = df.to_dict("records")
            label_mask = np.vstack(
                [_property_presence_vector(row, self.properties, rev_map) for row in rows]
            )

            if "spg_number" in df.columns:
                spg = pd.to_numeric(df["spg_number"], errors="coerce").to_numpy(dtype=float)
                spg_feat = np.zeros((len(df), 1), dtype=np.float32)
                finite = np.isfinite(spg)
                if finite.any():
                    denom = max(1.0, float(np.nanmax(spg[finite])))
                    spg_feat[finite, 0] = (spg[finite] / denom).astype(np.float32)
            else:
                spg_feat = np.zeros((len(df), 1), dtype=np.float32)

            comp_w = 1.0
            label_w = 0.35
            spg_w = 0.25
            features = np.hstack([comp * comp_w, label_mask * label_w, spg_feat * spg_w])
            selected = _kcenter_coreset_indices(features, self.max_samples, self.split_seed)
            return df.iloc[selected]

        # Safety net: should not be reachable due normalization.
        return df.sample(self.max_samples, random_state=self.split_seed)

    def _load_manifest_assignment(self) -> dict[str, str]:
        """Load sample_id -> split assignment from manifest-linked assignment file."""
        manifest_path = self.split_manifest_path
        if manifest_path is None:
            return {}
        if not manifest_path.exists():
            raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, dict):
            raise ValueError(f"Invalid split manifest format: {manifest_path}")

        strategy = str(manifest.get("split_strategy", "iid"))
        metadata = manifest.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        candidates: list[Path] = []
        csv_name = metadata.get("assignment_csv")
        json_name = metadata.get("assignment_json")
        if isinstance(csv_name, str) and csv_name:
            candidates.append(manifest_path.parent / csv_name)
        if isinstance(json_name, str) and json_name:
            candidates.append(manifest_path.parent / json_name)
        candidates.append(manifest_path.parent / f"split_assignment_{strategy}.csv")
        candidates.append(manifest_path.parent / f"split_assignment_{strategy}.json")

        for assignment_path in candidates:
            if not assignment_path.exists():
                continue
            if assignment_path.suffix.lower() == ".csv":
                table = pd.read_csv(assignment_path)
                if "sample_id" not in table.columns or self.assignment_col not in table.columns:
                    continue
                return {
                    str(sample_id): str(split_name)
                    for sample_id, split_name in zip(table["sample_id"], table[self.assignment_col])
                }
            if assignment_path.suffix.lower() == ".json":
                with open(assignment_path, encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    rows = [r for r in payload if isinstance(r, dict)]
                    if not rows:
                        continue
                    return {
                        str(row.get("sample_id")): str(row.get(self.assignment_col))
                        for row in rows
                        if row.get("sample_id") is not None and row.get(self.assignment_col) is not None
                    }

        raise FileNotFoundError(
            f"Could not find assignment file for manifest: {manifest_path}. "
            f"Tried: {', '.join(str(p) for p in candidates)}"
        )

    def __len__(self) -> int:
        if self._data_list is None:
            raise RuntimeError("Call .prepare() first")
        return len(self._data_list)

    def __getitem__(self, idx):
        if self._data_list is None:
            raise RuntimeError("Call .prepare() first")
        return self._data_list[idx]

    def to_pyg_loader(self, batch_size: int = 32, shuffle: bool = True, **kwargs):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self._data_list,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )

    def property_statistics(self) -> dict[str, dict[str, float]]:
        if self._data_list is None:
            raise RuntimeError("Call .prepare() first")

        stats = {}
        for prop in self.properties:
            values = []
            for data in self._data_list:
                if hasattr(data, prop):
                    values.append(getattr(data, prop).item())

            if values:
                arr = np.array(values, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    stats[prop] = {
                        "count": int(arr.size),
                        "mean": float(arr.mean()),
                        "std": float(arr.std()),
                        "min": float(arr.min()),
                        "max": float(arr.max()),
                    }
        return stats
