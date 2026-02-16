"""
Multi-Property Crystal Dataset

PyTorch Geometric dataset for multi-task property prediction.
Loads JARVIS-DFT data and converts crystal structures to graphs
with multiple property labels.
"""

import hashlib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm

from atlas.config import get_config


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

# Default properties for multi-task learning
DEFAULT_PROPERTIES = [
    "formation_energy",
    "band_gap",
    "bulk_modulus",
    "shear_modulus",
]


class CrystalPropertyDataset:
    """
    Multi-property crystal dataset backed by JARVIS-DFT.

    Converts crystal structures to PyG Data objects with multiple
    property labels. Handles missing values by masking.

    Args:
        properties: list of property names to include
        max_samples: cap on total samples (None = use all)
        stability_filter: max ehull in eV/atom (None = no filter)
        split: "train", "val", or "test"
        split_seed: random seed for reproducible splits
        split_ratio: (train, val, test) fractions
    """

    def __init__(
        self,
        properties: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        stability_filter: Optional[float] = 0.1,
        split: str = "train",
        split_seed: int = 42,
        split_ratio: tuple = (0.8, 0.1, 0.1),
    ):
        self.properties = properties or DEFAULT_PROPERTIES
        self.max_samples = max_samples
        self.stability_filter = stability_filter
        self.split = split
        self.split_seed = split_seed
        self.split_ratio = split_ratio

        self._data_list = None
        self._df = None

    def prepare(self, force_reload: bool = False) -> "CrystalPropertyDataset":
        """
        Load and prepare the dataset.

        1. Load JARVIS-DFT data
        2. Filter for materials with valid properties
        3. Split into train/val/test
        4. Convert structures to graphs

        Returns:
            self (for chaining)
        """
        from atlas.data import JARVISClient
        from atlas.models.graph_builder import CrystalGraphBuilder

        config = get_config()
        cache_dir = config.paths.processed_dir / "multi_property"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Property-specific cache key to prevent stale data reuse
        props_key = "_".join(sorted(self.properties))
        filter_key = f"stab{self.stability_filter}" if self.stability_filter else "nofilter"
        max_key = f"max{self.max_samples}" if self.max_samples else "full"
        cache_hash = hashlib.md5(f"{props_key}_{filter_key}_{max_key}".encode()).hexdigest()[:8]
        cache_file = cache_dir / f"{self.split}_{self.split_seed}_{cache_hash}.pt"
        if not force_reload and cache_file.exists():
            print(f"  Loading cached {self.split} dataset from {cache_file}")
            self._data_list = torch.load(cache_file, weights_only=False)
            print(f"  Loaded {len(self._data_list)} samples")
            return self

        # Load raw data
        client = JARVISClient()
        df = client.load_dft_3d()

        # Reverse property map: our name → JARVIS column
        rev_map = {v: k for k, v in PROPERTY_MAP.items()}

        # Filter: must have at least one valid property
        jarvis_cols = [rev_map[p] for p in self.properties if p in rev_map]
        valid_mask = pd.Series(False, index=df.index)
        for col in jarvis_cols:
            if col in df.columns:
                valid_mask |= df[col].notna() & (df[col] != "na")

        df = df[valid_mask].copy()

        # Stability filter
        if self.stability_filter is not None and "ehull" in df.columns:
            df = df[df["ehull"].notna() & (df["ehull"] <= self.stability_filter)]

        # Must have atoms data
        df = df[df["atoms"].notna()].reset_index(drop=True)

        if self.max_samples and len(df) > self.max_samples:
            df = df.sample(self.max_samples, random_state=self.split_seed)
            df = df.reset_index(drop=True)

        print(f"  Filtered dataset: {len(df)} materials with valid properties")

        # Split
        n = len(df)
        np.random.seed(self.split_seed)
        indices = np.random.permutation(n)

        n_train = int(n * self.split_ratio[0])
        n_val = int(n * self.split_ratio[1])

        if self.split == "train":
            split_idx = indices[:n_train]
        elif self.split == "val":
            split_idx = indices[n_train:n_train + n_val]
        elif self.split == "test":
            split_idx = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        df_split = df.iloc[split_idx].reset_index(drop=True)
        self._df = df_split
        print(f"  {self.split} split: {len(df_split)} materials")

        # Convert to PyG Data objects
        builder = CrystalGraphBuilder(cutoff=5.0, max_neighbors=12)
        data_list = []
        n_failed = 0

        for idx in tqdm(range(len(df_split)), desc=f"  Building {self.split} graphs"):
            row = df_split.iloc[idx]
            try:
                # Convert JARVIS atoms dict to pymatgen Structure
                from jarvis.core.atoms import Atoms as JarvisAtoms
                atoms = JarvisAtoms.from_dict(row["atoms"])
                structure = atoms.pymatgen_converter()

                # Extract properties
                props = {}
                for prop_name in self.properties:
                    jarvis_col = rev_map.get(prop_name)
                    if jarvis_col and jarvis_col in df_split.columns:
                        val = row.get(jarvis_col)
                        if val is not None and val != "na":
                            try:
                                props[prop_name] = float(val)
                            except (ValueError, TypeError):
                                pass

                if not props:
                    continue

                data = builder.structure_to_pyg(structure, **props)
                data.jid = row.get("jid", f"unknown_{idx}")
                data_list.append(data)

            except Exception as e:
                n_failed += 1
                if n_failed <= 5:
                    print(f"    Warning: Failed to convert {row.get('jid', idx)}: {e}")

        print(f"  Successfully built {len(data_list)} graphs ({n_failed} failed)")

        # Cache
        torch.save(data_list, cache_file)
        print(f"  Cached to {cache_file}")

        self._data_list = data_list
        return self

    def __len__(self) -> int:
        if self._data_list is None:
            raise RuntimeError("Call .prepare() first")
        return len(self._data_list)

    def __getitem__(self, idx):
        if self._data_list is None:
            raise RuntimeError("Call .prepare() first")
        return self._data_list[idx]

    def to_pyg_loader(self, batch_size: int = 32, shuffle: bool = True):
        """Create a PyTorch Geometric DataLoader."""
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self._data_list,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def property_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute mean, std, min, max for each property."""
        if self._data_list is None:
            raise RuntimeError("Call .prepare() first")

        stats = {}
        for prop in self.properties:
            values = []
            for data in self._data_list:
                if hasattr(data, prop):
                    values.append(getattr(data, prop).item())

            if values:
                arr = np.array(values)
                stats[prop] = {
                    "count": len(arr),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
        return stats
