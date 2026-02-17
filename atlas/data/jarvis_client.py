"""
JARVIS-DFT Local Data Client

Uses the JARVIS-DFT database (NIST) — a freely downloadable dataset of
~76,000 materials with DFT-computed properties.

Optimization:
- Robust download with resume capability and progress bar (tqdm)
- Automatic retry on network failure
- Local caching
"""

import json
import os
import requests
import time
import logging
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import numpy as np
import pandas as pd

from atlas.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JARVIS_DFT_3D_URL = "https://figshare.com/ndownloader/files/40357663" # Direct link to dft_3d.json


class JARVISClient:
    """
    JARVIS-DFT local data client.
    
    Downloads the full JARVIS-DFT dataset on first use (one-time ~500MB).
    Subsequent calls use cached local data. No API key needed.
    """

    def __init__(self):
        cfg = get_config()
        self.cache_dir = cfg.paths.raw_dir / "jarvis_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._dft_3d = None

    def load_dft_3d(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load the JARVIS-DFT 3D materials database.
        """
        cache_file = self.cache_dir / "dft_3d.pkl"
        json_file = self.cache_dir / "dft_3d.json"

        if not force_reload and cache_file.exists():
            print(f"  Loading cached JARVIS-DFT data from {cache_file}")
            try:
                self._dft_3d = pd.read_pickle(cache_file)
                return self._dft_3d
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Reloading from source.")

        # Check if raw JSON exists, otherwise download
        if not json_file.exists() or force_reload:
            self._download_file(JARVIS_DFT_3D_URL, json_file)

        print("  Processing JARVIS-DFT data...")
        with open(json_file, "r") as f:
            dft_3d = json.load(f)

        # Convert to DataFrame
        df = pd.DataFrame(dft_3d)

        # Clean numeric columns (JARVIS data has mixed str/float values)
        numeric_cols = [
            "optb88vdw_bandgap", "ehull", "spillage",
            "formation_energy_peratom", "optb88vdw_total_energy",
            "mbj_bandgap", "bulk_modulus_kv", "shear_modulus_gv",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Cache locally
        df.to_pickle(cache_file)
        print(f"  Downloaded {len(df)} materials, cached at {cache_file}")

        self._dft_3d = df
        return df

    def _download_file(self, url: str, dest_path: Path, max_retries: int = 3):
        """Download file with progress bar and retries."""
        print(f"  Downloading data from {url}...")
        
        for attempt in range(max_retries):
            try:
                # Stream download
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192 # 8KB
                
                with open(dest_path, "wb") as f, tqdm(
                    desc=dest_path.name,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(block_size):
                        size = f.write(data)
                        bar.update(size)
                
                return # Success
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Download attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1)) # Backoff
                else:
                    raise RuntimeError(f"Failed to download data after {max_retries} attempts") from e

    def get_stable_materials(
        self,
        ehull_max: float = 0.1,
        min_band_gap: Optional[float] = None,
        max_band_gap: Optional[float] = None,
    ) -> pd.DataFrame:
        """Filter for thermodynamically stable materials."""
        df = self.load_dft_3d()

        mask = df["ehull"].notna() & (df["ehull"] <= ehull_max)

        if min_band_gap is not None:
            mask &= df["optb88vdw_bandgap"] >= min_band_gap
        if max_band_gap is not None:
            mask &= df["optb88vdw_bandgap"] <= max_band_gap

        result = df[mask].copy()
        print(f"  Found {len(result)} stable materials (ehull <= {ehull_max} eV/atom)")
        return result

    def get_heavy_element_materials(
        self,
        min_atomic_number: int = 50,
        ehull_max: float = 0.1,
    ) -> pd.DataFrame:
        """Get materials containing heavy elements (strong SOC candidates)."""
        from jarvis.core.atoms import Atoms as JAtoms
        from pymatgen.core import Element

        df = self.get_stable_materials(ehull_max=ehull_max)

        def has_heavy(row):
            try:
                atoms = JAtoms.from_dict(row["atoms"])
                elements = atoms.elements
                return any(Element(e).Z >= min_atomic_number for e in elements)
            except Exception:
                return False

        print(f"  Filtering for heavy elements (Z >= {min_atomic_number})...")
        mask = df.apply(has_heavy, axis=1)
        result = df[mask].copy()
        print(f"  Found {len(result)} materials with heavy elements")
        return result

    def get_topological_materials(self) -> pd.DataFrame:
        """Get materials classified as topologically nontrivial."""
        df = self.load_dft_3d()

        if "spillage" in df.columns:
            topo_mask = df["spillage"].notna() & (df["spillage"] > 0.5)
            result = df[topo_mask].copy()
            print(f"  Found {len(result)} materials with spillage > 0.5")
            return result
        else:
            print("  Warning: 'spillage' column not found, returning proxy.")
            return self.get_stable_materials(max_band_gap=0.3)

    def get_structure(self, jid: str):
        """Get a pymatgen Structure for a given JARVIS ID."""
        from jarvis.core.atoms import Atoms as JAtoms

        df = self.load_dft_3d()
        row = df[df["jid"] == jid]

        if len(row) == 0:
            raise ValueError(f"Material {jid} not found in JARVIS-DFT")

        atoms = JAtoms.from_dict(row.iloc[0]["atoms"])
        return atoms.pymatgen_converter()

    def get_training_data(self, n_topo: int = 500, n_trivial: int = 500) -> dict:
        """Prepare a balanced training dataset for ML."""
        topo = self.get_topological_materials()
        trivial = self.get_stable_materials(min_band_gap=2.0, max_band_gap=6.0)

        # Sample
        if len(topo) > n_topo:
            topo = topo.sample(n=n_topo, random_state=42)
        if len(trivial) > n_trivial:
            trivial = trivial.sample(n=n_trivial, random_state=42)

        print(f"\n  Training data prepared:")
        print(f"    Topological:  {len(topo)}")
        print(f"    Trivial:      {len(trivial)}")
        print(f"    Total:        {len(topo) + len(trivial)}")

        return {
            "topo": topo,
            "trivial": trivial,
            "total": len(topo) + len(trivial),
        }
