"""
JARVIS-DFT Local Data Client

Uses the JARVIS-DFT database (NIST) — a freely downloadable dataset of
~76,000 materials with DFT-computed properties including band structures,
formation energies, and topological classifications.

NO API KEY REQUIRED. Data is downloaded directly via jarvis-tools.

Usage:
    from atlas.data.jarvis_client import JARVISClient
    client = JARVISClient()
    df = client.load_dft_3d()
    topo = client.get_topological_materials()
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from atlas.config import get_config


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
        
        Contains ~76,000 materials with properties:
        - jid: JARVIS ID
        - atoms: crystal structure
        - formation_energy_peratom: formation energy (eV/atom)
        - optb88vdw_bandgap: band gap (eV)
        - ehull: energy above hull (eV/atom)
        - spg_number: space group number
        - elements: list of elements
        - topological_class: spin-orbit spillage classification

        Returns:
            DataFrame with all materials
        """
        cache_file = self.cache_dir / "dft_3d.pkl"

        if not force_reload and cache_file.exists():
            print(f"  Loading cached JARVIS-DFT data from {cache_file}")
            self._dft_3d = pd.read_pickle(cache_file)
            return self._dft_3d

        print("  Downloading JARVIS-DFT 3D database (one-time, ~500MB)...")
        print("  This may take a few minutes...")

        from jarvis.db.figshare import data as jarvis_data
        dft_3d = jarvis_data("dft_3d")

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

    def get_stable_materials(
        self,
        ehull_max: float = 0.1,
        min_band_gap: Optional[float] = None,
        max_band_gap: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Filter for thermodynamically stable materials.

        Args:
            ehull_max: max energy above hull (eV/atom)
            min_band_gap: minimum band gap (eV), None for no filter
            max_band_gap: maximum band gap (eV), None for no filter

        Returns:
            Filtered DataFrame
        """
        df = self.load_dft_3d()

        mask = df["ehull"].notna() & (df["ehull"] <= ehull_max)

        if min_band_gap is not None:
            mask &= df["optb88vdw_bandgap"] >= min_band_gap
        if max_band_gap is not None:
            mask &= df["optb88vdw_bandgap"] <= max_band_gap

        result = df[mask].copy()
        print(f"  Found {len(result)} stable materials "
              f"(ehull ≤ {ehull_max} eV/atom)")
        return result

    def get_heavy_element_materials(
        self,
        min_atomic_number: int = 50,
        ehull_max: float = 0.1,
    ) -> pd.DataFrame:
        """
        Get materials containing heavy elements (strong SOC candidates).
        
        Heavy elements (Z ≥ 50): Sn, Sb, Te, I, Ba, ..., Bi, Pb, etc.
        These are prime candidates for topological behavior.

        Args:
            min_atomic_number: minimum atomic number to count as "heavy"
            ehull_max: stability filter

        Returns:
            Filtered DataFrame
        """
        from jarvis.core.atoms import Atoms as JAtoms
        from pymatgen.core import Element

        df = self.get_stable_materials(ehull_max=ehull_max)

        def has_heavy(row):
            try:
                atoms = JAtoms.from_dict(row["atoms"])
                elements = atoms.elements
                return any(
                    Element(e).Z >= min_atomic_number for e in elements
                )
            except Exception:
                return False

        print(f"  Filtering for heavy elements (Z ≥ {min_atomic_number})...")
        mask = df.apply(has_heavy, axis=1)
        result = df[mask].copy()
        print(f"  Found {len(result)} materials with heavy elements")
        return result

    def get_topological_materials(self) -> pd.DataFrame:
        """
        Get materials classified as topologically nontrivial
        based on JARVIS spin-orbit spillage analysis.
        
        Spillage > 0.5 indicates potential topological character.

        Returns:
            DataFrame of topological candidate materials
        """
        df = self.load_dft_3d()

        # JARVIS uses 'spillage' as topological indicator
        # spillage > 0.5 → likely topological
        if "spillage" in df.columns:
            topo_mask = df["spillage"].notna() & (df["spillage"] > 0.5)
            result = df[topo_mask].copy()
            print(f"  Found {len(result)} materials with spillage > 0.5 "
                  f"(topological candidates)")
            return result
        else:
            print("  Warning: 'spillage' column not found, "
                  "returning materials with small band gap as proxies")
            return self.get_stable_materials(max_band_gap=0.3)

    def get_structure(self, jid: str):
        """
        Get a pymatgen Structure for a given JARVIS ID.

        Args:
            jid: JARVIS ID, e.g., "JVASP-1002" (Si)

        Returns:
            pymatgen Structure
        """
        from jarvis.core.atoms import Atoms as JAtoms

        df = self.load_dft_3d()
        row = df[df["jid"] == jid]

        if len(row) == 0:
            raise ValueError(f"Material {jid} not found in JARVIS-DFT")

        atoms = JAtoms.from_dict(row.iloc[0]["atoms"])
        return atoms.pymatgen_converter()

    def get_training_data(
        self,
        n_topo: int = 500,
        n_trivial: int = 500,
    ) -> dict:
        """
        Prepare a balanced training dataset for ML.

        Args:
            n_topo: number of topological materials
            n_trivial: number of trivial materials

        Returns:
            dict with 'topo' and 'trivial' DataFrames and summary stats
        """
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

    def stats(self) -> dict:
        """Get database statistics."""
        df = self.load_dft_3d()

        stats = {
            "total_materials": len(df),
            "with_bandgap": df["optb88vdw_bandgap"].notna().sum(),
            "metals": (df["optb88vdw_bandgap"] == 0).sum(),
            "semiconductors": (
                (df["optb88vdw_bandgap"] > 0) & 
                (df["optb88vdw_bandgap"] < 3)
            ).sum(),
            "insulators": (df["optb88vdw_bandgap"] >= 3).sum(),
        }

        if "spillage" in df.columns:
            stats["with_spillage"] = df["spillage"].notna().sum()
            stats["topological_candidates"] = (
                df["spillage"].notna() & (df["spillage"] > 0.5)
            ).sum()

        return stats
