"""
Topological Materials Database Interface

Manages a database of topological materials (Topological Insulators,
Weyl Semimetals, etc.) with search capabilities.

Optimization:
- Added fuzzy search for chemical formulas (finds "Bi2Se3" from "BiSe")
- Robust CSV handling
- Enhanced query capabilities
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Union
import difflib
import logging

from atlas.config import get_config

logger = logging.getLogger(__name__)

# Default topological seeds (if DB missing)
TI_SEEDS = [
    {"formula": "Bi2Se3", "topo_class": "TI", "band_gap": 0.3},
    {"formula": "Bi2Te3", "topo_class": "TI", "band_gap": 0.15},
    {"formula": "Sb2Te3", "topo_class": "TI", "band_gap": 0.2},
]
TSM_SEEDS = [
    {"formula": "TaAs", "topo_class": "Weyl", "band_gap": 0.0},
    {"formula": "NbAs", "topo_class": "Weyl", "band_gap": 0.0},
    {"formula": "Cd3As2", "topo_class": "Dirac", "band_gap": 0.0},
]

class TopoDB:
    """
    Interface to the topological materials database.
    """

    def __init__(self):
        self.cfg = get_config()
        self.db_path = self.cfg.paths.raw_dir / "topological_materials.csv"
        self._df = self._load_or_create()

    def _load_or_create(self) -> pd.DataFrame:
        if self.db_path.exists():
            try:
                return pd.read_csv(self.db_path)
            except Exception as e:
                logger.warning(f"Failed to load TopoDB: {e}. Creating new.")
        
        # Create default
        df = pd.DataFrame(TI_SEEDS + TSM_SEEDS)
        self.save(df)
        return df

    def save(self, df: Optional[pd.DataFrame] = None):
        if df is None:
            df = self._df
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.db_path, index=False)

    def fuzzy_search(self, query: str, cutoff: float = 0.6) -> pd.DataFrame:
        """
        Search for materials with formulas similar to the query.
        Useful for typos or partial matches (e.g. 'BiSe' -> 'Bi2Se3').
        """
        if "formula" not in self._df.columns:
            return pd.DataFrame()

        formulas = self._df["formula"].astype(str).tolist()
        matches = difflib.get_close_matches(query, formulas, n=5, cutoff=cutoff)
        
        if not matches:
            return pd.DataFrame()

        return self._df[self._df["formula"].isin(matches)].copy()

    def query(
        self,
        topo_class: Optional[str] = None,
        elements: Optional[List[str]] = None,
        band_gap_range: Optional[tuple] = None,
        exact_formula: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filter database by multiple criteria.
        """
        df = self._df.copy()

        if exact_formula:
            return df[df["formula"] == exact_formula]

        if topo_class:
            df = df[df["topo_class"] == topo_class]

        if band_gap_range:
            lo, hi = band_gap_range
            # Ensure column is numeric
            if "band_gap" in df.columns:
                df["band_gap"] = pd.to_numeric(df["band_gap"], errors="coerce")
                df = df[(df["band_gap"] >= lo) & (df["band_gap"] <= hi)]

        if elements:
            # Filter rows where formula contains ALL elements
            # This is a simple string check, ideally use pymatgen Composition
            def has_elements(formula):
                return all(el in formula for el in elements)
            
            df = df[df["formula"].apply(has_elements)]

        return df

    def add_material(self, formula: str, topo_class: str, properties: dict):
        """Add a new material to the database."""
        new_row = {"formula": formula, "topo_class": topo_class, **properties}
        self._df = pd.concat([self._df, pd.DataFrame([new_row])], ignore_index=True)
        self.save()
        logger.info(f"Added {formula} ({topo_class}) to TopoDB")
