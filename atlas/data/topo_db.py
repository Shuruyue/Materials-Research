"""
Topological Materials Database Interface

Manages a database of topological materials (Topological Insulators,
Weyl Semimetals, etc.) with search and SQL capabilities.

Optimization:
- Fuzzy search for flexible chemical formula matching
- SQLite integration for scalable storage and complex queries
- Robust CSV/SQL loading
"""

import pandas as pd
import sqlite3
import difflib
import logging
from pathlib import Path
from typing import Optional, List, Union

from atlas.config import get_config

logger = logging.getLogger(__name__)

# Default topological seeds
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

TOPO_CLASSES = ["TI", "Weyl", "Dirac", "TCI", "LineNode", "Magnetic_TI"]

from dataclasses import dataclass

@dataclass
class TopoMaterial:
    formula: str
    topo_class: str
    band_gap: float = 0.0
    jid: str = ""
    space_group: int = 0


class TopoDB:
    """
    Interface to the topological materials database.
    Supports CSV (default) and SQLite backends.
    """

    def __init__(self, use_sql: bool = False):
        self.cfg = get_config()
        self.db_path = self.cfg.paths.raw_dir / "topological_materials.csv"
        self.sql_path = self.cfg.paths.raw_dir / "topological_materials.db"
        self.use_sql = use_sql
        
        # Load data
        if use_sql and self.sql_path.exists():
            self._df = self._load_from_sql()
        else:
            self._df = self._load_or_create_csv()

    def _load_or_create_csv(self) -> pd.DataFrame:
        if self.db_path.exists():
            try:
                return pd.read_csv(self.db_path)
            except Exception as e:
                logger.warning(f"Failed to load CSV TopoDB: {e}. Creating new.")
        
        df = pd.DataFrame(TI_SEEDS + TSM_SEEDS)
        self.save_csv(df)
        return df

    def _load_from_sql(self) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.sql_path)
            df = pd.read_sql("SELECT * FROM materials", conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Failed to load SQL TopoDB: {e}")
            return self._load_or_create_csv()

    def save_csv(self, df: Optional[pd.DataFrame] = None):
        if df is None: df = self._df
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.db_path, index=False)

    def to_sql(self, db_path: Optional[Path] = None):
        """Export current DB to SQLite."""
        path = db_path or self.sql_path
        try:
            conn = sqlite3.connect(path)
            self._df.to_sql("materials", conn, if_exists="replace", index=False)
            
            # Create indices for speed
            conn.execute("CREATE INDEX IF NOT EXISTS idx_formula ON materials (formula)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_topo ON materials (topo_class)")
            conn.commit()
            conn.close()
            logger.info(f"Saved TopoDB to SQL: {path}")
        except Exception as e:
            logger.error(f"SQL export failed: {e}")

    def query_sql(self, sql_query: str) -> pd.DataFrame:
        """Execute raw SQL query."""
        if not self.sql_path.exists():
            self.to_sql() # Sync first
            
        try:
            conn = sqlite3.connect(self.sql_path)
            df = pd.read_sql(sql_query, conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"SQL query failed: {e}")
            return pd.DataFrame()

    def fuzzy_search(self, query: str, cutoff: float = 0.6) -> pd.DataFrame:
        """
        Search for materials with formulas similar to the query.
        Works on in-memory DataFrame (hybrid approach).
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
        Filter database by multiple criteria (Pandas backend).
        """
        df = self._df.copy()

        if exact_formula:
            return df[df["formula"] == exact_formula]

        if topo_class:
            df = df[df["topo_class"] == topo_class]

        if band_gap_range:
            lo, hi = band_gap_range
            if "band_gap" in df.columns:
                df["band_gap"] = pd.to_numeric(df["band_gap"], errors="coerce")
                df = df[(df["band_gap"] >= lo) & (df["band_gap"] <= hi)]

        if elements:
            # Vectorized element check?
            # Creating a set column is slow but accurate
            def has_elements(formula):
                return all(el in formula for el in elements)
            
            df = df[df["formula"].apply(has_elements)]

        return df

    def add_material(self, formula: str, topo_class: str, properties: dict):
        """Add a new material to the database."""
        new_row = {"formula": formula, "topo_class": topo_class, **properties}
        self._df = pd.concat([self._df, pd.DataFrame([new_row])], ignore_index=True)
        self.save_csv()
        if self.use_sql:
            self.to_sql()
        logger.info(f"Added {formula} ({topo_class}) to TopoDB")
