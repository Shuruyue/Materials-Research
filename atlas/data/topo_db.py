"""
Topological materials database utility.

This module keeps a lightweight CSV-backed store and optional SQLite mirror.
It intentionally preserves a legacy API used by existing tests/scripts.
"""

from __future__ import annotations

import difflib
import logging
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from atlas.config import get_config

logger = logging.getLogger(__name__)

_DEFAULT_COLUMNS = [
    "jid",
    "formula",
    "space_group",
    "topo_class",
    "band_gap",
    "source",
]

_SEED_ROWS = [
    {"jid": "seed-ti-001", "formula": "Bi2Se3", "space_group": 166, "topo_class": "TI", "band_gap": 0.30, "source": "seed"},
    {"jid": "seed-ti-002", "formula": "Bi2Te3", "space_group": 166, "topo_class": "TI", "band_gap": 0.15, "source": "seed"},
    {"jid": "seed-tsm-001", "formula": "TaAs", "space_group": 109, "topo_class": "TSM", "band_gap": 0.00, "source": "seed"},
    {"jid": "seed-tsm-002", "formula": "Cd3As2", "space_group": 137, "topo_class": "TSM", "band_gap": 0.00, "source": "seed"},
    {"jid": "seed-tri-001", "formula": "Si", "space_group": 227, "topo_class": "TRIVIAL", "band_gap": 1.10, "source": "seed"},
]

TOPO_CLASSES = (
    "TRIVIAL",
    "TI",
    "TSM",
    "TCI",
    "WEYL",
    "DIRAC",
    "LINENODE",
    "MAGNETIC_TI",
)

_TOPO_POSITIVE = set(TOPO_CLASSES) - {"TRIVIAL"}


@dataclass
class TopoMaterial:
    # Legacy positional order used in tests:
    # TopoMaterial(jid, formula, space_group, topo_class, band_gap)
    jid: str
    formula: str
    space_group: int
    topo_class: str
    band_gap: float = 0.0
    source: str = "custom"

    def to_dict(self) -> dict:
        d = asdict(self)
        # Normalize class label casing on write.
        d["topo_class"] = str(self.topo_class).upper()
        return d

    def is_topological(self) -> bool:
        return str(self.topo_class).upper() in _TOPO_POSITIVE


class TopoDB:
    """
    CSV-first DB wrapper with optional SQLite export/query.
    """

    def __init__(self, use_sql: bool = False):
        cfg = get_config()
        self.db_dir: Path = cfg.paths.raw_dir
        self.db_file: Path = self.db_dir / "topological_materials.csv"
        self.sql_path: Path = self.db_dir / "topological_materials.db"
        self.use_sql = use_sql
        self._df: pd.DataFrame | None = None

        if self.use_sql and self.sql_path.exists():
            self._df = self._load_from_sql()
        elif self.db_file.exists():
            self._df = self._load_from_csv()

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            if self.db_file.exists():
                self._df = self._load_from_csv()
            else:
                self._df = self._empty_df()
        return self._df

    def _empty_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=_DEFAULT_COLUMNS)

    def _load_from_csv(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.db_file)
        except Exception as exc:
            logger.warning(f"Failed to read {self.db_file}: {exc}. Using empty DB.")
            return self._empty_df()

        for col in _DEFAULT_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df["topo_class"] = df["topo_class"].astype(str).str.upper()
        return df[_DEFAULT_COLUMNS]

    def _load_from_sql(self) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.sql_path)
            df = pd.read_sql("SELECT * FROM materials", conn)
            conn.close()
            for col in _DEFAULT_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            df["topo_class"] = df["topo_class"].astype(str).str.upper()
            return df[_DEFAULT_COLUMNS]
        except Exception as exc:
            logger.warning(f"Failed to read SQL DB {self.sql_path}: {exc}")
            return self._empty_df()

    def load_seed_data(self):
        seed_df = pd.DataFrame(_SEED_ROWS)
        if self.df.empty:
            merged = seed_df.copy()
        else:
            merged = pd.concat([self.df, seed_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["jid"], keep="last")
        merged["topo_class"] = merged["topo_class"].astype(str).str.upper()
        self._df = merged[_DEFAULT_COLUMNS].reset_index(drop=True)
        self.save()
        if self.use_sql:
            self.to_sql()

    def save(self):
        self.save_csv()

    def save_csv(self, df: pd.DataFrame | None = None):
        out = df if df is not None else self.df
        self.db_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(self.db_file, index=False)

    def to_sql(self, db_path: Path | None = None):
        path = db_path or self.sql_path
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        self.df.to_sql("materials", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_formula ON materials (formula)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_topo ON materials (topo_class)")
        conn.commit()
        conn.close()

    def query_sql(self, sql_query: str) -> pd.DataFrame:
        if not self.sql_path.exists():
            self.to_sql()
        conn = sqlite3.connect(self.sql_path)
        try:
            return pd.read_sql(sql_query, conn)
        finally:
            conn.close()

    def fuzzy_search(self, query: str, cutoff: float = 0.6) -> pd.DataFrame:
        formulas = self.df["formula"].astype(str).tolist()
        matches = difflib.get_close_matches(query, formulas, n=5, cutoff=cutoff)
        if not matches:
            return self._empty_df()
        return self.df[self.df["formula"].isin(matches)].copy()

    def query(
        self,
        topo_class: str | None = None,
        elements: list[str] | None = None,
        band_gap_range: tuple | None = None,
        exact_formula: str | None = None,
    ) -> pd.DataFrame:
        df = self.df.copy()

        if exact_formula:
            df = df[df["formula"] == exact_formula]

        if topo_class:
            df = df[df["topo_class"] == str(topo_class).upper()]

        if band_gap_range:
            lo, hi = band_gap_range
            if lo is not None:
                df = df[df["band_gap"] >= lo]
            if hi is not None:
                df = df[df["band_gap"] <= hi]

        if elements:
            def has_elements(formula: str) -> bool:
                return all(el in str(formula) for el in elements)
            df = df[df["formula"].apply(has_elements)]

        return df.reset_index(drop=True)

    def add_material(self, formula: str, topo_class: str, properties: dict):
        jid = properties.get("jid", f"mat-{len(self.df):06d}")
        mat = TopoMaterial(
            jid=jid,
            formula=formula,
            space_group=int(properties.get("space_group", 0)),
            topo_class=str(topo_class).upper(),
            band_gap=float(properties.get("band_gap", 0.0)),
            source=str(properties.get("source", "custom")),
        )
        self.add_materials([mat])

    def add_materials(self, materials: list[TopoMaterial]):
        if not materials:
            return
        new_df = pd.DataFrame([m.to_dict() for m in materials])
        if self.df.empty:
            merged = new_df.copy()
        else:
            merged = pd.concat([self.df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["jid"], keep="last")
        merged["topo_class"] = merged["topo_class"].astype(str).str.upper()
        self._df = merged[_DEFAULT_COLUMNS].reset_index(drop=True)
        self.save()
        if self.use_sql:
            self.to_sql()

    def stats(self) -> dict:
        df = self.df
        by_class = df["topo_class"].value_counts().to_dict() if len(df) else {}
        by_source = df["source"].value_counts().to_dict() if len(df) else {}
        return {
            "total": int(len(df)),
            "by_class": by_class,
            "by_source": by_source,
        }
