"""
Topological Materials Database Interface

Loads and queries the known topological materials from the
Topological Quantum Chemistry database (topologicalquantumchemistry.org)
and cross-references with Materials Project.

Classifications:
    - TI : Topological Insulator (Z₂ nontrivial)
    - TSM: Topological Semimetal (Weyl / Dirac)
    - NLSM: Nodal-Line Semimetal
    - TCI: Topological Crystalline Insulator
    - TRIVIAL: Topologically trivial
"""

import json
import csv
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import pandas as pd
from atlas.config import get_config


# Topological classification labels
TOPO_CLASSES = {
    "TI": "Topological Insulator",
    "TSM": "Topological Semimetal",
    "NLSM": "Nodal-Line Semimetal",
    "TCI": "Topological Crystalline Insulator",
    "TRIVIAL": "Trivial Insulator",
    "UNKNOWN": "Unknown / Not classified",
}


@dataclass
class TopoMaterial:
    """A material with topological classification."""
    material_id: str
    formula: str
    space_group: int
    topo_class: str  # One of TOPO_CLASSES keys
    band_gap: Optional[float] = None
    z2_invariant: Optional[tuple] = None
    chern_number: Optional[int] = None
    source: str = "TQC"  # TQC, MP, manual

    def is_topological(self) -> bool:
        return self.topo_class not in ("TRIVIAL", "UNKNOWN")


class TopoDB:
    """
    Topological Materials Database manager.

    Manages a local database of materials with their topological
    classifications. Supports loading from multiple sources and
    querying by properties.
    """

    def __init__(self):
        cfg = get_config()
        self.db_dir = cfg.paths.data_dir / "topo_db"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_file = self.db_dir / "topological_materials.csv"
        self._df: Optional[pd.DataFrame] = None

    @property
    def df(self) -> pd.DataFrame:
        """Lazy-load the database."""
        if self._df is None:
            if self.db_file.exists():
                self._df = pd.read_csv(self.db_file)
            else:
                self._df = pd.DataFrame(columns=[
                    "material_id", "formula", "space_group",
                    "topo_class", "band_gap", "source",
                ])
        return self._df

    def add_materials(self, materials: list[TopoMaterial]) -> None:
        """Add materials to the database."""
        new_rows = [
            {
                "material_id": m.material_id,
                "formula": m.formula,
                "space_group": m.space_group,
                "topo_class": m.topo_class,
                "band_gap": m.band_gap,
                "source": m.source,
            }
            for m in materials
        ]
        new_df = pd.DataFrame(new_rows)
        self._df = pd.concat([self.df, new_df], ignore_index=True)
        self._df.drop_duplicates(subset="material_id", keep="last", inplace=True)

    def save(self) -> None:
        """Save database to CSV."""
        self.df.to_csv(self.db_file, index=False)
        print(f"  Saved {len(self.df)} materials to {self.db_file}")

    def query(
        self,
        topo_class: Optional[str] = None,
        elements: Optional[list[str]] = None,
        band_gap_range: Optional[tuple[float, float]] = None,
    ) -> pd.DataFrame:
        """
        Query the topological materials database.

        Args:
            topo_class: Filter by topological class (e.g., "TI", "TSM")
            elements: Filter by elements (material must contain ALL listed)
            band_gap_range: (min, max) band gap in eV

        Returns:
            Filtered DataFrame
        """
        result = self.df.copy()

        if topo_class:
            result = result[result["topo_class"] == topo_class]

        if band_gap_range:
            lo, hi = band_gap_range
            result = result[
                (result["band_gap"] >= lo) & (result["band_gap"] <= hi)
            ]

        if elements:
            def contains_all(formula):
                return all(el in str(formula) for el in elements)
            result = result[result["formula"].apply(contains_all)]

        return result

    def stats(self) -> dict:
        """Get database statistics."""
        return {
            "total": len(self.df),
            "by_class": self.df["topo_class"].value_counts().to_dict(),
            "by_source": self.df["source"].value_counts().to_dict(),
        }

    def load_seed_data(self) -> None:
        """
        Load well-known topological materials as seed data.
        These are experimentally confirmed topological materials.
        """
        seeds = [
            TopoMaterial("mp-541837", "Bi2Se3", 166, "TI", 0.3),
            TopoMaterial("mp-34202", "Bi2Te3", 166, "TI", 0.165),
            TopoMaterial("mp-22598", "Sb2Te3", 166, "TI", 0.28),
            TopoMaterial("mp-22875", "Bi2Te2Se", 166, "TI", 0.35),
            TopoMaterial("mp-567290", "SmB6", 221, "TI", 0.02),
            TopoMaterial("mp-23092", "SnTe", 225, "TCI", 0.18),
            TopoMaterial("mp-7631", "Pb0.7Sn0.3Se", 225, "TCI", 0.0),
            TopoMaterial("mp-2815", "Na3Bi", 194, "TSM", 0.0),
            TopoMaterial("mp-5765", "Cd3As2", 137, "TSM", 0.0),
            TopoMaterial("mp-961652", "TaAs", 109, "TSM", 0.0),
            TopoMaterial("mp-10172", "NbAs", 109, "TSM", 0.0),
            TopoMaterial("mp-2998", "TaP", 109, "TSM", 0.0),
            TopoMaterial("mp-672", "NbP", 109, "TSM", 0.0),
            TopoMaterial("mp-3163", "WTe2", 31, "TSM", 0.0),
            TopoMaterial("mp-2070", "MoTe2", 11, "TSM", 0.0),
            TopoMaterial("mp-19717", "ZrSiS", 129, "NLSM", 0.0),
            TopoMaterial("mp-27175", "PbTaSe2", 187, "NLSM", 0.0),
            TopoMaterial("mp-149", "Si", 227, "TRIVIAL", 1.11),
            TopoMaterial("mp-2534", "GaAs", 216, "TRIVIAL", 1.42),
            TopoMaterial("mp-22862", "NaCl", 225, "TRIVIAL", 5.0),
        ]

        self.add_materials(seeds)
        self.save()
        print(f"  Loaded {len(seeds)} seed materials "
              f"({sum(1 for s in seeds if s.is_topological())} topological)")
