"""
Data source registry for materials datasets.

This module centralizes dataset metadata so experiments can be reproduced
with explicit source names and citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class DataSourceSpec:
    """Dataset descriptor used across training/inference pipelines."""

    key: str
    name: str
    domain: str
    primary_targets: List[str] = field(default_factory=list)
    url: str = ""
    citation: str = ""


class DataSourceRegistry:
    """Simple in-memory registry for named data sources."""

    def __init__(self):
        self._sources: Dict[str, DataSourceSpec] = {}

    def register(self, spec: DataSourceSpec):
        self._sources[spec.key] = spec

    def get(self, key: str) -> DataSourceSpec:
        if key not in self._sources:
            available = ", ".join(sorted(self._sources.keys()))
            raise KeyError(f"Unknown data source '{key}'. Available: {available}")
        return self._sources[key]

    def list_keys(self) -> List[str]:
        return sorted(self._sources.keys())

    def list_all(self) -> List[DataSourceSpec]:
        return [self._sources[k] for k in self.list_keys()]


DATA_SOURCES = DataSourceRegistry()

# Inorganic/metal/semiconductor-focused defaults for this project.
DATA_SOURCES.register(
    DataSourceSpec(
        key="jarvis_dft",
        name="JARVIS-DFT",
        domain="inorganic_crystals",
        primary_targets=[
            "formation_energy",
            "band_gap",
            "bulk_modulus",
            "shear_modulus",
        ],
        url="https://jarvis.nist.gov/",
        citation="Choudhary et al., NPJ Comput Mater (2020).",
    )
)
DATA_SOURCES.register(
    DataSourceSpec(
        key="materials_project",
        name="Materials Project",
        domain="inorganic_crystals",
        primary_targets=[
            "formation_energy",
            "band_gap",
            "elasticity",
            "dielectric",
        ],
        url="https://materialsproject.org/",
        citation="Jain et al., APL Materials (2013).",
    )
)
DATA_SOURCES.register(
    DataSourceSpec(
        key="matbench",
        name="Matbench",
        domain="benchmark_suite",
        primary_targets=["task_specific"],
        url="https://matbench.materialsproject.org/",
        citation="Dunn et al., NPJ Comput Mater (2020).",
    )
)
DATA_SOURCES.register(
    DataSourceSpec(
        key="oqmd",
        name="OQMD",
        domain="inorganic_crystals",
        primary_targets=["formation_energy"],
        url="https://oqmd.org/",
        citation="Saal et al., JOM (2013).",
    )
)
