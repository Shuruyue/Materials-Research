"""
ATLAS Configuration Module

Centralizes all project paths and hyperparameters.
No external API keys required — all data is freely downloadable.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathConfig:
    """File system paths."""
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = field(default=None)
    models_dir: Path = field(default=None)
    raw_dir: Path = field(default=None)
    processed_dir: Path = field(default=None)

    def __post_init__(self):
        self.data_dir = self.data_dir or self.project_root / "data"
        self.models_dir = self.models_dir or self.project_root / "models"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

    def ensure_dirs(self):
        """Create all directories if they don't exist."""
        for d in [self.data_dir, self.models_dir, self.raw_dir, self.processed_dir]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class DFTConfig:
    """DFT calculation parameters."""
    encut: float = 520.0          # plane-wave cutoff (eV)
    kpoints_density: float = 40.0  # k-points per Å⁻¹
    ediff: float = 1e-6           # SCF convergence (eV)
    ismear: int = 0               # Gaussian smearing
    sigma: float = 0.05           # smearing width (eV)
    nkpts_line: int = 40          # k-points per high-symmetry line
    nbands_factor: float = 1.5    # factor for number of bands


@dataclass
class MACEConfig:
    """MACE neural network potential hyperparameters."""
    # Architecture
    num_interactions: int = 2
    max_ell: int = 3
    correlation: int = 3
    hidden_irreps: str = "128x0e + 128x1o"
    r_max: float = 5.0            # cutoff radius (Å)

    # Training
    batch_size: int = 16
    lr: float = 0.01
    max_epochs: int = 500
    patience: int = 50
    weight_decay: float = 5e-7
    
    # Loss weights
    energy_weight: float = 1.0
    forces_weight: float = 10.0
    stress_weight: float = 1.0


@dataclass
class Config:
    """Master configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    dft: DFTConfig = field(default_factory=DFTConfig)
    mace: MACEConfig = field(default_factory=MACEConfig)

    def __post_init__(self):
        self.paths.ensure_dirs()

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "ATLAS Configuration",
            "=" * 60,
            f"Project root : {self.paths.project_root}",
            f"Data dir     : {self.paths.data_dir}",
            f"Models dir   : {self.paths.models_dir}",
            f"Data source  : JARVIS-DFT (no API key required)",
            f"MACE cutoff  : {self.mace.r_max} Å",
            f"MACE epochs  : {self.mace.max_epochs}",
            f"DFT ENCUT    : {self.dft.encut} eV",
            "=" * 60,
        ]
        return "\n".join(lines)


_config = None

def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
