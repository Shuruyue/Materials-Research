"""
ATLAS Configuration Module

Centralizes all project paths and hyperparameters.
No external API keys required — all data is freely downloadable.

Optimization:
- Device Management: Global `get_device()` helper.
- Environment Support: Override paths via .env if needed.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathConfig:
    """File system paths."""
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = field(default=None)
    models_dir: Path = field(default=None)
    raw_dir: Path = field(default=None)
    processed_dir: Path = field(default=None)
    artifacts_dir: Path = field(default=None)

    def __post_init__(self):
        # Allow env override
        data_env = os.environ.get("ATLAS_DATA_DIR")
        if data_env:
            self.data_dir = Path(data_env)
        else:
            self.data_dir = self.data_dir or self.project_root / "data"

        self.models_dir = self.models_dir or self.project_root / "models"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.artifacts_dir = self.project_root / "artifacts"

    def ensure_dirs(self):
        """Create all directories if they don't exist."""
        for d in [self.data_dir, self.models_dir, self.raw_dir, self.processed_dir, self.artifacts_dir]:
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
class TrainConfig:
    """General training parameters for Phase 1/Phase 2 models."""
    device: str = "auto"
    seed: int = 42
    deterministic: bool = True
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ProfileConfig:
    """Algorithm profiles to dynamically select registered components."""
    # Definer which model/relaxer to use from Registry
    model_name: str = "mace_default"
    relaxer_name: str = "ase_bfgs"
    screener_name: str = "crabnet_default"
    evaluator_name: str = "rustworkx_pathfinder"
    data_source_key: str = "jarvis_dft"
    method_key: str = "workflow_reproducible_graph"
    fallback_methods: tuple[str, str] = ("gp_active_learning", "descriptor_microstructure")

@dataclass
class Config:
    """Master configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    dft: DFTConfig = field(default_factory=DFTConfig)
    mace: MACEConfig = field(default_factory=MACEConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)

    def __post_init__(self):
        self.paths.ensure_dirs()
        self._set_device()

    def _set_device(self):
        try:
            self.device = self.get_device(self.train.device)
        except ImportError:
            self.device = "cpu"

    @staticmethod
    def get_device(requested: str = "auto"):
        """Resolve device string to torch.device."""
        import torch
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            # Potential check for MPS (Mac)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                 return torch.device("mps")
            return torch.device("cpu")
        return torch.device(requested)

    @staticmethod
    def source_label(source_key: str) -> str:
        labels = {
            "jarvis_dft": "JARVIS-DFT",
            "materials_project": "Materials Project",
            "matbench": "Matbench",
            "oqmd": "OQMD",
        }
        return labels.get(source_key, source_key)

    def summary(self) -> str:
        fallback_str = ", ".join(self.profile.fallback_methods)
        lines = [
            "=" * 60,
            "ATLAS Configuration",
            "=" * 60,
            f"Project root : {self.paths.project_root}",
            f"Data dir     : {self.paths.data_dir}",
            f"Data Source  : {self.source_label(self.profile.data_source_key)}",
            f"Device       : {self.device}",
            f"Seed         : {self.train.seed}",
            f"Deterministic: {self.train.deterministic}",
            f"Method       : {self.profile.method_key}",
            f"Fallbacks    : {fallback_str}",
            f"Model Profile: {self.profile.model_name}",
            f"Relaxer      : {self.profile.relaxer_name}",
            f"Screener     : {self.profile.screener_name}",
            f"Evaluator    : {self.profile.evaluator_name}",
            "=" * 60,
        ]
        return "\n".join(lines)


_config = None

def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
