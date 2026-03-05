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


def _coerce_non_negative_int(value: int, *, default: int, minimum: int = 0) -> int:
    if isinstance(value, bool):
        return max(int(default), int(minimum))
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return max(int(default), int(minimum))
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return max(int(default), int(minimum))
    rounded = int(round(parsed))
    if abs(parsed - rounded) > 1e-12:
        return max(int(default), int(minimum))
    return max(rounded, int(minimum))


def _coerce_int(value: int, *, default: int) -> int:
    if isinstance(value, bool):
        return int(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return int(default)
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return int(default)
    rounded = int(round(parsed))
    if abs(parsed - rounded) > 1e-12:
        return int(default)
    return int(rounded)


def _coerce_positive_float(value: float, *, default: float, floor: float = 1e-12) -> float:
    if isinstance(value, bool):
        return max(float(default), float(floor))
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return max(float(default), float(floor))
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return max(float(default), float(floor))
    return max(parsed, float(floor))


def _coerce_bool(value: bool, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        canon = value.strip().lower()
        if canon in {"1", "true", "yes", "y", "on"}:
            return True
        if canon in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


def _coerce_nonempty_string(value: str, *, default: str) -> str:
    text = str(value).strip()
    return text or str(default)


@dataclass
class PathConfig:
    """File system paths."""
    project_root: Path = _PROJECT_ROOT
    data_dir: Path | None = field(default=None)
    models_dir: Path | None = field(default=None)
    raw_dir: Path | None = field(default=None)
    processed_dir: Path | None = field(default=None)
    artifacts_dir: Path | None = field(default=None)

    def _normalize_path(self, candidate: Path | str | None, default: Path) -> Path:
        if candidate is None:
            base = default
        elif isinstance(candidate, bool):
            raise ValueError("path candidates must be path-like, not bool")
        elif isinstance(candidate, str):
            text = candidate.strip()
            base = default if not text else Path(text)
        else:
            base = Path(candidate)
        base = base.expanduser()
        if not base.is_absolute():
            base = self.project_root / base
        return base

    def __post_init__(self):
        self.project_root = self.project_root.expanduser().resolve()

        # Allow env override
        data_env = os.environ.get("ATLAS_DATA_DIR")
        data_candidate: Path | str | None = data_env if data_env else self.data_dir
        self.data_dir = self._normalize_path(data_candidate, self.project_root / "data")
        self.models_dir = self._normalize_path(self.models_dir, self.project_root / "models")
        self.raw_dir = self._normalize_path(self.raw_dir, self.data_dir / "raw")
        self.processed_dir = self._normalize_path(self.processed_dir, self.data_dir / "processed")
        self.artifacts_dir = self._normalize_path(self.artifacts_dir, self.project_root / "artifacts")

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

    def __post_init__(self):
        self.encut = _coerce_positive_float(self.encut, default=520.0)
        self.kpoints_density = _coerce_positive_float(self.kpoints_density, default=40.0)
        self.ediff = _coerce_positive_float(self.ediff, default=1e-6)
        self.sigma = _coerce_positive_float(self.sigma, default=0.05)
        self.nkpts_line = _coerce_non_negative_int(self.nkpts_line, default=40, minimum=1)
        self.nbands_factor = _coerce_positive_float(self.nbands_factor, default=1.5)
        self.ismear = _coerce_int(self.ismear, default=0)


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

    def __post_init__(self):
        self.num_interactions = _coerce_non_negative_int(self.num_interactions, default=2, minimum=1)
        self.max_ell = _coerce_non_negative_int(self.max_ell, default=3, minimum=0)
        self.correlation = _coerce_non_negative_int(self.correlation, default=3, minimum=1)
        self.hidden_irreps = _coerce_nonempty_string(self.hidden_irreps, default="128x0e + 128x1o")
        self.r_max = _coerce_positive_float(self.r_max, default=5.0)
        self.batch_size = _coerce_non_negative_int(self.batch_size, default=16, minimum=1)
        self.lr = _coerce_positive_float(self.lr, default=0.01)
        self.max_epochs = _coerce_non_negative_int(self.max_epochs, default=500, minimum=1)
        self.patience = _coerce_non_negative_int(self.patience, default=50, minimum=1)
        self.weight_decay = _coerce_positive_float(self.weight_decay, default=5e-7)
        self.energy_weight = _coerce_positive_float(self.energy_weight, default=1.0)
        self.forces_weight = _coerce_positive_float(self.forces_weight, default=10.0)
        self.stress_weight = _coerce_positive_float(self.stress_weight, default=1.0)


@dataclass
class TrainConfig:
    """General training parameters for Phase 1/Phase 2 models."""
    device: str = "auto"
    seed: int = 42
    deterministic: bool = True
    num_workers: int = 4
    pin_memory: bool = True

    def __post_init__(self):
        self.device = _coerce_nonempty_string(self.device, default="auto")
        self.seed = _coerce_non_negative_int(self.seed, default=42, minimum=0)
        self.deterministic = _coerce_bool(self.deterministic, default=True)
        self.num_workers = _coerce_non_negative_int(self.num_workers, default=4, minimum=0)
        self.pin_memory = _coerce_bool(self.pin_memory, default=True)


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
    fallback_methods: tuple[str, ...] = ("gp_active_learning", "descriptor_microstructure")

    def __post_init__(self):
        self.model_name = _coerce_nonempty_string(self.model_name, default="mace_default")
        self.relaxer_name = _coerce_nonempty_string(self.relaxer_name, default="ase_bfgs")
        self.screener_name = _coerce_nonempty_string(self.screener_name, default="crabnet_default")
        self.evaluator_name = _coerce_nonempty_string(
            self.evaluator_name,
            default="rustworkx_pathfinder",
        )
        self.data_source_key = _coerce_nonempty_string(self.data_source_key, default="jarvis_dft")
        self.method_key = _coerce_nonempty_string(
            self.method_key,
            default="workflow_reproducible_graph",
        )
        if isinstance(self.fallback_methods, (list, tuple)):
            fallbacks = [
                str(item).strip()
                for item in self.fallback_methods
                if str(item).strip()
            ]
        else:
            single = str(self.fallback_methods).strip()
            fallbacks = [single] if single else []
        if not fallbacks:
            fallbacks = ["gp_active_learning", "descriptor_microstructure"]
        self.fallback_methods = tuple(fallbacks)

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
        except Exception:
            # Keep config initialization robust even when the requested device
            # is invalid or torch is unavailable in minimal environments.
            self.device = "cpu"

    @staticmethod
    def get_device(requested: str = "auto"):
        """Resolve device string to torch.device."""
        import torch
        req = str(requested).strip().lower()
        if req == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            # Potential check for MPS (Mac)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        try:
            return torch.device(requested)
        except Exception as exc:
            raise ValueError(f"Invalid device specification: {requested!r}") from exc

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
