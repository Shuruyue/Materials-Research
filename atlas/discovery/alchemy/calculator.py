from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import ase
import numpy as np
import torch
import torch.nn.functional as F
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

logger = logging.getLogger(__name__)


def _normalize_device(device: str) -> str:
    value = str(device).strip().lower()
    if value == "cpu":
        return "cpu"
    if value.startswith("cuda"):
        if torch.cuda.is_available():
            return value
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    logger.warning("Unknown device '%s'; falling back to CPU.", device)
    return "cpu"


def _is_boolean_like(value: Any) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}


def _load_alchemy_api() -> tuple[Any, Any]:
    try:
        from .model import AlchemyManager, load_alchemical_model
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "Alchemical model dependencies are unavailable. "
            "Install optional stack (mace + e3nn + torch-geometric) to use AlchemicalMACECalculator."
        ) from exc
    return AlchemyManager, load_alchemical_model


def _validate_alchemical_pairs(
    alchemical_pairs: Sequence[Sequence[tuple[int, int]]],
    *,
    num_atoms: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if isinstance(alchemical_pairs, (str, bytes)):
        raise TypeError("alchemical_pairs must be a sequence of pair-groups, not a string")
    if len(alchemical_pairs) == 0:
        raise ValueError("alchemical_pairs must not be empty")

    normalized_groups: list[tuple[tuple[int, int], ...]] = []
    for group_idx, group in enumerate(alchemical_pairs):
        if isinstance(group, (str, bytes)):
            raise TypeError(
                f"alchemical_pairs[{group_idx}] must be a sequence of (atom_index, atomic_number) pairs"
            )
        pairs: list[tuple[int, int]] = []
        for pair_idx, pair in enumerate(group):
            if not isinstance(pair, Sequence) or len(pair) != 2:
                raise TypeError(
                    f"alchemical_pairs[{group_idx}][{pair_idx}] must be a 2-tuple/list"
                )
            idx_raw, z_raw = pair
            if _is_boolean_like(idx_raw) or _is_boolean_like(z_raw):
                raise ValueError(
                    f"alchemical_pairs[{group_idx}][{pair_idx}] entries must be integer-valued, not boolean"
                )
            idx = int(idx_raw)
            z = int(z_raw)
            if idx < 0 or idx >= num_atoms:
                raise ValueError(
                    f"alchemical atom index out of range: {idx} (num_atoms={num_atoms})"
                )
            if z <= 0:
                raise ValueError(f"atomic_number must be positive, got {z}")
            pairs.append((idx, z))
        if not pairs:
            raise ValueError(f"alchemical_pairs[{group_idx}] must not be empty")
        normalized_groups.append(tuple(pairs))
    return tuple(normalized_groups)


def _as_weight_array(alchemical_weights: Sequence[float], *, expected_size: int) -> np.ndarray:
    if expected_size <= 0:
        raise ValueError("expected_size must be positive")
    for idx, value in enumerate(alchemical_weights):
        if _is_boolean_like(value):
            raise ValueError(f"alchemical_weights[{idx}] must be numeric, not boolean")
    weights = np.asarray(alchemical_weights, dtype=np.float32).reshape(-1)
    if weights.size != int(expected_size):
        raise ValueError(
            f"alchemical_weights size mismatch: got {weights.size}, expected {expected_size}"
        )
    if not np.isfinite(weights).all():
        raise ValueError("alchemical_weights must be finite")
    if np.any(weights < -1e-6) or np.any(weights > 1.0 + 1e-6):
        raise ValueError("alchemical_weights must lie within [0, 1] (with small tolerance)")
    return np.clip(weights, 0.0, 1.0)


class AlchemicalMACECalculator(Calculator):
    """
    ASE Calculator for Alchemical MACE.
    Supports continuous interpolation of atomic species for optimization.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress", "alchemical_grad"]

    def __init__(
        self,
        atoms: ase.Atoms,
        alchemical_pairs: Sequence[Sequence[tuple[int, int]]],
        alchemical_weights: Sequence[float],
        device: str = "cpu",
        model_size: str = "medium",
        model_path: str | None = None,  # Reserved custom checkpoint path.
    ):
        """
        Initialize the Alchemical MACE calculator.

        Args:
            atoms: ASE Atoms object (provides initial structure and species).
            alchemical_pairs: List of lists of (index, atomic_number) tuples.
            alchemical_weights: Initial weights for the alchemical species [0, 1].
            device: 'cpu' or 'cuda'.
            model_size: 'small', 'medium', or 'large' (default MACE-MP models).
            model_path: Reserved for future custom checkpoint loading.
        """
        Calculator.__init__(self)
        self.results = {}
        if not isinstance(atoms, ase.Atoms):
            raise TypeError(f"atoms must be ase.Atoms, got {type(atoms)!r}")
        if len(atoms) == 0:
            raise ValueError("atoms must contain at least one site")

        self.device = _normalize_device(device)
        self.atoms = atoms
        self.alchemical_pairs = _validate_alchemical_pairs(
            alchemical_pairs,
            num_atoms=len(atoms),
        )
        AlchemyManager, load_alchemical_model = _load_alchemy_api()

        # Load model
        if model_path:
            logger.warning(
                "Custom model path is not supported yet (received: %s); "
                "falling back to model_size='%s'.",
                model_path,
                model_size,
            )
        self.model = load_alchemical_model(model_size=model_size, device=self.device)

        # Freeze model parameters (we only optimize alchemical weights or geometry)
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize Alchemy Manager
        # AtomicNumberTable is usually in mace.tools.
        from mace.tools import AtomicNumberTable

        z_table = AtomicNumberTable([int(z) for z in self.model.atomic_numbers])
        r_max = float(self.model.r_max.item())

        weights = _as_weight_array(
            alchemical_weights,
            expected_size=len(self.alchemical_pairs),
        )
        alchemical_weights_tensor = torch.tensor(
            weights,
            dtype=torch.float32,
            device=self.device,
        )

        self.alchemy_manager = AlchemyManager(
            atoms=atoms,
            alchemical_pairs=self.alchemical_pairs,
            alchemical_weights=alchemical_weights_tensor,
            z_table=z_table,
            r_max=r_max,
        ).to(self.device)

        # Optimization control
        self.calculate_alchemical_grad = False
        self.alchemy_manager.alchemical_weights.requires_grad = False

        self.num_atoms = len(atoms)

    def set_alchemical_weights(self, alchemical_weights: Sequence[float]):
        """Update the alchemical mixing weights."""
        weights = _as_weight_array(
            alchemical_weights,
            expected_size=len(self.alchemical_pairs),
        )
        tensor_weights = torch.tensor(
            weights,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            self.alchemy_manager.alchemical_weights.copy_(tensor_weights)
        # Force recalculation without clearing self.atoms (which reset() does).
        self.results = {}

    def get_alchemical_atomic_masses(self) -> np.ndarray:
        """
        Calculate effective atomic masses based on current alchemical weights.
        Useful for MD or vibrational analysis.
        """
        # 1. Get masses of all possible species in the system.
        node_masses = ase.data.atomic_masses[self.alchemy_manager.atomic_numbers]

        # 2. Get current weights for these nodes.
        # Pad with 1.0 for fixed atoms (index 0).
        weights = self.alchemy_manager.alchemical_weights.data
        weights = F.pad(weights, (1, 0), "constant", 1.0).cpu().numpy()
        node_weights = weights[self.alchemy_manager.weight_indices]

        # 3. Sum weighted masses for each original atom site.
        atom_masses = np.zeros(self.num_atoms, dtype=np.float32)
        np.add.at(atom_masses, self.alchemy_manager.atom_indices, node_masses * node_weights)
        return atom_masses

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Perform the electronic structure calculation (Energy, Forces, Stress).
        """
        Calculator.calculate(self, atoms, properties, system_changes)

        # Prepare inputs.
        tensor_kwargs = {"dtype": torch.float32, "device": self.device}
        positions = torch.tensor(self.atoms.get_positions(), **tensor_kwargs)
        cell = torch.tensor(self.atoms.get_cell().array, **tensor_kwargs)

        # Toggle gradient calculation for alchemical weights.
        if self.calculate_alchemical_grad:
            self.alchemy_manager.alchemical_weights.requires_grad = True

        # Build batch.
        batch = self.alchemy_manager(positions, cell).to(self.device)

        grad = np.zeros(self.alchemy_manager.alchemical_weights.shape[0], dtype=np.float32)
        out = None
        try:
            # Forward pass.
            if self.calculate_alchemical_grad:
                out = self.model(
                    batch,
                    compute_stress=True,
                    compute_alchemical_grad=True,
                    retain_graph=True,
                )
                (grad_weights,) = torch.autograd.grad(
                    outputs=[out["energy"]],
                    inputs=[self.alchemy_manager.alchemical_weights],
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                if grad_weights is not None:
                    grad = grad_weights.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                out = self.model(batch, retain_graph=False, compute_stress=True)
        finally:
            self.alchemy_manager.alchemical_weights.requires_grad = False
            self.alchemy_manager.alchemical_weights.grad = None

        if out is None:
            raise RuntimeError("Alchemical model failed to produce outputs.")
        required_keys = {"energy", "forces", "stress"}
        missing = [key for key in required_keys if key not in out]
        if missing:
            raise KeyError(f"Alchemical model output missing keys: {missing}")
        if not torch.isfinite(out["energy"]).all():
            raise RuntimeError("Model returned non-finite energy.")
        if not torch.isfinite(out["forces"]).all():
            raise RuntimeError("Model returned non-finite forces.")
        if not torch.isfinite(out["stress"]).all():
            raise RuntimeError("Model returned non-finite stress.")

        # Collect results.
        forces = out["forces"].detach().cpu().numpy()
        if forces.ndim != 2 or forces.shape[1] != 3:
            raise RuntimeError(f"Unexpected forces shape from model: {forces.shape}")
        stress = out["stress"][0].detach().cpu().numpy()
        if stress.shape != (3, 3):
            raise RuntimeError(f"Unexpected stress shape from model: {stress.shape}")

        energy_value = float(out["energy"].item())
        self.results = {
            "energy": energy_value,
            "free_energy": energy_value,
            "forces": forces,
            "stress": full_3x3_to_voigt_6_stress(stress),
            "alchemical_grad": grad,
        }
