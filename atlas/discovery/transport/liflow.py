import logging
import os
import sys

import ase
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Resolve local reference repo path:
# atlas/discovery/transport/liflow.py -> project root -> references/recisic/liflow
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
LIFLOW_REPO_PATH = os.path.join(ROOT_DIR, "references", "recisic", "liflow")
if LIFLOW_REPO_PATH not in sys.path:
    sys.path.append(LIFLOW_REPO_PATH)

try:
    from liflow.model.modules import FlowModule
    from liflow.utils.inference import FlowSimulator
    from liflow.utils.prior import get_prior
    _LIFLOW_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on optional external repo
    FlowModule = None
    FlowSimulator = None
    get_prior = None
    _LIFLOW_IMPORT_ERROR = exc


class LiFlowEvaluator:
    """
    Wrapper for LiFlow (flow-matching for atomic transport).
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        element_index_path: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temp_list: list[int] | None = None,
    ):
        if FlowModule is None:
            raise ImportError(
                f"Could not import LiFlow from {LIFLOW_REPO_PATH}. "
                f"Underlying error: {_LIFLOW_IMPORT_ERROR}"
            )

        self.device = device
        self.temp_list = temp_list or [600, 800, 1000]

        if checkpoint_path is None:
            checkpoint_path = os.path.join(LIFLOW_REPO_PATH, "ckpt", "P_universal.ckpt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"LiFlow checkpoint not found at {checkpoint_path}")

        self.model = FlowModule.load_from_checkpoint(checkpoint_path, map_location=device)
        self.model.eval()
        self.model.to(device)

        self.element_idx = self._load_element_index(element_index_path)

        if hasattr(self.model, "cfg") and hasattr(self.model.cfg, "propagate_prior"):
            cfg_prior = self.model.cfg.propagate_prior
            self.prior = get_prior(cfg_prior.class_name, **cfg_prior.params, seed=42)
        else:
            logger.warning(
                "LiFlow propagate_prior config missing, using AdaptiveMaxwellBoltzmannPrior fallback."
            )
            self.prior = get_prior("AdaptiveMaxwellBoltzmannPrior", seed=42)

    def _load_element_index(self, path: str | None) -> np.ndarray:
        if path and os.path.exists(path):
            return np.load(path)

        default_path = os.path.join(LIFLOW_REPO_PATH, "data", "universal", "element_index.npy")
        if os.path.exists(default_path):
            return np.load(default_path)

        logger.warning(
            "LiFlow element_index.npy missing, using fallback mapping Z->Z-1. "
            "Transport estimates may be noisy."
        )
        mapping = np.arange(119, dtype=int) - 1
        mapping[0] = 0
        return mapping

    def simulate(
        self,
        atoms: ase.Atoms,
        steps: int = 500,
        flow_steps: int = 10,
    ) -> tuple[list[ase.Atoms], float]:
        if FlowSimulator is None:
            raise RuntimeError("LiFlow simulator backend is unavailable.")

        temp = self.temp_list[0]
        simulator = FlowSimulator(
            propagate_model=self.model,
            propagate_prior=self.prior,
            atomic_numbers=atoms.get_atomic_numbers(),
            element_idx=self.element_idx,
            lattice=atoms.cell.array,
            temp=temp,
            correct_model=None,
            correct_prior=None,
            pbc=True,
            scale_Li_index=1,
            scale_frame_index=0,
        )

        traj_pos = simulator.run(
            positions=atoms.get_positions(),
            steps=steps,
            flow_steps=flow_steps,
            solver="euler",
            verbose=False,
            fix_com=True,
        )

        trajectory: list[ase.Atoms] = []
        for pos in traj_pos:
            new_atoms = atoms.copy()
            new_atoms.set_positions(pos)
            trajectory.append(new_atoms)

        # Rough diffusion estimate from Li MSD between first/last frame.
        diff_coeff = 0.0
        z = atoms.get_atomic_numbers()
        li_mask = z == 3
        if np.any(li_mask) and len(traj_pos) >= 2:
            disp = traj_pos[-1][li_mask] - traj_pos[0][li_mask]
            msd = float(np.mean(np.sum(disp**2, axis=1)))
            # Approximate dt in ps (coarse placeholder).
            dt_ps = max(steps * 1e-3, 1e-8)
            diff_coeff = msd / (6.0 * dt_ps)

        return trajectory, diff_coeff


__all__ = ["LiFlowEvaluator"]
