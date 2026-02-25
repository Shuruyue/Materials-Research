import logging
import os
import sys

import ase
import torch

logger = logging.getLogger(__name__)

# atlas/discovery/stability/mepin.py -> project root -> references/recisic/mepin
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
MEPIN_REPO_PATH = os.path.join(ROOT_DIR, "references", "recisic", "mepin")
if MEPIN_REPO_PATH not in sys.path:
    sys.path.append(MEPIN_REPO_PATH)

try:
    from mepin.model.modules import TripleCrossPaiNNModule
    from mepin.tools.inference import create_reaction_batch
    _MEPIN_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional external repo
    TripleCrossPaiNNModule = None
    create_reaction_batch = None
    _MEPIN_IMPORT_ERROR = exc


class MEPINStabilityEvaluator:
    """
    Wrapper for MEPIN (minimum-energy-path inference).
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_type: str = "cyclo_L",
    ):
        if TripleCrossPaiNNModule is None:
            raise ImportError(
                f"Could not import MEPIN from {MEPIN_REPO_PATH}. "
                f"Underlying error: {_MEPIN_IMPORT_ERROR}"
            )

        self.device = device
        self.model_type = model_type

        if checkpoint_path is None:
            if model_type not in ("cyclo_L", "t1x_L"):
                raise ValueError(f"Unknown model_type: {model_type}. Supported: cyclo_L, t1x_L")
            checkpoint_path = os.path.join(MEPIN_REPO_PATH, "ckpt", f"{model_type}.ckpt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"MEPIN checkpoint not found at {checkpoint_path}")

        self.model = TripleCrossPaiNNModule.load_from_checkpoint(checkpoint_path, map_location=device)
        self.model.eval()
        self.model.to(device)

    def predict_path(
        self,
        reactant: ase.Atoms,
        product: ase.Atoms,
        num_images: int = 20,
    ) -> list[ase.Atoms]:
        if create_reaction_batch is None:
            raise RuntimeError("MEPIN inference backend is unavailable.")

        use_geodesic = "G" in self.model_type
        batch = create_reaction_batch(
            reactant,
            product,
            interp_traj=None,
            use_geodesic=use_geodesic,
            num_images=num_images,
        )
        batch = batch.to(self.device)

        with torch.no_grad():
            out = self.model(batch)
            n_atoms = len(reactant)
            out = out.reshape(num_images, n_atoms, 3)
            output_positions = out.cpu().numpy()

        trajectory: list[ase.Atoms] = []
        for i in range(num_images):
            atoms = reactant.copy()
            atoms.set_positions(output_positions[i])
            trajectory.append(atoms)
        return trajectory


__all__ = ["MEPINStabilityEvaluator"]
