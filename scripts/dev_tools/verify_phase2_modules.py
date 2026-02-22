#!/usr/bin/env python3
"""
Verification script for current Phase 2 pipeline modules.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase2Verification")

# PyTorch 2.6+ defaults torch.load(weights_only=True), while e3nn constants
# use python's built-in `slice` in serialized content.
try:
    torch.serialization.add_safe_globals([slice])
except AttributeError:
    pass


def verify_phase2() -> bool:
    logger.info("Verifying Phase 2 modules...")

    # 1) Core model imports + lightweight instantiation
    try:
        from atlas.models.cgcnn import CGCNN
        from atlas.models.equivariant import EquivariantGNN
        from atlas.models.graph_builder import CrystalGraphBuilder
        from atlas.models.multi_task import MultiTaskGNN

        cgcnn = CGCNN(hidden_dim=32, n_conv=2, n_fc=1)
        e3nn = EquivariantGNN(
            irreps_hidden="16x0e + 8x1o",
            max_ell=1,
            n_layers=2,
            n_radial_basis=8,
            radial_hidden=32,
        )
        _ = MultiTaskGNN(
            encoder=e3nn,
            tasks={
                "formation_energy": {"type": "scalar"},
                "band_gap": {"type": "scalar"},
            },
            embed_dim=e3nn.scalar_dim,
        )
        _ = CrystalGraphBuilder(compute_3body=True)

        logger.info("[PASS] Core model modules are importable/instantiable")
        logger.info("[PASS] CGCNN hidden_dim=%s", cgcnn.hidden_dim)
    except Exception as exc:
        logger.error("[FAIL] Core model verification failed: %s", exc)
        return False

    # 2) Training utilities used by phase2 scripts
    try:
        from atlas.training.checkpoint import CheckpointManager
        from atlas.training.normalizers import MultiTargetNormalizer
        from atlas.training.run_utils import resolve_run_dir, write_run_manifest

        _ = CheckpointManager(PROJECT_ROOT / "artifacts" / "_verify_phase2_tmp")
        _ = MultiTargetNormalizer()
        _ = resolve_run_dir  # symbol check
        _ = write_run_manifest  # symbol check
        logger.info("[PASS] Training utility modules are available")
    except Exception as exc:
        logger.error("[FAIL] Training utility verification failed: %s", exc)
        return False

    # 3) Inference loader utility
    try:
        from atlas.models.utils import load_phase2_model

        _ = load_phase2_model  # symbol check
        logger.info("[PASS] Phase 2 model loader utility is available")
    except Exception as exc:
        logger.error("[FAIL] Inference utility verification failed: %s", exc)
        return False

    # 4) Ensure key phase2 scripts exist
    required_scripts = [
        PROJECT_ROOT / "scripts/phase2_multitask/run_phase2.py",
        PROJECT_ROOT / "scripts/phase2_multitask/train_multitask_lite.py",
        PROJECT_ROOT / "scripts/phase2_multitask/train_multitask_std.py",
        PROJECT_ROOT / "scripts/phase2_multitask/train_multitask_pro.py",
        PROJECT_ROOT / "scripts/phase2_multitask/train_multitask_cgcnn.py",
        PROJECT_ROOT / "scripts/phase2_multitask/inference_multitask.py",
    ]
    missing = [p for p in required_scripts if not p.exists()]
    if missing:
        for path in missing:
            logger.error("[FAIL] Missing script: %s", path)
        return False
    logger.info("[PASS] Phase 2 script set is complete")

    logger.info("ALL SYSTEMS GO: Phase 2 module verification passed.")
    return True


if __name__ == "__main__":
    ok = verify_phase2()
    raise SystemExit(0 if ok else 1)
