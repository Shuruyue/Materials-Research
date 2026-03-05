"""Unit tests for atlas.training.physics_losses."""

import numpy as np
import pytest
import torch

from atlas.training.physics_losses import PhysicsConstraintLoss, VoigtReussBoundsLoss


def test_voigt_reuss_bounds_loss_rejects_invalid_weight():
    with pytest.raises(ValueError, match="weight"):
        VoigtReussBoundsLoss(weight=-1.0)
    with pytest.raises(ValueError, match="weight"):
        VoigtReussBoundsLoss(weight=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weight"):
        VoigtReussBoundsLoss(weight=np.bool_(True))  # type: ignore[arg-type]


def test_voigt_reuss_bounds_loss_handles_singular_tensor():
    loss_fn = VoigtReussBoundsLoss(weight=0.5)
    loss = loss_fn(
        K_pred=torch.tensor([1.0]),
        G_pred=torch.tensor([1.0]),
        C_pred=torch.zeros(1, 6, 6),
    )
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_physics_constraint_loss_filters_non_finite_predictions():
    loss_fn = PhysicsConstraintLoss()
    predictions = {
        "bulk_modulus": torch.tensor([float("nan"), -2.0]),
        "shear_modulus": torch.tensor([float("inf"), -1.0]),
        "dielectric": torch.tensor([0.2, float("nan")]),
        "elastic_tensor": torch.stack(
            [torch.eye(6) * -0.5, torch.full((6, 6), float("nan"))]
        ),
    }
    loss = loss_fn(predictions)
    assert torch.isfinite(loss)
    assert float(loss) > 0.0


def test_physics_constraint_loss_rejects_unknown_alpha():
    with pytest.raises(KeyError, match="unsupported alpha key"):
        PhysicsConstraintLoss(alpha={"unknown": 0.1})


def test_physics_constraint_loss_rejects_invalid_alpha_values():
    with pytest.raises(ValueError, match=r"alpha\[positivity\]"):
        PhysicsConstraintLoss(alpha={"positivity": float("nan")})
    with pytest.raises(ValueError, match=r"alpha\[dielectric_lower\]"):
        PhysicsConstraintLoss(alpha={"dielectric_lower": True})  # type: ignore[arg-type]


def test_voigt_reuss_alignment_uses_joint_finite_mask():
    loss_fn = VoigtReussBoundsLoss(weight=1.0)

    c0 = torch.eye(6) * 1.0
    c1 = torch.eye(6) * 9.0
    c2 = torch.eye(6) * 25.0
    c_batch = torch.stack([c0, c1, c2], dim=0)

    # Only index 1 is jointly finite across K/G; indices 0 and 2 should be discarded.
    k_pred = torch.tensor([0.3333333, 3.0, float("nan")], dtype=torch.float32)
    g_pred = torch.tensor([float("nan"), 6.8, 20.0], dtype=torch.float32)
    loss = loss_fn(K_pred=k_pred, G_pred=g_pred, C_pred=c_batch)

    assert torch.isfinite(loss)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_voigt_reuss_empty_finite_rows_keeps_grad_path():
    loss_fn = VoigtReussBoundsLoss(weight=0.5)
    k_pred = torch.tensor([float("nan")], dtype=torch.float32, requires_grad=True)
    g_pred = torch.tensor([float("nan")], dtype=torch.float32, requires_grad=True)
    c_pred = torch.full((1, 6, 6), float("nan"), dtype=torch.float32, requires_grad=True)

    loss = loss_fn(K_pred=k_pred, G_pred=g_pred, C_pred=c_pred)
    assert torch.isfinite(loss)
    assert loss.requires_grad
