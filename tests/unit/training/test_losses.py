"""
Unit tests for atlas.training.losses

Tests:
- PropertyLoss: MSE/L1/Huber, NaN masking, physical constraint penalties
- EvidentialLoss: NIG distribution loss
- MultiTaskLoss: fixed/uncertainty weighting, multi-device safety
"""

import numpy as np
import pytest
import torch

from atlas.training.losses import EvidentialLoss, MultiTaskLoss, PropertyLoss

# ── PropertyLoss ──────────────────────────────────────────────


class TestPropertyLoss:
    def test_invalid_loss_type_raises(self):
        with pytest.raises(ValueError, match="Unknown loss_type"):
            PropertyLoss(loss_type="mae")

    def test_invalid_constraint_weight_raises(self):
        with pytest.raises(ValueError, match="constraint_weight"):
            PropertyLoss(constraint_weight=float("inf"))
        with pytest.raises(ValueError, match="constraint_weight"):
            PropertyLoss(constraint_weight=True)  # type: ignore[arg-type]

    def test_mse_basic(self):
        loss_fn = PropertyLoss(loss_type="mse")
        pred = torch.tensor([1.0, 2.0, 3.0])
        tgt = torch.tensor([1.0, 2.0, 3.0])
        assert loss_fn(pred, tgt).item() == pytest.approx(0.0)

    def test_l1_basic(self):
        loss_fn = PropertyLoss(loss_type="l1")
        pred = torch.tensor([1.0, 3.0])
        tgt = torch.tensor([2.0, 2.0])
        assert loss_fn(pred, tgt).item() == pytest.approx(1.0)

    def test_huber_basic(self):
        loss_fn = PropertyLoss(loss_type="huber")
        pred = torch.tensor([1.0])
        tgt = torch.tensor([1.0])
        assert loss_fn(pred, tgt).item() == pytest.approx(0.0)

    def test_nan_masking(self):
        """NaN targets should be masked out, not explode the loss."""
        loss_fn = PropertyLoss(loss_type="mse")
        pred = torch.tensor([1.0, 2.0, 3.0])
        tgt = torch.tensor([1.0, float("nan"), 3.0])
        loss = loss_fn(pred, tgt)
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(0.0)  # 1==1 and 3==3

    def test_non_finite_pred_masking(self):
        loss_fn = PropertyLoss(loss_type="mse")
        pred = torch.tensor([1.0, float("inf"), 3.0])
        tgt = torch.tensor([1.0, 2.0, 3.0])
        loss = loss_fn(pred, tgt)
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(0.0)

    def test_all_nan_returns_zero(self):
        loss_fn = PropertyLoss()
        pred = torch.tensor([1.0, 2.0])
        tgt = torch.tensor([float("nan"), float("nan")])
        loss = loss_fn(pred, tgt)
        assert loss.item() == pytest.approx(0.0)

    def test_positive_constraint(self):
        """Positive constraint penalizes negative predictions."""
        loss_fn = PropertyLoss(constraint="positive", constraint_weight=1.0)
        pred = torch.tensor([-1.0])  # Violates positive constraint
        tgt = torch.tensor([-1.0])   # Perfect regression...
        loss_constrained = loss_fn(pred, tgt)
        # Base loss is 0 (perfect match), but penalty should be > 0
        assert loss_constrained.item() > 0.0

    def test_positive_constraint_no_penalty_for_positive(self):
        loss_fn = PropertyLoss(constraint="positive", constraint_weight=1.0)
        pred = torch.tensor([1.0, 2.0])
        tgt = torch.tensor([1.0, 2.0])
        loss = loss_fn(pred, tgt)
        assert loss.item() == pytest.approx(0.0)

    def test_greater_than_one_constraint(self):
        """greater_than_one penalizes predictions < 1."""
        loss_fn = PropertyLoss(constraint="greater_than_one", constraint_weight=1.0)
        pred = torch.tensor([0.5])  # Violates >= 1
        tgt = torch.tensor([0.5])
        loss = loss_fn(pred, tgt)
        assert loss.item() > 0.0  # Penalty applied

    def test_invalid_constraint_raises(self):
        with pytest.raises(ValueError, match="Unsupported constraint"):
            PropertyLoss(constraint="unsupported")

    def test_gradient_flows(self):
        loss_fn = PropertyLoss(loss_type="mse")
        pred = torch.tensor([1.0], requires_grad=True)
        tgt = torch.tensor([2.0])
        loss = loss_fn(pred, tgt)
        loss.backward()
        assert pred.grad is not None

    def test_bce_targets_are_clamped_to_probability_domain(self):
        loss_fn = PropertyLoss(loss_type="bce")
        pred = torch.tensor([0.0, 0.0])
        tgt = torch.tensor([2.0, -1.0])  # should clamp to [1.0, 0.0]
        loss = loss_fn(pred, tgt)
        assert loss.item() == pytest.approx(0.69314718, rel=1e-5)


# ── EvidentialLoss ────────────────────────────────────────────


class TestEvidentialLoss:
    def test_invalid_coeff_raises(self):
        with pytest.raises(ValueError, match="coeff"):
            EvidentialLoss(coeff=-1.0)
        with pytest.raises(ValueError, match="coeff"):
            EvidentialLoss(coeff=np.bool_(True))  # type: ignore[arg-type]

    def _make_evidential_pred(self, n=5):
        return {
            "gamma": torch.randn(n),     # mean
            "nu": torch.abs(torch.randn(n)) + 1.0,      # > 0
            "alpha": torch.abs(torch.randn(n)) + 1.0,   # > 1
            "beta": torch.abs(torch.randn(n)) + 0.1,    # > 0
        }

    def test_basic_computation(self):
        loss_fn = EvidentialLoss(coeff=0.05)
        pred = self._make_evidential_pred()
        target = torch.randn(5)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_nan_masking(self):
        loss_fn = EvidentialLoss()
        pred = self._make_evidential_pred(3)
        target = torch.tensor([1.0, float("nan"), 2.0])
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_all_nan_returns_zero(self):
        loss_fn = EvidentialLoss()
        pred = self._make_evidential_pred(2)
        target = torch.tensor([float("nan"), float("nan")])
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(0.0)

    def test_missing_required_key_raises(self):
        loss_fn = EvidentialLoss()
        pred = self._make_evidential_pred(2)
        pred.pop("beta")
        with pytest.raises(KeyError, match="missing required keys"):
            loss_fn(pred, torch.randn(2))

    def test_invalid_distribution_rows_are_skipped(self):
        loss_fn = EvidentialLoss()
        pred = {
            "gamma": torch.tensor([0.0, 1.0]),
            "nu": torch.tensor([1.0, -1.0]),
            "alpha": torch.tensor([1.5, 1.5]),
            "beta": torch.tensor([0.5, 0.5]),
        }
        target = torch.tensor([0.0, 1.0])
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0

    def test_mismatched_shapes_are_safely_aligned(self):
        loss_fn = EvidentialLoss()
        pred = self._make_evidential_pred(4)
        target = torch.randn(2)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_integer_target_dtype_is_supported(self):
        loss_fn = EvidentialLoss()
        pred = self._make_evidential_pred(3)
        target = torch.tensor([0, 1, 2], dtype=torch.int64)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)


# ── MultiTaskLoss ─────────────────────────────────────────────


class TestMultiTaskLoss:
    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            MultiTaskLoss(task_names=["band_gap"], strategy="bad")

    def test_invalid_task_type_raises(self):
        with pytest.raises(ValueError, match="Unknown task type"):
            MultiTaskLoss(task_names=["band_gap"], task_types={"band_gap": "foo"})

    def test_empty_task_names_raises(self):
        with pytest.raises(ValueError, match="task_names"):
            MultiTaskLoss(task_names=[])

    def test_duplicate_task_names_raise(self):
        with pytest.raises(ValueError, match="unique"):
            MultiTaskLoss(task_names=["band_gap", "band_gap"])

    def test_invalid_fixed_weight_raises(self):
        with pytest.raises(ValueError, match="task_weights"):
            MultiTaskLoss(
                task_names=["band_gap"],
                strategy="fixed",
                task_weights={"band_gap": float("nan")},
            )
        with pytest.raises(ValueError, match="task_weights"):
            MultiTaskLoss(
                task_names=["band_gap"],
                strategy="fixed",
                task_weights={"band_gap": True},  # type: ignore[dict-item]
            )

    def test_task_type_unknown_task_key_raises(self):
        with pytest.raises(KeyError, match="task_types contains unknown task"):
            MultiTaskLoss(task_names=["band_gap"], task_types={"unknown": "regression"})

    def test_task_weights_unknown_task_key_raises(self):
        with pytest.raises(KeyError, match="task_weights contains unknown task"):
            MultiTaskLoss(task_names=["band_gap"], strategy="fixed", task_weights={"unknown": 1.0})

    def test_constraints_unknown_task_key_raises(self):
        with pytest.raises(KeyError, match="constraints contains unknown task"):
            MultiTaskLoss(task_names=["band_gap"], constraints={"unknown": "positive"})

    def test_normalized_mapping_duplicate_keys_raise(self):
        with pytest.raises(ValueError, match="duplicate task name"):
            MultiTaskLoss(
                task_names=["band_gap"],
                strategy="fixed",
                task_weights={"band_gap": 1.0, " band_gap ": 2.0},
            )

    def test_fixed_strategy(self):
        tasks = ["band_gap", "formation_energy"]
        loss_fn = MultiTaskLoss(
            task_names=tasks,
            strategy="fixed",
            task_weights={"band_gap": 1.0, "formation_energy": 2.0},
        )
        pred = {
            "band_gap": torch.tensor([1.0, 2.0]),
            "formation_energy": torch.tensor([0.5, 1.5]),
        }
        tgt = {
            "band_gap": torch.tensor([1.0, 2.0]),
            "formation_energy": torch.tensor([0.5, 1.5]),
        }
        result = loss_fn(pred, tgt)
        assert "total" in result
        assert result["total"].item() == pytest.approx(0.0, abs=1e-6)

    def test_uncertainty_strategy(self):
        tasks = ["band_gap", "formation_energy"]
        loss_fn = MultiTaskLoss(task_names=tasks, strategy="uncertainty")
        pred = {
            "band_gap": torch.tensor([1.0, 2.0]),
            "formation_energy": torch.tensor([0.5]),
        }
        tgt = {
            "band_gap": torch.tensor([1.5, 2.5]),
            "formation_energy": torch.tensor([1.0]),
        }
        result = loss_fn(pred, tgt)
        assert "total" in result
        assert torch.isfinite(result["total"])

    def test_missing_task_in_pred(self):
        """Tasks missing from predictions should be gracefully skipped."""
        tasks = ["a", "b"]
        loss_fn = MultiTaskLoss(task_names=tasks, strategy="fixed")
        pred = {"a": torch.tensor([1.0])}  # "b" missing
        tgt = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        result = loss_fn(pred, tgt)
        assert "total" in result

    def test_empty_predictions_returns_zero_total(self):
        loss_fn = MultiTaskLoss(task_names=["a"], strategy="fixed")
        result = loss_fn(predictions={}, targets={})
        assert "total" in result
        assert result["total"].item() == pytest.approx(0.0)
        assert result["total"].requires_grad

    def test_shape_mismatch_raises_when_numel_differs(self):
        loss_fn = MultiTaskLoss(task_names=["a"], strategy="fixed")
        pred = {"a": torch.tensor([[1.0, 2.0]])}
        tgt = {"a": torch.tensor([1.0, 2.0, 3.0])}
        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(pred, tgt)
