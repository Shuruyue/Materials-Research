"""
Unit tests for atlas.models.uncertainty

Tests:
- EvidentialRegression: output shape, positivity constraints, loss computation
- EnsembleUQ: factory construction, predict_with_uncertainty
- MCDropoutUQ: stochastic predictions via predict_with_uncertainty
"""

import pytest
import torch
import torch.nn as nn

from atlas.models.uncertainty import EnsembleUQ, EvidentialRegression, MCDropoutUQ

# ── Helper models ─────────────────────────────────────────────


class DictNet(nn.Module):
    """Model that returns a dict of predictions (required by Ensemble/MC API)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x, *args, **kwargs):
        return {"band_gap": self.fc(x).squeeze(-1)}


class TensorNet(nn.Module):
    """Model that returns a plain tensor output."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x, *args, **kwargs):
        return self.fc(x).squeeze(-1)


class OtherDictNet(nn.Module):
    """Model that returns a different task key for mismatch tests."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x, *args, **kwargs):
        return {"formation_energy": self.fc(x).squeeze(-1)}


class DifferentShapeDictNet(nn.Module):
    """Model that returns same task key but incompatible tensor shape."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x, *args, **kwargs):
        return {"band_gap": self.fc(x)}  # [N, 1]


class DropoutDictNet(nn.Module):
    """Model with dropout that returns dict outputs."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, *args, **kwargs):
        return {"band_gap": self.dropout(self.fc(x)).squeeze(-1)}


class FlakyKeyDropoutNet(nn.Module):
    """Dropout model that alternates key names between calls."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)
        self.dropout = nn.Dropout(0.5)
        self._toggle = False

    def forward(self, x, *args, **kwargs):
        self._toggle = not self._toggle
        key = "band_gap" if self._toggle else "formation_energy"
        return {key: self.dropout(self.fc(x)).squeeze(-1)}


class FlakyShapeDropoutNet(nn.Module):
    """Dropout model that alternates output tensor shapes."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)
        self.dropout = nn.Dropout(0.5)
        self._toggle = False

    def forward(self, x, *args, **kwargs):
        self._toggle = not self._toggle
        out = self.dropout(self.fc(x))
        if self._toggle:
            return {"band_gap": out.squeeze(-1)}  # [N]
        return {"band_gap": out}  # [N, 1]


class BadValueNet(nn.Module):
    """Model that violates dict[str, Tensor] payload contract."""

    def forward(self, x, *args, **kwargs):
        return {"band_gap": 1.0}


# ── EvidentialRegression ──────────────────────────────────────


class TestEvidentialRegression:
    def test_output_shape(self):
        ev = EvidentialRegression(input_dim=32, output_dim=1)
        x = torch.randn(8, 32)
        out = ev(x)
        assert isinstance(out, dict)
        for key in ["mean", "aleatoric", "epistemic", "total_std"]:
            assert key in out

    def test_internal_params_present(self):
        ev = EvidentialRegression(input_dim=16, output_dim=1)
        x = torch.randn(4, 16)
        out = ev(x)
        for key in ["_gamma", "_nu", "_alpha", "_beta"]:
            assert key in out

    def test_positivity_constraints(self):
        """nu > 0, alpha > 1, beta > 0."""
        ev = EvidentialRegression(input_dim=16, output_dim=1)
        x = torch.randn(10, 16)
        out = ev(x)
        assert (out["_nu"] > 0).all()
        assert (out["_alpha"] > 1).all()
        assert (out["_beta"] > 0).all()

    def test_uncertainty_positive(self):
        ev = EvidentialRegression(input_dim=8, output_dim=1)
        x = torch.randn(5, 8)
        out = ev(x)
        assert (out["aleatoric"] > 0).all()
        assert (out["epistemic"] > 0).all()
        assert (out["total_std"] > 0).all()

    def test_gradient_flows(self):
        ev = EvidentialRegression(input_dim=8, output_dim=1)
        x = torch.randn(4, 8, requires_grad=True)
        out = ev(x)
        loss = out["mean"].sum()
        loss.backward()
        assert x.grad is not None

    def test_evidential_loss_computes(self):
        ev = EvidentialRegression(input_dim=16, output_dim=1)
        x = torch.randn(5, 16)
        out = ev(x)
        target = torch.randn(5, 1)
        loss = EvidentialRegression.evidential_loss(out, target)
        assert torch.isfinite(loss)


# ── EnsembleUQ ────────────────────────────────────────────────


class TestEnsembleUQ:
    def test_invalid_n_models_raises(self):
        with pytest.raises(ValueError, match="n_models must be > 0"):
            EnsembleUQ(model_factory=DictNet, n_models=0)

    @pytest.mark.parametrize("value", [1.5, True, float("inf"), float("nan")])
    def test_invalid_non_integral_or_boolean_n_models_raises(self, value):
        with pytest.raises(ValueError, match="n_models"):
            EnsembleUQ(model_factory=DictNet, n_models=value)

    def test_construction(self):
        ensemble = EnsembleUQ(model_factory=DictNet, n_models=3)
        assert ensemble.n_models == 3
        assert len(ensemble.models) == 3

    def test_forward(self):
        ensemble = EnsembleUQ(model_factory=DictNet, n_models=3)
        x = torch.randn(5, 4)
        result = ensemble(x)
        assert "band_gap" in result
        mean, std = result["band_gap"]
        assert mean.shape == (5,)
        assert std.shape == (5,)
        assert (std >= 0).all()

    def test_predict_with_uncertainty(self):
        ensemble = EnsembleUQ(model_factory=DictNet, n_models=4)
        x = torch.randn(3, 4)
        result = ensemble.predict_with_uncertainty(x)
        assert "band_gap" in result
        assert "mean" in result["band_gap"]
        assert "std" in result["band_gap"]
        assert "all" in result["band_gap"]
        assert result["band_gap"]["all"].shape == (4, 3)  # 4 models, 3 samples

    def test_single_model_std_is_finite(self):
        ensemble = EnsembleUQ(model_factory=DictNet, n_models=1)
        x = torch.randn(4, 4)
        result = ensemble.predict_with_uncertainty(x)
        std = result["band_gap"]["std"]
        assert torch.isfinite(std).all()
        assert torch.all(std >= 0)

    def test_tensor_output_is_supported(self):
        ensemble = EnsembleUQ(model_factory=TensorNet, n_models=2)
        x = torch.randn(3, 4)
        result = ensemble.predict_with_uncertainty(x)
        assert "prediction" in result
        assert result["prediction"]["all"].shape == (2, 3)

    def test_inconsistent_prediction_keys_raise(self):
        state = {"idx": 0}

        def factory():
            state["idx"] += 1
            return DictNet() if state["idx"] == 1 else OtherDictNet()

        ensemble = EnsembleUQ(model_factory=factory, n_models=2)
        with pytest.raises(ValueError, match="Inconsistent prediction keys"):
            _ = ensemble.predict_with_uncertainty(torch.randn(3, 4))

    def test_non_tensor_prediction_value_raises(self):
        ensemble = EnsembleUQ(model_factory=BadValueNet, n_models=2)
        with pytest.raises(TypeError, match="must be Tensor"):
            _ = ensemble.predict_with_uncertainty(torch.randn(3, 4))

    def test_inconsistent_prediction_shape_raises(self):
        state = {"idx": 0}

        def factory():
            state["idx"] += 1
            return DictNet() if state["idx"] == 1 else DifferentShapeDictNet()

        ensemble = EnsembleUQ(model_factory=factory, n_models=2)
        with pytest.raises(ValueError, match="Inconsistent prediction shape"):
            _ = ensemble.predict_with_uncertainty(torch.randn(3, 4))


# ── MCDropoutUQ ───────────────────────────────────────────────


class TestMCDropoutUQ:
    def test_invalid_n_samples_raises(self):
        model = DropoutDictNet()
        with pytest.raises(ValueError, match="n_samples must be > 0"):
            MCDropoutUQ(model, n_samples=0)

    @pytest.mark.parametrize("value", [2.2, False, float("inf"), float("nan")])
    def test_invalid_non_integral_or_boolean_n_samples_raises(self, value):
        model = DropoutDictNet()
        with pytest.raises(ValueError, match="n_samples"):
            MCDropoutUQ(model, n_samples=value)

    def test_construction(self):
        model = DropoutDictNet()
        mc = MCDropoutUQ(model, n_samples=10)
        assert mc.n_samples == 10

    def test_predict_with_uncertainty(self):
        model = DropoutDictNet()
        mc = MCDropoutUQ(model, n_samples=10)
        x = torch.randn(5, 4)
        result = mc.predict_with_uncertainty(x)
        assert "band_gap" in result
        assert "mean" in result["band_gap"]
        assert "std" in result["band_gap"]
        assert result["band_gap"]["mean"].shape == (5,)

    def test_std_nonzero_with_dropout(self):
        """MC Dropout should produce non-zero std with 50% dropout."""
        model = DropoutDictNet()
        mc = MCDropoutUQ(model, n_samples=50)
        x = torch.randn(5, 4)
        result = mc.predict_with_uncertainty(x)
        # With 50% dropout and many samples, std should generally be > 0
        assert result["band_gap"]["std"].sum().item() > 0

    def test_predict_restores_training_state(self):
        model = DropoutDictNet()
        model.train(True)
        mc = MCDropoutUQ(model, n_samples=4)
        _ = mc.predict_with_uncertainty(torch.randn(2, 4))
        assert model.training is True

    def test_tensor_output_is_supported(self):
        model = TensorNet()
        mc = MCDropoutUQ(model, n_samples=5)
        out = mc.predict_with_uncertainty(torch.randn(3, 4))
        assert "prediction" in out
        assert out["prediction"]["mean"].shape == (3,)
        assert torch.isfinite(out["prediction"]["std"]).all()

    def test_mc_single_sample_std_is_finite(self):
        model = DropoutDictNet()
        mc = MCDropoutUQ(model, n_samples=1)
        out = mc.predict_with_uncertainty(torch.randn(4, 4))
        std = out["band_gap"]["std"]
        assert torch.isfinite(std).all()
        assert torch.all(std >= 0)

    def test_mc_inconsistent_prediction_keys_raise(self):
        model = FlakyKeyDropoutNet()
        mc = MCDropoutUQ(model, n_samples=3)
        with pytest.raises(ValueError, match="Inconsistent prediction keys"):
            _ = mc.predict_with_uncertainty(torch.randn(4, 4))

    def test_mc_inconsistent_prediction_shapes_raise(self):
        model = FlakyShapeDropoutNet()
        mc = MCDropoutUQ(model, n_samples=3)
        with pytest.raises(ValueError, match="Inconsistent prediction shape"):
            _ = mc.predict_with_uncertainty(torch.randn(4, 4))


def test_evidential_loss_non_finite_target_is_stable():
    ev = EvidentialRegression(input_dim=8, output_dim=1)
    x = torch.randn(4, 8)
    out = ev(x)
    target = torch.tensor([[1.0], [float("nan")], [2.0], [float("inf")]])
    loss = EvidentialRegression.evidential_loss(out, target, coeff=float("nan"))
    assert torch.isfinite(loss)


def test_evidential_loss_all_non_finite_targets_returns_zero():
    ev = EvidentialRegression(input_dim=8, output_dim=1)
    x = torch.randn(4, 8)
    out = ev(x)
    target = torch.tensor([[float("nan")], [float("inf")], [float("-inf")], [float("nan")]])
    loss = EvidentialRegression.evidential_loss(out, target, coeff=0.1)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)


def test_evidential_loss_accepts_integer_target_dtype():
    ev = EvidentialRegression(input_dim=8, output_dim=1)
    x = torch.randn(4, 8)
    out = ev(x)
    target = torch.tensor([[1], [0], [2], [3]], dtype=torch.int64)
    loss = EvidentialRegression.evidential_loss(out, target, coeff=0.05)
    assert torch.isfinite(loss)
