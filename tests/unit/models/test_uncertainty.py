"""
Unit tests for atlas.models.uncertainty

Tests:
- EvidentialRegression: output shape, positivity constraints, loss computation
- EnsembleUQ: factory construction, predict_with_uncertainty
- MCDropoutUQ: stochastic predictions via predict_with_uncertainty
"""

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


class DropoutDictNet(nn.Module):
    """Model with dropout that returns dict outputs."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, *args, **kwargs):
        return {"band_gap": self.dropout(self.fc(x)).squeeze(-1)}


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


# ── MCDropoutUQ ───────────────────────────────────────────────


class TestMCDropoutUQ:
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
