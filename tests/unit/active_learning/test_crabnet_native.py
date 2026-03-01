"""Algorithm-focused tests for NativeCrabnetScreener."""

from __future__ import annotations

import torch
import torch.nn as nn

import atlas.active_learning.crabnet_native as crabnet_native


class _DummyCrabNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.last_src = None
        self.last_frac = None
        self.kwargs = kwargs
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, src, frac):
        self.last_src = src.detach().clone()
        self.last_frac = frac.detach().clone()
        mean = (src.to(dtype=torch.float32) * frac).sum(dim=1, keepdim=True) + self.bias
        raw_scale = self.dropout(frac).sum(dim=1, keepdim=True) - 0.5
        return torch.cat([mean, raw_scale], dim=-1)


def _patch_dummy(monkeypatch):
    monkeypatch.setattr(crabnet_native, "OriginalCrabNet", _DummyCrabNet)


def test_forward_handles_reversed_input_order_and_fraction_sanitization(monkeypatch):
    _patch_dummy(monkeypatch)
    model = crabnet_native.NativeCrabnetScreener(
        return_distribution=False,
        simplex_transform="none",
        simplex_blend=0.0,
    )

    src = torch.tensor([[26, 8, 0]], dtype=torch.long)
    frac = torch.tensor([[0.7, -0.2, 0.4]], dtype=torch.float32)
    _ = model(frac, src)  # historical reversed signature

    seen_src = model.engine.last_src
    seen_frac = model.engine.last_frac
    assert torch.equal(seen_src, src)
    assert torch.all(seen_frac >= 0.0)
    assert torch.isclose(seen_frac[:, :2].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(seen_frac[:, 2].sum(), torch.tensor(0.0), atol=1e-6)


def test_escort_simplex_transform_matches_power_geometry(monkeypatch):
    _patch_dummy(monkeypatch)
    model = crabnet_native.NativeCrabnetScreener(
        return_distribution=False,
        simplex_transform="escort",
        simplex_blend=1.0,
        escort_power=0.5,
    )
    src = torch.tensor([[14, 8, 0]], dtype=torch.long)
    frac = torch.tensor([[0.9, 0.1, 0.0]], dtype=torch.float32)
    _ = model(src, frac)

    expected = torch.tensor([[0.75, 0.25, 0.0]], dtype=torch.float32)
    assert torch.allclose(model.engine.last_frac, expected, atol=1e-6)


def test_forward_returns_calibrated_distribution(monkeypatch):
    _patch_dummy(monkeypatch)
    model = crabnet_native.NativeCrabnetScreener(
        return_distribution=True,
        simplex_transform="none",
        simplex_blend=0.0,
        uncertainty_min_std=1e-4,
        uncertainty_temperature=1.5,
    )
    src = torch.tensor([[29, 8, 0]], dtype=torch.long)
    frac = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32)
    out = model(src, frac)

    assert isinstance(out, dict)
    assert "mean" in out and "std" in out
    assert torch.all(out["std"] > 0.0)


def test_mc_dropout_distribution_decomposes_uncertainty(monkeypatch):
    _patch_dummy(monkeypatch)
    model = crabnet_native.NativeCrabnetScreener(
        return_distribution=True,
        simplex_transform="none",
        simplex_blend=0.0,
    )
    src = torch.tensor([[26, 8, 0], [14, 8, 0]], dtype=torch.long)
    frac = torch.tensor([[0.5, 0.5, 0.0], [0.8, 0.2, 0.0]], dtype=torch.float32)
    out = model.predict_distribution(src, frac, mc_samples=8)

    assert "aleatoric_std" in out and "epistemic_std" in out and "total_std" in out
    assert torch.all(out["total_std"] >= out["aleatoric_std"])
    assert torch.all(out["epistemic_std"] >= 0.0)


def test_conformal_temperature_calibration_updates_scale(monkeypatch):
    _patch_dummy(monkeypatch)
    model = crabnet_native.NativeCrabnetScreener(
        return_distribution=True,
        simplex_transform="none",
        simplex_blend=0.0,
    )
    y_true = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    y_pred = torch.tensor([1.2, 1.8, 3.5], dtype=torch.float32)
    y_std = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    t = model.calibrate_uncertainty_temperature(y_true, y_pred, y_std, quantile=0.8)

    assert t > 0.0
    assert float(model.uncertainty_temperature.item()) == t


def test_log_var_head_mode_is_supported(monkeypatch):
    _patch_dummy(monkeypatch)
    model = crabnet_native.NativeCrabnetScreener(
        return_distribution=True,
        uncertainty_head_mode="log_var",
        simplex_transform="none",
        simplex_blend=0.0,
    )
    src = torch.tensor([[13, 8, 0]], dtype=torch.long)
    frac = torch.tensor([[0.4, 0.6, 0.0]], dtype=torch.float32)
    out = model(src, frac)
    assert isinstance(out, dict)
    assert torch.all(out["std"] > 0.0)


def test_grouped_calibration_applies_group_specific_scale(monkeypatch):
    _patch_dummy(monkeypatch)
    model = crabnet_native.NativeCrabnetScreener(
        return_distribution=True,
        simplex_transform="none",
        simplex_blend=0.0,
        grouped_calibration=True,
    )

    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    y_pred = torch.tensor([1.2, 1.9, 2.5, 3.8], dtype=torch.float32)
    y_std = torch.tensor([0.2, 0.2, 0.2, 0.2], dtype=torch.float32)
    src = torch.tensor(
        [
            [26, 8, 0],   # group 2
            [14, 8, 0],   # group 2
            [26, 8, 1],   # group 3
            [14, 8, 1],   # group 3
        ],
        dtype=torch.long,
    )
    table = model.calibrate_uncertainty_temperature_grouped(
        y_true,
        y_pred,
        y_std,
        src,
        quantile=0.8,
        min_group_size=2,
    )
    assert 2 in table and 3 in table

    frac = torch.tensor([[0.5, 0.5, 0.0], [0.4, 0.4, 0.2]], dtype=torch.float32)
    out = model.predict_distribution(src[:2], frac, mc_samples=0)
    assert torch.all(out["std"] > 0.0)


def test_ensemble_introduces_inter_member_epistemic_variance(monkeypatch):
    _patch_dummy(monkeypatch)
    model = crabnet_native.NativeCrabnetScreener(
        return_distribution=True,
        simplex_transform="none",
        simplex_blend=0.0,
        ensemble_size=3,
    )
    for i, member in enumerate(model.ensemble_members):
        member.bias.data.fill_(0.1 * i)

    src = torch.tensor([[26, 8, 0]], dtype=torch.long)
    frac = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32)
    out = model.predict_distribution(src, frac, mc_samples=0)
    assert torch.any(out["epistemic_std"] > 0.0)
    assert int(out["ensemble_size"].item()) == 3


def test_optimize_escort_power_updates_parameter(monkeypatch):
    _patch_dummy(monkeypatch)
    model = crabnet_native.NativeCrabnetScreener(
        return_distribution=False,
        simplex_transform="escort",
        simplex_blend=1.0,
        escort_power=1.5,
    )
    src = torch.tensor([[26, 8, 0], [14, 8, 0]], dtype=torch.long)
    frac = torch.tensor([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0]], dtype=torch.float32)
    q = model.optimize_escort_power(frac, src=src, target_entropy_ratio=0.9, q_min=0.2, q_max=1.5, q_steps=12)
    assert 0.2 <= q <= 1.5
    assert abs(model.escort_power - q) < 1e-12
