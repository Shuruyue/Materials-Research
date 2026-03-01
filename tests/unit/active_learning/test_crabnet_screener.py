"""Algorithm-focused tests for CompositionScreener."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from atlas.active_learning.crabnet_screener import CompositionScreener, ContinuousFractionalEncoder


def test_continuous_fractional_encoder_is_sensitive_to_tiny_changes():
    enc = ContinuousFractionalEncoder(d_model=16, resolution=5000, log10=False)
    x1 = torch.tensor([[0.500000]], dtype=torch.float32)
    x2 = torch.tensor([[0.500001]], dtype=torch.float32)
    diff = torch.max(torch.abs(enc(x1) - enc(x2))).item()
    assert diff > 0.0


def test_escort_transform_matches_expected_simplex_power_mapping():
    model = CompositionScreener(
        d_model=64,
        N=1,
        heads=4,
        dropout=0.0,
        simplex_transform="escort",
        simplex_blend=1.0,
        escort_power=0.5,
        uncertainty_head_mode="log_std",
    )
    model.eval()

    src = torch.tensor([[14, 8, 0]], dtype=torch.long)
    frac = torch.tensor([[0.9, 0.1, 0.0]], dtype=torch.float32)
    out = model.predict_distribution(src, frac)

    expected = torch.tensor([[0.75, 0.25, 0.0]], dtype=torch.float32)
    assert torch.allclose(out["transformed_frac"], expected, atol=1e-6)


def test_forward_is_permutation_invariant_for_composition_tokens():
    model = CompositionScreener(
        out_dims=1,
        d_model=64,
        N=2,
        heads=4,
        dropout=0.0,
        simplex_transform="none",
        simplex_blend=0.0,
    )
    model.eval()

    src_a = torch.tensor([[14, 8, 26, 0]], dtype=torch.long)
    frac_a = torch.tensor([[0.5, 0.3, 0.2, 0.0]], dtype=torch.float32)

    src_b = torch.tensor([[26, 14, 8, 0]], dtype=torch.long)
    frac_b = torch.tensor([[0.2, 0.5, 0.3, 0.0]], dtype=torch.float32)

    with torch.no_grad():
        out_a = model(src_a, frac_a)
        out_b = model(src_b, frac_b)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_distribution_output_returns_positive_std():
    model = CompositionScreener(
        out_dims=2,
        d_model=64,
        N=1,
        heads=4,
        dropout=0.0,
        uncertainty_head_mode="softplus_std",
        uncertainty_min_std=1e-5,
        return_distribution=True,
    )
    model.eval()

    src = torch.tensor([[29, 8, 0]], dtype=torch.long)
    frac = torch.tensor([[0.4, 0.6, 0.0]], dtype=torch.float32)
    out = model(src, frac)
    assert isinstance(out, dict)
    assert "mean" in out and "std" in out
    assert out["mean"].shape == (1, 2)
    assert out["std"].shape == (1, 2)
    assert torch.all(out["std"] > 0.0)


def test_forward_default_mode_keeps_point_prediction_shape():
    model = CompositionScreener(out_dims=3, d_model=64, N=1, heads=4, dropout=0.0)
    model.eval()
    src = torch.tensor([[13, 8, 0], [26, 8, 0]], dtype=torch.long)
    frac = torch.tensor([[0.5, 0.5, 0.0], [0.8, 0.2, 0.0]], dtype=torch.float32)
    out = model(src, frac)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 3)


def test_ilr_softmax_transform_is_simplex_valid_and_nontrivial():
    model = CompositionScreener(
        d_model=64,
        N=1,
        heads=4,
        dropout=0.0,
        simplex_transform="ilr_softmax",
        simplex_blend=1.0,
        ilr_temperature=2.0,
        uncertainty_head_mode="log_std",
    )
    model.eval()
    src = torch.tensor([[14, 8, 0]], dtype=torch.long)
    frac = torch.tensor([[0.9, 0.1, 0.0]], dtype=torch.float32)
    out = model.predict_distribution(src, frac)
    transformed = out["transformed_frac"]

    assert torch.allclose(transformed.sum(dim=1), torch.ones(1), atol=1e-6)
    assert transformed[0, 0] < 0.9
    assert transformed[0, 1] > 0.1


def test_ensemble_epistemic_uncertainty_is_reported():
    torch.manual_seed(0)
    model = CompositionScreener(
        out_dims=1,
        d_model=64,
        N=1,
        heads=4,
        dropout=0.0,
        uncertainty_head_mode="log_std",
        return_distribution=True,
        ensemble_size=3,
    )
    model.eval()
    for idx, head in enumerate(model.output_nns):
        for p in head.parameters():
            p.data.zero_()
        head.fc_out.bias.data.fill_(0.1 * idx)
    for head in model.uncertainty_nns:
        for p in head.parameters():
            p.data.zero_()

    src = torch.tensor([[26, 8, 0]], dtype=torch.long)
    frac = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32)
    out = model(src, frac)
    assert torch.all(out["epistemic_std"] > 0.0)
    assert torch.all(out["total_std"] >= out["aleatoric_std"])


def test_mc_dropout_augments_epistemic_component():
    torch.manual_seed(7)
    model = CompositionScreener(
        out_dims=1,
        d_model=64,
        N=2,
        heads=4,
        dropout=0.2,
        uncertainty_head_mode="log_std",
        ensemble_size=1,
    )
    model.eval()
    src = torch.tensor([[14, 8, 0], [26, 8, 0]], dtype=torch.long)
    frac = torch.tensor([[0.7, 0.3, 0.0], [0.4, 0.6, 0.0]], dtype=torch.float32)

    base = model.predict_distribution(src, frac, mc_samples=0)
    with_mc = model.predict_distribution(src, frac, mc_samples=6)
    assert "mc_epistemic_std" in with_mc
    assert torch.all(with_mc["total_std"] >= base["total_std"] - 1e-8)
    assert float(with_mc["mc_samples"].item()) == 6.0


def test_compute_training_loss_matches_mse_without_uncertainty_head():
    torch.manual_seed(11)
    model = CompositionScreener(out_dims=1, d_model=64, N=1, heads=4, dropout=0.0, uncertainty_head_mode="none")
    model.eval()
    src = torch.tensor([[13, 8, 0], [26, 8, 0]], dtype=torch.long)
    frac = torch.tensor([[0.5, 0.5, 0.0], [0.8, 0.2, 0.0]], dtype=torch.float32)
    target = torch.tensor([[0.2], [0.8]], dtype=torch.float32)
    loss = model.compute_training_loss(src, frac, target)
    pred = model(src, frac)
    expected = F.mse_loss(pred, target)
    assert torch.allclose(loss, expected, atol=1e-7)


def test_compute_training_loss_supports_gaussian_nll_with_uq_head():
    torch.manual_seed(13)
    model = CompositionScreener(
        out_dims=1,
        d_model=64,
        N=1,
        heads=4,
        dropout=0.0,
        uncertainty_head_mode="log_std",
        uncertainty_min_std=1e-4,
    )
    model.eval()
    src = torch.tensor([[13, 8, 0], [26, 8, 0]], dtype=torch.long)
    frac = torch.tensor([[0.6, 0.4, 0.0], [0.3, 0.7, 0.0]], dtype=torch.float32)
    target = torch.tensor([[0.4], [0.1]], dtype=torch.float32)
    loss = model.compute_training_loss(src, frac, target)
    assert torch.isfinite(loss)
