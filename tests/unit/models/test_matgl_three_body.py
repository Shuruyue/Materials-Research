from __future__ import annotations

import pytest
import torch

from atlas.models.matgl_three_body import (
    SimpleMLPAngleExpansion,
    SphericalBesselHarmonicsExpansion,
)


def test_simple_mlp_angle_expansion_validates_n_basis():
    with pytest.raises(ValueError, match="n_basis must be integer-valued"):
        SimpleMLPAngleExpansion(n_basis=12.5)
    with pytest.raises(ValueError, match="n_basis must be integer-valued, not boolean"):
        SimpleMLPAngleExpansion(n_basis=True)


def test_simple_mlp_angle_expansion_validates_input_rank_and_shape():
    layer = SimpleMLPAngleExpansion(n_basis=8)
    r_ij = torch.ones(4, 1)
    r_ik = torch.ones(4, 1)
    cos_theta = torch.ones(4, 1)
    out = layer(r_ij, r_ik, cos_theta)
    assert out.shape == (4, 8)

    with pytest.raises(ValueError, match="rank-2"):
        layer(r_ij.reshape(-1), r_ik, cos_theta)
    with pytest.raises(ValueError, match="identical shapes"):
        layer(r_ij, r_ik[:-1], cos_theta)
    with pytest.raises(ValueError, match="shape \\(T, 1\\)"):
        layer(torch.ones(4, 2), torch.ones(4, 2), torch.zeros(4, 2))


def test_spherical_bessel_expansion_validates_constructor_and_inputs():
    with pytest.raises(ValueError, match="max_n must be integer-valued"):
        SphericalBesselHarmonicsExpansion(max_n=4.2, max_l=3)
    with pytest.raises(ValueError, match="max_l must be integer-valued, not boolean"):
        SphericalBesselHarmonicsExpansion(max_n=4, max_l=False)

    layer = SphericalBesselHarmonicsExpansion(max_n=3, max_l=2)
    r_ij = torch.ones(5, 1)
    r_ik = torch.ones(5, 1)
    cos_theta = torch.zeros(5, 1)
    out = layer(r_ij, r_ik, cos_theta)
    assert out.shape == (5, 6)
    assert torch.isfinite(out).all()

    with pytest.raises(ValueError, match="rank-2"):
        layer(r_ij.squeeze(-1), r_ik, cos_theta)
    with pytest.raises(ValueError, match="identical shapes"):
        layer(r_ij, r_ik[:-1], cos_theta)
    with pytest.raises(ValueError, match="shape \\(T, 1\\)"):
        layer(torch.ones(5, 2), torch.ones(5, 2), torch.zeros(5, 2))


def test_spherical_bessel_expansion_sanitizes_non_finite_inputs():
    layer = SphericalBesselHarmonicsExpansion(max_n=4, max_l=3)
    r_ij = torch.tensor([[1.0], [float("nan")], [float("inf")]])
    r_ik = torch.tensor([[1.2], [0.8], [float("-inf")]])
    cos_theta = torch.tensor([[0.0], [float("nan")], [2.0]])
    out = layer(r_ij, r_ik, cos_theta)
    assert out.shape == (3, 12)
    assert torch.isfinite(out).all()
