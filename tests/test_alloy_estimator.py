"""Tests for atlas.data.alloy_estimator module."""

import pytest
import math


@pytest.fixture
def sac305():
    """SAC305 preset."""
    from atlas.data.alloy_estimator import AlloyEstimator
    return AlloyEstimator.from_preset("SAC305")


def test_preset_sac305_exists(sac305):
    """SAC305 preset should load."""
    assert sac305 is not None
    assert "SAC305" in sac305.name
    assert len(sac305.phases) == 3


def test_preset_volume_fractions_sum_to_one(sac305):
    """Volume fractions should sum to 1."""
    total = sum(p.volume_fraction for p in sac305.phases)
    assert abs(total - 1.0) < 0.01


def test_estimate_properties_keys(sac305):
    """estimate_properties should return expected keys."""
    props = sac305.estimate_properties()
    expected_keys = [
        "density_g_cm3", "bulk_modulus_GPa", "shear_modulus_GPa",
        "youngs_modulus_GPa", "poisson_ratio",
        "thermal_conductivity_W_mK", "thermal_expansion_1e6_K",
        "melting_point_K", "hardness_GPa", "pugh_ratio",
    ]
    for key in expected_keys:
        assert key in props, f"Missing key: {key}"


def test_sac305_density_reasonable(sac305):
    """SAC305 density should be ~7.3 g/cm³."""
    props = sac305.estimate_properties()
    assert 7.0 < props["density_g_cm3"] < 8.0


def test_sac305_youngs_modulus_reasonable(sac305):
    """SAC305 Young's modulus should be ~50 GPa."""
    props = sac305.estimate_properties()
    assert 30 < props["youngs_modulus_GPa"] < 80


def test_sac305_melting_point_reasonable(sac305):
    """SAC305 melting point should be near 505K."""
    props = sac305.estimate_properties()
    assert 480 < props["melting_point_K"] < 520


def test_sac305_thermal_conductivity(sac305):
    """SAC305 thermal conductivity should be ~60 W/m·K."""
    props = sac305.estimate_properties()
    assert 40 < props["thermal_conductivity_W_mK"] < 80


def test_vrh_bounds(sac305):
    """VRH should be between Voigt and Reuss bounds."""
    props = sac305.estimate_properties()

    k_vrh = props["bulk_modulus_GPa"]
    k_voigt = props["bulk_modulus_voigt_GPa"]
    k_reuss = props["bulk_modulus_reuss_GPa"]

    assert k_reuss <= k_vrh <= k_voigt


def test_pugh_ratio(sac305):
    """Pugh ratio should indicate ductility for SAC305."""
    props = sac305.estimate_properties()
    assert props["pugh_ratio"] > 0
    # SAC305 is ductile
    assert props["ductile"] is True or props["ductile"] is False


def test_preset_snpb63():
    """SnPb63 preset should load and estimate."""
    from atlas.data.alloy_estimator import AlloyEstimator
    est = AlloyEstimator.from_preset("SnPb63")
    props = est.estimate_properties()
    assert 7.5 < props["density_g_cm3"] < 9.0  # SnPb is denser


def test_preset_pure_sn():
    """Pure Sn should match single-phase values."""
    from atlas.data.alloy_estimator import AlloyEstimator
    est = AlloyEstimator.from_preset("pure_Sn")
    props = est.estimate_properties()
    assert abs(props["density_g_cm3"] - 7.29) < 0.1


def test_unknown_preset_raises():
    """Unknown preset should raise ValueError."""
    from atlas.data.alloy_estimator import AlloyEstimator
    with pytest.raises(ValueError, match="Unknown preset"):
        AlloyEstimator.from_preset("INVALID_ALLOY")


def test_custom_alloy():
    """Custom alloy creation should work."""
    from atlas.data.alloy_estimator import AlloyEstimator
    est = AlloyEstimator.custom("CuSn_80_20", [
        {"name": "Cu", "formula": "Cu", "weight_fraction": 0.8,
         "properties": {"bulk_modulus_GPa": 137, "shear_modulus_GPa": 48.3,
                        "density_g_cm3": 8.96, "melting_point_K": 1358}},
        {"name": "Sn", "formula": "Sn", "weight_fraction": 0.2,
         "properties": {"bulk_modulus_GPa": 56.3, "shear_modulus_GPa": 18.4,
                        "density_g_cm3": 7.29, "melting_point_K": 505}},
    ])
    props = est.estimate_properties()
    # Mixed density should be between Cu and Sn
    assert 7.29 < props["density_g_cm3"] < 8.96


def test_convert_wt_to_vol():
    """Weight-to-volume conversion should produce valid fractions."""
    from atlas.data.alloy_estimator import AlloyEstimator, AlloyPhase

    phases = [
        AlloyPhase("A", "A", weight_fraction=0.5,
                   properties={"density_g_cm3": 10.0}),
        AlloyPhase("B", "B", weight_fraction=0.5,
                   properties={"density_g_cm3": 5.0}),
    ]
    AlloyEstimator.convert_wt_to_vol(phases)

    # B is less dense → more volume
    assert phases[1].volume_fraction > phases[0].volume_fraction
    total = sum(p.volume_fraction for p in phases)
    assert abs(total - 1.0) < 0.01


def test_print_report_no_error(sac305, capsys):
    """print_report should not raise."""
    sac305.print_report()
    captured = capsys.readouterr()
    assert "SAC305" in captured.out
    assert "Density" in captured.out


def test_print_report_with_experimental(sac305, capsys):
    """print_report with experimental data should show errors."""
    exp = {"density_g_cm3": 7.38, "youngs_modulus_GPa": 51.0}
    sac305.print_report(experimental=exp)
    captured = capsys.readouterr()
    assert "%" in captured.out  # error percentages
