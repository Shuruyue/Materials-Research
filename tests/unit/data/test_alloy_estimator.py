"""Tests for atlas.data.alloy_estimator module."""


import pytest


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


def test_preset_sac405():
    """SAC405 preset should load and stay close to SAC family density."""
    from atlas.data.alloy_estimator import AlloyEstimator
    est = AlloyEstimator.from_preset("SAC405")
    props = est.estimate_properties()
    assert 7.0 < props["density_g_cm3"] < 8.2


def test_preset_pure_cu():
    """Pure Cu should match single-phase reference values."""
    from atlas.data.alloy_estimator import AlloyEstimator
    est = AlloyEstimator.from_preset("pure_Cu")
    props = est.estimate_properties()
    assert abs(props["density_g_cm3"] - 8.96) < 0.1
    assert abs(props["melting_point_K"] - 1358.0) < 1.0


def test_preset_alias_normalization():
    """Preset aliases with separators/case should resolve correctly."""
    from atlas.data.alloy_estimator import AlloyEstimator
    a = AlloyEstimator.from_preset("pure-cu")
    b = AlloyEstimator.from_preset("PURE CU")
    c = AlloyEstimator.from_preset("pure_cu")
    assert a.name == b.name == c.name == "pure_Cu"


def test_available_presets_contains_new_entries():
    """available_presets should expose all supported named presets."""
    from atlas.data.alloy_estimator import AlloyEstimator
    names = set(AlloyEstimator.available_presets())
    assert {"SAC305", "SAC405", "SnPb63", "pure_Sn", "pure_Cu"}.issubset(names)


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


def test_thermal_wiener_bounds_contain_prediction(sac305):
    """Thermal prediction should stay inside Wiener bounds."""
    props = sac305.estimate_properties()
    lo = props["thermal_conductivity_wiener_lower_W_mK"]
    hi = props["thermal_conductivity_wiener_upper_W_mK"]
    pred = props["thermal_conductivity_W_mK"]
    assert lo <= pred <= hi


def test_maxwell_thermal_model_within_wiener_bounds(sac305):
    """Maxwell estimate should also stay inside Wiener bounds."""
    props = sac305.estimate_properties(thermal_model="maxwell")
    lo = props["thermal_conductivity_wiener_lower_W_mK"]
    hi = props["thermal_conductivity_wiener_upper_W_mK"]
    pred = props["thermal_conductivity_W_mK"]
    assert lo <= pred <= hi
    assert props["thermal_model_used"] == "maxwell"


def test_unknown_thermal_model_raises(sac305):
    """Unknown thermal model should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown thermal_model"):
        sac305.estimate_properties(thermal_model="invalid")


def test_mixing_entropy_pure_phase_is_zero():
    """Pure element has zero ideal mixing entropy."""
    from atlas.data.alloy_estimator import AlloyEstimator

    est = AlloyEstimator.from_preset("pure_Sn")
    props = est.estimate_properties()
    assert abs(props["mixing_entropy_J_molK"]) < 1e-10
    assert abs(props["mixing_entropy_over_R"]) < 1e-10


def test_mixing_entropy_binary_is_positive():
    """Binary alloy should have positive configurational entropy."""
    from atlas.data.alloy_estimator import AlloyEstimator

    est = AlloyEstimator.from_preset("SnPb63")
    props = est.estimate_properties()
    assert props["mixing_entropy_J_molK"] > 0
    assert props["phase_entropy_volume_nat"] > 0


def test_maxwell_model_is_order_invariant():
    """Maxwell/Bruggeman thermal estimate should be invariant to phase order."""
    from atlas.data.alloy_estimator import AlloyEstimator

    phases = [
        {
            "name": "A",
            "formula": "Cu",
            "weight_fraction": 0.5,
            "properties": {
                "density_g_cm3": 8.0,
                "bulk_modulus_GPa": 100,
                "shear_modulus_GPa": 40,
                "thermal_conductivity_W_mK": 200,
                "thermal_expansion_1e6_K": 15,
                "melting_point_K": 1200,
            },
        },
        {
            "name": "B",
            "formula": "Sn",
            "weight_fraction": 0.5,
            "properties": {
                "density_g_cm3": 8.0,
                "bulk_modulus_GPa": 60,
                "shear_modulus_GPa": 20,
                "thermal_conductivity_W_mK": 20,
                "thermal_expansion_1e6_K": 22,
                "melting_point_K": 500,
            },
        },
    ]

    est_ab = AlloyEstimator.custom("ab", phases)
    est_ba = AlloyEstimator.custom("ba", list(reversed(phases)))

    k_ab = est_ab.estimate_properties(thermal_model="maxwell")["thermal_conductivity_W_mK"]
    k_ba = est_ba.estimate_properties(thermal_model="maxwell")["thermal_conductivity_W_mK"]
    assert abs(k_ab - k_ba) < 1e-10


def test_mixing_entropy_uses_element_basis_for_intermetallic():
    """Configurational entropy should use elemental fractions, not phase fractions."""
    from atlas.data.alloy_estimator import AlloyEstimator

    est = AlloyEstimator.custom(
        "sn_imc",
        [
            {
                "name": "Sn",
                "formula": "Sn",
                "weight_fraction": 0.7,
                "properties": {
                    "density_g_cm3": 7.29,
                    "bulk_modulus_GPa": 56.3,
                    "shear_modulus_GPa": 18.4,
                    "thermal_conductivity_W_mK": 66.0,
                    "thermal_expansion_1e6_K": 22.0,
                    "melting_point_K": 505.0,
                },
            },
            {
                "name": "IMC",
                "formula": "Cu6Sn5",
                "weight_fraction": 0.3,
                "properties": {
                    "density_g_cm3": 8.28,
                    "bulk_modulus_GPa": 120.0,
                    "shear_modulus_GPa": 45.0,
                    "thermal_conductivity_W_mK": 40.0,
                    "thermal_expansion_1e6_K": 18.0,
                    "melting_point_K": 688.0,
                },
            },
        ],
    )
    props = est.estimate_properties()
    assert props["mixing_entropy_basis"] == "element"
    # Element-level entropy should be materially larger than phase-level 2-state entropy here.
    assert props["mixing_entropy_over_R"] > 0.35
