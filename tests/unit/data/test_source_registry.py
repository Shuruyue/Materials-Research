"""Tests for atlas.data.source_registry."""


def test_data_source_registry_defaults():
    from atlas.data.source_registry import DATA_SOURCES

    keys = DATA_SOURCES.list_keys()
    assert "jarvis_dft" in keys
    assert "materials_project" in keys
    assert "matbench" in keys

    jarvis = DATA_SOURCES.get("jarvis_dft")
    assert "JARVIS" in jarvis.name
    assert "formation_energy" in jarvis.primary_targets


def test_data_source_spec_normalizes_key_and_targets():
    from atlas.data.source_registry import DataSourceRegistry, DataSourceSpec

    registry = DataSourceRegistry()
    registry.register(
        DataSourceSpec(
            key=" custom ",
            name=" Custom ",
            domain=" inorganic ",
            primary_targets=[" band_gap ", "", "band_gap"],
        )
    )
    assert registry.list_keys() == ["custom"]
    spec = registry.get(" custom ")
    assert spec.key == "custom"
    assert spec.name == "Custom"
    assert spec.domain == "inorganic"
    assert spec.primary_targets == ["band_gap"]


def test_data_source_spec_rejects_boolean_or_non_string_identity_fields():
    import pytest

    from atlas.data.source_registry import DataSourceSpec

    with pytest.raises(TypeError, match="DataSourceSpec.key"):
        DataSourceSpec(key=True, name="X", domain="d")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DataSourceSpec.name"):
        DataSourceSpec(key="k", name=1, domain="d")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="primary_targets"):
        DataSourceSpec(key="k", name="n", domain="d", primary_targets=[True])  # type: ignore[list-item]


def test_data_source_registry_missing_key():
    import pytest

    from atlas.data.source_registry import DATA_SOURCES

    with pytest.raises(KeyError):
        DATA_SOURCES.get("does_not_exist")


def test_source_reliability_update_posterior():
    from atlas.data.source_registry import DataSourceRegistry, DataSourceSpec

    registry = DataSourceRegistry()
    registry.register(DataSourceSpec(key="src", name="S", domain="d"))
    before = registry.get_reliability("src").mean
    registry.update_reliability("src", successes=8, failures=2)
    after = registry.get_reliability("src").mean
    assert after > before


def test_data_source_registry_duplicate_registration_requires_replace():
    import pytest

    from atlas.data.source_registry import DataSourceRegistry, DataSourceSpec

    registry = DataSourceRegistry()
    registry.register(DataSourceSpec(key="src", name="S", domain="d"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(DataSourceSpec(key="src", name="S2", domain="d2"))


def test_data_source_registry_replace_allows_overwrite():
    from atlas.data.source_registry import DataSourceRegistry, DataSourceSpec

    registry = DataSourceRegistry()
    registry.register(DataSourceSpec(key="src", name="S", domain="d"))
    registry.register(DataSourceSpec(key="src", name="S2", domain="d2"), replace=True)
    assert registry.get("src").name == "S2"


def test_rank_sources_prefers_target_coverage_and_low_drift():
    from atlas.data.source_registry import DataSourceRegistry, DataSourceSpec

    registry = DataSourceRegistry()
    registry.register(
        DataSourceSpec(
            key="a",
            name="A",
            domain="inorganic_crystals",
            primary_targets=["band_gap"],
        )
    )
    registry.register(
        DataSourceSpec(
            key="b",
            name="B",
            domain="inorganic_crystals",
            primary_targets=["formation_energy"],
        )
    )
    registry.update_reliability("a", successes=9, failures=1)
    registry.update_reliability("b", successes=9, failures=1)

    ranked = registry.rank_sources(
        target="band_gap",
        drift_by_source={"a": 0.1, "b": 0.1},
    )
    assert ranked[0][0] == "a"


def test_fuse_scalar_estimates_prefers_precise_and_reliable_source():
    from atlas.data.source_registry import (
        DataSourceRegistry,
        DataSourceSpec,
        SourceEstimate,
    )

    registry = DataSourceRegistry()
    registry.register(DataSourceSpec(key="x", name="X", domain="d"))
    registry.register(DataSourceSpec(key="y", name="Y", domain="d"))
    registry.update_reliability("x", successes=15, failures=1)
    registry.update_reliability("y", successes=1, failures=10)

    fused = registry.fuse_scalar_estimates(
        [
            SourceEstimate(source_key="x", value=1.0, std=0.1),
            SourceEstimate(source_key="y", value=3.0, std=0.5),
        ],
        pairwise_correlation=0.3,
    )
    assert 0.9 <= fused.mean <= 1.4
    assert fused.weights["x"] > fused.weights["y"]
    assert fused.std > 0


def test_fuse_scalar_estimates_correlation_inflates_uncertainty():
    from atlas.data.source_registry import (
        DataSourceRegistry,
        DataSourceSpec,
        SourceEstimate,
    )

    registry = DataSourceRegistry()
    registry.register(DataSourceSpec(key="x", name="X", domain="d"))
    registry.register(DataSourceSpec(key="y", name="Y", domain="d"))
    registry.update_reliability("x", successes=10, failures=1)
    registry.update_reliability("y", successes=10, failures=1)

    independent = registry.fuse_scalar_estimates(
        [
            SourceEstimate(source_key="x", value=2.0, std=0.2),
            SourceEstimate(source_key="y", value=2.2, std=0.2),
        ],
        pairwise_correlation=0.0,
    )
    correlated = registry.fuse_scalar_estimates(
        [
            SourceEstimate(source_key="x", value=2.0, std=0.2),
            SourceEstimate(source_key="y", value=2.2, std=0.2),
        ],
        pairwise_correlation=0.8,
    )
    assert correlated.std > independent.std


def test_reliability_snapshot_and_scope_restore_state():
    from atlas.data.source_registry import DataSourceRegistry, DataSourceSpec

    registry = DataSourceRegistry()
    registry.register(DataSourceSpec(key="x", name="X", domain="d"))

    before = registry.get_reliability("x")
    snapshot = registry.snapshot_reliability()
    registry.update_reliability("x", successes=9, failures=0)
    changed = registry.get_reliability("x")
    assert changed.mean > before.mean

    registry.restore_reliability(snapshot)
    restored = registry.get_reliability("x")
    assert restored.alpha == before.alpha
    assert restored.beta == before.beta

    with registry.reliability_scope():
        registry.update_reliability("x", successes=5, failures=0)
        assert registry.get_reliability("x").mean > restored.mean
    assert registry.get_reliability("x").mean == restored.mean


def test_fuse_scalar_estimates_nonnegative_constraint():
    from atlas.data.source_registry import (
        DataSourceRegistry,
        DataSourceSpec,
        SourceEstimate,
    )

    registry = DataSourceRegistry()
    registry.register(DataSourceSpec(key="noisy", name="Noisy", domain="d"))
    registry.register(DataSourceSpec(key="clean", name="Clean", domain="d"))
    registry.update_reliability("noisy", successes=10, failures=1)
    registry.update_reliability("clean", successes=10, failures=1)

    unconstrained = registry.fuse_scalar_estimates(
        [
            SourceEstimate(source_key="noisy", value=5.0, std=3.0),
            SourceEstimate(source_key="clean", value=1.0, std=1.0),
        ],
        pairwise_correlation=0.8,
        weight_constraint="unconstrained",
    )
    constrained = registry.fuse_scalar_estimates(
        [
            SourceEstimate(source_key="noisy", value=5.0, std=3.0),
            SourceEstimate(source_key="clean", value=1.0, std=1.0),
        ],
        pairwise_correlation=0.8,
        weight_constraint="nonnegative",
    )
    assert unconstrained.weights["noisy"] < 0.0
    assert constrained.weights["noisy"] >= 0.0
    assert constrained.weights["clean"] >= 0.0
    assert abs(sum(constrained.weights.values()) - 1.0) < 1e-8


def test_fuse_scalar_estimates_residual_based_correlation():
    from atlas.data.source_registry import (
        DataSourceRegistry,
        DataSourceSpec,
        SourceEstimate,
    )

    registry = DataSourceRegistry()
    registry.register(DataSourceSpec(key="a", name="A", domain="d"))
    registry.register(DataSourceSpec(key="b", name="B", domain="d"))
    registry.update_reliability("a", successes=12, failures=1)
    registry.update_reliability("b", successes=12, failures=1)

    independent = registry.fuse_scalar_estimates(
        [
            SourceEstimate(source_key="a", value=2.0, std=0.2),
            SourceEstimate(source_key="b", value=2.2, std=0.2),
        ],
        pairwise_correlation=0.0,
    )
    residuals = {
        "a": [0.10, -0.05, 0.08, -0.02, 0.06, -0.01],
        "b": [0.11, -0.04, 0.07, -0.03, 0.05, -0.02],
    }
    residual_based = registry.fuse_scalar_estimates(
        [
            SourceEstimate(source_key="a", value=2.0, std=0.2),
            SourceEstimate(source_key="b", value=2.2, std=0.2),
        ],
        residuals_by_source=residuals,
        correlation_shrinkage=0.1,
    )
    assert residual_based.std > independent.std


def test_register_sanitizes_invalid_reliability_priors():
    from atlas.data.source_registry import DataSourceRegistry, DataSourceSpec

    registry = DataSourceRegistry()
    registry.register(
        DataSourceSpec(
            key="bad",
            name="Bad",
            domain="d",
            reliability_prior_alpha=float("nan"),
            reliability_prior_beta=-5.0,
        )
    )
    rel = registry.get_reliability("bad")
    assert rel.alpha > 0.0
    assert rel.beta > 0.0
    assert 0.0 <= rel.mean <= 1.0


def test_update_reliability_rejects_non_finite_inputs():
    import pytest

    from atlas.data.source_registry import DataSourceRegistry, DataSourceSpec

    registry = DataSourceRegistry()
    registry.register(DataSourceSpec(key="x", name="X", domain="d"))
    with pytest.raises(ValueError):
        registry.update_reliability("x", successes=float("nan"), failures=1.0)


def test_update_reliability_rejects_boolean_inputs():
    import pytest

    from atlas.data.source_registry import DataSourceRegistry, DataSourceSpec

    registry = DataSourceRegistry()
    registry.register(DataSourceSpec(key="x", name="X", domain="d"))
    with pytest.raises(ValueError, match="successes"):
        registry.update_reliability("x", successes=True, failures=1.0)  # type: ignore[arg-type]


def test_estimate_correlation_matrix_rejects_duplicate_source_keys():
    import pytest

    from atlas.data.source_registry import DataSourceRegistry, DataSourceSpec

    registry = DataSourceRegistry()
    registry.register(DataSourceSpec(key="x", name="X", domain="d"))
    with pytest.raises(ValueError):
        registry.estimate_correlation_matrix(["x", "x"], {"x": [0.0, 1.0, 2.0]})


def test_source_score_remains_finite_with_non_finite_drift_inputs():
    from atlas.data.source_registry import DataSourceRegistry, DataSourceSpec

    registry = DataSourceRegistry()
    registry.register(DataSourceSpec(key="x", name="X", domain="d", primary_targets=["band_gap"]))
    score = registry.source_score("x", target="band_gap", drift_distance=float("nan"), drift_lambda=float("nan"))
    assert 0.0 <= score <= 1.0


def test_fuse_scalar_estimates_ignores_invalid_estimates():
    from atlas.data.source_registry import DataSourceRegistry, DataSourceSpec, SourceEstimate

    registry = DataSourceRegistry()
    registry.register(DataSourceSpec(key="a", name="A", domain="d"))
    registry.update_reliability("a", successes=5, failures=1)

    fused = registry.fuse_scalar_estimates(
        [
            SourceEstimate(source_key="a", value=2.5, std=0.2),
            SourceEstimate(source_key="a", value=float("nan"), std=0.2),
            SourceEstimate(source_key="a", value=3.0, std=float("inf")),
        ]
    )
    assert abs(fused.mean - 2.5) < 1e-12
    assert fused.std > 0.0


def test_fuse_scalar_estimates_duplicate_source_keys_are_disambiguated():
    from atlas.data.source_registry import DataSourceRegistry, DataSourceSpec, SourceEstimate

    registry = DataSourceRegistry()
    registry.register(DataSourceSpec(key="a", name="A", domain="d"))
    registry.update_reliability("a", successes=10, failures=1)

    fused = registry.fuse_scalar_estimates(
        [
            SourceEstimate(source_key="a", value=2.0, std=0.2),
            SourceEstimate(source_key="a", value=2.1, std=0.2),
        ],
        pairwise_correlation=0.5,
    )
    assert "a#0" in fused.weights
    assert "a#1" in fused.weights
    assert abs(sum(fused.weights.values()) - 1.0) < 1e-9
