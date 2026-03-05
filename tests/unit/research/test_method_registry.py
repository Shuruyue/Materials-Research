"""Tests for atlas.research.method_registry."""

import pytest


def test_method_registry_defaults():
    from atlas.research.method_registry import get_method, list_methods

    methods = list_methods()
    keys = [m.key for m in methods]
    assert "graph_equivariant" in keys
    assert "descriptor_tabular" in keys
    assert "physics_screened_graph" in keys
    assert "workflow_reproducible_graph" in keys
    assert "gp_active_learning" in keys
    assert "descriptor_microstructure" in keys

    primary = get_method("graph_equivariant")
    assert "GNN" in primary.summary or "equivariant" in primary.summary.lower()


def test_recommended_order():
    from atlas.research.method_registry import recommended_method_order

    order = recommended_method_order("workflow_reproducible_graph")
    assert order[0] == "workflow_reproducible_graph"


def test_method_spec_normalizes_and_freezes_text_fields():
    from atlas.research.method_registry import MethodSpec

    spec = MethodSpec(
        key=" custom_method ",
        name=" Custom Name ",
        summary=" Summary ",
        strengths=[" fast ", " robust "],
        tradeoffs=[" expensive "],
    )
    assert spec.key == "custom_method"
    assert spec.name == "Custom Name"
    assert spec.summary == "Summary"
    assert spec.strengths == ("fast", "robust")
    assert spec.tradeoffs == ("expensive",)


def test_registry_rejects_duplicate_keys_without_replace():
    from atlas.research.method_registry import MethodRegistry, MethodSpec

    registry = MethodRegistry()
    registry.register(MethodSpec(key="k", name="n1", summary="s1"))
    with pytest.raises(ValueError):
        registry.register(MethodSpec(key="k", name="n2", summary="s2"))


def test_method_spec_normalizes_string_strength_tradeoff_inputs():
    from atlas.research.method_registry import MethodSpec

    spec = MethodSpec(
        key="k",
        name="n",
        summary="s",
        strengths="single strength",
        tradeoffs="single tradeoff",
    )
    assert spec.strengths == ("single strength",)
    assert spec.tradeoffs == ("single tradeoff",)


def test_method_spec_strength_tradeoff_deduplicates_after_strip():
    from atlas.research.method_registry import MethodSpec

    spec = MethodSpec(
        key="k",
        name="n",
        summary="s",
        strengths=[" robust ", "robust", "fast"],
        tradeoffs=[" expensive ", "expensive"],
    )
    assert spec.strengths == ("robust", "fast")
    assert spec.tradeoffs == ("expensive",)


def test_method_spec_rejects_non_string_key_fields():
    from atlas.research.method_registry import MethodSpec

    with pytest.raises(TypeError, match="MethodSpec.key"):
        MethodSpec(key=1, name="n", summary="s")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="entries must be strings"):
        MethodSpec(key="k", name="n", summary="s", strengths=[1])  # type: ignore[list-item]


def test_registry_get_strips_lookup_key_whitespace():
    from atlas.research.method_registry import MethodRegistry, MethodSpec

    registry = MethodRegistry()
    registry.register(MethodSpec(key="workflow_reproducible_graph", name="n", summary="s"))
    resolved = registry.get("  workflow_reproducible_graph  ")
    assert resolved.key == "workflow_reproducible_graph"


def test_method_spec_key_is_normalized_to_lowercase():
    from atlas.research.method_registry import MethodSpec

    spec = MethodSpec(key="Graph_Equivariant", name="n", summary="s")
    assert spec.key == "graph_equivariant"


def test_registry_get_supports_case_insensitive_lookup():
    from atlas.research.method_registry import MethodRegistry, MethodSpec

    registry = MethodRegistry()
    registry.register(MethodSpec(key="workflow_reproducible_graph", name="n", summary="s"))
    resolved = registry.get("WORKFLOW_REPRODUCIBLE_GRAPH")
    assert resolved.key == "workflow_reproducible_graph"


def test_registry_register_requires_boolean_replace_flag():
    from atlas.research.method_registry import MethodRegistry, MethodSpec

    registry = MethodRegistry()
    spec = MethodSpec(key="k", name="n", summary="s")
    with pytest.raises(TypeError, match="replace"):
        registry.register(spec, replace="yes")  # type: ignore[arg-type]
