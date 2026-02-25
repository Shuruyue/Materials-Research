"""Tests for atlas.research.method_registry."""


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
