"""Tests for atlas.utils.registry helpers."""

from __future__ import annotations

import pytest

from atlas.utils.registry import Registry


def test_registry_register_get_and_build_with_normalized_names():
    registry = Registry("demo")

    @registry.register("  adder  ")
    class Adder:
        def __init__(self, left: int, right: int):
            self.total = left + right

    assert "adder" in registry
    assert " adder " in registry
    assert registry.get("adder") is Adder

    built = registry.build(" adder ", 2, right=5)
    assert built.total == 7


def test_registry_registered_names_is_sorted():
    registry = Registry("demo")
    registry.register("z")(object)
    registry.register("a")(object)
    assert registry.registered_names() == ("a", "z")


def test_registry_rejects_invalid_name_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        Registry("   ")

    registry = Registry("demo")
    with pytest.raises(ValueError, match="non-empty"):
        registry.register("   ")

    with pytest.raises(TypeError, match="must be a string"):
        registry.get(123)  # type: ignore[arg-type]

    assert 123 not in registry


def test_registry_rejects_non_callable_registration():
    registry = Registry("demo")
    with pytest.raises(TypeError, match="must be callable"):
        registry.register("bad")(42)


def test_registry_replace_flag_allows_intentional_override():
    registry = Registry("demo")

    class First:
        pass

    class Second:
        pass

    registry.register("item")(First)
    registry.register("item", replace=True)(Second)
    assert registry.get("item") is Second


def test_registry_rejects_duplicate_registration_without_replace():
    registry = Registry("demo")
    registry.register("item")(object)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("item")(object)


def test_registry_register_requires_boolean_replace_flag():
    registry = Registry("demo")
    with pytest.raises(TypeError, match="replace"):
        registry.register("item", replace="yes")  # type: ignore[arg-type]


def test_registry_get_error_includes_available_entries():
    registry = Registry("demo")
    registry.register("beta")(object)
    registry.register("alpha")(object)
    with pytest.raises(KeyError, match="Available: alpha, beta"):
        registry.get("missing")
