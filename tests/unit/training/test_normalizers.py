"""Unit tests for atlas.training.normalizers."""

from types import SimpleNamespace

import pytest
import torch

from atlas.training.normalizers import MultiTargetNormalizer, TargetNormalizer


def test_target_normalizer_computes_stats_from_finite_values_only():
    dataset = [
        SimpleNamespace(band_gap=torch.tensor(1.0)),
        SimpleNamespace(band_gap=torch.tensor(2.0)),
        SimpleNamespace(band_gap=torch.tensor(float("nan"))),
        SimpleNamespace(band_gap=torch.tensor(float("inf"))),
    ]
    norm = TargetNormalizer(dataset, "band_gap")
    assert norm.mean == pytest.approx(1.5)
    assert norm.std == pytest.approx(0.5)


def test_target_normalizer_empty_or_invalid_values_fallback_to_defaults():
    dataset = [
        SimpleNamespace(band_gap=torch.tensor(float("nan"))),
        SimpleNamespace(band_gap="not-a-number"),
    ]
    norm = TargetNormalizer(dataset, "band_gap")
    assert norm.mean == pytest.approx(0.0)
    assert norm.std == pytest.approx(1.0)


def test_target_normalizer_round_trip():
    norm = TargetNormalizer()
    norm.load_state_dict({"mean": 2.0, "std": 4.0})
    x = torch.tensor([2.0, 6.0])
    assert torch.allclose(norm.denormalize(norm.normalize(x)), x)


def test_target_normalizer_load_state_validates_schema():
    norm = TargetNormalizer()
    with pytest.raises(KeyError, match="mean"):
        norm.load_state_dict({"std": 1.0})
    with pytest.raises(ValueError, match="finite"):
        norm.load_state_dict({"mean": float("nan"), "std": 1.0})


def test_target_normalizer_load_state_non_finite_std_falls_back_to_one():
    norm = TargetNormalizer()
    norm.load_state_dict({"mean": 3.0, "std": float("nan")})
    assert norm.mean == pytest.approx(3.0)
    assert norm.std == pytest.approx(1.0)


def test_target_normalizer_rejects_boolean_mean_in_state():
    norm = TargetNormalizer()
    with pytest.raises(ValueError, match="finite"):
        norm.load_state_dict({"mean": True, "std": 1.0})  # type: ignore[arg-type]


def test_target_normalizer_boolean_std_falls_back_to_one():
    norm = TargetNormalizer()
    norm.load_state_dict({"mean": 2.0, "std": False})  # type: ignore[arg-type]
    assert norm.mean == pytest.approx(2.0)
    assert norm.std == pytest.approx(1.0)


def test_multi_target_normalizer_unknown_property_is_explicit():
    norm = MultiTargetNormalizer()
    with pytest.raises(KeyError, match="unknown property"):
        norm.normalize("missing", torch.tensor(1.0))


def test_multi_target_normalizer_normalize_denormalize_accepts_trimmed_property_name():
    norm = MultiTargetNormalizer()
    norm.load_state_dict({"band_gap": {"mean": 2.0, "std": 0.5}})
    value = torch.tensor(2.5)
    z = norm.normalize(" band_gap ", value)
    assert z.item() == pytest.approx(1.0)
    restored = norm.denormalize(" band_gap ", z)
    assert restored.item() == pytest.approx(2.5)


def test_multi_target_normalizer_load_state_adds_new_normalizer():
    norm = MultiTargetNormalizer()
    norm.load_state_dict({"bulk_modulus": {"mean": 10.0, "std": 2.0}})
    val = torch.tensor(14.0)
    z = norm.normalize("bulk_modulus", val)
    assert z.item() == pytest.approx(2.0)


def test_target_normalizer_supports_iterable_dataset_without_random_access():
    dataset = ({"band_gap": value} for value in [1.0, 2.0, float("nan")])
    norm = TargetNormalizer(dataset, "band_gap")
    assert norm.mean == pytest.approx(1.5)
    assert norm.std == pytest.approx(0.5)


def test_target_normalizer_supports_mapping_dataset_values():
    dataset = {
        "a": {"band_gap": 1.0},
        "b": {"band_gap": 2.0},
        "c": {"band_gap": float("nan")},
    }
    norm = TargetNormalizer(dataset, "band_gap")
    assert norm.mean == pytest.approx(1.5)
    assert norm.std == pytest.approx(0.5)


def test_multi_target_normalizer_normalizes_property_names():
    norm = MultiTargetNormalizer(
        dataset=[SimpleNamespace(eg=1.0)],
        properties=[" eg ", "eg", "", "  "],
    )
    assert list(norm.normalizers) == ["eg"]


def test_multi_target_normalizer_rejects_non_string_property_names():
    with pytest.raises(TypeError, match="string property names"):
        MultiTargetNormalizer(
            dataset=[SimpleNamespace(eg=1.0)],
            properties=["eg", 1],  # type: ignore[list-item]
        )


def test_multi_target_normalizer_rejects_empty_state_property_names():
    norm = MultiTargetNormalizer()
    with pytest.raises(ValueError, match="non-empty"):
        norm.load_state_dict({"   ": {"mean": 1.0, "std": 1.0}})


def test_multi_target_normalizer_rejects_non_mapping_property_state():
    norm = MultiTargetNormalizer()
    with pytest.raises(TypeError, match="state\\['eg'\\]"):
        norm.load_state_dict({"eg": 1.0})  # type: ignore[arg-type]


def test_multi_target_normalizer_rejects_empty_property_name_on_access():
    norm = MultiTargetNormalizer()
    norm.load_state_dict({"eg": {"mean": 1.0, "std": 1.0}})
    with pytest.raises(ValueError, match="non-empty"):
        norm.normalize("   ", torch.tensor(1.0))
