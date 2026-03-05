"""Tests for atlas.discovery.alchemy.calculator."""

from __future__ import annotations

import types

import ase
import numpy as np
import pytest
import torch

import atlas.discovery.alchemy.calculator as calc_mod


def _install_fake_mace_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    tools_mod = types.ModuleType("mace.tools")

    class _AtomicNumberTable:
        def __init__(self, numbers):
            self.numbers = list(numbers)

    tools_mod.AtomicNumberTable = _AtomicNumberTable
    monkeypatch.setitem(__import__("sys").modules, "mace.tools", tools_mod)


class _FakeBatch(dict):
    def to(self, _device):
        return self


class _FakeModel(torch.nn.Module):
    def __init__(self, *, missing_key: str | None = None):
        super().__init__()
        self.atomic_numbers = [14, 8]
        self.r_max = torch.tensor(5.0)
        self._missing_key = missing_key

    def forward(self, _batch, **_kwargs):
        payload: dict[str, torch.Tensor] = {
            "energy": torch.tensor(1.23, dtype=torch.float32),
            "forces": torch.zeros((2, 3), dtype=torch.float32),
            "stress": torch.zeros((1, 3, 3), dtype=torch.float32),
        }
        if self._missing_key is not None:
            payload.pop(self._missing_key, None)
        return payload


class _FakeAlchemyManager(torch.nn.Module):
    def __init__(
        self,
        *,
        atoms,
        alchemical_pairs,
        alchemical_weights,
        z_table,  # noqa: ARG002
        r_max,  # noqa: ARG002
    ):
        super().__init__()
        self.alchemical_weights = torch.nn.Parameter(alchemical_weights.clone())
        self.alchemical_pairs = alchemical_pairs
        self.atomic_numbers = np.array(atoms.get_atomic_numbers(), dtype=int)
        self.weight_indices = np.array([1, 0], dtype=int)
        self.atom_indices = np.array([0, 1], dtype=int)

    def forward(self, _positions, _cell):
        return _FakeBatch(dummy=torch.tensor(1.0))


def test_validate_alchemical_pairs_rejects_invalid_inputs():
    with pytest.raises(TypeError, match="sequence"):
        calc_mod._validate_alchemical_pairs("bad", num_atoms=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        calc_mod._validate_alchemical_pairs([], num_atoms=2)
    with pytest.raises(ValueError, match="integer-valued"):
        calc_mod._validate_alchemical_pairs([[(True, 14)]], num_atoms=2)  # type: ignore[list-item]


def test_weight_array_rejects_out_of_range_and_boolean_inputs():
    with pytest.raises(ValueError, match="numeric, not boolean"):
        calc_mod._as_weight_array([True], expected_size=1)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        calc_mod._as_weight_array([1.5], expected_size=1)


def test_calculator_uses_normalized_device_for_model_loading(monkeypatch: pytest.MonkeyPatch):
    _install_fake_mace_tools(monkeypatch)
    calls: list[str] = []

    def _fake_loader(*, model_size, device):  # noqa: ARG001
        calls.append(device)
        return _FakeModel()

    monkeypatch.setattr(
        calc_mod,
        "_load_alchemy_api",
        lambda: (_FakeAlchemyManager, _fake_loader),
    )

    atoms = ase.Atoms("SiO", positions=[[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]])
    calc = calc_mod.AlchemicalMACECalculator(
        atoms=atoms,
        alchemical_pairs=[[(0, 14)]],
        alchemical_weights=[1.0],
        device="cuda",
    )

    assert calc.device == "cpu"
    assert calls == ["cpu"]


def test_calculator_calculate_rejects_missing_model_outputs(monkeypatch: pytest.MonkeyPatch):
    _install_fake_mace_tools(monkeypatch)
    monkeypatch.setattr(
        calc_mod,
        "_load_alchemy_api",
        lambda: (_FakeAlchemyManager, lambda **_kwargs: _FakeModel(missing_key="forces")),
    )

    atoms = ase.Atoms("SiO", positions=[[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]])
    calc = calc_mod.AlchemicalMACECalculator(
        atoms=atoms,
        alchemical_pairs=[[(0, 14)]],
        alchemical_weights=[1.0],
    )
    with pytest.raises(KeyError, match="missing keys"):
        calc.calculate(atoms=atoms)
