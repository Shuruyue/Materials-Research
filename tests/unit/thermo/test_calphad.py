"""Unit tests for atlas.thermo.calphad."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from atlas.thermo.calphad import CalphadCalculator, _normalize_phase_fractions


def test_normalize_phase_fractions_filters_and_renormalizes():
    normalized = _normalize_phase_fractions(
        {
            "LIQUID": 0.7,
            "ALPHA": 0.4,
            "alpha ": 0.2,
            "": 0.1,
            "BETA": float("nan"),
        }
    )
    assert set(normalized) == {"LIQUID", "ALPHA"}
    assert sum(normalized.values()) == pytest.approx(1.0, abs=1e-8)
    assert normalized["LIQUID"] > normalized["ALPHA"]


def test_normalize_phase_fractions_renormalizes_subunit_sum():
    normalized = _normalize_phase_fractions({"LIQUID": 0.3, "ALPHA": 0.2})
    assert normalized["LIQUID"] == pytest.approx(0.6, abs=1e-8)
    assert normalized["ALPHA"] == pytest.approx(0.4, abs=1e-8)
    assert sum(normalized.values()) == pytest.approx(1.0, abs=1e-8)


def test_normalize_phase_fractions_rejects_boolean_fraction_values():
    with pytest.raises(ValueError, match="must be finite numeric"):
        _normalize_phase_fractions({"LIQUID": True})  # type: ignore[arg-type]


def test_find_transus_interpolates_threshold_crossings():
    calc = object.__new__(CalphadCalculator)
    temps = np.array([600.0, 500.0, 400.0], dtype=float)
    liquid = np.array([1.0, 0.5, 0.0], dtype=float)

    liquidus, solidus = CalphadCalculator._find_transus(calc, temps, liquid)

    assert liquidus == pytest.approx(598.0, abs=1e-6)
    assert solidus == pytest.approx(402.0, abs=1e-6)


def test_equilibrium_at_renormalizes_fraction_sum(monkeypatch):
    class _DummyVariables:
        T = "T"
        P = "P"
        N = "N"

        @staticmethod
        def X(elem: str) -> str:
            return f"X({elem})"

    class _DummyResult:
        Phase = np.array(["LIQUID", "ALPHA", "ALPHA"], dtype=object)
        NP = np.array([0.7, 0.4, 0.2], dtype=float)

    def _equilibrium(_db, _components, _phases, _conditions):
        return _DummyResult()

    monkeypatch.setitem(
        sys.modules,
        "pycalphad",
        types.SimpleNamespace(equilibrium=_equilibrium, variables=_DummyVariables),
    )

    calc = object.__new__(CalphadCalculator)
    calc.components = ["SN", "AG", "CU"]
    calc.dependent_component = "CU"
    calc.db = object()
    calc.all_phases = ["LIQUID", "ALPHA"]

    result = calc.equilibrium_at({"SN": 0.95, "AG": 0.05}, T=500.0)
    assert result.stable_phases == ["LIQUID", "ALPHA"]
    assert sum(result.phase_fractions.values()) == pytest.approx(1.0, abs=1e-8)
    assert result.phase_fractions["LIQUID"] > result.phase_fractions["ALPHA"]


def test_solidification_path_rejects_non_integral_steps():
    calc = object.__new__(CalphadCalculator)
    calc.components = ["SN", "AG", "CU"]
    calc.dependent_component = "CU"

    def _normalize(comp):
        return {"SN": 0.95, "AG": 0.03, "CU": 0.02}

    calc._normalize_composition = _normalize
    calc._equilibrium_simulation = lambda *_args, **_kwargs: None
    calc._scheil_simulation = lambda *_args, **_kwargs: None

    with pytest.raises(ValueError, match="n_steps must be an integer"):
        calc.solidification_path({"SN": 0.95, "AG": 0.03}, n_steps=20.5)


def test_get_composition_rejects_non_string_alloy_name():
    calc = object.__new__(CalphadCalculator)
    with pytest.raises(TypeError, match="alloy name must be a string"):
        calc.get_composition(True)  # type: ignore[arg-type]
