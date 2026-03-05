"""Unit tests for scripts.phase5_active_learning.test_enumeration helpers."""

from __future__ import annotations

import json

import pytest
from pymatgen.core import DummySpecies

import scripts.phase5_active_learning.test_enumeration as demo_module
from scripts.phase5_active_learning.test_enumeration import get_perovskite_structure, run_demo


def test_get_perovskite_structure_formula():
    structure = get_perovskite_structure()
    assert structure.composition.reduced_formula == "SrTiO3"


def test_run_demo_accepts_injected_enumerator_and_skip_vacancy():
    class DummyEnumerator:
        calls: list[tuple[dict[str, object], dict[str, object]]] = []

        def __init__(self, base):
            self.base = base

        def generate(self, substitutions, **kwargs):
            self.__class__.calls.append((substitutions, kwargs))
            return [self.base.copy()]

    DummyEnumerator.calls.clear()
    summary = run_demo(enumerator_cls=DummyEnumerator, include_vacancies=False, verbose=False)
    assert summary["simple_substitution_count"] == 1
    assert summary["vacancy_substitution_count"] == 0
    assert len(DummyEnumerator.calls) == 1

    DummyEnumerator.calls.clear()
    summary = run_demo(enumerator_cls=DummyEnumerator, include_vacancies=True, verbose=False)
    assert summary["vacancy_substitution_count"] == 1
    assert len(DummyEnumerator.calls) == 2
    vac_subs = DummyEnumerator.calls[1][0]
    assert "O" in vac_subs
    assert isinstance(vac_subs["O"][1], DummySpecies)


def test_run_demo_validates_enumerator_contract():
    class BadEnumerator:
        def __init__(self, _base):
            pass

    try:
        run_demo(enumerator_cls=BadEnumerator, verbose=False)
    except TypeError as exc:
        assert "generate" in str(exc)
    else:
        raise AssertionError("Expected TypeError for missing generate()")


def test_run_demo_rejects_non_structure_generate_results():
    class BadEnumerator:
        def __init__(self, _base):
            pass

        def generate(self, _substitutions, **_kwargs):
            return ["bad-payload"]

    with pytest.raises(TypeError, match="pymatgen Structure"):
        run_demo(enumerator_cls=BadEnumerator, verbose=False)


def test_summary_count_coercion_rejects_bool_and_fractional_inputs():
    with pytest.raises(ValueError, match="non-negative integer"):
        demo_module._coerce_non_negative_int(True, name="summary_count")
    with pytest.raises(ValueError, match="non-negative integer"):
        demo_module._coerce_non_negative_int(1.5, name="summary_count")


def test_main_json_mode_disables_verbose_output(monkeypatch, capsys):
    calls: dict[str, bool] = {}

    def _fake_run_demo(*, enumerator_cls=None, include_vacancies=True, verbose=True):  # noqa: ARG001
        calls["verbose"] = verbose
        return {
            "base_formula": "SrTiO3",
            "simple_substitution_count": 1,
            "vacancy_substitution_count": 0,
        }

    monkeypatch.setattr(demo_module, "run_demo", _fake_run_demo)
    code = demo_module.main(["--json"])

    captured = capsys.readouterr()
    assert code == 0
    assert calls["verbose"] is False
    payload = json.loads(captured.out.strip())
    assert payload["base_formula"] == "SrTiO3"
    assert captured.err == ""


def test_main_writes_errors_to_stderr(monkeypatch, capsys):
    def _boom(*, enumerator_cls=None, include_vacancies=True, verbose=True):  # noqa: ARG001
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(demo_module, "run_demo", _boom)
    code = demo_module.main([])

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "[ERROR] synthetic failure" in captured.err
