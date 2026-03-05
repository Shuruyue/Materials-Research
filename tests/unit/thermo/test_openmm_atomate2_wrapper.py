"""Tests for atlas.thermo.openmm.atomate2_wrapper."""

from __future__ import annotations

import types

import pytest

import atlas.thermo.openmm.atomate2_wrapper as wrapper


def test_native_atomate2_engine_rejects_invalid_init_args():
    with pytest.raises(ValueError, match="temperature"):
        wrapper.NativeAtomate2OpenMMEngine(temperature=0.0)
    with pytest.raises(ValueError, match="step_size"):
        wrapper.NativeAtomate2OpenMMEngine(step_size=-1.0)
    with pytest.raises(ValueError, match="ensemble"):
        wrapper.NativeAtomate2OpenMMEngine(ensemble="anneal")


def test_build_maker_requires_positive_steps_for_dynamics():
    jobs = types.SimpleNamespace(
        NVTMaker=lambda **kwargs: ("nvt", kwargs),
        NPTMaker=lambda **kwargs: ("npt", kwargs),
        EnergyMinimizationMaker=lambda: ("min", {}),
    )

    engine_nvt = wrapper.NativeAtomate2OpenMMEngine(ensemble="nvt")
    with pytest.raises(ValueError, match="steps must be > 0"):
        engine_nvt._build_maker(jobs, 0)

    engine_npt = wrapper.NativeAtomate2OpenMMEngine(ensemble="npt")
    with pytest.raises(ValueError, match="steps must be > 0"):
        engine_npt._build_maker(jobs, 0)

    engine_min = wrapper.NativeAtomate2OpenMMEngine(ensemble="minimize")
    maker = engine_min._build_maker(jobs, 0)
    assert maker[0] == "min"

    with pytest.raises(ValueError, match="must be an integer"):
        engine_nvt._build_maker(jobs, 10.5)


def test_load_atomate2_jobs_module_caches(monkeypatch):
    calls = {"count": 0}
    fake_jobs = types.SimpleNamespace()

    def _fake_import(name: str):
        assert name == "atlas.third_party.atomate2.openmm.jobs.core"
        calls["count"] += 1
        return fake_jobs

    monkeypatch.setattr(wrapper, "_ATOMATE2_JOBS_MODULE", None)
    monkeypatch.setattr(wrapper.importlib, "import_module", _fake_import)

    first = wrapper._load_atomate2_jobs_module()
    second = wrapper._load_atomate2_jobs_module()
    assert first is fake_jobs
    assert second is fake_jobs
    assert calls["count"] == 1


def test_run_simulation_wraps_import_error(monkeypatch):
    engine = wrapper.NativeAtomate2OpenMMEngine()

    def _fake_import(_name: str):
        raise ImportError("missing optional dependency")

    monkeypatch.setattr(wrapper, "_ATOMATE2_JOBS_MODULE", None)
    monkeypatch.setattr(wrapper.importlib, "import_module", _fake_import)

    with pytest.raises(RuntimeError, match="Assimilated Atomate2 OpenMM jobs are unavailable"):
        engine.run_simulation(object(), steps=10)


def test_run_simulation_requires_maker_make(monkeypatch):
    class _BadMaker:
        pass

    jobs = types.SimpleNamespace(
        NVTMaker=lambda **kwargs: _BadMaker(),  # noqa: ARG005
        NPTMaker=lambda **kwargs: _BadMaker(),  # noqa: ARG005
        EnergyMinimizationMaker=lambda: _BadMaker(),
    )

    monkeypatch.setattr(wrapper, "_ATOMATE2_JOBS_MODULE", jobs)
    engine = wrapper.NativeAtomate2OpenMMEngine(ensemble="nvt")
    with pytest.raises(TypeError, match="does not define callable make"):
        engine.run_simulation(object(), steps=10)


def test_run_simulation_invokes_maker_make(monkeypatch):
    class _Maker:
        def __init__(self, payload):
            self.payload = payload

        def make(self, interchange_data):
            return {"payload": self.payload, "input": interchange_data}

    jobs = types.SimpleNamespace(
        NVTMaker=lambda **kwargs: _Maker(("nvt", kwargs)),
        NPTMaker=lambda **kwargs: _Maker(("npt", kwargs)),
        EnergyMinimizationMaker=lambda: _Maker(("minimize", {})),
    )

    monkeypatch.setattr(wrapper, "_ATOMATE2_JOBS_MODULE", jobs)
    engine = wrapper.NativeAtomate2OpenMMEngine(ensemble="nvt", temperature=350.0, step_size=2.0)
    result = engine.run_simulation({"state": 1}, steps=20)

    assert result["input"] == {"state": 1}
    mode, kwargs = result["payload"]
    assert mode == "nvt"
    assert kwargs["temperature"] == pytest.approx(350.0)
    assert kwargs["step_size"] == pytest.approx(2.0)
    assert kwargs["n_steps"] == 20
