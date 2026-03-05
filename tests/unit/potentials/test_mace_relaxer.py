"""Tests for atlas.potentials.mace_relaxer."""

from __future__ import annotations

import pytest

from atlas.potentials.mace_relaxer import MACERelaxer, _coerce_non_negative_int, _normalize_device


def test_mace_relaxer_int_and_device_normalizers():
    assert _coerce_non_negative_int(0, name="steps") == 0
    assert _coerce_non_negative_int(3.0, name="steps") == 3
    with pytest.raises(ValueError, match="steps"):
        _coerce_non_negative_int(True, name="steps")
    with pytest.raises(ValueError, match="steps"):
        _coerce_non_negative_int(2.5, name="steps")
    with pytest.raises(ValueError, match="Invalid device string"):
        _normalize_device(True)  # type: ignore[arg-type]


def test_mace_relaxer_batch_relax_rejects_non_integral_n_jobs(monkeypatch):
    relaxer = MACERelaxer(use_foundation=False)
    monkeypatch.setattr(relaxer, "relax_structure", lambda *_args, **_kwargs: {"ok": True})
    with pytest.raises(ValueError, match="n_jobs"):
        relaxer.batch_relax([], n_jobs=1.5)
