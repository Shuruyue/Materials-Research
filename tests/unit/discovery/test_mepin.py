"""Tests for atlas.discovery.stability.mepin."""

from __future__ import annotations

import ase
import numpy as np
import pytest
import torch

import atlas.discovery.stability.mepin as mepin


def test_normalize_model_type_and_count_helpers():
    assert mepin._normalize_model_type("cyclo_l") == "cyclo_L"
    assert mepin._normalize_model_type(" T1X_L ") == "t1x_L"
    with pytest.raises(ValueError, match="Unknown model_type"):
        mepin._normalize_model_type("bad")
    with pytest.raises(ValueError, match="num_images"):
        mepin._coerce_int_with_min(True, name="num_images", minimum=2)
    with pytest.raises(ValueError, match="num_images"):
        mepin._coerce_int_with_min(2.5, name="num_images", minimum=2)


def test_mepin_predict_path_validates_inputs(monkeypatch, tmp_path):
    class _FakeModel:
        def eval(self):
            return self

        def to(self, _device):
            return self

        def __call__(self, _batch):
            return torch.zeros((2 * 2 * 3,), dtype=torch.float32)

    class _TripleCross:
        @staticmethod
        def load_from_checkpoint(_path, map_location=None):  # noqa: ARG004
            return _FakeModel()

    class _Batch:
        def to(self, _device):
            return self

    def _create_batch(*_args, **_kwargs):
        return _Batch()

    monkeypatch.setattr(mepin, "_load_mepin_api", lambda: (_TripleCross, _create_batch))

    ckpt = tmp_path / "cyclo_L.ckpt"
    ckpt.write_text("ok", encoding="utf-8")

    evaluator = mepin.MEPINStabilityEvaluator(
        checkpoint_path=str(ckpt),
        device="cpu",
        model_type="cyclo_l",
    )

    reactant = ase.Atoms("Li2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    product = ase.Atoms("Li2", positions=[[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]])
    traj = evaluator.predict_path(reactant, product, num_images=2)
    assert len(traj) == 2

    with pytest.raises(ValueError, match="num_images"):
        evaluator.predict_path(reactant, product, num_images=1)
    with pytest.raises(TypeError, match="ase.Atoms"):
        evaluator.predict_path("bad", product, num_images=2)  # type: ignore[arg-type]

    bad_product = product.copy()
    bad_pos = bad_product.get_positions()
    bad_pos[0, 0] = np.nan
    bad_product.set_positions(bad_pos)
    with pytest.raises(ValueError, match="positions must be finite"):
        evaluator.predict_path(reactant, bad_product, num_images=2)
