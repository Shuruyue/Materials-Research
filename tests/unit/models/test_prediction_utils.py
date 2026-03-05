from __future__ import annotations

import pytest
import torch

from atlas.models.prediction_utils import (
    extract_mean_and_std,
    forward_graph_model,
    resolve_primary_edge_features,
)


def test_extract_mean_and_std_tensor():
    x = torch.tensor([[1.0], [2.0]])
    mean, std = extract_mean_and_std(x)
    assert torch.allclose(mean, x)
    assert std is None


def test_extract_mean_and_std_tensor_sanitizes_non_finite():
    x = torch.tensor([[float("nan")], [float("inf")], [float("-inf")]])
    mean, std = extract_mean_and_std(x)
    assert std is None
    assert torch.equal(mean, torch.tensor([[0.0], [1e6], [-1e6]]))


def test_extract_mean_and_std_mean_std_dict():
    payload = {
        "mean": torch.tensor([[1.0], [2.0]]),
        "std": torch.tensor([[0.1], [0.2]]),
    }
    mean, std = extract_mean_and_std(payload)
    assert torch.allclose(mean, payload["mean"])
    assert torch.allclose(std, payload["std"])


def test_extract_mean_and_std_evidential_dict():
    payload = {
        "gamma": torch.tensor([[1.5], [2.5]]),
        "nu": torch.tensor([[2.0], [2.0]]),
        "alpha": torch.tensor([[2.5], [3.0]]),
        "beta": torch.tensor([[0.5], [0.8]]),
    }
    mean, std = extract_mean_and_std(payload)
    assert torch.allclose(mean, payload["gamma"])
    assert std is not None
    assert torch.all(std > 0)


def test_extract_mean_and_std_tuple_with_none_std():
    mean, std = extract_mean_and_std((torch.tensor([1.0, 2.0]), None))
    assert torch.allclose(mean, torch.tensor([1.0, 2.0]))
    assert std is None


def test_extract_mean_and_std_broadcasts_scalar_std():
    mean, std = extract_mean_and_std((torch.tensor([1.0, 2.0]), 0.25))
    assert std is not None
    assert torch.equal(std, torch.tensor([0.25, 0.25]))


def test_extract_mean_and_std_rejects_std_that_expands_mean_shape():
    with pytest.raises(ValueError, match="would expand mean shape"):
        extract_mean_and_std((torch.tensor([1.0, 2.0]), torch.ones(2, 2)))


def test_extract_mean_and_std_sanitizes_non_finite_std():
    payload = {
        "mean": torch.tensor([[1.0], [2.0], [3.0]]),
        "std": torch.tensor([[0.1], [float("nan")], [float("inf")]]),
    }
    mean, std = extract_mean_and_std(payload)
    assert torch.allclose(mean, payload["mean"])
    assert std is not None
    assert torch.isfinite(std).all()
    assert torch.all(std >= 0)


def test_extract_mean_and_std_casts_non_floating_mean_to_float():
    payload = {"mean": torch.tensor([[1], [2]], dtype=torch.int64), "std": torch.tensor(0.5)}
    mean, std = extract_mean_and_std(payload)
    assert mean.dtype.is_floating_point
    assert std is not None and std.dtype.is_floating_point


def test_extract_mean_and_std_sanitizes_non_finite_mean_payload():
    payload = {
        "mean": torch.tensor([[float("nan")], [float("inf")], [float("-inf")]]),
        "std": torch.tensor([[0.1], [0.2], [0.3]]),
    }
    mean, std = extract_mean_and_std(payload)
    assert std is not None
    assert torch.isfinite(mean).all()
    assert torch.equal(mean, torch.tensor([[0.0], [1e6], [-1e6]]))


def test_extract_mean_and_std_evidential_non_finite_payload_is_stable():
    payload = {
        "gamma": torch.tensor([[1.0], [2.0]]),
        "nu": torch.tensor([[float("nan")], [1.5]]),
        "alpha": torch.tensor([[float("inf")], [2.0]]),
        "beta": torch.tensor([[float("nan")], [0.5]]),
    }
    _, std = extract_mean_and_std(payload)
    assert std is not None
    assert torch.isfinite(std).all()
    assert torch.all(std >= 0)


def test_extract_mean_and_std_evidential_non_finite_gamma_is_sanitized():
    payload = {
        "gamma": torch.tensor([[float("nan")], [float("inf")], [float("-inf")]]),
        "nu": torch.tensor([[1.0], [1.0], [1.0]]),
        "alpha": torch.tensor([[2.5], [2.5], [2.5]]),
        "beta": torch.tensor([[0.5], [0.5], [0.5]]),
    }
    mean, _ = extract_mean_and_std(payload)
    assert torch.isfinite(mean).all()
    assert torch.equal(mean, torch.tensor([[0.0], [1e6], [-1e6]]))


def test_extract_mean_and_std_evidential_casts_aux_dtype_to_gamma_dtype():
    payload = {
        "gamma": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        "nu": torch.tensor([[2.0], [2.0]], dtype=torch.float64),
        "alpha": torch.tensor([[2.5], [3.0]], dtype=torch.float64),
        "beta": torch.tensor([[0.5], [0.8]], dtype=torch.float64),
    }
    mean, std = extract_mean_and_std(payload)
    assert mean.dtype == torch.float32
    assert std is not None
    assert std.dtype == torch.float32


class _DummyBatch:
    def __init__(self):
        self.x = torch.randn(5, 4)
        self.edge_index = torch.randint(0, 5, (2, 8))
        self.edge_attr = torch.randn(8, 20)
        self.edge_vec = torch.randn(8, 3)
        self.edge_index_3body = torch.randint(0, 8, (2, 6))
        self.batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)


class _EncoderEq:
    sh_irreps = "1x0e"


class _ModelPrefersEdgeVec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _EncoderEq()

    def forward(self, x, edge_index, edge_feats, batch):
        return edge_feats


class _ModelPrefersEdgeAttr(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = object()

    def forward(self, x, edge_index, edge_feats, batch):
        return edge_feats


class _ModelWithKwargs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _EncoderEq()
        self.seen = {}

    def forward(
        self,
        x,
        edge_index,
        edge_feats,
        batch=None,
        tasks=None,
        edge_vectors=None,
        edge_index_3body=None,
        encoder_kwargs=None,
    ):
        self.seen = {
            "tasks": tasks,
            "edge_vectors": edge_vectors,
            "edge_index_3body": edge_index_3body,
            "encoder_kwargs": encoder_kwargs,
        }
        return {"ok": edge_feats}


def test_resolve_primary_edge_features_prefers_equivariant_edge_vec():
    batch = _DummyBatch()
    model = _ModelPrefersEdgeVec()
    edge = resolve_primary_edge_features(model, batch)
    assert torch.equal(edge, batch.edge_vec)


def test_resolve_primary_edge_features_falls_back_to_edge_attr():
    batch = _DummyBatch()
    model = _ModelPrefersEdgeAttr()
    edge = resolve_primary_edge_features(model, batch)
    assert torch.equal(edge, batch.edge_attr)


def test_forward_graph_model_passes_optional_kwargs_when_supported():
    batch = _DummyBatch()
    model = _ModelWithKwargs()
    out = forward_graph_model(
        model,
        batch,
        tasks=["formation_energy"],
        encoder_kwargs={"flag": 1},
    )
    assert "ok" in out
    assert model.seen["tasks"] == ["formation_energy"]
    # edge_vectors is intentionally NOT passed when edge_feats is already
    # batch.edge_vec (equivariant path) to avoid duplicate data.
    assert model.seen["edge_vectors"] is None
    assert model.seen["edge_index_3body"] is batch.edge_index_3body
    assert model.seen["encoder_kwargs"] == {"flag": 1}


def test_forward_graph_model_handles_uninspectable_signature(monkeypatch):
    class _ModelNoSignature(_ModelPrefersEdgeAttr):
        pass

    batch = _DummyBatch()
    model = _ModelNoSignature()
    monkeypatch.setattr(
        "atlas.models.prediction_utils.inspect.signature",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("no signature")),
    )
    out = forward_graph_model(model, batch)
    assert torch.equal(out, batch.edge_attr)


def test_forward_graph_model_validates_tasks_and_encoder_kwargs():
    batch = _DummyBatch()
    model = _ModelWithKwargs()
    with pytest.raises(ValueError, match="tasks"):
        forward_graph_model(model, batch, tasks="formation_energy")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="encoder_kwargs"):
        forward_graph_model(model, batch, encoder_kwargs=1)  # type: ignore[arg-type]


def test_forward_graph_model_infers_batch_when_missing():
    class _BatchNoBatch(_DummyBatch):
        def __init__(self):
            super().__init__()
            self.batch = None

    batch = _BatchNoBatch()
    model = _ModelPrefersEdgeAttr()
    out = forward_graph_model(model, batch)
    assert torch.equal(out, batch.edge_attr)


def test_resolve_primary_edge_features_raises_when_missing():
    class _BatchNoEdges:
        edge_attr = None
        edge_vec = None

    with pytest.raises(AttributeError):
        resolve_primary_edge_features(_ModelPrefersEdgeAttr(), _BatchNoEdges())

