from __future__ import annotations

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
    assert model.seen["edge_vectors"] is batch.edge_vec
    assert model.seen["edge_index_3body"] is batch.edge_index_3body
    assert model.seen["encoder_kwargs"] == {"flag": 1}
