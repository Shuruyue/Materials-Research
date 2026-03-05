"""Tests for MultiTaskGNN encoder-kwargs passthrough."""

import pytest
import torch

from atlas.models.multi_task import MultiTaskGNN, TensorHead


class _EncoderNoKwargs(torch.nn.Module):
    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.embed_dim = embed_dim

    def encode(self, node_feats, edge_index, edge_feats, batch=None):
        if batch is None:
            batch = torch.zeros(node_feats.size(0), dtype=torch.long, device=node_feats.device)
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        return torch.ones(batch_size, self.embed_dim, device=node_feats.device)


class _EncoderWithKwargs(torch.nn.Module):
    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.last_kwargs = {}

    def encode(
        self,
        node_feats,
        edge_index,
        edge_feats,
        batch=None,
        edge_vectors=None,
        edge_index_3body=None,
        flag=None,
    ):
        self.last_kwargs = {
            "edge_vectors": edge_vectors,
            "edge_index_3body": edge_index_3body,
            "flag": flag,
        }
        if batch is None:
            batch = torch.zeros(node_feats.size(0), dtype=torch.long, device=node_feats.device)
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        return torch.ones(batch_size, self.embed_dim, device=node_feats.device)


def _dummy_inputs():
    n_nodes = 6
    n_edges = 12
    x = torch.randn(n_nodes, 5)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    return x, edge_index, edge_attr, batch


def test_multitask_filters_unknown_encoder_kwargs():
    encoder = _EncoderNoKwargs(embed_dim=8)
    model = MultiTaskGNN(
        encoder=encoder,
        tasks={"formation_energy": {"type": "scalar"}},
        embed_dim=8,
    )
    x, edge_index, edge_attr, batch = _dummy_inputs()
    out = model(
        x,
        edge_index,
        edge_attr,
        batch,
        edge_vectors=torch.randn(edge_attr.size(0), 3),
        edge_index_3body=torch.randint(0, edge_attr.size(0), (2, 5)),
        encoder_kwargs={"custom_flag": True},
    )
    assert "formation_energy" in out
    assert out["formation_energy"].shape == (2, 1)


def test_multitask_passes_supported_encoder_kwargs():
    encoder = _EncoderWithKwargs(embed_dim=8)
    model = MultiTaskGNN(
        encoder=encoder,
        tasks={"formation_energy": {"type": "scalar"}},
        embed_dim=8,
    )
    x, edge_index, edge_attr, batch = _dummy_inputs()
    edge_vectors = torch.randn(edge_attr.size(0), 3)
    edge_index_3body = torch.randint(0, edge_attr.size(0), (2, 5))
    out = model(
        x,
        edge_index,
        edge_attr,
        batch,
        edge_vectors=edge_vectors,
        edge_index_3body=edge_index_3body,
        encoder_kwargs={"flag": "ok"},
    )

    assert "formation_energy" in out
    assert out["formation_energy"].shape == (2, 1)
    assert encoder.last_kwargs["edge_vectors"] is edge_vectors
    assert encoder.last_kwargs["edge_index_3body"] is edge_index_3body
    assert encoder.last_kwargs["flag"] == "ok"


def test_multitask_rejects_unknown_task_type():
    with pytest.raises(ValueError, match="Unsupported task type"):
        MultiTaskGNN(
            encoder=_EncoderNoKwargs(embed_dim=8),
            tasks={"formation_energy": {"type": "unknown"}},
            embed_dim=8,
        )
    with pytest.raises(ValueError, match="Unsupported tensor_type"):
        MultiTaskGNN(
            encoder=_EncoderNoKwargs(embed_dim=8),
            tasks={"elastic_tensor": {"type": "tensor", "tensor_type": "invalid"}},
            embed_dim=8,
        )


def test_multitask_tasks_string_selects_single_task():
    encoder = _EncoderNoKwargs(embed_dim=8)
    model = MultiTaskGNN(
        encoder=encoder,
        tasks={
            "formation_energy": {"type": "scalar"},
            "band_gap": {"type": "scalar"},
        },
        embed_dim=8,
    )
    x, edge_index, edge_attr, batch = _dummy_inputs()
    out = model(x, edge_index, edge_attr, batch, tasks="band_gap")
    assert set(out.keys()) == {"band_gap"}


def test_multitask_unknown_selected_task_raises():
    encoder = _EncoderNoKwargs(embed_dim=8)
    model = MultiTaskGNN(
        encoder=encoder,
        tasks={"formation_energy": {"type": "scalar"}},
        embed_dim=8,
    )
    x, edge_index, edge_attr, batch = _dummy_inputs()
    with pytest.raises(ValueError, match="Unknown task"):
        model(x, edge_index, edge_attr, batch, tasks=["formation_energy", "nonexistent_task"])


def test_multitask_add_task_validates_duplicates_and_type():
    encoder = _EncoderNoKwargs(embed_dim=8)
    model = MultiTaskGNN(
        encoder=encoder,
        tasks={"formation_energy": {"type": "scalar"}},
        embed_dim=8,
    )
    with pytest.raises(ValueError, match="already exists"):
        model.add_task("formation_energy", task_type="scalar")
    with pytest.raises(ValueError, match="Unsupported task type"):
        model.add_task("new_task", task_type="bad")


def test_multitask_rejects_encoder_without_encode():
    with pytest.raises(ValueError, match="callable encode"):
        MultiTaskGNN(
            encoder=torch.nn.Linear(8, 8),
            tasks={"formation_energy": {"type": "scalar"}},
            embed_dim=8,
        )


def test_multitask_normalizes_and_deduplicates_selected_tasks():
    encoder = _EncoderNoKwargs(embed_dim=8)
    model = MultiTaskGNN(
        encoder=encoder,
        tasks={
            "formation_energy": {"type": "scalar"},
            "band_gap": {"type": "scalar"},
        },
        embed_dim=8,
    )
    x, edge_index, edge_attr, batch = _dummy_inputs()
    out = model(
        x,
        edge_index,
        edge_attr,
        batch,
        tasks=[" formation_energy ", "formation_energy", "band_gap"],
    )
    assert set(out.keys()) == {"formation_energy", "band_gap"}


def test_multitask_rejects_empty_task_name_in_selection():
    encoder = _EncoderNoKwargs(embed_dim=8)
    model = MultiTaskGNN(
        encoder=encoder,
        tasks={"formation_energy": {"type": "scalar"}},
        embed_dim=8,
    )
    x, edge_index, edge_attr, batch = _dummy_inputs()
    with pytest.raises(ValueError, match="empty task name"):
        model(x, edge_index, edge_attr, batch, tasks=["  "])


def test_tensor_head_to_full_tensor_validates_components_and_preserves_dtype():
    head = TensorHead(embed_dim=8, tensor_type="elastic")
    components = torch.randn(2, 21, dtype=torch.float64)
    full = head.to_full_tensor(components)
    assert full.shape == (2, 6, 6)
    assert full.dtype == torch.float64

    with pytest.raises(ValueError, match="rank-2"):
        head.to_full_tensor(torch.randn(2, 21, 1))
    with pytest.raises(ValueError, match="expected n_components=21"):
        head.to_full_tensor(torch.randn(2, 20))
    bad = components.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        head.to_full_tensor(bad)


def test_multitask_signature_failure_falls_back_without_extra_kwargs(monkeypatch):
    encoder = _EncoderWithKwargs(embed_dim=8)
    model = MultiTaskGNN(
        encoder=encoder,
        tasks={"formation_energy": {"type": "scalar"}},
        embed_dim=8,
    )
    x, edge_index, edge_attr, batch = _dummy_inputs()
    edge_vectors = torch.randn(edge_attr.size(0), 3)
    monkeypatch.setattr(
        "atlas.models.multi_task.inspect.signature",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("no signature")),
    )
    out = model(
        x,
        edge_index,
        edge_attr,
        batch,
        edge_vectors=edge_vectors,
        encoder_kwargs={"flag": "ok"},
    )
    assert "formation_energy" in out
    assert encoder.last_kwargs["edge_vectors"] is None
    assert encoder.last_kwargs["flag"] is None
