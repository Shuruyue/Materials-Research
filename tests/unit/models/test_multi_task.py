"""Tests for MultiTaskGNN encoder-kwargs passthrough."""

import torch

from atlas.models.multi_task import MultiTaskGNN


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
