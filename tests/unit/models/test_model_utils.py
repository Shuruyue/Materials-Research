from __future__ import annotations

import pytest
import torch

import atlas.models as models_pkg
from atlas.models import utils as model_utils
from atlas.models.cgcnn import CGCNN
from atlas.models.m3gnet import M3GNet
from atlas.models.multi_task import MultiTaskGNN
from atlas.training.normalizers import TargetNormalizer


def _build_multitask_cgcnn(
    *,
    hidden_dim: int,
    n_conv: int,
    n_fc: int,
    pooling: str,
    jk: str,
    use_edge_gates: bool,
) -> MultiTaskGNN:
    encoder = CGCNN(
        node_dim=91,
        edge_dim=20,
        hidden_dim=hidden_dim,
        n_conv=n_conv,
        n_fc=n_fc,
        output_dim=1,
        pooling=pooling,
        jk=jk,
        message_aggr="mean",
        use_edge_gates=use_edge_gates,
    )
    return MultiTaskGNN(
        encoder=encoder,
        tasks={"formation_energy": {"type": "scalar"}},
        embed_dim=encoder.graph_dim,
    )


def _dummy_graph(batch_size: int = 2):
    n_nodes = batch_size * 6
    n_edges = batch_size * 20
    x = torch.randn(n_nodes, 91)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, 20)
    batch = torch.arange(batch_size).repeat_interleave(6)
    return x, edge_index, edge_attr, batch


def _dummy_graph_with_3body(batch_size: int = 2):
    n_nodes = batch_size * 6
    n_edges = batch_size * 24
    x = torch.randn(n_nodes, 91)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, 20)
    edge_vec = torch.randn(n_edges, 3)
    edge_index_3body = torch.randint(0, n_edges, (2, batch_size * 18))
    batch = torch.arange(batch_size).repeat_interleave(6)
    return x, edge_index, edge_attr, edge_vec, edge_index_3body, batch


def test_infer_cgcnn_config_from_state_dict_mean_max_concat():
    model = _build_multitask_cgcnn(
        hidden_dim=48,
        n_conv=4,
        n_fc=3,
        pooling="mean_max",
        jk="concat",
        use_edge_gates=True,
    )
    cfg = model_utils._infer_cgcnn_config_from_state_dict(model.state_dict())

    assert cfg["node_dim"] == 91
    assert cfg["edge_dim"] == 20
    assert cfg["hidden_dim"] == 48
    assert cfg["n_conv"] == 4
    assert cfg["n_fc"] == 3
    assert cfg["pooling"] == "mean_max"
    assert cfg["jk"] == "concat"
    assert cfg["use_edge_gates"] is True


def test_infer_cgcnn_config_from_state_dict_attn_no_gates():
    model = _build_multitask_cgcnn(
        hidden_dim=32,
        n_conv=2,
        n_fc=2,
        pooling="attn",
        jk="last",
        use_edge_gates=False,
    )
    cfg = model_utils._infer_cgcnn_config_from_state_dict(model.state_dict())

    assert cfg["hidden_dim"] == 32
    assert cfg["n_conv"] == 2
    assert cfg["n_fc"] == 2
    assert cfg["pooling"] == "attn"
    assert cfg["jk"] == "last"
    assert cfg["use_edge_gates"] is False


def test_load_phase2_model_with_cgcnn_checkpoint(tmp_path):
    model = _build_multitask_cgcnn(
        hidden_dim=40,
        n_conv=3,
        n_fc=2,
        pooling="mean_max",
        jk="concat",
        use_edge_gates=True,
    )
    checkpoint_path = tmp_path / "phase2_cgcnn_ckpt.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    loaded_model, normalizer = model_utils.load_phase2_model(checkpoint_path, device="cpu")
    assert normalizer is None

    x, edge_index, edge_attr, batch = _dummy_graph(batch_size=2)
    out = loaded_model(x, edge_index, edge_attr, batch)
    assert "formation_energy" in out
    assert out["formation_energy"].shape == (2, 1)


def test_load_phase1_model_infers_architecture_and_normalizer(tmp_path):
    model = CGCNN(
        node_dim=91,
        edge_dim=20,
        hidden_dim=72,
        n_conv=4,
        n_fc=3,
        output_dim=1,
        pooling="attn",
        jk="concat",
        message_aggr="sum",
        use_edge_gates=False,
    )
    normalizer = TargetNormalizer()
    normalizer.mean = -1.23
    normalizer.std = 0.45
    ckpt = {
        "model_state_dict": model.state_dict(),
        "normalizer": normalizer.state_dict(),
    }
    checkpoint_path = tmp_path / "phase1_ckpt.pt"
    torch.save(ckpt, checkpoint_path)

    loaded_model, loaded_norm = model_utils.load_phase1_model(checkpoint_path, device="cpu")
    assert loaded_norm is not None
    assert loaded_norm.mean == normalizer.mean
    assert loaded_norm.std == normalizer.std

    x, edge_index, edge_attr, batch = _dummy_graph(batch_size=2)
    out = loaded_model(x, edge_index, edge_attr, batch)
    assert out.shape == (2, 1)


def test_load_phase2_model_with_m3gnet_checkpoint(tmp_path):
    encoder = M3GNet(
        n_species=86,
        embed_dim=48,
        n_layers=2,
        n_rbf=20,
        max_radius=5.0,
    )
    model = MultiTaskGNN(
        encoder=encoder,
        tasks={"formation_energy": {"type": "scalar"}},
        embed_dim=encoder.embed_dim,
    )
    checkpoint_path = tmp_path / "phase2_m3gnet_ckpt.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    loaded_model, normalizer = model_utils.load_phase2_model(checkpoint_path, device="cpu")
    assert normalizer is None

    x, edge_index, edge_attr, edge_vec, edge_index_3body, batch = _dummy_graph_with_3body(batch_size=2)
    out = loaded_model(
        x,
        edge_index,
        edge_attr,
        batch,
        edge_vectors=edge_vec,
        edge_index_3body=edge_index_3body,
    )
    assert "formation_energy" in out
    assert out["formation_energy"].shape == (2, 1)


def test_normalize_state_dict_keys_conflict_raises():
    state = {
        "module.cgcnn.weight": torch.tensor([1.0]),
        "weight": torch.tensor([2.0]),
    }
    with pytest.raises(ValueError, match="Conflicting state_dict keys"):
        model_utils._normalize_state_dict_keys(state)


def test_load_scalar_normalizer_malformed_payload_returns_none():
    payload = {"normalizer": {"mean": "bad", "std": "bad"}}
    norm = model_utils._load_scalar_normalizer(payload)
    assert norm is None


def test_load_phase2_model_rejects_invalid_state_dict_payload(tmp_path):
    checkpoint_path = tmp_path / "bad_phase2.pt"
    torch.save({"model_state_dict": ["not", "a", "dict"]}, checkpoint_path)
    with pytest.raises(ValueError, match="invalid 'model_state_dict'"):
        model_utils.load_phase2_model(checkpoint_path, device="cpu")


def test_load_phase2_model_rejects_boolean_checkpoint_path():
    with pytest.raises(ValueError, match="checkpoint_path"):
        model_utils.load_phase2_model(True, device="cpu")  # type: ignore[arg-type]


def test_load_phase1_model_rejects_missing_checkpoint_file(tmp_path):
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        model_utils.load_phase1_model(missing, device="cpu")


def test_models_package_lazy_export_is_cached():
    models_pkg.__dict__.pop("CrystalGraphBuilder", None)
    first = models_pkg.CrystalGraphBuilder
    second = models_pkg.CrystalGraphBuilder
    assert first is second


def test_models_package_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        getattr(models_pkg, "DefinitelyMissingSymbol")


def test_models_package_export_mismatch_raises_helpful_error(monkeypatch):
    models_pkg.__dict__.pop("MessagePassingLayer", None)
    monkeypatch.setattr(models_pkg, "import_module", lambda _name: object())
    with pytest.raises(AttributeError, match="does not define expected export"):
        getattr(models_pkg, "MessagePassingLayer")


def test_models_package_lazy_import_error_is_recorded(monkeypatch):
    models_pkg.__dict__.pop("BrokenModelExport", None)
    monkeypatch.setattr(
        models_pkg,
        "_EXPORTS",
        {
            **dict(models_pkg._EXPORTS),  # type: ignore[attr-defined]
            "BrokenModelExport": ("atlas.__missing_model_module_for_test__", "X"),
        },
    )
    with pytest.raises(ImportError, match="Unable to import dependency"):
        getattr(models_pkg, "BrokenModelExport")
    errors = models_pkg.get_import_errors()
    assert "BrokenModelExport" in errors
