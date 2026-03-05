"""Unit tests for benchmark CLI argument parsing and validation."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from atlas.benchmark.cli import (
    _extract_state_dict,
    _load_model,
    _parse_model_kwargs,
    _validate_cli_args,
    build_parser,
    main,
)


def _valid_task() -> str:
    from atlas.benchmark.runner import MatbenchRunner

    return next(iter(MatbenchRunner.TASKS.keys()))


def test_parse_model_kwargs_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="valid JSON object"):
        _parse_model_kwargs("{not-json")


def test_parse_model_kwargs_rejects_non_object_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        _parse_model_kwargs("[]")


def test_validate_cli_args_rejects_invalid_coverage_range() -> None:
    parser = build_parser()
    args = parser.parse_args(["--task", _valid_task(), "--min-coverage-required", "1.2"])
    with pytest.raises(SystemExit):
        _validate_cli_args(parser, args)


def test_validate_cli_args_rejects_non_finite_probability_controls() -> None:
    parser = build_parser()
    args = parser.parse_args(["--task", _valid_task()])
    args.min_coverage_required = float("nan")
    with pytest.raises(SystemExit):
        _validate_cli_args(parser, args)

    args = parser.parse_args(["--task", _valid_task()])
    args.conformal_coverage = float("inf")
    with pytest.raises(SystemExit):
        _validate_cli_args(parser, args)


def test_validate_cli_args_rejects_bool_probability_controls() -> None:
    parser = build_parser()
    args = parser.parse_args(["--task", _valid_task()])
    args.min_coverage_required = True
    with pytest.raises(SystemExit):
        _validate_cli_args(parser, args)


def test_validate_cli_args_rejects_missing_checkpoint(tmp_path: Path) -> None:
    parser = build_parser()
    missing = tmp_path / "missing.ckpt"
    args = parser.parse_args(["--task", _valid_task(), "--checkpoint", str(missing)])
    with pytest.raises(SystemExit):
        _validate_cli_args(parser, args)


def test_validate_cli_args_rejects_invalid_preflight_property_group() -> None:
    parser = build_parser()
    args = parser.parse_args(["--task", _valid_task(), "--preflight-property-group", "bad group"])
    with pytest.raises(SystemExit):
        _validate_cli_args(parser, args)


def test_validate_cli_args_skip_preflight_allows_blank_property_group() -> None:
    parser = build_parser()
    args = parser.parse_args(["--task", _valid_task(), "--skip-preflight", "--dry-run"])
    args.preflight_property_group = "  "
    _validate_cli_args(parser, args)


def test_validate_cli_args_rejects_output_conflict() -> None:
    parser = build_parser()
    args = parser.parse_args(["--task", _valid_task(), "--output", "a.json", "--output-dir", "reports"])
    with pytest.raises(SystemExit):
        _validate_cli_args(parser, args)


def test_validate_cli_args_rejects_preflight_only_skip_preflight_conflict() -> None:
    parser = build_parser()
    args = parser.parse_args(["--task", _valid_task(), "--preflight-only", "--skip-preflight", "--dry-run"])
    with pytest.raises(SystemExit):
        _validate_cli_args(parser, args)


def test_validate_cli_args_rejects_non_integral_integer_controls() -> None:
    parser = build_parser()
    args = parser.parse_args(["--task", _valid_task()])
    args.batch_size = 1.5
    with pytest.raises(SystemExit):
        _validate_cli_args(parser, args)

    args = parser.parse_args(["--task", _valid_task(), "--folds", "1", "2"])
    args.folds = [1, 2.5]
    with pytest.raises(SystemExit):
        _validate_cli_args(parser, args)


def test_extract_state_dict_supports_nested_dataparallel_payload() -> None:
    payload = {
        "epoch": 1,
        "model_state_dict": {
            "module.layer.weight": torch.zeros(1),
            "module.layer.bias": torch.ones(1),
        },
    }
    state_dict = _extract_state_dict(payload)
    assert "layer.weight" in state_dict
    assert "layer.bias" in state_dict


def test_extract_state_dict_rejects_invalid_payload() -> None:
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        _extract_state_dict({"epoch": 3})


def test_load_model_accepts_nested_dataparallel_checkpoint(tmp_path: Path) -> None:
    module_name = "tests.benchmark_cli_dummy_module"
    module = types.ModuleType(module_name)

    class TinyModel(torch.nn.Module):
        def __init__(self, input_dim: int = 2, out_dim: int = 1):
            super().__init__()
            self.fc = torch.nn.Linear(input_dim, out_dim)

    module.TinyModel = TinyModel
    sys.modules[module_name] = module
    try:
        reference_model = TinyModel(input_dim=2, out_dim=1)
        wrapped_state = {
            f"module.{key}": value.clone()
            for key, value in reference_model.state_dict().items()
        }
        checkpoint = tmp_path / "tiny_model.pt"
        torch.save({"model_state_dict": wrapped_state}, checkpoint)

        model, load_info = _load_model(
            module_name=module_name,
            class_name="TinyModel",
            model_kwargs_json='{"input_dim": 2, "out_dim": 1}',
            checkpoint=str(checkpoint),
            strict_checkpoint=False,
        )
        assert isinstance(model, TinyModel)
        assert load_info["missing_keys"] == []
        assert load_info["unexpected_keys"] == []
    finally:
        sys.modules.pop(module_name, None)


def test_validate_cli_args_normalizes_fold_list() -> None:
    parser = build_parser()
    args = parser.parse_args(["--task", _valid_task(), "--folds", "3", "1", "3", "0"])
    _validate_cli_args(parser, args)
    assert args.folds == [0, 1, 3]


def test_validate_cli_args_normalizes_text_fields() -> None:
    parser = build_parser()
    task = _valid_task()
    args = parser.parse_args(["--task", f"  {task}  "])
    _validate_cli_args(parser, args)
    assert args.task == task


def test_list_tasks_bypasses_non_list_validation() -> None:
    code = main(["--list-tasks", "--batch-size", "0"])
    assert code == 0


def test_main_preflight_failure_writes_to_stderr(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "atlas.benchmark.cli.run_preflight",
        lambda **_kwargs: SimpleNamespace(return_code=9, error_message="synthetic"),
    )
    rc = main(["--task", _valid_task(), "--preflight-only"])
    captured = capsys.readouterr()
    assert rc == 9
    assert "[ERROR] Preflight failed with return code 9 (synthetic)" in captured.err
