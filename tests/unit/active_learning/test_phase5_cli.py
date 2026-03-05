"""Tests for phase5 launcher command construction."""

from __future__ import annotations

from argparse import Namespace

import pytest

from scripts.phase5_active_learning.run_discovery import (
    _extract_classifier_state_dict,
    _validate_discovery_args,
)
from scripts.phase5_active_learning.run_discovery import (
    main as run_discovery_main,
)
from scripts.phase5_active_learning.run_phase5 import _validate_args, build_command
from scripts.phase5_active_learning.run_phase5 import main as run_phase5_main


def _phase5_args(**overrides) -> Namespace:
    payload = {
        "iterations": 1,
        "candidates": 5,
        "top": 3,
        "seeds": 2,
        "calibration_window": 64,
        "preflight_max_samples": 0,
        "preflight_split_seed": 42,
        "preflight_timeout_sec": 1800,
        "preflight_property_group": "priority7",
        "acq_kappa": 1.0,
        "acq_best_f": -0.5,
        "acq_jitter": 0.01,
        "run_id": None,
        "results_dir": None,
        "preflight_only": False,
        "skip_preflight": False,
        "resume": False,
    }
    payload.update(overrides)
    return Namespace(**payload)


def test_build_command_includes_policy_flags():
    args = Namespace(
        competition=False,
        level="smoke",
        iterations=None,
        candidates=None,
        top=None,
        seeds=None,
        no_mace=False,
        resume=False,
        run_id="demo",
        results_dir=None,
        acq_strategy="hybrid",
        acq_kappa=2.0,
        acq_best_f=-0.5,
        acq_jitter=0.01,
        policy="cmoeic",
        risk_mode="hybrid",
        cost_aware=True,
        calibration_window=96,
    )

    cmd = build_command(args)

    assert "--policy" in cmd
    assert "cmoeic" in cmd
    assert "--risk-mode" in cmd
    assert "hybrid" in cmd
    assert "--cost-aware" in cmd
    assert "--calibration-window" in cmd
    assert "96" in cmd


def test_build_command_replaces_profile_defaults_instead_of_appending_duplicates():
    args = Namespace(
        competition=False,
        level="smoke",
        iterations=3,
        candidates=22,
        top=7,
        seeds=9,
        no_mace=False,
        resume=False,
        run_id=None,
        results_dir=None,
        acq_strategy=None,
        acq_kappa=None,
        acq_best_f=None,
        acq_jitter=None,
        policy="legacy",
        risk_mode="soft",
        cost_aware=False,
        calibration_window=128,
    )
    cmd = build_command(args)
    assert cmd.count("--iterations") == 1
    assert cmd[cmd.index("--iterations") + 1] == "3"
    assert cmd.count("--candidates") == 1
    assert cmd[cmd.index("--candidates") + 1] == "22"
    assert cmd.count("--top") == 1
    assert cmd[cmd.index("--top") + 1] == "7"
    assert cmd.count("--seeds") == 1
    assert cmd[cmd.index("--seeds") + 1] == "9"


def test_validate_args_rejects_top_greater_than_candidates():
    args = _phase5_args(top=8, run_id="demo", acq_kappa=2.0)
    ok, message = _validate_args(args)
    assert ok is False
    assert "--top cannot be greater than --candidates" in message


def test_validate_args_rejects_non_integral_integer_controls():
    args = _phase5_args(iterations=1.2)
    ok, message = _validate_args(args)
    assert ok is False
    assert "--iterations must be an integer" in message

    args = _phase5_args(preflight_split_seed=True)
    ok, message = _validate_args(args)
    assert ok is False
    assert "--preflight-split-seed must be an integer" in message


def test_validate_args_rejects_non_finite_acquisition_values_and_unsafe_run_id():
    args = _phase5_args(acq_kappa=float("nan"))
    ok, message = _validate_args(args)
    assert ok is False
    assert "--acq-kappa must be finite and >= 0" in message

    args.acq_kappa = 1.0
    args.acq_jitter = float("inf")
    ok, message = _validate_args(args)
    assert ok is False
    assert "--acq-jitter must be finite and >= 0" in message

    args.acq_jitter = 0.01
    args.run_id = "../escape"
    ok, message = _validate_args(args)
    assert ok is False
    assert "--run-id contains unsafe characters" in message


def test_validate_args_rejects_invalid_preflight_timeout():
    args = _phase5_args(preflight_timeout_sec=0)
    ok, message = _validate_args(args)
    assert ok is False
    assert "--preflight-timeout-sec must be > 0" in message


def test_validate_args_rejects_invalid_preflight_property_group_and_empty_results_dir():
    args = _phase5_args(preflight_property_group="bad group")
    ok, message = _validate_args(args)
    assert ok is False
    assert "--preflight-property-group" in message

    args.preflight_property_group = "priority7"
    args.results_dir = "   "
    ok, message = _validate_args(args)
    assert ok is False
    assert "--results-dir must not be empty" in message


def test_validate_args_rejects_conflicting_run_directory_and_preflight_modes():
    args = _phase5_args(run_id="demo", results_dir="artifacts/discovery")
    ok, message = _validate_args(args)
    assert ok is False
    assert "--run-id and --results-dir cannot be used together" in message

    args = _phase5_args(resume=True, results_dir="artifacts/discovery")
    ok, message = _validate_args(args)
    assert ok is False
    assert "--resume and --results-dir cannot be used together" in message

    args = _phase5_args(preflight_only=True, skip_preflight=True)
    ok, message = _validate_args(args)
    assert ok is False
    assert "--preflight-only cannot be used with --skip-preflight" in message


def test_run_discovery_validate_args_rejects_invalid_ranges():
    args = Namespace(
        iterations=2,
        candidates=5,
        top=7,
        seeds=3,
        calibration_window=64,
        acq_kappa=1.0,
        acq_jitter=0.01,
        acq_best_f=-0.5,
        run_id="demo",
        results_dir=None,
    )
    ok, message = _validate_discovery_args(args)
    assert ok is False
    assert "--top cannot be greater than --candidates" in message


def test_run_discovery_validate_args_rejects_non_integral_controls():
    args = Namespace(
        iterations=2.5,
        candidates=8,
        top=4,
        seeds=3,
        calibration_window=64,
        acq_kappa=1.0,
        acq_jitter=0.01,
        acq_best_f=-0.5,
        run_id="demo",
        results_dir=None,
        resume=False,
    )
    ok, message = _validate_discovery_args(args)
    assert ok is False
    assert "--iterations must be an integer" in message


def test_run_discovery_validate_args_rejects_invalid_floats_and_unsafe_run_id():
    args = Namespace(
        iterations=2,
        candidates=8,
        top=4,
        seeds=3,
        calibration_window=64,
        acq_kappa=float("nan"),
        acq_jitter=0.01,
        acq_best_f=-0.5,
        run_id="demo",
        results_dir=None,
    )
    ok, message = _validate_discovery_args(args)
    assert ok is False
    assert "--acq-kappa must be finite and >= 0" in message

    args.acq_kappa = 1.0
    args.acq_jitter = float("inf")
    ok, message = _validate_discovery_args(args)
    assert ok is False
    assert "--acq-jitter must be finite and >= 0" in message

    args.acq_jitter = 0.01
    args.run_id = "..\\escape"
    ok, message = _validate_discovery_args(args)
    assert ok is False
    assert "--run-id contains unsafe characters" in message


def test_run_discovery_validate_args_rejects_boolean_acquisition_values():
    args = Namespace(
        iterations=2,
        candidates=8,
        top=4,
        seeds=3,
        calibration_window=64,
        acq_kappa=True,
        acq_jitter=0.01,
        acq_best_f=-0.5,
        run_id="demo",
        results_dir=None,
        resume=False,
    )
    ok, message = _validate_discovery_args(args)
    assert ok is False
    assert "--acq-kappa must be finite and >= 0" in message

    args.acq_kappa = 1.0
    args.acq_jitter = False
    ok, message = _validate_discovery_args(args)
    assert ok is False
    assert "--acq-jitter must be finite and >= 0" in message

    args.acq_jitter = 0.01
    args.acq_best_f = True
    ok, message = _validate_discovery_args(args)
    assert ok is False
    assert "--acq-best-f must be finite" in message


def test_run_discovery_validate_args_rejects_conflicting_run_directory_flags():
    args = Namespace(
        iterations=2,
        candidates=8,
        top=4,
        seeds=3,
        calibration_window=64,
        acq_kappa=1.0,
        acq_jitter=0.01,
        acq_best_f=-0.5,
        run_id="safe",
        results_dir="out/discovery",
        resume=False,
    )
    ok, message = _validate_discovery_args(args)
    assert ok is False
    assert "--run-id and --results-dir cannot be used together" in message

    args.run_id = None
    args.resume = True
    ok, message = _validate_discovery_args(args)
    assert ok is False
    assert "--resume and --results-dir cannot be used together" in message


def test_extract_classifier_state_dict_supports_nested_and_dataparallel_payload():
    payload = {
        "epoch": 5,
        "state_dict": {
            "module.layer.weight": 1,
            "module.layer.bias": 2,
        },
    }
    state_dict = _extract_classifier_state_dict(payload)
    assert state_dict == {
        "layer.weight": 1,
        "layer.bias": 2,
    }


def test_extract_classifier_state_dict_rejects_invalid_payload():
    with pytest.raises(TypeError, match="missing state dict"):
        _extract_classifier_state_dict({"epoch": 1, "step": 3})


def test_main_validation_errors_are_reported_to_stderr(capsys):
    code = run_phase5_main(["--iterations", "0"])
    captured = capsys.readouterr()
    assert code == 2
    assert "[ERROR]" in captured.err

    code = run_discovery_main(["--iterations", "0"])
    captured = capsys.readouterr()
    assert code == 2
    assert "[ERROR]" in captured.err
