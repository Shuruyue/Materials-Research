"""Tests for phase5 launcher command construction."""

from __future__ import annotations

from argparse import Namespace

from scripts.phase5_active_learning.run_phase5 import build_command


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
