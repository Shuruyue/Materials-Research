"""Tests for typed active-learning policy config/state."""

from __future__ import annotations

from types import SimpleNamespace

from atlas.active_learning.policy_state import ActiveLearningPolicyConfig, PolicyState


def test_policy_config_from_profile_and_validation():
    profile = SimpleNamespace(
        policy_name="CMOEIC",
        risk_mode="HYBRID",
        cost_aware=True,
        calibration_window=96,
        ood_combination="or",
    )
    cfg = ActiveLearningPolicyConfig.from_profile(profile)
    assert cfg.policy_name == "cmoeic"
    assert cfg.risk_mode == "hybrid"
    assert cfg.cost_aware is True
    assert cfg.calibration_window == 96
    assert cfg.ood_combination == "or"


def test_policy_state_calibration_updates_scales_from_history():
    cfg = ActiveLearningPolicyConfig(policy_name="cmoeic", calibration_window=64)
    st = PolicyState()

    history = []
    for i in range(20):
        pred = -1.0 + 0.01 * i
        obs = pred + (0.02 if i % 2 == 0 else -0.02)
        history.append(SimpleNamespace(energy_mean=pred, energy_per_atom=obs, energy_std=0.01))

    st.update_calibration(history, cfg=cfg)

    assert st.calibration_points >= cfg.calibration_min_points
    assert st.std_scale >= 1.0
    assert st.conformal_scale >= 1.0
    assert st.last_calibration_error > 0.0
