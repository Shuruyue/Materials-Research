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


def test_policy_config_from_profile_coerces_invalid_types_to_safe_defaults():
    profile = SimpleNamespace(
        policy_name="cmoeic",
        risk_mode="soft",
        cost_aware="yes",
        calibration_window="bad-int",
        relax_timeout_sec="bad-float",
        relax_retry_backoff_sec="0.2",
        relax_retry_backoff_max_sec="bad-float",
        relax_retry_jitter="1.7",
    )
    cfg = ActiveLearningPolicyConfig.from_profile(profile)
    assert cfg.cost_aware is True
    assert cfg.calibration_window >= 1
    assert cfg.relax_timeout_sec >= 0.0
    assert cfg.relax_retry_backoff_sec >= 0.0
    assert cfg.relax_retry_backoff_max_sec >= cfg.relax_retry_backoff_sec
    assert 0.0 <= cfg.relax_retry_jitter <= 1.0


def test_policy_config_from_profile_rejects_non_integral_integer_fields():
    profile = SimpleNamespace(
        policy_name="cmoeic",
        risk_mode="soft",
        calibration_window=96.5,
        calibration_min_points=9.9,
        relax_max_retries=2.4,
    )
    cfg = ActiveLearningPolicyConfig.from_profile(profile)
    assert cfg.calibration_window == 128
    assert cfg.calibration_min_points == 12
    assert cfg.relax_max_retries == 1


def test_policy_config_unknown_bool_string_falls_back_to_default():
    profile = SimpleNamespace(
        policy_name="legacy",
        risk_mode="soft",
        cost_aware="maybe",
    )
    cfg = ActiveLearningPolicyConfig.from_profile(profile, cost_aware=False)
    assert cfg.cost_aware is False


def test_policy_config_validated_sanitizes_nonfinite_numeric_fields():
    cfg = ActiveLearningPolicyConfig(
        policy_name="cmoeic",
        risk_mode="soft",
        ood_gate_threshold=float("nan"),
        conformal_gate_threshold=float("inf"),
        conformal_alpha=float("nan"),
        relax_cost_base=float("nan"),
        classify_cost_base=float("nan"),
        relax_retry_backoff_max_sec=float("nan"),
        relax_retry_jitter=float("nan"),
    ).validated()
    assert 0.0 <= cfg.ood_gate_threshold <= 1.0
    assert 0.0 <= cfg.conformal_gate_threshold <= 1.0
    assert 1e-6 <= cfg.conformal_alpha <= 0.25
    assert cfg.relax_cost_base > 0.0
    assert cfg.classify_cost_base > 0.0
    assert cfg.relax_retry_backoff_max_sec >= cfg.relax_retry_backoff_sec
    assert 0.0 <= cfg.relax_retry_jitter <= 1.0


def test_policy_state_from_dict_sanitizes_negative_and_nonfinite_values():
    state = PolicyState.from_dict(
        {
            "iteration": -3,
            "calibration_points": -7,
            "std_scale": float("nan"),
            "conformal_scale": -2.0,
            "last_calibration_error": -0.5,
            "relax_circuit_open_until_iter": -4,
            "relax_consecutive_failures": -2,
        }
    )
    assert state.iteration == 0
    assert state.calibration_points == 0
    assert state.std_scale > 0.0
    assert state.conformal_scale > 0.0
    assert state.last_calibration_error >= 0.0
    assert state.relax_circuit_open_until_iter == 0
    assert state.relax_consecutive_failures == 0


def test_policy_state_from_dict_rejects_non_integral_integer_fields():
    state = PolicyState.from_dict(
        {
            "iteration": 5.7,
            "calibration_points": 11.3,
            "relax_circuit_open_until_iter": 2.9,
            "relax_consecutive_failures": 4.2,
        }
    )
    assert state.iteration == 0
    assert state.calibration_points == 0
    assert state.relax_circuit_open_until_iter == 0
    assert state.relax_consecutive_failures == 0


def test_policy_state_validated_sanitizes_direct_nonfinite_values():
    state = PolicyState(
        iteration=float("nan"),
        calibration_points=float("inf"),
        std_scale=float("nan"),
        conformal_scale=float("-inf"),
        last_calibration_error=float("nan"),
    ).validated()
    assert state.iteration == 0
    assert state.calibration_points >= 0
    assert state.std_scale > 0.0
    assert state.conformal_scale > 0.0
    assert state.last_calibration_error >= 0.0


def test_policy_state_calibration_with_finite_sample_quantile_remains_stable():
    cfg = ActiveLearningPolicyConfig(
        policy_name="cmoeic",
        calibration_window=16,
        calibration_min_points=4,
        conformal_alpha=0.2,
    )
    st = PolicyState()
    history = []
    for i in range(8):
        pred = -1.0 + 0.01 * i
        obs = pred + (0.03 if i < 4 else -0.01)
        history.append(SimpleNamespace(energy_mean=pred, energy_per_atom=obs, energy_std=0.01))

    st.update_calibration(history, cfg=cfg)
    assert st.calibration_points == 8
    assert st.std_scale >= 1.0
    assert st.conformal_scale >= 1.0
