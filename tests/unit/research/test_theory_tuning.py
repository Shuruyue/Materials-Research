from __future__ import annotations

from atlas.training.theory_tuning import (
    adapt_params_for_next_round,
    extract_score_from_manifest,
    get_profile,
)


def test_get_profile_phase1_std():
    profile = get_profile("phase1", "cgcnn", "std")
    assert profile.phase == "phase1"
    assert profile.algorithm == "cgcnn"
    assert profile.params["epochs"] == 300
    assert profile.objective.mode == "min"


def test_extract_score_from_manifest_result_key():
    profile = get_profile("phase4", "rf", "std")
    manifest = {"result": {"validation_f1": 0.8123}}
    score = extract_score_from_manifest(manifest, profile.objective)
    assert score == 0.8123


def test_adapt_recovery_when_failed():
    profile = get_profile("phase1", "cgcnn", "std")
    params = dict(profile.params)
    next_params, reason, improvement = adapt_params_for_next_round(
        profile=profile,
        current_params=params,
        previous_score=0.40,
        current_score=None,
        failed=True,
    )
    assert reason.startswith("metric_missing_or_run_failed")
    assert improvement is None
    assert next_params["lr"] < params["lr"]
    assert next_params["epochs"] > params["epochs"]


def test_adapt_when_improvement_is_good():
    profile = get_profile("phase4", "topognn", "std")
    params = dict(profile.params)
    next_params, reason, improvement = adapt_params_for_next_round(
        profile=profile,
        current_params=params,
        previous_score=0.78,
        current_score=0.82,
        failed=False,
    )
    assert reason.startswith("improvement_good_anneal")
    assert improvement is not None and improvement > 0
    assert next_params["lr"] < params["lr"]
    assert next_params["epochs"] >= params["epochs"]
