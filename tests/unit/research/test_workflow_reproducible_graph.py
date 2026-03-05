"""Tests for reproducible workflow scaffolding."""

import json

import pytest

from atlas.research.workflow_reproducible_graph import IterationSnapshot, RunManifest, WorkflowReproducibleGraph


def test_workflow_manifest_lifecycle(tmp_path):
    workflow = WorkflowReproducibleGraph(output_dir=tmp_path)
    manifest = workflow.start(extra_metrics={"unit_test": True})
    assert manifest.status == "running"
    assert manifest.metrics["unit_test"] is True

    workflow.record_iteration(
        IterationSnapshot(
            iteration=1,
            generated=10,
            unique=8,
            relaxed=8,
            selected=3,
            duration_sec=1.25,
            stage_timings_sec={"generate": 0.5, "relax": 0.4},
            seed_pool_size=5,
        )
    )
    workflow.finalize(status="completed", extra_metrics={"top_candidates": 3})

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1

    with open(files[0], encoding="utf-8") as fp:
        data = json.load(fp)

    assert data["status"] == "completed"
    assert data["metrics"]["top_candidates"] == 3
    assert data["seed"] == 42
    assert "python_version" in data["runtime_metadata"]
    assert len(data["iterations"]) == 1
    assert data["iterations"][0]["iteration"] == 1
    assert data["iterations"][0]["seed_pool_size"] == 5
    assert data["iterations"][0]["stage_timings_sec"]["generate"] == 0.5


def test_workflow_rejects_non_monotonic_iteration_index(tmp_path):
    workflow = WorkflowReproducibleGraph(output_dir=tmp_path)
    workflow.start()
    workflow.record_iteration(IterationSnapshot(iteration=1, generated=1))
    with pytest.raises(ValueError):
        workflow.record_iteration(IterationSnapshot(iteration=1, generated=2))


def test_make_run_id_sanitizes_tokens():
    run_id = WorkflowReproducibleGraph.make_run_id(
        method_key="workflow/reproducible graph",
        data_source_key="mp 2024:Q4",
        timestamp=1700000000.0,
    )
    assert "/" not in run_id
    assert " " not in run_id
    assert "workflow-reproducible-graph" in run_id
    assert "mp-2024-Q4" in run_id


def test_workflow_start_avoids_manifest_filename_collision(tmp_path):
    workflow_a = WorkflowReproducibleGraph(output_dir=tmp_path)
    workflow_b = WorkflowReproducibleGraph(output_dir=tmp_path)

    manifest_a = workflow_a.start()
    manifest_b = workflow_b.start()

    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == 2
    assert files[0] != files[1]
    assert manifest_a.run_id in files[0].stem or manifest_a.run_id in files[1].stem
    assert manifest_b.run_id in files[0].stem or manifest_b.run_id in files[1].stem


def test_workflow_persist_sanitizes_non_finite_metric_values(tmp_path):
    workflow = WorkflowReproducibleGraph(output_dir=tmp_path)
    workflow.start()
    workflow.set_metric("nan_metric", float("nan"))
    workflow.finalize()

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["metrics"]["nan_metric"] is None


def test_workflow_finalize_rejects_invalid_status(tmp_path):
    workflow = WorkflowReproducibleGraph(output_dir=tmp_path)
    workflow.start()
    with pytest.raises(ValueError, match="status"):
        workflow.finalize(status="   ")


def test_iteration_snapshot_rejects_boolean_and_fractional_counts():
    with pytest.raises(ValueError, match="iteration"):
        IterationSnapshot(iteration=True, generated=1)
    with pytest.raises(ValueError, match="generated"):
        IterationSnapshot(iteration=1, generated=3.2)


def test_iteration_snapshot_rejects_inconsistent_count_order():
    with pytest.raises(ValueError, match="unique"):
        IterationSnapshot(iteration=1, generated=3, unique=4)
    with pytest.raises(ValueError, match="relaxed"):
        IterationSnapshot(iteration=1, generated=5, unique=4, relaxed=5)
    with pytest.raises(ValueError, match="selected"):
        IterationSnapshot(iteration=1, generated=5, unique=4, relaxed=3, selected=4)


def test_workflow_start_rejects_string_stage_plan(tmp_path):
    workflow = WorkflowReproducibleGraph(output_dir=tmp_path)
    with pytest.raises(ValueError, match="stage_plan"):
        workflow.start(stage_plan="ingest->generate")  # type: ignore[arg-type]


def test_workflow_start_rejects_explicit_empty_stage_plan(tmp_path):
    workflow = WorkflowReproducibleGraph(output_dir=tmp_path)
    with pytest.raises(ValueError, match="stage_plan must not be empty"):
        workflow.start(stage_plan=[])


def test_workflow_start_rejects_non_string_stage_entries(tmp_path):
    workflow = WorkflowReproducibleGraph(output_dir=tmp_path)
    with pytest.raises(ValueError, match="stage_plan entry"):
        workflow.start(stage_plan=["ingest", 2])  # type: ignore[list-item]


def test_run_manifest_normalizes_and_deduplicates_fallback_methods():
    manifest = RunManifest(
        run_id="run",
        method_key="method",
        data_source_key="source",
        fallback_methods=[" aux ", "aux", "fallback"],
        model_name="m",
        relaxer_name="r",
        evaluator_name="e",
        seed=42,
        deterministic=True,
        stage_plan=["ingest"],
        started_at=1.0,
    )
    assert manifest.fallback_methods == ("aux", "fallback")

    with pytest.raises(ValueError, match="fallback method"):
        RunManifest(
            run_id="run",
            method_key="method",
            data_source_key="source",
            fallback_methods=[True],  # type: ignore[list-item]
            model_name="m",
            relaxer_name="r",
            evaluator_name="e",
            seed=42,
            deterministic=True,
            stage_plan=["ingest"],
            started_at=1.0,
        )


def test_run_manifest_rejects_non_string_model_name():
    with pytest.raises(ValueError, match="model_name"):
        RunManifest(
            run_id="run",
            method_key="method",
            data_source_key="source",
            fallback_methods=["fallback"],
            model_name=1,  # type: ignore[arg-type]
            relaxer_name="r",
            evaluator_name="e",
            seed=42,
            deterministic=True,
            stage_plan=["ingest"],
            started_at=1.0,
        )


def test_run_manifest_accepts_bool_like_deterministic_string():
    manifest = RunManifest(
        run_id="run",
        method_key="method",
        data_source_key="source",
        fallback_methods=["fallback"],
        model_name="m",
        relaxer_name="r",
        evaluator_name="e",
        seed=42,
        deterministic="false",  # type: ignore[arg-type]
        stage_plan=["ingest"],
        started_at=1.0,
    )
    assert manifest.deterministic is False
