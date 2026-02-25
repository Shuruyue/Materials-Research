"""Tests for reproducible workflow scaffolding."""

import json

from atlas.research.workflow_reproducible_graph import IterationSnapshot, WorkflowReproducibleGraph


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
