"""
CLI entrypoint for Matbench benchmark runs.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Sequence

import torch

from atlas.benchmark.runner import MatbenchRunner


def _load_model(
    module_name: str,
    class_name: str,
    model_kwargs_json: str,
    checkpoint: str | None,
):
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in module '{module_name}'")
    model_cls = getattr(module, class_name)

    kwargs = json.loads(model_kwargs_json or "{}")
    if not isinstance(kwargs, dict):
        raise ValueError("--model-kwargs must decode to a JSON object")

    model = model_cls(**kwargs)

    if checkpoint:
        ckpt_path = Path(checkpoint)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        elif isinstance(payload, dict):
            state_dict = payload
        else:
            raise ValueError(
                "Unsupported checkpoint format. Expected state_dict or dict with model_state_dict."
            )
        model.load_state_dict(state_dict, strict=False)

    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ATLAS model on Matbench task(s).")
    parser.add_argument("--list-tasks", action="store_true", help="List supported Matbench tasks and exit")
    parser.add_argument("--task", type=str, default="", help="Matbench task key")
    parser.add_argument("--property", type=str, default="", help="ATLAS property name in model output dict")
    parser.add_argument("--model-module", type=str, default="", help="Python module containing model class")
    parser.add_argument("--model-class", type=str, default="", help="Model class name")
    parser.add_argument("--model-kwargs", type=str, default="{}", help="JSON kwargs for model constructor")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="cpu/cuda/auto")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel structure conversion workers")
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="Specific folds to run")
    parser.add_argument("--output", type=str, default="", help="Output JSON path")
    parser.add_argument("--output-dir", type=str, default="", help="Report directory if --output omitted")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_tasks:
        for task, prop in MatbenchRunner.TASKS.items():
            print(f"{task}: default_property={prop}")
        return 0

    if not args.task:
        parser.error("--task is required unless --list-tasks is used")
    if not args.model_module or not args.model_class:
        parser.error("--model-module and --model-class are required")

    property_name = args.property or MatbenchRunner.TASKS.get(args.task, "")
    if not property_name:
        parser.error("--property is required when task has no default mapping")

    model = _load_model(
        module_name=args.model_module,
        class_name=args.model_class,
        model_kwargs_json=args.model_kwargs,
        checkpoint=args.checkpoint or None,
    )

    output_dir = Path(args.output_dir) if args.output_dir else None
    runner = MatbenchRunner(
        model=model,
        property_name=property_name,
        device=args.device,
        batch_size=args.batch_size,
        n_jobs=args.jobs,
        output_dir=output_dir,
    )

    save_path = Path(args.output) if args.output else None
    report = runner.run_task(
        task_name=args.task,
        folds=args.folds,
        save_path=save_path,
        show_progress=True,
    )

    print("Task:", report["task_name"])
    print("Property:", report["property_name"])
    print("Model:", report["model_name"])
    print("MAE (mean):", report["aggregate_metrics"].get("mae_mean"))
    print("RMSE (mean):", report["aggregate_metrics"].get("rmse_mean"))
    print("R2 (mean):", report["aggregate_metrics"].get("r2_mean"))
    print("Coverage (mean):", report["aggregate_metrics"].get("coverage_mean"))
    print("Report:", report.get("report_path"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
