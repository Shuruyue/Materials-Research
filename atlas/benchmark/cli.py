"""
CLI entrypoint for Matbench benchmark runs.
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from atlas.benchmark.runner import MatbenchRunner
from atlas.training.preflight import run_preflight


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
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-property-group", type=str, default="priority7")
    parser.add_argument("--preflight-max-samples", type=int, default=0)
    parser.add_argument("--preflight-split-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_tasks:
        for task, prop in MatbenchRunner.TASKS.items():
            print(f"{task}: default_property={prop}")
        return 0

    if args.skip_preflight and not args.dry_run:
        print("[ERROR] --skip-preflight is only allowed together with --dry-run.")
        return 2
    if not args.skip_preflight:
        project_root = Path(__file__).resolve().parents[2]
        preflight = run_preflight(
            project_root=project_root,
            property_group=args.preflight_property_group,
            max_samples=args.preflight_max_samples,
            split_seed=args.preflight_split_seed,
            dry_run=args.dry_run,
        )
        if preflight.return_code != 0:
            print(f"[ERROR] Preflight failed with return code {preflight.return_code}")
            return preflight.return_code
    if args.preflight_only:
        print("[Benchmark] Preflight-only mode completed.")
        return 0
    if args.dry_run:
        print("[Benchmark] Dry run complete.")
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
    print("PI95 Coverage (mean):", report["aggregate_metrics"].get("pi95_coverage_mean"))
    print("Gaussian NLL (mean):", report["aggregate_metrics"].get("gaussian_nll_mean"))
    print("Report:", report.get("report_path"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
