"""
CLI entrypoint for Matbench benchmark runs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path

import torch

from atlas.benchmark.runner import MatbenchRunner
from atlas.training.preflight import run_preflight

_SAFE_PROPERTY_GROUP = re.compile(r"^[A-Za-z0-9._-]+$")
_STATE_DICT_CONTAINER_KEYS = ("model_state_dict", "state_dict", "model")


def _normalize_optional_text(value: object) -> str:
    return str(value).strip()


def _coerce_int_with_min(value: object, *, arg_name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{arg_name} must be an integer")
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        number_f = float(value)
        if not number_f.is_integer():
            raise ValueError(f"{arg_name} must be an integer")
        number = int(number_f)
    else:
        try:
            number = int(value)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(f"{arg_name} must be an integer") from exc
    if number < int(minimum):
        comparator = "> 0" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{arg_name} must be {comparator}")
    return number


def _coerce_probability_in_range(
    value: object,
    *,
    arg_name: str,
    lo: float,
    hi: float,
    inclusive_lo: bool = True,
    inclusive_hi: bool = True,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{arg_name} must be a finite float")
    if isinstance(value, Real):
        out = float(value)
    else:
        try:
            out = float(value)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(f"{arg_name} must be a finite float") from exc
    if not (out == out and out not in {float("inf"), float("-inf")}):
        raise ValueError(f"{arg_name} must be finite")

    lo_ok = out >= lo if inclusive_lo else out > lo
    hi_ok = out <= hi if inclusive_hi else out < hi
    if not (lo_ok and hi_ok):
        left = "[" if inclusive_lo else "("
        right = "]" if inclusive_hi else ")"
        raise ValueError(f"{arg_name} must be in {left}{lo}, {hi}{right}")
    return float(out)


def _parse_model_kwargs(model_kwargs_json: str) -> dict:
    payload = model_kwargs_json or "{}"
    try:
        kwargs = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--model-kwargs must be valid JSON object: {exc}") from exc
    if not isinstance(kwargs, dict):
        raise ValueError("--model-kwargs must decode to a JSON object")
    return kwargs


def _looks_like_state_dict(payload: Mapping[object, object]) -> bool:
    if not payload:
        return False
    for key in payload:
        if not isinstance(key, str) or not key:
            return False
    return any(
        "." in key
        or key in {"weight", "bias"}
        or key.endswith(("weight", "bias", "running_mean", "running_var", "num_batches_tracked"))
        for key in payload
    )


def _extract_state_dict(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(
            "Unsupported checkpoint format. Expected state_dict mapping or mapping with model_state_dict/state_dict/model."
        )

    candidate: Mapping[object, object] | None = None
    for container_key in _STATE_DICT_CONTAINER_KEYS:
        nested = payload.get(container_key)
        if isinstance(nested, Mapping) and _looks_like_state_dict(nested):
            candidate = nested
            break
    if candidate is None:
        if not _looks_like_state_dict(payload):
            raise ValueError(
                "Unsupported checkpoint format. Expected state_dict mapping or mapping with model_state_dict/state_dict/model."
            )
        candidate = payload

    normalized: dict[str, object] = {}
    for raw_key, value in candidate.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError("Unsupported checkpoint format. Found non-string state_dict key.")
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if not key:
            raise ValueError("Unsupported checkpoint format. Found empty normalized state_dict key.")
        normalized[key] = value
    return normalized


def _validate_cli_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    def _require(condition: bool, message: str) -> None:
        if not condition:
            parser.error(message)

    args.task = _normalize_optional_text(args.task)
    args.property = _normalize_optional_text(args.property)
    args.model_module = _normalize_optional_text(args.model_module)
    args.model_class = _normalize_optional_text(args.model_class)
    args.checkpoint = _normalize_optional_text(args.checkpoint)
    args.output = _normalize_optional_text(args.output)
    args.output_dir = _normalize_optional_text(args.output_dir)
    args.preflight_property_group = _normalize_optional_text(args.preflight_property_group)

    if args.task:
        _require(args.task in MatbenchRunner.TASKS, f"Unknown --task '{args.task}'")
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        _require(ckpt.exists(), f"--checkpoint not found: {ckpt}")
        _require(ckpt.is_file(), f"--checkpoint must be a file: {ckpt}")
    try:
        args.batch_size = _coerce_int_with_min(args.batch_size, arg_name="--batch-size", minimum=1)
        args.bootstrap_samples = _coerce_int_with_min(
            args.bootstrap_samples,
            arg_name="--bootstrap-samples",
            minimum=0,
        )
        args.bootstrap_seed = _coerce_int_with_min(args.bootstrap_seed, arg_name="--bootstrap-seed", minimum=0)
        args.conformal_max_calibration_samples = _coerce_int_with_min(
            args.conformal_max_calibration_samples,
            arg_name="--conformal-max-calibration-samples",
            minimum=0,
        )
        args.preflight_max_samples = _coerce_int_with_min(
            args.preflight_max_samples,
            arg_name="--preflight-max-samples",
            minimum=0,
        )
        args.preflight_split_seed = _coerce_int_with_min(
            args.preflight_split_seed,
            arg_name="--preflight-split-seed",
            minimum=0,
        )
        args.preflight_timeout_sec = _coerce_int_with_min(
            args.preflight_timeout_sec,
            arg_name="--preflight-timeout-sec",
            minimum=1,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        args.min_coverage_required = _coerce_probability_in_range(
            args.min_coverage_required,
            arg_name="--min-coverage-required",
            lo=0.0,
            hi=1.0,
            inclusive_lo=True,
            inclusive_hi=True,
        )
        args.conformal_coverage = _coerce_probability_in_range(
            args.conformal_coverage,
            arg_name="--conformal-coverage",
            lo=0.0,
            hi=1.0,
            inclusive_lo=False,
            inclusive_hi=False,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        jobs = _coerce_int_with_min(args.jobs, arg_name="--jobs", minimum=-10**9)
    except ValueError as exc:
        parser.error(str(exc))
    _require(int(jobs) != 0, "--jobs must be non-zero (-1 means all cores)")
    args.jobs = int(jobs)
    if not args.skip_preflight:
        property_group = args.preflight_property_group
        _require(bool(property_group), "--preflight-property-group must not be empty")
        _require(
            _SAFE_PROPERTY_GROUP.fullmatch(property_group) is not None,
            "--preflight-property-group contains unsupported characters",
        )
    if args.folds is not None:
        _require(len(args.folds) > 0, "--folds requires at least one fold index")
        try:
            normalized_folds = [
                _coerce_int_with_min(v, arg_name="--folds entries", minimum=0)
                for v in args.folds
            ]
        except ValueError as exc:
            parser.error(str(exc))
        args.folds = sorted(set(normalized_folds))
    if args.output and args.output_dir:
        _require(False, "--output and --output-dir cannot be used together")
    if args.skip_preflight and not args.dry_run:
        parser.error("--skip-preflight is only allowed together with --dry-run.")
    if args.preflight_only and args.skip_preflight:
        parser.error("--preflight-only cannot be used with --skip-preflight.")


def _load_model(
    module_name: str,
    class_name: str,
    model_kwargs_json: str,
    checkpoint: str | None,
    strict_checkpoint: bool = False,
):
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in module '{module_name}'")
    model_cls = getattr(module, class_name)

    kwargs = _parse_model_kwargs(model_kwargs_json)

    model = model_cls(**kwargs)

    load_info = {
        "checkpoint_path": str(checkpoint) if checkpoint else "",
        "missing_keys": [],
        "unexpected_keys": [],
    }

    if checkpoint:
        ckpt_path = Path(checkpoint)
        try:
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            # Compatibility fallback for older torch releases.
            payload = torch.load(ckpt_path, map_location="cpu")
        state_dict = _extract_state_dict(payload)
        incompatible = model.load_state_dict(state_dict, strict=bool(strict_checkpoint))
        if hasattr(incompatible, "missing_keys"):
            load_info["missing_keys"] = list(incompatible.missing_keys)
        if hasattr(incompatible, "unexpected_keys"):
            load_info["unexpected_keys"] = list(incompatible.unexpected_keys)

    return model, load_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ATLAS model on Matbench task(s).")
    parser.add_argument("--list-tasks", action="store_true", help="List supported Matbench tasks and exit")
    parser.add_argument("--task", type=str, default="", help="Matbench task key")
    parser.add_argument("--property", type=str, default="", help="ATLAS property name in model output dict")
    parser.add_argument("--model-module", type=str, default="", help="Python module containing model class")
    parser.add_argument("--model-class", type=str, default="", help="Model class name")
    parser.add_argument("--model-kwargs", type=str, default="{}", help="JSON kwargs for model constructor")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path")
    parser.add_argument(
        "--strict-checkpoint",
        action="store_true",
        help="Require exact checkpoint key match (recommended for benchmark fairness).",
    )
    parser.add_argument("--device", type=str, default="auto", help="cpu/cuda/auto")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel structure conversion workers")
    parser.add_argument("--bootstrap-samples", type=int, default=400, help="Bootstrap samples for global MAE CI")
    parser.add_argument("--bootstrap-seed", type=int, default=42, help="Bootstrap RNG seed")
    parser.add_argument(
        "--min-coverage-required",
        type=float,
        default=0.0,
        help="Mark fold as low_coverage when valid structure ratio is below this threshold.",
    )
    parser.add_argument(
        "--use-conformal",
        action="store_true",
        help="Enable split-conformal diagnostics using fold train/val residuals.",
    )
    parser.add_argument(
        "--conformal-coverage",
        type=float,
        default=0.95,
        help="Target coverage for conformal intervals.",
    )
    parser.add_argument(
        "--conformal-max-calibration-samples",
        type=int,
        default=0,
        help="Cap calibration samples per fold (0 means full set).",
    )
    parser.add_argument(
        "--no-structure-cache",
        action="store_true",
        help="Disable structure-to-graph conversion cache during this run.",
    )
    parser.add_argument(
        "--fail-on-fallback-signature",
        action="store_true",
        help="Fail when model forward requires fallback call signature.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="Specific folds to run")
    parser.add_argument("--output", type=str, default="", help="Output JSON path")
    parser.add_argument("--output-dir", type=str, default="", help="Report directory if --output omitted")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-property-group", type=str, default="priority7")
    parser.add_argument("--preflight-max-samples", type=int, default=0)
    parser.add_argument("--preflight-split-seed", type=int, default=42)
    parser.add_argument("--preflight-timeout-sec", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_tasks:
        for task, prop in MatbenchRunner.TASKS.items():
            print(f"{task}: default_property={prop}")
        return 0
    _validate_cli_args(parser, args)

    if not args.skip_preflight:
        project_root = Path(__file__).resolve().parents[2]
        preflight = run_preflight(
            project_root=project_root,
            property_group=args.preflight_property_group,
            max_samples=args.preflight_max_samples,
            split_seed=args.preflight_split_seed,
            dry_run=args.dry_run,
            timeout_sec=args.preflight_timeout_sec,
        )
        if preflight.return_code != 0:
            detail = f" ({preflight.error_message})" if preflight.error_message else ""
            print(f"[ERROR] Preflight failed with return code {preflight.return_code}{detail}", file=sys.stderr)
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

    model, load_info = _load_model(
        module_name=args.model_module,
        class_name=args.model_class,
        model_kwargs_json=args.model_kwargs,
        checkpoint=args.checkpoint or None,
        strict_checkpoint=args.strict_checkpoint,
    )
    if load_info["missing_keys"] or load_info["unexpected_keys"]:
        print(
            "[Benchmark] Checkpoint compatibility:",
            f"missing={len(load_info['missing_keys'])}, unexpected={len(load_info['unexpected_keys'])}",
        )
        if args.strict_checkpoint:
            print("[Benchmark] strict checkpoint mode enabled.")

    output_dir = Path(args.output_dir) if args.output_dir else None
    runner = MatbenchRunner(
        model=model,
        property_name=property_name,
        device=args.device,
        batch_size=args.batch_size,
        n_jobs=args.jobs,
        output_dir=output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        min_coverage_required=args.min_coverage_required,
        use_conformal=args.use_conformal,
        conformal_coverage=args.conformal_coverage,
        conformal_max_calibration_samples=args.conformal_max_calibration_samples,
        structure_cache=not args.no_structure_cache,
        fail_on_fallback_signature=args.fail_on_fallback_signature,
    )

    save_path = Path(args.output) if args.output else None
    report = runner.run_task(
        task_name=args.task,
        folds=args.folds,
        save_path=save_path,
        show_progress=not args.no_progress,
    )

    aggregate = report["aggregate_metrics"]
    print("Task:", report["task_name"])
    print("Property:", report["property_name"])
    print("Model:", report["model_name"])
    print("Global MAE:", aggregate.get("global_mae"))
    print("Global MAE CI95:", (aggregate.get("global_mae_ci95_lo"), aggregate.get("global_mae_ci95_hi")))
    print("Global RMSE:", aggregate.get("global_rmse"))
    print("Global R2:", aggregate.get("global_r2"))
    print("Success rate:", aggregate.get("success_rate"))
    print("Fold MAE (weighted mean):", aggregate.get("mae_weighted_mean"))
    print("Fold RMSE (weighted mean):", aggregate.get("rmse_weighted_mean"))
    print("Fold Coverage (weighted mean):", aggregate.get("coverage_weighted_mean"))
    print("PI95 Coverage (global):", aggregate.get("global_pi95_coverage"))
    print("Gaussian NLL (global):", aggregate.get("global_gaussian_nll"))
    print("CRPS (global):", aggregate.get("global_crps_gaussian"))
    print("Regression ECE (global):", aggregate.get("global_regression_ece"))
    print("Conformal PI coverage (weighted):", aggregate.get("conformal_pi_coverage_weighted_mean"))
    print("Report:", report.get("report_path"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
