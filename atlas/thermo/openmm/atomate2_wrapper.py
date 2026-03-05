from __future__ import annotations

import importlib
import logging
import math
from numbers import Integral, Real
from typing import Any

from atlas.utils.registry import RELAXERS

logger = logging.getLogger(__name__)

_VALID_ENSEMBLES = {"nvt", "npt", "minimize"}
_ATOMATE2_JOBS_MODULE: Any | None = None


def _is_boolean_like(value: Any) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}


def _coerce_non_negative_int(value: Any, name: str) -> int:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be an integer, got boolean {value!r}")
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        number_f = float(value)
        if not math.isfinite(number_f) or not number_f.is_integer():
            raise ValueError(f"{name} must be an integer, got {value!r}")
        number = int(number_f)
    else:
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if number < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return number


def _load_atomate2_jobs_module() -> Any:
    global _ATOMATE2_JOBS_MODULE
    if _ATOMATE2_JOBS_MODULE is not None:
        return _ATOMATE2_JOBS_MODULE
    try:
        _ATOMATE2_JOBS_MODULE = importlib.import_module(
            "atlas.third_party.atomate2.openmm.jobs.core"
        )
        return _ATOMATE2_JOBS_MODULE
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "Assimilated Atomate2 OpenMM jobs are unavailable. "
            "Ensure optional OpenMM/atomate2 dependencies are installed."
        ) from exc


@RELAXERS.register("atomate2_native")
class NativeAtomate2OpenMMEngine:
    """
    Native Atomate2 OpenMM wrapper using job makers directly.

    Supports `nvt`, `npt`, and `minimize` execution modes.
    """

    def __init__(
        self,
        temperature: float = 300.0,
        step_size: float = 1.0,
        ensemble: str = "nvt",
    ):
        if not math.isfinite(float(temperature)) or float(temperature) <= 0:
            raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
        if not math.isfinite(float(step_size)) or float(step_size) <= 0:
            raise ValueError(f"step_size must be finite and > 0, got {step_size!r}")

        mode = str(ensemble).strip().lower()
        if mode not in _VALID_ENSEMBLES:
            raise ValueError(f"ensemble must be one of {sorted(_VALID_ENSEMBLES)}, got {ensemble!r}")

        self.temperature = float(temperature)
        self.step_size = float(step_size)
        self.ensemble = mode

    def _build_maker(self, atomate2_jobs: Any, steps: int):
        n_steps = _coerce_non_negative_int(steps, "steps")

        if self.ensemble == "npt":
            if n_steps == 0:
                raise ValueError("steps must be > 0 for npt ensemble")
            return atomate2_jobs.NPTMaker(
                temperature=self.temperature,
                step_size=self.step_size,
                n_steps=n_steps,
            )
        if self.ensemble == "nvt":
            if n_steps == 0:
                raise ValueError("steps must be > 0 for nvt ensemble")
            return atomate2_jobs.NVTMaker(
                temperature=self.temperature,
                step_size=self.step_size,
                n_steps=n_steps,
            )
        return atomate2_jobs.EnergyMinimizationMaker()

    def run_simulation(self, interchange_data: Any, steps: int = 1000) -> Any:
        """Run simulation through native Atomate2 OpenMM makers."""
        if interchange_data is None:
            raise ValueError("interchange_data must not be None")

        atomate2_jobs = _load_atomate2_jobs_module()

        try:
            maker = self._build_maker(atomate2_jobs, steps)
            make_fn = getattr(maker, "make", None)
            if not callable(make_fn):
                raise TypeError(f"Maker {maker.__class__.__name__} does not define callable make(...)")
            logger.info("Instantiated native Atomate2 maker: %s", maker.__class__.__name__)
            return make_fn(interchange_data)
        except Exception as exc:
            logger.error("Native Atomate2 OpenMM simulation failed: %s", exc)
            raise
