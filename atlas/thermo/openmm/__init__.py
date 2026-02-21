try:
    from .engine import OpenMMEngine
except Exception as exc:  # pragma: no cover - optional dependency
    _OPENMM_IMPORT_ERROR = str(exc)

    class OpenMMEngine:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "OpenMMEngine is unavailable because openmm/openmm-ml is not installed. "
                f"Underlying error: {_OPENMM_IMPORT_ERROR}"
            )
