"""
Alchemical Discovery Module for ATLAS.

This module integrates the Alchemical-MLIP logic for continuous chemical space exploration.
Ported and optimized from recisic/alchemical-mlip.
"""

try:
    from .model import AlchemicalModel, AlchemyManager
    from .calculator import AlchemicalMACECalculator
    _ALCHEMY_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional heavy dependency path
    AlchemicalModel = None
    AlchemyManager = None
    _ALCHEMY_IMPORT_ERROR = exc

    class AlchemicalMACECalculator:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Alchemical module is unavailable. "
                f"Underlying error: {_ALCHEMY_IMPORT_ERROR}"
            )

__all__ = [
    "AlchemicalModel",
    "AlchemyManager",
    "AlchemicalMACECalculator",
]
