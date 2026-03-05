"""Registry and factory helpers for dynamic component instantiation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_registry_key(name: str, *, field_name: str = "name") -> str:
    """Normalize and validate a registry key."""
    if not isinstance(name, str):
        raise TypeError(f"{field_name} must be a string, got {type(name).__name__}")
    normalized = name.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


class Registry:
    """Store and resolve classes/functions by string key."""

    def __init__(self, name: str):
        self.name = _normalize_registry_key(name, field_name="registry name")
        self._registry: dict[str, Any] = {}

    def register(self, name: str, *, replace: bool = False) -> Callable[[Any], Any]:
        """
        Decorator to register a class or function.

        Example:
            @MODELS.register("my_model")
            class MyModel:
                ...
        """
        key = _normalize_registry_key(name)
        if not isinstance(replace, bool):
            raise TypeError("replace must be a boolean")

        def inner_wrapper(wrapped_class: Any) -> Any:
            if not callable(wrapped_class):
                raise TypeError(
                    f"Registered object for {key!r} in {self.name} registry must be callable"
                )
            if key in self._registry:
                if not replace:
                    raise ValueError(
                        f"{key!r} is already registered in {self.name} registry. "
                        "Use replace=True to overwrite."
                    )
                logger.info("Replacing existing registration for %r in %s registry.", key, self.name)
            self._registry[key] = wrapped_class
            return wrapped_class

        return inner_wrapper

    def get(self, name: str) -> Any:
        """Retrieve a registered class/function by name."""
        key = _normalize_registry_key(name)
        if key not in self._registry:
            available = ", ".join(sorted(self._registry)) or "<empty>"
            raise KeyError(f"{key!r} not found in {self.name} registry. Available: {available}")
        return self._registry[key]

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Instantiate a registered class/function with provided args/kwargs."""
        obj_type = self.get(name)
        return obj_type(*args, **kwargs)

    def registered_names(self) -> tuple[str, ...]:
        """Return sorted names for deterministic diagnostics."""
        return tuple(sorted(self._registry))

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            normalized = _normalize_registry_key(name)
        except ValueError:
            return False
        return normalized in self._registry


# Global Registries
MODELS = Registry("models")
RELAXERS = Registry("relaxers")
FEATURE_EXTRACTORS = Registry("feature_extractors")
EVALUATORS = Registry("evaluators")


# Factory Helpers
class ModelFactory:
    @staticmethod
    def create(name: str, **kwargs: Any) -> Any:
        return MODELS.build(name, **kwargs)


class RelaxerFactory:
    @staticmethod
    def create(name: str, **kwargs: Any) -> Any:
        return RELAXERS.build(name, **kwargs)


class FeatureExtractorFactory:
    @staticmethod
    def create(name: str, **kwargs: Any) -> Any:
        return FEATURE_EXTRACTORS.build(name, **kwargs)


class EvaluatorFactory:
    @staticmethod
    def create(name: str, **kwargs: Any) -> Any:
        return EVALUATORS.build(name, **kwargs)
