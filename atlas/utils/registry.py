"""
Registry and Factory Pattern implementation for ATLAS.
Allows dynamic registration and instantiation of algorithms via configuration.
"""

from typing import Callable, Dict, Any, Type
import logging

logger = logging.getLogger(__name__)

class Registry:
    """
    A unified registry to store and retrieve classes/functions dynamically.
    """
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: str) -> Callable:
        """
        Decorator to register a class or function.
        Usage:
            @MODELS.register("my_model")
            class MyModel:
                pass
        """
        def inner_wrapper(wrapped_class: Any) -> Any:
            if name in self._registry:
                logger.warning(f"Overwriting existing registration for '{name}' in {self.name} registry.")
            self._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    def get(self, name: str) -> Any:
        """
        Retrieve a class/function by name.
        """
        if name not in self._registry:
            raise KeyError(f"'{name}' not found in {self.name} registry. Available: {list(self._registry.keys())}")
        return self._registry[name]

    def build(self, name: str, **kwargs) -> Any:
        """
        Instantiate a registered class with provided kwargs.
        """
        obj_type = self.get(name)
        return obj_type(**kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

# Global Registries
MODELS = Registry("models")
RELAXERS = Registry("relaxers")
FEATURE_EXTRACTORS = Registry("feature_extractors")
EVALUATORS = Registry("evaluators")

# Factory Helpers
class ModelFactory:
    @staticmethod
    def create(name: str, **kwargs) -> Any:
        return MODELS.build(name, **kwargs)

class RelaxerFactory:
    @staticmethod
    def create(name: str, **kwargs) -> Any:
        return RELAXERS.build(name, **kwargs)

class FeatureExtractorFactory:
    @staticmethod
    def create(name: str, **kwargs) -> Any:
        return FEATURE_EXTRACTORS.build(name, **kwargs)
