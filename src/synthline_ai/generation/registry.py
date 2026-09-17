from __future__ import annotations

from synthline_ai.generation.base import BaseGenerator
from synthline_ai.generation.procedural.scratch import ScratchGenerator

_REGISTRY: dict[str, type[BaseGenerator]] = {}


def register(name: str, cls: type[BaseGenerator]) -> None:
    """Register a generator class under a given name."""
    _REGISTRY[name] = cls


def get_generator(name: str) -> BaseGenerator:
    """Instantiate and return the generator registered with the given name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown generator: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]()


def available_generators() -> list[str]:
    """Return a list of all registered generator names."""
    return list(_REGISTRY.keys())


# Auto-register default generators
register("scratch", ScratchGenerator)

