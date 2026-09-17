from __future__ import annotations

from synthline_ai.generation.base import BaseGenerator
from synthline_ai.generation.procedural.crack import CrackGenerator
from synthline_ai.generation.procedural.discoloration import DiscolorationGenerator
from synthline_ai.generation.procedural.pinhole import PinholeGenerator
from synthline_ai.generation.procedural.scratch import ScratchGenerator
from synthline_ai.generation.procedural.stain import StainGenerator

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
register("stain", StainGenerator)
register("discoloration", DiscolorationGenerator)
register("crack", CrackGenerator)
register("pinhole", PinholeGenerator)
