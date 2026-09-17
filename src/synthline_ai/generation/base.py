"""Abstract base class and result type for defect generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GenerationResult:
    """Output of a single image generation.

    Attributes:
        image: BGR uint8 array, same shape as input.
        mask: Single-channel uint8 array, 0=background, 255=defect.
        metadata: Generator-specific parameters used for this image.
        source_seed: Filename of the source seed image.
        defect_type: String name of the defect type (e.g. "scratch").
        random_seed: The random seed used for this specific generation.
        split: Optional dataset partition assignment ('train', 'val', 'test').
    """

    image: np.ndarray
    mask: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)
    source_seed: str = ""
    defect_type: str = ""
    random_seed: int = 0
    split: str = "train"


class BaseGenerator(ABC):
    """Interface that all defect generators must implement."""

    @abstractmethod
    def generate(
        self,
        image: np.ndarray,
        seed_name: str,
        random_seed: int,
        severity: float,
        frequency: float = 1.0,
    ) -> GenerationResult:
        """Generate a single defective image from a source image.

        Args:
            image: Source BGR uint8 image.
            seed_name: Filename of the source seed image.
            random_seed: Random seed for reproducibility.
            severity: Defect severity from 0.0 (subtle) to 1.0 (severe).
            frequency: Defect frequency / occurrence density (default: 1.0).

        Returns:
            A GenerationResult containing the modified image, mask, and metadata.
        """
        ...

    @property
    @abstractmethod
    def defect_type(self) -> str:
        """Return the string name of the defect type this generator produces."""
        ...
