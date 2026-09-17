from __future__ import annotations

import numpy as np

from synthline_ai.config.models import GenerationConfig, ImageInfo
from synthline_ai.generation.base import GenerationResult
from synthline_ai.generation.registry import get_generator


def run_generation(
    config: GenerationConfig,
    images: list[np.ndarray],
    image_infos: list[ImageInfo],
) -> list[GenerationResult]:
    """
    Orchestrate the generation loop.

    Args:
        config: Generation configuration.
        images: List of source images.
        image_infos: List of metadata for the source images.

    Returns:
        List of generated results.
    """
    results: list[GenerationResult] = []

    if not images or not image_infos:
        return results

    generator = get_generator(config.defect_type.value)

    for i in range(config.count):
        image = images[i % len(images)]
        info = image_infos[i % len(image_infos)]

        per_image_seed = config.random_seed + i

        result = generator.generate(
            image=image,
            seed_name=info.path.name,
            random_seed=per_image_seed,
            severity=config.severity,
        )
        results.append(result)

    return results
