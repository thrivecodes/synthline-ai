"""Generation pipeline orchestration with seed-aware partition splitting."""

from __future__ import annotations

import numpy as np

from synthline_ai.config.models import DatasetSplit, GenerationConfig, ImageInfo
from synthline_ai.generation.base import GenerationResult
from synthline_ai.generation.randomization import (
    apply_geometry_variation,
    apply_lighting_variation,
    apply_texture_variation,
)
from synthline_ai.generation.registry import get_generator


def partition_seeds(
    image_infos: list[ImageInfo],
    config: GenerationConfig,
) -> dict[str, list[int]]:
    """Partition seed indices into train/val/test splits to prevent data leakage.

    Splits source seeds first, so all generated variants of a seed stay strictly
    within the same dataset split.
    """
    num_seeds = len(image_infos)
    if not config.enable_split or num_seeds == 0:
        return {"train": list(range(num_seeds)), "val": [], "test": []}

    rng = np.random.RandomState(config.random_seed)
    indices = list(range(num_seeds))
    rng.shuffle(indices)

    ratios = config.split_ratio
    total_ratio = ratios.train + ratios.val + ratios.test
    if total_ratio <= 0:
        norm_train, norm_val, norm_test = 0.7, 0.2, 0.1
    else:
        norm_train = ratios.train / total_ratio
        norm_val = ratios.val / total_ratio
        norm_test = ratios.test / total_ratio

    train_end = max(1, int(round(num_seeds * norm_train)))
    val_end = train_end + int(round(num_seeds * norm_val))

    # Boundary safety
    if num_seeds >= 3:
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        # Ensure at least 1 in val and test if ratios requested and seeds >= 3
        if not val_indices and norm_val > 0:
            val_indices = [train_indices.pop()]
        if not test_indices and norm_test > 0 and train_indices:
            test_indices = [train_indices.pop()]
    elif num_seeds == 2:
        train_indices = [indices[0]]
        val_indices = [indices[1]]
        test_indices = []
    else:
        train_indices = indices
        val_indices = []
        test_indices = []

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def _apply_randomization(
    result: GenerationResult,
    config: GenerationConfig,
    seed: int,
) -> GenerationResult:
    """Apply optional surface lighting, texture, and geometry variations."""
    if not config.enable_variations:
        return result

    rng = np.random.RandomState(seed)
    img = result.image
    mask = result.mask

    if config.geometry_intensity > 0.0:
        max_rot = 3.0 * config.geometry_intensity
        max_shift = int(4 * config.geometry_intensity) or 1
        img, mask = apply_geometry_variation(
            img,
            mask,
            rng,
            max_rotation_deg=max_rot,
            max_shift_px=max_shift,
        )

    if config.lighting_intensity > 0.0:
        img = apply_lighting_variation(img, rng, intensity=config.lighting_intensity)

    if config.texture_intensity > 0.0:
        img = apply_texture_variation(img, rng, intensity=config.texture_intensity)

    result.image = img
    result.mask = mask
    return result


def run_generation(
    config: GenerationConfig,
    images: list[np.ndarray],
    image_infos: list[ImageInfo],
) -> list[GenerationResult]:
    """Orchestrate the generation loop with optional leak-free dataset splitting."""
    results: list[GenerationResult] = []

    if not images or not image_infos:
        return results

    generator = get_generator(config.defect_type.value)

    if not config.enable_split:
        # Standard generation without splits
        for i in range(config.count):
            image = images[i % len(images)]
            info = image_infos[i % len(image_infos)]
            per_image_seed = config.random_seed + i

            result = generator.generate(
                image=image,
                seed_name=info.path.name,
                random_seed=per_image_seed,
                severity=config.severity,
                frequency=config.frequency,
            )
            result = _apply_randomization(result, config, per_image_seed)
            result.split = DatasetSplit.TRAIN.value
            results.append(result)
        return results

    # Partition seeds by split first
    seed_splits = partition_seeds(image_infos, config)
    active_splits = [
        (split_name, s_indices)
        for split_name, s_indices in seed_splits.items()
        if len(s_indices) > 0
    ]

    ratios = config.split_ratio
    split_weights = {
        "train": ratios.train,
        "val": ratios.val,
        "test": ratios.test,
    }

    # Allocate target sample counts across active splits
    total_active_weight = sum(split_weights[s] for s, _ in active_splits) or 1.0
    split_counts: dict[str, int] = {}
    allocated = 0
    for idx, (s_name, _) in enumerate(active_splits):
        if idx == len(active_splits) - 1:
            split_counts[s_name] = max(0, config.count - allocated)
        else:
            w = split_weights[s_name] / total_active_weight
            cnt = int(round(config.count * w))
            split_counts[s_name] = cnt
            allocated += cnt

    # Generate samples per split using only that split's seeds
    sample_idx = 0
    for s_name, s_indices in active_splits:
        target_count = split_counts[s_name]
        for i in range(target_count):
            seed_idx = s_indices[i % len(s_indices)]
            image = images[seed_idx]
            info = image_infos[seed_idx]
            per_image_seed = config.random_seed + sample_idx

            result = generator.generate(
                image=image,
                seed_name=info.path.name,
                random_seed=per_image_seed,
                severity=config.severity,
                frequency=config.frequency,
            )
            result = _apply_randomization(result, config, per_image_seed)
            result.split = s_name
            results.append(result)
            sample_idx += 1

    return results
