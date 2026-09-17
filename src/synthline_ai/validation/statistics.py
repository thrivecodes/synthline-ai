from __future__ import annotations

import numpy as np

from synthline_ai.generation.base import GenerationResult
from synthline_ai.labeling.masks import mask_area


def compute_statistics(results: list[GenerationResult]) -> dict[str, object]:
    """Compute dataset-level statistics.

    Returns:
        {
            'total_images': int,
            'total_annotations': int (same as total_images for Phase 0, 1 defect per image),
            'defect_types': {type_name: count},
            'mask_area_stats': {
                'min': float (% of image area),
                'max': float,
                'mean': float,
                'median': float,
            },
            'generation_success_rate': float (% with non-empty mask),
            'empty_mask_count': int,
            'brightness_stats': {
                'min': float (mean brightness),
                'max': float,
                'mean': float,
            },
        }
    """
    total_images = len(results)

    if total_images == 0:
        return {
            "total_images": 0,
            "total_annotations": 0,
            "defect_types": {},
            "mask_area_stats": {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0},
            "generation_success_rate": 0.0,
            "empty_mask_count": 0,
            "brightness_stats": {"min": 0.0, "max": 0.0, "mean": 0.0},
        }

    defect_types: dict[str, int] = {}
    areas = []
    brights = []
    empty_mask_count = 0

    for res in results:
        defect_types[res.defect_type] = defect_types.get(res.defect_type, 0) + 1

        area = mask_area(res.mask)
        img_h, img_w = res.image.shape[:2]
        total_pixels = img_h * img_w
        if total_pixels > 0:
            areas.append((area / total_pixels) * 100.0)
        else:
            areas.append(0.0)

        if area == 0:
            empty_mask_count += 1

        brights.append(np.mean(res.image))

    return {
        "total_images": total_images,
        "total_annotations": total_images,
        "defect_types": defect_types,
        "mask_area_stats": {
            "min": float(np.min(areas)),
            "max": float(np.max(areas)),
            "mean": float(np.mean(areas)),
            "median": float(np.median(areas)),
        },
        "generation_success_rate": (total_images - empty_mask_count) / total_images * 100.0,
        "empty_mask_count": empty_mask_count,
        "brightness_stats": {
            "min": float(np.min(brights)),
            "max": float(np.max(brights)),
            "mean": float(np.mean(brights)),
        },
    }


def compute_split_statistics(
    results: list[GenerationResult],
) -> dict[str, object]:
    """Compute per-split statistics for datasets with train/val/test partitioning."""
    from synthline_ai.labeling.masks import mask_area

    splits_data: dict[str, list[GenerationResult]] = {}
    for res in results:
        split = res.split
        if split not in splits_data:
            splits_data[split] = []
        splits_data[split].append(res)

    splits_stats = {}
    for split, split_results in splits_data.items():
        count = len(split_results)
        areas = []
        brights = []
        defect_types: dict[str, int] = {}
        for res in split_results:
            defect_types[res.defect_type] = defect_types.get(res.defect_type, 0) + 1
            area = mask_area(res.mask)
            img_h, img_w = res.image.shape[:2]
            total_pixels = img_h * img_w
            if total_pixels > 0:
                areas.append((area / total_pixels) * 100.0)
            else:
                areas.append(0.0)
            brights.append(float(np.mean(res.image)))

        splits_stats[split] = {
            "count": count,
            "mean_mask_area_pct": float(np.mean(areas)) if areas else 0.0,
            "mean_brightness": float(np.mean(brights)) if brights else 0.0,
            "defect_types": defect_types,
        }

    total_images = len(results)
    is_balanced = True
    if total_images > 0:
        for _split, stat in splits_stats.items():
            count_val = stat.get("count", 0)
            split_count = count_val if isinstance(count_val, int) else 0
            if split_count > 0 and (split_count / total_images) < 0.1:
                is_balanced = False

    return {
        "splits": splits_stats,
        "total": total_images,
        "is_balanced": is_balanced,
    }
