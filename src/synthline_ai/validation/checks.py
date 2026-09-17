from __future__ import annotations

from synthline_ai.generation.base import GenerationResult
from synthline_ai.labeling.boxes import mask_to_bbox
from synthline_ai.labeling.masks import mask_area, validate_mask


def validate_results(results: list[GenerationResult]) -> dict[str, object]:
    """Post-generation validation.

    Checks each result for:
    - empty masks (area == 0)
    - invalid mask values (not just 0 and 255)
    - invalid bounding boxes (width or height == 0 when mask is non-empty)
    - image shape mismatch (image H,W doesn't match mask H,W)

    Returns:
        {
            'total': int,
            'valid_count': int,
            'invalid_count': int,
            'empty_mask_count': int,
            'invalid_mask_count': int,
            'invalid_bbox_count': int,
            'shape_mismatch_count': int,
            'issues': [{'index': int, 'codes': [str], 'details': str}, ...]
        }
    """
    stats = {
        "total": len(results),
        "valid_count": 0,
        "invalid_count": 0,
        "empty_mask_count": 0,
        "invalid_mask_count": 0,
        "invalid_bbox_count": 0,
        "shape_mismatch_count": 0,
        "issues": [],
    }

    for i, res in enumerate(results):
        codes = []
        details = []

        # Check shape mismatch
        img_h, img_w = res.image.shape[:2]
        mask_h, mask_w = res.mask.shape[:2]
        if img_h != mask_h or img_w != mask_w:
            codes.append("shape_mismatch")
            details.append(
                f"Image shape {res.image.shape} does not match mask shape {res.mask.shape}"
            )
            stats["shape_mismatch_count"] = int(stats["shape_mismatch_count"]) + 1  # type: ignore

        # Check invalid mask values
        if not validate_mask(res.mask):
            codes.append("invalid_mask")
            details.append("Mask contains values other than 0 and 255")
            stats["invalid_mask_count"] = int(stats["invalid_mask_count"]) + 1  # type: ignore

        # Check empty mask
        area = mask_area(res.mask)
        if area == 0:
            codes.append("empty_mask")
            details.append("Mask area is 0")
            stats["empty_mask_count"] = int(stats["empty_mask_count"]) + 1  # type: ignore
        else:
            # Check bounding box
            x, y, w, h = mask_to_bbox(res.mask)
            if w == 0 or h == 0:
                codes.append("invalid_bbox")
                details.append("Bounding box has 0 width or height")
                stats["invalid_bbox_count"] = int(stats["invalid_bbox_count"]) + 1  # type: ignore

        if codes:
            stats["invalid_count"] = int(stats["invalid_count"]) + 1  # type: ignore
            issues = stats["issues"]
            assert isinstance(issues, list)
            issues.append({"index": i, "codes": codes, "details": "; ".join(details)})
        else:
            stats["valid_count"] = int(stats["valid_count"]) + 1  # type: ignore

    return stats


def check_output_duplicates(
    results: list[GenerationResult],
    threshold: int = 4,
) -> dict[str, object]:
    """Check generated output images for duplicates using dHash."""
    from synthline_ai.ingestion.deduplication import compute_dhash, hamming_distance

    hashes = [compute_dhash(res.image) for res in results]
    duplicates = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            dist = hamming_distance(hashes[i], hashes[j])
            if dist <= threshold:
                duplicates.append({"idx_a": i, "idx_b": j, "distance": dist})

    return {
        "total": len(results),
        "duplicate_pairs": len(duplicates),
        "duplicates": duplicates,
    }


def check_split_leakage(
    results: list[GenerationResult],
) -> dict[str, object]:
    """Verify no source seed appears in multiple dataset splits."""
    seed_to_splits: dict[str, set[str]] = {}
    seeds_per_split = {"train": 0, "val": 0, "test": 0}

    for res in results:
        split = res.split
        seed = res.source_seed
        if split in seeds_per_split:
            seeds_per_split[split] += 1
        elif split:
            seeds_per_split[split] = 1

        if seed not in seed_to_splits:
            seed_to_splits[seed] = set()
        seed_to_splits[seed].add(split)

    leaked_seeds = []
    for seed, splits in seed_to_splits.items():
        if len(splits) > 1:
            leaked_seeds.append({"seed": seed, "splits": sorted(list(splits))})

    return {
        "has_leakage": len(leaked_seeds) > 0,
        "leaked_seeds": leaked_seeds,
        "seeds_per_split": seeds_per_split,
    }


def check_brightness_distribution(
    results: list[GenerationResult],
) -> dict[str, object]:
    """Analyze brightness and contrast distribution of generated images."""
    import numpy as np

    if not results:
        return {
            "brightness": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
            "contrast": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
            "outlier_count": 0,
            "outliers": [],
        }

    means = []
    stds = []
    for res in results:
        means.append(float(np.mean(res.image)))
        stds.append(float(np.std(res.image)))

    b_mean = float(np.mean(means))
    b_std = float(np.std(means))
    c_mean = float(np.mean(stds))
    c_std = float(np.std(stds))

    outliers = []
    for i, (b, c) in enumerate(zip(means, stds, strict=False)):
        reasons = []
        if b_std > 0 and abs(b - b_mean) > 2 * b_std:
            reasons.append("brightness")
        if c_std > 0 and abs(c - c_mean) > 2 * c_std:
            reasons.append("contrast")

        if reasons:
            outliers.append(
                {
                    "index": i,
                    "brightness": b,
                    "contrast": c,
                    "reason": " and ".join(reasons) + " outlier",
                }
            )

    return {
        "brightness": {
            "min": float(np.min(means)),
            "max": float(np.max(means)),
            "mean": b_mean,
            "std": b_std,
        },
        "contrast": {
            "min": float(np.min(stds)),
            "max": float(np.max(stds)),
            "mean": c_mean,
            "std": c_std,
        },
        "outlier_count": len(outliers),
        "outliers": outliers,
    }
