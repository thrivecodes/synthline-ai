import numpy as np

from synthline_ai.generation.base import GenerationResult
from synthline_ai.validation.checks import validate_results


def _make_result(h=64, w=64, mask_fill=True, seed=0, invalid_mask_val=False, shape_mismatch=False):
    rng = np.random.RandomState(seed)
    image = rng.randint(100, 200, (h, w, 3), dtype=np.uint8)
    if shape_mismatch:
        mask = np.zeros((h // 2, w // 2), dtype=np.uint8)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)

    if mask_fill:
        mask[10:20, 10:30] = 128 if invalid_mask_val else 255

    return GenerationResult(
        image=image,
        mask=mask,
        source_seed=f"seed_{seed}.png",
        defect_type="scratch",
        random_seed=seed,
    )


def test_all_valid():
    results = [_make_result(seed=i) for i in range(3)]
    stats = validate_results(results)
    assert stats["valid_count"] == 3
    assert stats["invalid_count"] == 0


def test_empty_mask():
    results = [_make_result(mask_fill=False)]
    stats = validate_results(results)
    assert stats["empty_mask_count"] == 1
    assert stats["invalid_count"] == 1


def test_invalid_mask_values():
    results = [_make_result(invalid_mask_val=True)]
    stats = validate_results(results)
    assert stats["invalid_mask_count"] == 1
    assert stats["invalid_count"] == 1


def test_shape_mismatch():
    results = [_make_result(shape_mismatch=True)]
    stats = validate_results(results)
    assert stats["shape_mismatch_count"] == 1
    assert stats["invalid_count"] == 1
