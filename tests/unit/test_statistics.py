import numpy as np

from synthline_ai.generation.base import GenerationResult
from synthline_ai.validation.statistics import compute_statistics


def _make_result(h=64, w=64, mask_fill=True, seed=0):
    rng = np.random.RandomState(seed)
    image = rng.randint(100, 200, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    if mask_fill:
        mask[10:20, 10:30] = 255  # area is 10 * 20 = 200
    return GenerationResult(
        image=image,
        mask=mask,
        source_seed=f"seed_{seed}.png",
        defect_type="scratch",
        random_seed=seed,
    )


def test_basic_stats():
    results = [_make_result(seed=i) for i in range(5)]
    stats = compute_statistics(results)

    assert stats["total_images"] == 5
    assert stats["defect_types"] == {"scratch": 5}
    assert stats["generation_success_rate"] == 100.0

    # 200 pixels out of 64x64=4096 is ~4.88%
    area_pct = (200 / 4096) * 100.0
    mask_stats = stats["mask_area_stats"]
    assert np.isclose(mask_stats["min"], area_pct)
    assert np.isclose(mask_stats["max"], area_pct)


def test_empty_results():
    stats = compute_statistics([])
    assert stats["total_images"] == 0
    assert stats["generation_success_rate"] == 0.0
