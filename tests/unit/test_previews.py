from pathlib import Path

import cv2
import numpy as np

from synthline_ai.generation.base import GenerationResult
from synthline_ai.validation.previews import create_contact_sheet


def _make_result(h=64, w=64, mask_fill=True, seed=0):
    rng = np.random.RandomState(seed)
    image = rng.randint(100, 200, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    if mask_fill:
        mask[10:20, 10:30] = 255
    return GenerationResult(
        image=image,
        mask=mask,
        source_seed=f"seed_{seed}.png",
        defect_type="scratch",
        random_seed=seed,
    )


def test_contact_sheet_created(tmp_path: Path):
    results = [_make_result(seed=i) for i in range(4)]
    out_path = tmp_path / "sheet.jpg"

    returned_path = create_contact_sheet(results, out_path)

    assert returned_path == out_path
    assert out_path.exists()

    img = cv2.imread(str(out_path))
    assert img is not None
    assert img.shape[2] == 3


def test_contact_sheet_max_samples(tmp_path: Path):
    results = [_make_result(seed=i) for i in range(20)]
    out_path = tmp_path / "sheet_max.jpg"

    create_contact_sheet(results, out_path, max_samples=8, thumb_size=128)

    assert out_path.exists()
    img = cv2.imread(str(out_path))

    # 8 samples -> max 4 cols -> 4 cols.
    # 8 samples -> 2 pairs -> 4 rows.
    # Expected shape: 4 rows * 128 = 512 height. 4 cols * 128 = 512 width.
    assert img.shape[:2] == (512, 512)
