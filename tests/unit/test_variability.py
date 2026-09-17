from __future__ import annotations

import cv2
import numpy as np

from synthline_ai.generation.procedural.variability import apply_defect_variability


def create_test_data() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[22:42, 22:42] = 255
    return image, mask


def test_variability_preserves_shape() -> None:
    image, mask = create_test_data()
    rng = np.random.RandomState(42)

    out_img, out_mask = apply_defect_variability(
        image, mask, rng, color_shift=0.3, opacity_jitter=0.2, edge_roughness=0.3
    )

    assert out_img.shape == image.shape
    assert out_mask.shape == mask.shape


def test_variability_preserves_dtype() -> None:
    image, mask = create_test_data()
    rng = np.random.RandomState(42)

    out_img, out_mask = apply_defect_variability(
        image, mask, rng, color_shift=0.3, opacity_jitter=0.2, edge_roughness=0.3
    )

    assert out_img.dtype == np.uint8
    assert out_mask.dtype == np.uint8


def test_variability_mask_binary() -> None:
    image, mask = create_test_data()
    rng = np.random.RandomState(42)

    out_img, out_mask = apply_defect_variability(
        image, mask, rng, color_shift=0.3, opacity_jitter=0.2, edge_roughness=0.3
    )

    unique_vals = np.unique(out_mask)
    assert set(unique_vals).issubset({0, 255})


def test_variability_preserves_background() -> None:
    image, mask = create_test_data()
    rng = np.random.RandomState(42)

    out_img, out_mask = apply_defect_variability(
        image, mask, rng, color_shift=0.3, opacity_jitter=0.2, edge_roughness=0.3
    )

    bg_mask = (mask == 0) & (out_mask == 0)
    assert np.array_equal(out_img[bg_mask], image[bg_mask])


def test_variability_different_seeds() -> None:
    image, mask = create_test_data()

    rng1 = np.random.RandomState(42)
    out_img1, out_mask1 = apply_defect_variability(
        image, mask, rng1, color_shift=0.9, opacity_jitter=0.9, edge_roughness=0.9
    )

    rng2 = np.random.RandomState(99)
    out_img2, out_mask2 = apply_defect_variability(
        image, mask, rng2, color_shift=0.9, opacity_jitter=0.9, edge_roughness=0.9
    )

    assert not np.array_equal(out_img1, out_img2) or not np.array_equal(out_mask1, out_mask2)


def test_variability_zero_intensity() -> None:
    image, mask = create_test_data()
    rng = np.random.RandomState(42)

    out_img, out_mask = apply_defect_variability(
        image, mask, rng, color_shift=0.0, opacity_jitter=0.0, edge_roughness=0.0
    )

    assert np.array_equal(out_img, image)
    assert np.array_equal(out_mask, mask)


def test_variability_color_shift_changes_hue() -> None:
    image, mask = create_test_data()
    image[:] = [255, 0, 0]  # red

    rng = np.random.RandomState(42)
    out_img, out_mask = apply_defect_variability(
        image, mask, rng, color_shift=1.0, opacity_jitter=0.0, edge_roughness=0.0
    )

    hsv_in = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_out = cv2.cvtColor(out_img, cv2.COLOR_RGB2HSV)

    active = out_mask > 0
    hue_in = hsv_in[active][:, 0]
    hue_out = hsv_out[active][:, 0]

    assert not np.array_equal(hue_in, hue_out)
