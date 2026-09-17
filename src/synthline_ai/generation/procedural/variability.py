"""Per-defect appearance variability: color shifts, opacity jitter, edge roughness."""

from __future__ import annotations

import cv2
import numpy as np


def apply_defect_variability(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.RandomState,
    color_shift: float = 0.3,
    opacity_jitter: float = 0.2,
    edge_roughness: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies appearance variability to a defect region.
    """
    out_image = image.copy()
    out_mask = mask.copy()

    # Edge roughness
    if edge_roughness > 0:
        kernel_size = int(round(edge_roughness * 3))
        if kernel_size > 0:
            kernel_size = max(3, kernel_size | 1)
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            op = rng.choice(["erode", "dilate"])

            if op == "erode":
                out_mask = cv2.erode(out_mask, kernel, iterations=1)
            else:
                out_mask = cv2.dilate(out_mask, kernel, iterations=1)

            out_mask = np.where(out_mask > 127, 255, 0).astype(np.uint8)

    active_mask = out_mask > 0

    if not np.any(active_mask):
        return out_image, out_mask

    # Color shift
    if color_shift > 0 and len(out_image.shape) == 3 and out_image.shape[2] == 3:
        hsv = cv2.cvtColor(out_image, cv2.COLOR_RGB2HSV).astype(np.float32)

        hue_shift = rng.uniform(-15.0 * color_shift, 15.0 * color_shift)
        sat_shift = rng.uniform(-30.0 * color_shift, 30.0 * color_shift)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]

        h[active_mask] = (h[active_mask] + hue_shift) % 180.0
        s[active_mask] = np.clip(s[active_mask] + sat_shift, 0, 255)

        hsv[:, :, 0] = h
        hsv[:, :, 1] = s

        shifted_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        out_image[active_mask] = shifted_rgb[active_mask]

    # Opacity jitter
    if opacity_jitter > 0:
        jitter = rng.uniform(-opacity_jitter, opacity_jitter, size=out_image.shape[:2])
        if len(out_image.shape) == 3:
            jitter = jitter[:, :, np.newaxis]

        jittered = out_image.astype(np.float32) * (1.0 + jitter)
        jittered = np.clip(jittered, 0, 255).astype(np.uint8)

        if len(out_image.shape) == 3:
            out_image = np.where(np.expand_dims(active_mask, -1), jittered, out_image)
        else:
            out_image = np.where(active_mask, jittered, out_image)

    return out_image, out_mask
