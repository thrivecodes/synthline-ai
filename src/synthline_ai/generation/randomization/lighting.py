"""Lighting and illumination field randomization."""

from __future__ import annotations

import numpy as np


def apply_lighting_variation(
    image: np.ndarray,
    rng: np.random.RandomState,
    intensity: float = 0.2,
) -> np.ndarray:
    """Apply non-uniform illumination gradient and optional vignetting.

    Args:
        image: Source image array (H, W, C) or (H, W) in uint8.
        rng: Reproducible numpy RandomState.
        intensity: Strength of the lighting gradient (0.0 to 1.0).

    Returns:
        Illuminated image array in uint8.
    """
    if intensity <= 0.0:
        return image.copy()

    h, w = image.shape[:2]

    # Generate linear gradient across random direction
    angle = rng.uniform(0, 2 * np.pi)
    grad_x = np.linspace(-1.0, 1.0, w) * np.cos(angle)
    grad_y = np.linspace(-1.0, 1.0, h) * np.sin(angle)
    xx, yy = np.meshgrid(grad_x, grad_y)
    gradient = xx + yy

    # Normalize gradient to [-1.0, 1.0]
    max_val = np.max(np.abs(gradient))
    if max_val > 0:
        gradient /= max_val

    # Scale by intensity (e.g. up to +/- 20% brightness change)
    delta = gradient * (intensity * 40.0)

    # Optional subtle radial vignette
    if rng.rand() > 0.5:
        y_c, x_c = np.indices((h, w))
        dist_from_center = np.sqrt(((x_c - w / 2) / (w / 2)) ** 2 + ((y_c - h / 2) / (h / 2)) ** 2)
        vignette = 1.0 - (dist_from_center * intensity * 0.3)
        vignette = np.clip(vignette, 0.7, 1.0)
    else:
        vignette = 1.0

    float_img = image.astype(np.float32)

    if image.ndim == 3:
        delta = delta[:, :, np.newaxis]
        if isinstance(vignette, np.ndarray):
            vignette = vignette[:, :, np.newaxis]

    adjusted = (float_img * vignette) + delta
    output: np.ndarray = np.clip(adjusted, 0, 255).astype(np.uint8)
    return output
