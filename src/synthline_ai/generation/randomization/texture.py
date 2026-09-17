"""Micro-texture and camera sensor noise randomization."""

from __future__ import annotations

import cv2
import numpy as np


def apply_texture_variation(
    image: np.ndarray,
    rng: np.random.RandomState,
    intensity: float = 0.15,
) -> np.ndarray:
    """Add subtle micro-surface roughness and camera sensor noise.

    Args:
        image: Source image array (H, W, C) or (H, W) in uint8.
        rng: Reproducible numpy RandomState.
        intensity: Noise intensity (0.0 to 1.0).

    Returns:
        Texture-augmented image array in uint8.
    """
    if intensity <= 0.0:
        return image.copy()

    h, w = image.shape[:2]

    # High-frequency sensor noise (Gaussian)
    sigma = intensity * 8.0
    noise = rng.normal(0, sigma, image.shape).astype(np.float32)

    # Low-frequency surface roughness
    roughness = rng.uniform(-1.0, 1.0, (h // 4 or 1, w // 4 or 1)).astype(np.float32)
    roughness_resized = cv2.resize(roughness, (w, h), interpolation=cv2.INTER_CUBIC)
    roughness_scaled = roughness_resized * (intensity * 12.0)

    float_img = image.astype(np.float32)
    if image.ndim == 3:
        roughness_scaled = roughness_scaled[:, :, np.newaxis]

    output = float_img + noise + roughness_scaled
    result: np.ndarray = np.clip(output, 0, 255).astype(np.uint8)
    return result
