"""Industrial camera sensor and lens optics noise simulation."""

from __future__ import annotations

import cv2
import numpy as np


def apply_sensor_noise(
    image: np.ndarray,
    rng: np.random.RandomState,
    intensity: float = 0.2,
) -> np.ndarray:
    """Apply realistic factory camera sensor and optical noise.

    Simulates:
    1. Poisson shot noise + Gaussian thermal read noise.
    2. Chromatic aberration (slight lateral RGB channel displacement).
    3. Defocus or high-speed conveyor vibration blur.

    Args:
        image: BGR uint8 input image.
        rng: Deterministic numpy RandomState.
        intensity: Effect intensity from 0.0 (clean) to 1.0 (heavy sensor degradation).

    Returns:
        Augmented BGR uint8 image.
    """
    if intensity <= 0.0:
        return image.copy()

    h, w, c = image.shape
    img_f = image.astype(np.float32)

    # 1. Poisson-Gaussian Sensor Noise
    # Shot noise variance scales with pixel intensity; read noise is constant variance
    read_noise_std = float(intensity * 12.0)
    shot_noise_scale = float(intensity * 0.8)

    read_noise = rng.normal(0, read_noise_std, (h, w, c)).astype(np.float32)
    # Scaled shot noise
    raw_shot = rng.normal(0, 1.0, (h, w, c)).astype(np.float32)
    shot_noise = raw_shot * np.sqrt(np.maximum(img_f, 1.0)) * shot_noise_scale

    noisy = img_f + read_noise + shot_noise
    out: np.ndarray = np.clip(noisy, 0, 255).astype(np.uint8)

    # 2. Chromatic Aberration (lateral lens dispersion)
    if intensity > 0.25:
        shift_px = float(max(1, int(round(intensity * 2.5))))
        # Shift Blue and Red in opposite directions
        b, g, r = cv2.split(out)
        # Shift blue channel left/up
        m_b = np.array([[1.0, 0.0, -shift_px], [0.0, 1.0, -shift_px]], dtype=np.float32)
        b_shifted = cv2.warpAffine(b, m_b, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Shift red channel right/down
        m_r = np.array([[1.0, 0.0, shift_px], [0.0, 1.0, shift_px]], dtype=np.float32)
        r_shifted = cv2.warpAffine(r, m_r, (w, h), borderMode=cv2.BORDER_REFLECT)

        merged = cv2.merge([b_shifted, g, r_shifted])
        out = merged.astype(np.uint8)

    # 3. Conveyor Motion Blur / Lens Defocus
    if intensity > 0.5:
        # 1D motion blur kernel
        ksize = int(3 + int(intensity * 4))
        if ksize % 2 == 0:
            ksize += 1
        motion_kernel = np.zeros((ksize, ksize), dtype=np.float32)
        angle = rng.uniform(0, np.pi)
        # Draw line through center of kernel
        cx, cy = ksize // 2, ksize // 2
        dx = int(np.cos(angle) * (ksize // 2))
        dy = int(np.sin(angle) * (ksize // 2))
        cv2.line(motion_kernel, (cx - dx, cy - dy), (cx + dx, cy + dy), 1.0, thickness=1)
        k_sum = float(np.sum(motion_kernel))
        if k_sum > 0:
            motion_kernel /= k_sum
            blurred = cv2.filter2D(out, -1, motion_kernel)
            out = blurred.astype(np.uint8)

    return out
