"""Standardized benchmark surface fixtures for reproducible evaluation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def generate_brushed_metal(size: int = 256, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Generate synthetic brushed metal surface with directional streaks."""
    if rng is None:
        rng = np.random.RandomState(42)

    # Base gray metal
    base = np.full((size, size), 180, dtype=np.float32)

    # Horizontal streaks (anisotropic texture)
    streak_profile = rng.normal(0, 15, (size, 1)).astype(np.float32)
    streaks = np.repeat(streak_profile, size, axis=1)

    # High frequency noise
    noise = rng.normal(0, 5, (size, size)).astype(np.float32)

    metal = base + streaks + noise
    metal = np.clip(metal, 0, 255).astype(np.uint8)

    # 3-channel BGR
    result: np.ndarray = cv2.cvtColor(metal, cv2.COLOR_GRAY2BGR)
    return result


def generate_ceramic_tile(size: int = 256, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Generate ceramic tile surface with slight mottling and subtle specular sheen."""
    if rng is None:
        rng = np.random.RandomState(42)

    # Smooth base with subtle large-scale variation
    coords = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(coords, coords)
    gradient = (np.sin(xx * 3.14) * np.sin(yy * 3.14) * 15).astype(np.float32)

    noise = cv2.GaussianBlur(
        rng.normal(0, 8, (size, size)).astype(np.float32),
        (15, 15),
        0,
    )
    tile = 210.0 + gradient + noise
    tile = np.clip(tile, 0, 255).astype(np.uint8)

    # Subtle off-white warm tone
    bgr = np.zeros((size, size, 3), dtype=np.uint8)
    bgr[:, :, 0] = np.clip(tile.astype(np.int16) - 5, 0, 255).astype(np.uint8)
    bgr[:, :, 1] = tile
    bgr[:, :, 2] = np.clip(tile.astype(np.int16) + 3, 0, 255).astype(np.uint8)
    return bgr


def generate_woven_fabric(size: int = 256, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Generate textile woven fabric surface with cross-hatched thread pattern."""
    if rng is None:
        rng = np.random.RandomState(42)

    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)

    # Weave frequency: 8 pixels per thread
    pattern_x = np.sin(xx * (2 * np.pi / 8.0))
    pattern_y = np.sin(yy * (2 * np.pi / 8.0))
    weave = (pattern_x * pattern_y * 20.0).astype(np.float32)

    base = np.full((size, size), 140, dtype=np.float32)
    noise = rng.normal(0, 4, (size, size)).astype(np.float32)

    fabric = np.clip(base + weave + noise, 0, 255).astype(np.uint8)
    result: np.ndarray = cv2.cvtColor(fabric, cv2.COLOR_GRAY2BGR)
    return result


def generate_smooth_plastic(
    size: int = 256,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Generate smooth molded polymer surface with soft vignette and micro-texture."""
    if rng is None:
        rng = np.random.RandomState(42)

    base = np.full((size, size), 90, dtype=np.float32)
    fine_noise = rng.normal(0, 2.5, (size, size)).astype(np.float32)

    # Soft radial vignette
    center = size / 2.0
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    vignette = (1.0 - (dist_from_center / (size * 0.75)) * 0.15).astype(np.float32)

    plastic = np.clip((base + fine_noise) * vignette, 0, 255).astype(np.uint8)
    # Dark blue-gray polymer tone
    bgr = np.zeros((size, size, 3), dtype=np.uint8)
    bgr[:, :, 0] = np.clip(plastic.astype(np.int16) + 12, 0, 255).astype(np.uint8)
    bgr[:, :, 1] = plastic
    bgr[:, :, 2] = np.clip(plastic.astype(np.int16) - 10, 0, 255).astype(np.uint8)
    return bgr


def create_benchmark_suite(
    output_dir: Path,
    image_size: int = 256,
    random_seed: int = 42,
) -> dict[str, Path]:
    """Create a standardized benchmark fixture environment with clean seeds and held-out real data.

    Args:
        output_dir: Destination folder.
        image_size: Resolution of generated square images.
        random_seed: Deterministic random seed.

    Returns:
        dict with paths to 'seeds' and 'real_heldout' directories.
    """
    rng = np.random.RandomState(random_seed)

    seeds_dir = output_dir / "seeds"
    real_dir = output_dir / "real_heldout"
    real_masks_dir = real_dir / "masks"

    seeds_dir.mkdir(parents=True, exist_ok=True)
    real_masks_dir.mkdir(parents=True, exist_ok=True)

    surfaces = {
        "brushed_metal": generate_brushed_metal,
        "ceramic_tile": generate_ceramic_tile,
        "woven_fabric": generate_woven_fabric,
        "smooth_plastic": generate_smooth_plastic,
    }

    # 1. Generate clean reference seed images
    for name, gen_fn in surfaces.items():
        seed_img = gen_fn(image_size, rng=rng)
        cv2.imwrite(str(seeds_dir / f"{name}_seed.png"), seed_img)

    # 2. Generate simulated ground-truth real test set (12 images: 6 normal, 6 defect)
    for i in range(12):
        surf_name = list(surfaces.keys())[i % len(surfaces)]
        gen_fn = surfaces[surf_name]
        img = gen_fn(image_size, rng=rng)

        # Environmental shift (simulating different line lighting / sensor conditions)
        exposure_jitter = rng.uniform(0.9, 1.1)
        img = np.clip(img.astype(np.float32) * exposure_jitter, 0, 255).astype(np.uint8)

        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        is_defect = i % 2 == 1

        if is_defect:
            # Draw real-world simulated defect
            d_type = i % 3
            if d_type == 0:
                # Scratch / crack
                pt1 = (int(rng.uniform(20, 100)), int(rng.uniform(20, 200)))
                pt2 = (int(rng.uniform(150, 240)), int(rng.uniform(50, 230)))
                cv2.line(mask, pt1, pt2, 255, thickness=int(rng.uniform(2, 4)))
                img[mask > 0] = (img[mask > 0] * 0.3).astype(np.uint8)
            elif d_type == 1:
                # Stain / blotch
                center = (int(rng.uniform(50, 200)), int(rng.uniform(50, 200)))
                axes = (int(rng.uniform(15, 35)), int(rng.uniform(10, 25)))
                cv2.ellipse(mask, center, axes, rng.uniform(0, 180), 0, 360, 255, -1)
                img[mask > 0] = (img[mask > 0] * 0.45).astype(np.uint8)
            else:
                # Discoloration patch
                center = (int(rng.uniform(60, 190)), int(rng.uniform(60, 190)))
                cv2.circle(mask, center, int(rng.uniform(20, 45)), 255, -1)
                img[mask > 0] = np.clip(img[mask > 0] * 1.3, 0, 255).astype(np.uint8)

        file_name = f"real_{i:03d}_{surf_name}_{'defect' if is_defect else 'normal'}.png"
        cv2.imwrite(str(real_dir / file_name), img)
        cv2.imwrite(str(real_masks_dir / file_name), mask)

    return {
        "seeds": seeds_dir,
        "real_heldout": real_dir,
    }
