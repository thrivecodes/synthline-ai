"""Generate procedural seed images of industrial surfaces for testing and prototyping."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def generate_brushed_metal(size: int = 256, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Generate brushed aluminum/steel surface texture with horizontal brush lines."""
    if rng is None:
        rng = np.random.RandomState(42)

    # Base mid-tone metallic gray
    base = np.full((size, size), 180, dtype=np.float32)

    # 1D line noise stretched horizontally
    line_noise = rng.normal(0, 15, (size, 1)).astype(np.float32)
    brushed = base + np.tile(line_noise, (1, size))

    # Fine high-frequency grain
    grain = rng.normal(0, 4, (size, size)).astype(np.float32)
    texture = brushed + grain

    # Directional illumination gradient
    gradient = np.linspace(-10, 10, size, dtype=np.float32).reshape(1, size)
    texture = texture + gradient

    clipped = np.clip(texture, 0, 255).astype(np.uint8)
    return np.asarray(cv2.cvtColor(clipped, cv2.COLOR_GRAY2BGR))


def generate_ceramic_tile(size: int = 256, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Generate clean polished ceramic/polymer surface with subtle tonal variations."""
    if rng is None:
        rng = np.random.RandomState(43)

    # Warm off-white base
    base = np.full((size, size, 3), [235, 238, 240], dtype=np.float32)

    # Low frequency smooth shading (surface curvature)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    vignette = (1.0 - 0.08 * (xx**2 + yy**2))[:, :, np.newaxis].astype(np.float32)
    base = base * vignette

    # Micro-texture noise
    noise = rng.normal(0, 2.5, (size, size, 3)).astype(np.float32)
    texture = base + noise

    return np.asarray(np.clip(texture, 0, 255).astype(np.uint8))


def generate_matte_polymer(size: int = 256, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Generate dark matte plastic/polymer texture with subtle stipple roughness."""
    if rng is None:
        rng = np.random.RandomState(44)

    base = np.full((size, size), 70, dtype=np.float32)

    # Multi-scale Gaussian noise
    coarse = cv2.resize(rng.normal(0, 6, (size // 8, size // 8)), (size, size))
    fine = rng.normal(0, 3, (size, size))

    texture = base + coarse + fine
    clipped = np.clip(texture, 0, 255).astype(np.uint8)
    return np.asarray(cv2.cvtColor(clipped, cv2.COLOR_GRAY2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample industrial surface seed images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/seeds"),
        help="Directory to save generated seed images (default: data/seeds)",
    )
    parser.add_argument("--size", type=int, default=256, help="Image dimension in pixels")
    parser.add_argument("--count-per-type", type=int, default=3, help="Seeds per surface type")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(100)

    generators = [
        ("metal", generate_brushed_metal),
        ("ceramic", generate_ceramic_tile),
        ("polymer", generate_matte_polymer),
    ]

    saved_paths: list[Path] = []
    for type_name, gen_func in generators:
        for i in range(args.count_per_type):
            img = gen_func(size=args.size, rng=rng)
            filename = f"seed_{type_name}_{i + 1:02d}.png"
            path = args.output_dir / filename
            cv2.imwrite(str(path), img)
            saved_paths.append(path)
            print(f"Generated {filename} ({args.size}x{args.size})")

    print(f"\nSuccessfully generated {len(saved_paths)} seed images in {args.output_dir}")


if __name__ == "__main__":
    main()
