"""Procedural generator for pinholes, porosity voids, and surface pitting defects."""

from __future__ import annotations

import cv2
import numpy as np

from synthline_ai.generation.base import BaseGenerator, GenerationResult


class PinholeGenerator(BaseGenerator):
    """Procedural generator for pinholes, voids, and pitting clusters."""

    @property
    def defect_type(self) -> str:
        """Return the defect type identifier."""
        return "pinhole"

    def generate(
        self,
        image: np.ndarray,
        seed_name: str,
        random_seed: int,
        severity: float,
        frequency: float = 1.0,
    ) -> GenerationResult:
        """Generate pinhole / pitting defect voids on the image.

        Args:
            image: Source BGR uint8 image.
            seed_name: Filename of the source seed image.
            random_seed: Random seed for reproducibility.
            severity: Defect severity from 0.0 (micro-pores) to 1.0 (large pits).
            frequency: Defect count / cluster density multiplier (default: 1.0).

        Returns:
            GenerationResult containing modified image, binary mask, and metadata.
        """
        rng = np.random.RandomState(random_seed)
        h, w = image.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        output = image.copy()

        # Determine count of pinhole pits
        base_count = rng.randint(3, 8)
        num_pinholes = max(1, int(round(base_count * frequency)))

        # Cluster center or dispersed
        is_cluster = rng.rand() < 0.65
        cluster_cx = float(rng.randint(int(w * 0.2), int(w * 0.8)))
        cluster_cy = float(rng.randint(int(h * 0.2), int(h * 0.8)))
        cluster_radius = float(min(w, h)) * rng.uniform(0.1, 0.25)

        total_pit_area = 0
        radii_list: list[float] = []

        for _ in range(num_pinholes):
            if is_cluster:
                angle = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(0, cluster_radius)
                cx = int(np.clip(cluster_cx + np.cos(angle) * dist, 5, w - 6))
                cy = int(np.clip(cluster_cy + np.sin(angle) * dist, 5, h - 6))
            else:
                cx = rng.randint(5, w - 5)
                cy = rng.randint(5, h - 5)

            # Radius from 1px to 8px based on severity
            radius = max(1, int(round(1.5 + severity * 6.0 * rng.uniform(0.6, 1.3))))
            radii_list.append(float(radius))

            # Draw pit crater onto mask
            axes = (radius, max(1, int(radius * rng.uniform(0.8, 1.2))))
            rot_angle = rng.uniform(0, 180)
            cv2.ellipse(mask, (cx, cy), axes, rot_angle, 0, 360, 255, -1)

            # Pit crater shading: deep dark center
            pit_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(pit_mask, (cx, cy), axes, rot_angle, 0, 360, 255, -1)
            pit_pixels = pit_mask > 0
            total_pit_area += int(np.sum(pit_pixels))

            # Darken void core
            core_darkness = max(0.05, 0.35 - severity * 0.25)
            output[pit_pixels] = (output[pit_pixels] * core_darkness).astype(np.uint8)

            # Specular rim highlight on upper-left lip (simulating light catching crater edge)
            rim_x = int(np.clip(cx - int(radius * 0.7), 0, w - 1))
            rim_y = int(np.clip(cy - int(radius * 0.7), 0, h - 1))
            rim_radius = max(1, int(radius * 0.5))
            rim_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(rim_mask, (rim_x, rim_y), rim_radius, 255, -1)
            rim_pixels = (rim_mask > 0) & (~pit_pixels)
            if rim_pixels.any():
                output[rim_pixels] = np.clip(
                    output[rim_pixels].astype(np.int16) + int(40 + severity * 40),
                    0,
                    255,
                ).astype(np.uint8)

        # Ensure single channel binary uint8 mask
        mask = (mask > 0).astype(np.uint8) * 255

        avg_radius = float(np.mean(radii_list)) if radii_list else 0.0
        metadata: dict[str, object] = {
            "num_pinholes": num_pinholes,
            "avg_radius_pixels": round(avg_radius, 2),
            "total_pit_area": total_pit_area,
            "is_cluster": is_cluster,
            "severity": severity,
            "frequency": frequency,
        }

        return GenerationResult(
            image=output,
            mask=mask,
            metadata=metadata,
            source_seed=seed_name,
            defect_type=self.defect_type,
            random_seed=random_seed,
        )
