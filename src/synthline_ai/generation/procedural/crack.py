"""Procedural generator for branching crack and fracture defects."""

from __future__ import annotations

import cv2
import numpy as np

from synthline_ai.generation.base import BaseGenerator, GenerationResult


class CrackGenerator(BaseGenerator):
    """Procedural generator for structural cracks and branching fracture fissures."""

    @property
    def defect_type(self) -> str:
        """Return the defect type identifier."""
        return "crack"

    def generate(
        self,
        image: np.ndarray,
        seed_name: str,
        random_seed: int,
        severity: float,
        frequency: float = 1.0,
    ) -> GenerationResult:
        """Generate a branching crack fracture defect on the image.

        Args:
            image: Source BGR uint8 image.
            seed_name: Filename of the source seed image.
            random_seed: Random seed for reproducibility.
            severity: Defect severity from 0.0 (micro-crack) to 1.0 (severe fissure).
            frequency: Controls number of primary crack clusters (default: 1.0).

        Returns:
            GenerationResult containing modified image, binary mask, and metadata.
        """
        rng = np.random.RandomState(random_seed)
        h, w = image.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        num_primary_cracks = max(1, int(round(frequency)))
        total_branches = 0
        total_length_pixels = 0.0
        crack_width = max(1, int(1 + severity * 3))

        for _ in range(num_primary_cracks):
            # Pick a starting point near edge or within bounds
            if rng.rand() < 0.6:
                # Start near edge
                edge = rng.choice(["top", "bottom", "left", "right"])
                if edge == "top":
                    cur_x, cur_y = float(rng.randint(int(w * 0.1), int(w * 0.9))), 0.0
                    angle = rng.uniform(0.2 * np.pi, 0.8 * np.pi)
                elif edge == "bottom":
                    cur_x, cur_y = float(rng.randint(int(w * 0.1), int(w * 0.9))), float(h - 1)
                    angle = rng.uniform(-0.8 * np.pi, -0.2 * np.pi)
                elif edge == "left":
                    cur_x, cur_y = 0.0, float(rng.randint(int(h * 0.1), int(h * 0.9)))
                    angle = rng.uniform(-0.3 * np.pi, 0.3 * np.pi)
                else:
                    cur_x, cur_y = float(w - 1), float(rng.randint(int(h * 0.1), int(h * 0.9)))
                    angle = rng.uniform(0.7 * np.pi, 1.3 * np.pi)
            else:
                cur_x = float(rng.randint(int(w * 0.2), int(w * 0.8)))
                cur_y = float(rng.randint(int(h * 0.2), int(h * 0.8)))
                angle = rng.uniform(0, 2 * np.pi)

            # Main trunk propagation
            trunk_steps = rng.randint(15, max(20, int(25 + severity * 40)))
            step_size = max(3.0, min(12.0, (w + h) / 60.0))

            trunk_points: list[tuple[int, int]] = [(int(cur_x), int(cur_y))]
            branch_seeds: list[tuple[float, float, float]] = []

            for step_i in range(trunk_steps):
                angle += rng.normal(0.0, 0.35)  # Tortuous random walk
                next_x = cur_x + np.cos(angle) * step_size
                next_y = cur_y + np.sin(angle) * step_size

                if not (0 <= next_x < w and 0 <= next_y < h):
                    break

                pt1 = (int(cur_x), int(cur_y))
                pt2 = (int(next_x), int(next_y))
                cv2.line(mask, pt1, pt2, 255, thickness=crack_width)
                total_length_pixels += float(np.hypot(next_x - cur_x, next_y - cur_y))

                cur_x, cur_y = next_x, next_y
                trunk_points.append((int(cur_x), int(cur_y)))

                # Potential branch candidate
                if step_i > 4 and rng.rand() < (0.2 + severity * 0.15):
                    # Branch diverges by 35 to 65 degrees
                    branch_dir = rng.choice([-1.0, 1.0])
                    branch_angle = angle + branch_dir * rng.uniform(0.6, 1.2)
                    branch_seeds.append((cur_x, cur_y, branch_angle))

            # Propagate secondary branches
            branch_count = min(len(branch_seeds), max(1, int(severity * 4)))
            for b_idx in range(branch_count):
                bx, by, b_ang = branch_seeds[b_idx]
                b_steps = rng.randint(5, max(8, int(10 + severity * 15)))
                b_width = max(1, crack_width - 1)
                total_branches += 1

                for _ in range(b_steps):
                    b_ang += rng.normal(0.0, 0.3)
                    nbx = bx + np.cos(b_ang) * (step_size * 0.8)
                    nby = by + np.sin(b_ang) * (step_size * 0.8)

                    if not (0 <= nbx < w and 0 <= nby < h):
                        break

                    cv2.line(
                        mask,
                        (int(bx), int(by)),
                        (int(nbx), int(nby)),
                        255,
                        thickness=b_width,
                    )
                    total_length_pixels += float(np.hypot(nbx - bx, nby - by))
                    bx, by = nbx, nby

        # Composite crack onto image
        output = image.copy()
        crack_pixels = mask > 0

        if crack_pixels.any():
            # Crack core is significantly darkened (shadow inside fissure)
            darkness_factor = max(0.1, 0.5 - severity * 0.35)
            output[crack_pixels] = (output[crack_pixels] * darkness_factor).astype(np.uint8)

            # Optional stress halo (slight darkening/highlighting around crack boundary)
            if severity > 0.4:
                dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
                halo = (dilated > 0) & (~crack_pixels)
                if halo.any():
                    darkened = output[halo].astype(np.int16) - 15
                    output[halo] = np.clip(darkened, 0, 255).astype(np.uint8)

        # Ensure clean binary mask
        mask = (mask > 0).astype(np.uint8) * 255

        metadata: dict[str, object] = {
            "num_primary_cracks": num_primary_cracks,
            "total_branches": total_branches,
            "total_length_pixels": round(total_length_pixels, 1),
            "crack_width": crack_width,
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
