from __future__ import annotations

import cv2
import numpy as np

from synthline_ai.generation.base import BaseGenerator, GenerationResult


class ScratchGenerator(BaseGenerator):
    """Procedural generator for scratch defects."""

    @property
    def defect_type(self) -> str:
        """Return the defect type identifier."""
        return "scratch"

    def generate(
        self,
        image: np.ndarray,
        seed_name: str,
        random_seed: int,
        severity: float,
    ) -> GenerationResult:
        """
        Generate a scratch defect on the image.

        Args:
            image: Source BGR uint8 image.
            seed_name: Filename of the source seed image.
            random_seed: Random seed for reproducibility.
            severity: Defect severity from 0.0 (subtle) to 1.0 (severe).

        Returns:
            A GenerationResult containing the modified image, mask, and metadata.
        """
        rng = np.random.RandomState(random_seed)
        h, w = image.shape[:2]

        # Create blank mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # Generate random scratch path
        num_points = rng.randint(2, max(3, int(severity * 5) + 3))
        points = []
        for _ in range(num_points):
            points.append([rng.randint(0, w), rng.randint(0, h)])

        # Interpolate a smooth polyline
        if num_points > 2:
            pts = np.array(points)
            t = np.linspace(0, 1, len(pts))
            t_interp = np.linspace(0, 1, max(10, len(pts) * 5))
            x_interp = np.interp(t_interp, t, pts[:, 0])
            y_interp = np.interp(t_interp, t, pts[:, 1])
            points_interp = np.column_stack((x_interp, y_interp))
            curve_pts = points_interp.astype(np.int32).reshape((-1, 1, 2))
        else:
            curve_pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            points_interp = np.array(points, dtype=np.float64)


        # Draw polyline on the mask
        width = 1 + int(severity * 4)
        cv2.polylines(
            mask, [curve_pts], isClosed=False, color=255, thickness=width, lineType=cv2.LINE_AA
        )

        # Optionally apply slight Gaussian blur to the mask for softer edges
        if severity > 0.0:
            mask = np.asarray(cv2.GaussianBlur(mask, (3, 3), 0), dtype=np.uint8)


        # Composite the scratch onto a COPY of the source image
        output = image.copy()

        # Approximate path length
        if len(points_interp) > 1:
            length_pixels = float(np.sum(np.linalg.norm(np.diff(points_interp, axis=0), axis=1)))
        else:
            length_pixels = 0.0

        alpha = np.clip(0.3 + severity * 0.6, 0.3, 0.9)

        mask_bool = mask > 0
        if not np.any(mask_bool):
            metadata_empty = {
                "width": width,
                "length_pixels": 0.0,
                "opacity": alpha,
                "num_control_points": num_points,
            }
            return GenerationResult(
                image=output,
                mask=np.zeros_like(mask),
                metadata=metadata_empty,
                source_seed=seed_name,
                defect_type=self.defect_type,
                random_seed=random_seed,
            )

        # Determine scratch color and blend
        if len(image.shape) == 3:
            mean_vals = cv2.mean(image, mask=mask)[:3]
            darken = min(255, max(30, int(30 + severity * 50)))
            scratch_color = np.clip(np.array(mean_vals) - darken, 0, 255)

            for c in range(3):
                c_out = output[:, :, c]
                c_img = image[:, :, c]
                c_out[mask_bool] = np.clip(
                    c_img[mask_bool] * (1 - alpha) + scratch_color[c] * alpha, 0, 255
                )
        else:
            gray_mean = float(cv2.mean(image, mask=mask)[0])
            darken = min(255, max(30, int(30 + severity * 50)))
            gray_scratch = max(0.0, gray_mean - darken)
            output[mask_bool] = np.clip(
                image[mask_bool] * (1 - alpha) + gray_scratch * alpha, 0, 255
            )

        # Ensure the final mask is binary
        _, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        mask = np.asarray(mask_bin, dtype=np.uint8)


        metadata = {
            "width": width,
            "length_pixels": length_pixels,
            "opacity": alpha,
            "num_control_points": num_points,
        }

        return GenerationResult(
            image=output,
            mask=mask,
            metadata=metadata,
            source_seed=seed_name,
            defect_type=self.defect_type,
            random_seed=random_seed,
        )
