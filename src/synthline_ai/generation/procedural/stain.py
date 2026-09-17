"""Procedural generator for stain and blotch defects."""

from __future__ import annotations

import cv2
import numpy as np

from synthline_ai.generation.base import BaseGenerator, GenerationResult


class StainGenerator(BaseGenerator):
    """Procedural generator for localized stains, splatters, and liquid blotches."""

    @property
    def defect_type(self) -> str:
        """Return the defect type identifier."""
        return "stain"

    def generate(
        self,
        image: np.ndarray,
        seed_name: str,
        random_seed: int,
        severity: float,
        frequency: float = 1.0,
    ) -> GenerationResult:
        """Generate a procedural stain/blotch on the image."""
        rng = np.random.RandomState(random_seed)
        h, w = image.shape[:2]

        full_mask = np.zeros((h, w), dtype=np.uint8)
        output = image.copy()

        # Determine number of stain spots based on frequency and severity
        num_spots = max(1, int(rng.randint(1, 3) * frequency))
        total_stain_pixels = 0

        for _ in range(num_spots):
            # Center of the stain
            center_x = rng.randint(int(w * 0.1), int(w * 0.9))
            center_y = rng.randint(int(h * 0.1), int(h * 0.9))

            # Base radius scaled by severity
            base_radius = int(min(h, w) * (0.04 + severity * 0.12))
            base_radius = max(4, base_radius)

            # Generate an organic polygon or deformed ellipse
            num_vertices = rng.randint(7, 14)
            angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
            radii = base_radius * (1.0 + rng.uniform(-0.35, 0.35, num_vertices))

            vx = (center_x + radii * np.cos(angles)).astype(np.int32)
            vy = (center_y + radii * np.sin(angles)).astype(np.int32)
            pts = np.column_stack((vx, vy)).reshape((-1, 1, 2))

            spot_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(spot_mask, [pts], 255)

            # Smooth out and feather the edges using Gaussian blur
            ksize = int(max(3, (base_radius // 2) * 2 + 1))
            spot_mask = np.asarray(
                cv2.GaussianBlur(spot_mask, (ksize, ksize), 0), dtype=np.uint8
            )

            # Organic thresholding with noise
            noise = rng.randint(0, 50, (h, w), dtype=np.uint8)
            spot_mask = np.asarray(cv2.subtract(spot_mask, noise), dtype=np.uint8)
            _, spot_bin = cv2.threshold(spot_mask, 30, 255, cv2.THRESH_BINARY)
            spot_bin_u8 = np.asarray(spot_bin, dtype=np.uint8)

            full_mask = np.asarray(cv2.bitwise_or(full_mask, spot_bin_u8), dtype=np.uint8)


        mask_bool = full_mask > 0
        total_stain_pixels = int(np.count_nonzero(mask_bool))

        # Alpha opacity based on severity
        alpha = float(np.clip(0.35 + severity * 0.5, 0.2, 0.85))

        if total_stain_pixels > 0:
            if len(image.shape) == 3:
                # Tint stain with realistic organic coloration (dark brownish or oil-slick shift)
                mean_bgr = cv2.mean(image, mask=full_mask)[:3]
                # Shift toward dark brownish/yellowish
                target_color = np.array(
                    [
                        max(0, mean_bgr[0] * 0.4 - 20),
                        max(0, mean_bgr[1] * 0.6 - 15),
                        max(0, mean_bgr[2] * 0.7 + 10),
                    ]
                )
                target_color = np.clip(target_color, 0, 255)

                for c in range(3):
                    c_out = output[:, :, c]
                    c_img = image[:, :, c]
                    c_out[mask_bool] = np.clip(
                        c_img[mask_bool] * (1.0 - alpha) + target_color[c] * alpha, 0, 255
                    )
            else:
                gray_mean = float(cv2.mean(image, mask=full_mask)[0])
                darken = min(150, max(25, int(30 + severity * 60)))
                gray_stain = max(0.0, gray_mean - darken)
                output[mask_bool] = np.clip(
                    image[mask_bool] * (1.0 - alpha) + gray_stain * alpha, 0, 255
                )

        metadata: dict[str, object] = {
            "num_spots": num_spots,
            "stain_pixels": total_stain_pixels,
            "opacity": alpha,
            "frequency": frequency,
        }


        return GenerationResult(
            image=output,
            mask=full_mask,
            metadata=metadata,
            source_seed=seed_name,
            defect_type=self.defect_type,
            random_seed=random_seed,
        )
