"""Procedural generator for discoloration and thermal/chemical fading defects."""

from __future__ import annotations

import cv2
import numpy as np

from synthline_ai.generation.base import BaseGenerator, GenerationResult


class DiscolorationGenerator(BaseGenerator):
    """Procedural generator for oxidation, thermal discoloration, or color bleaching."""

    @property
    def defect_type(self) -> str:
        """Return the defect type identifier."""
        return "discoloration"

    def generate(
        self,
        image: np.ndarray,
        seed_name: str,
        random_seed: int,
        severity: float,
        frequency: float = 1.0,
    ) -> GenerationResult:
        """Generate a procedural discoloration zone on the image."""
        rng = np.random.RandomState(random_seed)
        h, w = image.shape[:2]

        output = image.copy()
        mask = np.zeros((h, w), dtype=np.uint8)

        # Discoloration patch center and axes
        center_x = rng.randint(int(w * 0.15), int(w * 0.85))
        center_y = rng.randint(int(h * 0.15), int(h * 0.85))

        major_axis = int(min(h, w) * (0.1 + severity * 0.25 * frequency))
        minor_axis = int(major_axis * rng.uniform(0.4, 0.9))
        angle = rng.uniform(0, 180)

        # Draw smooth elliptic zone
        cv2.ellipse(
            mask,
            (center_x, center_y),
            (max(8, major_axis), max(5, minor_axis)),
            angle,
            0,
            360,
            255,
            -1,
        )

        # Add soft perlin-like gradient blur
        blur_kernel = max(7, (major_axis // 3) * 2 + 1)
        soft_mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)

        # Create binary ground truth mask with a conservative boundary
        _, binary_mask = cv2.threshold(soft_mask, 50, 255, cv2.THRESH_BINARY)
        mask = np.asarray(binary_mask, dtype=np.uint8)

        mask_bool = mask > 0
        total_pixels = int(np.count_nonzero(mask_bool))

        # Color shift calculation
        alpha = float(np.clip(0.3 + severity * 0.45, 0.25, 0.75))

        if total_pixels > 0:
            if len(image.shape) == 3:
                # Convert to HSV for realistic hue shift / oxidation patina
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
                # Shift hue (e.g. rust/heat tinting)
                hue_shift = rng.choice([-30.0, 25.0, 45.0, -15.0])
                hsv[mask_bool, 0] = (hsv[mask_bool, 0] + hue_shift) % 180
                # Increase or decrease saturation
                sat_mult = 1.0 + severity * 0.6
                hsv[mask_bool, 1] = np.clip(hsv[mask_bool, 1] * sat_mult, 0, 255)

                modified_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

                for c in range(3):
                    c_out = output[:, :, c]
                    c_mod = modified_bgr[:, :, c]
                    c_img = image[:, :, c]
                    c_out[mask_bool] = np.clip(
                        c_img[mask_bool] * (1.0 - alpha) + c_mod[mask_bool] * alpha, 0, 255
                    )
            else:
                # Contrast/brightness fade for single channel
                factor = 1.0 + (severity * 0.4 if rng.rand() > 0.5 else -severity * 0.4)
                output[mask_bool] = np.clip(image[mask_bool] * factor, 0, 255).astype(np.uint8)

        metadata: dict[str, object] = {
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "angle": angle,
            "opacity": alpha,
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
