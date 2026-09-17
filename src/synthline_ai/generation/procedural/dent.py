"""Procedural dent and surface impression generator for industrial inspection."""

from __future__ import annotations

import cv2
import numpy as np

from synthline_ai.generation.base import BaseGenerator, GenerationResult


class DentGenerator(BaseGenerator):
    """Generates realistic 3D surface dents, bumps, and tool impressions.

    Simulates surface deformation under directional industrial lighting:
    highlighting the rim facing the light source and casting shadows in the trough.
    """

    @property
    def defect_type(self) -> str:
        return "dent"

    def generate(
        self,
        image: np.ndarray,
        seed_name: str,
        random_seed: int,
        severity: float,
        frequency: float = 1.0,
    ) -> GenerationResult:
        """Generate one or more surface dents/impressions on the workpiece."""
        rng = np.random.RandomState(random_seed)
        h, w = image.shape[:2]
        output = image.copy()
        mask = np.zeros((h, w), dtype=np.uint8)

        # Number of dents
        num_dents = max(1, int(round(frequency)))
        # Margin from image borders
        margin = max(10, int(min(h, w) * 0.1))

        dents_info = []

        for _ in range(num_dents):
            if w <= 2 * margin or h <= 2 * margin:
                cx, cy = w // 2, h // 2
            else:
                cx = rng.randint(margin, w - margin)
                cy = rng.randint(margin, h - margin)

            # Dent dimensions
            base_radius = min(h, w) * (0.04 + severity * 0.10)
            aspect = rng.uniform(0.7, 1.3)
            rx = max(4.0, base_radius * aspect)
            ry = max(4.0, base_radius / aspect)

            # Orientation and illumination angle
            rot_angle = rng.uniform(0.0, np.pi)
            light_angle = rng.uniform(0.0, 2.0 * np.pi)
            light_dir = np.array([np.cos(light_angle), np.sin(light_angle)], dtype=np.float32)

            # Bounding box for local computation to maximize speed
            pad = int(max(rx, ry) * 1.5) + 2
            x0 = max(0, cx - pad)
            x1 = min(w, cx + pad + 1)
            y0 = max(0, cy - pad)
            y1 = min(h, cy + pad + 1)

            if x1 <= x0 or y1 <= y0:
                continue

            # Local coordinates grid
            grid_y, grid_x = np.ogrid[y0:y1, x0:x1]
            dx = (grid_x - cx).astype(np.float32)
            dy = (grid_y - cy).astype(np.float32)

            # Rotate coordinates into ellipse principal axes
            cos_r = np.cos(rot_angle)
            sin_r = np.sin(rot_angle)
            xr = dx * cos_r + dy * sin_r
            yr = -dx * sin_r + dy * cos_r

            # Normalized elliptic radius
            norm_dist_sq = (xr / rx) ** 2 + (yr / ry) ** 2
            dent_mask_local = norm_dist_sq <= 1.0

            if not np.any(dent_mask_local):
                continue

            # Surface deformation profile: Z(d) = (1 - d^2)^1.5
            # Gradient gives directional slope
            profile = np.maximum(0.0, 1.0 - norm_dist_sq) ** 1.5

            # Gradients along original X and Y
            grad_xr = -2.0 * (xr / (rx**2)) * profile
            grad_yr = -2.0 * (yr / (ry**2)) * profile

            # Rotate gradients back to image coordinate frame
            grad_x = grad_xr * cos_r - grad_yr * sin_r
            grad_y = grad_xr * sin_r + grad_yr * cos_r

            # Directional shading: dot product with light vector
            shading = grad_x * light_dir[0] + grad_y * light_dir[1]

            # Normalize shading magnitude
            max_s = np.max(np.abs(shading)) or 1.0
            shading_norm = shading / max_s

            # Shading intensity scaled by severity
            intensity_scale = 30.0 + severity * 70.0
            shade_delta = shading_norm * intensity_scale

            # Apply shading to local patch in output image
            patch = output[y0:y1, x0:x1].astype(np.float32)
            for c in range(patch.shape[2]):
                patch[:, :, c] += shade_delta * dent_mask_local

            output[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)

            # Update binary mask
            mask[y0:y1, x0:x1][dent_mask_local] = 255

            dents_info.append(
                {
                    "center": (cx, cy),
                    "rx": float(rx),
                    "ry": float(ry),
                    "light_angle": float(light_angle),
                }
            )

        # Final cleanup: ensure clean binary mask
        mask = (mask > 0).astype(np.uint8) * 255

        # Fallback if no pixels were marked
        if not np.any(mask):
            cv2.circle(mask, (w // 2, h // 2), max(3, int(min(h, w) * 0.05)), 255, -1)
            cv2.circle(output, (w // 2, h // 2), max(3, int(min(h, w) * 0.05)), (30, 30, 30), -1)

        metadata: dict[str, object] = {
            "num_dents": len(dents_info),
            "dents": dents_info,
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
