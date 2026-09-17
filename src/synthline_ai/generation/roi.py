"""Region of Interest (ROI) extraction and defect constraint engine."""

from __future__ import annotations

import cv2
import numpy as np


def compute_roi_mask(
    image: np.ndarray,
    min_area_ratio: float = 0.05,
) -> np.ndarray:
    """Extract workpiece foreground ROI mask using adaptive segmentation.

    Separates the physical part/workpiece from conveyor, bench, or background lighting.
    If the image is already a cropped workpiece texture, returns an all-foreground mask.

    Args:
        image: BGR uint8 input image.
        min_area_ratio: Minimum fraction of image area required for workpiece contour.

    Returns:
        Single-channel uint8 mask (0 = background/conveyor, 255 = workpiece surface).
    """
    h, w = image.shape[:2]
    total_pixels = h * w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Blur to suppress micro-texture noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu thresholding
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine whether background is bright or dark by checking 4 image corners
    corners = [
        gray[0, 0],
        gray[0, -1],
        gray[-1, 0],
        gray[-1, -1],
    ]
    mean_corner = float(np.mean(corners))
    mean_center = float(np.mean(gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]))

    # If corners are brighter than center, invert threshold mask
    fg_candidate = cv2.bitwise_not(otsu_thresh) if mean_corner > mean_center else otsu_thresh

    # Morphological closing to seal internal cavities in workpiece
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(fg_candidate, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find external contours and select workpiece
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full((h, w), 255, dtype=np.uint8)

    # Filter by minimum area
    valid_contours = [c for c in contours if cv2.contourArea(c) >= total_pixels * min_area_ratio]
    if not valid_contours:
        # If no distinct object was found, treat entire canvas as valid workpiece
        return np.full((h, w), 255, dtype=np.uint8)

    # Render selected workpiece contours
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(roi_mask, valid_contours, -1, 255, thickness=cv2.FILLED)

    return roi_mask


def apply_roi_constraint(
    image: np.ndarray,
    mask: np.ndarray,
    clean_image: np.ndarray,
    roi_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Constrain defect modifications strictly to the workpiece ROI.

    Restores any background/conveyor pixels that were unintentionally modified,
    and zeroes out the defect mask outside the workpiece surface.

    Args:
        image: Modified defective image.
        mask: Binary defect mask.
        clean_image: Original pristine seed image.
        roi_mask: Binary workpiece ROI mask (255 = part, 0 = background).

    Returns:
        (constrained_image, constrained_mask) tuple.
    """
    outside_roi = roi_mask == 0
    constrained_img = image.copy()
    constrained_mask = mask.copy()

    # Revert image pixels outside workpiece
    constrained_img[outside_roi] = clean_image[outside_roi]
    # Zero out mask outside workpiece
    constrained_mask[outside_roi] = 0

    return constrained_img, constrained_mask
