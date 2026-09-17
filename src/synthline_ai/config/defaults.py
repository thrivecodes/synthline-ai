"""Default thresholds for quality checks."""

from __future__ import annotations

#: Minimum image dimension (width or height) in pixels.
MIN_DIMENSION: int = 64

#: Maximum image dimension (width or height) in pixels.
MAX_DIMENSION: int = 8192

#: Mean pixel value below which an image is considered too dark.
BRIGHTNESS_LOW: float = 30.0

#: Mean pixel value above which an image is considered too bright.
BRIGHTNESS_HIGH: float = 225.0

#: Laplacian variance below which an image is considered blurry.
BLUR_THRESHOLD: float = 100.0

#: Supported image file extensions.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"})
