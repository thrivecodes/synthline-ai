"""Core Pydantic models used across SynthLine AI."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class DefectType(StrEnum):
    """Supported defect generation types."""

    SCRATCH = "scratch"


class GenerationConfig(BaseModel):
    """Configuration for a single generation run."""

    seeds_dir: Path
    defect_type: DefectType = DefectType.SCRATCH
    count: int = Field(default=100, ge=1, le=10000)
    output_dir: Path
    random_seed: int = Field(default=42)
    severity: float = Field(default=0.5, ge=0.0, le=1.0)


class ImageInfo(BaseModel):
    """Metadata about a loaded seed image."""

    path: Path
    width: int
    height: int
    channels: int
    mean_brightness: float
    blur_score: float  # Laplacian variance


class QualityWarning(BaseModel):
    """A quality warning for a seed image."""

    path: Path
    code: str  # e.g. "too_dark", "too_bright", "blurry", "odd_dimensions"
    message: str
    severity: str = "warning"  # "warning" | "error"


class SeedSetReport(BaseModel):
    """Report on the quality of the seed image set."""

    images: list[ImageInfo]
    warnings: list[QualityWarning]
    valid_count: int
    rejected_count: int
