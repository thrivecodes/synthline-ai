"""Core Pydantic models used across SynthLine AI."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class DefectType(StrEnum):
    """Supported defect generation types."""

    SCRATCH = "scratch"
    STAIN = "stain"
    DISCOLORATION = "discoloration"
    CRACK = "crack"
    PINHOLE = "pinhole"


class DatasetSplit(StrEnum):
    """Dataset partition types."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class ExportFormat(StrEnum):
    """Supported dataset export formats."""

    COCO = "coco"
    YOLO = "yolo"
    ALL = "all"


class SplitRatio(BaseModel):
    """Train/Val/Test partitioning ratios."""

    train: float = Field(default=0.7, ge=0.0, le=1.0)
    val: float = Field(default=0.2, ge=0.0, le=1.0)
    test: float = Field(default=0.1, ge=0.0, le=1.0)


class GenerationConfig(BaseModel):
    """Configuration for a single generation run."""

    seeds_dir: Path
    defect_type: DefectType = DefectType.SCRATCH
    count: int = Field(default=100, ge=1, le=10000)
    output_dir: Path
    random_seed: int = Field(default=42)
    severity: float = Field(default=0.5, ge=0.0, le=1.0)
    frequency: float = Field(default=1.0, ge=0.1, le=5.0)  # defect density / occurrences
    enable_split: bool = Field(default=False)
    split_ratio: SplitRatio = Field(default_factory=SplitRatio)
    export_format: ExportFormat = ExportFormat.COCO
    enable_variations: bool = Field(default=False)
    lighting_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    texture_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    geometry_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    sensor_intensity: float = Field(default=0.0, ge=0.0, le=1.0)


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
