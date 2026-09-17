"""Project and generation run data models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from synthline_ai.config.models import (
    DefectType,
    ExportFormat,
    ImageInfo,
    QualityWarning,
    SplitRatio,
)


class ProjectCreate(BaseModel):
    """Payload to create a new visual inspection project."""

    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field(default="")


class Project(BaseModel):
    """A project containing seed images and generation runs."""

    id: str
    name: str
    description: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    seeds_count: int = 0
    runs_count: int = 0


class RunCreate(BaseModel):
    """Payload to launch a generation run in a project."""

    defect_type: DefectType = DefectType.SCRATCH
    count: int = Field(default=50, ge=1, le=10000)
    random_seed: int = Field(default=42)
    severity: float = Field(default=0.5, ge=0.0, le=1.0)
    frequency: float = Field(default=1.0, ge=0.1, le=5.0)
    enable_split: bool = Field(default=False)
    split_ratio: SplitRatio = Field(default_factory=SplitRatio)
    export_format: ExportFormat = ExportFormat.COCO
    enable_variations: bool = Field(default=False)
    lighting_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    texture_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    geometry_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    sensor_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    compound_defects: bool = Field(default=False)
    defects_per_image: int = Field(default=1, ge=1, le=5)


class Run(BaseModel):
    """Metadata for a completed or active generation run."""

    id: str
    project_id: str
    defect_type: str
    count: int
    severity: float
    frequency: float
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "completed"  # pending, completed, failed
    output_dir: str
    statistics: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)


class ProjectDetail(Project):
    """Full details of a project including seed info and run history."""

    seeds: list[ImageInfo] = Field(default_factory=list)
    seed_warnings: list[QualityWarning] = Field(default_factory=list)
    runs: list[Run] = Field(default_factory=list)
