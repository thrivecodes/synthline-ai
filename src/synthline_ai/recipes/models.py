"""Declarative recipe data model for reproducible industrial defect synthesis."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from synthline_ai.config.models import (
    DefectType,
    ExportFormat,
    GenerationConfig,
    SplitRatio,
)


class GenerationRecipe(BaseModel):
    """Specification recipe for domain-specific defect generation."""

    name: str = Field(..., description="Unique slug for the recipe (e.g. automotive_stamping)")
    title: str = Field(..., description="Human-readable title")
    description: str = Field(default="", description="Domain explanation and visual context")
    domain: str = Field(default="general", description="Industry domain category")

    defect_type: DefectType = DefectType.SCRATCH
    severity: float = Field(default=0.5, ge=0.0, le=1.0)
    frequency: float = Field(default=1.0, ge=0.1, le=5.0)

    # Environmental & domain variations
    enable_variations: bool = Field(default=True)
    lighting_intensity: float = Field(default=0.2, ge=0.0, le=1.0)
    texture_intensity: float = Field(default=0.15, ge=0.0, le=1.0)
    geometry_intensity: float = Field(default=0.2, ge=0.0, le=1.0)
    sensor_intensity: float = Field(default=0.1, ge=0.0, le=1.0)

    # Multi-defect compound settings
    compound_defects: bool = Field(default=False)
    defects_per_image: int = Field(default=1, ge=1, le=5)

    # Workpiece boundary constraint
    auto_roi: bool = Field(default=False)

    # Dataset export & partitioning
    export_format: ExportFormat = ExportFormat.ALL
    enable_split: bool = Field(default=True)
    split_ratio: SplitRatio = Field(default_factory=SplitRatio)

    def to_config(
        self,
        seeds_dir: Path,
        output_dir: Path,
        count: int = 100,
        random_seed: int = 42,
    ) -> GenerationConfig:
        """Convert recipe into executable GenerationConfig."""
        return GenerationConfig(
            seeds_dir=seeds_dir,
            defect_type=self.defect_type,
            count=count,
            output_dir=output_dir,
            random_seed=random_seed,
            severity=self.severity,
            frequency=self.frequency,
            enable_split=self.enable_split,
            split_ratio=self.split_ratio,
            export_format=self.export_format,
            enable_variations=self.enable_variations,
            lighting_intensity=self.lighting_intensity,
            texture_intensity=self.texture_intensity,
            geometry_intensity=self.geometry_intensity,
            sensor_intensity=self.sensor_intensity,
            compound_defects=self.compound_defects,
            defects_per_image=self.defects_per_image,
            auto_roi=self.auto_roi,
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize recipe to JSON string."""
        return self.model_dump_json(indent=indent)

    def save(self, path: Path) -> Path:
        """Save recipe to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> GenerationRecipe:
        """Load recipe from JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Recipe file '{path}' does not exist.")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)
