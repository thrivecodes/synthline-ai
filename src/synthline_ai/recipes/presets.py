"""Pre-configured industrial domain recipes for quality control benchmarks."""

from __future__ import annotations

from enum import StrEnum

from synthline_ai.config.models import DefectType, ExportFormat, SplitRatio
from synthline_ai.recipes.models import GenerationRecipe


class IndustryPreset(StrEnum):
    """Supported industry domain presets."""

    AUTOMOTIVE_STAMPING = "automotive_stamping"
    SEMICONDUCTOR_WAFER = "semiconductor_wafer"
    PCB_ELECTRONICS = "pcb_electronics"
    PHARMACEUTICAL_PACKAGING = "pharmaceutical_packaging"
    TEXTILE_FABRIC = "textile_fabric"
    GLASS_OPTICS = "glass_optics"


_PRESETS: dict[str, GenerationRecipe] = {
    IndustryPreset.AUTOMOTIVE_STAMPING.value: GenerationRecipe(
        name=IndustryPreset.AUTOMOTIVE_STAMPING.value,
        title="Automotive Sheet Metal Stamping",
        description=(
            "Die-forming scratches, stamping tool abrasions, and sheet-metal dents with "
            "directional illumination highlights and workpiece boundary containment."
        ),
        domain="automotive",
        defect_type=DefectType.MIXED,
        severity=0.65,
        frequency=1.2,
        enable_variations=True,
        lighting_intensity=0.35,
        texture_intensity=0.15,
        geometry_intensity=0.25,
        sensor_intensity=0.12,
        compound_defects=True,
        defects_per_image=2,
        auto_roi=True,
        export_format=ExportFormat.ALL,
        enable_split=True,
        split_ratio=SplitRatio(train=0.7, val=0.2, test=0.1),
    ),
    IndustryPreset.SEMICONDUCTOR_WAFER.value: GenerationRecipe(
        name=IndustryPreset.SEMICONDUCTOR_WAFER.value,
        title="Semiconductor Wafer Inspection",
        description=(
            "Micro-fracture cracks, etching pinholes, and chemical vapor discoloration rings "
            "simulating cleanroom optical inspection."
        ),
        domain="semiconductor",
        defect_type=DefectType.MIXED,
        severity=0.40,
        frequency=0.8,
        enable_variations=True,
        lighting_intensity=0.10,
        texture_intensity=0.10,
        geometry_intensity=0.10,
        sensor_intensity=0.18,
        compound_defects=False,
        defects_per_image=1,
        auto_roi=True,
        export_format=ExportFormat.ALL,
        enable_split=True,
        split_ratio=SplitRatio(train=0.7, val=0.2, test=0.1),
    ),
    IndustryPreset.PCB_ELECTRONICS.value: GenerationRecipe(
        name=IndustryPreset.PCB_ELECTRONICS.value,
        title="PCB & SMT Surface Mount",
        description=(
            "Hairline trace cracks, solder voids, and chemical flux staining on electronic "
            "substrates with multi-defect compound presence."
        ),
        domain="electronics",
        defect_type=DefectType.MIXED,
        severity=0.50,
        frequency=1.0,
        enable_variations=True,
        lighting_intensity=0.20,
        texture_intensity=0.15,
        geometry_intensity=0.15,
        sensor_intensity=0.12,
        compound_defects=True,
        defects_per_image=2,
        auto_roi=True,
        export_format=ExportFormat.ALL,
        enable_split=True,
        split_ratio=SplitRatio(train=0.7, val=0.2, test=0.1),
    ),
    IndustryPreset.PHARMACEUTICAL_PACKAGING.value: GenerationRecipe(
        name=IndustryPreset.PHARMACEUTICAL_PACKAGING.value,
        title="Pharmaceutical Packaging & Blister Seals",
        description=(
            "Blister foil pinholes, crimping stress cracks, and fluid residue contamination "
            "for sterile pharmaceutical container inspection."
        ),
        domain="pharmaceutical",
        defect_type=DefectType.MIXED,
        severity=0.55,
        frequency=0.9,
        enable_variations=True,
        lighting_intensity=0.25,
        texture_intensity=0.10,
        geometry_intensity=0.20,
        sensor_intensity=0.10,
        compound_defects=True,
        defects_per_image=2,
        auto_roi=True,
        export_format=ExportFormat.ALL,
        enable_split=True,
        split_ratio=SplitRatio(train=0.7, val=0.2, test=0.1),
    ),
    IndustryPreset.TEXTILE_FABRIC.value: GenerationRecipe(
        name=IndustryPreset.TEXTILE_FABRIC.value,
        title="Textile & Woven Fabric Inspection",
        description=(
            "Yarn abrasions, oil droplet stains, and localized dye discoloration patches "
            "across high-frequency woven textures."
        ),
        domain="textiles",
        defect_type=DefectType.MIXED,
        severity=0.60,
        frequency=1.5,
        enable_variations=True,
        lighting_intensity=0.20,
        texture_intensity=0.40,
        geometry_intensity=0.25,
        sensor_intensity=0.10,
        compound_defects=False,
        defects_per_image=1,
        auto_roi=False,
        export_format=ExportFormat.ALL,
        enable_split=True,
        split_ratio=SplitRatio(train=0.7, val=0.2, test=0.1),
    ),
    IndustryPreset.GLASS_OPTICS.value: GenerationRecipe(
        name=IndustryPreset.GLASS_OPTICS.value,
        title="Glass & Optical Components",
        description=(
            "Surface scratches, micro-pits, edge chipping, and specular refraction anomalies "
            "on transparent and optical glass substrates."
        ),
        domain="optics",
        defect_type=DefectType.MIXED,
        severity=0.45,
        frequency=0.8,
        enable_variations=True,
        lighting_intensity=0.30,
        texture_intensity=0.08,
        geometry_intensity=0.15,
        sensor_intensity=0.15,
        compound_defects=True,
        defects_per_image=2,
        auto_roi=True,
        export_format=ExportFormat.ALL,
        enable_split=True,
        split_ratio=SplitRatio(train=0.7, val=0.2, test=0.1),
    ),
}


def list_presets() -> list[GenerationRecipe]:
    """Retrieve all available industry domain presets."""
    return list(_PRESETS.values())


def get_preset(name: str) -> GenerationRecipe:
    """Retrieve a specific preset recipe by name.

    Raises:
        KeyError: If preset name is not recognized.
    """
    key = name.lower().strip()
    if key not in _PRESETS:
        available = list(_PRESETS.keys())
        raise KeyError(f"Unknown preset '{name}'. Available presets: {available}")
    return _PRESETS[key]
