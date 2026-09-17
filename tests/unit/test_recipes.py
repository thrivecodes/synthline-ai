"""Unit tests for declarative recipe models and industry presets."""

from __future__ import annotations

from pathlib import Path

import pytest

from synthline_ai.config.models import DefectType, ExportFormat
from synthline_ai.recipes.models import GenerationRecipe
from synthline_ai.recipes.presets import IndustryPreset, get_preset, list_presets


def test_list_presets() -> None:
    """Verify all defined presets are returned."""
    presets = list_presets()
    assert len(presets) == 6
    names = {p.name for p in presets}
    assert "automotive_stamping" in names
    assert "semiconductor_wafer" in names
    assert "pcb_electronics" in names
    assert "pharmaceutical_packaging" in names
    assert "textile_fabric" in names
    assert "glass_optics" in names


def test_get_preset_success() -> None:
    """Verify getting presets by name is case-insensitive and returns expected recipe."""
    recipe = get_preset("automotive_stamping")
    assert recipe.name == "automotive_stamping"
    assert recipe.domain == "automotive"
    assert recipe.auto_roi is True
    assert recipe.compound_defects is True

    # Case-insensitive
    recipe_upper = get_preset("AUTOMOTIVE_STAMPING")
    assert recipe_upper.name == "automotive_stamping"


def test_get_preset_unknown() -> None:
    """Verify unknown preset name raises KeyError."""
    with pytest.raises(KeyError, match="Unknown preset 'invalid_preset'"):
        get_preset("invalid_preset")


def test_recipe_to_config(tmp_path: Path) -> None:
    """Verify GenerationRecipe correctly generates an executable GenerationConfig."""
    recipe = get_preset("pcb_electronics")
    seeds_dir = tmp_path / "seeds"
    output_dir = tmp_path / "output"

    config = recipe.to_config(
        seeds_dir=seeds_dir,
        output_dir=output_dir,
        count=50,
        random_seed=123,
    )

    assert config.seeds_dir == seeds_dir
    assert config.output_dir == output_dir
    assert config.count == 50
    assert config.random_seed == 123
    assert config.defect_type == DefectType.MIXED
    assert config.compound_defects is True
    assert config.auto_roi is True
    assert config.export_format == ExportFormat.ALL


def test_recipe_json_serialization(tmp_path: Path) -> None:
    """Verify recipe saving to file and loading from file maintains fidelity."""
    recipe = GenerationRecipe(
        name="custom_inspection",
        title="Custom Foundry Inspection",
        description="Custom thermal stress crack recipe",
        domain="metallurgy",
        defect_type=DefectType.CRACK,
        severity=0.75,
        frequency=2.0,
        enable_variations=True,
        lighting_intensity=0.4,
        texture_intensity=0.3,
        geometry_intensity=0.2,
        sensor_intensity=0.15,
        compound_defects=False,
        defects_per_image=1,
        auto_roi=True,
    )

    recipe_file = tmp_path / "custom_recipe.json"
    saved_path = recipe.save(recipe_file)
    assert saved_path.exists()

    loaded = GenerationRecipe.load(recipe_file)
    assert loaded.name == recipe.name
    assert loaded.title == recipe.title
    assert loaded.defect_type == DefectType.CRACK
    assert loaded.severity == 0.75
    assert loaded.auto_roi is True


def test_recipe_load_nonexistent_raises(tmp_path: Path) -> None:
    """Verify loading from nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        GenerationRecipe.load(tmp_path / "missing.json")


def test_industry_preset_enum_values() -> None:
    """Verify IndustryPreset enum values align with string values."""
    assert IndustryPreset.AUTOMOTIVE_STAMPING == "automotive_stamping"
    assert IndustryPreset.SEMICONDUCTOR_WAFER == "semiconductor_wafer"
    assert IndustryPreset.PCB_ELECTRONICS == "pcb_electronics"
