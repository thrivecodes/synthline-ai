"""Domain recipes and presets for industrial visual inspection."""

from synthline_ai.recipes.models import GenerationRecipe
from synthline_ai.recipes.presets import IndustryPreset, get_preset, list_presets

__all__ = [
    "GenerationRecipe",
    "IndustryPreset",
    "get_preset",
    "list_presets",
]
