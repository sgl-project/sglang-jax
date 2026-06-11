"""Registry for mapping model names to stage config YAML files."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Get the directory where this file is located
_STATIC_CONFIGS_DIR = Path(__file__).parent


class StageConfigRegistry:
    """Registry that maps model names to their stage config YAML paths.

    The registry supports matching by:
    1. Exact model name match (e.g., "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    2. Model name suffix match (e.g., "Wan2.1-T2V-1.3B-Diffusers")
    3. Partial keyword match (e.g., "Wan2.1")
    """

    # Model name -> YAML filename mapping
    _REGISTRY: dict[str, str] = {
        # Wan2.1 series (single transformer)
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": "wan2_1_stage_config.yaml",
        "Wan2.1-T2V-1.3B-Diffusers": "wan2_1_stage_config.yaml",
        "Wan2.1-T2V-14B-Diffusers": "wan2_1_stage_config.yaml",
        "Wan2.1-I2V-14B-480P-Diffusers": "wan2_1_stage_config.yaml",
        "Wan2.1-I2V-14B-720P-Diffusers": "wan2_1_stage_config.yaml",
        # Wan2.2 series (MoE with dual transformers)
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "wan2_2_stage_config.yaml",
        "Wan2.2-T2V-A14B-Diffusers": "wan2_2_stage_config.yaml",
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers": "wan2_2_stage_config.yaml",
        "Wan2.2-I2V-A14B-Diffusers": "wan2_2_stage_config.yaml",
        # Qwen2.5-VL series
        "Qwen/Qwen2.5-VL-7B-Instruct": "qwen2_5_vl_stage_config.yaml",
        "Qwen2.5-VL-7B-Instruct": "qwen2_5_vl_stage_config.yaml",
        "Qwen/Qwen2.5-VL-32B-Instruct": "qwen2_5_vl_stage_config_tp4.yaml",
        "Qwen2.5-VL-32B-Instruct": "qwen2_5_vl_stage_config_tp4.yaml",
        "Qwen/Qwen2.5-VL-72B-Instruct": "qwen2_5_vl_stage_config_tp4.yaml",
        "Qwen2.5-VL-72B-Instruct": "qwen2_5_vl_stage_config_tp4.yaml",
        # qwen3-omni
        "Qwen/Qwen3-Omni-30B-A3B-Instruct": "qwen3_omni_stage_config.yaml",
        # MiMo-V2.5 omni
        "XiaomiMiMo/MiMo-V2.5": "mimo_v2_5_stage_config.yaml",
        "MiMo-V2.5": "mimo_v2_5_stage_config.yaml",
        # MiMo Audio series
        "XiaomiMiMo/MiMo-Audio-7B-Instruct": "mimo_audio_stage_config.yaml",
        "XiaomiMiMo/MiMo-Audio-7B-Base": "mimo_audio_stage_config.yaml",
        "black-forest-labs/FLUX.1-dev": "flux1_dev_stage_config.yaml",
    }

    # Keyword patterns for fallback matching (order matters - more specific first).
    # NOTE: MiMo-V2.5 is handled separately (see _match_mimo_v25_omni) so the broad
    # "mimo-v2.5" substring does not also route the text-only "MiMo-V2.5-Pro/Flash"
    # variants to the omni two-stage config.
    _KEYWORD_PATTERNS: list[tuple[str, str]] = [
        ("Wan2.2", "wan2_2_stage_config.yaml"),
        ("Wan2.1", "wan2_1_stage_config.yaml"),
        ("Qwen2.5-VL", "qwen2_5_vl_stage_config.yaml"),
        ("Qwen3-Omni", "qwen3_omni_stage_config.yaml"),
        ("MiMo-Audio-7B-Instruct", "mimo_audio_stage_config.yaml"),
        ("MiMo-Audio-7B-Base", "mimo_audio_stage_config.yaml"),
        ("FLUX.1-dev", "flux1_dev_stage_config.yaml"),
    ]

    # Text-only MiMo-V2.5 variants that must NOT route to the omni stage config.
    # Matched as whole tokens (word boundaries) so an incidental substring in an
    # unrelated path segment (e.g. ".../prod/...", ".../flashattn/...") does not
    # wrongly exclude the omni model (review R2-17).
    _MIMO_V25_TEXT_ONLY_MARKERS: tuple[str, ...] = ("pro", "flash")

    @classmethod
    def _match_mimo_v25_omni(cls, model_name: str, model_path: str) -> str | None:
        """Resolve MiMo-V2.5 omni by substring, excluding text-only variants."""
        haystack = f"{model_name} {model_path}".lower()
        if "mimo-v2.5" not in haystack:
            return None
        if any(
            re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", haystack)
            for marker in cls._MIMO_V25_TEXT_ONLY_MARKERS
        ):
            return None
        return "mimo_v2_5_stage_config.yaml"

    @classmethod
    def register(cls, model_name: str, yaml_filename: str) -> None:
        """Register a new model name to YAML mapping.

        Args:
            model_name: The model name or path pattern to register.
            yaml_filename: The YAML filename (relative to static_configs directory).
        """
        cls._REGISTRY[model_name] = yaml_filename
        logger.info("Registered model '%s' -> '%s'", model_name, yaml_filename)

    @classmethod
    def get_yaml_path(cls, model_path: str) -> Path:
        """Get the stage config YAML path for a given model path.

        Args:
            model_path: The model path from server args (can be local path or HF repo ID).

        Returns:
            Absolute path to the stage config YAML file.

        Raises:
            ValueError: If no matching config is found for the model.
        """
        # Extract model name from path (handle both local paths and HF repo IDs)
        model_name = cls._extract_model_name(model_path)

        # Try exact match first
        yaml_filename = cls._REGISTRY.get(model_name)
        if yaml_filename:
            logger.debug("Found exact match for model '%s'", model_name)
            return _STATIC_CONFIGS_DIR / yaml_filename

        # Try matching with full model_path (for HF-style repo IDs)
        yaml_filename = cls._REGISTRY.get(model_path)
        if yaml_filename:
            logger.debug("Found match for full model path '%s'", model_path)
            return _STATIC_CONFIGS_DIR / yaml_filename

        # Try keyword pattern matching
        model_name_lower = model_name.lower()
        model_path_lower = model_path.lower()

        # MiMo-V2.5 omni needs a variant-aware check (exclude Pro/Flash text-only)
        mimo_v25_yaml = cls._match_mimo_v25_omni(model_name, model_path)
        if mimo_v25_yaml:
            logger.debug("Found MiMo-V2.5 omni match for model '%s'", model_name)
            return _STATIC_CONFIGS_DIR / mimo_v25_yaml

        for keyword, yaml_file in cls._KEYWORD_PATTERNS:
            keyword_lower = keyword.lower()
            if keyword_lower in model_name_lower or keyword_lower in model_path_lower:
                logger.debug("Found keyword match '%s' for model '%s'", keyword, model_name)
                return _STATIC_CONFIGS_DIR / yaml_file

        # No match found
        available_models = list(cls._REGISTRY.keys())
        raise ValueError(
            f"No stage config found for model '{model_path}'. "
            f"Available models: {available_models}. "
            f"You can register new models using StageConfigRegistry.register()."
        )

    @classmethod
    def _extract_model_name(cls, model_path: str) -> str:
        """Extract the model name from a model path.

        Handles both:
        - Local paths: /models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers -> Wan2.1-T2V-1.3B-Diffusers
        - HF repo IDs: Wan-AI/Wan2.1-T2V-1.3B-Diffusers -> Wan2.1-T2V-1.3B-Diffusers
        """
        # Remove trailing slashes
        model_path = model_path.rstrip("/")

        # Get the basename (last component of path)
        basename = os.path.basename(model_path)

        return basename

    @classmethod
    def list_registered_models(cls) -> list[str]:
        """List all registered model names."""
        return list(cls._REGISTRY.keys())


def get_stage_config_path(model_path: str) -> str:
    """Convenience function to get stage config path as string.

    Args:
        model_path: The model path from server args.

    Returns:
        String path to the stage config YAML file.
    """
    return str(StageConfigRegistry.get_yaml_path(model_path))
