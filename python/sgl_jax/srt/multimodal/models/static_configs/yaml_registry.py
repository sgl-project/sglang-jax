"""Registry for mapping model names to stage config YAML files."""

import logging
import os
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
    }

    # Keyword patterns for fallback matching (order matters - more specific first)
    _KEYWORD_PATTERNS: list[tuple[str, str]] = [
        ("Wan2.2", "wan2_2_stage_config.yaml"),
        ("Wan2.1", "wan2_1_stage_config.yaml"),
    ]

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
        for keyword, yaml_file in cls._KEYWORD_PATTERNS:
            if keyword in model_name or keyword in model_path:
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
