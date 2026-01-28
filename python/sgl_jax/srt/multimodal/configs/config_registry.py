"""Registry for mapping model names to model configs."""

import logging
import os

from sgl_jax.srt.multimodal.configs.dits.wan_model_config import WanModelConfig
from sgl_jax.srt.multimodal.configs.qwen3_omni.qwen3_omni_vision_config import (
    Qwen3OmniMoeVisionConfig,
)
from sgl_jax.srt.multimodal.configs.vaes.wan_vae_config import WanVAEConfig

logger = logging.getLogger(__name__)


class DiffusionConfigRegistry:
    """Registry that maps model names to their diffusion model configs.

    The registry supports matching by:
    1. Exact model name match (e.g., "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    2. Model name suffix match (e.g., "Wan2.1-T2V-1.3B-Diffusers")
    3. Partial keyword match (e.g., "Wan2.1-T2V-14B")
    """

    # Wan2.1 14B model architecture params (different from 1.3B defaults)
    _WAN_14B_ARCH = {
        "num_layers": 40,
        "num_attention_heads": 40,
        "attention_head_dim": 128,
        "hidden_size": 5120,  # 40 * 128
        "ffn_dim": 13824,
    }

    # Model name -> config factory mapping
    # Each factory is a callable that returns a config instance
    _REGISTRY: dict[str, callable] = {
        # Wan2.1 T2V 1.3B (480P, flow_shift=3.0) - uses default WanModelConfig params
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": lambda: WanModelConfig(flow_shift=3.0),
        "Wan2.1-T2V-1.3B-Diffusers": lambda: WanModelConfig(flow_shift=3.0),
        # Wan2.1 T2V 14B (720P, flow_shift=5.0)
        "Wan-AI/Wan2.1-T2V-14B-Diffusers": lambda: WanModelConfig(
            flow_shift=5.0,
            **DiffusionConfigRegistry._WAN_14B_ARCH,
        ),
        "Wan2.1-T2V-14B-Diffusers": lambda: WanModelConfig(
            flow_shift=5.0,
            **DiffusionConfigRegistry._WAN_14B_ARCH,
        ),
        # Wan2.1 I2V 14B 480P
        "Wan2.1-I2V-14B-480P-Diffusers": lambda: WanModelConfig(
            flow_shift=3.0,
            image_dim=1280,
            added_kv_proj_dim=5120,
            **DiffusionConfigRegistry._WAN_14B_ARCH,
        ),
        # Wan2.1 I2V 14B 720P
        "Wan2.1-I2V-14B-720P-Diffusers": lambda: WanModelConfig(
            flow_shift=5.0,
            image_dim=1280,
            added_kv_proj_dim=5120,
            **DiffusionConfigRegistry._WAN_14B_ARCH,
        ),
        # Wan2.2 T2V A14B (MoE)
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers": lambda: WanModelConfig(
            flow_shift=12.0,
            boundary_ratio=0.875,
            **DiffusionConfigRegistry._WAN_14B_ARCH,
        ),
        "Wan2.2-T2V-A14B-Diffusers": lambda: WanModelConfig(
            flow_shift=12.0,
            boundary_ratio=0.875,
            **DiffusionConfigRegistry._WAN_14B_ARCH,
        ),
    }

    # Keyword patterns for fallback matching (order matters - more specific first)
    _KEYWORD_PATTERNS: list[tuple[str, callable]] = [
        # Wan2.2 A14B MoE pattern (check first as it's more specific)
        (
            "T2V-A14B",
            lambda: WanModelConfig(
                flow_shift=12.0,
                boundary_ratio=0.875,
                **DiffusionConfigRegistry._WAN_14B_ARCH,
            ),
        ),
        # 14B patterns
        (
            "T2V-14B",
            lambda: WanModelConfig(flow_shift=5.0, **DiffusionConfigRegistry._WAN_14B_ARCH),
        ),
        (
            "I2V-14B-720P",
            lambda: WanModelConfig(
                flow_shift=5.0,
                image_dim=1280,
                added_kv_proj_dim=5120,
                **DiffusionConfigRegistry._WAN_14B_ARCH,
            ),
        ),
        (
            "I2V-14B-480P",
            lambda: WanModelConfig(
                flow_shift=3.0,
                image_dim=1280,
                added_kv_proj_dim=5120,
                **DiffusionConfigRegistry._WAN_14B_ARCH,
            ),
        ),
        # 1.3B patterns (uses default WanModelConfig params)
        ("T2V-1.3B", lambda: WanModelConfig(flow_shift=3.0)),
        # Generic Wan2.1 fallback (default to 1.3B/480P)
        ("Wan2.1", lambda: WanModelConfig(flow_shift=3.0)),
    ]

    @classmethod
    def register(cls, model_name: str, config_factory: callable) -> None:
        """Register a new model name to config factory mapping.

        Args:
            model_name: The model name or path pattern to register.
            config_factory: A callable that returns a model config instance.
        """
        cls._REGISTRY[model_name] = config_factory
        logger.info("Registered diffusion config '%s'", model_name)

    @classmethod
    def get_config(cls, model_path: str) -> WanModelConfig:
        """Get the diffusion model config for a given model path.

        Args:
            model_path: The model path from server args (can be local path or HF repo ID).

        Returns:
            A model config instance configured for the specified model.

        Raises:
            ValueError: If no matching config is found for the model.
        """
        # Extract model name from path (handle both local paths and HF repo IDs)
        model_name = cls._extract_model_name(model_path)

        # Try exact match first
        config_factory = cls._REGISTRY.get(model_name)
        if config_factory:
            logger.debug("Found exact diffusion config match for model '%s'", model_name)
            return config_factory()

        # Try matching with full model_path (for HF-style repo IDs)
        config_factory = cls._REGISTRY.get(model_path)
        if config_factory:
            logger.debug("Found diffusion config match for full model path '%s'", model_path)
            return config_factory()

        # Try keyword pattern matching
        for keyword, factory in cls._KEYWORD_PATTERNS:
            if keyword in model_name or keyword in model_path:
                logger.debug(
                    "Found diffusion config keyword match '%s' for model '%s'", keyword, model_name
                )
                return factory()

        # No match found
        available_models = list(cls._REGISTRY.keys())
        raise ValueError(
            f"No diffusion config found for model '{model_path}'. "
            f"Available models: {available_models}. "
            f"You can register new models using DiffusionConfigRegistry.register()."
        )

    @classmethod
    def _extract_model_name(cls, model_path: str) -> str:
        """Extract the model name from a model path."""
        model_path = model_path.rstrip("/")
        return os.path.basename(model_path)

    @classmethod
    def list_registered_models(cls) -> list[str]:
        """List all registered model names."""
        return list(cls._REGISTRY.keys())


class VAEConfigRegistry:
    """Registry that maps model names to their VAE configs.

    The registry supports matching by:
    1. Exact model name match (e.g., "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    2. Model name suffix match (e.g., "Wan2.1-T2V-1.3B-Diffusers")
    3. Partial keyword match (e.g., "Wan2.1")
    """

    # Model name -> config factory mapping
    _REGISTRY: dict[str, callable] = {
        # Wan2.1 T2V 1.3B
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": lambda: WanVAEConfig(),
        "Wan2.1-T2V-1.3B-Diffusers": lambda: WanVAEConfig(),
        # Wan2.1 T2V 14B
        "Wan-AI/Wan2.1-T2V-14B-Diffusers": lambda: WanVAEConfig(),
        "Wan2.1-T2V-14B-Diffusers": lambda: WanVAEConfig(),
        # Wan2.1 I2V 14B 480P
        "Wan2.1-I2V-14B-480P-Diffusers": lambda: WanVAEConfig(),
        # Wan2.1 I2V 14B 720P
        "Wan2.1-I2V-14B-720P-Diffusers": lambda: WanVAEConfig(),
        # Wan2.2 T2V A14B (MoE)
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers": lambda: WanVAEConfig(is_residual=False),
        "Wan2.2-T2V-A14B-Diffusers": lambda: WanVAEConfig(is_residual=False),
    }

    # Keyword patterns for fallback matching (order matters - more specific first)
    _KEYWORD_PATTERNS: list[tuple[str, callable]] = [
        # Wan2.2 patterns
        ("Wan2.2", lambda: WanVAEConfig(is_residual=False)),
        # Wan2.1 patterns
        ("Wan2.1", lambda: WanVAEConfig()),
    ]

    @classmethod
    def register(cls, model_name: str, config_factory: callable) -> None:
        """Register a new model name to VAE config factory mapping."""
        cls._REGISTRY[model_name] = config_factory
        logger.info("Registered VAE config '%s'", model_name)

    @classmethod
    def get_config(cls, model_path: str) -> WanVAEConfig:
        """Get the VAE config for a given model path."""
        model_name = cls._extract_model_name(model_path)

        # Try exact match first
        config_factory = cls._REGISTRY.get(model_name)
        if config_factory:
            logger.debug("Found exact VAE config match for model '%s'", model_name)
            return config_factory()

        # Try matching with full model_path (for HF-style repo IDs)
        config_factory = cls._REGISTRY.get(model_path)
        if config_factory:
            logger.debug("Found VAE config match for full model path '%s'", model_path)
            return config_factory()

        # Try keyword pattern matching
        for keyword, factory in cls._KEYWORD_PATTERNS:
            if keyword in model_name or keyword in model_path:
                logger.debug(
                    "Found VAE config keyword match '%s' for model '%s'", keyword, model_name
                )
                return factory()

        # No match found
        available_models = list(cls._REGISTRY.keys())
        raise ValueError(
            f"No VAE config found for model '{model_path}'. "
            f"Available models: {available_models}. "
            f"You can register new models using VAEConfigRegistry.register()."
        )

    @classmethod
    def _extract_model_name(cls, model_path: str) -> str:
        """Extract the model name from a model path."""
        model_path = model_path.rstrip("/")
        return os.path.basename(model_path)

    @classmethod
    def list_registered_models(cls) -> list[str]:
        """List all registered model names."""
        return list(cls._REGISTRY.keys())


def get_diffusion_config(model_path: str) -> WanModelConfig:
    """Convenience function to get diffusion model config.

    Args:
        model_path: The model path from server args.

    Returns:
        A model config instance configured for the specified model.
    """
    return DiffusionConfigRegistry.get_config(model_path)


def get_vae_config(model_path: str) -> WanVAEConfig:
    """Convenience function to get VAE config.

    Args:
        model_path: The model path from server args.

    Returns:
        A VAE config instance configured for the specified model.
    """
    return VAEConfigRegistry.get_config(model_path)


class VisionEncoderConfigRegistry:
    """Registry that maps model names to their Vision Encoder configs.

    The registry supports matching by:
    1. Exact model name match (e.g., "Qwen/Qwen3-Omni-MoE")
    2. Model name suffix match (e.g., "Qwen3-Omni-MoE")
    3. Partial keyword match (e.g., "Qwen3-Omni")
    """

    # Model name -> config factory mapping
    _REGISTRY: dict[str, callable] = {
        # Qwen3 Omni MoE
        "Qwen/Qwen3-Omni-MoE": lambda: Qwen3OmniMoeVisionConfig(),
        "Qwen3-Omni-MoE": lambda: Qwen3OmniMoeVisionConfig(),
        "Qwen3-Omni": lambda: Qwen3OmniMoeVisionConfig(),
    }

    # Keyword patterns for fallback matching (order matters - more specific first)
    _KEYWORD_PATTERNS: list[tuple[str, callable]] = [
        # Qwen3 Omni patterns
        ("Qwen3-Omni-MoE", lambda: Qwen3OmniMoeVisionConfig()),
        ("Qwen3-Omni", lambda: Qwen3OmniMoeVisionConfig()),
        ("qwen3-omni", lambda: Qwen3OmniMoeVisionConfig()),
    ]

    @classmethod
    def register(cls, model_name: str, config_factory: callable) -> None:
        """Register a new model name to Vision Encoder config factory mapping."""
        cls._REGISTRY[model_name] = config_factory
        logger.info("Registered Vision Encoder config '%s'", model_name)

    @classmethod
    def get_config(cls, model_path: str) -> Qwen3OmniMoeVisionConfig:
        """Get the Vision Encoder config for a given model path."""
        model_name = cls._extract_model_name(model_path)

        # Try exact match first
        config_factory = cls._REGISTRY.get(model_name)
        if config_factory:
            logger.debug("Found exact Vision Encoder config match for model '%s'", model_name)
            return config_factory()

        # Try matching with full model_path (for HF-style repo IDs)
        config_factory = cls._REGISTRY.get(model_path)
        if config_factory:
            logger.debug("Found Vision Encoder config match for full model path '%s'", model_path)
            return config_factory()

        # Try keyword pattern matching
        for keyword, factory in cls._KEYWORD_PATTERNS:
            if keyword in model_name or keyword in model_path:
                logger.debug(
                    "Found Vision Encoder config keyword match '%s' for model '%s'",
                    keyword,
                    model_name,
                )
                return factory()

        # No match found
        available_models = list(cls._REGISTRY.keys())
        raise ValueError(
            f"No Vision Encoder config found for model '{model_path}'. "
            f"Available models: {available_models}. "
            f"You can register new models using VisionEncoderConfigRegistry.register()."
        )

    @classmethod
    def _extract_model_name(cls, model_path: str) -> str:
        """Extract the model name from a model path."""
        model_path = model_path.rstrip("/")
        return os.path.basename(model_path)

    @classmethod
    def list_registered_models(cls) -> list[str]:
        """List all registered model names."""
        return list(cls._REGISTRY.keys())


def get_vision_encoder_config(model_path: str) -> Qwen3OmniMoeVisionConfig:
    """Convenience function to get Vision Encoder config.

    Args:
        model_path: The model path from server args.

    Returns:
        A Vision Encoder config instance configured for the specified model.
    """
    return VisionEncoderConfigRegistry.get_config(model_path)
