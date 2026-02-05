"""Registry for mapping model names to model configs."""

import contextlib
import json
import logging
import os

from sgl_jax.srt.multimodal.configs.audio.mimo_audio_config import MiMoAudioConfig
from sgl_jax.srt.multimodal.configs.audio.mimo_audio_backbone_config import MiMoAudioBackboneConfig
from sgl_jax.srt.multimodal.configs.dits.wan_model_config import WanModelConfig
from sgl_jax.srt.multimodal.configs.qwen_vl.qwen_2_5_vl_config import (
    QwenVLModelVitConfig,
)
from sgl_jax.srt.multimodal.configs.vaes.wan_vae_config import WanVAEConfig

logger = logging.getLogger(__name__)


_QWEN_VL_VISION_KEY_MAP = {
    "depth": "depth",
    "num_hidden_layers": "depth",
    "hidden_size": "hidden_size",
    "intermediate_size": "intermediate_size",
    "num_attention_heads": "num_heads",
    "num_heads": "num_heads",
    "in_channels": "in_channels",
    "patch_size": "patch_size",
    "spatial_merge_size": "spatial_merge_size",
    "temporal_patch_size": "temporal_patch_size",
    "tokens_per_second": "tokens_per_second",
    "window_size": "window_size",
    "out_hidden_size": "out_hidden_size",
    "output_hidden_size": "out_hidden_size",
    "rms_norm_eps": "rms_norm_eps",
    "initializer_range": "initializer_range",
    "fullatt_block_indexes": "fullatt_block_indexes",
    "full_attn_block_indexes": "fullatt_block_indexes",
}
_QWEN_VL_VISION_INT_FIELDS = {
    "depth",
    "hidden_size",
    "intermediate_size",
    "num_heads",
    "in_channels",
    "patch_size",
    "spatial_merge_size",
    "temporal_patch_size",
    "tokens_per_second",
    "window_size",
    "out_hidden_size",
}
_QWEN_VL_VISION_FLOAT_FIELDS = {
    "initializer_range",
    "rms_norm_eps",
}
_QWEN_VL_VISION_LIST_FIELDS = {
    "fullatt_block_indexes",
}


def _load_local_config_dict(model_path: str) -> dict | None:
    if not isinstance(model_path, str):
        return None
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to read config.json from %s: %s", config_path, exc)
        return None


def _apply_qwen_vl_vision_overrides(
    config: QwenVLModelVitConfig, model_path: str
) -> QwenVLModelVitConfig:
    config_dict = _load_local_config_dict(model_path)
    if not config_dict:
        return config

    vision_cfg = (
        config_dict.get("vision_config")
        or config_dict.get("vision_config_dict")
        or config_dict.get("vision_cfg")
    )
    if not isinstance(vision_cfg, dict):
        return config

    updated_fields: set[str] = set()
    for src_key, dst_attr in _QWEN_VL_VISION_KEY_MAP.items():
        if src_key not in vision_cfg:
            continue
        value = vision_cfg[src_key]
        if dst_attr in _QWEN_VL_VISION_INT_FIELDS:
            with contextlib.suppress(Exception):
                value = int(value)
        elif dst_attr in _QWEN_VL_VISION_FLOAT_FIELDS:
            with contextlib.suppress(Exception):
                value = float(value)
        elif dst_attr in _QWEN_VL_VISION_LIST_FIELDS and not isinstance(value, list):
            value = list(value)
        setattr(config, dst_attr, value)
        updated_fields.add(dst_attr)

    if (
        "out_hidden_size" not in updated_fields
        and (top_hidden_size := config_dict.get("hidden_size")) is not None
    ):
        with contextlib.suppress(Exception):
            top_hidden_size = int(top_hidden_size)
        config.out_hidden_size = top_hidden_size

    if updated_fields:
        logger.info("Loaded QwenVL vision config overrides from %s", model_path)
    return config


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


class QwenVLConfigRegistry:
    # Model name -> config factory mapping
    _REGISTRY: dict[str, callable] = {
        "Qwen/Qwen2.5-VL-3B-Instruct": lambda: QwenVLModelVitConfig(),
    }

    # Keyword patterns for fallback matching (order matters - more specific first)
    _KEYWORD_PATTERNS: list[tuple[str, callable]] = [
        ("Qwen2.5-VL", lambda: QwenVLModelVitConfig()),
    ]

    @classmethod
    def register(cls, model_name: str, config_factory: callable) -> None:
        cls._REGISTRY[model_name] = config_factory
        logger.info("Registered QwenVL config '%s'", model_name)

    @classmethod
    def get_config(cls, model_path: str) -> QwenVLModelVitConfig:
        model_name = cls._extract_model_name(model_path)

        # Try exact match first
        config_factory = cls._REGISTRY.get(model_name)
        if config_factory:
            logger.debug("Found exact QwenVL config match for model '%s'", model_name)
            return _apply_qwen_vl_vision_overrides(config_factory(), model_path)

        # Try matching with full model_path (for HF-style repo IDs)
        config_factory = cls._REGISTRY.get(model_path)
        if config_factory:
            logger.debug("Found QwenVL config match for full model path '%s'", model_path)
            return _apply_qwen_vl_vision_overrides(config_factory(), model_path)

        # Try keyword pattern matching
        for keyword, factory in cls._KEYWORD_PATTERNS:
            if keyword in model_name or keyword in model_path:
                logger.debug(
                    "Found QwenVL config keyword match '%s' for model '%s'", keyword, model_name
                )
                return _apply_qwen_vl_vision_overrides(factory(), model_path)

        # No match found
        available_models = list(cls._REGISTRY.keys())
        raise ValueError(
            f"No VAE config found for model '{model_path}'. "
            f"Available models: {available_models}. "
            f"You can register new models using QwenVLConfig.register()."
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


class AudioConfigRegistry:
    """Registry that maps model names to their audio tokenizer configs."""

    _REGISTRY: dict[str, callable] = {
        "xiaomi/MiMo-Audio-Tokenizer": lambda: MiMoAudioConfig(),
        "MiMo-Audio-Tokenizer": lambda: MiMoAudioConfig(),
    }

    _KEYWORD_PATTERNS: list[tuple[str, callable]] = [
        ("MiMo-Audio", lambda: MiMoAudioConfig()),
        ("mimo-audio", lambda: MiMoAudioConfig()),
    ]

    @classmethod
    def register(cls, model_name: str, config_factory: callable) -> None:
        cls._REGISTRY[model_name] = config_factory
        logger.info("Registered audio config '%s'", model_name)

    @classmethod
    def get_config(cls, model_path: str) -> MiMoAudioConfig:
        model_name = cls._extract_model_name(model_path)

        config_factory = cls._REGISTRY.get(model_name)
        if config_factory:
            logger.debug("Found exact audio config match for model '%s'", model_name)
            return config_factory()

        config_factory = cls._REGISTRY.get(model_path)
        if config_factory:
            logger.debug("Found audio config match for full model path '%s'", model_path)
            return config_factory()

        for keyword, factory in cls._KEYWORD_PATTERNS:
            if keyword.lower() in model_name.lower() or keyword.lower() in model_path.lower():
                logger.debug(
                    "Found audio config keyword match '%s' for model '%s'", keyword, model_name
                )
                return factory()

        available_models = list(cls._REGISTRY.keys())
        raise ValueError(
            f"No audio config found for model '{model_path}'. "
            f"Available models: {available_models}. "
            f"You can register new models using AudioConfigRegistry.register()."
        )

    @classmethod
    def _extract_model_name(cls, model_path: str) -> str:
        model_path = model_path.rstrip("/")
        return os.path.basename(model_path)

    @classmethod
    def list_registered_models(cls) -> list[str]:
        return list(cls._REGISTRY.keys())


def get_audio_config(model_path: str) -> MiMoAudioConfig:
    """Convenience function to get audio tokenizer config."""
    return AudioConfigRegistry.get_config(model_path)


class AudioBackboneConfigRegistry:
    """Registry that maps model names to their audio backbone (LLM) configs."""

    _REGISTRY: dict[str, callable] = {
        "xiaomi/MiMo-7B-RL": lambda: MiMoAudioBackboneConfig(),
        "MiMo-7B-RL": lambda: MiMoAudioBackboneConfig(),
        "xiaomi/MiMo-Audio": lambda: MiMoAudioBackboneConfig(),
        "MiMo-Audio": lambda: MiMoAudioBackboneConfig(),
        "XiaomiMiMo/MiMo-Audio-7B-Instruct": lambda: MiMoAudioBackboneConfig(),
        "MiMo-Audio-7B-Instruct": lambda: MiMoAudioBackboneConfig(),
    }

    _KEYWORD_PATTERNS: list[tuple[str, callable]] = [
        ("MiMo-Audio-7B-Instruct", lambda: MiMoAudioBackboneConfig()),
        ("MiMo-7B", lambda: MiMoAudioBackboneConfig()),
        ("mimo-7b", lambda: MiMoAudioBackboneConfig()),
        ("MiMo-Audio", lambda: MiMoAudioBackboneConfig()),
    ]

    @classmethod
    def register(cls, model_name: str, config_factory: callable) -> None:
        cls._REGISTRY[model_name] = config_factory
        logger.info("Registered audio backbone config '%s'", model_name)

    @classmethod
    def get_config(cls, model_path: str) -> MiMoAudioBackboneConfig:
        model_name = cls._extract_model_name(model_path)

        config_factory = cls._REGISTRY.get(model_name)
        if config_factory:
            logger.debug("Found exact audio backbone config match for model '%s'", model_name)
            return config_factory()

        config_factory = cls._REGISTRY.get(model_path)
        if config_factory:
            logger.debug("Found audio backbone config match for full model path '%s'", model_path)
            return config_factory()

        for keyword, factory in cls._KEYWORD_PATTERNS:
            if keyword.lower() in model_name.lower() or keyword.lower() in model_path.lower():
                logger.debug(
                    "Found audio backbone config keyword match '%s' for model '%s'",
                    keyword,
                    model_name,
                )
                return factory()

        available_models = list(cls._REGISTRY.keys())
        raise ValueError(
            f"No audio backbone config found for model '{model_path}'. "
            f"Available models: {available_models}. "
            f"You can register new models using AudioBackboneConfigRegistry.register()."
        )

    @classmethod
    def _extract_model_name(cls, model_path: str) -> str:
        model_path = model_path.rstrip("/")
        return os.path.basename(model_path)

    @classmethod
    def list_registered_models(cls) -> list[str]:
        return list(cls._REGISTRY.keys())


def get_audio_backbone_config(model_path: str) -> MiMoAudioBackboneConfig:
    """Convenience function to get audio backbone config."""
    return AudioBackboneConfigRegistry.get_config(model_path)


def get_qwen_vl_config(model_path: str) -> QwenVLModelVitConfig:
    """Convenience function to get Qwen vl config.

    Args:
        model_path: The model path from server args.

    Returns:
        A VAE config instance configured for the specified model.
    """
    return QwenVLConfigRegistry.get_config(model_path)
