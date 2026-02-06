"""Qwen3-VL model implementation for SGLang-JAX.

This module provides the Qwen3-VL multimodal model for high-performance
distributed inference on TPUs.

Components:
- Qwen3_VL_VisionModel: Vision encoder with DeepStack feature extraction
- Qwen3_VL_Generation: Text decoder with M-RoPE for conditional generation
"""

from sgl_jax.srt.multimodal.models.qwen3_VL.qwen3_vl_generation import (
    MRotaryEmbedding,
    Qwen3_VL_Generation,
    Qwen3_VL_Model,
)
from sgl_jax.srt.multimodal.models.qwen3_VL.qwen3_vl_vit import (
    Qwen3_VL_VisionModel,
    Qwen3_VL_VisionTransformer,
    Qwen3_VLImageInputs,
    Qwen3_VLVisionAttention,
    Qwen3_VLVisionBlock,
    Qwen3_VLVisionMLP,
    Qwen3_VLVisionPatchEmbed,
    Qwen3_VLVisionPatchMerger,
    Qwen3_VLVisionRotaryEmbedding,
)

__all__ = [
    # Vision components
    "Qwen3_VL_VisionModel",
    "Qwen3_VL_VisionTransformer",
    "Qwen3_VLVisionPatchEmbed",
    "Qwen3_VLVisionRotaryEmbedding",
    "Qwen3_VLVisionMLP",
    "Qwen3_VLVisionAttention",
    "Qwen3_VLVisionBlock",
    "Qwen3_VLVisionPatchMerger",
    "Qwen3_VLImageInputs",
    # Generation components
    "MRotaryEmbedding",
    "Qwen3_VL_Model",
    "Qwen3_VL_Generation",
]
