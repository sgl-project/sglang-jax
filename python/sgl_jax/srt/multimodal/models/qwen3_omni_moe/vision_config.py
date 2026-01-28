# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0

"""Qwen3OmniMoe Vision Encoder configuration."""

from dataclasses import dataclass, field


@dataclass
class Qwen3OmniMoeVisionConfig:
    """
    Configuration for Qwen3OmniMoe Vision Encoder.

    Args:
        depth: Number of transformer layers
        hidden_size: Hidden dimension size
        hidden_act: Activation function name (e.g., "gelu", "gelu_pytorch_tanh")
        intermediate_size: FFN intermediate dimension
        num_heads: Number of attention heads
        in_channels: Number of input channels (3 for RGB)
        patch_size: Spatial patch size
        spatial_merge_size: Spatial merge factor for patch merging
        temporal_patch_size: Temporal patch size for video
        out_hidden_size: Output dimension after patch merger
        num_position_embeddings: Number of learnable position embeddings
        deepstack_visual_indexes: Layer indices to extract intermediate features
        rope_theta: Base for rotary position embedding
        layer_norm_eps: Epsilon for layer normalization
        dtype: Data type for model parameters
    """

    # Model architecture
    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"  # Match PyTorch default for Qwen3OmniMoe
    intermediate_size: int = 4304
    num_heads: int = 16

    # Input configuration
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2

    # Output configuration
    out_hidden_size: int = 3584  # Match language model dimension
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: list[int] = field(default_factory=lambda: [8, 16, 24])

    # Position embedding
    rope_theta: float = 10000.0

    # Normalization
    layer_norm_eps: float = 1e-6

    # Computation
    dtype: str = "bfloat16"

    @property
    def head_dim(self) -> int:
        """Attention head dimension"""
        return self.hidden_size // self.num_heads

    @property
    def num_grid_per_side(self) -> int:
        """Grid size per side for position embeddings"""
        return int(self.num_position_embeddings**0.5)

    def __post_init__(self):
        """Validate configuration"""
        assert (
            self.hidden_size % self.num_heads == 0
        ), f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"

        assert len(self.deepstack_visual_indexes) > 0, "deepstack_visual_indexes must not be empty"

        for idx in self.deepstack_visual_indexes:
            assert (
                0 <= idx < self.depth
            ), f"deepstack_visual_indexes contains invalid layer index: {idx}"
