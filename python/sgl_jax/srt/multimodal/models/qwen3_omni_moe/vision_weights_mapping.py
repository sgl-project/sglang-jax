"""
Weight mappings for Qwen3OmniMoe Vision Encoder.

Maps HuggingFace PyTorch checkpoint keys to JAX/Flax model paths.

Key transformations:
- PyTorch Linear: weight shape (out, in), transpose=True
- JAX Dense: kernel shape (in, out)
- Conv3D: kernel axes permutation required
- LayerNorm: weight -> scale, bias -> bias
"""


def create_vision_encoder_weight_mappings(config) -> dict:
    """
    Create weight mappings for Qwen3OmniMoe Vision Encoder.

    Args:
        config: Qwen3OmniMoeVisionConfig instance

    Returns:
        Dictionary mapping HF keys to JAX paths with transformation specs
    """
    # Lazy import to avoid fastapi dependency issues
    from sgl_jax.srt.utils.weight_utils import WeightMapping

    mappings = {}

    # ==================== Patch Embedding ====================
    # Conv3d weights: PyTorch (out_channels, in_channels, T, H, W)
    # JAX Conv: (T, H, W, in_channels, out_channels)
    # Need transpose_axes to reorder dimensions
    mappings["patch_embed.proj.weight"] = WeightMapping(
        target_path="patch_embed.proj.kernel",
        sharding=(None, None, None, None, None),
        transpose=False,
        transpose_axes=(2, 3, 4, 1, 0),  # (out, in, T, H, W) -> (T, H, W, in, out)
    )

    mappings["patch_embed.proj.bias"] = WeightMapping(
        target_path="patch_embed.proj.bias",
        sharding=(None,),
        transpose=False,
    )

    # ==================== Position Embedding ====================
    mappings["pos_embed.weight"] = WeightMapping(
        target_path="pos_embed.embedding",
        sharding=(None, None),
        transpose=False,
    )

    # ==================== Transformer Blocks ====================
    num_layers = config.depth

    for layer_idx in range(num_layers):
        prefix = f"blocks.{layer_idx}"
        target_prefix = f"blocks.{layer_idx}"

        # LayerNorm 1
        mappings[f"{prefix}.norm1.weight"] = WeightMapping(
            target_path=f"{target_prefix}.norm1.scale",
            sharding=(None,),
            transpose=False,
        )
        mappings[f"{prefix}.norm1.bias"] = WeightMapping(
            target_path=f"{target_prefix}.norm1.bias",
            sharding=(None,),
            transpose=False,
        )

        # Attention QKV (combined projection)
        # PyTorch: (hidden_size, hidden_size*3) needs transpose to (hidden_size*3, hidden_size)
        # Then JAX expects (hidden_size, hidden_size*3)
        mappings[f"{prefix}.attn.qkv.weight"] = WeightMapping(
            target_path=f"{target_prefix}.attn.qkv_proj.weight",
            sharding=(None, "tensor"),  # Shard on output dimension
            transpose=True,  # PyTorch (out, in) -> JAX (in, out)
        )
        mappings[f"{prefix}.attn.qkv.bias"] = WeightMapping(
            target_path=f"{target_prefix}.attn.qkv_proj.bias",
            sharding=("tensor",),  # Shard on feature dimension
            transpose=False,
        )

        # Attention Output Projection
        mappings[f"{prefix}.attn.proj.weight"] = WeightMapping(
            target_path=f"{target_prefix}.attn.o_proj.weight",
            sharding=("tensor", None),  # Shard on input dimension
            transpose=True,
        )
        mappings[f"{prefix}.attn.proj.bias"] = WeightMapping(
            target_path=f"{target_prefix}.attn.o_proj.bias",
            sharding=(None,),
            transpose=False,
        )

        # LayerNorm 2
        mappings[f"{prefix}.norm2.weight"] = WeightMapping(
            target_path=f"{target_prefix}.norm2.scale",
            sharding=(None,),
            transpose=False,
        )
        mappings[f"{prefix}.norm2.bias"] = WeightMapping(
            target_path=f"{target_prefix}.norm2.bias",
            sharding=(None,),
            transpose=False,
        )

        # MLP
        mappings[f"{prefix}.mlp.linear_fc1.weight"] = WeightMapping(
            target_path=f"{target_prefix}.mlp.fc1.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings[f"{prefix}.mlp.linear_fc1.bias"] = WeightMapping(
            target_path=f"{target_prefix}.mlp.fc1.bias",
            sharding=("tensor",),
            transpose=False,
        )

        mappings[f"{prefix}.mlp.linear_fc2.weight"] = WeightMapping(
            target_path=f"{target_prefix}.mlp.fc2.weight",
            sharding=("tensor", None),
            transpose=True,
        )
        mappings[f"{prefix}.mlp.linear_fc2.bias"] = WeightMapping(
            target_path=f"{target_prefix}.mlp.fc2.bias",
            sharding=(None,),
            transpose=False,
        )

    # ==================== Final Merger ====================
    # TP Strategy: mlp_fc1 column-wise, mlp_fc2 row-wise (same as VisionMLP)
    mappings["merger.ln_q.weight"] = WeightMapping(
        target_path="merger.ln_q.scale",
        sharding=(None,),
        transpose=False,
    )
    mappings["merger.ln_q.bias"] = WeightMapping(
        target_path="merger.ln_q.bias",
        sharding=(None,),
        transpose=False,
    )

    # Merger MLP: HuggingFace uses nn.ModuleList[Linear, GELU, Linear]
    # Index 0: First Linear (column-wise TP sharding)
    # Index 1: GELU (no parameters)
    # Index 2: Second Linear (row-wise TP sharding)
    mappings["merger.mlp.0.weight"] = WeightMapping(
        target_path="merger.mlp_fc1.weight",
        sharding=(None, "tensor"),  # TP: column-wise sharding
        transpose=True,
    )
    mappings["merger.mlp.0.bias"] = WeightMapping(
        target_path="merger.mlp_fc1.bias",
        sharding=("tensor",),  # TP: shard bias along output dimension
        transpose=False,
    )

    mappings["merger.mlp.2.weight"] = WeightMapping(
        target_path="merger.mlp_fc2.weight",
        sharding=("tensor", None),  # TP: row-wise sharding
        transpose=True,
    )
    mappings["merger.mlp.2.bias"] = WeightMapping(
        target_path="merger.mlp_fc2.bias",
        sharding=(None,),  # TP: bias not sharded (after all-reduce)
        transpose=False,
    )

    # ==================== Deepstack Mergers ====================
    # Same TP strategy as final merger
    deepstack_indexes = getattr(config, "deepstack_visual_indexes", [8, 16, 24])

    for merger_idx in range(len(deepstack_indexes)):
        prefix = f"merger_list.{merger_idx}"
        target_prefix = f"deepstack_mergers.{merger_idx}"

        mappings[f"{prefix}.ln_q.weight"] = WeightMapping(
            target_path=f"{target_prefix}.ln_q.scale",
            sharding=(None,),
            transpose=False,
        )
        mappings[f"{prefix}.ln_q.bias"] = WeightMapping(
            target_path=f"{target_prefix}.ln_q.bias",
            sharding=(None,),
            transpose=False,
        )

        # mlp_fc1: column-wise TP sharding
        mappings[f"{prefix}.mlp.0.weight"] = WeightMapping(
            target_path=f"{target_prefix}.mlp_fc1.weight",
            sharding=(None, "tensor"),  # TP: column-wise sharding
            transpose=True,
        )
        mappings[f"{prefix}.mlp.0.bias"] = WeightMapping(
            target_path=f"{target_prefix}.mlp_fc1.bias",
            sharding=("tensor",),  # TP: shard bias along output dimension
            transpose=False,
        )

        # mlp_fc2: row-wise TP sharding
        mappings[f"{prefix}.mlp.2.weight"] = WeightMapping(
            target_path=f"{target_prefix}.mlp_fc2.weight",
            sharding=("tensor", None),  # TP: row-wise sharding
            transpose=True,
        )
        mappings[f"{prefix}.mlp.2.bias"] = WeightMapping(
            target_path=f"{target_prefix}.mlp_fc2.bias",
            sharding=(None,),  # TP: bias not sharded (after all-reduce)
            transpose=False,
        )

    return mappings
