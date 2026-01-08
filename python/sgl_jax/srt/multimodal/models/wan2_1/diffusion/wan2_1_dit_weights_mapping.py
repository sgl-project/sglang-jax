from sgl_jax.srt.utils.weight_utils import WeightMapping


def to_mappings(num_layers: int = 30) -> dict[str, WeightMapping]:
    """
    Create weight mappings from HuggingFace WAN 2.1 DiT weights to JAX model.

    HF weight format -> JAX model path

    Args:
        num_layers: Number of transformer blocks (default 30 for WAN 2.1 1.3B)
    """
    mappings: dict[str, WeightMapping] = {}

    # ==========================================================================
    # Patch Embedding
    # ==========================================================================
    # HF: patch_embedding is a Conv3d with kernel_size=patch_size
    # JAX: patch_embedding.proj is nnx.Conv
    mappings["patch_embedding.weight"] = WeightMapping(
        target_path="patch_embedding.proj.kernel",
        sharding=(None, None),
        transpose_axes=(2, 3, 4, 1, 0),  # PyTorch OIDHW -> JAX DHWIO
    )
    mappings["patch_embedding.bias"] = WeightMapping(
        target_path="patch_embedding.proj.bias",
        sharding=(None,),
    )

    # ==========================================================================
    # Condition Embedder
    # ==========================================================================
    # Time Embedder (TimestepEmbedder -> MLP with 2 linear layers)
    mappings["condition_embedder.time_embedder.linear_1.weight"] = WeightMapping(
        target_path="condition_embedder.time_embedder.mlp.fc_in.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["condition_embedder.time_embedder.linear_1.bias"] = WeightMapping(
        target_path="condition_embedder.time_embedder.mlp.fc_in.bias",
        sharding=(None,),
    )
    mappings["condition_embedder.time_embedder.linear_2.weight"] = WeightMapping(
        target_path="condition_embedder.time_embedder.mlp.fc_out.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["condition_embedder.time_embedder.linear_2.bias"] = WeightMapping(
        target_path="condition_embedder.time_embedder.mlp.fc_out.bias",
        sharding=(None,),
    )

    # Time Modulation (ModulateProjection -> linear)
    mappings["condition_embedder.time_proj.weight"] = WeightMapping(
        target_path="condition_embedder.time_modulation.linear.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["condition_embedder.time_proj.bias"] = WeightMapping(
        target_path="condition_embedder.time_modulation.linear.bias",
        sharding=(None,),
    )

    # Text Embedder (MLP with 2 linear layers)
    mappings["condition_embedder.text_embedder.linear_1.weight"] = WeightMapping(
        target_path="condition_embedder.text_embedder.fc_in.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["condition_embedder.text_embedder.linear_1.bias"] = WeightMapping(
        target_path="condition_embedder.text_embedder.fc_in.bias",
        sharding=(None,),
    )
    mappings["condition_embedder.text_embedder.linear_2.weight"] = WeightMapping(
        target_path="condition_embedder.text_embedder.fc_out.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["condition_embedder.text_embedder.linear_2.bias"] = WeightMapping(
        target_path="condition_embedder.text_embedder.fc_out.bias",
        sharding=(None,),
    )

    # ==========================================================================
    # Transformer Blocks (using wildcard pattern)
    # ==========================================================================
    # Self-Attention (attn1 in HF -> to_q/to_k/to_v/to_out in JAX block)
    mappings["blocks.*.attn1.to_q.weight"] = WeightMapping(
        target_path="blocks.*.to_q.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["blocks.*.attn1.to_q.bias"] = WeightMapping(
        target_path="blocks.*.to_q.bias",
        sharding=(None,),
    )
    mappings["blocks.*.attn1.to_k.weight"] = WeightMapping(
        target_path="blocks.*.to_k.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["blocks.*.attn1.to_k.bias"] = WeightMapping(
        target_path="blocks.*.to_k.bias",
        sharding=(None,),
    )
    mappings["blocks.*.attn1.to_v.weight"] = WeightMapping(
        target_path="blocks.*.to_v.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["blocks.*.attn1.to_v.bias"] = WeightMapping(
        target_path="blocks.*.to_v.bias",
        sharding=(None,),
    )
    # HF has to_out.0, JAX has to_out directly
    mappings["blocks.*.attn1.to_out.0.weight"] = WeightMapping(
        target_path="blocks.*.to_out.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["blocks.*.attn1.to_out.0.bias"] = WeightMapping(
        target_path="blocks.*.to_out.bias",
        sharding=(None,),
    )

    # Self-Attention QK Norm (RMSNorm)
    mappings["blocks.*.attn1.norm_q.weight"] = WeightMapping(
        target_path="blocks.*.norm_q.scale",
        sharding=(None,),
    )
    mappings["blocks.*.attn1.norm_k.weight"] = WeightMapping(
        target_path="blocks.*.norm_k.scale",
        sharding=(None,),
    )

    # Cross-Attention (attn2 in HF -> attn2.to_q/to_k/to_v/to_out in JAX)
    mappings["blocks.*.attn2.to_q.weight"] = WeightMapping(
        target_path="blocks.*.attn2.to_q.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["blocks.*.attn2.to_q.bias"] = WeightMapping(
        target_path="blocks.*.attn2.to_q.bias",
        sharding=(None,),
    )
    mappings["blocks.*.attn2.to_k.weight"] = WeightMapping(
        target_path="blocks.*.attn2.to_k.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["blocks.*.attn2.to_k.bias"] = WeightMapping(
        target_path="blocks.*.attn2.to_k.bias",
        sharding=(None,),
    )
    mappings["blocks.*.attn2.to_v.weight"] = WeightMapping(
        target_path="blocks.*.attn2.to_v.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["blocks.*.attn2.to_v.bias"] = WeightMapping(
        target_path="blocks.*.attn2.to_v.bias",
        sharding=(None,),
    )
    mappings["blocks.*.attn2.to_out.0.weight"] = WeightMapping(
        target_path="blocks.*.attn2.to_out.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["blocks.*.attn2.to_out.0.bias"] = WeightMapping(
        target_path="blocks.*.attn2.to_out.bias",
        sharding=(None,),
    )

    # Cross-Attention QK Norm
    mappings["blocks.*.attn2.norm_q.weight"] = WeightMapping(
        target_path="blocks.*.attn2.norm_q.scale",
        sharding=(None,),
    )
    mappings["blocks.*.attn2.norm_k.weight"] = WeightMapping(
        target_path="blocks.*.attn2.norm_k.scale",
        sharding=(None,),
    )

    # FFN (MLP)
    # HF: ffn.net.0.proj (GELU activation layer) -> JAX: ffn.fc_in
    mappings["blocks.*.ffn.net.0.proj.weight"] = WeightMapping(
        target_path="blocks.*.ffn.fc_in.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["blocks.*.ffn.net.0.proj.bias"] = WeightMapping(
        target_path="blocks.*.ffn.fc_in.bias",
        sharding=(None,),
    )
    # HF: ffn.net.2 (output linear) -> JAX: ffn.fc_out
    mappings["blocks.*.ffn.net.2.weight"] = WeightMapping(
        target_path="blocks.*.ffn.fc_out.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["blocks.*.ffn.net.2.bias"] = WeightMapping(
        target_path="blocks.*.ffn.fc_out.bias",
        sharding=(None,),
    )

    # Self-Attention Residual Norm (norm2 in HF -> self_attn_residual_norm in JAX)
    mappings["blocks.*.norm2.weight"] = WeightMapping(
        target_path="blocks.*.self_attn_residual_norm.norm.scale",
        sharding=(None,),
    )
    mappings["blocks.*.norm2.bias"] = WeightMapping(
        target_path="blocks.*.self_attn_residual_norm.norm.bias",
        sharding=(None,),
    )

    # Scale Shift Table
    mappings["blocks.*.scale_shift_table"] = WeightMapping(
        target_path="blocks.*.scale_shift_table",
        sharding=(None, None, None),
    )

    # ==========================================================================
    # Output Layers
    # ==========================================================================
    # Output projection
    mappings["proj_out.weight"] = WeightMapping(
        target_path="proj_out.kernel",
        sharding=(None, None),
        transpose=True,
    )
    mappings["proj_out.bias"] = WeightMapping(
        target_path="proj_out.bias",
        sharding=(None,),
    )

    # Global scale shift table
    mappings["scale_shift_table"] = WeightMapping(
        target_path="scale_shift_table",
        sharding=(None, None, None),
    )

    return mappings


def to_i2v_mappings(num_layers: int = 30) -> dict[str, WeightMapping]:
    """
    Create weight mappings for WAN 2.1 I2V (Image-to-Video) model.
    Includes additional mappings for image embedder and I2V cross-attention.

    Args:
        num_layers: Number of transformer blocks
    """
    # Start with base T2V mappings
    mappings = to_mappings(num_layers)

    # ==========================================================================
    # Image Embedder (only for I2V)
    # ==========================================================================
    mappings["condition_embedder.image_embedder.norm1.weight"] = WeightMapping(
        target_path="condition_embedder.image_embedder.norm1.scale",
        sharding=(None,),
    )
    mappings["condition_embedder.image_embedder.norm1.bias"] = WeightMapping(
        target_path="condition_embedder.image_embedder.norm1.bias",
        sharding=(None,),
    )
    mappings["condition_embedder.image_embedder.ff.fc_in.weight"] = WeightMapping(
        target_path="condition_embedder.image_embedder.ff.fc_in.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["condition_embedder.image_embedder.ff.fc_in.bias"] = WeightMapping(
        target_path="condition_embedder.image_embedder.ff.fc_in.bias",
        sharding=(None,),
    )
    mappings["condition_embedder.image_embedder.ff.fc_out.weight"] = WeightMapping(
        target_path="condition_embedder.image_embedder.ff.fc_out.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["condition_embedder.image_embedder.ff.fc_out.bias"] = WeightMapping(
        target_path="condition_embedder.image_embedder.ff.fc_out.bias",
        sharding=(None,),
    )
    mappings["condition_embedder.image_embedder.norm2.weight"] = WeightMapping(
        target_path="condition_embedder.image_embedder.norm2.scale",
        sharding=(None,),
    )
    mappings["condition_embedder.image_embedder.norm2.bias"] = WeightMapping(
        target_path="condition_embedder.image_embedder.norm2.bias",
        sharding=(None,),
    )

    # ==========================================================================
    # I2V Cross-Attention additional projections (add_k_proj, add_v_proj)
    # ==========================================================================
    mappings["blocks.*.attn2.add_k_proj.weight"] = WeightMapping(
        target_path="blocks.*.attn2.add_k_proj.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["blocks.*.attn2.add_k_proj.bias"] = WeightMapping(
        target_path="blocks.*.attn2.add_k_proj.bias",
        sharding=(None,),
    )
    mappings["blocks.*.attn2.add_v_proj.weight"] = WeightMapping(
        target_path="blocks.*.attn2.add_v_proj.weight",
        sharding=(None, None),
        transpose=True,
    )
    mappings["blocks.*.attn2.add_v_proj.bias"] = WeightMapping(
        target_path="blocks.*.attn2.add_v_proj.bias",
        sharding=(None,),
    )

    # I2V additional norms
    mappings["blocks.*.attn2.norm_added_k.weight"] = WeightMapping(
        target_path="blocks.*.attn2.norm_added_k.scale",
        sharding=(None,),
    )
    mappings["blocks.*.attn2.norm_added_q.weight"] = WeightMapping(
        target_path="blocks.*.attn2.norm_added_q.scale",
        sharding=(None,),
    )

    return mappings
