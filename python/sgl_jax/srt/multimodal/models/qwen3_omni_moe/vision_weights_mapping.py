"""Weight mappings for Qwen3OmniMoe Vision Encoder."""


def create_vision_encoder_weight_mappings(config) -> dict:
    """
    Create weight mappings for Qwen3OmniMoe Vision Encoder.

    Args:
        config: Qwen3OmniMoeVisionConfig instance

    Returns:
        Dictionary mapping HF keys to JAX paths with transformation specs
    """
    from sgl_jax.srt.utils.weight_utils import WeightMapping

    mappings = {}

    # Helper functions to reduce repetition
    def add_linear(src: str, dst: str, tp_col: bool = False, tp_row: bool = False):
        """Add linear layer mapping with optional TP sharding."""
        w_sharding = (None, "tensor") if tp_col else ("tensor", None) if tp_row else (None, None)
        b_sharding = ("tensor",) if tp_col else (None,)
        mappings[f"{src}.weight"] = WeightMapping(
            target_path=f"{dst}.weight", sharding=w_sharding, transpose=True
        )
        mappings[f"{src}.bias"] = WeightMapping(
            target_path=f"{dst}.bias", sharding=b_sharding, transpose=False
        )

    def add_layernorm(src: str, dst: str):
        """Add layernorm mapping."""
        mappings[f"{src}.weight"] = WeightMapping(
            target_path=f"{dst}.scale", sharding=(None,), transpose=False
        )
        mappings[f"{src}.bias"] = WeightMapping(
            target_path=f"{dst}.bias", sharding=(None,), transpose=False
        )

    # ==================== Patch Embedding ====================
    # Conv3d: PyTorch (out, in, T, H, W) -> JAX (T, H, W, in, out)
    mappings["patch_embed.proj.weight"] = WeightMapping(
        target_path="patch_embed.proj.kernel",
        sharding=(None, None, None, None, None),
        transpose=False,
        transpose_axes=(2, 3, 4, 1, 0),
    )
    mappings["patch_embed.proj.bias"] = WeightMapping(
        target_path="patch_embed.proj.bias", sharding=(None,), transpose=False
    )

    # ==================== Position Embedding ====================
    mappings["pos_embed.weight"] = WeightMapping(
        target_path="pos_embed.embedding", sharding=(None, None), transpose=False
    )

    # ==================== Transformer Blocks ====================
    for i in range(config.depth):
        prefix = f"blocks.{i}"

        # LayerNorm
        add_layernorm(f"{prefix}.norm1", f"{prefix}.norm1")
        add_layernorm(f"{prefix}.norm2", f"{prefix}.norm2")

        # Attention: QKV (column-wise TP), Output (row-wise TP)
        add_linear(f"{prefix}.attn.qkv", f"{prefix}.attn.qkv_proj", tp_col=True)
        add_linear(f"{prefix}.attn.proj", f"{prefix}.attn.o_proj", tp_row=True)

        # MLP: fc1 (column-wise TP), fc2 (row-wise TP)
        add_linear(f"{prefix}.mlp.linear_fc1", f"{prefix}.mlp.fc1", tp_col=True)
        add_linear(f"{prefix}.mlp.linear_fc2", f"{prefix}.mlp.fc2", tp_row=True)

    # ==================== Final Merger ====================
    add_layernorm("merger.ln_q", "merger.ln_q")
    # Merger MLP: [0]=Linear, [1]=GELU, [2]=Linear
    add_linear("merger.mlp.0", "merger.mlp_fc1", tp_col=True)
    add_linear("merger.mlp.2", "merger.mlp_fc2", tp_row=True)

    # ==================== Deepstack Mergers ====================
    deepstack_indexes = getattr(config, "deepstack_visual_indexes", [8, 16, 24])
    for idx in range(len(deepstack_indexes)):
        src = f"merger_list.{idx}"
        dst = f"deepstack_mergers.{idx}"

        add_layernorm(f"{src}.ln_q", f"{dst}.ln_q")
        add_linear(f"{src}.mlp.0", f"{dst}.mlp_fc1", tp_col=True)
        add_linear(f"{src}.mlp.2", f"{dst}.mlp_fc2", tp_row=True)

    return mappings
