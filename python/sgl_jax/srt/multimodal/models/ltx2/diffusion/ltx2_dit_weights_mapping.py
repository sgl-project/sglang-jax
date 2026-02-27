from sgl_jax.srt.utils.weight_utils import WeightMapping


def to_ltx2_video_only_mappings(num_layers: int = 48) -> dict[str, WeightMapping]:
    """
    Create weight mappings from LTX-2 checkpoint to JAX model parameters.

    Checkpoint format: keys under "model.diffusion_model.*"
    (pass keys WITHOUT the "model.diffusion_model." prefix)

    JAX model attribute structure:
      patch_embedding: nnx.Linear -> .kernel, .bias
      adaln_single: TimestepEmbedder
        .mlp.fc_in: LinearBase -> .weight, .bias
        .mlp.fc_out: LinearBase -> .weight, .bias
        .linear: LinearBase -> .weight, .bias
      caption_projection: MLP
        .fc_in: LinearBase -> .weight, .bias
        .fc_out: LinearBase -> .weight, .bias
      scale_shift_table: nnx.Param
      proj_out: nnx.Linear -> .kernel, .bias
      blocks[i]:
        .attn1: LTX2Attention
          .to_q: LinearBase -> .weight, .bias
          .to_k: LinearBase -> .weight, .bias
          .to_v: LinearBase -> .weight, .bias
          .to_out: LinearBase -> .weight, .bias
          .norm_q: RMSNorm -> .scale
          .norm_k: RMSNorm -> .scale
        .attn2: LTX2Attention (same structure)
        .ff: MLP
          .fc_in: LinearBase -> .weight, .bias
          .fc_out: LinearBase -> .weight, .bias
        .scale_shift_table: nnx.Param

    Checkpoint key differences from the old (wrong) mappings:
      - patchify_proj.weight -> patch_embedding.kernel (nnx.Linear)
      - adaln_single.emb.timestep_embedder.linear_{1,2} (not linear_{1,2} directly)
      - adaln_single.linear -> adaln_single.linear.weight (the final linear in TimestepEmbedder)
      - caption_projection.linear_{1,2} (not y_proj)
      - attn1.q_norm.weight (not norm_q.weight)
      - attn1.to_out.0.weight (not to_out.weight)
    """
    mappings: dict[str, WeightMapping] = {}

    # ==========================================================================
    # Patch Embedding (patchify_proj) - nnx.Linear
    # PyTorch: [out, in] = [4096, 128]  ->  JAX kernel: [in, out] = [128, 4096]
    # ==========================================================================
    mappings["patchify_proj.weight"] = WeightMapping(
        target_path="patch_embedding.kernel",
        sharding=(None, None),
        transpose=True,
    )
    mappings["patchify_proj.bias"] = WeightMapping(
        target_path="patch_embedding.bias",
        sharding=(None,),
    )

    # ==========================================================================
    # AdaLN Single (Timestep Embedder) - TimestepEmbedder
    # checkpoint: adaln_single.emb.timestep_embedder.linear_{1,2}.*
    #             adaln_single.linear.*
    # JAX:        adaln_single.mlp.fc_in.weight / adaln_single.mlp.fc_out.weight
    #             adaln_single.linear.weight
    # ==========================================================================
    # MLP fc_in: [4096, 256] (PyTorch) -> JAX LinearBase.weight [256, 4096], transpose
    mappings["adaln_single.emb.timestep_embedder.linear_1.weight"] = WeightMapping(
        target_path="adaln_single.mlp.fc_in.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["adaln_single.emb.timestep_embedder.linear_1.bias"] = WeightMapping(
        target_path="adaln_single.mlp.fc_in.bias",
        sharding=("tensor",),
    )
    # MLP fc_out: [4096, 4096] (PyTorch) -> JAX LinearBase.weight [4096, 4096], transpose
    mappings["adaln_single.emb.timestep_embedder.linear_2.weight"] = WeightMapping(
        target_path="adaln_single.mlp.fc_out.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["adaln_single.emb.timestep_embedder.linear_2.bias"] = WeightMapping(
        target_path="adaln_single.mlp.fc_out.bias",
        sharding=(None,),
    )
    # Final AdaLN linear: [24576, 4096] -> JAX LinearBase.weight [4096, 24576], transpose
    mappings["adaln_single.linear.weight"] = WeightMapping(
        target_path="adaln_single.linear.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["adaln_single.linear.bias"] = WeightMapping(
        target_path="adaln_single.linear.bias",
        sharding=("tensor",),
    )

    # ==========================================================================
    # Caption Projection - MLP (2-layer with GELU tanh, PixArtAlphaTextProjection)
    # checkpoint: caption_projection.linear_{1,2}.*
    # JAX: caption_projection.fc_in.weight / caption_projection.fc_out.weight
    # ==========================================================================
    # fc_in: [4096, 3840] (PyTorch) -> [3840, 4096] (JAX LinearBase), transpose
    mappings["caption_projection.linear_1.weight"] = WeightMapping(
        target_path="caption_projection.fc_in.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["caption_projection.linear_1.bias"] = WeightMapping(
        target_path="caption_projection.fc_in.bias",
        sharding=("tensor",),
    )
    # fc_out: [4096, 4096] (PyTorch) -> [4096, 4096] (JAX LinearBase), transpose
    mappings["caption_projection.linear_2.weight"] = WeightMapping(
        target_path="caption_projection.fc_out.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["caption_projection.linear_2.bias"] = WeightMapping(
        target_path="caption_projection.fc_out.bias",
        sharding=(None,),
    )

    # ==========================================================================
    # Transformer Blocks - using wildcard * for block index
    # ==========================================================================

    # --- Self-Attention (attn1) ---
    # to_q: [4096, 4096] (PyTorch [out,in]) -> JAX LinearBase.weight [in,out], transpose
    mappings["transformer_blocks.*.attn1.to_q.weight"] = WeightMapping(
        target_path="blocks.*.attn1.to_q.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.attn1.to_q.bias"] = WeightMapping(
        target_path="blocks.*.attn1.to_q.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.attn1.to_k.weight"] = WeightMapping(
        target_path="blocks.*.attn1.to_k.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.attn1.to_k.bias"] = WeightMapping(
        target_path="blocks.*.attn1.to_k.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.attn1.to_v.weight"] = WeightMapping(
        target_path="blocks.*.attn1.to_v.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.attn1.to_v.bias"] = WeightMapping(
        target_path="blocks.*.attn1.to_v.bias",
        sharding=("tensor",),
    )
    # to_out.0 in PyTorch -> to_out in JAX (LinearBase)
    mappings["transformer_blocks.*.attn1.to_out.0.weight"] = WeightMapping(
        target_path="blocks.*.attn1.to_out.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["transformer_blocks.*.attn1.to_out.0.bias"] = WeightMapping(
        target_path="blocks.*.attn1.to_out.bias",
        sharding=(None,),
    )
    # QK Norm: q_norm/k_norm in PyTorch -> norm_q/norm_k in JAX (RMSNorm.scale)
    mappings["transformer_blocks.*.attn1.q_norm.weight"] = WeightMapping(
        target_path="blocks.*.attn1.norm_q.scale",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.attn1.k_norm.weight"] = WeightMapping(
        target_path="blocks.*.attn1.norm_k.scale",
        sharding=(None,),
    )

    # --- Cross-Attention (attn2) ---
    mappings["transformer_blocks.*.attn2.to_q.weight"] = WeightMapping(
        target_path="blocks.*.attn2.to_q.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.attn2.to_q.bias"] = WeightMapping(
        target_path="blocks.*.attn2.to_q.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.attn2.to_k.weight"] = WeightMapping(
        target_path="blocks.*.attn2.to_k.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.attn2.to_k.bias"] = WeightMapping(
        target_path="blocks.*.attn2.to_k.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.attn2.to_v.weight"] = WeightMapping(
        target_path="blocks.*.attn2.to_v.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.attn2.to_v.bias"] = WeightMapping(
        target_path="blocks.*.attn2.to_v.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.attn2.to_out.0.weight"] = WeightMapping(
        target_path="blocks.*.attn2.to_out.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["transformer_blocks.*.attn2.to_out.0.bias"] = WeightMapping(
        target_path="blocks.*.attn2.to_out.bias",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.attn2.q_norm.weight"] = WeightMapping(
        target_path="blocks.*.attn2.norm_q.scale",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.attn2.k_norm.weight"] = WeightMapping(
        target_path="blocks.*.attn2.norm_k.scale",
        sharding=(None,),
    )

    # --- Feed-Forward Network (ff) ---
    # net.0.proj = fc_in (GLU projection, PyTorch [16384, 4096] -> JAX [4096, 16384])
    mappings["transformer_blocks.*.ff.net.0.proj.weight"] = WeightMapping(
        target_path="blocks.*.ff.fc_in.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.ff.net.0.proj.bias"] = WeightMapping(
        target_path="blocks.*.ff.fc_in.bias",
        sharding=("tensor",),
    )
    # net.2 = fc_out (PyTorch [4096, 16384] -> JAX [16384, 4096])
    mappings["transformer_blocks.*.ff.net.2.weight"] = WeightMapping(
        target_path="blocks.*.ff.fc_out.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["transformer_blocks.*.ff.net.2.bias"] = WeightMapping(
        target_path="blocks.*.ff.fc_out.bias",
        sharding=(None,),
    )

    # Scale-shift table: [6, dim] - per-block AdaLN parameters
    mappings["transformer_blocks.*.scale_shift_table"] = WeightMapping(
        target_path="blocks.*.scale_shift_table",
        sharding=(None, None),
    )

    # ==========================================================================
    # Output Layers
    # ==========================================================================
    # proj_out: nnx.Linear, [128, 4096] (PyTorch) -> kernel [4096, 128] (JAX), transpose
    mappings["proj_out.weight"] = WeightMapping(
        target_path="proj_out.kernel",
        sharding=(None, None),
        transpose=True,
    )
    mappings["proj_out.bias"] = WeightMapping(
        target_path="proj_out.bias",
        sharding=(None,),
    )

    # Global scale shift table: [2, dim]
    mappings["scale_shift_table"] = WeightMapping(
        target_path="scale_shift_table",
        sharding=(None, None),
    )

    return mappings


def to_ltx2_audio_video_mappings(num_layers: int = 48) -> dict[str, WeightMapping]:
    """
    Create weight mappings for LTX-2 audio-video model.
    Includes mappings for both video and audio components plus audio-video cross-attention.
    """
    # Start with video-only mappings
    mappings = to_ltx2_video_only_mappings(num_layers)

    # ==========================================================================
    # Audio Components
    # ==========================================================================

    # Audio patchify projection (nnx.Linear)
    mappings["audio_patchify_proj.weight"] = WeightMapping(
        target_path="audio_patch_embedding.kernel",
        sharding=(None, None),
        transpose=True,
    )
    mappings["audio_patchify_proj.bias"] = WeightMapping(
        target_path="audio_patch_embedding.bias",
        sharding=(None,),
    )

    # Audio AdaLN Single (TimestepEmbedder)
    mappings["audio_adaln_single.emb.timestep_embedder.linear_1.weight"] = WeightMapping(
        target_path="audio_adaln_single.mlp.fc_in.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["audio_adaln_single.emb.timestep_embedder.linear_1.bias"] = WeightMapping(
        target_path="audio_adaln_single.mlp.fc_in.bias",
        sharding=("tensor",),
    )
    mappings["audio_adaln_single.emb.timestep_embedder.linear_2.weight"] = WeightMapping(
        target_path="audio_adaln_single.mlp.fc_out.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["audio_adaln_single.emb.timestep_embedder.linear_2.bias"] = WeightMapping(
        target_path="audio_adaln_single.mlp.fc_out.bias",
        sharding=(None,),
    )
    mappings["audio_adaln_single.linear.weight"] = WeightMapping(
        target_path="audio_adaln_single.linear.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["audio_adaln_single.linear.bias"] = WeightMapping(
        target_path="audio_adaln_single.linear.bias",
        sharding=("tensor",),
    )

    # Audio Caption Projection (MLP)
    mappings["audio_caption_projection.linear_1.weight"] = WeightMapping(
        target_path="audio_caption_projection.fc_in.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["audio_caption_projection.linear_1.bias"] = WeightMapping(
        target_path="audio_caption_projection.fc_in.bias",
        sharding=("tensor",),
    )
    mappings["audio_caption_projection.linear_2.weight"] = WeightMapping(
        target_path="audio_caption_projection.fc_out.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["audio_caption_projection.linear_2.bias"] = WeightMapping(
        target_path="audio_caption_projection.fc_out.bias",
        sharding=(None,),
    )

    # ==========================================================================
    # Audio Transformer Block Components
    # ==========================================================================

    # Audio Self-Attention (audio_attn1)
    mappings["transformer_blocks.*.audio_attn1.to_q.weight"] = WeightMapping(
        target_path="blocks.*.audio_attn1.to_q.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.audio_attn1.to_q.bias"] = WeightMapping(
        target_path="blocks.*.audio_attn1.to_q.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.audio_attn1.to_k.weight"] = WeightMapping(
        target_path="blocks.*.audio_attn1.to_k.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.audio_attn1.to_k.bias"] = WeightMapping(
        target_path="blocks.*.audio_attn1.to_k.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.audio_attn1.to_v.weight"] = WeightMapping(
        target_path="blocks.*.audio_attn1.to_v.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.audio_attn1.to_v.bias"] = WeightMapping(
        target_path="blocks.*.audio_attn1.to_v.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.audio_attn1.to_out.0.weight"] = WeightMapping(
        target_path="blocks.*.audio_attn1.to_out.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["transformer_blocks.*.audio_attn1.to_out.0.bias"] = WeightMapping(
        target_path="blocks.*.audio_attn1.to_out.bias",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.audio_attn1.q_norm.weight"] = WeightMapping(
        target_path="blocks.*.audio_attn1.norm_q.scale",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.audio_attn1.k_norm.weight"] = WeightMapping(
        target_path="blocks.*.audio_attn1.norm_k.scale",
        sharding=(None,),
    )

    # Audio Cross-Attention (audio_attn2)
    mappings["transformer_blocks.*.audio_attn2.to_q.weight"] = WeightMapping(
        target_path="blocks.*.audio_attn2.to_q.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.audio_attn2.to_q.bias"] = WeightMapping(
        target_path="blocks.*.audio_attn2.to_q.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.audio_attn2.to_k.weight"] = WeightMapping(
        target_path="blocks.*.audio_attn2.to_k.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.audio_attn2.to_k.bias"] = WeightMapping(
        target_path="blocks.*.audio_attn2.to_k.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.audio_attn2.to_v.weight"] = WeightMapping(
        target_path="blocks.*.audio_attn2.to_v.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.audio_attn2.to_v.bias"] = WeightMapping(
        target_path="blocks.*.audio_attn2.to_v.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.audio_attn2.to_out.0.weight"] = WeightMapping(
        target_path="blocks.*.audio_attn2.to_out.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["transformer_blocks.*.audio_attn2.to_out.0.bias"] = WeightMapping(
        target_path="blocks.*.audio_attn2.to_out.bias",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.audio_attn2.q_norm.weight"] = WeightMapping(
        target_path="blocks.*.audio_attn2.norm_q.scale",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.audio_attn2.k_norm.weight"] = WeightMapping(
        target_path="blocks.*.audio_attn2.norm_k.scale",
        sharding=(None,),
    )

    # Audio Feed-Forward
    mappings["transformer_blocks.*.audio_ff.net.0.proj.weight"] = WeightMapping(
        target_path="blocks.*.audio_ff.fc_in.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.audio_ff.net.0.proj.bias"] = WeightMapping(
        target_path="blocks.*.audio_ff.fc_in.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.audio_ff.net.2.weight"] = WeightMapping(
        target_path="blocks.*.audio_ff.fc_out.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["transformer_blocks.*.audio_ff.net.2.bias"] = WeightMapping(
        target_path="blocks.*.audio_ff.fc_out.bias",
        sharding=(None,),
    )

    # Audio Scale Shift Table
    mappings["transformer_blocks.*.audio_scale_shift_table"] = WeightMapping(
        target_path="blocks.*.audio_scale_shift_table",
        sharding=(None, None),
    )

    # ==========================================================================
    # Audio-Video Cross-Attention
    # ==========================================================================

    # Audio to Video Cross-Attention (audio_to_video_attn)
    mappings["transformer_blocks.*.audio_to_video_attn.to_q.weight"] = WeightMapping(
        target_path="blocks.*.audio_to_video_attn.to_q.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.audio_to_video_attn.to_q.bias"] = WeightMapping(
        target_path="blocks.*.audio_to_video_attn.to_q.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.audio_to_video_attn.to_k.weight"] = WeightMapping(
        target_path="blocks.*.audio_to_video_attn.to_k.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.audio_to_video_attn.to_k.bias"] = WeightMapping(
        target_path="blocks.*.audio_to_video_attn.to_k.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.audio_to_video_attn.to_v.weight"] = WeightMapping(
        target_path="blocks.*.audio_to_video_attn.to_v.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.audio_to_video_attn.to_v.bias"] = WeightMapping(
        target_path="blocks.*.audio_to_video_attn.to_v.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.audio_to_video_attn.to_out.0.weight"] = WeightMapping(
        target_path="blocks.*.audio_to_video_attn.to_out.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["transformer_blocks.*.audio_to_video_attn.to_out.0.bias"] = WeightMapping(
        target_path="blocks.*.audio_to_video_attn.to_out.bias",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.audio_to_video_attn.q_norm.weight"] = WeightMapping(
        target_path="blocks.*.audio_to_video_attn.norm_q.scale",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.audio_to_video_attn.k_norm.weight"] = WeightMapping(
        target_path="blocks.*.audio_to_video_attn.norm_k.scale",
        sharding=(None,),
    )

    # Video to Audio Cross-Attention (video_to_audio_attn)
    mappings["transformer_blocks.*.video_to_audio_attn.to_q.weight"] = WeightMapping(
        target_path="blocks.*.video_to_audio_attn.to_q.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.video_to_audio_attn.to_q.bias"] = WeightMapping(
        target_path="blocks.*.video_to_audio_attn.to_q.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.video_to_audio_attn.to_k.weight"] = WeightMapping(
        target_path="blocks.*.video_to_audio_attn.to_k.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.video_to_audio_attn.to_k.bias"] = WeightMapping(
        target_path="blocks.*.video_to_audio_attn.to_k.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.video_to_audio_attn.to_v.weight"] = WeightMapping(
        target_path="blocks.*.video_to_audio_attn.to_v.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.video_to_audio_attn.to_v.bias"] = WeightMapping(
        target_path="blocks.*.video_to_audio_attn.to_v.bias",
        sharding=("tensor",),
    )
    mappings["transformer_blocks.*.video_to_audio_attn.to_out.0.weight"] = WeightMapping(
        target_path="blocks.*.video_to_audio_attn.to_out.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["transformer_blocks.*.video_to_audio_attn.to_out.0.bias"] = WeightMapping(
        target_path="blocks.*.video_to_audio_attn.to_out.bias",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.video_to_audio_attn.q_norm.weight"] = WeightMapping(
        target_path="blocks.*.video_to_audio_attn.norm_q.scale",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.video_to_audio_attn.k_norm.weight"] = WeightMapping(
        target_path="blocks.*.video_to_audio_attn.norm_k.scale",
        sharding=(None,),
    )

    # Audio-Video Scale Shift Tables (per-block)
    mappings["transformer_blocks.*.scale_shift_table_a2v_ca_audio"] = WeightMapping(
        target_path="blocks.*.scale_shift_table_a2v_ca_audio",
        sharding=(None, None),
    )
    mappings["transformer_blocks.*.scale_shift_table_a2v_ca_video"] = WeightMapping(
        target_path="blocks.*.scale_shift_table_a2v_ca_video",
        sharding=(None, None),
    )

    # ==========================================================================
    # Audio Output Layers
    # ==========================================================================
    mappings["audio_proj_out.weight"] = WeightMapping(
        target_path="audio_proj_out.kernel",
        sharding=(None, None),
        transpose=True,
    )
    mappings["audio_proj_out.bias"] = WeightMapping(
        target_path="audio_proj_out.bias",
        sharding=(None,),
    )
    mappings["audio_scale_shift_table"] = WeightMapping(
        target_path="audio_scale_shift_table",
        sharding=(None, None),
    )

    # ==========================================================================
    # AV Cross-Attention AdaLN Singles (TimestepEmbedder)
    # Pattern: {name}_adaln_single.emb.timestep_embedder.linear_{1,2} + {name}_adaln_single.linear
    # ==========================================================================
    for name in ["av_ca_video_scale_shift", "av_ca_a2v_gate", "av_ca_audio_scale_shift", "av_ca_v2a_gate"]:
        jax_name = {
            "av_ca_video_scale_shift": "av_ca_video_scale_shift_adaln",
            "av_ca_a2v_gate": "av_ca_a2v_gate_adaln",
            "av_ca_audio_scale_shift": "av_ca_audio_scale_shift_adaln",
            "av_ca_v2a_gate": "av_ca_v2a_gate_adaln",
        }[name]

        mappings[f"{name}_adaln_single.emb.timestep_embedder.linear_1.weight"] = WeightMapping(
            target_path=f"{jax_name}.mlp.fc_in.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings[f"{name}_adaln_single.emb.timestep_embedder.linear_1.bias"] = WeightMapping(
            target_path=f"{jax_name}.mlp.fc_in.bias",
            sharding=("tensor",),
        )
        mappings[f"{name}_adaln_single.emb.timestep_embedder.linear_2.weight"] = WeightMapping(
            target_path=f"{jax_name}.mlp.fc_out.weight",
            sharding=("tensor", None),
            transpose=True,
        )
        mappings[f"{name}_adaln_single.emb.timestep_embedder.linear_2.bias"] = WeightMapping(
            target_path=f"{jax_name}.mlp.fc_out.bias",
            sharding=(None,),
        )
        mappings[f"{name}_adaln_single.linear.weight"] = WeightMapping(
            target_path=f"{jax_name}.linear.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings[f"{name}_adaln_single.linear.bias"] = WeightMapping(
            target_path=f"{jax_name}.linear.bias",
            sharding=("tensor",),
        )

    return mappings
