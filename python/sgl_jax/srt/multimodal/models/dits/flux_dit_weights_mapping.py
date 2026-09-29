from sgl_jax.srt.model_loader.weights import WeightSpec


def to_mappings(has_guidance_embeds: bool = False) -> dict[str, WeightSpec]:
    mappings: dict[str, WeightSpec] = {}

    mappings["time_text_embed.timestep_embedder.linear_1.weight"] = WeightSpec(
        target_path="time_text_embed.timestep_embedder.linear_1.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["time_text_embed.timestep_embedder.linear_1.bias"] = WeightSpec(
        target_path="time_text_embed.timestep_embedder.linear_1.bias",
        sharding=(None,),
    )
    mappings["time_text_embed.timestep_embedder.linear_2.weight"] = WeightSpec(
        target_path="time_text_embed.timestep_embedder.linear_2.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["time_text_embed.timestep_embedder.linear_2.bias"] = WeightSpec(
        target_path="time_text_embed.timestep_embedder.linear_2.bias",
        sharding=(None,),
    )
    mappings["time_text_embed.text_embedder.linear_1.weight"] = WeightSpec(
        target_path="time_text_embed.text_embedder.linear_1.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["time_text_embed.text_embedder.linear_1.bias"] = WeightSpec(
        target_path="time_text_embed.text_embedder.linear_1.bias",
        sharding=(None,),
    )
    mappings["time_text_embed.text_embedder.linear_2.weight"] = WeightSpec(
        target_path="time_text_embed.text_embedder.linear_2.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["time_text_embed.text_embedder.linear_2.bias"] = WeightSpec(
        target_path="time_text_embed.text_embedder.linear_2.bias",
        sharding=(None,),
    )

    if has_guidance_embeds:
        mappings["time_text_embed.guidance_embedder.linear_1.weight"] = WeightSpec(
            target_path="time_text_embed.guidance_embedder.linear_1.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings["time_text_embed.guidance_embedder.linear_1.bias"] = WeightSpec(
            target_path="time_text_embed.guidance_embedder.linear_1.bias",
            sharding=(None,),
        )
        mappings["time_text_embed.guidance_embedder.linear_2.weight"] = WeightSpec(
            target_path="time_text_embed.guidance_embedder.linear_2.weight",
            sharding=("tensor", None),
            transpose=True,
        )
        mappings["time_text_embed.guidance_embedder.linear_2.bias"] = WeightSpec(
            target_path="time_text_embed.guidance_embedder.linear_2.bias",
            sharding=(None,),
        )

    mappings["context_embedder.weight"] = WeightSpec(
        target_path="context_embedder.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["context_embedder.bias"] = WeightSpec(
        target_path="context_embedder.bias",
        sharding=(None,),
    )
    mappings["x_embedder.weight"] = WeightSpec(
        target_path="x_embedder.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["x_embedder.bias"] = WeightSpec(
        target_path="x_embedder.bias",
        sharding=(None,),
    )

    mappings["transformer_blocks.*.norm1.linear.weight"] = WeightSpec(
        target_path="transformer_blocks.*.norm1.linear.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.norm1.linear.bias"] = WeightSpec(
        target_path="transformer_blocks.*.norm1.linear.bias",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.norm1_context.linear.weight"] = WeightSpec(
        target_path="transformer_blocks.*.norm1_context.linear.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.norm1_context.linear.bias"] = WeightSpec(
        target_path="transformer_blocks.*.norm1_context.linear.bias",
        sharding=(None,),
    )

    for proj_name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
        mappings[f"transformer_blocks.*.attn.{proj_name}.weight"] = WeightSpec(
            target_path=f"transformer_blocks.*.attn.{proj_name}.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings[f"transformer_blocks.*.attn.{proj_name}.bias"] = WeightSpec(
            target_path=f"transformer_blocks.*.attn.{proj_name}.bias",
            sharding=(None,),
        )

    mappings["transformer_blocks.*.attn.to_out.0.weight"] = WeightSpec(
        target_path="transformer_blocks.*.attn.to_out.0.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["transformer_blocks.*.attn.to_out.0.bias"] = WeightSpec(
        target_path="transformer_blocks.*.attn.to_out.0.bias",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.attn.to_add_out.weight"] = WeightSpec(
        target_path="transformer_blocks.*.attn.to_add_out.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["transformer_blocks.*.attn.to_add_out.bias"] = WeightSpec(
        target_path="transformer_blocks.*.attn.to_add_out.bias",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.attn.norm_q.weight"] = WeightSpec(
        target_path="transformer_blocks.*.attn.norm_q.scale",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.attn.norm_k.weight"] = WeightSpec(
        target_path="transformer_blocks.*.attn.norm_k.scale",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.attn.norm_added_q.weight"] = WeightSpec(
        target_path="transformer_blocks.*.attn.norm_added_q.scale",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.attn.norm_added_k.weight"] = WeightSpec(
        target_path="transformer_blocks.*.attn.norm_added_k.scale",
        sharding=(None,),
    )

    mappings["transformer_blocks.*.ff.net.0.proj.weight"] = WeightSpec(
        target_path="transformer_blocks.*.ff.net.0.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.ff.net.0.proj.bias"] = WeightSpec(
        target_path="transformer_blocks.*.ff.net.0.bias",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.ff.net.2.weight"] = WeightSpec(
        target_path="transformer_blocks.*.ff.net.2.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["transformer_blocks.*.ff.net.2.bias"] = WeightSpec(
        target_path="transformer_blocks.*.ff.net.2.bias",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.ff_context.net.0.proj.weight"] = WeightSpec(
        target_path="transformer_blocks.*.ff_context.net.0.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["transformer_blocks.*.ff_context.net.0.proj.bias"] = WeightSpec(
        target_path="transformer_blocks.*.ff_context.net.0.bias",
        sharding=(None,),
    )
    mappings["transformer_blocks.*.ff_context.net.2.weight"] = WeightSpec(
        target_path="transformer_blocks.*.ff_context.net.2.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["transformer_blocks.*.ff_context.net.2.bias"] = WeightSpec(
        target_path="transformer_blocks.*.ff_context.net.2.bias",
        sharding=(None,),
    )

    mappings["single_transformer_blocks.*.norm.linear.weight"] = WeightSpec(
        target_path="single_transformer_blocks.*.norm.linear.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["single_transformer_blocks.*.norm.linear.bias"] = WeightSpec(
        target_path="single_transformer_blocks.*.norm.linear.bias",
        sharding=(None,),
    )
    for proj_name in ("to_q", "to_k", "to_v"):
        mappings[f"single_transformer_blocks.*.attn.{proj_name}.weight"] = WeightSpec(
            target_path=f"single_transformer_blocks.*.attn.{proj_name}.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings[f"single_transformer_blocks.*.attn.{proj_name}.bias"] = WeightSpec(
            target_path=f"single_transformer_blocks.*.attn.{proj_name}.bias",
            sharding=(None,),
        )
    mappings["single_transformer_blocks.*.attn.norm_q.weight"] = WeightSpec(
        target_path="single_transformer_blocks.*.attn.norm_q.scale",
        sharding=(None,),
    )
    mappings["single_transformer_blocks.*.attn.norm_k.weight"] = WeightSpec(
        target_path="single_transformer_blocks.*.attn.norm_k.scale",
        sharding=(None,),
    )
    mappings["single_transformer_blocks.*.proj_mlp.weight"] = WeightSpec(
        target_path="single_transformer_blocks.*.proj_mlp.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["single_transformer_blocks.*.proj_mlp.bias"] = WeightSpec(
        target_path="single_transformer_blocks.*.proj_mlp.bias",
        sharding=(None,),
    )
    mappings["single_transformer_blocks.*.proj_out.weight"] = WeightSpec(
        target_path="single_transformer_blocks.*.proj_out.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["single_transformer_blocks.*.proj_out.bias"] = WeightSpec(
        target_path="single_transformer_blocks.*.proj_out.bias",
        sharding=(None,),
    )

    mappings["norm_out.linear.weight"] = WeightSpec(
        target_path="norm_out.linear.weight",
        sharding=(None, "tensor"),
        transpose=True,
    )
    mappings["norm_out.linear.bias"] = WeightSpec(
        target_path="norm_out.linear.bias",
        sharding=(None,),
    )
    mappings["proj_out.weight"] = WeightSpec(
        target_path="proj_out.weight",
        sharding=("tensor", None),
        transpose=True,
    )
    mappings["proj_out.bias"] = WeightSpec(
        target_path="proj_out.bias",
        sharding=(None,),
    )

    return mappings
