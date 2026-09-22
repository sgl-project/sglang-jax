"""Abstract cache state and attention metadata for Kimi Linear exports."""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    HybridLinearAttnBackendMetadata,
    LinearRecurrentAttnBackendMetadata,
)
from sgl_jax.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sgl_jax.srt.layers.attention.mla_backend import (
    MLAAttentionBackend,
    MLAAttentionMetadata,
)
from sgl_jax.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MemoryPools,
    MLATokenToKVPool,
)
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool


def validate_config(config, options):
    if options.attention_backend != "fa" or options.moe_backend != "epmoe":
        raise ValueError("Kimi Linear requires --attention-backend=fa --moe-backend=epmoe")
    if options.ep_size != options.tp_size or config.num_experts % options.ep_size:
        raise ValueError("Kimi export requires ep_size=tp_size and experts divisible by ep_size")
    linear = config.linear_attn_config
    layers = linear["kda_layers"] + linear["full_attn_layers"]
    if sorted(layers) != list(range(1, config.num_hidden_layers + 1)):
        raise ValueError("Kimi linear/full attention layer IDs must partition all layers (1-based)")
    if linear["num_heads"] % (options.tp_size // options.dp_size):
        raise ValueError("KDA heads must be divisible by attention TP")
    if options.page_size < 2 or options.page_size % 2:
        raise ValueError("BF16 MLA requires an even page_size of at least 2")
    if not config.is_mla or not getattr(config, "use_absorbed_mla", True):
        raise ValueError("Kimi Linear export uses absorbed MLA")
    config.moe_backend = options.moe_backend
    config.ep_size = options.ep_size


def bind_parameter_specs(model, config, parameter_specs):
    # Serving post_load_weights reshapes [latent, heads * (nope + value)]
    # and splits the last dimension. Preserve the original head partition.
    for layer_id in config.full_attention_layer_ids:
        prefix = f"model.layers.{layer_id}.self_attn"
        source = parameter_specs.pop(f"{prefix}.kv_b_proj.weight")
        for name in ("w_uk", "w_uv"):
            parameter_specs[f"{prefix}.{name}"] = P(source[0], source[1], None)
    # EPMoE uses its own (expert, tensor) mesh over the same devices.
    return {
        f"model.layers.{layer_id}.block_sparse_moe.{name}": layer.block_sparse_moe.moe_mesh
        for layer_id, layer in enumerate(model.model.layers)
        if layer.is_moe_layer
        for name in ("wi_0", "wi_1", "wo")
    }


def build_resources(config, options, mesh):
    def vector(length, dtype=jnp.int32):
        return jax.ShapeDtypeStruct((length,), dtype, sharding=NamedSharding(mesh, P("data")))

    batch_size = options.batch_size
    linear = config.linear_attn_config
    pool = HybridLinearKVPool(
        size=options.kv_capacity,
        page_size=options.page_size,
        dtype=jnp.bfloat16,
        full_attention_layer_ids=config.full_attention_layer_ids,
        mesh=mesh,
        token_to_kv_pool_class=MLATokenToKVPool,
        kv_lora_rank=config.kv_lora_rank,
        qk_rope_head_dim=config.qk_rope_head_dim,
        dp_size=options.dp_size,
        abstract=True,
    )
    recurrent_pool = RecurrentStatePool(
        linear_recurrent_layer_ids=config.linear_layer_ids,
        size=options.recurrent_capacity or batch_size,
        num_heads=linear["num_heads"],
        head_dim=linear["head_dim"],
        conv_kernel_size=linear["short_conv_kernel_size"],
        mesh=mesh,
        dp_size=options.dp_size,
        abstract=True,
    )
    backend = HybridLinearAttnBackend(
        full_attn_backend=MLAAttentionBackend(
            num_attn_heads=config.num_attention_heads,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            page_size=options.page_size,
            mesh=mesh,
        ),
        linear_attn_backend=KDAAttnBackend(mesh=mesh),
        full_attn_layers=config.full_attention_layer_ids,
    )
    pages_per_request = -(-options.context_length // options.page_size)
    # Match serving's per-DP decode metadata. Indices, initial-state flags,
    # sequence lengths, and distributions remain dynamic graph inputs.
    backend.forward_metadata = HybridLinearAttnBackendMetadata(
        full_attn_metadata=MLAAttentionMetadata(
            cu_q_lens=vector(batch_size + options.dp_size),
            cu_kv_lens=vector(batch_size + options.dp_size),
            page_indices=vector(batch_size * pages_per_request),
            seq_lens=vector(batch_size),
            distribution=vector(3 * options.dp_size),
        ),
        linear_attn_metadata=LinearRecurrentAttnBackendMetadata(
            cu_q_lens=vector(batch_size + options.dp_size),
            recurrent_indices=vector(batch_size),
            has_initial_state=vector(batch_size, jnp.bool_),
        ),
    )
    return backend, MemoryPools(token_to_kv_pool=pool, recurrent_state_pool=recurrent_pool)
