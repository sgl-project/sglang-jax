import logging
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
from flax import nnx
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig, MoEBackend
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.kernels.dsa.ref import build_index_share_map
from sgl_jax.srt.kernels.fused_mlp import apply_fused_mlp_with_padding
from sgl_jax.srt.layers.attention.dsa_indexer_ops import (
    DSAIndexerOutput,
    update_index_cache_and_select,
)
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import (
    EPMoE,
    FusedEPMoE,
    FusedEPMoEV2,
    GateLogit,
    TopK,
    create_moe_weights_mapping,
)
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.quantization.quantization_utils import (
    dequantize_tensor,
    quantize_tensor,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


@partial(jax.jit, static_argnames=("quantized_dtype",))
def _requantize_blockwise_shared_weight(
    weight_q: jax.Array,
    block_scale: jax.Array,
    *,
    quantized_dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
    """Convert a transposed HF block-wise FP8 weight to per-channel FP8.

    GLM-5.2-FP8 stores shared-expert weights as ``[out, in]`` with a
    ``[out_blocks, in_blocks]`` scale grid. The fused-v2 layer stores the
    transposed weight as ``[in, out]``, while its shared-expert kernel accepts
    one scale per output channel. Dequantize with the transposed block grid
    before requantizing; reshaping the original block scales would change the
    represented weight values.
    """
    weight = dequantize_tensor(
        weight_q,
        jnp.transpose(block_scale),
        axis=(0, 1),
        out_dtype=jnp.float32,
    )
    return quantize_tensor(quantized_dtype, weight, axis=0)


def _requantize_glm5_shared_expert(mlp: FusedEPMoEV2) -> None:
    """Finish loading GLM-5.2 static block-wise shared-expert weights.

    TODO: This is a temporary checkpoint-compatibility bridge. GLM-5.2 stores
    shared-expert FP8 weights with 2D block-wise scales, while the fused-v2
    in-kernel shared-expert path currently accepts only one scale per output
    channel. Dequantizing and requantizing introduces a second FP8 rounding;
    remove this conversion once that kernel consumes block-wise scales directly.
    """
    if not hasattr(mlp, "w1_shared_block_scale"):
        return

    if mlp.quantized_dtype is None:
        raise ValueError("GLM-5.2 block-wise shared-expert conversion requires FP8 weights")

    with jax.set_mesh(mlp.mesh):
        for weight_name in ("w1_shared", "w3_shared", "w2_shared"):
            scale_name = f"{weight_name}_scale"
            block_scale_name = f"{weight_name}_block_scale"
            weight_q, scale = _requantize_blockwise_shared_weight(
                getattr(mlp, weight_name).value,
                getattr(mlp, block_scale_name).value,
                quantized_dtype=mlp.quantized_dtype,
            )
            # Model loading runs once per layer. Drain each conversion before
            # dropping the block-scale input so 75 layers do not queue all
            # dequant/requant temporaries in device memory at once.
            weight_q.block_until_ready()
            scale.block_until_ready()
            setattr(
                mlp,
                weight_name,
                nnx.Param(weight_q, out_sharding=P(None, None)),
            )
            setattr(
                mlp,
                scale_name,
                nnx.Param(scale.reshape(1, 1, -1), out_sharding=P(None, None, None)),
            )
            delattr(mlp, block_scale_name)

    logger.info("Requantized GLM-5.2 shared expert from block-wise to per-channel FP8")


# No-op: FP32 accumulation logic removed to keep native BF16 execution.


class GlmNorm(nnx.Module):
    def __init__(self, dim: int, dtype: jnp.dtype = jnp.bfloat16):
        self.weight = nnx.Param(jnp.ones((dim,), dtype=dtype))
        self.bias = nnx.Param(jnp.zeros((dim,), dtype=dtype))

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.var(x, axis=-1, keepdims=True)
        eps = 1e-5
        normalized = (x - mean) / jnp.sqrt(variance + eps)
        return normalized * self.weight.value + self.bias.value


def get_hadamard_matrix(n):
    if n == 1:
        return jnp.array([[1.0]])
    h = get_hadamard_matrix(n // 2)
    return jnp.block([[h, h], [h, -h]])


@dataclass(frozen=True)
class GlmDsaLayerPlan:
    """Static model policy for one layer's dense and IndexShare roles."""

    indexer_type: str
    index_cache_slot: int | None
    is_dense_attention: bool
    produces_topk: bool


def _full_indexer_serves_sparse_layer(
    indexer_types: list[str] | tuple[str, ...],
    layer_id: int,
    dense_layer_count: int,
) -> bool:
    """Whether a full layer's IndexShare group reaches sparse attention."""

    if indexer_types[layer_id] != "full":
        return False
    group_end = len(indexer_types)
    for next_layer in range(layer_id + 1, len(indexer_types)):
        if indexer_types[next_layer] == "full":
            group_end = next_layer
            break
    return max(layer_id, dense_layer_count) < group_end


def _build_dsa_layer_plans(config: PretrainedConfig) -> tuple[GlmDsaLayerPlan, ...]:
    """Build the model-owned dense/IndexShare plan exactly once."""

    num_layers = config.num_hidden_layers
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is None:
        indexer_types = ["full"] * num_layers
    if len(indexer_types) != num_layers:
        raise ValueError(f"indexer_types has {len(indexer_types)} entries for {num_layers} layers")

    dense_layer_count = getattr(config, "index_skip_topk_offset", 0)
    if not 0 <= dense_layer_count <= num_layers:
        raise ValueError(
            f"index_skip_topk_offset must be in [0, {num_layers}], got {dense_layer_count}"
        )

    full_slot, _, _ = build_index_share_map(
        indexer_types,
        dense_layer_count,
        num_layers,
    )
    return tuple(
        GlmDsaLayerPlan(
            indexer_type=indexer_type,
            index_cache_slot=full_slot.get(layer_id),
            is_dense_attention=layer_id < dense_layer_count,
            produces_topk=_full_indexer_serves_sparse_layer(
                indexer_types,
                layer_id,
                dense_layer_count,
            ),
        )
        for layer_id, indexer_type in enumerate(indexer_types)
    )


class GlmDsaIndexer(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        index_head_dim: int,
        index_n_heads: int,
        index_topk: int,
        cache_slot: int,
        mesh: jax.sharding.Mesh,
        topk_impl: str = "exact_lax",
        attention_data_partition_axis: str = "data",
        dtype: jnp.dtype = jnp.bfloat16,
        scope_name: str = "indexer",
    ):
        self.head_dim = index_head_dim
        self.n_head = index_n_heads
        self.index_topk = index_topk
        self.cache_slot = cache_slot
        self.mesh = mesh
        self.topk_impl = topk_impl
        self.attention_data_partition_axis = attention_data_partition_axis
        if topk_impl not in ("approx", "exact_lax", "radix"):
            raise ValueError(f"unknown DSA top-k implementation: {topk_impl}")

        self.wq_b = LinearBase(
            input_size=q_lora_rank,
            output_size=index_head_dim * index_n_heads,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="wq_b",
        )
        self.wk = LinearBase(
            input_size=hidden_size,
            output_size=index_head_dim,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="wk",
        )
        self.k_norm = GlmNorm(index_head_dim, dtype)

        self.weights_proj = LinearBase(
            input_size=hidden_size,
            output_size=index_n_heads,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="weights_proj",
        )

    def _project(
        self, hidden_states: jax.Array, qr: jax.Array, positions: jax.Array, rotary_emb: Any
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Project query, key, and per-head gates for Indexer scoring.

        Returns query [T, n_head, head_dim], key [T, head_dim], weights [T, n_head]
        after RoPE + Hadamard, ready for the DSA indexer runtime.
        """
        query, _ = self.wq_b(qr)
        query = query.reshape(-1, self.n_head, self.head_dim)

        key, _ = self.wk(hidden_states)
        key = self.k_norm(key)

        rope_dim = 64
        q_rope = query[:, :, :rope_dim]
        k_rope = key[:, :rope_dim][:, None, :]
        q_rope, k_rope = rotary_emb(positions, q_rope, k_rope)
        query = query.at[:, :, :rope_dim].set(q_rope)
        key = key.at[:, :rope_dim].set(k_rope.squeeze(1))

        h_matrix = get_hadamard_matrix(128) * (128**-0.5)
        query = jnp.einsum("thd,de->the", query, h_matrix)
        key = jnp.einsum("td,de->te", key, h_matrix)

        weights, _ = self.weights_proj(hidden_states)
        return query, key, weights

    def __call__(
        self,
        hidden_states: jax.Array,
        q_compressed: jax.Array,
        positions: jax.Array,
        rotary_emb: Any,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        *,
        compute_topk: bool,
    ) -> DSAIndexerOutput:
        """Run the complete Indexer pipeline for one full IndexShare layer.

        The owning module performs projection and delegates only the functional
        cache/scoring primitive.  Shared layers never instantiate or call this
        module.
        """

        q_idx, k_idx, idx_weights = self._project(
            hidden_states,
            q_compressed,
            positions,
            rotary_emb,
        )

        attention_backend = forward_batch.attn_backend
        metadata_owner = getattr(attention_backend, "full_attn_backend", attention_backend)
        metadata = getattr(metadata_owner, "forward_metadata", None)
        if metadata is None:
            raise ValueError("DSA Indexer requires initialized MLA forward metadata")

        index_cache = token_to_kv_pool.get_indexer_key_buffer(self.cache_slot)
        return update_index_cache_and_select(
            q_idx,
            k_idx,
            idx_weights,
            index_cache,
            metadata,
            index_topk=self.index_topk,
            compute_topk=compute_topk,
            topk_impl=self.topk_impl,
            one_token_per_seq=forward_batch.forward_mode.is_decode(),
            attention_data_partition_axis=self.attention_data_partition_axis,
        )


class Glm5Attention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        mesh: jax.sharding.Mesh,
        rope_theta: float = 1000000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        rms_norm_eps: float = None,
        use_qk_norm: bool = True,
        rotary_dim: int = 0,
        layer_id: int = 0,
        attention_bias: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
        use_absorbed: bool = True,
        has_indexer: bool = True,
        indexer_type: str = "full",
        index_topk: int = 2048,
        index_cache_slot: int | None = None,
        produces_topk: bool = True,
        is_dense_attention: bool = False,
        sparse_impl: str = "exact",
        topk_impl: str = "exact_lax",
        use_dsa_sparse: bool = False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.mesh = mesh
        self.num_heads = num_heads
        self.kv_head_num = num_kv_heads
        self.indexer_type = indexer_type
        self.produces_topk = produces_topk
        self.is_dense_attention = is_dense_attention
        self.sparse_impl = sparse_impl
        self.use_dsa_sparse = use_dsa_sparse
        if indexer_type not in ("full", "shared"):
            raise ValueError(f"unknown indexer_type {indexer_type!r} at layer {layer_id}")
        if sparse_impl not in ("page", "exact"):
            raise ValueError(f"unknown DSA sparse implementation: {sparse_impl}")

        self.qk_nope_head_dim = 192
        self.qk_rope_head_dim = 64
        self.qk_head_dim = 256
        self.v_head_dim = 256
        self.kv_lora_rank = 512
        self.q_lora_rank = 2048

        self.scaling = 256**-0.5

        self.use_qk_norm = use_qk_norm

        if use_qk_norm:
            self.q_norm = RMSNorm(256, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="q_norm")
            self.k_norm = RMSNorm(256, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="k_norm")
        else:
            self.q_norm = None
            self.k_norm = None

        self.q_a_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.q_lora_rank,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(
            self.q_lora_rank, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="q_a_layernorm"
        )
        self.q_b_proj = LinearBase(
            input_size=self.q_lora_rank,
            output_size=num_heads * self.qk_head_dim,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_b_proj",
        )
        self.kv_a_proj_with_mqa = LinearBase(
            input_size=hidden_size,
            output_size=self.kv_lora_rank + self.qk_rope_head_dim,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="kv_a_layernorm"
        )

        self.kv_b_proj = LinearBase(
            input_size=self.kv_lora_rank,
            output_size=num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="kv_b_proj",
        )

        self.o_proj = LinearBase(
            input_size=num_heads * self.v_head_dim,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )

        if has_indexer:
            if index_cache_slot is None:
                raise ValueError(f"full Indexer layer {layer_id} requires an index-cache slot")
            self.indexer = GlmDsaIndexer(
                hidden_size=hidden_size,
                q_lora_rank=self.q_lora_rank,
                index_head_dim=128,
                index_n_heads=32,
                index_topk=index_topk,
                cache_slot=index_cache_slot,
                mesh=mesh,
                topk_impl=topk_impl,
                dtype=dtype,
                scope_name="indexer",
            )
        else:
            self.indexer = None
        self.rotary_emb = RotaryEmbedding(
            head_size=self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=False,
            dtype=dtype,
            mesh=mesh,
        )

        self.use_absorbed = use_absorbed

        if use_absorbed:
            uk_axes = (None, "tensor", None)
            self.w_uk = nnx.Param(
                jnp.zeros(
                    (self.kv_lora_rank, num_heads, self.qk_nope_head_dim),
                    dtype=dtype,
                    out_sharding=P(*uk_axes),
                )
            )
            self.w_uv = nnx.Param(
                jnp.zeros(
                    (self.kv_lora_rank, num_heads, self.v_head_dim),
                    dtype=dtype,
                    out_sharding=P(*uk_axes),
                )
            )
            self.attn_mqa = RadixAttention(
                num_heads=num_heads,
                head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
                scaling=self.scaling,
                num_kv_heads=1,
                v_head_dim=self.kv_lora_rank,
                layer_id=layer_id,
            )
        else:
            self.w_uk = None
            self.w_uv = None
            self.attn_mqa = None

        self.attn_mha = RadixAttention(
            num_heads=num_heads,
            head_dim=self.qk_head_dim,
            scaling=self.scaling,
            num_kv_heads=num_heads,
            v_head_dim=self.qk_head_dim,
            layer_id=layer_id,
        )

    def post_load_weights(self):
        if not self.use_absorbed:
            return
        if self.kv_b_proj is None:
            return
        if hasattr(self.kv_b_proj, "weight"):
            raw_weight = self.kv_b_proj.weight.value
        else:
            wq = self.kv_b_proj.weight_q.value
            ws = self.kv_b_proj.weight_scale.value
            wq_f32 = wq.T.astype(jnp.float32)
            if ws.ndim == 3:
                in_blocks, _, n_out = ws.shape
                block_k = wq.shape[1] // in_blocks
                wq_f32 = wq_f32.reshape(in_blocks, block_k, n_out)
                wq_f32 = (wq_f32 * ws.astype(jnp.float32)).reshape(in_blocks * block_k, n_out)
            else:
                wq_f32 = wq_f32 * ws.astype(jnp.float32)[None, :]
            raw_weight = wq_f32.astype(jnp.bfloat16)
        w_kv = raw_weight.reshape(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        self.w_uk.value = w_kv[:, :, : self.qk_nope_head_dim]
        self.w_uv.value = w_kv[:, :, self.qk_nope_head_dim :]
        self.kv_b_proj = None

    def _forward_mqa(
        self,
        q_nope: jax.Array,
        q_rope: jax.Array,
        compressed: jax.Array,
        k_rope: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        topk_indices: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        # "thd,rhd->thr" — fp32 accumulate: on v7x the default bf16
        # accumulator drops enough precision on this small batched dot
        # (per-device [T, H/tp, 128]) that decode drifts into repetition
        # over 78 layers; v6e's default happens to be tighter. Cost is
        # negligible vs q/o_proj.
        ql_nope = jax.lax.dot_general(
            q_nope,
            self.w_uk.value,
            (((2,), (2,)), ((1,), (1,))),
            preferred_element_type=jnp.float32,
        ).astype(q_nope.dtype)
        ql_nope = ql_nope.transpose(1, 0, 2)

        c_kv_3d = compressed[:, None, :]
        attn_output, kv_fused = self.attn_mqa(
            ql_nope,
            c_kv_3d,
            c_kv_3d,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            q_rope=q_rope,
            k_rope=k_rope,
            topk_indices=topk_indices,
        )
        # "thr,rhd->thd" — fp32 accumulate; see ql_nope above.
        o_v = jax.lax.dot_general(
            attn_output,
            self.w_uv.value,
            (((2,), (0,)), ((1,), (1,))),
            preferred_element_type=jnp.float32,
        ).astype(attn_output.dtype)
        o_v = o_v.transpose(1, 0, 2)
        attn_output = o_v.reshape(-1, self.num_heads * self.v_head_dim)
        return attn_output, kv_fused

    def _forward_mha(
        self,
        q_nope: jax.Array,
        q_rope: jax.Array,
        compressed: jax.Array,
        k_rope: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        kv, _ = self.kv_b_proj(compressed)
        kv = kv.reshape(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = jnp.split(kv, [self.qk_nope_head_dim], axis=-1)

        k_rope = jnp.broadcast_to(
            k_rope,
            (k_rope.shape[0], self.num_heads, self.qk_rope_head_dim),
            out_sharding=P("data", "tensor", None),
        )

        q = jnp.concatenate([q_nope, q_rope], axis=-1)
        k = jnp.concatenate([k_nope, k_rope], axis=-1)

        attn_output, kv_fused = self.attn_mha(
            q, k, v, forward_batch=forward_batch, token_to_kv_pool=token_to_kv_pool
        )
        attn_output = attn_output.reshape(-1, self.num_heads * self.v_head_dim)
        return attn_output, kv_fused

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        prev_topk_indices: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array | None]:
        q_compressed, _ = self.q_a_proj(hidden_states)
        q_compressed = self.q_a_layernorm(q_compressed)

        indexer_output = None
        if self.use_dsa_sparse:
            if self.indexer_type == "full":
                if self.indexer is None:
                    raise ValueError(f"full Indexer layer {self.layer_id} has no Indexer module")
                compute_topk = self.produces_topk and (
                    self.sparse_impl == "exact" or forward_batch.forward_mode.is_decode()
                )
                indexer_output = self.indexer(
                    hidden_states,
                    q_compressed,
                    positions,
                    self.rotary_emb,
                    forward_batch,
                    token_to_kv_pool,
                    compute_topk=compute_topk,
                )
                selected_topk = indexer_output.topk_indices
            else:
                selected_topk = prev_topk_indices

            if (
                not self.is_dense_attention
                and self.sparse_impl == "exact"
                and selected_topk is None
            ):
                raise ValueError(
                    f"sparse shared layer {self.layer_id} requires top-k from a preceding full layer"
                )
            attention_topk = None if self.is_dense_attention else selected_topk
        else:
            attention_topk = None

        q, _ = self.q_b_proj(q_compressed)
        q = q.reshape(-1, self.num_heads, self.qk_head_dim)

        q_nope = q[:, :, : self.qk_nope_head_dim]
        q_rope = q[:, :, self.qk_nope_head_dim :]

        latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
        compressed, k_rope = jnp.split(latent_cache, [self.kv_lora_rank], axis=-1)
        compressed = self.kv_a_layernorm(compressed)

        k_rope = k_rope.reshape(-1, 1, self.qk_rope_head_dim)
        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        if self.use_absorbed:
            attn_output, kv_fused = self._forward_mqa(
                q_nope,
                q_rope,
                compressed,
                k_rope,
                forward_batch,
                token_to_kv_pool,
                topk_indices=attention_topk,
            )
        else:
            attn_output, kv_fused = self._forward_mha(
                q_nope, q_rope, compressed, k_rope, forward_batch, token_to_kv_pool
            )

        output, _ = self.o_proj(attn_output)
        return (
            output,
            kv_fused,
            indexer_output.index_cache if indexer_output is not None else None,
            indexer_output.topk_indices if indexer_output is not None else None,
        )


class Glm5MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        use_fused: bool = True,
    ) -> None:
        self.layer_id = layer_id
        self.mesh = mesh
        self.use_fused = use_fused

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="gate_proj",
        )

        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="up_proj",
        )

        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="down_proj",
        )

        self.act_fn = jax.nn.silu

        if use_fused:
            tp_size = mesh.shape["tensor"]
            local_inter_size = intermediate_size // tp_size

            # Dynamically choose block size (B_INTER) based on local intermediate size
            # to ensure that num_blocks is always a multiple of the TP size.
            if local_inter_size >= 128:
                self.b_inter = 128
            elif local_inter_size >= 64:
                self.b_inter = 64
            else:
                self.b_inter = 32

            pad_inter = (self.b_inter - (local_inter_size % self.b_inter)) % self.b_inter
            local_inter_size_padded = local_inter_size + pad_inter
            global_inter_size_padded = local_inter_size_padded * tp_size

            # Pre-allocate fused parameters with correct global shape and sharding
            # under the active constructor mesh context.
            self.w_gu = nnx.Param(
                jnp.zeros((hidden_size, global_inter_size_padded * 2), dtype=dtype),
                out_sharding=P(None, "tensor"),
            )
            self.w_d = nnx.Param(
                jnp.zeros((global_inter_size_padded, hidden_size), dtype=dtype),
                out_sharding=P("tensor", None),
            )

    def post_load_weights(self):
        if not self.use_fused:
            return
        if not hasattr(self.gate_proj, "weight"):
            # static fp8 checkpoint: gate_proj is already QuantizedLinear
            # (weight_q/weight_scale), fused-merge path from #1344 only
            # handles bf16 LinearBase. Fall back to unfused (forward checks
            # hasattr(self, "w_gu")).
            return

        wg = self.gate_proj.weight.value
        wu = self.up_proj.weight.value
        wd = self.down_proj.weight.value

        # Use dynamically chosen block size
        b_inter = self.b_inter
        hidden_size, local_inter_size = wg.shape

        # Pad local intermediate dimension to a multiple of b_inter
        pad_inter = (b_inter - (local_inter_size % b_inter)) % b_inter
        if pad_inter > 0:
            wg = jnp.pad(wg, ((0, 0), (0, pad_inter)), mode="constant")
            wu = jnp.pad(wu, ((0, 0), (0, pad_inter)), mode="constant")
            wd = jnp.pad(wd, ((0, pad_inter), (0, 0)), mode="constant")
            local_inter_size += pad_inter

        # Combine wg and wu block-by-block using jax.lax.reshape to explicitly
        # specify the sharding for the split/merged dimensions under JAX SPMD.
        num_blocks = local_inter_size // b_inter
        sharding_3d = jax.sharding.NamedSharding(self.mesh, P(None, "tensor", None))
        wg_reshaped = jax.lax.reshape(
            wg, (hidden_size, num_blocks, b_inter), out_sharding=sharding_3d
        )
        wu_reshaped = jax.lax.reshape(
            wu, (hidden_size, num_blocks, b_inter), out_sharding=sharding_3d
        )

        # Concat along block dimension and flatten
        w_gu = jnp.concatenate([wg_reshaped, wu_reshaped], axis=-1)

        sharding_2d = jax.sharding.NamedSharding(self.mesh, P(None, "tensor"))
        w_gu = jax.lax.reshape(w_gu, (hidden_size, local_inter_size * 2), out_sharding=sharding_2d)

        # Assign values directly to pre-allocated sharded parameters
        self.w_gu.value = w_gu
        self.w_d.value = wd

        # Free original projection modules to save HBM
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None

    def __call__(self, hidden_states: jax.Array):
        if self.use_fused and hasattr(self, "w_gu"):
            seq_len, _ = hidden_states.shape
            b_seq = 64 if seq_len <= 8 else 256

            return apply_fused_mlp_with_padding(
                hidden_states,
                self.w_gu.value,
                self.w_d.value,
                self.mesh,
                b_seq=b_seq,
                b_inter=self.b_inter,
            )

        # Fallback non-fused path
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel)
        return output


class Glm5DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        dsa_plan: GlmDsaLayerPlan | None = None,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_params = getattr(config, "rope_parameters", None) or {}
        rope_theta = getattr(config, "rope_theta", None) or rope_params.get("rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 131072)
        self.head_dim = getattr(config, "head_dim", None) or 128
        use_qk_norm = getattr(config, "use_qk_norm", True)

        partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
        rotary_dim = int(self.head_dim * partial_rotary_factor)

        # Dense-attention policy and IndexShare ownership are orthogonal. A
        # dense full layer can still seed top-k for sparse shared followers.
        if dsa_plan is None:
            dsa_plan = _build_dsa_layer_plans(config)[layer_id]
        has_indexer = dsa_plan.indexer_type == "full"
        use_dsa_sparse = getattr(config, "use_dsa_sparse", False)

        self.self_attn = Glm5Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=self.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=use_qk_norm,
            rotary_dim=rotary_dim,
            layer_id=layer_id,
            attention_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            mesh=mesh,
            use_absorbed=True,
            has_indexer=has_indexer,
            indexer_type=dsa_plan.indexer_type,
            index_topk=getattr(config, "index_topk", 2048),
            index_cache_slot=dsa_plan.index_cache_slot,
            produces_topk=dsa_plan.produces_topk,
            is_dense_attention=dsa_plan.is_dense_attention,
            sparse_impl=getattr(config, "dsa_sparse_impl", "exact"),
            topk_impl=getattr(config, "dsa_topk_impl", "exact_lax"),
            use_dsa_sparse=use_dsa_sparse,
        )

        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
        use_fused_mlp = getattr(config, "_sgl_use_fused_mlp", True)

        if layer_id < first_k_dense_replace:
            self.mlp = Glm5MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_id=layer_id,
                dtype=dtype,
                mesh=mesh,
                use_fused=use_fused_mlp,
            )
            self.is_moe_layer = False
            self.moe_gate = None
        else:
            router_dtype = jnp.float32
            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                num_experts=config.n_routed_experts,
                enable_expert_bias=True,
                weight_dtype=router_dtype,
                score_func=getattr(config, "scoring_func", "sigmoid"),
            )

            self.moe_backend = getattr(config, "moe_backend", MoEBackend.EPMOE)
            self.use_fused = self.moe_backend in (MoEBackend.FUSED, MoEBackend.FUSED_V2)
            num_shared_experts = getattr(config, "n_shared_experts", 0)
            use_inkernel_se = self.moe_backend == MoEBackend.FUSED_V2 and num_shared_experts > 0

            self.topk = TopK(
                topk=config.num_experts_per_tok,
                renormalize=config.norm_topk_prob,
                num_expert_group=getattr(config, "n_group", 1),
                topk_group=getattr(config, "topk_group", 1),
                routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
                layer_id=layer_id,
                mesh=mesh,
            )

            if self.moe_backend == MoEBackend.FUSED_V2:
                self.mlp = FusedEPMoEV2(
                    hidden_size=config.hidden_size,
                    num_experts=config.n_routed_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    intermediate_dim=config.moe_intermediate_size,
                    mesh=mesh,
                    ep_size=getattr(config, "ep_size", 1),
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_id,
                    renormalize_topk_logits=config.norm_topk_prob,
                    routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
                    use_grouped_topk=getattr(config, "n_group", 1) > 1,
                    num_groups=getattr(config, "n_group", 1),
                    top_k_groups=getattr(config, "topk_group", 1),
                    num_shared_experts=num_shared_experts if use_inkernel_se else 0,
                    moe_shared_expert_intermediate_size=config.moe_intermediate_size,
                    quantization_config=getattr(config, "quantization_config", None),
                )

                quant_config = getattr(config, "quantization_config", None)
                weight_block_size = (
                    getattr(quant_config, "weight_block_size", None) if quant_config else None
                )
                if (
                    use_inkernel_se
                    and getattr(quant_config, "is_static_checkpoint", False)
                    and weight_block_size is not None
                ):
                    block_n, block_k = map(int, weight_block_size)
                    shared_intermediate = config.moe_intermediate_size * num_shared_experts
                    if (
                        config.hidden_size % block_k
                        or shared_intermediate % block_n
                        or config.hidden_size % block_n
                        or shared_intermediate % block_k
                    ):
                        raise ValueError(
                            "GLM-5.2 shared-expert dimensions must be divisible by "
                            f"weight_block_size={weight_block_size}"
                        )
                    self.mlp.w1_shared_block_scale = nnx.Param(
                        jnp.zeros(
                            (shared_intermediate // block_n, config.hidden_size // block_k),
                            dtype=jnp.float32,
                        ),
                        out_sharding=P(None, None),
                    )
                    self.mlp.w3_shared_block_scale = nnx.Param(
                        jnp.zeros(
                            (shared_intermediate // block_n, config.hidden_size // block_k),
                            dtype=jnp.float32,
                        ),
                        out_sharding=P(None, None),
                    )
                    self.mlp.w2_shared_block_scale = nnx.Param(
                        jnp.zeros(
                            (config.hidden_size // block_n, shared_intermediate // block_k),
                            dtype=jnp.float32,
                        ),
                        out_sharding=P(None, None),
                    )
            elif self.use_fused:
                self.mlp = FusedEPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=config.n_routed_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    intermediate_dim=config.moe_intermediate_size,
                    mesh=mesh,
                    ep_size=getattr(config, "ep_size", 1),
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_id,
                    renormalize_topk_logits=config.norm_topk_prob,
                    routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
                    use_grouped_topk=getattr(config, "n_group", 1) > 1,
                    num_groups=getattr(config, "n_group", 1),
                    top_k_groups=getattr(config, "topk_group", 1),
                    num_shared_experts=getattr(config, "n_shared_experts", 0),
                    moe_shared_expert_intermediate_size=config.moe_intermediate_size,
                    quantization_config=getattr(config, "quantization_config", None),
                )
            else:
                self.mlp = EPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=config.n_routed_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    intermediate_dim=config.moe_intermediate_size,
                    mesh=mesh,
                    ep_size=getattr(config, "ep_size", 1),
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_id,
                    quantization_config=getattr(config, "quantization_config", None),
                )

            if num_shared_experts > 0 and not self.use_fused:
                self.shared_experts = Glm5MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size * num_shared_experts,
                    layer_id=layer_id,
                    dtype=dtype,
                    mesh=mesh,
                    use_fused=use_fused_mlp,
                )
            else:
                self.shared_experts = None
            self.is_moe_layer = True

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="input_layernorm",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="post_attention_layernorm",
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
        dispatch_info: ExpertLocationMetadata | None = None,
        prev_topk_indices: jax.Array | None = None,
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array | None,
        jax.Array | None,
        jax.Array | None,
    ]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, kv_fused, index_cache, fresh_topk_indices = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            prev_topk_indices=prev_topk_indices,
        )
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe_layer:
            if self.shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)
            else:
                shared_output = None
            router_logits = self.moe_gate(hidden_states)

            correction_bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
            topk_weights, topk_ids = self.topk(
                router_logits,
                correction_bias,
                dispatch_info=dispatch_info,
            )

            hidden_states = self.mlp(hidden_states, topk_weights, topk_ids)

            if shared_output is not None:
                hidden_states = hidden_states + shared_output
        else:
            hidden_states = self.mlp(hidden_states)
            topk_ids = None

        return hidden_states, residual, kv_fused, index_cache, fresh_topk_indices, topk_ids


class Glm5Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

        dsa_layer_plans = _build_dsa_layer_plans(config)
        self.layers = nnx.data(
            [
                Glm5DecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                    dsa_plan=dsa_layer_plans[i],
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype, scope_name="norm"
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> jax.Array:
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        residual = None
        layers_kv_fused = []
        layers_idx_fused = []
        layers_topk_ids = []
        prev_topk_indices = None
        for layer in self.layers:
            (
                hidden_states,
                residual,
                kv_fused,
                index_cache,
                fresh_topk_indices,
                topk_ids,
            ) = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
                dispatch_info=forward_batch.expert_location_metadata,
                prev_topk_indices=prev_topk_indices,
            )
            layers_kv_fused.append(kv_fused)
            if index_cache is not None:
                layers_idx_fused.append(index_cache)
            if layer.self_attn.indexer_type == "full":
                # A full layer starts a new IndexShare group.  ``None`` clears
                # stale state when that group never reaches sparse attention.
                prev_topk_indices = fresh_topk_indices
            layers_topk_ids.append(topk_ids)

        if residual is not None:
            hidden_states += residual

        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused, layers_idx_fused, layers_topk_ids


class Glm5ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        self.model = Glm5Model(config, dtype=self.dtype, mesh=mesh)
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            mesh=self.mesh,
            soft_cap=getattr(config, "final_logit_softcapping", None),
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools,
        logits_metadata: LogitsMetadata,
    ):
        kv_pool = memory_pools.token_to_kv_pool
        hidden_states, layers_kv_fused, layers_idx_fused, layers_topk_ids = self.model(
            forward_batch,
            kv_pool,
        )

        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)

        kv_update = (layers_kv_fused, layers_idx_fused) if layers_idx_fused else layers_kv_fused
        return output, {"token_to_kv_pool": kv_update}, True, layers_topk_ids

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_glm5_weight_mappings(model_config)
        loader.load_weights_from_safetensors(weight_mappings)

        for layer in self.model.layers:
            layer.self_attn.post_load_weights()
            if isinstance(getattr(layer, "mlp", None), FusedEPMoEV2):
                _requantize_glm5_shared_expert(layer.mlp)
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "post_load_weights"):
                layer.mlp.post_load_weights()
            if (
                hasattr(layer, "shared_experts")
                and layer.shared_experts is not None
                and hasattr(layer.shared_experts, "post_load_weights")
            ):
                layer.shared_experts.post_load_weights()
        logger.info("Absorbed MLA weights and Fused MLP weights processed successfully!")

        # Skipping scale inversion for BF16
        logger.info("Skipping scale inversion for BF16 model.")

    def _create_glm5_weight_mappings(self, model_config: ModelConfig) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }

        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )

        num_layers = self.config.num_hidden_layers
        first_k_dense_replace = getattr(self.config, "first_k_dense_replace", 0)
        indexer_types = getattr(self.config, "indexer_types", None)

        quant_config = getattr(model_config, "quantization_config", None)
        is_static_quant = quant_config is not None and quant_config.is_static_checkpoint

        hf_layer_indices = list(range(num_layers))
        for layer_idx in range(num_layers):
            target_idx = hf_layer_indices[layer_idx]
            layer_mappings = self._create_moe_layer_mappings(
                layer_idx,
                target_idx,
                target_idx < first_k_dense_replace,
                is_static_quant=is_static_quant,
                has_indexer=indexer_types is None or indexer_types[target_idx] == "full",
            )
            mappings.update(layer_mappings)

        return mappings

    def _create_moe_layer_mappings(
        self,
        layer_idx: int,
        target_idx: int,
        is_mlp_layer: bool,
        is_static_quant: bool = False,
        has_indexer: bool = True,
    ) -> dict:
        prefix = f"model.layers.{target_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        mappings = {
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        def add_linear(hf: str, tgt: str, sharding_std: tuple, force_unquant: bool = False):
            """Mirror deepseek_v3._create_layer_mappings.add_linear.

            HF weight is [out, in]. Unquantized → LinearBase.weight [in, out]
            (transpose=True, sharding=kernel_axes). Static FP8 → QuantizedLinear
            .weight_q [out, in] (transpose=False, sharding swapped) plus the
            block-wise weight_scale_inv sidecar. force_unquant covers modules in
            the FP8 checkpoint's modules_to_not_convert (indexer.weights_proj).
            """
            if force_unquant or not is_static_quant:
                mappings[f"{hf}.weight"] = WeightMapping(
                    target_path=f"{tgt}.weight", sharding=sharding_std, transpose=True
                )
                return
            sharding_q = (sharding_std[1], sharding_std[0])
            mappings[f"{hf}.weight"] = WeightMapping(
                target_path=f"{tgt}.weight_q", sharding=sharding_q, transpose=False
            )
            # Load 2D block scale [out_blocks, in_blocks] replicated: GLM-5.1 head_dim
            # 448 → out_blocks not always tp-divisible (kv_b_proj: 224 % 64 ≠ 0).
            # _maybe_expand_linear_block_scale runs after _shard_weight and expands to
            # [in_blocks, 1, n_out]; assignment into model_param then reshards to the
            # QuantizedLinear placeholder's 3D sharding.
            mappings[f"{hf}.weight_scale_inv"] = WeightMapping(
                target_path=f"{tgt}.weight_scale", sharding=(None, None), transpose=False
            )

        ap = f"{prefix}.self_attn"
        tp = f"{target_prefix}.self_attn"
        add_linear(f"{ap}.q_a_proj", f"{tp}.q_a_proj", (None, None))
        mappings[f"{ap}.q_a_layernorm.weight"] = WeightMapping(
            target_path=f"{tp}.q_a_layernorm.scale", sharding=(None,)
        )
        add_linear(f"{ap}.q_b_proj", f"{tp}.q_b_proj", (None, "tensor"))
        add_linear(f"{ap}.kv_a_proj_with_mqa", f"{tp}.kv_a_proj_with_mqa", (None, None))
        mappings[f"{ap}.kv_a_layernorm.weight"] = WeightMapping(
            target_path=f"{tp}.kv_a_layernorm.scale", sharding=(None,)
        )
        add_linear(f"{ap}.kv_b_proj", f"{tp}.kv_b_proj", (None, "tensor"))
        add_linear(f"{ap}.o_proj", f"{tp}.o_proj", ("tensor", None))

        if has_indexer:
            add_linear(f"{ap}.indexer.wq_b", f"{tp}.indexer.wq_b", (None, None))
            add_linear(f"{ap}.indexer.wk", f"{tp}.indexer.wk", (None, None))
            # weights_proj is in modules_to_not_convert (HF: indexers_proj) → unquantized.
            add_linear(
                f"{ap}.indexer.weights_proj",
                f"{tp}.indexer.weights_proj",
                (None, None),
                force_unquant=True,
            )
            mappings[f"{ap}.indexer.k_norm.weight"] = WeightMapping(
                target_path=f"{tp}.indexer.k_norm.weight", sharding=(None,)
            )
            mappings[f"{ap}.indexer.k_norm.bias"] = WeightMapping(
                target_path=f"{tp}.indexer.k_norm.bias", sharding=(None,)
            )

        if is_mlp_layer:
            add_linear(
                f"{prefix}.mlp.gate_proj", f"{target_prefix}.mlp.gate_proj", (None, "tensor")
            )
            add_linear(f"{prefix}.mlp.up_proj", f"{target_prefix}.mlp.up_proj", (None, "tensor"))
            add_linear(
                f"{prefix}.mlp.down_proj", f"{target_prefix}.mlp.down_proj", ("tensor", None)
            )
        else:
            mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )
            # GLM-4 uses e_score_correction_bias
            mappings[f"{prefix}.mlp.gate.e_score_correction_bias"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.bias", sharding=(None,)
            )

            num_logical_experts = self.config.n_routed_experts
            moe_backend = getattr(self.config, "moe_backend", "epmoe")

            moe_mappings = create_moe_weights_mapping(
                prefix=prefix,
                target_prefix=target_prefix,
                num_experts=num_logical_experts,
                expert_type_names=("gate_proj", "up_proj", "down_proj"),
                moe_backend=moe_backend,
                physical_to_logical_map=None,  # Handle physical mapping if needed later
            )

            if is_static_quant:
                new_moe_mappings = {}

                for key, mapping in moe_mappings.items():
                    target_param = mapping.target_path[0]
                    src_paths = mapping.target_path[1:]

                    new_moe_mappings[key] = WeightMapping(
                        target_path=[target_param] + src_paths,
                        sharding=mapping.sharding,
                        transpose=True,
                        concat_axis=mapping.concat_axis,
                        physical_to_logical_map=mapping.physical_to_logical_map,
                    )

                    scale_key = key + "_scale"
                    target_scale_param = target_param + "_scale"
                    scale_src_paths = [p.replace(".weight", ".weight_scale_inv") for p in src_paths]

                    # Stacked HF scale is [E, out_blocks, in_blocks]. Load EP-sharded
                    # and replicated on the block dims (matches deepseek_v3); the
                    # loader's _maybe_convert_epmoe_scale_for_kernel handles the
                    # [E, out_blocks, k_blocks] → [E, k_blocks, 1, n_out] expand.
                    new_moe_mappings[scale_key] = WeightMapping(
                        target_path=[target_scale_param] + scale_src_paths,
                        sharding=("expert", None, None),
                        transpose=False,
                        concat_axis=mapping.concat_axis,
                        physical_to_logical_map=mapping.physical_to_logical_map,
                    )
                moe_mappings = new_moe_mappings

            mappings.update(moe_mappings)

            num_shared = getattr(self.config, "n_shared_experts", 0)
            if num_shared > 0:
                sp = f"{prefix}.mlp.shared_experts"
                if moe_backend == "fused_v2":
                    for hf_name, target_name in (
                        ("gate_proj", "w1_shared"),
                        ("up_proj", "w3_shared"),
                        ("down_proj", "w2_shared"),
                    ):
                        target_path = f"{target_prefix}.mlp.{target_name}"
                        mappings[f"{sp}.{hf_name}.weight"] = WeightMapping(
                            target_path=target_path,
                            sharding=(None, None),
                            transpose=True,
                        )
                        if is_static_quant:
                            mappings[f"{sp}.{hf_name}.weight_scale_inv"] = WeightMapping(
                                target_path=f"{target_path}_block_scale",
                                sharding=(None, None),
                                transpose=False,
                            )
                else:
                    st = f"{target_prefix}.shared_experts"
                    add_linear(f"{sp}.gate_proj", f"{st}.gate_proj", (None, "tensor"))
                    add_linear(f"{sp}.up_proj", f"{st}.up_proj", (None, "tensor"))
                    add_linear(f"{sp}.down_proj", f"{st}.down_proj", ("tensor", None))

        return mappings


class GlmMoeDsaForCausalLM(Glm5ForCausalLM):
    @classmethod
    def patch_model_config(cls, mc: ModelConfig) -> None:
        from sgl_jax.srt.configs.model_config import AttentionArch

        # GLM-5 uses 256 for attention head dim (192 nope + 64 pe)
        mc.head_dim = 256
        mc.hf_config.head_dim = 256
        mc.v_head_dim = getattr(mc.hf_text_config, "v_head_dim", 256)
        # GLM-5 uses MLA architecture
        mc.attention_arch = AttentionArch.MLA
        # GLM-5.1-FP8 ships modules_to_not_convert with HF naming (e.g.
        # `self_attn.indexers_proj`); translate to sglang-jax module paths so
        # quantize_model leaves the unquantized indexer head-gate as LinearBase.
        if mc.quantization_config is not None and mc.quantization_config.is_static_checkpoint:
            mc.quantization_config.ignored_layers = list(
                mc.quantization_config.ignored_layers or []
            ) + ["indexer.weights_proj"]
            # indexer.wk has out_dim=128 == block_size_out (single N-block); the
            # narrow-N guard would reject it. Keep the checkpoint-compatible
            # exception used by the existing GLM/DeepSeek Indexer loading path.
            mc.quantization_config.allow_narrow_n_blockwise = True
        # Under dynamic (in-framework) quant, Glm5MLP.post_load_weights merges
        # BF16 w_gu/w_d and nulls gate/up/down_proj *before* quantize_model
        # runs, so the fused weights bypass quantization and regress decode
        # TPOT on HBM-bound hardware (#1378). Keep the unfused path there so
        # the LinearBase modules get quantized as before.
        # Static fp8 checkpoint also breaks fused: gate_proj/up_proj/down_proj
        # become QuantizedLinear (no .weight), so post_load_weights cannot
        # populate w_gu/w_d and the abstract ShapeDtypeStruct placeholders
        # leak into jit inputs. Keep fused for bf16-only.
        mc.hf_config._sgl_use_fused_mlp = mc.quantization_config is None


EntryClass = [Glm5ForCausalLM, GlmMoeDsaForCausalLM]
