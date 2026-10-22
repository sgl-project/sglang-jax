import copy
import json
import logging
import math
import os
from functools import partial
from typing import Any

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig, MoEBackend
from sgl_jax.srt.environ import envs
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.kernels.collectives.oneshot_allreduce import (
    defer_reduce,
    force_fused_reduce,
    fused_all_reduce,
    fused_moe_enabled,
)
from sgl_jax.srt.kernels.fused_mlp import apply_fused_mlp_with_padding
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import (
    LinearBase,
    fuse_quantized_linears,
    prepad_replicated_quantized_linears,
)
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
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec
from sgl_jax.srt.utils.quantization.int4_experts import routed_expert_scale_suffix
from sgl_jax.srt.utils.quantization.quantization_utils import (
    dequantize_tensor,
    quantize_tensor,
)

logger = logging.getLogger(__name__)
_LOGGED_ONCE: set[str] = set()


def _log_once(msg: str) -> None:
    if msg not in _LOGGED_ONCE:
        _LOGGED_ONCE.add(msg)
        logger.info(msg)


_INDEXER_ROPE_CONCAT = envs.SGLANG_JAX_INDEXER_ROPE_CONCAT.get()


# Fuse q_a_proj + kv_a_proj_with_mqa + indexer.wk (all replicated, same
# hidden_states input) into one fp8 blockwise matmul per layer after weight
# load. 3 -> 1 launches/layer; outputs are byte-identical slices.
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


def _load_shared_expert(inputs, *, mesh, quantized_dtype):
    arrays = [
        jax.make_array_from_callback(
            value.shape,
            jax.sharding.NamedSharding(mesh, P()),
            lambda index, value=value: value[index],
        )
        for value in inputs
    ]
    with jax.set_mesh(mesh):
        weight, scale = _requantize_blockwise_shared_weight(
            arrays[0].T, arrays[1], quantized_dtype=quantized_dtype
        )
        return weight, scale.reshape(1, 1, -1)


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


class GlmDsaIndexer(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        index_head_dim: int,
        index_n_heads: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        scope_name: str = "indexer",
    ):
        self.head_dim = index_head_dim
        self.n_head = index_n_heads
        self.mesh = mesh

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

    def project(
        self,
        hidden_states: jax.Array,
        qr: jax.Array,
        positions: jax.Array,
        rotary_emb: Any,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Indexer projections only (query, key, per-head weights); no top-k.

        Returns query [T, n_head, head_dim], key [T, head_dim], weights [T, n_head]
        after RoPE + Hadamard, ready for the DSA backend's streamindex_topk.
        """
        query, _ = self.wq_b(qr)
        query = query.reshape(-1, self.n_head, self.head_dim)

        if key is None:
            key, _ = self.wk(hidden_states)
        key = self.k_norm(key)

        rope_dim = 64
        q_rope = query[:, :, :rope_dim]
        k_rope = key[:, :rope_dim][:, None, :]
        q_rope, k_rope = rotary_emb(positions, q_rope, k_rope)
        if _INDEXER_ROPE_CONCAT:
            # concat instead of at[].set: avoids a read-modify-write of the
            # full [T, n_head, head_dim] tensor (values are identical).
            query = jnp.concatenate((q_rope, query[:, :, rope_dim:]), axis=-1)
            key = jnp.concatenate((k_rope.squeeze(1), key[:, rope_dim:]), axis=-1)
        else:
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
        qr: jax.Array,
        positions: jax.Array,
        rotary_emb: Any,
        key: jax.Array | None = None,
    ) -> jax.Array:
        # 1. Project Query and Key
        query, _ = self.wq_b(qr)
        query = query.reshape(-1, self.n_head, self.head_dim)

        if key is None:
            key, _ = self.wk(hidden_states)
        key = self.k_norm(key)

        # Apply RoPE
        rope_dim = 64
        q_rope = query[:, :, :rope_dim]
        k_rope = key[:, :rope_dim]
        k_rope = k_rope[:, None, :]  # Add head dim for RoPE

        q_rope, k_rope = rotary_emb(positions, q_rope, k_rope)
        k_rope = k_rope.squeeze(1)  # Remove head dim

        if _INDEXER_ROPE_CONCAT:
            query = jnp.concatenate((q_rope, query[:, :, rope_dim:]), axis=-1)
            key = jnp.concatenate((k_rope, key[:, rope_dim:]), axis=-1)
        else:
            query = query.at[:, :, :rope_dim].set(q_rope)
            key = key.at[:, :rope_dim].set(k_rope)

        # Apply Hadamard Transform
        h_matrix = get_hadamard_matrix(128)
        h_matrix = h_matrix * (128**-0.5)

        query = jnp.einsum("thd,de->the", query, h_matrix)
        key = jnp.einsum("td,de->te", key, h_matrix)

        # 2. Compute Logits (simplified dense dot product)
        key_replicated = jax.sharding.reshard(
            key, jax.sharding.NamedSharding(self.mesh, P(None, None))
        )
        logits = jnp.einsum("ijk,lk->ijl", query, key_replicated)

        # 3. Apply weights_proj
        weights, _ = self.weights_proj(hidden_states)

        # Scale and apply weights
        scaling = self.head_dim**-0.5
        logits = logits * scaling * weights[:, :, None]

        # 4. Top-K Selection (Top-1 for now to match dummy shape [T, n_head])
        _, topk_ids = jax.lax.top_k(logits, 1)
        topk_ids = topk_ids.squeeze(-1)

        return topk_ids


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
        use_dsa_sparse: bool = False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.mesh = mesh
        self.num_heads = num_heads
        self.kv_head_num = num_kv_heads
        self.indexer_type = indexer_type
        self.use_dsa_sparse = use_dsa_sparse

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
            self.q_lora_rank,
            epsilon=rms_norm_eps,
            param_dtype=jnp.float32,
            dtype=dtype,
            scope_name="q_a_layernorm",
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
            self.kv_lora_rank,
            epsilon=rms_norm_eps,
            param_dtype=jnp.float32,
            dtype=dtype,
            scope_name="kv_a_layernorm",
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
            self.indexer = GlmDsaIndexer(
                hidden_size=hidden_size,
                q_lora_rank=self.q_lora_rank,
                index_head_dim=128,
                index_n_heads=32,
                mesh=mesh,
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

    def prepare_weight_loading(self, loader, mappings, prefix):
        from sgl_jax.srt.layers.weight_loading import prepare_absorbed_mla

        return prepare_absorbed_mla(self, loader, mappings, prefix)

    def fuse_in_proj(self):
        """Fold q_a_proj, kv_a_proj_with_mqa and
        indexer.wk into one fp8 blockwise matmul. Runs after WeightLoader.load, once the
        weights are concrete (the old post_load_weights hook no longer exists)."""
        # NOTE: nnx Pytree attributes are typed on first assignment (static vs
        # data), so the fused module is only assigned when built (never pre-set
        # to None) and the per-module linears stay in place; the forward path
        # checks ``getattr(self, "fused_in_proj", None)``.
        if getattr(self, "fused_in_proj", None) is None:
            mods = [self.q_a_proj, self.kv_a_proj_with_mqa]
            member_paths = ["q_a_proj", "kv_a_proj_with_mqa"]
            if self.indexer is not None:
                mods.append(self.indexer.wk)
                member_paths.append("indexer/wk")
            fused = fuse_quantized_linears(
                mods, scope_name="fused_in_proj", member_paths=member_paths
            )
            if fused is not None:
                fused_mod, splits = fused
                self._fused_in_splits = tuple(splits)
                self.fused_in_proj = fused_mod
                # Release the now-unused per-module weights (~17 MB/layer fp8 +
                # scales) so the fusion does not shrink the KV pool: keep the
                # modules (nnx attribute typing) but point their params at
                # empty arrays of the same dtype/rank.
                for m in mods:
                    wq = m.weight_q.value
                    ws = m.weight_scale.value
                    m.weight_q = nnx.Param(jnp.zeros((0, wq.shape[1]), wq.dtype))
                    m.weight_scale = nnx.Param(jnp.zeros(ws.shape[:2] + (0,), ws.dtype))
                logger.info(
                    "fused_in_proj built: splits=%s in=%d",
                    self._fused_in_splits,
                    fused_mod.weight_q.value.shape[1],
                )

    def _forward_mqa(
        self,
        q_nope: jax.Array,
        q_rope: jax.Array,
        compressed: jax.Array,
        k_rope: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        **dsa_kwargs,
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
            **dsa_kwargs,
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
        dsa_topk_in: jax.Array | None = None,
        dsa_topk_pages_in: jax.Array | None = None,
        dsa_topk_reuse: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        k_idx_raw = None
        hs_plain = hidden_states
        if getattr(self, "fused_in_proj", None) is not None:
            fused_out, _ = self.fused_in_proj(hidden_states)
            # split at every segment end so the build-time zero-padding tail
            # (if any) lands in its own, unused trailing part
            bounds = [
                sum(self._fused_in_splits[: k + 1]) for k in range(len(self._fused_in_splits))
            ]
            parts = jnp.split(fused_out, bounds, axis=-1)
            q_compressed, latent_cache = parts[0], parts[1]
            if len(parts) > 2:
                k_idx_raw = parts[2]
        else:
            q_compressed, _ = self.q_a_proj(hidden_states)
            latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
        q_compressed = self.q_a_layernorm(q_compressed)
        q, _ = self.q_b_proj(q_compressed)
        q = q.reshape(-1, self.num_heads, self.qk_head_dim)

        dsa_kwargs = {}
        if self.use_dsa_sparse:
            dsa_kwargs["indexer_type"] = self.indexer_type
            dsa_kwargs["dsa_topk_in"] = dsa_topk_in
            dsa_kwargs["dsa_topk_pages_in"] = dsa_topk_pages_in
            dsa_kwargs["dsa_topk_reuse"] = dsa_topk_reuse
            if self.indexer is not None:
                q_idx, k_idx, idx_w = self.indexer.project(
                    hs_plain, q_compressed, positions, self.rotary_emb, key=k_idx_raw
                )
                dsa_kwargs["q_idx"] = q_idx
                dsa_kwargs["k_idx"] = k_idx
                dsa_kwargs["idx_weights"] = idx_w
        elif self.indexer is not None:
            _ = self.indexer(hs_plain, q_compressed, positions, self.rotary_emb, key=k_idx_raw)

        q_nope = q[:, :, : self.qk_nope_head_dim]
        q_rope = q[:, :, self.qk_nope_head_dim :]

        compressed, k_rope = jnp.split(latent_cache, [self.kv_lora_rank], axis=-1)
        compressed = self.kv_a_layernorm(compressed)

        k_rope = k_rope.reshape(-1, 1, self.qk_rope_head_dim)
        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        if self.use_absorbed:
            attn_output, kv_fused = self._forward_mqa(
                q_nope, q_rope, compressed, k_rope, forward_batch, token_to_kv_pool, **dsa_kwargs
            )
        else:
            attn_output, kv_fused = self._forward_mha(
                q_nope, q_rope, compressed, k_rope, forward_batch, token_to_kv_pool
            )

        output, _ = self.o_proj(attn_output)
        return output, kv_fused


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

    def prepare_weight_loading(self, loader, mappings, prefix):
        if not self.use_fused or not hasattr(self.gate_proj, "weight"):
            return mappings
        targets = tuple(
            prefix + "." + name + ".weight" for name in ("gate_proj", "up_proj", "down_proj")
        )
        by_target = {
            spec.target_path: source
            for source, spec in mappings.items()
            if isinstance(spec.target_path, str)
        }
        if not any(target in by_target for target in targets):
            return mappings
        sources = tuple(by_target[target] for target in targets)
        mappings = {key: spec for key, spec in mappings.items() if key not in sources}
        mappings[sources[0]] = WeightSpec(
            [prefix + ".w_gu", prefix + ".w_d"],
            sources=sources,
            recipe=self._pack_weights,
        )
        self.gate_proj = self.up_proj = self.down_proj = None
        return mappings

    def _pack_weights(self, inputs):
        wg, wu, wd = (weight.T.astype(self.w_gu.value.dtype) for weight in inputs)
        hidden, intermediate = wg.shape
        pad = self.w_gu.value.shape[1] // 2 - intermediate
        wg, wu = (np.pad(weight, ((0, 0), (0, pad))) for weight in (wg, wu))
        wd = np.pad(wd, ((0, pad), (0, 0)))
        wg, wu = (weight.reshape(hidden, -1, self.b_inter) for weight in (wg, wu))
        packed = np.concatenate((wg, wu), axis=-1).reshape(hidden, -1)
        result = []
        for weight, axes in ((packed, (None, "tensor")), (wd, ("tensor", None))):
            result.append(
                jax.make_array_from_callback(
                    weight.shape,
                    jax.sharding.NamedSharding(self.mesh, P(*axes)),
                    lambda index, weight=weight: weight[index],
                )
            )
        return tuple(result)

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
        if getattr(self.down_proj, "kernel_axes", (None, None))[0] is None:
            # Block-quantized down_proj whose reduce dim does not split into whole
            # blocks across TP (e.g. the 2048-wide shared expert at tp32: 64 per
            # shard < 128-block) runs with a replicated reduce axis; gate/up still
            # emit the tensor-sharded activation, so gather it first.
            sh = jax.typeof(intermediate_parallel).sharding
            if (
                isinstance(sh, NamedSharding)
                and sh.spec
                and len(sh.spec) > 1
                and sh.spec[1] is not None
            ):
                intermediate_parallel = jax.sharding.reshard(
                    intermediate_parallel, NamedSharding(sh.mesh, P(sh.spec[0], None))
                )
        output, _ = self.down_proj(intermediate_parallel)
        return output


def fused_moe_reduce_supported(mesh, shared_experts, use_fused: bool) -> bool:
    """Whether a MoE layer may defer its reduces and finish routed + shared with one
    ``fused_all_reduce`` over the model ``tensor`` axis. Three conditions, each one a wrong
    sum otherwise:

    * shared experts live outside the fused MoE kernel (``use_fused`` keeps them in-kernel);
    * the model ``tensor`` axis spans every device: the deferred EPMoE combine is a psum over
      the ``expert`` axis of the flattened device list, which only equals the ``tensor``
      all-reduce when no other mesh axis (``data``) is larger than one;
    * the shared down projection is a row-parallel ``QuantizedLinear`` (reduce axis
      ``tensor``), whose psum goes through ``maybe_oneshot_psum`` and can be deferred.
      ``LinearBase`` reduces inside ``dot_general`` and a quantized layer whose reduce axis fell
      back to replicated never reduces; both already return the full shared output, which the
      fused all-reduce would multiply by the tensor size.
    """
    from sgl_jax.srt.layers.linear import QuantizedLinear

    if shared_experts is None or use_fused:
        return False
    sizes = dict(mesh.shape)
    if math.prod(sizes.values()) != sizes.get("tensor", 0):
        return False
    down = getattr(shared_experts, "down_proj", None)
    if not isinstance(down, QuantizedLinear):
        return False
    return tuple(down.kernel_axes or (None, None))[0] == "tensor"


class Glm5DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        force_moe: bool = False,
    ):
        self.layer_id = layer_id
        self.mesh = mesh
        self.hidden_size = config.hidden_size
        rope_theta = config.rope_parameters["rope_theta"]
        rope_scaling = config.rope_parameters
        max_position_embeddings = getattr(config, "max_position_embeddings", 131072)
        self.head_dim = getattr(config, "head_dim", None) or 128
        # GLM-5.x checkpoints carry no self_attn.q_norm/k_norm tensors and the
        # attention forward never applies them; creating the params only leaves
        # unloaded parameters behind (the loader's final validation rejects them).
        use_qk_norm = getattr(config, "use_qk_norm", False)

        partial_rotary_factor = rope_scaling.get("partial_rotary_factor", 0.5)
        rotary_dim = int(self.head_dim * partial_rotary_factor)

        # GLM-5.2 IndexShare: layers tagged "shared" reuse the previous "full"
        # layer's top-k and ship no indexer weights. Dense MLA discards indexer
        # output anyway, so just skip building the module on shared layers.
        indexer_types = getattr(config, "indexer_types", None)
        indexer_type = (
            "full"
            if (indexer_types is None or layer_id >= len(indexer_types))
            else indexer_types[layer_id]
        )
        has_indexer = indexer_type == "full"
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
            indexer_type=indexer_type,
            use_dsa_sparse=use_dsa_sparse,
        )

        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
        use_fused_mlp = getattr(config, "_sgl_use_fused_mlp", True)

        if layer_id < first_k_dense_replace and not force_moe:
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
                # GLM-5.2 checkpoints ship the gate weight in BF16 and the
                # e_score_correction_bias in F32: store both checkpoint-native,
                # keep the router dot in f32 (unchanged numerics).
                kernel_dtype=jnp.bfloat16,
                compute_dtype=jnp.float32,
                bias_dtype=jnp.float32,
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
            param_dtype=jnp.float32,
            dtype=dtype,
            scope_name="input_layernorm",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=jnp.float32,
            dtype=dtype,
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
        dsa_topk_in: jax.Array | None = None,
        dsa_topk_pages_in: jax.Array | None = None,
        dsa_topk_reuse: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        hidden_states, residual, kv_fused = self.attn_block(
            positions,
            hidden_states,
            forward_batch,
            token_to_kv_pool,
            residual,
            dsa_topk_in=dsa_topk_in,
            dsa_topk_pages_in=dsa_topk_pages_in,
            dsa_topk_reuse=dsa_topk_reuse,
        )
        hidden_states, topk_ids = self.mlp_block(hidden_states, dispatch_info)
        return hidden_states, residual, kv_fused, topk_ids

    def attn_block(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
        *,
        dsa_topk_in: jax.Array | None = None,
        dsa_topk_pages_in: jax.Array | None = None,
        dsa_topk_reuse: bool = False,
    ):
        """Input norm -> attention (incl. o_proj and its all-reduce) -> residual -> post-attention
        norm. Returns ``(hidden_states, residual, kv_fused)``; ``mlp_block`` finishes the layer."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            dsa_topk_in=dsa_topk_in,
            dsa_topk_pages_in=dsa_topk_pages_in,
            dsa_topk_reuse=dsa_topk_reuse,
        )
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return hidden_states, residual, kv_fused

    def fused_reduce_supported(self) -> bool:
        """Whether this layer may take the fused MoE reduce path (``defer_reduce`` around the
        shared experts + EPMoE combine, one ``fused_all_reduce``); see
        :func:`fused_moe_reduce_supported`."""
        return self.is_moe_layer and fused_moe_reduce_supported(
            self.mesh, self.shared_experts, self.use_fused
        )

    def mlp_block(
        self, hidden_states: jax.Array, dispatch_info: ExpertLocationMetadata | None = None
    ):
        """MoE (routed + shared, incl. the combine all-reduce) or dense MLP on the
        post-attention-norm hidden states. Returns ``(hidden_states, topk_ids)``."""
        value, topk_ids, deferred = self.mlp_partial(hidden_states, dispatch_info)
        return self.mlp_reduce(value, deferred), topk_ids

    def mlp_reduce(self, value: jax.Array, deferred: bool) -> jax.Array:
        """Finish ``mlp_partial``: the fused all-reduce of the routed + shared partial sums
        when ``deferred``, identity otherwise."""
        return fused_all_reduce(value, self.mesh) if deferred else value

    def mlp_partial(
        self, hidden_states: jax.Array, dispatch_info: ExpertLocationMetadata | None = None
    ):
        """``mlp_block`` up to the layer-level all-reduce. Returns ``(value, topk_ids, deferred)``:
        on the fused all-reduce path ``value`` is the per-device partial sum (routed + shared)
        and ``deferred`` is True (``mlp_reduce`` must finish it); otherwise ``value`` is final."""
        if self.is_moe_layer:
            # Opt-in (SGLANG_JAX_ONESHOT_AR_MIN_BYTES + SITES=fused): let the shared-expert
            # down_proj and the EPMoE combine return per-device partials and reduce their sum
            # with one Pallas all-reduce -- the single collective XLA would otherwise emit for
            # psum(shared) + psum(routed). Only valid when the MoE output is fully replicated.
            fuse_ar = fused_moe_enabled(hidden_states) and self.fused_reduce_supported()
            router_logits = self.moe_gate(hidden_states)

            correction_bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
            topk_weights, topk_ids = self.topk(
                router_logits,
                correction_bias,
                dispatch_info=dispatch_info,
            )

            if fuse_ar:
                with defer_reduce():
                    shared_output = self.shared_experts(hidden_states)
                    hidden_states = self.mlp(hidden_states, topk_weights, topk_ids)
                return hidden_states + shared_output, topk_ids, True
            else:
                if self.shared_experts is not None:
                    shared_output = self.shared_experts(hidden_states)
                else:
                    shared_output = None

                hidden_states = self.mlp(hidden_states, topk_weights, topk_ids)

                if shared_output is not None:
                    hidden_states = hidden_states + shared_output
        else:
            hidden_states = self.mlp(hidden_states)
            topk_ids = None

        return hidden_states, topk_ids, False

    def mlp_dispatch(
        self, hidden_states: jax.Array, dispatch_info: ExpertLocationMetadata | None = None
    ):
        """PIN=4 stage 1 of ``mlp_partial`` (MoE layers on the fused all-reduce path only):
        router, top-k, shared experts (deferred partial) and ``EPMoE.dispatch_stage`` (token
        sort + SparseCore dispatch gather). Returns a dict for ``mlp_experts``; ``d["anchor"]``
        is a tiny array that depends on this window's dispatch gather."""
        fuse_ar = fused_moe_enabled(hidden_states) and self.fused_reduce_supported()
        if not fuse_ar:
            raise NotImplementedError(
                "mlp_dispatch needs an EPMoE layer with deferrable shared experts on the fused "
                "all-reduce path (SGLANG_JAX_ONESHOT_AR_SITES=fused / force_fused_reduce); "
                "callers check fused_reduce_supported() and use mlp_partial otherwise"
            )
        router_logits = self.moe_gate(hidden_states)
        correction_bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
        topk_weights, topk_ids = self.topk(
            router_logits, correction_bias, dispatch_info=dispatch_info
        )
        with defer_reduce():
            shared_output = self.shared_experts(hidden_states)
        state, anchor = self.mlp.dispatch_stage(hidden_states, topk_ids)
        return dict(
            state=state,
            anchor=anchor,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared=shared_output,
        )

    def mlp_experts(self, d, combine_pin: jax.Array | None = None):
        """PIN=4 stage 2: grouped matmuls + combine (``combine_pin`` keeps the SparseCore combine
        behind that array's producer) + shared partial. Returns ``(value, topk_ids, True)`` like
        ``mlp_partial`` on the fused path."""
        with defer_reduce():
            routed = self.mlp.experts_stage(d["state"], d["topk_weights"], combine_pin=combine_pin)
        return routed + d["shared"], d["topk_ids"], True


# ── prefill micro-batching (DSA_PREFILL_MICROBATCH, default 2) ──────────────
# Run an EXTEND chunk as N consecutive token windows inside one forward: at
# every layer the windows go through the layer one after the other, so XLA
# can schedule window A's SparseCore all-reduces (MoE combine, o_proj site;
# ~27% of a 4k chunk on v7x tp8) under window B's qmm / gmm / attend and vice
# versa. Window B attends over window A's KV of the same layer through the
# pool overlay below (the kernels return the rewritten paged buffers), and
# its attention metadata is the same chunk re-expressed as a chunked-prefill
# continuation (``window_forward_metadata``). Numerically each window is the
# plain prefill of its own rows; only the schedule changes. Pair it with
# SGLANG_JAX_PREFILL_XLA_OPTIONS=xla_jf_crs_combiner_threshold_in_bytes=1 --
# otherwise XLA's all-reduce combiner merges the two windows' same-shaped
# all-reduces back into one and nothing overlaps.
_PREFILL_MICROBATCH = envs.DSA_PREFILL_MICROBATCH.get()
# below this many (padded) tokens a chunk is not split (1k prompts: fixed
# per-forward costs dominate, halves would run below MXU-efficient M)
_PREFILL_MICROBATCH_MIN_TOKENS = envs.DSA_PREFILL_MICROBATCH_MIN_TOKENS.get()
# The window schedule ("PIN=7", 10-03; the earlier variants 1-6 and the later 8-13 were measured and
# dropped): the windows run block-interleaved per layer (A.attn, B.attn, then the
# MoE block), the MoE all-reduce is taken to the model level (force_fused_reduce: one fused_all_reduce
# of routed + shared partials) and every all-reduce consumer is pinned behind the OTHER window's
# pre-all-reduce value with a zero-valued data edge (_pin_after; gp92 HLO: XLA schedules call-done right
# before the consumer, so where the consumer sits decides what the all-reduce overlaps). Two windows add
# SparseCore-stream ordering for the MoE all-reduces (A.dispatch -> A.experts -> B.dispatch ->
# A.all-reduce (operand pinned behind B's dispatch gather) -> B.experts (combine pinned behind A's
# post-all-reduce value) -> B.all-reduce; the SC stream completes in start order) and one XLA
# scheduling group per layer holding window B's MoE all-reduce of layer L and window A's attend
# kernel of layer L+1 (dsa_sparse_backend.PREFILL_SCHED_GROUP), see _microbatch_pinned_layers.
# Needs xla_jf_enable_producer_consumer_multi_output_fusion=false in SGLANG_JAX_PREFILL_XLA_OPTIONS.


def _pin_after(x: jax.Array, anchor: jax.Array) -> jax.Array:
    """``x`` with a scheduling dependency on ``anchor`` and the same value: adds
    ``anchor[0, 0] * 0`` (XLA keeps float ``mul(_, 0)`` for NaN/Inf semantics, so
    the edge survives optimization; the broadcast add fuses into the consumer)."""
    tiny = anchor[tuple(slice(0, 1) for _ in range(anchor.ndim))].astype(jnp.float32)
    # 0 even for a NaN / Inf anchor (tiny * 0 would be NaN); still a real data edge
    zero = jnp.where(jnp.isfinite(tiny), tiny * 0, 0.0).reshape(()).astype(x.dtype)
    return x + zero


def prefill_microbatch_bounds(num_tokens: int, parts: int, align: int, min_tokens: int):
    """``[(lo, hi), ...]`` token windows for a chunk of ``num_tokens`` rows, or
    ``None`` when micro-batching does not apply (disabled, chunk too small, or
    the windows would not sit on ``align`` -- the TSA all_to_all tiling /
    zigzag block, tp * DSA_TSA_PREFILL_QB)."""
    if parts < 2 or num_tokens < min_tokens:
        return None
    unit = align if align > 0 else 1
    if num_tokens % unit:
        return None
    units = num_tokens // unit
    if units < parts:
        return None
    # equal split in alignment units; a remainder of r units goes to the first r windows
    base, rem = divmod(units, parts)
    widths = [(base + (1 if i < rem else 0)) * unit for i in range(parts)]
    edges = [0]
    for w in widths:
        edges.append(edges[-1] + w)
    return [(edges[i], edges[i + 1]) for i in range(parts)]


def _slice_forward_batch(forward_batch: ForwardBatch, lo: int, hi: int) -> ForwardBatch:
    """The window ``[lo, hi)`` of an EXTEND batch as its own ForwardBatch view:
    token-indexed arrays sliced, attention metadata windowed, everything else
    shared with the original (which is left untouched)."""
    from sgl_jax.srt.layers.attention.dsa_sparse_backend import window_forward_metadata

    part = copy.copy(forward_batch)
    part.input_ids = forward_batch.input_ids[lo:hi]
    part.positions = forward_batch.positions[lo:hi]
    part.out_cache_loc = forward_batch.out_cache_loc[lo:hi]
    backend = copy.copy(forward_batch.attn_backend)
    backend.forward_metadata = window_forward_metadata(
        forward_batch.attn_backend.forward_metadata, lo, hi - lo
    )
    part.attn_backend = backend
    return part


class _MicrobatchPoolView:
    """``token_to_kv_pool`` view for the later micro-batches of one forward: the
    fused KV / indexer-key buffers a previous window already rewrote in this
    forward are served instead of the pool's (the kernels update the paged
    buffers functionally and return them); everything else passes through."""

    def __init__(self, pool):
        self._pool = pool
        self._kv = {}
        self._idx = {}

    def __getattr__(self, name):
        return getattr(self._pool, name)

    def get_fused_kv_buffer(self, layer_id):
        if layer_id in self._kv:
            return self._kv[layer_id]
        return self._pool.get_fused_kv_buffer(layer_id)

    def get_indexer_key_buffer(self, slot_id):
        if slot_id in self._idx:
            return self._idx[slot_id]
        return self._pool.get_indexer_key_buffer(slot_id)

    def update(self, layer_id, slot_id, kv, idx):
        if kv is not None:
            self._kv[layer_id] = kv
        if slot_id is not None and idx is not None:
            self._idx[slot_id] = idx


class Glm5Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.mesh = mesh
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=_EMBED_KERNEL_AXES,
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                Glm5DecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=jnp.float32,
            dtype=dtype,
            scope_name="norm",
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> jax.Array:
        from sgl_jax.srt.layers.attention.dsa_sparse_backend import DSAFusedCache

        bounds = self._prefill_microbatch_bounds(forward_batch)
        if bounds is not None:
            return self._call_microbatched(forward_batch, token_to_kv_pool, bounds)

        hidden_states = self.embed_tokens(forward_batch.input_ids)
        residual = None
        layers_kv_fused = []
        layers_idx_fused = []
        layers_topk_ids = []
        dsa_topk = None
        dsa_topk_pages = None
        for layer in self.layers:
            hidden_states, residual, kv_fused, topk_ids = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
                dispatch_info=forward_batch.expert_location_metadata,
                dsa_topk_in=dsa_topk,
                dsa_topk_pages_in=dsa_topk_pages,
            )
            if isinstance(kv_fused, DSAFusedCache):
                layers_kv_fused.append(kv_fused.kv)
                if kv_fused.idx is not None:
                    layers_idx_fused.append(kv_fused.idx)
                if kv_fused.topk is not None:
                    dsa_topk = kv_fused.topk
                if kv_fused.topk_pages is not None:
                    dsa_topk_pages = kv_fused.topk_pages
            else:
                layers_kv_fused.append(kv_fused)
            layers_topk_ids.append(topk_ids)

        if residual is not None:
            hidden_states += residual

        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused, layers_idx_fused, layers_topk_ids

    def _prefill_microbatch_bounds(self, forward_batch: ForwardBatch):
        """Token windows for this batch, or ``None`` (see ``prefill_microbatch_bounds``)."""
        if _PREFILL_MICROBATCH < 2 or forward_batch.forward_mode != ForwardMode.EXTEND:
            return None
        md = getattr(forward_batch.attn_backend, "forward_metadata", None)
        cu_q_lens = getattr(md, "cu_q_lens", None)
        seq_lens = getattr(md, "seq_lens", None)
        if cu_q_lens is None or seq_lens is None:
            return None
        if cu_q_lens.shape[0] != seq_lens.shape[0] + 1:
            _log_once("prefill micro-batch: off (attention metadata is not dp_size == 1)")
            return None  # window_forward_metadata needs flat [S+1] / [S] metadata
        from sgl_jax.srt.layers.attention import dsa_sparse_backend as dsb

        align = self.mesh.shape["tensor"] * dsb._TSA_PREFILL_QB
        num_tokens = int(forward_batch.input_ids.shape[0])
        parts = _PREFILL_MICROBATCH
        bounds = prefill_microbatch_bounds(num_tokens, parts, align, _PREFILL_MICROBATCH_MIN_TOKENS)
        _log_once(
            f"prefill micro-batch: {bounds if bounds else 'off'} for {num_tokens} tokens "
            f"(parts={parts}, align={align}, min_tokens={_PREFILL_MICROBATCH_MIN_TOKENS})"
        )
        return bounds

    def _call_microbatched(self, forward_batch: ForwardBatch, token_to_kv_pool: KVCache, bounds):
        """``__call__`` with the chunk run as consecutive token windows on the PIN=7 schedule (see
        the DSA_PREFILL_MICROBATCH note above). Same outputs as the single pass: hidden states in
        token order, one rewritten buffer per layer."""
        parts = [_slice_forward_batch(forward_batch, lo, hi) for lo, hi in bounds]
        n = len(parts)
        hidden = [self.embed_tokens(part.input_ids) for part in parts]
        residual = [None] * n
        dsa_topk = [None] * n
        dsa_topk_pages = [None] * n
        pool = _MicrobatchPoolView(token_to_kv_pool)
        full_slot = getattr(forward_batch.attn_backend, "full_slot", None) or {}
        with force_fused_reduce():
            hidden, residual, layers_kv_fused, layers_idx_fused, layers_topk_ids = (
                _microbatch_pinned_layers(
                    self.layers, parts, hidden, residual, dsa_topk, dsa_topk_pages, pool, full_slot
                )
            )
        outs = [h + r if r is not None else h for h, r in zip(hidden, residual)]
        hidden_states = self.norm(jnp.concatenate(outs, axis=0))
        return hidden_states, layers_kv_fused, layers_idx_fused, layers_topk_ids


def _microbatch_pinned_layers(
    layers,
    parts,
    hidden,
    residual,
    dsa_topk,
    dsa_topk_pages,
    pool,
    full_slot,
    pin=None,
):
    """PIN=7 schedule (see the DSA_PREFILL_MICROBATCH note). Per layer:
    attn_0..attn_{n-1} then the MoE / MLP blocks; every all-reduce consumer is pinned behind the
    other window's pre-all-reduce anchor (attention-kernel KV output / MoE partial sum). With two
    windows the MoE layers additionally order the SparseCore stream (two_stage) and tag window B's
    MoE all-reduce + window A's next-layer attend kernel with one XLA scheduling group.
    Returns (hidden, residual, layers_kv_fused, layers_idx_fused, layers_topk_ids)."""
    from sgl_jax.srt.layers.attention.dsa_sparse_backend import DSAFusedCache

    pin = pin or _pin_after
    n = len(parts)
    two_stage = n == 2  # SparseCore-stream ordering pins + scheduling groups
    # window B's MoE all-reduce of layer L and window A's attend kernel of layer L+1 share one
    # _scheduling_group_id, so the scheduler overlaps them without any program-order / data-edge
    # trick (in the HLO the edges only move the done; the start follows the scheduler's own SC-AR
    # latency estimate).
    sched_groups = two_stage
    if sched_groups:
        from jax.experimental import xla_metadata as _xla_metadata

        from sgl_jax.srt.layers.attention import dsa_sparse_backend as _dsa_backend_mod
    _SCHED_GROUP_BASE = 1000
    layers_kv_fused, layers_idx_fused, layers_topk_ids = [], [], []
    prev_mlp_anchor = [None] * n  # previous layer's MoE partial per window (pre all-reduce #2)
    for li, layer in enumerate(layers):
        kv_last = idx_last = None
        attn_anchor = [None] * n  # attention-kernel KV output
        for i, part in enumerate(parts):
            # consumer of window i's previous-layer MoE all-reduce: pin behind window i-1's
            # attention of this layer, or (window 0) behind the last window's MoE partial
            anchor_val = prev_mlp_anchor[n - 1] if i == 0 else attn_anchor[i - 1]
            if anchor_val is not None:
                hidden[i] = pin(hidden[i], anchor_val)
                if residual[i] is not None:
                    residual[i] = pin(residual[i], anchor_val)
            # window A's attend kernel of layer li joins the group of window B's MoE all-reduce
            # of layer li-1 (tagged below with the same id)
            if sched_groups and i == 0 and li >= 1:
                _dsa_backend_mod.PREFILL_SCHED_GROUP = _SCHED_GROUP_BASE + li
            try:
                hidden[i], residual[i], kv_fused = layer.attn_block(
                    part.positions,
                    hidden[i],
                    part,
                    pool,
                    residual[i],
                    dsa_topk_in=dsa_topk[i],
                    dsa_topk_pages_in=dsa_topk_pages[i],
                )
            finally:
                if sched_groups:
                    _dsa_backend_mod.PREFILL_SCHED_GROUP = None
            if isinstance(kv_fused, DSAFusedCache):
                kv_last, idx_last = kv_fused.kv, kv_fused.idx
                if kv_fused.topk is not None:
                    dsa_topk[i] = kv_fused.topk
                if kv_fused.topk_pages is not None:
                    dsa_topk_pages[i] = kv_fused.topk_pages
            else:
                kv_last, idx_last = kv_fused, None
            attn_anchor[i] = kv_last
            pool.update(layer.layer_id, full_slot.get(layer.layer_id), kv_last, idx_last)
        mlp_anchor = [None] * n
        topk_ids_parts = []
        if two_stage and getattr(layer, "is_moe_layer", False) and layer.fused_reduce_supported():
            # (AOT replica): with SparseCore permute kernels in the
            # MoE block, XLA hoists window B's combine call-start ahead of window A's MoE
            # all-reduce; the SparseCore stream completes in start order, so A's all-reduce
            # data phase waits behind B's gmm and its done is exposed (50 ms per 4k window).
            # Two extra zero-valued edges fix the stream order: A's all-reduce operand is pinned
            # behind B's dispatch gather output (A's all-reduce enters the stream after B's
            # dispatch, B's gmm then runs under it) and B's combine input behind A's
            # post-all-reduce value (B's combine cannot be hoisted ahead of A's all-reduce).
            if attn_anchor[1] is not None:
                hidden[0] = pin(hidden[0], attn_anchor[1])
            d_a = layer.mlp_dispatch(hidden[0], parts[0].expert_location_metadata)
            value_a, topk_a, _ = layer.mlp_experts(d_a)
            hidden[1] = pin(hidden[1], value_a)  # B's MLP behind A's partial
            d_b = layer.mlp_dispatch(hidden[1], parts[1].expert_location_metadata)
            pinned_a = pin(value_a, d_b["anchor"])
            hidden[0] = layer.mlp_reduce(pinned_a, True)
            value_b, topk_b, _ = layer.mlp_experts(d_b, combine_pin=hidden[0])
            if sched_groups and li + 1 < len(layers):
                # same group id as window A's attend kernel of the next layer (see the attention phase)
                with _xla_metadata.set_xla_metadata(
                    _scheduling_group_id=_SCHED_GROUP_BASE + li + 1
                ):
                    hidden[1] = layer.mlp_reduce(value_b, True)
            else:
                hidden[1] = layer.mlp_reduce(value_b, True)
            mlp_anchor = [value_a, value_b]
            topk_ids_parts = [topk_a, topk_b]
            prev_mlp_anchor = mlp_anchor
            layers_kv_fused.append(kv_last)
            if idx_last is not None:
                layers_idx_fused.append(idx_last)
            if all(t is None for t in topk_ids_parts):
                layers_topk_ids.append(None)
            else:
                layers_topk_ids.append(jnp.concatenate(topk_ids_parts, axis=0))
            continue
        for i, part in enumerate(parts):
            # consumer of window i's o_proj all-reduce: pin behind window i+1's attention
            # (kernel output), or (last window) behind window 0's MoE partial
            anchor_val = attn_anchor[i + 1] if i + 1 < n else mlp_anchor[0]
            if anchor_val is not None:
                hidden[i] = pin(hidden[i], anchor_val)
            value, topk_ids, deferred = layer.mlp_partial(hidden[i], part.expert_location_metadata)
            mlp_anchor[i] = value if deferred else None
            hidden[i] = layer.mlp_reduce(value, deferred)
            topk_ids_parts.append(topk_ids)
        prev_mlp_anchor = mlp_anchor
        layers_kv_fused.append(kv_last)
        if idx_last is not None:
            layers_idx_fused.append(idx_last)
        if all(t is None for t in topk_ids_parts):
            layers_topk_ids.append(None)
        else:
            layers_topk_ids.append(jnp.concatenate(topk_ids_parts, axis=0))
    return hidden, residual, layers_kv_fused, layers_idx_fused, layers_topk_ids


def _prepare_fused_shared_experts(mappings, blocks):
    """Regroup FusedEPMoEV2 shared-expert weight + block scale into one recipe.

    ``blocks`` lists ``(target_prefix, decoder_layer)`` pairs; the target model
    passes ``model.layers.{i}`` and the NextN draft passes ``mtp_block``.
    """
    mappings = dict(mappings)
    by_target = {
        spec.target_path: source
        for source, spec in mappings.items()
        if isinstance(spec.target_path, str)
    }
    for block_prefix, layer in blocks:
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, FusedEPMoEV2) or not hasattr(mlp, "w1_shared_block_scale"):
            continue
        for name in ("w1_shared", "w3_shared", "w2_shared"):
            path = f"{block_prefix}.mlp.{name}"
            wk, sk = by_target[path], by_target[path + "_block_scale"]
            mappings.pop(wk)
            mappings.pop(sk)
            mappings[wk] = WeightSpec(
                [path, path + "_scale"],
                sources=(wk, sk),
                recipe=partial(
                    _load_shared_expert,
                    mesh=mlp.mesh,
                    quantized_dtype=mlp.quantized_dtype,
                ),
            )
            shape = getattr(mlp, name).value.shape
            getattr(mlp, name + "_scale").value = jax.ShapeDtypeStruct(
                (1, 1, shape[-1]),
                jnp.float32,
                sharding=jax.sharding.NamedSharding(mlp.mesh, P(None, None, None)),
            )
            delattr(mlp, name + "_block_scale")
    return mappings


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
                mesh=mesh,
                enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
            )
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            mesh=self.mesh,
            soft_cap=getattr(config, "final_logit_softcapping", None),
            enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
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

    def get_embed_and_head(self):
        return self.model.embed_tokens.embedding.value, self.lm_head.embedding.value

    def set_embed_and_head(self, embed, head) -> None:
        self.model.embed_tokens.embedding.value = embed
        self.lm_head.embedding.value = head

    def set_embed(self, embed) -> None:
        self.model.embed_tokens.embedding.value = embed

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_glm5_weight_mappings(model_config)
        loader.load(weight_mappings)
        for layer in self.model.layers:
            layer.self_attn.fuse_in_proj()
        prepad_replicated_quantized_linears(self)

        logger.info("Absorbed MLA weights and Fused MLP weights processed successfully!")

        # Skipping scale inversion for BF16
        logger.info("Skipping scale inversion for BF16 model.")

    def prepare_weight_loading(self, loader, mappings):
        return _prepare_fused_shared_experts(
            mappings, [(f"model.layers.{i}", layer) for i, layer in enumerate(self.model.layers)]
        )

    def _create_glm5_weight_mappings(self, model_config: ModelConfig) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightSpec(
                target_path="model.embed_tokens.embedding",
                sharding=_EMBED_KERNEL_AXES,  # follows SGLANG_JAX_EMBED_REPLICATED (a hardcoded shard left the table split while the module declared it replicated)
                transpose=False,
            ),
            "model.norm.weight": WeightSpec(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }

        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = self.lm_head.weight_mapping("lm_head.embedding")

        num_layers = self.config.num_hidden_layers
        first_k_dense_replace = getattr(self.config, "first_k_dense_replace", 0)
        indexer_types = getattr(self.config, "indexer_types", None)

        quant_config = getattr(model_config, "quantization_config", None)
        is_static_quant = quant_config is not None and quant_config.is_static_checkpoint
        moe_scale_suffix = routed_expert_scale_suffix(quant_config)

        hf_layer_indices = list(range(num_layers))
        for layer_idx in range(num_layers):
            target_idx = hf_layer_indices[layer_idx]
            layer_mappings = self._create_moe_layer_mappings(
                layer_idx,
                target_idx,
                target_idx < first_k_dense_replace,
                is_static_quant=is_static_quant,
                has_indexer=indexer_types is None or indexer_types[target_idx] == "full",
                moe_scale_suffix=moe_scale_suffix,
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
        moe_scale_suffix: str = ".weight_scale_inv",
    ) -> dict:
        prefix = f"model.layers.{target_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        mappings = {
            f"{prefix}.input_layernorm.weight": WeightSpec(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightSpec(
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
                mappings[f"{hf}.weight"] = WeightSpec(
                    target_path=f"{tgt}.weight", sharding=sharding_std, transpose=True
                )
                return
            sharding_q = (sharding_std[1], sharding_std[0])
            mappings[f"{hf}.weight"] = WeightSpec(
                target_path=f"{tgt}.weight_q", sharding=sharding_q, transpose=False
            )
            # Load 2D block scale [out_blocks, in_blocks] replicated: GLM-5.1 head_dim
            # 448 → out_blocks not always tp-divisible (kv_b_proj: 224 % 64 ≠ 0).
            # _maybe_expand_linear_block_scale runs after _shard_weight and expands to
            # [in_blocks, 1, n_out]; assignment into model_param then reshards to the
            # QuantizedLinear placeholder's 3D sharding.
            mappings[f"{hf}.weight_scale_inv"] = WeightSpec(
                target_path=f"{tgt}.weight_scale",
                sharding=(None, None),
                transpose=False,
            )

        ap = f"{prefix}.self_attn"
        tp = f"{target_prefix}.self_attn"
        add_linear(f"{ap}.q_a_proj", f"{tp}.q_a_proj", (None, None))
        mappings[f"{ap}.q_a_layernorm.weight"] = WeightSpec(
            target_path=f"{tp}.q_a_layernorm.scale", sharding=(None,)
        )
        add_linear(f"{ap}.q_b_proj", f"{tp}.q_b_proj", (None, "tensor"))
        add_linear(f"{ap}.kv_a_proj_with_mqa", f"{tp}.kv_a_proj_with_mqa", (None, None))
        mappings[f"{ap}.kv_a_layernorm.weight"] = WeightSpec(
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
            mappings[f"{ap}.indexer.k_norm.weight"] = WeightSpec(
                target_path=f"{tp}.indexer.k_norm.weight", sharding=(None,)
            )
            mappings[f"{ap}.indexer.k_norm.bias"] = WeightSpec(
                target_path=f"{tp}.indexer.k_norm.bias", sharding=(None,)
            )

        if is_mlp_layer:
            add_linear(
                f"{prefix}.mlp.gate_proj",
                f"{target_prefix}.mlp.gate_proj",
                (None, "tensor"),
            )
            add_linear(
                f"{prefix}.mlp.up_proj",
                f"{target_prefix}.mlp.up_proj",
                (None, "tensor"),
            )
            add_linear(
                f"{prefix}.mlp.down_proj",
                f"{target_prefix}.mlp.down_proj",
                ("tensor", None),
            )
        else:
            mappings[f"{prefix}.mlp.gate.weight"] = WeightSpec(
                target_path=f"{target_prefix}.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )
            # GLM-4 uses e_score_correction_bias
            mappings[f"{prefix}.mlp.gate.e_score_correction_bias"] = WeightSpec(
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
                    target_param = mapping.target_path
                    src_paths = mapping.sources

                    new_moe_mappings[key] = WeightSpec(
                        target_path=target_param,
                        sources=tuple(src_paths),
                        sharding=mapping.sharding,
                        transpose=True,
                        concat_axis=mapping.concat_axis,
                        physical_to_logical_map=mapping.physical_to_logical_map,
                    )

                    scale_key = key + "_scale"
                    target_scale_param = target_param + "_scale"
                    # FP8 block checkpoints: ``weight_scale_inv`` [out_blocks, in_blocks];
                    # int4 per-channel expert checkpoints: ``weight_scale`` [out, 1].
                    scale_src_paths = [p.replace(".weight", moe_scale_suffix) for p in src_paths]

                    # Stacked HF scale is [E, out_blocks, in_blocks]. Load EP-sharded
                    # and replicated on the block dims (matches deepseek_v3); the
                    # loader's _maybe_convert_epmoe_scale_for_kernel handles the
                    # [E, out_blocks, k_blocks] → [E, k_blocks, 1, n_out] expand.
                    # The expert-dim spec must follow the weight mapping's:
                    # epmoe loads on the ("expert","tensor") moe_mesh, but
                    # fused/fused_v2 run shard_map on the main mesh with
                    # P(("data","tensor")) — a hardcoded "expert" here makes
                    # shard_map reject the scale under fused_v2 static FP8.
                    new_moe_mappings[scale_key] = WeightSpec(
                        target_path=target_scale_param,
                        sources=tuple(scale_src_paths),
                        sharding=(mapping.sharding[0], None, None),
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
                        mappings[f"{sp}.{hf_name}.weight"] = WeightSpec(
                            target_path=target_path,
                            sharding=(None, None),
                            transpose=True,
                        )
                        if is_static_quant:
                            mappings[f"{sp}.{hf_name}.weight_scale_inv"] = WeightSpec(
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
            # narrow-N guard would reject it but the indexer output is currently
            # discarded so accuracy is unaffected. Match deepseek_v3 config.
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
        # The fused MLP is a TPU Pallas kernel; CPU (interpret/smoke) runs keep the plain path.
        mc.hf_config._sgl_use_fused_mlp = (
            mc.quantization_config is None and jax.default_backend() == "tpu"
        )


_NEXTN_TOP_LEVEL_MODULES = ("eh_proj", "enorm", "hnorm", "shared_head", "embed_tokens", "lm_head")


def _checkpoint_modules_to_not_convert(mc) -> list[str] | None:
    """The checkpoint's raw ``modules_to_not_convert`` / ``ignored_layers`` list.

    ``ModelConfig`` overwrites ``hf_config.quantization_config`` with the unified
    ``QuantizationConfig`` object, so prefer the raw dict it keeps on
    ``hf_quantization_config_raw``; fall back to the hf_config dict if still raw.
    """
    raw = getattr(mc, "hf_quantization_config_raw", None)
    if not isinstance(raw, dict):
        raw = getattr(getattr(mc, "hf_config", None), "quantization_config", None)
    if not isinstance(raw, dict):
        # hf_config is lru-cached across the target and draft ModelConfigs, so
        # the draft can only see the overwritten object: read config.json.
        model_path = getattr(mc, "model_path", None)
        if not model_path:
            return None
        try:
            with open(os.path.join(model_path, "config.json")) as f:
                raw = json.load(f).get("quantization_config")
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
    return raw.get("ignored_layers") or raw.get("modules_to_not_convert")


def nextn_ignored_layers(ignored_layers, nextn_layer_idx: int, hf_ignored=None) -> list[str]:
    """Translate the checkpoint's ``modules_to_not_convert`` for the MTP layer.

    ``hf_ignored`` is the checkpoint's raw list: a user yaml
    (``--quantization-config-path``) builds ``ignored_layers`` without it, so
    the MTP layer's bf16 modules (``eh_proj`` ...) are merged from the checkpoint
    here; entries for other layers are left to the main model's own config.

    The FP8 checkpoint lists the unquantized MTP modules under
    ``model.layers.{num_hidden_layers}.*`` (e.g. ``eh_proj``, ``mlp.gate``), but
    :class:`GlmMoeDsaForCausalLMNextN` hosts them at ``eh_proj`` / ``mtp_block.*``.
    ``quantize_model`` matches module paths against these entries, so without the
    translation ``eh_proj`` is converted to a ``QuantizedLinear`` and the bf16
    checkpoint weight has no ``.weight`` to load into.
    """
    ignored = list(ignored_layers or [])
    prefix = f"model.layers.{nextn_layer_idx}."
    for ig in hf_ignored or []:
        if ig.startswith(prefix) and ig not in ignored:
            ignored.append(ig)
    extra = []
    for ig in ignored:
        if not ig.startswith(prefix):
            continue
        rest = ig[len(prefix) :]
        mapped = rest if rest.split(".")[0] in _NEXTN_TOP_LEVEL_MODULES else f"mtp_block.{rest}"
        if mapped not in ignored and mapped not in extra:
            extra.append(mapped)
    return ignored + extra


def nextn_pool_updates(kv_fused, is_dsa: bool):
    """Pool write-back payload of one NextN forward, in the target model's format.

    On the DSA path the NextN block updates two page buffers: the latent KV and the
    indexer keys. ``MLATokenToKVPool.replace_buffer`` only touches ``indexer_key_buffer``
    when it is handed the ``(kv_list, idx_list)`` tuple, so returning the bare KV list
    silently drops every indexer-key write of the draft model: each later step-0 call
    then scores pages against a buffer whose historical positions were never written
    (#1639 review; accept length recovers once the keys survive across calls).
    """
    if not is_dsa:
        return [kv_fused]
    kv = [kv_fused.kv]
    if kv_fused.idx is None:
        return {"token_to_kv_pool": kv}
    return {"token_to_kv_pool": (kv, [kv_fused.idx])}


# SGLANG_JAX_EMBED_REPLICATED (default on) keeps the token-embedding table replicated on every
# device instead of vocab-sharded on "tensor". The vocab-sharded lookup is an
# XLA gather + one bf16[tokens, hidden] all-reduce over all devices per lookup (embeddings.py:112;
# 3 per fused draft-extend module + 1 per verify step at cc64, 63-88 us each; one 96 MiB AR per 8k
# prefill window). The replicated lookup is a local gather: same values (x + 0 == x exactly in bf16),
# no all-reduce; cost = +vocab x hidden x 2 B HBM per device (GLM-5.2: 154880 x 6144 = 1.9 GB).
_EMBED_REPLICATED = envs.SGLANG_JAX_EMBED_REPLICATED.get()
_EMBED_KERNEL_AXES = (None, None) if _EMBED_REPLICATED else ("tensor", None)


class GlmMoeDsaForCausalLMNextN(nnx.Module):
    load_lm_head_from_target = True

    @classmethod
    def patch_model_config(cls, mc: ModelConfig) -> None:
        GlmMoeDsaForCausalLM.patch_model_config(mc)
        qc = mc.quantization_config
        if qc is not None and qc.is_static_checkpoint:
            idx = getattr(mc.hf_config, "num_hidden_layers", 78)
            qc.ignored_layers = nextn_ignored_layers(
                qc.ignored_layers, idx, hf_ignored=_checkpoint_modules_to_not_convert(mc)
            )

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        # The Draft Worker only passes its own isolated memory pool (which inherits model_config.num_hidden_layers length)
        # We must index layer_id=0 so it does not exceed the KV buffer array bounds!
        self.mtp_layer_idx = 0

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=_EMBED_KERNEL_AXES,
            param_dtype=dtype,
            mesh=mesh,
        )
        self.enorm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype, scope_name="enorm"
        )
        self.hnorm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype, scope_name="hnorm"
        )
        self.eh_proj = LinearBase(
            input_size=2 * config.hidden_size,
            output_size=config.hidden_size,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.mtp_block = Glm5DecoderLayer(
            config, layer_id=self.mtp_layer_idx, dtype=dtype, mesh=mesh, force_moe=True
        )
        self.shared_head = nnx.Module()
        self.shared_head.norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype, scope_name="norm"
        )
        # Same call face as Glm5ForCausalLM: load_lm_head_from_target copies the
        # target's lm_head array, so the DP lm-head layout must match on both sides.
        enable_dp_lm_head = getattr(config, "enable_dp_lm_head", False)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            mesh=mesh,
            enable_dp_lm_head=enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(
            config.vocab_size, mesh=self.mesh, enable_dp_lm_head=enable_dp_lm_head
        )
        self.hot_token_ids = None

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools,
        logits_metadata: LogitsMetadata,
        dsa_topk_pages_in: jax.Array | None = None,
        dsa_topk_reuse: bool = False,
        return_dsa_topk_pages: bool = False,
    ):
        """One MTP step.

        GLM-5.2 IndexShare across MTP iterations (``index_share_for_mtp_iteration``):
        the fused draft loop passes ``return_dsa_topk_pages=True`` to receive the
        page-topk this step's indexer selected, and on later steps hands it back via
        ``dsa_topk_pages_in`` with ``dsa_topk_reuse=True`` so the block attends over
        the reused selection instead of re-running the indexer.
        """
        from sgl_jax.srt.layers.attention.dsa_sparse_backend import DSAFusedCache

        embed = self.embed_tokens(forward_batch.input_ids)
        hidden_in = forward_batch.spec_info.hidden_states
        emb_sh = jax.typeof(embed).sharding
        if isinstance(emb_sh, jax.sharding.NamedSharding):
            hidden_in = jax.sharding.reshard(hidden_in, emb_sh)

        concat_in = jnp.concatenate((self.enorm(embed), self.hnorm(hidden_in)), axis=-1)
        hidden_states, _ = self.eh_proj(concat_in)

        token_to_kv_pool = (
            memory_pools.token_to_kv_pool
            if hasattr(memory_pools, "token_to_kv_pool")
            else memory_pools
        )
        hidden_states, residual, kv_fused, _ = self.mtp_block(
            forward_batch.positions,
            hidden_states,
            forward_batch,
            token_to_kv_pool,
            None,
            dispatch_info=forward_batch.expert_location_metadata,
            dsa_topk_in=None,
            dsa_topk_pages_in=dsa_topk_pages_in,
            dsa_topk_reuse=dsa_topk_reuse,
        )

        if residual is not None:
            hidden_states = hidden_states + residual

        hidden_states = self.shared_head.norm(hidden_states)
        output = self.logits_processor(
            hidden_states, self.lm_head, logits_metadata, aux_hidden_states=None
        )

        is_dsa = isinstance(kv_fused, DSAFusedCache)
        kv_cache_list = nextn_pool_updates(kv_fused, is_dsa)
        if return_dsa_topk_pages:
            return output, kv_cache_list, True, None, (kv_fused.topk_pages if is_dsa else None)
        return output, kv_cache_list, True, None

    def load_weights(self, model_config: ModelConfig):
        self.loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        mappings = self._create_weight_mappings(model_config)
        # MLA absorption and fused-MLP packing run inside loader.load via the
        # modules' prepare_weight_loading hooks (#1716); no post-load pass.
        self.loader.load(mappings)
        prepad_replicated_quantized_linears(self)

    def get_shared_weight_paths(self):
        # Filled from the target model by EagleDraftWorker._share_embed_head.
        return ("embed_tokens.embedding", "lm_head.embedding")

    def prepare_weight_loading(self, loader, mappings):
        return _prepare_fused_shared_experts(mappings, [("mtp_block", self.mtp_block)])

    @classmethod
    def _create_weight_mappings(cls, model_config: ModelConfig) -> dict[str, WeightSpec]:
        mappings = {}
        idx = getattr(model_config.hf_config, "num_hidden_layers", 78)
        prefix = f"model.layers.{idx}"

        mappings[f"{prefix}.enorm.weight"] = WeightSpec(
            target_path="enorm.scale", sharding=(None,), transpose=False
        )
        mappings[f"{prefix}.hnorm.weight"] = WeightSpec(
            target_path="hnorm.scale", sharding=(None,), transpose=False
        )
        mappings[f"{prefix}.eh_proj.weight"] = WeightSpec(
            target_path="eh_proj.weight", sharding=(None, None), transpose=True
        )
        mappings[f"{prefix}.shared_head.norm.weight"] = WeightSpec(
            target_path="shared_head.norm.scale", sharding=(None,), transpose=False
        )

        class MockConfig:
            pass

        mock_self = MockConfig()
        mock_self.config = model_config.hf_config
        quant_config = getattr(model_config, "quantization_config", None)
        is_static_quant = quant_config is not None and quant_config.is_static_checkpoint

        decoder_mappings = Glm5ForCausalLM._create_moe_layer_mappings(
            mock_self,
            layer_idx=idx,
            target_idx=idx,
            is_mlp_layer=False,
            is_static_quant=is_static_quant,
            has_indexer=True,
            moe_scale_suffix=routed_expert_scale_suffix(quant_config),
        )
        for k, v in decoder_mappings.items():
            if isinstance(v.target_path, str):
                v.target_path = v.target_path.replace(f"model.layers.{idx}", "mtp_block")
            elif isinstance(v.target_path, list):
                new_path = [v.target_path[0].replace(f"model.layers.{idx}", "mtp_block")]
                new_path.extend(v.target_path[1:])
                v.target_path = new_path
            mappings[k] = v

        return mappings

    def get_embed_and_head(self):
        return self.embed_tokens.embedding.value, self.lm_head.embedding.value

    def set_embed_and_head(self, embed: jax.Array, head: jax.Array) -> None:
        self.embed_tokens.embedding.value = embed
        self.lm_head.embedding.value = head

    def set_embed(self, embed: jax.Array) -> None:
        self.embed_tokens.embedding.value = embed


EntryClass = [Glm5ForCausalLM, GlmMoeDsaForCausalLM, GlmMoeDsaForCausalLMNextN]
