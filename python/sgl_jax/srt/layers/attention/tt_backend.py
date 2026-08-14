"""TT attention backend for paged Qwen-style decoder serving."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MHATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils import cdiv
from sgl_jax.srt.utils.jax_utils import device_array
from sgl_jax.srt.utils.profiling_utils import named_scope

logger = logging.getLogger(__name__)


def _tt_call(target, result, *operands, config=""):
    return jax.ffi.ffi_call(
        target,
        jax.ShapeDtypeStruct(result.shape, result.dtype),
        custom_call_api_version=2,
        legacy_backend_config=config,
    )(*operands)


def _prefill_attention(query, key, value):
    return _tt_call("tt.scaled_dot_product_attention", query, query, key, value)


def _decode_attention(query, key, value, page_table, positions):
    return _tt_call(
        "tt.paged_scaled_dot_product_attention_decode",
        query,
        query,
        key,
        value,
        page_table,
        positions,
    )


def _update_cache(cache, value, positions, page_table):
    return _tt_call(
        "tt.paged_update_cache", cache, cache, value, positions, page_table
    )


def _fill_cache(cache, value, page_table, batch_indices):
    return _tt_call(
        "tt.paged_fill_cache", cache, cache, value, page_table, batch_indices
    )


def prepare_weight(tensor):
    """Mark a JIT argument as a TT parameter and select its storage dtype."""
    original_shape = tensor.shape
    if tensor.ndim < 3:
        tensor = jnp.reshape(tensor, (1,) * (3 - tensor.ndim) + original_shape)
    dtype = "bf16" if len(original_shape) == 1 else "bfp_bf8"
    tensor = _tt_call("tt.weight_dtype_override", tensor, tensor, config=dtype)
    return jnp.reshape(tensor, original_shape)


@register_pytree_node_class
class TTTokenToKVPool(MHATokenToKVPool):
    """Separate K/V pages in the layout consumed by TTNN attention."""

    def _create_buffers(self):
        self.kv_sharding = NamedSharding(self.mesh, P("data", "tensor", None, None))
        shape = (
            (self.size + self.page_size * self.dp_size) // self.page_size,
            self.head_num,
            self.page_size,
            self.head_dim,
        )
        zeros = np.zeros(shape, dtype=np.dtype(self.dtype))
        start = time.time()
        with self.mesh:
            self.kv_buffer = [
                (
                    jax.device_put(zeros.copy(), self.kv_sharding),
                    jax.device_put(zeros.copy(), self.kv_sharding),
                )
                for _ in range(self.layer_num)
            ]
        logger.info("Created TT KV buffers in %.2f seconds", time.time() - start)

    def get_fused_kv_buffer(self, layer_id):
        return self.get_kv_buffer(layer_id)

    def get_kv_buffer(self, layer_id):
        return self.kv_buffer[layer_id - self.start_layer]

    def replace_buffer(self, new_buffer):
        if new_buffer is None or all(value is None for value in new_buffer):
            return
        super().replace_buffer(new_buffer)


@register_pytree_node_class
@dataclass
class TTAttentionMetadata:
    page_table: jax.Array | None = None
    positions: jax.Array | None = None
    fill_page_table: jax.Array | None = None
    fill_batch_indices: jax.Array | None = None
    tokens_per_sequence: int | None = None

    def tree_flatten(self):
        children = (
            self.page_table,
            self.positions,
            self.fill_page_table,
            self.fill_batch_indices,
        )
        return children, self.tokens_per_sequence

    @classmethod
    def tree_unflatten(cls, tokens_per_sequence, children):
        return cls(*children, tokens_per_sequence=tokens_per_sequence)


def _pad_page_table(table: np.ndarray, users: int) -> np.ndarray:
    return np.pad(
        table,
        ((0, max(0, users - table.shape[0])), (0, max(0, 16 - table.shape[1]))),
    )


def _decode_page_table(
    cache_locations: np.ndarray,
    sequence_lengths: np.ndarray,
    block_size: int,
    dp_size: int,
    batch_size_per_dp: int,
) -> np.ndarray:
    aligned_lengths = cdiv(sequence_lengths, block_size) * block_size
    page_counts = cdiv(sequence_lengths, block_size)
    table = np.zeros(
        (len(sequence_lengths), max(int(page_counts.max(initial=0)), 1)),
        dtype=np.int32,
    )
    locations_per_dp = len(cache_locations) // dp_size

    for dp_rank in range(dp_size):
        row_base = dp_rank * batch_size_per_dp
        token_base = dp_rank * locations_per_dp
        rank_lengths = aligned_lengths[row_base : row_base + batch_size_per_dp]
        offsets = np.zeros(batch_size_per_dp, dtype=np.int64)
        if batch_size_per_dp > 1:
            offsets[1:] = np.cumsum(rank_lengths[:-1], dtype=np.int64)

        for local_row in range(batch_size_per_dp):
            row = row_base + local_row
            count = int(page_counts[row])
            start = token_base + int(offsets[local_row])
            locations = cache_locations[
                start : start + int(rank_lengths[local_row]) : block_size
            ]
            table[row, :count] = locations[:count] // block_size

    return table


class TTAttention(AttentionBackend):
    """TTNN prefill and paged-decode attention with an in-place KV cache."""

    token_to_kv_pool_class = TTTokenToKVPool
    updates_cache_in_place = True
    use_fast_greedy_sampler = True
    compiler_options = {
        "math_fidelity": "hifi4",
        "fp32_dest_acc_en": "true",
        "experimental_enable_permute_matmul_fusion": "true",
        "optimization_level": "1",
        "experimental_weight_dtype": "bfp_bf8",
        "enable_trace": "true",
    }

    def __init__(
        self,
        num_attn_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        mesh: jax.sharding.Mesh,
    ):
        if page_size < 32 or page_size % 32:
            raise ValueError("TT attention requires a page size divisible by 32")
        self.num_heads = num_attn_heads
        self.num_kv_heads = num_kv_heads or num_attn_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.mesh = mesh
        self.forward_metadata = nnx.data(TTAttentionMetadata())

    def tree_flatten(self):
        return (self.forward_metadata,), {
            "num_attn_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "page_size": self.page_size,
            "mesh": self.mesh,
        }

    @classmethod
    def tree_unflatten(cls, attributes, children):
        backend = cls(**attributes)
        backend.forward_metadata = children[0]
        return backend

    def prepare_model_state(self, leaves):
        return tuple(
            prepare_weight(leaf)
            if getattr(leaf, "ndim", 0) > 0
            and getattr(leaf, "dtype", None) in (jnp.bfloat16, jnp.float32)
            else leaf
            for leaf in leaves
        )

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        if batch.forward_mode == ForwardMode.EXTEND:
            return self._prefill_metadata(batch)
        if batch.forward_mode == ForwardMode.DECODE:
            return self._decode_metadata(batch)
        raise ValueError(f"TT attention does not support {batch.forward_mode}")

    def _decode_metadata(self, batch: ModelWorkerBatch) -> TTAttentionMetadata:
        if batch.dp_size <= 0 or batch.per_dp_bs_size <= 0:
            raise ValueError("TT attention received invalid DP batch metadata")
        if len(batch.cache_loc) % batch.dp_size:
            raise ValueError("TT cache locations must be evenly partitioned across DP ranks")

        batch_size_per_dp = max(max(batch.real_bs_per_dp), 1)
        sequence_lengths = np.zeros(
            batch.dp_size * batch_size_per_dp, dtype=np.int32
        )
        source_lengths = np.asarray(batch.seq_lens, dtype=np.int32)
        for dp_rank, real_batch_size in enumerate(batch.real_bs_per_dp):
            source = dp_rank * batch.per_dp_bs_size
            target = dp_rank * batch_size_per_dp
            sequence_lengths[target : target + real_batch_size] = source_lengths[
                source : source + real_batch_size
            ]

        users = max(len(batch.input_ids), 1)
        page_table = _pad_page_table(
            _decode_page_table(
                np.asarray(batch.cache_loc, dtype=np.int32),
                sequence_lengths,
                self.page_size,
                batch.dp_size,
                batch_size_per_dp,
            ),
            users,
        )
        positions = np.where(sequence_lengths > 0, sequence_lengths - 1, -1)
        positions = np.pad(
            positions.astype(np.int32),
            (0, max(0, users - len(positions))),
            constant_values=-1,
        )
        return TTAttentionMetadata(
            page_table=device_array(
                page_table, sharding=NamedSharding(self.mesh, P("data", None))
            ),
            positions=device_array(
                positions, sharding=NamedSharding(self.mesh, P("data"))
            ),
        )

    def _prefill_metadata(self, batch: ModelWorkerBatch) -> TTAttentionMetadata:
        real_batch_size = int(batch.real_bs)
        sequence_lengths = np.asarray(
            batch.extend_seq_lens[:real_batch_size], dtype=np.int32
        )
        prefix_lengths = np.asarray(
            batch.extend_prefix_lens[:real_batch_size], dtype=np.int32
        )
        if (
            real_batch_size == 0
            or batch.out_cache_loc is None
            or np.any(prefix_lengths)
            or np.any(sequence_lengths != sequence_lengths[0])
        ):
            raise ValueError(
                "TT prefill requires non-empty, equal-length requests without cached prefixes"
            )

        tokens_per_sequence = int(sequence_lengths[0])
        locations = np.asarray(batch.out_cache_loc, dtype=np.int32)[
            : real_batch_size * tokens_per_sequence
        ].reshape(real_batch_size, tokens_per_sequence)
        expected = locations[:, :1] + np.arange(tokens_per_sequence, dtype=np.int32)
        page_starts = locations[:, :: self.page_size]
        if np.any(locations < 0) or np.any(locations != expected) or np.any(
            page_starts % self.page_size
        ):
            raise ValueError("TT prefill requires contiguous, page-aligned cache locations")

        page_table = _pad_page_table(page_starts // self.page_size, users=8)
        return TTAttentionMetadata(
            fill_page_table=device_array(
                page_table, sharding=NamedSharding(self.mesh, P("data", None))
            ),
            fill_batch_indices=device_array(
                np.arange(real_batch_size, dtype=np.int32),
                sharding=NamedSharding(self.mesh, P("data")),
            ),
            tokens_per_sequence=tokens_per_sequence,
        )

    @named_scope
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        **kwargs,
    ):
        if kwargs.get("attention_sink") is not None:
            raise ValueError("TT attention does not support attention sinks")
        if layer.sliding_window_size:
            raise ValueError("TT attention does not support sliding-window layers")
        if getattr(layer, "xai_temperature_len", -1) > 0:
            raise ValueError("TT attention does not support xAI temperature scaling")

        if forward_batch.forward_mode == ForwardMode.EXTEND:
            return self._prefill(q, k, v, layer, token_to_kv_pool)
        if forward_batch.forward_mode == ForwardMode.DECODE:
            return self._decode(q, k, v, layer, token_to_kv_pool)
        raise ValueError(f"TT attention does not support {forward_batch.forward_mode}")

    def _prefill(self, q, k, v, layer, token_to_kv_pool):
        metadata = self.forward_metadata
        if metadata.tokens_per_sequence is None:
            raise ValueError("TT prefill metadata is missing")

        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k_cache = _fill_cache(
            k_cache,
            self._prefill_cache_value(k, k_cache.shape[-1]),
            metadata.fill_page_table,
            metadata.fill_batch_indices,
        )
        v_cache = _fill_cache(
            v_cache,
            self._prefill_cache_value(v, v_cache.shape[-1]),
            metadata.fill_page_table,
            metadata.fill_batch_indices,
        )
        k_cache, v_cache = self._cache_barrier((k_cache, v_cache))

        tokens = metadata.tokens_per_sequence
        batch_size = metadata.fill_batch_indices.shape[0]
        active_tokens = batch_size * tokens
        total_tokens = q.shape[0]
        q = q[:active_tokens].reshape(batch_size, tokens, q.shape[1], q.shape[2])
        k = k[:active_tokens].reshape(batch_size, tokens, k.shape[1], k.shape[2])
        v = v[:active_tokens].reshape(batch_size, tokens, v.shape[1], v.shape[2])

        padded_tokens = cdiv(tokens, 32) * 32
        if padded_tokens != tokens:
            padding = ((0, 0), (0, padded_tokens - tokens), (0, 0), (0, 0))
            q, k, v = (jnp.pad(value, padding) for value in (q, k, v))

        output = _prefill_attention(
            jnp.transpose(q, (0, 2, 1, 3)),
            jnp.transpose(k, (0, 2, 1, 3)),
            jnp.transpose(v, (0, 2, 1, 3)),
        )
        output = jnp.transpose(output, (0, 2, 1, 3))[:, :tokens]
        output = output.reshape(active_tokens, -1)
        if active_tokens < total_tokens:
            output = jnp.pad(output, ((0, total_tokens - active_tokens), (0, 0)))
        return output, (k_cache, v_cache)

    def _decode(self, q, k, v, layer, token_to_kv_pool):
        metadata = self.forward_metadata
        if metadata.page_table is None or metadata.positions is None:
            raise ValueError("TT decode metadata is missing")

        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
        num_tokens = q.shape[0]
        q = q.reshape(1, num_tokens, q.shape[1], q.shape[2])
        users = q.shape[1]
        page_table = metadata.page_table[:users]
        positions = metadata.positions[:users]

        k_update = self._decode_cache_value(k, k_cache.shape[-1])
        v_update = self._decode_cache_value(v, v_cache.shape[-1])
        k_cache = _update_cache(k_cache, k_update, positions, page_table)
        v_cache = _update_cache(v_cache, v_update, positions, page_table)
        k_cache, v_cache = self._cache_barrier((k_cache, v_cache))

        output = _decode_attention(
            q,
            k_cache,
            v_cache,
            page_table,
            positions,
        )
        output_sharding = NamedSharding(self.mesh, P("data", "tensor"))
        return output[:, :num_tokens].reshape(
            num_tokens, -1, out_sharding=output_sharding
        ), None

    def _prefill_cache_value(self, value, head_dim):
        tokens = self.forward_metadata.tokens_per_sequence
        batch_size = self.forward_metadata.fill_batch_indices.shape[0]
        value = self._pad_head_dim(value[: batch_size * tokens], head_dim)
        value = value.reshape(batch_size, tokens, value.shape[1], value.shape[2])
        return jnp.transpose(value, (0, 2, 1, 3))

    @staticmethod
    def _decode_cache_value(value, head_dim):
        return TTAttention._pad_head_dim(value, head_dim).reshape(
            1, value.shape[0], value.shape[1], head_dim
        )

    @staticmethod
    def _pad_head_dim(value, head_dim):
        if value.shape[-1] < head_dim:
            return jnp.pad(
                value, ((0, 0), (0, 0), (0, head_dim - value.shape[-1]))
            )
        return value[..., :head_dim]

    @staticmethod
    def _cache_barrier(cache):
        return tuple(jax.lax.optimization_barrier(value) for value in cache)

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        pages_per_request = cdiv(max_context_len, page_size)
        return max(1, 1024 * 1024 // 2 // pages_per_request // 4)
