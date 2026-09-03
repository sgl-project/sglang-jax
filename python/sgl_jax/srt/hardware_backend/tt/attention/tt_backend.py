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

from sgl_jax.srt.hardware_backend.tt.attention import ops as tt_ops
from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MHATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils import cdiv
from sgl_jax.srt.utils.jax_utils import device_array
from sgl_jax.srt.utils.profiling_utils import named_scope

logger = logging.getLogger(__name__)


def prepare_weight(tensor):
    dtype = "bf16" if tensor.ndim == 1 else "bfp_bf8"
    return tt_ops.annotate_weight_dtype(tensor, dtype)


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


@jax.tree_util.register_dataclass
@dataclass
class TTAttentionMetadata:
    page_table: jax.Array | None = None
    positions: jax.Array | None = None
    fill_page_table: jax.Array | None = None
    prefill_chunk_start: jax.Array | None = None
    prefill_input_indices: jax.Array | None = None
    prefill_output_indices: jax.Array | None = None


def _pad_page_table(table: np.ndarray, users: int) -> np.ndarray:
    return np.pad(
        table,
        ((0, max(0, users - table.shape[0])), (0, max(0, 16 - table.shape[1]))),
    )


def _page_table(
    cache_locations: np.ndarray,
    sequence_lengths: np.ndarray,
    block_size: int,
) -> np.ndarray:
    if len(cache_locations) % block_size:
        raise ValueError("TT cache-location capacity must be page-aligned")

    aligned_lengths = cdiv(sequence_lengths, block_size) * block_size
    page_counts = cdiv(sequence_lengths, block_size)
    page_capacity = max(len(cache_locations) // block_size, 1)
    if np.any(page_counts > page_capacity):
        raise ValueError("TT sequence exceeds the cache-location page capacity")

    # cache_locations is scheduler-bucketed, so its capacity is stable while
    # sequences grow. Use that capacity for the page table instead of its live
    # width to keep the JAX input shape fixed.
    table = np.zeros(
        (len(sequence_lengths), page_capacity),
        dtype=np.int32,
    )

    offsets = np.zeros(len(sequence_lengths), dtype=np.int64)
    if len(sequence_lengths) > 1:
        offsets[1:] = np.cumsum(aligned_lengths[:-1], dtype=np.int64)
    for row, start in enumerate(offsets):
        length = aligned_lengths[row]
        count = page_counts[row]
        locations = cache_locations[start : start + length : block_size]
        table[row, :count] = locations[:count] // block_size

    return table


class TTAttention(AttentionBackend):
    """TTNN prefill and paged-decode attention with an in-place KV cache."""

    token_to_kv_pool_class = TTTokenToKVPool
    compiler_options = {
        "experimental_enable_permute_matmul_fusion": "true",
        "optimization_level": "1",
        "experimental_weight_dtype": "bfp_bf8",
        "enable_trace": "true",
    }
    sampler_compiler_options = {"enable_trace": "true"}

    def __init__(self, page_size: int, mesh: jax.sharding.Mesh):
        if page_size < 32 or page_size % 32:
            raise ValueError("TT attention requires a page size divisible by 32")
        if mesh.shape["data"] != 1:
            raise NotImplementedError("TT attention currently supports dp_size=1 only")
        self.page_size = page_size
        self.mesh = mesh
        self.forward_metadata = nnx.data(TTAttentionMetadata())

    def tree_flatten(self):
        return (self.forward_metadata,), {
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
            (
                prepare_weight(leaf)
                if getattr(leaf, "ndim", 0) > 0
                and getattr(leaf, "dtype", None) in (jnp.bfloat16, jnp.float32)
                else leaf
            )
            for leaf in leaves
        )

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        if batch.forward_mode == ForwardMode.EXTEND:
            return self._prefill_metadata(batch)
        if batch.forward_mode == ForwardMode.DECODE:
            return self._decode_metadata(batch)
        raise ValueError(f"TT attention does not support {batch.forward_mode}")

    def _decode_metadata(self, batch: ModelWorkerBatch) -> TTAttentionMetadata:
        sequence_lengths = np.asarray(batch.seq_lens, dtype=np.int32)[: batch.real_bs]

        users = max(len(batch.input_ids), 1)
        page_table = _pad_page_table(
            _page_table(
                np.asarray(batch.cache_loc, dtype=np.int32),
                sequence_lengths,
                self.page_size,
            ),
            users,
        )
        positions = np.where(sequence_lengths > 0, sequence_lengths - 1, -1)
        positions = np.pad(
            positions.astype(np.int32),
            (0, max(0, users - len(positions))),
            constant_values=-1,
        )
        metadata = TTAttentionMetadata()
        metadata.page_table, metadata.positions = device_array(
            (page_table, positions),
            sharding=NamedSharding(self.mesh, P("data")),
        )
        return metadata

    def _prefill_metadata(self, batch: ModelWorkerBatch) -> TTAttentionMetadata:
        active_slots = np.asarray(batch.logits_indices_selector, dtype=np.int32)
        chunk_lengths = np.asarray(batch.extend_seq_lens, dtype=np.int32)[active_slots]
        prefix_lengths = np.asarray(batch.extend_prefix_lens, dtype=np.int32)[active_slots]
        if np.any(prefix_lengths % self.page_size):
            raise ValueError("TT prefill prefixes must be page-aligned")

        # The scheduler owns semantic chunking. Bucket one chunk to a
        # power-of-two page count, matching SGLang's token bucketing without
        # imposing another chunk-size policy in the attention backend.
        live_page_counts = cdiv(chunk_lengths, self.page_size)
        max_live_pages = int(np.max(live_page_counts))
        bucket_pages = max(16, 1 << (max_live_pages - 1).bit_length())
        tokens_per_sequence = bucket_pages * self.page_size

        page_table = _page_table(
            np.asarray(batch.cache_loc, dtype=np.int32),
            np.asarray(batch.seq_lens, dtype=np.int32),
            self.page_size,
        )[active_slots]

        # SGLang packs each request's live tokens inside an input bucket.
        # Repack those ragged ranges into rectangular TT rows and remember how
        # to restore the scheduler layout after attention.
        all_chunk_lengths = np.asarray(batch.extend_seq_lens, dtype=np.int32)
        input_starts = (np.cumsum(all_chunk_lengths, dtype=np.int32) - all_chunk_lengths)[
            active_slots
        ]

        token_offsets = np.arange(tokens_per_sequence, dtype=np.int32)
        input_indices = input_starts[:, None] + token_offsets
        live_tokens = token_offsets < chunk_lengths[:, None]
        prefill_input_indices = np.where(live_tokens, input_indices, 0)

        output_indices = (
            np.arange(len(active_slots), dtype=np.int32)[:, None] * tokens_per_sequence
            + token_offsets
        )
        prefill_output_indices = np.zeros(len(batch.input_ids), dtype=np.int32)
        prefill_output_indices[input_indices[live_tokens]] = output_indices[live_tokens]

        page_offsets = np.arange(bucket_pages, dtype=np.int32)
        live_pages = page_offsets < live_page_counts[:, None]
        page_columns = prefix_lengths[:, None] // self.page_size + page_offsets
        fill_page_table = np.take_along_axis(
            page_table,
            np.where(live_pages, page_columns, 0),
            axis=1,
        )
        fill_page_table[~live_pages] = 0

        metadata = TTAttentionMetadata()
        (
            metadata.page_table,
            metadata.fill_page_table,
            metadata.prefill_chunk_start,
            metadata.prefill_input_indices,
            metadata.prefill_output_indices,
        ) = device_array(
            (
                page_table,
                _pad_page_table(fill_page_table, users=8),
                prefix_lengths,
                prefill_input_indices,
                prefill_output_indices,
            ),
            sharding=NamedSharding(self.mesh, P("data")),
        )
        return metadata

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
        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
        batch_size = metadata.page_table.shape[0]
        fill_batch_indices = jnp.arange(
            batch_size,
            dtype=jnp.int32,
            out_sharding=NamedSharding(self.mesh, P("data")),
        )
        k_value = self._prefill_cache_value(k, k_cache.shape[-1])
        v_value = self._prefill_cache_value(v, v_cache.shape[-1])
        k_cache = tt_ops.paged_fill_cache(
            k_cache,
            k_value,
            metadata.fill_page_table,
            fill_batch_indices,
        )
        v_cache = tt_ops.paged_fill_cache(
            v_cache,
            v_value,
            metadata.fill_page_table,
            fill_batch_indices,
        )

        tokens = metadata.prefill_input_indices.shape[1]
        q = q.at[metadata.prefill_input_indices].get(
            out_sharding=NamedSharding(self.mesh, P("data", None, "tensor", None))
        )
        q = jnp.transpose(q, (0, 2, 1, 3))
        output = tt_ops.chunked_scaled_dot_product_attention(
            q,
            k_cache,
            v_cache,
            metadata.page_table,
            metadata.prefill_chunk_start,
            scale=layer.scaling,
        )

        output = jnp.transpose(output, (0, 2, 1, 3)).reshape(
            batch_size * tokens,
            -1,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor")),
        )
        output = output.at[metadata.prefill_output_indices].get(
            out_sharding=NamedSharding(self.mesh, P("data", "tensor"))
        )
        return output, (k_cache, v_cache)

    def _decode(self, q, k, v, layer, token_to_kv_pool):
        metadata = self.forward_metadata
        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
        num_tokens = q.shape[0]
        q = q.reshape(1, num_tokens, q.shape[1], q.shape[2])
        users = q.shape[1]
        page_table = metadata.page_table[:users]
        positions = metadata.positions[:users]

        k_update = self._decode_cache_value(k, k_cache.shape[-1])
        v_update = self._decode_cache_value(v, v_cache.shape[-1])
        k_cache = tt_ops.paged_update_cache(k_cache, k_update, positions, page_table)
        v_cache = tt_ops.paged_update_cache(v_cache, v_update, positions, page_table)

        output = tt_ops.paged_scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            page_table,
            positions,
        )
        output_sharding = NamedSharding(self.mesh, P("data", "tensor"))
        return output.reshape(num_tokens, -1, out_sharding=output_sharding), (k_cache, v_cache)

    def _prefill_cache_value(self, value, head_dim):
        value = value.at[self.forward_metadata.prefill_input_indices].get(
            out_sharding=NamedSharding(self.mesh, P("data", None, "tensor", None))
        )
        value = self._pad_head_dim(value, head_dim)
        return jnp.transpose(value, (0, 2, 1, 3))

    @staticmethod
    def _decode_cache_value(value, head_dim):
        return TTAttention._pad_head_dim(value, head_dim).reshape(
            1, value.shape[0], value.shape[1], head_dim
        )

    @staticmethod
    def _pad_head_dim(value, head_dim):
        if value.shape[-1] < head_dim:
            return jnp.pad(value, ((0, 0), (0, 0), (0, head_dim - value.shape[-1])))
        return value[..., :head_dim]

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        pages_per_request = cdiv(max_context_len, page_size)
        return max(1, 1024 * 1024 // 2 // pages_per_request // 4)
