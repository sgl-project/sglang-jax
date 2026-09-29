"""Single-device CSA: compressor -> shared indexer/Top-K -> joint attention.

Queries and the uncompressed KV projection are caller inputs. Cache buffers are
caller-owned and donated; use the returned buffers for the next forward.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.csa_attention import CSAAttentionMetadata, csa_joint_attention
from sgl_jax.srt.kernels.csa_compressor import CompressorMetadata, csa_compressor
from sgl_jax.srt.kernels.csa_compressor.tune import (
    CSA_CACHE_PACKING,
    CSA_COMPRESSION_RATIO,
    CSA_FP8_BLOCK_SIZE,
    CSA_INDEX_DIM,
    CSA_INDEX_RECORD_BYTES,
)
from sgl_jax.srt.kernels.dsa.streamindex_topk import streamindex_topk


class CSACache(NamedTuple):
    main_state: jax.Array
    index_state: jax.Array
    main_nope: jax.Array
    main_rope: jax.Array
    index: jax.Array
    window: jax.Array


class CSAMetadata(NamedTuple):
    compressor: CompressorMetadata
    attention: CSAAttentionMetadata
    distribution: jax.Array


@partial(jax.jit, static_argnames=("top_k", "kv_pages_per_block", "queries_per_block"))
def csa_topk(
    index_q, index_weights, index_cache, metadata, *, top_k, kv_pages_per_block, queries_per_block
):
    """Reuse DSA retrieval; bypass it when every completed entry fits in Top-K."""
    if (
        index_q.ndim != 3
        or index_q.shape[-1] != CSA_INDEX_DIM
        or index_q.dtype != jnp.bfloat16
        or index_weights.shape != index_q.shape[:2]
        or index_weights.dtype != jnp.float32
    ):
        raise ValueError("index_q must be BF16 [T,H,128] and index_weights FP32 [T,H]")
    if (
        index_cache.ndim != 4
        or index_cache.shape[-2:] != (CSA_CACHE_PACKING, CSA_INDEX_RECORD_BYTES)
        or index_cache.dtype != jnp.uint8
    ):
        raise ValueError("index_cache must use the compressor's packed uint8 cache format")
    if min(top_k, kv_pages_per_block, queries_per_block) <= 0:
        raise ValueError("Top-K and indexer block sizes must be positive")
    attention = metadata.attention
    batch = attention.seq_lens.size
    if (
        batch == 0
        or attention.compressed_page_indices.size == 0
        or attention.compressed_page_indices.size % batch
        or attention.query_seq_ids.shape != (index_q.shape[0],)
        or metadata.compressor.positions.shape != (index_q.shape[0],)
    ):
        raise ValueError("retrieval requires matching token metadata and fixed-stride page tables")
    reqs = attention.query_seq_ids
    visible = (metadata.compressor.positions + 1) // CSA_COMPRESSION_RATIO
    visible = jnp.where(reqs >= 0, visible, 0)
    candidates = jnp.broadcast_to(jnp.arange(top_k, dtype=jnp.int32), (index_q.shape[0], top_k))

    def retrieve():
        return streamindex_topk(
            index_q,
            index_weights,
            index_cache,
            attention.seq_lens,
            attention.compressed_page_indices,
            attention.cu_q_lens,
            metadata.distribution,
            k=top_k,
            compression_ratio=CSA_COMPRESSION_RATIO,
            num_kv_pages_per_block=kv_pages_per_block,
            num_queries_per_block=queries_per_block,
            topk_backend="xla",
        )

    page_size = index_cache.shape[1] * index_cache.shape[2]
    capacity = attention.compressed_page_indices.size // attention.seq_lens.size * page_size
    if capacity <= top_k:
        indices = candidates
    else:
        indices = jax.lax.cond(jnp.max(visible) <= top_k, lambda: candidates, retrieve)
    return jnp.where((indices >= 0) & (indices < visible[:, None]), indices, -1)


@partial(
    jax.jit,
    static_argnames=(
        "compressor_schedule",
        "attention_schedule",
        "scale",
        "window_size",
        "top_k",
        "kv_pages_per_block",
        "queries_per_block",
    ),
    donate_argnames=("cache",),
)
def csa_attention(
    x,
    fused_weight,
    main_ape,
    index_ape,
    main_norm,
    index_norm,
    cos,
    sin,
    index_q,
    index_weights,
    q,
    new_kv,
    attention_sink,
    cache,
    metadata,
    *,
    compressor_schedule,
    attention_schedule,
    scale,
    window_size,
    top_k,
    kv_pages_per_block,
    queries_per_block,
):
    """Return (attention output, updated cache), including SWA ring writeback.

    Index queries are BF16 [T,H,128], with FP32 [T,H] index weights.
    Metadata uses fixed-stride compressed page tables shared with DSA. Only the
    last window_size new tokens per request may have nonnegative ring write slots.
    """
    tokens = x.shape[0]
    if (
        q.ndim != 3
        or q.shape[0] != tokens
        or new_kv.shape != (tokens, q.shape[-1])
        or index_q.ndim != 3
        or index_q.shape[0] != tokens
        or index_weights.shape != index_q.shape[:2]
        or metadata.attention.window_write_locations.shape != (tokens,)
    ):
        raise ValueError("CSA queries, projections and write locations must share the token axis")
    compressed = csa_compressor(
        x,
        fused_weight,
        main_ape,
        index_ape,
        main_norm,
        index_norm,
        cos,
        sin,
        *cache[:5],
        metadata.compressor,
        schedule=compressor_schedule,
    )
    if attention_schedule.decode:
        capacity = (
            metadata.attention.compressed_page_indices.size
            // metadata.attention.seq_lens.size
            * compressed[2].shape[1]
        )
        top_k = min(top_k, 1 << (capacity - 1).bit_length())
    indices = csa_topk(
        index_q,
        index_weights,
        compressed[4],
        metadata,
        top_k=top_k,
        kv_pages_per_block=kv_pages_per_block,
        queries_per_block=queries_per_block,
    )
    output, window = csa_joint_attention(
        q,
        new_kv,
        cache.window,
        compressed[2],
        compressed[3],
        indices,
        attention_sink,
        metadata.attention,
        scale=scale,
        schedule=attention_schedule,
        window_size=window_size,
        compression_ratio=CSA_COMPRESSION_RATIO,
        fp8_scale_block=CSA_FP8_BLOCK_SIZE,
        rows_per_group=CSA_CACHE_PACKING,
    )
    return output, CSACache(*compressed, window)
