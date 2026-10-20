"""One rank-local CSA step; allocation and update commits stay with the caller."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.csa_attention import CSAAttentionMetadata, csa_joint_attention
from sgl_jax.srt.kernels.csa_attention.tune import LANES
from sgl_jax.srt.kernels.csa_compressor.compressor import (
    CompressorMetadata,
    csa_compressor,
)
from sgl_jax.srt.kernels.dsa.streamindex_topk import get_dtype_packing, streamindex_topk


class CSAMetadata(NamedTuple):
    """Backend-produced views sharing request order, positions and compressed pages.

    index_page_indices:[B, pages_per_request], with zero padding; distribution:[3]
    follows streamindex_topk. All views share token boundaries and logical page
    order; positions/lengths use original tokens, cache locations compressed rows.
    """

    compressor: CompressorMetadata
    attention: CSAAttentionMetadata
    index_page_indices: jax.Array
    distribution: jax.Array


def csa_step(
    x,
    q,
    new_kv,
    index_q,
    index_weights,
    main_state,
    index_state,
    window_cache,
    main_cache,
    index_cache,
    fused_weight,
    main_ape,
    index_ape,
    main_norm,
    index_norm,
    cos,
    sin,
    attention_sink,
    metadata: CSAMetadata,
    *,
    compressor_schedule,
    attention_schedule,
    indexer_schedule,
    softmax_scale: float,
    top_k: int = 512,
    window_size: int = 128,
    page_size: int = 128,
    norm_eps: float = 1e-6,
):
    """Return output and complete state/indexer_state/compressed/indexer/swa updates.

    Q/new KV and index Q/weights are model-prepared. Fresh state is initialized
    by the backend; continuation keeps state. No pool allocation or commit occurs here.
    Caller maps state/indexer_state to state-pool compressor/indexer; the other
    three keys go to the KV pool. Skip backend writes for all five owned families.
    Equal-score Top-K ties follow the existing indexer, with no extra index-order rule.
    """
    md = metadata.attention
    if page_size not in (128, 256) or top_k <= 0 or top_k % LANES:
        raise ValueError("CSA requires page_size 128/256 and top_k a positive multiple of 128")
    if x.ndim != 2 or q.ndim != 3:
        raise ValueError("x must be [T,hidden] and q must be [T,heads,512]")
    tokens, requests = q.shape[0], md.seq_lens.size
    cm = metadata.compressor
    for value, shape in (
        (md.query_seq_ids, (tokens,)),
        (md.seq_lens, (requests,)),
        (md.cu_q_lens, (requests + 1,)),
        (cm.cu_q_lens, (requests + 1,)),
        (cm.state_indices, (requests,)),
        (cm.positions, (tokens,)),
        (cm.cache_locations, (tokens,)),
    ):
        if value.shape != shape or value.dtype != jnp.int32:
            raise ValueError("CSA metadata must share int32 token/request dimensions")
    if tokens and not requests:
        raise ValueError("nonempty CSA input requires at least one request row")
    if index_q.ndim != 3 or index_q.dtype != jnp.bfloat16:
        raise ValueError("index_q must be BF16 [T,heads,128]")
    if (
        metadata.index_page_indices.ndim != 2
        or metadata.index_page_indices.shape[0] != md.seq_lens.size
        or metadata.index_page_indices.shape[1] == 0
    ):
        raise ValueError("index_page_indices must be [requests, pages_per_request]")
    if (
        metadata.index_page_indices.dtype != jnp.int32
        or metadata.distribution.shape != (3,)
        or metadata.distribution.dtype != jnp.int32
    ):
        raise ValueError("indexer page tables and distribution must be int32")
    if (
        x.shape[0] != q.shape[0]
        or index_q.shape[0] != q.shape[0]
        or index_q.shape[-1] != 128
        or index_weights.shape != index_q.shape[:2]
    ):
        raise ValueError("CSA inputs must share token order and index_q must have dimension 128")
    if main_cache.shape[:2] != index_cache.shape[:2] or main_cache.shape[1] != page_size // 4:
        raise ValueError("c4 and indexer caches must share physical pages")
    state, istate, compressed, indexer = csa_compressor(
        x,
        fused_weight,
        main_ape,
        index_ape,
        main_norm,
        index_norm,
        cos,
        sin,
        main_state,
        index_state,
        main_cache,
        index_cache,
        metadata.compressor,
        schedule=compressor_schedule,
        norm_eps=norm_eps,
    )
    if q.shape[0] == 0:
        return jnp.zeros_like(q), dict(
            state=state,
            indexer_state=istate,
            compressed=compressed,
            indexer=indexer,
            swa=window_cache,
        )
    # The indexer packs BF16 rows in pairs; preserve the pool's contiguous row order.
    packing = get_dtype_packing(indexer.dtype)
    indexer_view = indexer.reshape(
        indexer.shape[0], indexer.shape[1] // packing, packing, indexer.shape[2]
    )
    selected = streamindex_topk(
        index_q,
        index_weights,
        indexer_view,
        md.seq_lens,
        metadata.index_page_indices.reshape(-1),
        md.cu_q_lens,
        metadata.distribution,
        k=top_k,
        compression_ratio=4,
        num_kv_pages_per_block=indexer_schedule.kv_pages_per_block,
        num_queries_per_block=indexer_schedule.query_tile,
        decode_req_batch_size=indexer_schedule.decode_request_tile,
        topk_backend="xla",
    )
    request = jnp.clip(md.query_seq_ids, 0, md.seq_lens.size - 1)
    token = jnp.arange(q.shape[0])
    live = (md.query_seq_ids >= 0) & (md.query_seq_ids < md.seq_lens.size)
    live &= (token >= md.cu_q_lens[request]) & (token < md.cu_q_lens[request + 1])
    selected = jnp.where(live[:, None], selected, -1)
    output, swa = csa_joint_attention(
        q,
        new_kv,
        window_cache,
        compressed,
        selected,
        attention_sink,
        md,
        scale=softmax_scale,
        schedule=attention_schedule,
        window_size=window_size,
        window_page_size=page_size,
    )
    return output, dict(
        state=state, indexer_state=istate, compressed=compressed, indexer=indexer, swa=swa
    )
