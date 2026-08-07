"""Functional JAX operations for the model-owned GLM-5.2 DSA Indexer.

The public Indexer lives with its projection weights in ``glm5_moe``.  This
module contains only the JAX cache/scoring primitives used by that module; it
does not own IndexShare policy or sparse-attention dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.dsa.ref import score_and_select_index_tokens
from sgl_jax.srt.layers.attention.dsa_cache_ops import scatter_paged_cache
from sgl_jax.srt.utils.profiling_utils import named_scope

if TYPE_CHECKING:
    from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionMetadata


@register_pytree_node_class
@dataclass
class DSAIndexerOutput:
    """Persistent index-cache update plus an optional fresh token top-k."""

    index_cache: jax.Array
    topk_indices: jax.Array | None

    def tree_flatten(self):
        return ((self.index_cache, self.topk_indices), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)


@named_scope("CacheAndTopK")
def update_index_cache_and_select(
    q_idx: jax.Array,
    k_idx: jax.Array,
    idx_weights: jax.Array,
    index_cache: jax.Array,
    metadata: MLAAttentionMetadata,
    *,
    index_topk: int,
    compute_topk: bool,
    topk_impl: str,
    one_token_per_seq: bool,
    attention_data_partition_axis: str = "data",
) -> DSAIndexerOutput:
    """Write current index keys, then optionally select sequence-local tokens.

    Projection and orchestration remain in the owning Indexer module.  This
    helper is intentionally a single ``shard_map`` boundary so the cache write
    is ordered before scoring without introducing host/device transfers.
    """

    dpa = attention_data_partition_axis
    cache_spec = P(dpa, None, None) if index_cache.ndim == 3 else P(dpa, None, None, None)
    in_specs = (
        P(dpa, None, None),
        P(dpa, None),
        P(dpa, None),
        cache_spec,
        P(dpa),
        P(dpa),
        P(dpa),
        P(dpa),
        P(dpa),
    )
    out_specs = (cache_spec, P(dpa, None))

    def _run(q_, k_, w_, cache_, seq_lens_, pi_, cuq_, cukv_, dist_):
        flat_page_layout = cache_.ndim == 3
        page_size = cache_.shape[1] if flat_page_layout else cache_.shape[1] * cache_.shape[2]
        idx_dim = cache_.shape[-1]
        pages_per_seq = pi_.shape[0] // seq_lens_.shape[0]
        cache3d = (
            cache_ if flat_page_layout else cache_.reshape(cache_.shape[0], page_size, idx_dim)
        )
        cache3d = scatter_paged_cache(
            cache3d,
            k_,
            seq_lens_,
            pi_,
            cuq_,
            cukv_,
        )

        if compute_topk:
            topk = score_and_select_index_tokens(
                q_,
                w_,
                cache3d,
                seq_lens_,
                pi_,
                cuq_,
                cukv_,
                dist_,
                k=index_topk,
                pages_per_seq=pages_per_seq,
                one_token_per_seq=one_token_per_seq,
                topk_impl=topk_impl,
            )
        else:
            # Keep shard_map's output tree static.  The placeholder is discarded
            # immediately after the call and never reaches attention.
            topk = jnp.full((q_.shape[0], 1), -1, jnp.int32)
        updated_cache = cache3d if flat_page_layout else cache3d.reshape(cache_.shape)
        return updated_cache, topk

    index_cache, topk_indices = jax.shard_map(
        _run,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(
        q_idx,
        k_idx,
        idx_weights,
        index_cache,
        metadata.seq_lens,
        metadata.page_indices,
        metadata.cu_q_lens,
        metadata.cu_kv_lens,
        metadata.distribution,
    )
    return DSAIndexerOutput(
        index_cache=index_cache,
        topk_indices=topk_indices if compute_topk else None,
    )
