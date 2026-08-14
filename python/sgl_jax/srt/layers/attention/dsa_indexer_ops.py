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

from sgl_jax.srt.kernels.dsa.fused_k_projection_scatter import (
    fused_k_projection_scatter_pallas,
)
from sgl_jax.srt.kernels.dsa.indexer import compute_scores_and_select_topk_indices
from sgl_jax.srt.layers.attention.dsa_cache_ops import scatter_paged_cache

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
    output_physical_slots: bool = False,
    attention_data_partition_axis: str = "data",
) -> DSAIndexerOutput:
    """Write current index keys, then optionally select index tokens.

    Projection and orchestration remain in the owning Indexer module.  This
    helper is intentionally a single ``shard_map`` boundary so the cache write
    is ordered before scoring without introducing host/device transfers.
    ``output_physical_slots`` switches exact sparse attention to the resolver-
    free ABI where the indexer returns flattened KV-cache slots.
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
        with jax.named_scope("IndexCacheWrite"):
            cache3d = scatter_paged_cache(
                cache3d,
                k_,
                seq_lens_,
                pi_,
                cuq_,
                cukv_,
            )

        if compute_topk:
            topk = compute_scores_and_select_topk_indices(
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
                output_physical_slots=output_physical_slots,
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


def project_index_keys_update_cache_and_select(
    q_idx: jax.Array,
    hidden_states: jax.Array,
    wk: jax.Array,
    k_norm_weight: jax.Array,
    k_norm_bias: jax.Array,
    rope_cos: jax.Array,
    rope_sin: jax.Array,
    hadamard: jax.Array,
    idx_weights: jax.Array,
    index_cache: jax.Array,
    metadata: MLAAttentionMetadata,
    *,
    index_topk: int,
    compute_topk: bool,
    topk_impl: str,
    output_physical_slots: bool = False,
    attention_data_partition_axis: str = "data",
) -> DSAIndexerOutput:
    """Fuse page-aligned Extend K projection into its paged-cache scatter.

    Q and index-weight projection remain native XLA operations in the owning
    model.  This helper selects the persistent Pallas K path only when every
    sequence appends complete cache pages.  Other Extend layouts retain the
    original native K projection plus functional scatter semantics.
    """

    dpa = attention_data_partition_axis
    cache_spec = P(dpa, None, None) if index_cache.ndim == 3 else P(dpa, None, None, None)
    in_specs = (
        P(dpa, None, None),
        P(dpa, None),
        P(None, None),
        P(None),
        P(None),
        P(dpa, None),
        P(dpa, None),
        P(None, None),
        P(dpa, None),
        cache_spec,
        P(dpa),
        P(dpa),
        P(dpa),
        P(dpa),
        P(dpa),
    )
    out_specs = (cache_spec, P(dpa, None))

    def _run(
        q_,
        hidden_,
        wk_,
        norm_weight_,
        norm_bias_,
        cos_,
        sin_,
        hadamard_,
        weights_,
        cache_,
        seq_lens_,
        page_indices_,
        cu_q_lens_,
        cu_kv_lens_,
        distribution_,
    ):
        flat_page_layout = cache_.ndim == 3
        page_size = cache_.shape[1] if flat_page_layout else cache_.shape[1] * cache_.shape[2]
        idx_dim = cache_.shape[-1]
        pages_per_seq = page_indices_.shape[0] // seq_lens_.shape[0]
        cache3d = (
            cache_ if flat_page_layout else cache_.reshape(cache_.shape[0], page_size, idx_dim)
        )

        def project_keys_xla() -> jax.Array:
            key = jax.lax.dot_general(
                hidden_,
                wk_,
                (((1,), (0,)), ((), ())),
                preferred_element_type=hidden_.dtype,
            )
            mean = jnp.mean(key, axis=-1, keepdims=True)
            variance = jnp.var(key, axis=-1, keepdims=True)
            key = (key - mean) / jnp.sqrt(variance + 1e-5)
            key = key * norm_weight_ + norm_bias_

            key_rope = key[:, :64]
            even = key_rope[:, ::2]
            odd = key_rope[:, 1::2]
            rotated = jnp.stack(
                (even * cos_ - odd * sin_, odd * cos_ + even * sin_),
                axis=-1,
            ).reshape(key_rope.shape)
            key = key.at[:, :64].set(rotated)
            return jnp.einsum("td,de->te", key, hadamard_)

        def project_and_scatter_xla(current_cache: jax.Array) -> jax.Array:
            with jax.named_scope("IndexKeyProjection"):
                key = project_keys_xla()
            with jax.named_scope("IndexCacheWrite"):
                return scatter_paged_cache(
                    current_cache,
                    key,
                    seq_lens_,
                    page_indices_,
                    cu_q_lens_,
                    cu_kv_lens_,
                )

        can_use_fused_kernel = (
            jax.default_backend() == "tpu"
            and page_size == 64
            and idx_dim == 128
            and hidden_.shape[0] >= page_size
            and hidden_.shape[0] % page_size == 0
            and hidden_.dtype == jnp.bfloat16
            and wk_.dtype == jnp.bfloat16
            and cache3d.dtype == jnp.bfloat16
            and cos_.shape == (hidden_.shape[0], 32)
            and sin_.shape == (hidden_.shape[0], 32)
            and hadamard_.shape == (128, 128)
            and hadamard_.dtype == jnp.float32
        )

        if can_use_fused_kernel:
            num_tokens = hidden_.shape[0]
            num_token_pages = num_tokens // page_size
            q_lens = cu_q_lens_[1:] - cu_q_lens_[:-1]
            prefix_lens = seq_lens_ - q_lens
            token_starts = jnp.arange(num_token_pages, dtype=jnp.int32) * page_size
            seq_ids = jnp.searchsorted(cu_q_lens_[1:], token_starts, side="right")
            seq_ids = jnp.clip(seq_ids, 0, seq_lens_.shape[0] - 1)
            sequence_token_starts = cu_q_lens_[seq_ids]
            logical_token_starts = prefix_lens[seq_ids] + token_starts - sequence_token_starts
            page_table_offsets = cu_kv_lens_[seq_ids] // page_size
            page_table_indices = page_table_offsets + logical_token_starts // page_size
            safe_page_table_indices = jnp.clip(
                page_table_indices,
                0,
                page_indices_.shape[0] - 1,
            )
            target_pages = page_indices_[safe_page_table_indices].astype(jnp.int32)
            aligned = (
                (cu_q_lens_[0] == 0)
                & (cu_q_lens_[-1] == num_tokens)
                & jnp.all(q_lens >= 0)
                & jnp.all(q_lens % page_size == 0)
                & jnp.all(prefix_lens >= 0)
                & jnp.all(prefix_lens % page_size == 0)
                & jnp.all(page_table_indices == safe_page_table_indices)
                & jnp.all(target_pages >= 0)
                & jnp.all(target_pages < cache3d.shape[0])
            )

            def run_fused(current_cache: jax.Array) -> jax.Array:
                with jax.named_scope("FusedIndexKeyProjectionScatter"):
                    return fused_k_projection_scatter_pallas(
                        hidden_,
                        wk_,
                        norm_weight_,
                        norm_bias_,
                        cos_,
                        sin_,
                        hadamard_,
                        current_cache,
                        target_pages,
                        page_size=page_size,
                    )

            cache3d = jax.lax.cond(
                aligned,
                run_fused,
                project_and_scatter_xla,
                cache3d,
            )
        else:
            cache3d = project_and_scatter_xla(cache3d)

        if compute_topk:
            topk = compute_scores_and_select_topk_indices(
                q_,
                weights_,
                cache3d,
                seq_lens_,
                page_indices_,
                cu_q_lens_,
                cu_kv_lens_,
                distribution_,
                k=index_topk,
                pages_per_seq=pages_per_seq,
                one_token_per_seq=False,
                topk_impl=topk_impl,
                output_physical_slots=output_physical_slots,
            )
        else:
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
        hidden_states,
        wk,
        k_norm_weight,
        k_norm_bias,
        rope_cos,
        rope_sin,
        hadamard,
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
