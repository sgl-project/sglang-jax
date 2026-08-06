"""Sparse MLA backend consuming finalized DSA top-k selections.

The model-owned Indexer performs projection, index-key cache management, and
score/top-k.  This module only updates the standard MLA KV cache and chooses
dense or sparse MLA from the supplied finalized token indices.

The legacy path runs page-level sparse MLA for decode and dense MLA for
extend. The exact path uses a functional ``lax.top_k`` StreamIndex fallback,
maps sequence-local positions to physical cache slots, and runs the fused
SparseCore-gather + TensorCore-attention kernel for both extend and decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.dsa.exact_attention import sparse_core_tensor_core_dsa
from sgl_jax.srt.kernels.dsa.sparse_mla import sparse_mla_page_level
from sgl_jax.srt.kernels.mla.v2.kernel import mla_ragged_paged_attention
from sgl_jax.srt.layers.attention.dsa_cache_ops import scatter_paged_cache
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend
from sgl_jax.srt.utils.profiling_utils import named_scope

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_attention import RadixAttention
    from sgl_jax.srt.mem_cache.memory_pool import KVCache
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class DSASparseAttentionBackend(MLAAttentionBackend):
    """Absorbed MLA that consumes finalized token indices from the Indexer."""

    def __init__(
        self,
        *,
        sparse_impl: str = "exact",
        **mla_kwargs,
    ):
        if sparse_impl not in ("page", "exact"):
            raise ValueError(f"unknown DSA sparse implementation: {sparse_impl}")
        self.sparse_impl = sparse_impl
        super().__init__(**mla_kwargs)

    def tree_flatten(self):
        children, aux = super().tree_flatten()
        aux = {
            **aux,
            "sparse_impl": self.sparse_impl,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(
            sparse_impl=aux_data.get("sparse_impl", "exact"),
            num_attn_heads=aux_data["num_attn_heads"],
            kv_lora_rank=aux_data["kv_lora_rank"],
            qk_nope_head_dim=aux_data["qk_nope_head_dim"],
            qk_rope_head_dim=aux_data["qk_rope_head_dim"],
            v_head_dim=aux_data["v_head_dim"],
            page_size=aux_data["page_size"],
            mesh=aux_data.get("mesh"),
            attention_data_partition_axis=aux_data.get("attention_data_partition_axis", "data"),
            vmem_limit_bytes=aux_data["vmem_limit_bytes"],
            num_kv_pages_per_block=aux_data["num_kv_pages_per_block"],
            num_queries_per_block=aux_data["num_queries_per_block"],
            decode_batch_size=aux_data["decode_batch_size"],
        )
        obj.forward_metadata = children[0]
        return obj

    @named_scope
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        *,
        q_rope: jax.Array,
        k_rope: jax.Array,
        topk_indices: jax.Array | None = None,
    ):
        del v
        layer_id = layer.layer_id
        new_kv_c = k if k.ndim == 2 else jnp.squeeze(k, axis=1)
        new_k_pe = k_rope if k_rope.ndim == 2 else jnp.squeeze(k_rope, axis=1)

        kv_cache = token_to_kv_pool.get_fused_kv_buffer(layer_id)
        sm_scale = (
            (1.0 / jnp.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim))
            if (layer is None or layer.scaling is None)
            else layer.scaling
        )
        dpa = self.attention_data_partition_axis
        md = self.forward_metadata

        if topk_indices is None:
            return self._run_dense(
                q, q_rope, new_kv_c, new_k_pe, kv_cache, sm_scale, layer, dpa, md
            )

        if self.sparse_impl == "exact":
            return self._run_exact(
                q,
                q_rope,
                new_kv_c,
                new_k_pe,
                kv_cache,
                topk_indices,
                sm_scale,
                dpa,
                md,
            )
        return self._run_sparse(
            q,
            q_rope,
            new_kv_c,
            new_k_pe,
            kv_cache,
            topk_indices,
            sm_scale,
            dpa,
            md,
        )

    def _run_exact(self, ql, qpe, kvc, kpe, cache, topk, sm_scale, dpa, md):
        """Write the fused KV cache, map logical top-k to slots, then attend."""
        in_specs = (
            P(dpa, "tensor", None),
            P(dpa, "tensor", None),
            P(dpa, None),
            P(dpa, None),
            P(dpa, None, None, None),
            P(dpa, None),
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),
        )
        out_specs = (P(dpa, "tensor", None), P(dpa, None, None, None))

        def _run(ql_, qpe_, kvc_, kpe_, cache_, topk_, seq_lens_, pi_, cuq_, cukv_, dist_):
            del dist_
            page_size = cache_.shape[1] * cache_.shape[2]
            cache3d = cache_.reshape(cache_.shape[0], page_size, cache_.shape[-1])
            cache3d = _scatter_fused_kv_paged(
                cache3d,
                kvc_,
                kpe_,
                seq_lens_,
                pi_,
                cuq_,
                cukv_,
                kv_lora_rank=self.kv_lora_rank,
            )
            physical_slots, selected_counts = _logical_topk_to_physical_slots(
                topk_, seq_lens_, pi_, cuq_, cukv_, page_size
            )

            # SparseCore requires bq_sparse*K to be a gather-wave multiple.
            # For C=2 decode, bq_sparse=2 is the smallest valid choice at K=2048.
            # Larger extend batches use the kernel's validated 128-query tile.
            bq_sparse = 2 if ql_.shape[0] < 128 else 128
            bq = 1 if bq_sparse == 2 else 32
            q_padded = _pad_first_axis(ql_, bq_sparse)
            qpe_padded = _pad_first_axis(qpe_, bq_sparse)
            slots_padded = _pad_first_axis(physical_slots, bq_sparse)
            counts_padded = _pad_first_axis(selected_counts, bq_sparse)
            output = sparse_core_tensor_core_dsa(
                q_padded,
                qpe_padded,
                cache3d.reshape(-1, cache3d.shape[-1]),
                slots_padded,
                counts_padded,
                sm_scale,
                bq_sparse=bq_sparse,
                bq=bq,
                b_topk=128,
            )
            return output[: ql_.shape[0]], cache3d.reshape(cache_.shape)

        return jax.shard_map(_run, in_specs=in_specs, out_specs=out_specs, check_vma=False)(
            ql,
            qpe,
            kvc,
            kpe,
            cache,
            topk,
            md.seq_lens,
            md.page_indices,
            md.cu_q_lens,
            md.cu_kv_lens,
            md.distribution,
        )

    def _run_sparse(self, ql, qpe, kvc, kpe, cache, topk, sm_scale, dpa, md):
        in_specs = (
            P(dpa, "tensor", None),
            P(dpa, "tensor", None),
            P(dpa, None),
            P(dpa, None),
            P(dpa, None, None, None),
            P(dpa, None),  # topk [T, k]
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),
        )
        out_specs = (P(dpa, "tensor", None), P(dpa, None, None, None))

        def _run(ql_, qpe_, kvc_, kpe_, cache_, topk_, seq_lens_, pi_, cuq_, cukv_, dist_):
            page_size = cache_.shape[1] * cache_.shape[2]
            pages_per_seq = pi_.shape[0] // seq_lens_.shape[0]
            return sparse_mla_page_level(
                ql_,
                qpe_,
                kvc_,
                kpe_,
                cache_,
                seq_lens_,
                topk_,
                pi_,
                cuq_,
                cukv_,
                dist_,
                None,
                sm_scale=float(sm_scale),
                page_size=page_size,
                pages_per_seq=pages_per_seq,
                kv_lora_rank=self.kv_lora_rank,
                k_pages_max=512,
                vmem_limit_bytes=self.vmem_limit_bytes,
            )

        return jax.shard_map(_run, in_specs=in_specs, out_specs=out_specs, check_vma=False)(
            ql,
            qpe,
            kvc,
            kpe,
            cache,
            topk,
            md.seq_lens,
            md.page_indices,
            md.cu_q_lens,
            md.cu_kv_lens,
            md.distribution,
        )

    def _run_dense(self, ql, qpe, kvc, kpe, cache, sm_scale, layer, dpa, md):
        in_specs = (
            P(dpa, "tensor", None),
            P(dpa, "tensor", None),
            P(dpa, None),
            P(dpa, None),
            P(dpa, None, None, None),
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),
        )
        out_specs = (P(dpa, "tensor", None), P(dpa, None, None, None))
        sw = layer.sliding_window_size if layer is not None else None
        sc = layer.logit_cap if layer is not None else None

        def _run(ql_, qpe_, kvc_, kpe_, cache_, seq_lens_, pi_, cuq_, cukv_, dist_):
            return mla_ragged_paged_attention(
                ql_,
                qpe_,
                kvc_,
                kpe_,
                cache_,
                seq_lens_,
                pi_,
                cuq_,
                cukv_,
                dist_,
                sm_scale=sm_scale,
                sliding_window=sw,
                soft_cap=sc,
                num_kv_pages_per_block=self.num_kv_pages_per_block,
                num_queries_per_block=self.num_queries_per_block,
                decode_batch_size=self.decode_batch_size,
                vmem_limit_bytes=self.vmem_limit_bytes,
            )

        return jax.shard_map(_run, in_specs=in_specs, out_specs=out_specs, check_vma=False)(
            ql,
            qpe,
            kvc,
            kpe,
            cache,
            md.seq_lens,
            md.page_indices,
            md.cu_q_lens,
            md.cu_kv_lens,
            md.distribution,
        )


def _scatter_fused_kv_paged(
    cache3d: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    *,
    kv_lora_rank: int,
) -> jax.Array:
    """Pack latent KV + RoPE exactly as ``MLATokenToKVPool`` and scatter it."""
    rope_offset = (kv_lora_rank + 127) // 128 * 128
    packed = jnp.zeros((new_kv_c.shape[0], cache3d.shape[-1]), dtype=cache3d.dtype)
    packed = packed.at[:, :kv_lora_rank].set(new_kv_c.astype(cache3d.dtype))
    packed = packed.at[:, rope_offset : rope_offset + new_k_pe.shape[-1]].set(
        new_k_pe.astype(cache3d.dtype)
    )
    return scatter_paged_cache(
        cache3d,
        packed,
        seq_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
    )


def _logical_topk_to_physical_slots(
    topk: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    page_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Map per-sequence logical token positions to flattened cache rows.

    ``page_indices`` is ragged-packed. Sequence ``i`` starts at
    ``cu_kv_lens[i] // page_size``; using a fixed ``i * pages_per_seq`` stride
    would silently cross sequence boundaries for unequal context lengths.
    """
    num_tokens, topk_size = topk.shape
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_ids = jnp.searchsorted(cu_q_lens[1:], token_ids, side="right")
    seq_ids = jnp.clip(seq_ids, 0, seq_lens.shape[0] - 1)

    logical = jnp.maximum(topk, 0)
    page_ptr = cu_kv_lens[seq_ids, None] // page_size + logical // page_size
    ptr_in_bounds = (page_ptr >= 0) & (page_ptr < page_indices.shape[0])
    safe_ptr = jnp.clip(page_ptr, 0, page_indices.shape[0] - 1)
    physical_pages = page_indices[safe_ptr]
    query_valid = token_ids < cu_q_lens[-1]
    valid = (
        query_valid[:, None]
        & (topk >= 0)
        & (logical < seq_lens[seq_ids, None])
        & ptr_in_bounds
        & (physical_pages >= 0)
    )
    physical_slots = physical_pages * page_size + logical % page_size
    physical_slots = jnp.where(valid, physical_slots, jnp.int32(0))
    selected_counts = jnp.sum(valid, axis=1, dtype=jnp.int32)

    # ``lax.top_k`` places all finite values before -inf padding, so valid
    # entries are a prefix as required by selected_counts. Keep this helper
    # pure/JIT-compatible; correctness tests cover the prefix invariant.
    del topk_size
    return physical_slots.astype(jnp.int32), selected_counts


def _pad_first_axis(x: jax.Array, multiple: int) -> jax.Array:
    pad = (-x.shape[0]) % multiple
    return jnp.pad(x, ((0, pad), *((0, 0),) * (x.ndim - 1)))
