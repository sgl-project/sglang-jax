import logging
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    ragged_paged_attention as ragged_paged_attention_v3,
)
from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput
from sgl_jax.srt.utils import cdiv
from sgl_jax.srt.utils.jax_utils import device_array
from sgl_jax.srt.utils.profiling_utils import named_scope

logger = logging.getLogger(__name__)


def _per_dp_cumsum(lens, dp_size: int, per_dp_bs: int) -> np.ndarray:
    """`(dp*(per_dp_bs+1),)` row-wise cumsum with leading 0 per DP rank.

    At dp=1 reduces to `[0, *cumsum(lens)]`. Replaces the previous
    ``if dp>1: 2D else: 1D`` branches (review #1108).
    """
    cu = np.zeros((dp_size, per_dp_bs + 1), dtype=np.int32)
    cu[:, 1:] = np.cumsum(np.asarray(lens, dtype=np.int32).reshape(dp_size, per_dp_bs), axis=1)
    return cu.ravel()


@register_pytree_node_class
@dataclass
class FlashAttentionMetadata:
    """Metadata to be init once in the model forward pass,
    each layer's forward pass can reuse the metadata.

    For each init metadata function, we will try set up them in below order
    """

    cu_q_lens: jax.Array = None
    cu_kv_lens: jax.Array = None
    page_indices: jax.Array = None
    swa_page_indices: jax.Array = None
    seq_lens: jax.Array = None
    distribution: jax.Array = None
    custom_mask: jax.Array = None

    def tree_flatten(self):
        children = (
            self.cu_q_lens,
            self.cu_kv_lens,
            self.page_indices,
            self.swa_page_indices,
            self.seq_lens,
            self.distribution,
            self.custom_mask,
        )

        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)

        obj.cu_q_lens = children[0]
        obj.cu_kv_lens = children[1]
        obj.page_indices = children[2]
        obj.swa_page_indices = children[3]
        obj.seq_lens = children[4]
        obj.distribution = children[5]
        obj.custom_mask = children[6]

        return obj


@dataclass
class FlashAttention(AttentionBackend):
    """Native Attention layer for variable-length sequences using ForwardBatch."""

    def __init__(
        self,
        num_attn_heads,
        num_kv_heads,
        head_dim,
        page_size: int = 1,
        kv_partition_axis: str = "tensor",
        attention_data_partition_axis: str = "data",
        mesh: jax.sharding.Mesh = None,
    ):
        self.num_heads = num_attn_heads
        if num_kv_heads is not None:
            self.num_kv_heads = num_kv_heads
        else:
            self.num_kv_heads = num_attn_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.kv_partition_axis = kv_partition_axis
        self.attention_data_partition_axis = attention_data_partition_axis
        self.forward_metadata = nnx.data(FlashAttentionMetadata())
        self.mesh = mesh
        # SWA dual-pool support: set by model_runner after pool creation.
        # Accessed on host during metadata construction.
        #
        # MSA ctx-bucket precompile (G3): when set to a sorted page-count list
        # (e.g. [16, 64, 512]) by model_runner for MSA models, decode-time
        # page_indices is truncated to the smallest bucket >= max(seq_pages) so
        # the traced graph's pages_per_seq — and hence _msa_inner's ik_buf
        # gather — scales with actual context instead of max_context_len.
        # None → original behaviour (single max_ctx-sized page_indices).
        self.msa_ctx_page_buckets: list[int] | None = None

    def get_forward_metadata(
        self,
        batch: ModelWorkerBatch,
    ):
        """Return the metadata for a forward pass."""
        metadata = FlashAttentionMetadata()

        if batch.dp_size <= 0:
            raise ValueError(f"Invalid dp_size: {batch.dp_size}")
        if batch.per_dp_bs_size <= 0:
            raise ValueError(f"Invalid per_dp_bs_size: {batch.per_dp_bs_size}")
        if batch.per_dp_bs_size * batch.dp_size != len(batch.seq_lens):
            raise ValueError(
                "Inconsistent DP batch metadata: expected per_dp_bs_size * dp_size == len(seq_lens), "
                f"got {batch.per_dp_bs_size} * {batch.dp_size} != {len(batch.seq_lens)}"
            )
        if len(batch.cache_loc) % batch.dp_size != 0:
            raise ValueError(
                "Inconsistent cache_loc layout for DP sharding: "
                f"len(cache_loc)={len(batch.cache_loc)} is not divisible by dp_size={batch.dp_size}"
            )

        total_loc_len = len(batch.cache_loc)
        per_dp_loc_len = total_loc_len // batch.dp_size

        # Reshape cache_loc to (dp_size, per_dp_loc_len) — O(1) view
        cache_loc_2d = batch.cache_loc.reshape(batch.dp_size, per_dp_loc_len)
        # Stride by page_size to pick one slot per page — O(1) view
        strided_2d = cache_loc_2d[:, :: self.page_size]

        # MSA ctx-bucket: drop the per-DP zero-padded page tail. cache_loc is
        # cumsum-packed within a DP rank (schedule_batch._merge_cache_loc), so
        # all valid pages live in the [:, :sum(seq_pages)] prefix and
        # sum(seq_pages) <= bs_per_dp * max(seq_pages) <= bs_per_dp * bucket.
        # RPA indexes via cu_kv_lens (unaffected); _msa_inner derives
        # pages_per_seq from the truncated shape (the point of this opt).
        # DECODE only — EXTEND uses the largest cache_loc bucket unconditionally.
        if self.msa_ctx_page_buckets is not None and batch.forward_mode == ForwardMode.DECODE:
            seq_pages = (np.asarray(batch.seq_lens) + self.page_size - 1) // self.page_size
            max_pages = int(seq_pages.max(initial=0))
            bucket = next((b for b in self.msa_ctx_page_buckets if b >= max_pages), None)
            if bucket is not None:
                n_keep = batch.per_dp_bs_size * bucket
                if n_keep < strided_2d.shape[1]:
                    strided_2d = strided_2d[:, :n_keep]

        # Physical slot -> Physical page index
        page_indices = (strided_2d // self.page_size).ravel()

        # SWA page indices: stride first, then apply mapping on ~N_pages entries
        # instead of ~N_tokens entries (256x fewer random accesses)
        swa_page_indices = None
        swa_mapping = getattr(self, "swa_index_mapping", None)
        if swa_mapping is not None:
            n_pages = strided_2d.shape[1]
            swa_strided = np.empty((batch.dp_size, n_pages), dtype=np.int32)
            for i in range(batch.dp_size):
                mapping = swa_mapping[i] if isinstance(swa_mapping, list) else swa_mapping
                swa_strided[i] = mapping[strided_2d[i]]
            swa_page_indices = (swa_strided // self.page_size).ravel()

        # cu_q_lens per DP rank section (each section starts from 0)
        if batch.forward_mode == ForwardMode.EXTEND:
            ext_2d = batch.extend_seq_lens.reshape(batch.dp_size, batch.per_dp_bs_size)
            cu_q_2d = np.zeros((batch.dp_size, batch.per_dp_bs_size + 1), dtype=np.int32)
            cu_q_2d[:, 1:] = np.cumsum(ext_2d, axis=1)
            cu_q_lens = cu_q_2d.ravel()
        elif batch.forward_mode == ForwardMode.DECODE:
            single_cu = np.arange(batch.per_dp_bs_size + 1, dtype=np.int32)
            cu_q_lens = np.tile(single_cu, batch.dp_size)
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        seq_lens = batch.seq_lens

        aligned_seq_lens = (
            (batch.seq_lens + self.page_size - 1) // self.page_size
        ) * self.page_size

        # cu_kv_lens per DP rank section — vectorized 2D cumsum
        aligned_2d = aligned_seq_lens.reshape(batch.dp_size, batch.per_dp_bs_size)
        cu_kv_2d = np.zeros((batch.dp_size, batch.per_dp_bs_size + 1), dtype=np.int32)
        cu_kv_2d[:, 1:] = np.cumsum(aligned_2d, axis=1)
        cu_kv_lens = cu_kv_2d.ravel()

        # distribution — vectorized
        seq_lens_2d = batch.seq_lens.reshape(batch.dp_size, batch.per_dp_bs_size)
        local_num_seqs = np.sum(seq_lens_2d > 0, axis=1, dtype=np.int32)
        if batch.forward_mode == ForwardMode.DECODE:
            distribution = np.repeat(local_num_seqs, 3)
        elif batch.forward_mode == ForwardMode.EXTEND:
            distribution = np.column_stack(
                [np.zeros_like(local_num_seqs), local_num_seqs, local_num_seqs]
            ).ravel()
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        (
            metadata.cu_q_lens,
            metadata.cu_kv_lens,
            metadata.page_indices,
            metadata.swa_page_indices,
            metadata.seq_lens,
            metadata.distribution,
        ) = device_array(
            (cu_q_lens, cu_kv_lens, page_indices, swa_page_indices, seq_lens, distribution),
            sharding=(NamedSharding(self.mesh, P("data"))),
        )
        return metadata

    def get_eagle_forward_metadata(self, batch: ModelWorkerBatch):
        """Return the metadata for a forward pass."""
        # below code is for verify and draft extend phase
        metadata = FlashAttentionMetadata()
        indices = np.arange(0, len(batch.cache_loc), self.page_size)
        selected_cache_locs = batch.cache_loc[indices]
        page_indices = (selected_cache_locs // self.page_size).astype(np.int32)

        if batch.forward_mode == ForwardMode.TARGET_VERIFY:
            # convert custom_mask from bool to int32, because dma not support bool type
            if batch.spec_info_padded.custom_mask.dtype == jnp.bool:
                # FIXME(pc) rm this dtype convert
                logger.warning(
                    "batch.spec_info_padded.custom_mask type is  %s, it may make performance very low",
                    batch.spec_info_padded.custom_mask.dtype,
                )
                metadata.custom_mask = batch.spec_info_padded.custom_mask.astype(jnp.int32)
            else:
                metadata.custom_mask = batch.spec_info_padded.custom_mask
        else:
            metadata.custom_mask = None

        dp_size = batch.dp_size
        per_dp_bs = batch.per_dp_bs_size if dp_size > 1 else len(batch.seq_lens)
        if batch.forward_mode.is_target_verify():
            padded_batch_size = len(batch.seq_lens)
            extend_seq_lens = np.zeros(padded_batch_size, dtype=np.int32)
            extend_seq_lens[batch.logits_indices_selector] = batch.spec_info_padded.draft_token_num
        else:
            extend_seq_lens = batch.extend_seq_lens
        cu_q_lens = _per_dp_cumsum(extend_seq_lens, dp_size, per_dp_bs)

        seq_lens = np.copy(batch.seq_lens)

        if batch.forward_mode.is_target_verify():
            seq_lens += extend_seq_lens
            aligned_seq_lens = ((seq_lens + self.page_size - 1) // self.page_size) * self.page_size
            # Verify mask must be (a) DP-segmented per rank when dp>1 so each
            # rank's P("data") shard sees its own slots, and (b) padded so each
            # row width = aligned_seq_lens (= cu_kv_lens delta). The RPA kernel
            # always takes the cu_kv_lens-aligned path now (#1089 used to gate
            # on page_size>=256, which broke dp=1 + smaller pages — Mosaic
            # could not prove tiling(8) on the unaligned slice). dp=1 reduces
            # to a single rank chunk; dp>1 keeps the per-rank repack from
            # #1108 P1-7.
            if metadata.custom_mask is not None:
                q = batch.spec_info_padded.draft_token_num
                cm = np.asarray(jax.device_get(metadata.custom_mask))
                # Pin per-rank mask target from the pre-repacking mask capacity
                # (tree_mask_capacity from build_tree, already bucket-stable).
                cm_total = len(cm)
                per_rank_mask_target = ((cm_total // dp_size + 7) // 8) * 8 or 8
                # cm is DP-slot-ordered (build_tree got verified_seq_len = mwb.seq_lens-1
                # over total_bs). Per-slot cm length = q*(verified_seq_len[s]+q); for pad
                # slots verified_seq_len=-1 → q*(q-1).
                cm_kl = np.where(seq_lens > 0, seq_lens, q - 1).astype(np.int64)
                cm_off = np.concatenate([[0], np.cumsum(q * cm_kl)])
                row_width = aligned_seq_lens
                rank_chunks: list[np.ndarray] = []
                for r in range(dp_size):
                    parts = []
                    for j in range(per_dp_bs):
                        s = r * per_dp_bs + j
                        kla = int(row_width[s])
                        if seq_lens[s] > 0:
                            kl = int(seq_lens[s])
                            row = cm[cm_off[s] : cm_off[s] + q * kl].reshape(q, kl)
                            parts.append(np.pad(row, ((0, 0), (0, kla - kl))).reshape(-1))
                        elif kla > 0:
                            parts.append(np.zeros(q * kla, dtype=cm.dtype))
                    rank_chunks.append(
                        np.concatenate(parts) if parts else np.zeros(0, dtype=cm.dtype)
                    )
                max_len = max(max((len(c) for c in rank_chunks), default=0), per_rank_mask_target)
                packed = np.concatenate(
                    [np.pad(c, (0, max_len - len(c))) for c in rank_chunks]
                ).astype(np.int32)
                metadata.custom_mask = device_array(
                    packed,
                    sharding=NamedSharding(self.mesh, P("data")),
                )
        else:
            aligned_seq_lens = (
                (batch.seq_lens + self.page_size - 1) // self.page_size
            ) * self.page_size
        cu_kv_lens = _per_dp_cumsum(aligned_seq_lens, dp_size, per_dp_bs)

        if batch.forward_mode == ForwardMode.DRAFT_EXTEND:
            # Truncate each req's page list from allocate_len → seq_len, keeping
            # the DP-segmented layout from padding_for_decode (rank r's pages
            # at [r*per_dp_pg : ...]). page_indices (line 212) is already
            # cache_loc[::page_size]//page_size, so re-gather from it per-rank.
            allocate_lens = batch.spec_info_padded.allocate_lens
            if hasattr(allocate_lens, "device"):
                allocate_lens = jax.device_get(allocate_lens)
            allocate_lens = np.asarray(allocate_lens)
            sel = np.asarray(batch.logits_indices_selector)
            # allocate_lens here is global-flat (real_bs,) (cur_allocate_lens via
            # verify); sel is DP-slot indices, only used for rank_of/aligned_seq.
            assert len(allocate_lens) == len(sel), (len(allocate_lens), len(sel))
            full_pg = page_indices.shape[0]
            assert full_pg % dp_size == 0, (full_pg, dp_size)
            per_dp_pg = full_pg // dp_size
            alloc_pg = cdiv(allocate_lens.astype(np.int64), self.page_size)
            need_pg = (aligned_seq_lens[sel] // self.page_size).astype(np.int64)
            rank_of = (sel // per_dp_bs).astype(np.int64)
            new_pi = np.zeros(full_pg, dtype=np.int32)
            src_off = np.zeros(dp_size, dtype=np.int64)
            dst_off = np.zeros(dp_size, dtype=np.int64)
            for k in range(len(sel)):
                r = int(rank_of[k])
                n = int(need_pg[k])
                s = r * per_dp_pg + src_off[r]
                d = r * per_dp_pg + dst_off[r]
                new_pi[d : d + n] = page_indices[s : s + n]
                src_off[r] += int(alloc_pg[k])
                dst_off[r] += n
            page_indices = new_pi

        seq_2d = np.asarray(batch.seq_lens).reshape(dp_size, per_dp_bs)
        local_n = np.sum(seq_2d > 0, axis=1, dtype=np.int32)
        distribution = np.column_stack([np.zeros_like(local_n), local_n, local_n]).ravel()
        page_indices = np.array(page_indices)
        seq_lens = np.array(seq_lens)
        (
            metadata.cu_q_lens,
            metadata.cu_kv_lens,
            metadata.page_indices,
            metadata.seq_lens,
            metadata.distribution,
        ) = device_array(
            (cu_q_lens, cu_kv_lens, page_indices, seq_lens, distribution),
            sharding=(NamedSharding(self.mesh, P("data"))),
        )
        # Hybrid SWA targets need swa_page_indices for TARGET_VERIFY too,
        # otherwise SWA layers index the swa sub-pool with full-pool page ids.
        swa_mapping = getattr(self, "swa_index_mapping", None)
        if swa_mapping is not None:
            full_loc = (page_indices.astype(np.int64) * self.page_size).astype(np.int32)
            if isinstance(swa_mapping, list):
                full_2d = full_loc.reshape(dp_size, -1)
                swa_2d = np.empty_like(full_2d)
                for r in range(dp_size):
                    swa_2d[r] = np.asarray(swa_mapping[r])[full_2d[r]]
                swa_loc = swa_2d.ravel()
            else:
                swa_loc = np.asarray(swa_mapping)[full_loc]
            metadata.swa_page_indices = device_array(
                (swa_loc // self.page_size).astype(np.int32),
                sharding=NamedSharding(self.mesh, P("data")),
            )
        return metadata

    def get_eagle_multi_step_metadata(self, batch: ModelWorkerBatch):

        indices = np.arange(0, len(batch.cache_loc), self.page_size)
        # NOTE: Use original_selected_cache_locs as the source of truth for all steps
        # to avoid the bug where selected_cache_locs is overwritten by truncated data in loops.
        original_selected_cache_locs = batch.cache_loc[indices]
        assert batch.forward_mode is ForwardMode.DECODE

        page_indices = []
        cu_kv_lens = []
        seq_lens = np.copy(batch.seq_lens)

        # Vectorized preparation. selector maps global-flat req k → DP-padded
        # slot s_k; cache_loc/page_indices are laid out in valid-slot order
        # (== selector order), so the per-req gather below stays aligned.
        sel = np.asarray(batch.logits_indices_selector)
        current_seq_lens = np.asarray(batch.seq_lens)[sel]
        allocate_lens = np.asarray(batch.spec_info_padded.allocate_lens)[sel]

        draft_allocs = allocate_lens - current_seq_lens

        alloc_tokens = current_seq_lens + draft_allocs
        alloc_pages = cdiv(alloc_tokens, self.page_size)

        full_size = len(original_selected_cache_locs)
        dp_size = batch.dp_size
        per_dp_bs = batch.per_dp_bs_size if dp_size > 1 else len(batch.seq_lens)
        assert full_size % dp_size == 0, (full_size, dp_size)
        # cache_loc is DP-segmented (padding_for_decode); src_starts must point
        # into rank r's section [r*per_dp_src_pages : ...], and result_locs
        # must be written DP-segmented so the P("data") shard gives each rank
        # its own draft page_indices (otherwise rank>0 reads page 0 → accept~1).
        per_dp_src_pages = full_size // dp_size
        TARGET_PADDING = 16384
        assert TARGET_PADDING % dp_size == 0
        per_dp_dst_pages = TARGET_PADDING // dp_size
        rank_of_req = (sel // per_dp_bs).astype(np.int64)

        def _dp_starts(pages, per_dp_base):
            starts = np.zeros(len(pages), dtype=np.int64)
            for r in range(dp_size):
                m = rank_of_req == r
                if not np.any(m):
                    continue
                c = np.cumsum(pages[m])
                starts[m] = r * per_dp_base + np.concatenate(([0], c[:-1]))
            return starts

        src_starts = _dp_starts(alloc_pages, per_dp_src_pages)
        seq_lens_list = []
        valid_slot = np.asarray(batch.seq_lens) > 0
        for speculative_step_id in range(batch.speculative_num_steps):
            seq_lens = np.where(valid_slot, batch.seq_lens + speculative_step_id, 0)
            seq_lens_list.append(seq_lens)
            aligned_seq_lens = ((seq_lens + self.page_size - 1) // self.page_size) * self.page_size
            cu_kv_lens.append(_per_dp_cumsum(aligned_seq_lens, dp_size, per_dp_bs))

            # Vectorized calculation of spec_pages
            step_spec_tokens = (
                current_seq_lens + (speculative_step_id) * batch.speculative_eagle_topk
            )
            step_spec_pages = cdiv(step_spec_tokens, self.page_size)

            total_spec_pages = int(np.sum(step_spec_pages))
            dst_starts = _dp_starts(step_spec_pages, per_dp_dst_pages)
            flat_dst_cum = np.concatenate(([0], np.cumsum(step_spec_pages)[:-1]))

            repeats = step_spec_pages
            local_off = np.arange(total_spec_pages) - np.repeat(flat_dst_cum, repeats)
            gather_indices = np.repeat(src_starts, repeats) + local_off
            write_indices = np.repeat(dst_starts, repeats) + local_off
            gathered_locs = original_selected_cache_locs[gather_indices]

            result_locs = np.zeros(TARGET_PADDING, dtype=original_selected_cache_locs.dtype)
            result_locs[write_indices] = gathered_locs
            page_indices.append((result_locs // self.page_size).astype(np.int32))

        if batch.spec_algorithm.is_none():
            raise RuntimeError("should not reach here")
        assert isinstance(batch.spec_info_padded, EagleDraftInput)
        topk = batch.speculative_eagle_topk
        cu_q_lens = np.tile(np.arange(0, per_dp_bs * topk + 1, topk, dtype=np.int32), dp_size)
        seq_2d = np.asarray(batch.seq_lens).reshape(dp_size, per_dp_bs)
        local_n = np.sum(seq_2d > 0, axis=1, dtype=np.int32)
        distribution = np.column_stack(
            [np.zeros_like(local_n), np.zeros_like(local_n), local_n]
        ).ravel()
        metadata = []
        for i in range(batch.speculative_num_steps):
            metadata_tmp = FlashAttentionMetadata()
            (
                metadata_tmp.cu_q_lens,
                metadata_tmp.cu_kv_lens,
                metadata_tmp.page_indices,
                metadata_tmp.seq_lens,
                metadata_tmp.distribution,
            ) = device_array(
                (
                    cu_q_lens,
                    cu_kv_lens[i],
                    page_indices[i],
                    seq_lens_list[i],
                    distribution,
                ),
                sharding=(NamedSharding(self.mesh, P("data"))),
            )
            metadata.append(metadata_tmp)
        return metadata

    def tree_flatten(self):
        children = (self.forward_metadata,)
        aux_data = {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "page_size": self.page_size,
            "kv_partition_axis": self.kv_partition_axis,
            "attention_data_partition_axis": self.attention_data_partition_axis,
            "mesh": self.mesh,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(
            aux_data["num_heads"],
            aux_data["num_kv_heads"],
            aux_data["head_dim"],
            aux_data["page_size"],
            kv_partition_axis=aux_data.get("kv_partition_axis", "tensor"),
            attention_data_partition_axis=aux_data.get("attention_data_partition_axis", "data"),
            mesh=aux_data.get("mesh"),
        )

        obj.forward_metadata = children[0]

        return obj

    @named_scope
    def __call__(
        self,
        q: jax.Array,  # [total_tokens, num_heads, head_dim]
        k: jax.Array,  # [total_tokens, num_heads, head_dim]
        v: jax.Array,  # [total_tokens, num_heads, head_dim]
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        causal: int = 1,
        attention_sink: jax.Array = None,
        index_q: jax.Array | None = None,
        index_k: jax.Array | None = None,
        msa_topk: int = 0,
        msa_local_blocks: int = 1,
    ):
        """
        Args:
            q, k, v: Input tensors of shape [total_tokens, num_heads, head_dim]
            forward_batch: ForwardBatch object containing seq_lens and batch_size
            attention_mask: Optional attention mask
            is_causal: Whether to apply causal masking
        Returns:
            Output tensor of shape [total_tokens, hidden_size]
        """
        if forward_batch is not None and token_to_kv_pool is not None:
            kv_cache_fused = self._get_fused_kv_cache(
                forward_batch, token_to_kv_pool, layer.layer_id
            )
        else:
            kv_cache_fused = jnp.zeros((0, self.num_kv_heads * 2, self.head_dim), dtype=q.dtype)
        scale = (
            1.0 / jnp.sqrt(layer.head_dim)
            if (layer is None or layer.scaling is None)
            else layer.scaling
        )

        if self.forward_metadata.custom_mask is not None:
            causal = 0
        # Select page indices and remap to SWA pool if KV cache supports it
        page_indices_arg = self.forward_metadata.page_indices
        is_swa_layer = layer.sliding_window_size is not None and layer.sliding_window_size > 0
        if is_swa_layer and self.forward_metadata.swa_page_indices is not None:
            page_indices_arg = self.forward_metadata.swa_page_indices
        elif hasattr(token_to_kv_pool, "remap_cache_loc") and self.page_size == 1:
            page_indices_arg = token_to_kv_pool.remap_cache_loc(page_indices_arg, layer.layer_id)

        in_specs = (
            P(self.attention_data_partition_axis, self.kv_partition_axis),  # queries
            P(self.attention_data_partition_axis, self.kv_partition_axis),  # keys (new tokens)
            P(self.attention_data_partition_axis, self.kv_partition_axis),  # values (new tokens)
            P(
                self.attention_data_partition_axis, None, self.kv_partition_axis, None, None
            ),  # kv_cache_fused (head interleaved)
            P(self.attention_data_partition_axis),  # kv_lens
            P(self.attention_data_partition_axis),  # page_indices
            P(self.attention_data_partition_axis),  # cu_q_lens
            P(self.attention_data_partition_axis),  # cu_kv_lens
            P(self.attention_data_partition_axis),  # distribution
            (
                P(self.attention_data_partition_axis)
                if self.forward_metadata.custom_mask is not None
                else P()
            ),  # custom_mask: DP-segmented per-rank (cu_seq_mask_lens is rank-local)
            (
                P(self.kv_partition_axis) if attention_sink is not None else P()
            ),  # attention sink: (num_q_heads,), sharded by heads
        )

        out_specs = (
            P(self.attention_data_partition_axis, self.kv_partition_axis),  # attention output
            P(
                self.attention_data_partition_axis, None, self.kv_partition_axis, None, None
            ),  # updated kv_cache_fused (head interleaved) - 3D: [total_tokens, num_kv_heads*2, head_dim]
        )

        mask_aligned_to_cu_kv = (
            self.forward_metadata.custom_mask is not None
            and forward_batch.forward_mode.is_target_verify()
        )

        def _ragged_paged_attention_with_fused_kv(*args):
            queries, keys, values, kv_cache_fused = args[:4]
            other_args = args[4:]

            # Call fused KV kernel with head interleaving
            result, updated_kv_cache_fused = ragged_paged_attention_v3(
                queries,
                keys,
                values,
                kv_cache_fused,
                *other_args,
                causal=causal,
                sm_scale=scale,
                sliding_window=layer.sliding_window_size,
                soft_cap=layer.logit_cap,
                xai_temperature_len=(
                    layer.xai_temperature_len if layer.xai_temperature_len > 0 else None
                ),
                mask_aligned_to_cu_kv=mask_aligned_to_cu_kv,
            )

            return result, updated_kv_cache_fused

        rpa_args = (
            q.reshape(q.shape[0], -1, getattr(layer, "head_dim", self.head_dim)),
            k.reshape(k.shape[0], -1, getattr(layer, "head_dim", self.head_dim)),
            v.reshape(v.shape[0], -1, getattr(layer, "head_dim", self.head_dim)),
            kv_cache_fused,
            self.forward_metadata.seq_lens,
            page_indices_arg,
            self.forward_metadata.cu_q_lens,
            self.forward_metadata.cu_kv_lens,
            self.forward_metadata.distribution,
            self.forward_metadata.custom_mask,
            attention_sink,
        )

        _MSA_POOLED = os.environ.get("SGLANG_MSA_POOLED", "0") == "1"
        is_msa = index_k is not None and hasattr(token_to_kv_pool, "get_index_k_buffer")
        if not is_msa:
            attn_output, updated_kv_cache_fused = jax.shard_map(
                _ragged_paged_attention_with_fused_kv,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            )(*rpa_args)
            return attn_output.reshape(q.shape[0], -1), updated_kv_cache_fused

        # --- MSA layer: per-DP-local index_k write + (decode) topk page selection ---
        ik_buf = token_to_kv_pool.get_index_k_buffer(layer.layer_id)
        ik_pooled = token_to_kv_pool.get_index_k_pooled(layer.layer_id)
        is_decode = forward_batch.forward_mode.is_decode()
        page_size = self.page_size
        data = self.attention_data_partition_axis
        msa_in_specs = in_specs + (
            P(data, None, None, None),  # ik_buf [pages, ps, 1, d_idx]
            P(data, None, None),  # ik_pooled [pages, 1, d_idx]
            P(data, None, None),  # index_q [bs, H_idx, d_idx]
            P(data, None, None),  # index_k [bs, 1, d_idx]
            P(data),  # out_cache_loc [bs]
        )
        msa_out_specs = out_specs + (P(data, None, None, None), P(data, None, None))

        def _msa_inner(*args):
            (
                queries,
                keys,
                values,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                distribution,
                custom_mask,
                attn_sink,
                ik_buf_l,
                ikp_l,
                iq_l,
                ik_new_l,
                out_loc_l,
            ) = args
            q_page = out_loc_l // page_size
            slot_in_page = out_loc_l % page_size
            ik_cast = ik_new_l[:, 0].astype(ik_buf_l.dtype)
            # 1. write new index_k into ik_buf (per-DP local page addressing).
            # Prefill: sort the flat index so XLA emits a sorted+unique scatter
            # (v7x: 1065→367us/layer, ×57=37ms). Decode bs is small and idx is
            # already scattered across pages — plain .at[].set is faster there.
            if is_decode:
                ik_buf_upd = ik_buf_l.at[q_page, slot_in_page, 0].set(ik_cast)
            else:
                flat_idx = q_page * page_size + slot_in_page
                order = jnp.argsort(flat_idx)
                d_idx = ik_buf_l.shape[-1]
                _dn = jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(1,),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                )
                ik_buf_upd = jax.lax.scatter(
                    ik_buf_l.reshape(-1, d_idx),
                    flat_idx[order][:, None],
                    ik_cast[order],
                    _dn,
                    indices_are_sorted=True,
                    unique_indices=True,
                    mode="promise_in_bounds",
                ).reshape(ik_buf_l.shape)
            # 1b. update per-block element-wise-max pool. slot==0 means this token
            #     opens a fresh page (possibly reused from a freed req) so the
            #     stale pooled value must be discarded, not max-ed against.
            if is_decode:
                ikp_prev = ikp_l[q_page, 0]
                ikp_new = jnp.where(
                    (slot_in_page == 0)[:, None], ik_cast, jnp.maximum(ikp_prev, ik_cast)
                )
                ikp_upd = ikp_l.at[q_page, 0].set(ikp_new)
            else:
                neg_inf = jnp.finfo(ikp_l.dtype).min
                ikp_upd = ikp_l.at[q_page, 0].set(neg_inf).at[q_page, 0].max(ik_cast)
            bs_l = kv_lens.shape[0]
            assert page_indices.shape[0] % bs_l == 0
            pages_per_seq = page_indices.shape[0] // bs_l
            # Skip MSA topk when the (static) page budget cannot exceed topk —
            # selecting topk from ≤topk blocks is the identity (≡ dense). This
            # avoids the O(pages_per_seq) ik gather on short-ctx deployments.
            if is_decode and msa_topk > 0 and pages_per_seq > msa_topk:
                # page_indices is cumsum-packed ragged (schedule_batch._merge_cache_loc),
                # NOT [bs, P] rectangular. reshape(bs, P) misaligns req[k>0] and reads
                # stale cache_loc_host_buf entries (NOT re-zeroed, see schedule_batch
                # safety note — that invariant assumes cu_kv_lens-bounded reads).
                cu_pages = cu_kv_lens[:bs_l] // page_size
                col = jnp.arange(pages_per_seq, dtype=jnp.int32)
                gidx = jnp.minimum(cu_pages[:, None] + col[None, :], page_indices.shape[0] - 1)
                pi_2d = page_indices[gidx]
                n_blocks_v = (kv_lens + page_size - 1) // page_size
                if _MSA_POOLED:
                    # P0a approximate: gather pooled ik (O(n_blocks))
                    ikp_seq = ikp_upd[pi_2d][:, :, 0]
                    bscores = jnp.einsum(
                        "bhd,bnd->bhn", iq_l.astype(jnp.float32), ikp_seq.astype(jnp.float32)
                    ).max(1)
                else:
                    # v2 exact: page-level einsum, bf16 in / f32 accum (no f32
                    # materialization of ik_hist). The last (q_block) block's
                    # zero-padded tail contributes score=0 to its max, but that
                    # block is force-selected as local anyway.
                    ik_pages = ik_buf_upd[pi_2d][:, :, :, 0]  # [bs, n_pages, ps, d] bf16
                    s_tok = jnp.einsum(
                        "bhd,bnpd->bhnp", iq_l, ik_pages, preferred_element_type=jnp.float32
                    )
                    bscores = s_tok.max(-1).max(1)
                block_mask = jnp.arange(pages_per_seq)[None, :] < n_blocks_v[:, None]
                bscores = jnp.where(block_mask, bscores, -jnp.inf)
                q_block = (kv_lens - 1) // page_size
                ar = jnp.arange(bs_l)
                for j in range(msa_local_blocks):
                    bscores = bscores.at[ar, jnp.maximum(q_block - j, 0)].set(jnp.inf)
                _, topk_idx = jax.lax.top_k(bscores, msa_topk)
                n_valid = jnp.minimum(n_blocks_v, msa_topk)
                # 4. sort ascending; pad invalid slots with pages_per_seq so q_block
                #    (largest valid) lands at [n_valid-1] — RPA writes new K/V there.
                pad_val = jnp.int32(pages_per_seq)
                slot = jnp.arange(msa_topk)[None, :]
                topk_idx = jnp.where(slot < n_valid[:, None], topk_idx, pad_val)
                topk_idx = jnp.sort(topk_idx, axis=-1)
                # 5. logical block -> physical page; build MSA metadata
                msa_pi = jnp.take_along_axis(
                    jnp.pad(pi_2d, ((0, 0), (0, 1))), topk_idx, axis=-1
                ).reshape(-1)
                msa_kvl = (n_valid - 1) * page_size + (kv_lens - 1) % page_size + 1
                msa_cu = jnp.arange(bs_l + 1, dtype=jnp.int32) * (msa_topk * page_size)
                page_indices, kv_lens, cu_kv_lens = msa_pi, msa_kvl, msa_cu
            result, kv_upd = ragged_paged_attention_v3(
                queries,
                keys,
                values,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                distribution,
                custom_mask,
                attn_sink,
                causal=causal,
                sm_scale=scale,
                sliding_window=layer.sliding_window_size,
                soft_cap=layer.logit_cap,
                xai_temperature_len=(
                    layer.xai_temperature_len if layer.xai_temperature_len > 0 else None
                ),
                mask_aligned_to_cu_kv=mask_aligned_to_cu_kv,
            )
            return result, kv_upd, ik_buf_upd, ikp_upd

        attn_output, updated_kv_cache_fused, updated_ik, updated_ikp = jax.shard_map(
            _msa_inner, in_specs=msa_in_specs, out_specs=msa_out_specs, check_vma=False
        )(*rpa_args, ik_buf, ik_pooled, index_q, index_k, forward_batch.out_cache_loc)

        return (
            attn_output.reshape(q.shape[0], -1),
            updated_kv_cache_fused,
            (updated_ik, updated_ikp),
        )

    def _get_fused_kv_cache(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        layer_id: int,
    ) -> jax.Array:
        return token_to_kv_pool.get_fused_kv_buffer(layer_id)

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        num_page_per_req = cdiv(max_context_len, page_size)
        res = 1024 * 1024 // 2 // num_page_per_req // 4
        assert (
            res > 0
        ), f"max running requests: {res} must larger than 0, please increase page size or decrease max context length"
        return res
