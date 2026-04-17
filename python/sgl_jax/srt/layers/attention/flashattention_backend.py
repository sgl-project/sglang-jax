import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    ragged_paged_attention as ragged_paged_attention,
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


@register_pytree_node_class
@dataclass
class FlashAttentionMetadata:
    """Metadata to be init once in the model forward pass,
    each layer's forward pass can reuse the metadata.

    For each init metadata function, we will try set up them in below order
    """

    num_seqs: jax.Array = None
    cu_q_lens: jax.Array = None
    page_indices: jax.Array = None
    seq_lens: jax.Array = None
    distribution: jax.Array = None

    def tree_flatten(self):
        children = (
            self.num_seqs,
            self.cu_q_lens,
            self.page_indices,
            self.seq_lens,
            self.distribution,
        )

        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)

        obj.num_seqs = children[0]
        obj.cu_q_lens = children[1]
        obj.page_indices = children[2]
        obj.seq_lens = children[3]
        obj.distribution = children[4]

        return obj


@dataclass
class FlashAttention(AttentionBackend):
    """Native Attention layer for variable-length sequences using ForwardBatch."""

    def __init__(
        self,
        num_attn_heads,
        num_kv_heads,
        head_dim,
        vmem_limit_bytes: int = 64 * (1 << 20),  # 64MB
        page_size: int = 1,
        kv_partition_axis: str = "tensor",
        mesh: jax.sharding.Mesh = None,
        max_context_len: int = 131072,
    ):
        self.vmem_limit_bytes = vmem_limit_bytes
        self.num_heads = num_attn_heads
        if num_kv_heads is not None:
            self.num_kv_heads = num_kv_heads
        else:
            self.num_kv_heads = num_attn_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.pages_per_seq = cdiv(max_context_len, page_size)
        self.kv_partition_axis = kv_partition_axis
        self.forward_metadata = nnx.data(FlashAttentionMetadata())
        self.mesh = mesh

    def get_forward_metadata(
        self,
        batch: ModelWorkerBatch,
    ):
        """Return the metadata for a forward pass."""
        metadata = FlashAttentionMetadata()

        max_num_seqs = len(batch.seq_lens)
        pages_per_seq = self.pages_per_seq

        # Build page_indices with uniform stride: [max_num_seqs * pages_per_seq]
        # Each sequence gets exactly pages_per_seq slots, padded with 0.
        page_indices = np.zeros(max_num_seqs * pages_per_seq, dtype=np.int32)
        offset = 0
        for i in range(max_num_seqs):
            seq_len = batch.seq_lens[i]
            if seq_len <= 0:
                continue
            num_pages = cdiv(seq_len, self.page_size)
            aligned_len = num_pages * self.page_size
            seq_cache_locs = batch.cache_loc[offset : offset + aligned_len]
            seq_page_indices = (seq_cache_locs[:: self.page_size] // self.page_size).astype(
                np.int32
            )
            dst_start = i * pages_per_seq
            page_indices[dst_start : dst_start + num_pages] = seq_page_indices[:num_pages]
            offset += aligned_len

        if batch.forward_mode == ForwardMode.EXTEND:
            cu_q_lens = np.concatenate(
                [
                    np.array([0], dtype=np.int32),
                    np.cumsum(batch.extend_seq_lens, dtype=np.int32),
                ]
            )
        elif batch.forward_mode == ForwardMode.DECODE:
            cu_q_lens = np.concatenate(
                [
                    np.array([0], dtype=np.int32),
                    np.cumsum(np.ones(len(batch.seq_lens), dtype=np.int32)),
                ]
            )
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        seq_lens = np.copy(batch.seq_lens)

        num_seqs = np.sum(batch.seq_lens > 0, dtype=np.int32).reshape(
            1,
        )

        # Construct distribution: [decode_end, prefill_end, mixed_end]
        # sequences[0:i] are decode-only, sequences[i:j] are prefill-only,
        # sequences[j:k] are mixed. Extend sequences go through the MIXED
        # bucket until bucketed prefill padding is wired up.
        if batch.forward_mode == ForwardMode.DECODE:
            distribution = np.array(
                [num_seqs.item(), num_seqs.item(), num_seqs.item()], dtype=np.int32
            )
        elif batch.forward_mode == ForwardMode.EXTEND:
            distribution = np.array([0, num_seqs.item(), num_seqs.item()], dtype=np.int32)
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        (
            metadata.num_seqs,
            metadata.cu_q_lens,
            metadata.page_indices,
            metadata.seq_lens,
            metadata.distribution,
        ) = device_array(
            (num_seqs, cu_q_lens, page_indices, seq_lens, distribution),
            sharding=(NamedSharding(self.mesh, P()) if jax.process_count() == 1 else None),
        )
        return metadata

    def get_eagle_forward_metadata(self, batch: ModelWorkerBatch):
        """Return the metadata for a forward pass."""
        # below code is for verify and draft extend phase
        metadata = FlashAttentionMetadata()

        max_num_seqs = len(batch.seq_lens)
        pages_per_seq = self.pages_per_seq

        if batch.forward_mode.is_target_verify():
            padded_batch_size = len(batch.seq_lens)
            real_batch_size = batch.real_bs
            q_lens = np.array([batch.spec_info.draft_token_num] * real_batch_size, dtype=np.int32)
            extend_seq_lens = np.pad(q_lens, (0, padded_batch_size - real_batch_size))
        else:
            extend_seq_lens = batch.extend_seq_lens
        cu_q_lens = np.concatenate(
            [
                np.array([0], dtype=np.int32),
                np.cumsum(extend_seq_lens),
            ]
        )

        seq_lens = np.copy(batch.seq_lens)

        if batch.forward_mode.is_target_verify():
            seq_lens += extend_seq_lens

        if batch.forward_mode == ForwardMode.DRAFT_EXTEND:
            # Reconstruct page_indices with uniform stride for v3 kernel
            page_indices = np.zeros(max_num_seqs * pages_per_seq, dtype=np.int32)
            offset = 0
            allocate_lens = batch.spec_info.allocate_lens
            if hasattr(allocate_lens, "device"):
                allocate_lens = jax.device_get(allocate_lens)

            for i in range(batch.real_bs):
                alloc_len = (
                    (int(allocate_lens[i]) + self.page_size - 1) // self.page_size
                ) * self.page_size
                needed_pages = cdiv(int(seq_lens[i]), self.page_size)

                if needed_pages > 0:
                    req_cache_loc = batch.cache_loc[offset : offset + alloc_len]
                    indices = np.arange(needed_pages) * self.page_size
                    selected = req_cache_loc[indices]
                    dst_start = i * pages_per_seq
                    page_indices[dst_start : dst_start + needed_pages] = selected // self.page_size

                offset += alloc_len
        else:
            # Build page_indices with uniform stride from cache_loc
            page_indices = np.zeros(max_num_seqs * pages_per_seq, dtype=np.int32)
            offset = 0
            for i in range(max_num_seqs):
                s_len = seq_lens[i]
                if s_len <= 0:
                    continue
                num_pages = cdiv(s_len, self.page_size)
                aligned_len = num_pages * self.page_size
                seq_cache_locs = batch.cache_loc[offset : offset + aligned_len]
                seq_page_indices = (seq_cache_locs[:: self.page_size] // self.page_size).astype(
                    np.int32
                )
                dst_start = i * pages_per_seq
                page_indices[dst_start : dst_start + num_pages] = seq_page_indices[:num_pages]
                offset += aligned_len

        num_seqs = np.sum(batch.seq_lens > 0, dtype=np.int32).reshape(
            1,
        )

        # Route to MIXED bucket (always runs); PREFILL kernel requires prefill_size.
        distribution = np.array([0, num_seqs.item(), num_seqs.item()], dtype=np.int32)

        num_seqs = np.array(num_seqs)
        cu_q_lens = np.array(cu_q_lens)
        page_indices = np.array(page_indices)
        seq_lens = np.array(seq_lens)
        (
            metadata.num_seqs,
            metadata.cu_q_lens,
            metadata.page_indices,
            metadata.seq_lens,
            metadata.distribution,
        ) = device_array(
            (num_seqs, cu_q_lens, page_indices, seq_lens, distribution),
            sharding=(NamedSharding(self.mesh, P()) if jax.process_count() == 1 else None),
        )
        return metadata

    def get_eagle_multi_step_metadata(self, batch: ModelWorkerBatch):

        indices = np.arange(0, len(batch.cache_loc), self.page_size)
        # NOTE: Use original_selected_cache_locs as the source of truth for all steps
        # to avoid the bug where selected_cache_locs is overwritten by truncated data in loops.
        original_selected_cache_locs = batch.cache_loc[indices]
        assert batch.forward_mode is ForwardMode.DECODE

        max_num_seqs = len(batch.seq_lens)
        pages_per_seq = self.pages_per_seq

        page_indices = []
        seq_lens = np.copy(batch.seq_lens)

        # Vectorized preparation
        real_bs = batch.real_bs
        current_seq_lens = batch.seq_lens[:real_bs]
        allocate_lens = batch.spec_info.allocate_lens[:real_bs]

        draft_allocs = allocate_lens - current_seq_lens

        alloc_tokens = current_seq_lens + draft_allocs
        alloc_pages = cdiv(alloc_tokens, self.page_size)

        # src_starts (offset2) is constant across steps
        src_starts = np.concatenate(([0], np.cumsum(alloc_pages)[:-1]))

        seq_lens_list = []
        for speculative_step_id in range(batch.speculative_num_steps):
            seq_lens = batch.seq_lens + (speculative_step_id)
            seq_lens[batch.real_bs :] = 0
            seq_lens_list.append(seq_lens)

            # Vectorized calculation of spec_pages
            step_spec_tokens = (
                current_seq_lens + (speculative_step_id) * batch.speculative_eagle_topk
            )
            step_spec_pages = cdiv(step_spec_tokens, self.page_size)

            total_spec_pages = np.sum(step_spec_pages)
            dst_starts = np.concatenate(([0], np.cumsum(step_spec_pages)[:-1]))

            # Vectorized Gather
            repeats = step_spec_pages
            gather_indices = np.repeat(src_starts, repeats) + (
                np.arange(total_spec_pages) - np.repeat(dst_starts, repeats)
            )

            gathered_locs = original_selected_cache_locs[gather_indices]

            # Build page_indices with uniform stride for v3 kernel
            page_indices_cur_step = np.zeros(max_num_seqs * pages_per_seq, dtype=np.int32)
            ragged_offset = 0
            for i in range(real_bs):
                num_pages = int(step_spec_pages[i])
                if num_pages > 0:
                    ragged_pages = (
                        gathered_locs[ragged_offset : ragged_offset + num_pages]
                    ).astype(np.int32)
                    dst_start = i * pages_per_seq
                    page_indices_cur_step[dst_start : dst_start + num_pages] = ragged_pages
                    ragged_offset += num_pages

            page_indices.append(page_indices_cur_step)

        if batch.spec_algorithm.is_none():
            raise RuntimeError("should not reach here")
        else:
            assert isinstance(batch.spec_info, EagleDraftInput)
            # it is same across every step
            cu_q_lens = np.arange(
                0,
                len(batch.seq_lens) * batch.speculative_eagle_topk + 1,
                step=batch.speculative_eagle_topk,
                dtype=np.int32,
            )
        num_seqs = np.sum(batch.seq_lens > 0, dtype=np.int32).reshape(
            1,
        )

        distribution = np.array([0, 0, num_seqs.item()], dtype=np.int32)
        metadata = []
        for i in range(batch.speculative_num_steps):
            metadata_tmp = FlashAttentionMetadata()
            (
                metadata_tmp.num_seqs,
                metadata_tmp.cu_q_lens,
                metadata_tmp.page_indices,
                metadata_tmp.seq_lens,
                metadata_tmp.distribution,
            ) = device_array(
                (
                    num_seqs,
                    cu_q_lens,
                    page_indices[i],
                    seq_lens_list[i],
                    distribution,
                ),
                sharding=(NamedSharding(self.mesh, P()) if jax.process_count() == 1 else None),
            )
            metadata.append(metadata_tmp)
        return metadata

    def tree_flatten(self):
        children = (self.forward_metadata,)
        aux_data = {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "vmem_limit_bytes": self.vmem_limit_bytes,
            "head_dim": self.head_dim,
            "page_size": self.page_size,
            "pages_per_seq": self.pages_per_seq,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(
            aux_data["num_heads"],
            aux_data["num_kv_heads"],
            aux_data["head_dim"],
            aux_data["vmem_limit_bytes"],
            aux_data["page_size"],
        )
        obj.pages_per_seq = aux_data[
            "pages_per_seq"
        ]  # override the value computed from max_context_len
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

        # Prepare fused KV cache as 4D paged format (keep heads axis unsplit for sharding):
        # [num_pages, page_size, num_kv_heads_x2, head_dim]
        total_tokens = kv_cache_fused.shape[0]
        num_pages = total_tokens // self.page_size
        padded_head_dim = (self.head_dim + 127) // 128 * 128
        kv_packing = 32 // (jnp.dtype(kv_cache_fused.dtype).itemsize * 8)
        num_kv_heads_x2 = kv_cache_fused.shape[1]
        kv_cache_fused_paged = kv_cache_fused.reshape(
            num_pages, self.page_size, num_kv_heads_x2, padded_head_dim
        )

        xai_temp_len = getattr(layer, "xai_temperature_len", None)
        if xai_temp_len is not None and xai_temp_len <= 0:
            raise AssertionError(
                f"xai_temperature_len must be a positive integer, got {xai_temp_len}"
            )

        # Select page indices and remap to SWA pool if KV cache supports it
        page_indices_arg = self.forward_metadata.page_indices
        if hasattr(token_to_kv_pool, "remap_cache_loc") and self.page_size == 1:
            page_indices_arg = token_to_kv_pool.remap_cache_loc(page_indices_arg, layer.layer_id)

        in_specs = (
            P(None, self.kv_partition_axis),  # queries
            P(None, self.kv_partition_axis),  # keys (new tokens)
            P(None, self.kv_partition_axis),  # values (new tokens)
            P(None, None, self.kv_partition_axis, None),  # kv_cache 4D
            P(),  # kv_lens
            P(),  # page_indices
            P(),  # cu_q_lens
            P(),  # distribution
        )
        out_specs = (
            P(None, self.kv_partition_axis),  # attention output
            P(None, None, self.kv_partition_axis, None),  # updated kv_cache 4D
        )

        def _ragged_paged_attention_with_fused_kv(*args):
            queries, keys, values, kv_cache_4d = args[:4]
            other_args = args[4:]

            n_pages, pg_sz, local_kv_heads_x2, hdim = kv_cache_4d.shape

            # Pool and v3 both use [K0,V0,K1,V1,...] interleaving — just reshape to 5D
            kv_cache_5d = kv_cache_4d.reshape(
                n_pages,
                pg_sz,
                local_kv_heads_x2 // kv_packing,
                kv_packing,
                hdim,
            )

            result, updated_kv_cache_5d = ragged_paged_attention(
                queries,
                keys,
                values,
                kv_cache_5d,
                *other_args,
                use_causal_mask=causal,
                sm_scale=scale,
                sliding_window=layer.sliding_window_size,
                soft_cap=layer.logit_cap,
                xai_temperature_len=xai_temp_len,
                prefill_size=queries.shape[0],
            )

            updated_kv_cache_4d = updated_kv_cache_5d.reshape(
                n_pages, pg_sz, local_kv_heads_x2, hdim
            )
            return result, updated_kv_cache_4d

        (
            attn_output,
            updated_kv_cache_fused,
        ) = jax.shard_map(
            _ragged_paged_attention_with_fused_kv,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(
            q.reshape(q.shape[0], -1, self.head_dim),
            k.reshape(k.shape[0], -1, self.head_dim),
            v.reshape(v.shape[0], -1, self.head_dim),
            kv_cache_fused_paged,
            self.forward_metadata.seq_lens,
            page_indices_arg,
            self.forward_metadata.cu_q_lens,
            self.forward_metadata.distribution,
        )
        # Reshape 4D back to 3D for pool storage
        updated_kv_cache_fused = updated_kv_cache_fused.reshape(
            total_tokens, num_kv_heads_x2, padded_head_dim
        )
        # jax.debug.print("updated_kv_cache_fused: {updated_kv_cache_fused}", updated_kv_cache_fused=updated_kv_cache_fused)
        # jax.debug.print("updated_kv_cache_fused shape: {s}", s=updated_kv_cache_fused.shape)
        # jax.debug.print(
        #     "kv_cache nonzero count: {c}, min: {mn}, max: {mx}, sum: {s}",
        #     c=jnp.count_nonzero(updated_kv_cache_fused),
        #     mn=jnp.min(updated_kv_cache_fused),
        #     mx=jnp.max(updated_kv_cache_fused),
        #     s=jnp.sum(updated_kv_cache_fused),
        # )

        return (
            attn_output.reshape(q.shape[0], -1),
            updated_kv_cache_fused,
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
