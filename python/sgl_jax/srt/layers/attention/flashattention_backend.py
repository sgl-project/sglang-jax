import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.radix_attention import AttentionType, RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils import cdiv
from sgl_jax.srt.utils.jax_utils import device_array, is_tpu_runtime

# Conditional imports based on runtime
if is_tpu_runtime():
    from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention import (
        ragged_paged_attention,
    )
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput
else:
    from flash_attn_jax import flash_mha, flash_mha_varlen

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
    cu_kv_lens: jax.Array = None
    page_indices: jax.Array = None
    seq_lens: jax.Array = None
    distribution: jax.Array = None
    custom_mask: jax.Array = None

    def tree_flatten(self):
        children = (
            self.num_seqs,
            self.cu_q_lens,
            self.cu_kv_lens,
            self.page_indices,
            self.seq_lens,
            self.distribution,
            self.custom_mask,
        )

        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)

        obj.num_seqs = children[0]
        obj.cu_q_lens = children[1]
        obj.cu_kv_lens = children[2]
        obj.page_indices = children[3]
        obj.seq_lens = children[4]
        obj.distribution = children[5]
        obj.custom_mask = children[6]

        return obj


@dataclass
class FlashAttention(AttentionBackend):
    """Flash Attention layer for variable-length sequences using ForwardBatch.

    Uses ragged_paged_attention on TPU and flash_attn_jax on GPU.
    """

    def __init__(
        self,
        num_attn_heads,
        num_kv_heads,
        head_dim,
        vmem_limit_bytes: int = 64 * (1 << 20),  # 64MB
        page_size: int = 1,
        kv_partition_axis: str = "tensor",
        mesh: jax.sharding.Mesh = None,
    ):
        self.vmem_limit_bytes = vmem_limit_bytes
        self.num_heads = num_attn_heads
        if num_kv_heads is not None:
            self.num_kv_heads = num_kv_heads
        else:
            self.num_kv_heads = num_attn_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.kv_partition_axis = kv_partition_axis
        self.forward_metadata = nnx.data(FlashAttentionMetadata())
        self.mesh = mesh
        self.kv_sharding = NamedSharding(self.mesh, P(None, "tensor", None))

    def get_forward_metadata(
        self,
        batch: ModelWorkerBatch,
    ):
        # GPU backend doesn't need forward metadata
        if not is_tpu_runtime():
            return None

        # NOTE: Removed undefined is_eagle, speculative_step_id, topk variables
        # if is_eagle:
        #     return self.get_eagle_forward_metadata(
        #         batch, speculative_step_id=speculative_step_id, topk=topk
        #     )

        """Return the metadata for a forward pass."""
        metadata = FlashAttentionMetadata()

        indices = np.arange(0, len(batch.cache_loc), self.page_size)
        selected_cache_locs = batch.cache_loc[indices]
        page_indices = (selected_cache_locs // self.page_size).astype(np.int32)

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

        aligned_seq_lens = (
            (batch.seq_lens + self.page_size - 1) // self.page_size
        ) * self.page_size
        cu_kv_lens = np.concatenate(
            [
                np.array([0], dtype=np.int32),
                np.cumsum(aligned_seq_lens),
            ]
        )

        num_seqs = np.sum(batch.seq_lens > 0, dtype=np.int32).reshape(
            1,
        )

        # Construct distribution for V2 kernel: [decode_end, prefill_end, mixed_end]
        if batch.forward_mode == ForwardMode.DECODE:
            # All sequences are decode/mixed mode
            distribution = np.array([0, 0, num_seqs.item()], dtype=np.int32)
        elif batch.forward_mode == ForwardMode.EXTEND:
            # All sequences are prefill mode
            distribution = np.array([0, num_seqs.item(), num_seqs.item()], dtype=np.int32)
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        (
            metadata.num_seqs,
            metadata.cu_q_lens,
            metadata.cu_kv_lens,
            metadata.page_indices,
            metadata.seq_lens,
            metadata.distribution,
        ) = device_array(
            (num_seqs, cu_q_lens, cu_kv_lens, page_indices, seq_lens, distribution),
            sharding=(NamedSharding(self.mesh, P()) if jax.process_count() == 1 else None),
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
            if batch.spec_info.custom_mask.dtype == jnp.bool:
                # FIXME(pc) rm this dtype convert
                logger.warning(
                    "batch.spec_info.custom_mask type is  %s, it may make performance very low",
                    batch.spec_info.custom_mask.dtype,
                )
                metadata.custom_mask = batch.spec_info.custom_mask.astype(jnp.int32)
            else:
                metadata.custom_mask = batch.spec_info.custom_mask
        else:
            metadata.custom_mask = None

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
            aligned_seq_lens = ((seq_lens + self.page_size - 1) // self.page_size) * self.page_size
        else:
            aligned_seq_lens = (
                (batch.seq_lens + self.page_size - 1) // self.page_size
            ) * self.page_size
        cu_kv_lens = np.concatenate(
            [
                np.array([0], dtype=np.int32),
                np.cumsum(aligned_seq_lens),
            ]
        )

        if batch.forward_mode == ForwardMode.DRAFT_EXTEND:
            # Reconstruct page_indices properly respecting ragged allocation
            page_indices_list = []
            offset = 0
            allocate_lens = batch.spec_info.allocate_lens
            # Ensure it's accessible as array
            if hasattr(allocate_lens, "device"):
                allocate_lens = jax.device_get(allocate_lens)

            num_pages_per_seq = aligned_seq_lens // self.page_size

            for i in range(batch.real_bs):
                alloc_len = (
                    (int(allocate_lens[i]) + self.page_size - 1) // self.page_size
                ) * self.page_size
                needed_pages = int(num_pages_per_seq[i])

                if needed_pages > 0:
                    # Get the slice of cache_loc for this request
                    # We assume batch.cache_loc is ordered and packed according to allocate_lens
                    req_cache_loc = batch.cache_loc[offset : offset + alloc_len]

                    # Select the first token of each page
                    # The tokens are at indices 0, page_size, 2*page_size...
                    # We need `needed_pages` entries.

                    indices = np.arange(needed_pages) * self.page_size
                    selected = req_cache_loc[indices]
                    page_indices_list.extend(selected // self.page_size)

                offset += alloc_len

            page_indices = np.pad(
                np.array(page_indices_list, dtype=np.int32),
                (0, page_indices.shape[0] - len(page_indices_list)),
            )

        num_seqs = np.sum(batch.seq_lens > 0, dtype=np.int32).reshape(
            1,
        )
        # Construct distribution for V2 kernel: [decode_end, prefill_end, mixed_end]

        # All sequences are prefill mode
        distribution = np.array([0, num_seqs.item(), num_seqs.item()], dtype=np.int32)

        num_seqs = np.array(num_seqs)
        cu_q_lens = np.array(cu_q_lens)
        cu_kv_lens = np.array(cu_kv_lens)
        page_indices = np.array(page_indices)
        seq_lens = np.array(seq_lens)
        (
            metadata.num_seqs,
            metadata.cu_q_lens,
            metadata.cu_kv_lens,
            metadata.page_indices,
            metadata.seq_lens,
            metadata.distribution,
        ) = device_array(
            (num_seqs, cu_q_lens, cu_kv_lens, page_indices, seq_lens, distribution),
            sharding=(NamedSharding(self.mesh, P()) if jax.process_count() == 1 else None),
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

        # Vectorized preparation
        real_bs = batch.real_bs
        current_seq_lens = batch.seq_lens[:real_bs]
        allocate_lens = batch.spec_info.allocate_lens[:real_bs]

        draft_allocs = allocate_lens - current_seq_lens

        alloc_tokens = current_seq_lens + draft_allocs
        alloc_pages = cdiv(alloc_tokens, self.page_size)

        # src_starts (offset2) is constant across steps
        src_starts = np.concatenate(([0], np.cumsum(alloc_pages)[:-1]))

        full_size = len(original_selected_cache_locs)
        seq_lens_list = []
        for speculative_step_id in range(batch.speculative_num_steps):
            seq_lens = batch.seq_lens + (speculative_step_id)
            seq_lens[batch.real_bs :] = 0
            seq_lens_list.append(seq_lens)
            aligned_seq_lens = ((seq_lens + self.page_size - 1) // self.page_size) * self.page_size
            cu_kv_lens.append(
                np.concatenate(
                    [
                        np.array([0], dtype=np.int32),
                        np.cumsum(aligned_seq_lens),
                    ]
                )
            )

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

            # Reconstruct the full array (sparse/padded)
            result_locs = np.zeros(full_size, dtype=original_selected_cache_locs.dtype)
            result_locs[:total_spec_pages] = gathered_locs

            page_indices_cur_step = (result_locs // self.page_size).astype(np.int32)

            # FIXME Handle padding, this will be move to precompile
            TARGET_PADDING = 16384
            if page_indices_cur_step.shape[0] < TARGET_PADDING:
                padding_size = TARGET_PADDING - page_indices_cur_step.shape[0]
                # Use np.pad to keep it on CPU/Numpy until device_array call
                page_indices_cur_step = np.pad(page_indices_cur_step, (0, padding_size))

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
                metadata_tmp.cu_kv_lens,
                metadata_tmp.page_indices,
                metadata_tmp.seq_lens,
                metadata_tmp.distribution,
            ) = device_array(
                (
                    num_seqs,
                    cu_q_lens,
                    cu_kv_lens[i],
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
            "kv_partition_axis": self.kv_partition_axis,
            "mesh": self.mesh,
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
            aux_data.get("kv_partition_axis", "tensor"),
            aux_data.get("mesh", None),
        )

        obj.forward_metadata = children[0]

        return obj

    def __call__(
        self,
        q: jax.Array,  # [total_tokens, num_heads, head_dim]
        k: jax.Array,  # [total_tokens, num_heads, head_dim]
        v: jax.Array,  # [total_tokens, num_heads, head_dim]
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        attention_mask: jax.Array = None,
        kv_partition_axis: str = "tensor",
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
        if is_tpu_runtime():
            return self._forward_tpu(
                q, k, v, layer, forward_batch, token_to_kv_pool, attention_mask
            )
        else:
            return self._forward_gpu(q, k, v, layer, forward_batch, token_to_kv_pool)

    def _forward_tpu(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        attention_mask: jax.Array = None,
    ):
        """TPU forward pass using ragged_paged_attention."""
        kv_cache_fused = self._get_fused_kv_cache(forward_batch, token_to_kv_pool, layer.layer_id)

        scale = 1.0 / jnp.sqrt(layer.head_dim) if layer.scaling is None else layer.scaling

        # Prepare fused KV cache for paged format: [num_pages, page_size, num_kv_heads * 2, head_dim]
        total_tokens = kv_cache_fused.shape[0]
        num_pages = total_tokens // self.page_size
        kv_cache_fused_paged = kv_cache_fused.reshape(
            num_pages, self.page_size, -1, (self.head_dim + 127) // 128 * 128
        )

        causal = 1

        # custom_mask = self.forward_metadata.custom_mask
        if self.forward_metadata.custom_mask is not None:
            causal = 0
        # Select page indices and remap to SWA pool if KV cache supports it
        page_indices_arg = self.forward_metadata.page_indices
        if hasattr(token_to_kv_pool, "remap_cache_loc") and self.page_size == 1:
            page_indices_arg = token_to_kv_pool.remap_cache_loc(page_indices_arg, layer.layer_id)

        in_specs = (
            P(None, self.kv_partition_axis),  # queries
            P(None, self.kv_partition_axis),  # keys (new tokens)
            P(None, self.kv_partition_axis),  # values (new tokens)
            P(None, None, self.kv_partition_axis, None),  # kv_cache_fused (head interleaved)
            P(),  # kv_lens
            P(),  # page_indices
            P(),  # cu_q_lens
            P(),  # cu_kv_lens
            P(),  # distribution
            P(),  # custom_mask
        )
        out_specs = (
            P(None, self.kv_partition_axis),  # attention output
            P(
                None, self.kv_partition_axis, None
            ),  # updated kv_cache_fused (head interleaved) - 3D: [total_tokens, num_kv_heads*2, head_dim]
        )

        def _ragged_paged_attention_with_fused_kv(*args):
            queries, keys, values, kv_cache_fused = args[:4]
            other_args = args[4:]

            # Call fused KV kernel with head interleaving
            result, updated_kv_cache_fused = ragged_paged_attention(
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
                vmem_limit_bytes=self.vmem_limit_bytes,
            )

            return result, updated_kv_cache_fused

        (
            attn_output,
            updated_kv_cache_fused,
        ) = jax.shard_map(  # Fused KV kernel handles cache updates internally
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
            self.forward_metadata.cu_kv_lens,
            self.forward_metadata.distribution,
            self.forward_metadata.custom_mask,
        )
        pad_width = (self.head_dim + 127) // 128 * 128 - self.head_dim
        if pad_width > 0:
            updated_kv_cache_fused = jnp.pad(
                updated_kv_cache_fused,
                ((0, 0), (0, 0), (0, pad_width)),
                mode="constant",
            )

        return (
            attn_output.reshape(q.shape[0], -1),
            updated_kv_cache_fused,
        )

    def _forward_gpu(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        """GPU forward pass using flash_attn_jax."""
        k_buffer, v_buffer, kv_fused = self._get_and_update_kv_cache_gpu(
            k, v, forward_batch, token_to_kv_pool, self.kv_sharding, layer.layer_id
        )

        scale = 1.0 / jnp.sqrt(layer.head_dim) if layer.scaling is None else layer.scaling

        is_causal = True
        if (
            forward_batch.forward_mode == ForwardMode.DECODE
            or layer.attn_type == AttentionType.ENCODER_ONLY
        ):
            is_causal = False

        attn_output = _forward_flash_attention_gpu(
            q,
            k_buffer,
            v_buffer,
            forward_batch.seq_lens,
            forward_batch.cache_loc,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            layer.q_head_num,
            layer.kv_head_num,
            scale,
            is_causal,
            forward_batch.forward_mode,
            self.kv_sharding,
        )

        return attn_output, kv_fused

    def _get_and_update_kv_cache_gpu(
        self,
        k: jax.Array,
        v: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        kv_sharding: jax.NamedSharding,
        layer_id: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Get the kv cache from the forward batch (GPU path)."""
        updated_layer = token_to_kv_pool.set_kv_buffer_legacy(
            layer_id, forward_batch.out_cache_loc, k, v
        )
        # Functional style: treat updated_layer as authoritative fused buffer for this layer in this step
        # Derive K/V views for attention computation from fused buffer directly
        k = updated_layer.at[:, ::2, :].get(out_sharding=kv_sharding, mode="fill", fill_value=0)
        v = updated_layer.at[:, 1::2, :].get(out_sharding=kv_sharding, mode="fill", fill_value=0)
        # Return fused buffer directly for persistence outside JIT
        fused_return = updated_layer
        return k, v, fused_return

    def _get_fused_kv_cache(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        layer_id: int,
    ) -> jax.Array:
        return token_to_kv_pool.get_fused_kv_buffer(layer_id)

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        if is_tpu_runtime():
            num_page_per_req = cdiv(max_context_len, page_size)
            res = 1024 * 1024 // 2 // num_page_per_req // 4
            assert (
                res > 0
            ), f"max running requests: {res} must larger than 0, please increase page size or decrease max context length"
            return res
        else:
            # GPU flash attention backend doesn't care about max running requests
            return 4096


def _forward_flash_attention_gpu(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    seq_lengths: jax.Array,
    loc: jax.Array,
    extend_prefix_lens: jax.Array,
    extend_seq_lens: jax.Array,
    num_heads: int,
    num_kv_heads: int,
    scale: float = None,
    is_causal: bool = True,
    mode: ForwardMode = ForwardMode.DECODE,
    kv_sharding=None,
):
    """
    Forward pass using Flash Attention for variable-length sequences (GPU).

    Args:
        q: input tokens, shape (num_tokens, num_heads, head_dim) or (num_tokens, hidden_size)
        k_cache: key cache, shape (cache_size, num_kv_heads, head_dim)
        v_cache: value cache, shape (cache_size, num_kv_heads, head_dim)
        seq_lengths: cumulative sequence lengths for each batch
        loc: location of the key/value cache
        extend_prefix_lens: prefix lengths of each batch in extend mode
        extend_seq_lens: sequence lengths of each batch in extend mode
        num_heads: number of query heads
        num_kv_heads: number of key/value heads
        scale: scale for the attention weights (softmax_scale)
        is_causal: whether to apply causal masking
        mode: forward mode (DECODE or EXTEND)
        kv_sharding: sharding for KV cache

    Returns:
        Output tensor of shape [num_tokens, hidden_size]
    """
    cache_size = k_cache.shape[0]
    safe_loc = jnp.where(loc > 0, loc, cache_size)
    k_cache = k_cache.at[safe_loc].get(out_sharding=kv_sharding, mode="fill", fill_value=0)
    v_cache = v_cache.at[safe_loc].get(out_sharding=kv_sharding, mode="fill", fill_value=0)

    # Handle both 2D and 3D input formats for q
    if len(q.shape) == 2:
        # Traditional format: [num_tokens, hidden_size]
        num_tokens, hidden_size = q.shape
        head_dim = hidden_size // num_heads
        q_heads = q.reshape(num_tokens, num_heads, head_dim)
    else:
        # Already in multi-head format: [num_tokens, num_heads, head_dim]
        num_tokens, num_heads_input, head_dim = q.shape
        assert num_heads_input == num_heads, f"Expected {num_heads} heads, got {num_heads_input}"
        hidden_size = num_heads * head_dim
        q_heads = q

    # KV cache from get_kv_buffer is already in multi-head format: [cache_size, num_kv_heads, head_dim]
    k_heads = k_cache
    v_heads = v_cache

    # flash_attn_jax expects:
    # q: [total_tokens_q, num_heads_q, head_dim]
    # k: [total_tokens_k, num_kv_heads, head_dim]
    # v: [total_tokens_k, num_kv_heads, head_dim]
    # seqlens_q: [batch_size + 1] cumulative sequence lengths
    # seqlens_k: [batch_size + 1] cumulative sequence lengths

    # Convert seq_lengths to cumulative format for flash_mha_varlen
    # flash_attn_jax expects seqlens as cumulative (e.g., [0, len1, len1+len2, ...])
    if mode == ForwardMode.EXTEND:
        # In extend mode, use extend_seq_lens for query and seq_lengths for key
        seqlens_q = jnp.concatenate(
            [jnp.array([0], dtype=jnp.int32), jnp.cumsum(extend_seq_lens, dtype=jnp.int32)]
        )
        seqlens_k = jnp.concatenate(
            [jnp.array([0], dtype=jnp.int32), jnp.cumsum(seq_lengths, dtype=jnp.int32)]
        )
    else:
        # In decode mode, each query has length 1
        batch_size = seq_lengths.shape[0]
        seqlens_q = jnp.arange(batch_size + 1, dtype=jnp.int32)
        seqlens_k = jnp.concatenate(
            [jnp.array([0], dtype=jnp.int32), jnp.cumsum(seq_lengths, dtype=jnp.int32)]
        )

    # Ensure proper dtype for flash attention (bf16 or fp16)
    original_dtype = q_heads.dtype
    if q_heads.dtype not in [jnp.bfloat16, jnp.float16]:
        q_heads = q_heads.astype(jnp.bfloat16)
        k_heads = k_heads.astype(jnp.bfloat16)
        v_heads = v_heads.astype(jnp.bfloat16)

    # Call flash_mha_varlen from flash_attn_jax
    # Use -1 for max_seqlen_q and max_seqlen_k to let flash_attn_jax infer from tensor shapes
    # This avoids ConcretizationTypeError during JIT tracing
    attn_output = flash_mha_varlen(
        q_heads,
        k_heads,
        v_heads,
        seqlens_q=seqlens_q,
        seqlens_k=seqlens_k,
        softmax_scale=scale,
        is_causal=is_causal,
    )

    # Convert back to original dtype if needed
    if attn_output.dtype != original_dtype:
        attn_output = attn_output.astype(original_dtype)

    # Reshape output back to [num_tokens, hidden_size]
    return attn_output.reshape(num_tokens, hidden_size)


def vision_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
    window_size: int = -1,
) -> jax.Array:
    """
    Compute vision attention using flash attention on GPU or native attention on TPU.

    This is a simple attention function for vision models (no KV cache, no causal masking).

    Args:
        q, k, v: Input tensors of shape [B, T, N, H] (batch, seq_len, num_heads, head_dim)
        scale: Attention scale factor (1/sqrt(head_dim))
        window_size: Window size for local attention. -1 means full attention.

    Returns:
        Output tensor of shape [B, T, N, H]
    """
    if not is_tpu_runtime():
        # GPU: use flash_mha
        original_dtype = q.dtype
        if q.dtype not in [jnp.bfloat16, jnp.float16]:
            q = q.astype(jnp.bfloat16)
            k = k.astype(jnp.bfloat16)
            v = v.astype(jnp.bfloat16)

        if window_size > 0:
            output = flash_mha(
                q,
                k,
                v,
                softmax_scale=scale,
                is_causal=False,
                window_size=(window_size, window_size),
            )
        else:
            output = flash_mha(q, k, v, softmax_scale=scale, is_causal=False)

        if output.dtype != original_dtype:
            output = output.astype(original_dtype)
        return output
    else:
        # TPU: native attention
        B, T, N, H = q.shape
        q = jnp.transpose(q, (0, 2, 1, 3))  # [B, N, T, H]
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        attn_weights = jnp.einsum("bnth,bnsh->bnts", q, k) * scale

        if window_size > 0:
            # Create window mask for local attention
            positions = jnp.arange(T)
            distance = jnp.abs(positions[:, None] - positions[None, :])
            window_mask = distance > window_size
            attn_weights = jnp.where(
                window_mask[None, None, :, :], jnp.finfo(attn_weights.dtype).min, attn_weights
            )

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        output = jnp.einsum("bnts,bnsh->bnth", attn_weights, v)
        return jnp.transpose(output, (0, 2, 1, 3))  # [B, T, N, H]
