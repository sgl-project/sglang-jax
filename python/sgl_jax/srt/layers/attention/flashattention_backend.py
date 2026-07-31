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
    ragged_paged_attention as ragged_paged_attention_v3,
)
from sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes_v3 import (
    get_tuned_block_sizes_v3,
)
from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.attention.flashattention_metadata import (
    PagedKVLayout,
    pad_page_indices,
)
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
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

    def prepare_paged_kv_layout(
        self,
        batch: ModelWorkerBatch,
        *,
        page_indices: np.ndarray | None = None,
        fixed_capacity: int | None = None,
    ) -> PagedKVLayout:
        """Upload physical KV pages for metadata materialization inside a fused JIT."""
        if page_indices is None:
            page_indices = (
                np.asarray(batch.cache_loc[:: self.page_size], dtype=np.int32) // self.page_size
            )
        else:
            page_indices = np.asarray(page_indices, dtype=np.int32)
        per_dp_bs = int(getattr(batch, "per_dp_bs_size", 0))
        if per_dp_bs <= 0:
            per_dp_bs = len(batch.seq_lens)
        max_num_seqs = batch.dp_size * per_dp_bs
        page_indices = pad_page_indices(
            page_indices,
            max_num_seqs,
            fixed_capacity=fixed_capacity,
        )
        data_sharding = NamedSharding(self.mesh, P("data"))
        page_indices_device = device_array(page_indices, sharding=data_sharding)

        swa_page_indices = None
        swa_mapping = getattr(self, "swa_index_mapping", None)
        if swa_mapping is not None:
            full_loc = (page_indices.astype(np.int64) * self.page_size).astype(np.int32)
            if isinstance(swa_mapping, list):
                full_2d = full_loc.reshape(batch.dp_size, -1)
                swa_2d = np.empty_like(full_2d)
                for r in range(batch.dp_size):
                    swa_2d[r] = np.asarray(swa_mapping[r])[full_2d[r]]
                swa_loc = swa_2d.ravel()
            else:
                swa_loc = np.asarray(swa_mapping)[full_loc]
            swa_page_indices = device_array(
                (swa_loc // self.page_size).astype(np.int32),
                sharding=data_sharding,
            )
        return PagedKVLayout(
            page_indices=page_indices_device,
            swa_page_indices=swa_page_indices,
        )

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

        attn_type = getattr(layer, "attn_type", None)
        if getattr(attn_type, "value", attn_type) == "encoder_only":
            causal = 0
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
        target_verify_tokens_per_seq = (
            getattr(forward_batch.spec_info, "draft_token_num", None)
            if forward_batch.forward_mode.is_target_verify()
            else None
        )

        def _ragged_paged_attention_with_fused_kv(*args):
            queries, keys, values, kv_cache_fused = args[:4]
            other_args = args[4:]
            # Target verify is semantically many short decode-like query
            # segments over long prefixes, although it must execute in the
            # MIXED kernel for q_len > 1. Keep its tuning namespace separate
            # from ordinary mixed/prefill. The current table is for causal
            # chain verification; custom tree masks retain the generic path.
            target_verify_m_block_sizes = (
                get_tuned_block_sizes_v3(
                    "v",
                    queries.dtype,
                    kv_cache_fused.dtype,
                    queries.shape[1],
                    keys.shape[1],
                    queries.shape[2],
                    kv_cache_fused.shape[1],
                    queries.shape[0],
                    sliding_window=layer.sliding_window_size,
                    tokens_per_seq=target_verify_tokens_per_seq,
                )
                if target_verify_tokens_per_seq is not None
                and self.forward_metadata.custom_mask is None
                else None
            )

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
                softmax_dtype=layer.softmax_dtype,
                mask_aligned_to_cu_kv=mask_aligned_to_cu_kv,
                m_block_sizes=target_verify_m_block_sizes,
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
