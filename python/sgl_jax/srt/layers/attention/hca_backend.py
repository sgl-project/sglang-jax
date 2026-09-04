"""SGLang-JAX attention backend for HCA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class
from sgl_jax.srt.kernels.hca.attention import INERT_QUERY_OFFSET
from sgl_jax.srt.kernels.hca.hca import (
    HCAMetadata,
    fused_projection_weight,
    hca_step,
)
from sgl_jax.srt.kernels.hca.tuned_block_sizes import (
    HCAKernelSchedule,
    get_hca_kernel_schedule,
)
from sgl_jax.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    AttentionBackendMetadata,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.jax_utils import device_array

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


@register_pytree_node_class
@dataclass
class HCABackendMetadata(AttentionBackendMetadata):
    """Per-forward HCA metadata plus static framework/optimization choices."""

    kernel: HCAMetadata | None = None
    schedule: HCAKernelSchedule | None = None
    use_uniform_prefill_fast_path: bool = False

    def tree_flatten(self):
        return (self.kernel,), (self.schedule, self.use_uniform_prefill_fast_path)

    @classmethod
    def tree_unflatten(cls, static_options, children):
        schedule, use_uniform_prefill_fast_path = static_options
        return cls(
            kernel=children[0],
            schedule=schedule,
            use_uniform_prefill_fast_path=use_uniform_prefill_fast_path,
        )


# Minimum padded capacities: small floors stop tiny batches re-bucketing every
# few steps, and the page-table floors hold one large batch's tables.
_DECODE_IDS_FLOOR = 8
_BOUNDARY_FLOOR = 8
_WINDOW_TABLE_FLOOR = 512
_COMPRESSED_TABLE_FLOOR = 64


def _query_schedule(cu_q_lens: np.ndarray, query_block_size: int):
    """Build execution-only query blocks after the platform schedule is known.

    Returns possibly-empty arrays; ``get_forward_metadata`` pads them to stable
    capacities."""
    q_lens = np.diff(cu_q_lens).astype(np.int32)
    block_counts = np.where(
        q_lens == 1,
        0,
        (q_lens + query_block_size - 1) // query_block_size,
    )
    request_ids = np.repeat(np.arange(q_lens.size, dtype=np.int32), block_counts)
    offsets = np.concatenate(
        [
            np.arange(0, int(q_len), query_block_size, dtype=np.int32)
            if q_len != 1
            else np.empty((0,), np.int32)
            for q_len in q_lens
        ]
    )
    return request_ids, offsets.astype(np.int32), np.flatnonzero(q_lens == 1).astype(np.int32)


def _pad_capacity(values: np.ndarray, capacity: int, fill) -> np.ndarray:
    """Right-pad to an exact batch-shape-derived capacity with an inert fill."""
    values = np.asarray(values, np.int32)
    if values.shape[0] > capacity:
        raise ValueError(f"HCA metadata length {values.shape[0]} exceeds capacity {capacity}")
    return np.pad(values, (0, capacity - values.shape[0]), constant_values=fill)


def _bucket_capacity(length: int, floor: int, bound: int | None = None) -> int:
    """Smallest power-of-two capacity covering ``length``, capped at ``bound``."""
    capacity = floor
    while capacity < length:
        capacity *= 2
    return capacity if bound is None else min(capacity, bound)


def _bucket_max_queries(max_queries: int, floor: int) -> int:
    """Bucket the per-request query capacity to a bounded ladder.

    Decode (1) keeps its dedicated value; longer chunks round up onto powers of
    two interleaved with 1.5x steps (..., 128, 192, 256, 384, ...), so a chunk
    just past a power of two pays 1.5x KV staging instead of 2x.
    """
    if max_queries <= 1:
        return max_queries
    power = 1 << max((max_queries - 1).bit_length() - 1, 0)
    bucket = power * 3 // 2 if max_queries <= power * 3 // 2 else power * 2
    return max(floor, bucket)


def _metadata_partition_spec(metadata: HCAMetadata) -> HCAMetadata:
    """Every metadata leaf rides the leading data axis, like SGLang batch fields.

    Derived rather than hand-listed so a new field cannot silently miss a spec.
    """
    return jax.tree.map(lambda _: P("data"), metadata)


@dataclass
class HCABackend(AttentionBackend):
    """Run cache-aware HCA through the SGLang-JAX attention interface."""

    def __init__(
        self,
        *,
        num_attn_heads: int = 64,
        head_dim: int = 512,
        compressor_hidden_size: int = 4096,
        page_size: int = 128,
        compress_ratio: int = 128,
        window_size: int = 128,
        mesh: jax.sharding.Mesh,
    ):
        if mesh is None:
            raise ValueError("production HCABackend requires the SGLang device mesh")
        if page_size < 2 or window_size % page_size:
            raise ValueError("HCA page_size must be >=2 and divide window_size")
        if (
            num_attn_heads != 64
            or head_dim != 512
            or compressor_hidden_size != 4096
            or compress_ratio != 128
            or window_size != 128
        ):
            raise ValueError(
                "production HCA requires H=64, D=512, hidden=4096, ratio=128, and window=128"
            )
        self.num_heads = num_attn_heads
        self.head_dim = head_dim
        self.compressor_hidden_size = compressor_hidden_size
        self.page_size = page_size
        self.compress_ratio = compress_ratio
        self.window_size = window_size
        self.mesh = mesh
        self.forward_metadata = nnx.data(HCABackendMetadata())
        # HCA page ownership: set by model_runner after pool creation, read on
        # host during metadata construction, like FlashAttention's swa_index_mapping.
        self.allocator = None

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        """Derive HCA metadata from the worker batch and place it on the data mesh.

        Reads only standard batch fields plus the attached allocator's page
        tables; the scheduler must already have grown the compressed tier via
        ``ensure_compressed_capacity``.
        """
        if self.allocator is None:
            raise RuntimeError("model_runner must attach an HCAKVPoolAllocator first")
        req_pool_indices = np.asarray(batch.req_pool_indices, np.int32)
        seq_lens = np.asarray(batch.seq_lens, np.int32)
        positions = np.asarray(batch.positions, np.int32).reshape(-1)
        if req_pool_indices.shape != seq_lens.shape:
            raise ValueError("req_pool_indices and seq_lens must have the same shape")
        if batch.forward_mode == ForwardMode.DECODE:
            q_lens = np.ones_like(seq_lens)
            uniform_prefill = False
        elif batch.forward_mode == ForwardMode.EXTEND:
            q_lens = np.asarray(batch.extend_seq_lens, np.int32)
            if q_lens.shape != seq_lens.shape:
                raise ValueError("extend_seq_lens must have one value per request")
            prefix_lens = (
                seq_lens - q_lens
                if batch.extend_prefix_lens is None
                else np.asarray(batch.extend_prefix_lens, np.int32)
            )
            uniform_prefill = bool(
                q_lens.size and np.all(prefix_lens == 0) and np.all(q_lens == q_lens[0])
            )
        else:
            raise ValueError(f"HCA does not support {batch.forward_mode}")
        if np.any(q_lens <= 0):
            raise ValueError("every HCA request must contain at least one query token")

        valid_token_count = int(q_lens.sum())
        if valid_token_count > positions.size:
            raise ValueError("positions does not contain every HCA query token")
        query_seq_ids = np.repeat(np.arange(q_lens.size, dtype=np.int32), q_lens)
        valid_token_mask = np.arange(positions.size) < valid_token_count
        if positions.size > valid_token_count:
            query_seq_ids = np.pad(query_seq_ids, (0, positions.size - valid_token_count))
        cu_q_lens = np.concatenate((np.zeros((1,), np.int32), np.cumsum(q_lens, dtype=np.int32)))
        emit_mask = valid_token_mask & (np.mod(positions + 1, self.compress_ratio) == 0)
        boundary_tokens = np.flatnonzero(emit_mask).astype(np.int32)

        # Recurrent slots ride the standard hybrid-recurrent batch field.
        if batch.recurrent_indices is None:
            raise ValueError("HCA requires batch.recurrent_indices from hybrid scheduling")
        state_by_request = np.asarray(batch.recurrent_indices, np.int32)
        if state_by_request.shape != seq_lens.shape or np.any(state_by_request == 0):
            raise ValueError("HCA forward references an unallocated recurrent slot")
        safe_ids = np.where(valid_token_mask, query_seq_ids, 0)
        state_slots = np.where(valid_token_mask, state_by_request[safe_ids], 0).astype(np.int32)

        (
            window_page_indices,
            window_cu_kv_lens,
            compressed_page_indices,
            compressed_cu_kv_lens,
            compressed_kv_lens,
        ) = self.allocator.page_tables(req_pool_indices, seq_lens)

        tensor_shards = int(self.mesh.shape.get("tensor", 1))
        if self.num_heads % tensor_shards:
            raise ValueError("HCA attention heads must divide the tensor mesh")
        device_kind = str(np.asarray(self.mesh.devices).reshape(-1)[0].device_kind)
        schedule = get_hca_kernel_schedule(
            device_kind,
            page_size=self.page_size,
            max_compressed_entries=max(1, int(compressed_kv_lens.max(initial=0))),
            local_heads=self.num_heads // tensor_shards,
            head_dim=self.head_dim,
        )
        block_requests, block_offsets, decode_requests = _query_schedule(
            cu_q_lens, schedule.query_block_size
        )
        # Pad every batch-dependent length to a bucketed capacity so metadata
        # drift never recompiles; ``HCAMetadata`` documents the inert sentinels.
        tokens = int(positions.size)
        batch_size = int(seq_lens.shape[0])
        block_capacity = tokens // schedule.query_block_size + batch_size
        if block_requests.shape[0]:
            block_requests = _pad_capacity(block_requests, block_capacity, 0)
            block_offsets = _pad_capacity(block_offsets, block_capacity, INERT_QUERY_OFFSET)
        if decode_requests.shape[0]:
            decode_capacity = _bucket_capacity(
                decode_requests.shape[0], _DECODE_IDS_FLOOR, bound=batch_size
            )
            decode_requests = _pad_capacity(decode_requests, decode_capacity, -1)
        if boundary_tokens.shape[0]:
            boundary_bound = tokens // self.compress_ratio + batch_size
            boundary_capacity = _bucket_capacity(
                boundary_tokens.shape[0], _BOUNDARY_FLOOR, bound=boundary_bound
            )
            boundary_tokens = _pad_capacity(boundary_tokens, boundary_capacity, tokens)
        window_pages = _pad_capacity(
            window_page_indices,
            _bucket_capacity(window_page_indices.shape[0], _WINDOW_TABLE_FLOOR),
            0,
        )
        compressed_pages = _pad_capacity(
            compressed_page_indices,
            _bucket_capacity(compressed_page_indices.shape[0], _COMPRESSED_TABLE_FLOOR),
            0,
        )
        max_queries = _bucket_max_queries(int(q_lens.max()), schedule.query_block_size)
        arrays = device_array(
            (
                state_slots,
                query_seq_ids.astype(np.int32),
                cu_q_lens,
                valid_token_mask,
                boundary_tokens,
                window_pages,
                window_cu_kv_lens,
                seq_lens,
                compressed_pages,
                compressed_cu_kv_lens,
                compressed_kv_lens,
                block_requests,
                block_offsets,
                decode_requests,
            ),
            sharding=NamedSharding(self.mesh, P("data")),
        )
        kernel_metadata = HCAMetadata(
            *arrays,
            max_queries_per_request=max_queries,
        )
        return HCABackendMetadata(
            kernel=kernel_metadata,
            schedule=schedule,
            use_uniform_prefill_fast_path=uniform_prefill,
        )

    def tree_flatten(self):
        children = (self.forward_metadata,)
        aux = {
            "num_attn_heads": self.num_heads,
            "head_dim": self.head_dim,
            "compressor_hidden_size": self.compressor_hidden_size,
            "page_size": self.page_size,
            "compress_ratio": self.compress_ratio,
            "window_size": self.window_size,
            "mesh": self.mesh,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls(**aux)
        obj.forward_metadata = children[0]
        return obj

    def _check_constants(
        self,
        wkv,
        wgate,
        ape,
        norm_weight,
        cos,
        sin,
        attention_sink,
        fused_weight,
        max_context_len,
    ) -> None:
        """Validate the shapes that cannot change between steps.

        Factored out of the hot path for readability; under jit these run at
        trace time only. Must not mutate ``self`` -- nnx forbids it in a trace.
        """
        if jax.default_backend() != "tpu":
            raise RuntimeError("production HCABackend requires a TPU backend")
        weight_shape = (self.head_dim, self.compressor_hidden_size)
        if wkv.shape != weight_shape or wgate.shape != weight_shape:
            raise ValueError("wkv and wgate must both be [512,4096]")
        if ape.shape != (self.compress_ratio, self.head_dim):
            raise ValueError("ape must be [128,512]")
        if norm_weight.shape != (self.head_dim,):
            raise ValueError("norm_weight must be [512]")
        if attention_sink.shape != (self.num_heads,):
            raise ValueError("attention_sink must be [64]")
        if cos.ndim != 2 or cos.shape[1] != 32 or sin.shape != cos.shape:
            raise ValueError("production HCA RoPE tables must both be [positions,32]")
        # Boundary emission gathers the row at each group start; a shorter table
        # would silently rotate records with clamped or filled frequencies.
        min_rope_rows = max(1, max_context_len - self.compress_ratio + 1)
        if cos.shape[0] < min_rope_rows:
            raise ValueError(
                f"RoPE tables cover {cos.shape[0]} positions but max_context_len="
                f"{max_context_len} requires at least {min_rope_rows}"
            )
        if fused_weight is not None and fused_weight.shape != (
            self.compressor_hidden_size,
            2 * self.head_dim,
        ):
            raise ValueError("fused_weight must be [4096,1024]")

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer,
        forward_batch: ForwardBatch,
        token_to_kv_pool,
        *,
        recurrent_state_pool,
        compressor_input: jax.Array,
        wkv: jax.Array,
        wgate: jax.Array,
        ape: jax.Array,
        norm_weight: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        attention_sink: jax.Array,
        fused_weight: jax.Array | None = None,
        **_kwargs,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        """Run complete cache-aware HCA and return explicit pool updates."""
        metadata = self.forward_metadata
        if metadata.kernel is None:
            raise RuntimeError("HCABackend.forward_metadata has not been prepared")
        if metadata.schedule is None:
            raise RuntimeError("HCABackend has no HCA kernel schedule")
        # Only the per-token shapes can change between steps; the model
        # constants are validated in _check_constants below.
        if q.ndim != 3 or q.shape[1:] != (self.num_heads, self.head_dim):
            raise ValueError("q must be [T,num_attn_heads,head_dim]")
        if k.ndim == 3 and k.shape[1] == 1:
            new_kv = k[:, 0]
        elif k.ndim == 2:
            new_kv = k
        else:
            raise ValueError("HCA k/v must be [T,D] or [T,1,D]")
        if v.shape != k.shape or new_kv.shape != (q.shape[0], self.head_dim):
            raise ValueError("HCA k and v must share the same KV shape")
        if compressor_input.shape != (q.shape[0], self.compressor_hidden_size):
            raise ValueError("compressor_input must be [T,4096]")
        self._check_constants(
            wkv,
            wgate,
            ape,
            norm_weight,
            cos,
            sin,
            attention_sink,
            fused_weight,
            token_to_kv_pool.max_context_len,
        )

        layer_index = token_to_kv_pool._layer_index(int(layer.layer_id))
        recurrent_state_pool._layer_index(int(layer.layer_id))

        kernel_options = {
            "softmax_scale": (
                self.head_dim**-0.5
                if getattr(layer, "scaling", None) is None
                else float(layer.scaling)
            ),
            "compress_ratio": self.compress_ratio,
            "head_dim": self.head_dim,
            "window_size": self.window_size,
            "schedule": metadata.schedule,
        }
        fused_weight = fused_projection_weight(wkv, wgate, fused_weight)

        forward_mode = forward_batch.forward_mode
        if forward_mode.is_decode():
            kernel_options["mode"] = "decode"
        elif forward_mode.is_extend():
            kernel_options["mode"] = (
                "uniform" if metadata.use_uniform_prefill_fast_path else "ragged"
            )
        else:
            raise ValueError(f"unsupported HCA forward mode: {forward_mode}")

        # Built inline like the MLA and GDN backends: under the model's outer
        # jit this is traced once, so a cached callable buys nothing.
        def rank_local(
            x,
            q_,
            new_kv_,
            state,
            window,
            compressed,
            wkv_,
            wgate_,
            ape_,
            norm_,
            cos_,
            sin_,
            positions_,
            sink_,
            md,
            fused,
        ):
            output, state, window, compressed = hca_step(
                x,
                q_,
                new_kv_,
                state,
                window,
                compressed,
                wkv_,
                wgate_,
                ape_,
                norm_,
                cos_,
                sin_,
                positions_,
                sink_,
                md,
                fused_weight=fused,
                **kernel_options,
            )
            return output.reshape(output.shape[0], -1), state, window, compressed

        output, state, window, compressed = jax.shard_map(
            rank_local,
            mesh=self.mesh,
            in_specs=(
                P("data", None),  # compressor_input [T, hidden]
                P("data", "tensor", None),  # q                [T, H/tp, D]
                P("data", None),  # new_kv           [T, D]
                P("data", None, None, None),  # recurrent state pool
                P("data", None, None, None),  # window cache
                P("data", None, None, None),  # compressed cache
                P(None, None),  # wkv
                P(None, None),  # wgate
                P(None, None),  # ape
                P(None),  # norm_weight
                P(None, None),  # cos
                P(None, None),  # sin
                P("data"),  # positions        [T]
                P("tensor"),  # attention_sink   [H/tp]
                _metadata_partition_spec(metadata.kernel),
                P(None, None),  # fused_weight
            ),
            out_specs=(
                P("data", "tensor"),  # output [T, H/tp*D]
                P("data", None, None, None),  # state pool
                P("data", None, None, None),  # window cache
                P("data", None, None, None),  # compressed cache
            ),
            check_vma=False,
        )(
            compressor_input,
            q,
            new_kv,
            recurrent_state_pool.get_hca_state(int(layer.layer_id)),
            token_to_kv_pool.window_buffer[layer_index],
            token_to_kv_pool.compressed_buffer[layer_index],
            wkv,
            wgate,
            ape,
            norm_weight,
            cos,
            sin,
            forward_batch.positions,
            attention_sink,
            metadata.kernel,
            fused_weight,
        )
        return output.astype(q.dtype), (state, window, compressed)

    @staticmethod
    def pack_pool_updates(layer_updates) -> dict:
        """Regroup per-layer ``(state, window, compressed)`` for their owning pools."""
        states, windows, compressed = zip(*layer_updates, strict=True)
        return {
            "token_to_kv_pool": {
                "window_buffer": list(windows),
                "compressed_buffer": list(compressed),
            },
            "recurrent_state_pool": {"state_buffers": list(states)},
        }

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        pages_per_request = (max_context_len + page_size - 1) // page_size
        return max(1, 1024 * 1024 // 2 // pages_per_request // 4)


__all__ = ["HCABackend", "HCABackendMetadata"]
