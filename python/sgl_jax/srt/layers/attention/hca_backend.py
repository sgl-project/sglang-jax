"""SGLang-JAX attention backend for HCA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.hca.attention import INERT_QUERY_OFFSET
from sgl_jax.srt.kernels.hca.hca import HCAMetadata
from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule
from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.attention.dsv4.hca import DeepseekV4HCABackendMixin
from sgl_jax.srt.layers.attention.hca_execution import run_hca
from sgl_jax.srt.layers.attention.hca_metadata import (
    _BOUNDARY_FLOOR,
    _COMPRESSED_TABLE_FLOOR,
    _DECODE_IDS_FLOOR,
    _WINDOW_TABLE_FLOOR,
    HCABackendMetadata,
    _bucket_capacity,
    _bucket_max_queries,
    _pad_capacity,
    _query_schedule,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.jax_utils import device_array

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class HCABackend(AttentionBackend, DeepseekV4HCABackendMixin):
    """HCA execution with legacy pools or request-owned V4 resources."""

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
        max_context_len: int | None = None,
    ):
        if mesh is None:
            raise ValueError("production HCABackend requires the SGLang device mesh")
        if max_context_len is not None and (max_context_len <= 0 or page_size not in (128, 256)):
            raise ValueError("V4 HCA requires a positive context capacity and page size 128 or 256")
        if max_context_len is None and (page_size < 2 or window_size % page_size):
            raise ValueError("HCA page_size must be >=2 and divide window_size")
        if (
            num_attn_heads != 64
            or head_dim != 512
            or compressor_hidden_size not in (4096, 7168)
            or compress_ratio != 128
            or window_size != 128
        ):
            raise ValueError(
                "production HCA requires H=64, D=512, hidden in (4096,7168), ratio=128, and window=128"
            )
        self.num_heads = num_attn_heads
        self.head_dim = head_dim
        self.compressor_hidden_size = compressor_hidden_size
        self.page_size = page_size
        self.compress_ratio = compress_ratio
        self.window_size = window_size
        self.mesh = mesh
        self.max_context_len = max_context_len
        self.request_capacity = None
        self.forward_metadata = nnx.data(HCABackendMetadata())
        # HCA page ownership: set by model_runner after pool creation, read on
        # host during metadata construction, like FlashAttention's swa_index_mapping.
        self.allocator = None

    def bind_resources(self, request_pool, allocator):
        """Bind resource geometry without capturing host owners in the model graph."""
        if self.max_context_len is None:
            raise ValueError("V4 resources require max_context_len")
        if allocator.dp_size != self.mesh.shape["data"] or allocator.page_size != self.page_size:
            raise ValueError("HCA and V4 resource geometry disagree")
        self.request_capacity = request_pool.size

    def get_forward_metadata(self, batch: ModelWorkerBatch, *, request_pool=None, allocator=None):
        """Build read/write tables from caller-owned allocations; never allocate.

        V4 callers supply request_pool and allocator; legacy callers attach self.allocator.
        """
        if request_pool is not None or allocator is not None:
            if request_pool is None or allocator is None or self.request_capacity is None:
                raise ValueError("bind V4 resources and supply both request_pool and allocator")
            return self._get_hca_metadata(batch, request_pool=request_pool, allocator=allocator)
        if self.request_capacity is not None:
            raise ValueError("V4 metadata requires request_pool and allocator")
        if self.allocator is None:
            raise RuntimeError("model_runner must attach an HCAKVPoolAllocator first")
        req_pool_indices = np.asarray(batch.req_pool_indices, np.int32)
        seq_lens = np.asarray(batch.seq_lens, np.int32)
        positions = np.asarray(batch.positions, np.int32).reshape(-1)
        if req_pool_indices.shape != seq_lens.shape:
            raise ValueError("req_pool_indices and seq_lens must have the same shape")
        if np.any(seq_lens < 0):
            raise ValueError("HCA sequence lengths must be non-negative")
        active_requests = seq_lens > 0
        if np.any(req_pool_indices[active_requests] < 0):
            raise ValueError("active HCA requests require allocated request slots")
        if batch.forward_mode == ForwardMode.DECODE:
            q_lens = active_requests.astype(np.int32)
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
            if prefix_lens.shape != seq_lens.shape:
                raise ValueError("extend_prefix_lens must have one value per request")
            active_q_lens = q_lens[active_requests]
            uniform_prefill = bool(
                active_q_lens.size
                and int(active_q_lens.sum()) == positions.size
                and np.all(prefix_lens[active_requests] == 0)
                and np.all(active_q_lens == active_q_lens[0])
            )
        else:
            raise ValueError(f"HCA does not support {batch.forward_mode}")
        if np.any(q_lens < 0) or np.any((q_lens > 0) != active_requests):
            raise ValueError("only active HCA requests may contain query tokens")

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
        if state_by_request.shape != seq_lens.shape:
            raise ValueError("HCA requires one recurrent slot per request")
        if np.any(state_by_request[active_requests] == 0):
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
            "max_context_len": self.max_context_len,
            "request_capacity": self.request_capacity,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        aux = dict(aux)
        request_capacity = aux.pop("request_capacity", None)
        obj = cls(**aux)
        obj.request_capacity = request_capacity
        obj.forward_metadata = children[0]
        return obj

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer,
        forward_batch: ForwardBatch,
        token_to_kv_pool,
        *,
        recurrent_state_pool=None,
        compressor_state_pool=None,
        compressor_input: jax.Array,
        wkv: jax.Array,
        wgate: jax.Array,
        ape: jax.Array,
        norm_weight: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        attention_sink: jax.Array,
        fused_weight: jax.Array | None = None,
        norm_eps: float = 1e-6,
        metadata=None,
        **_kwargs,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array] | dict[str, jax.Array]]:
        """Run complete cache-aware HCA and return explicit pool updates."""
        metadata = self.forward_metadata if metadata is None else metadata
        if compressor_state_pool is not None:
            if recurrent_state_pool is not None or self.request_capacity is None:
                raise ValueError(
                    "V4 execution requires bound resources and only compressor_state_pool"
                )
            output, (state, window, compressed) = self._forward_hca(
                q,
                k,
                v,
                layer,
                forward_batch,
                token_to_kv_pool,
                compressor_state_pool=compressor_state_pool,
                metadata=metadata,
                compressor_input=compressor_input,
                wkv=wkv,
                wgate=wgate,
                ape=ape,
                norm_weight=norm_weight,
                cos=cos,
                sin=sin,
                attention_sink=attention_sink,
                fused_weight=fused_weight,
                norm_eps=norm_eps,
            )
            return output.reshape(q.shape), {
                "state": state,
                "swa": window,
                "compressed": compressed,
            }
        if self.request_capacity is not None or recurrent_state_pool is None:
            raise ValueError("supply the state pool matching the bound HCA resources")
        layer_id = int(layer.layer_id)
        layer_index = token_to_kv_pool._layer_index(layer_id)
        recurrent_state_pool._layer_index(layer_id)
        return run_hca(
            q,
            k,
            v,
            mesh=self.mesh,
            compressor_hidden_size=self.compressor_hidden_size,
            page_size=self.page_size,
            max_context_len=token_to_kv_pool.max_context_len,
            positions=forward_batch.positions,
            forward_mode=forward_batch.forward_mode,
            state_arg=recurrent_state_pool.get_hca_state(layer_id),
            window_arg=token_to_kv_pool.window_buffer[layer_index],
            compressed_arg=token_to_kv_pool.compressed_buffer[layer_index],
            metadata=metadata,
            compressor_input=compressor_input,
            wkv=wkv,
            wgate=wgate,
            ape=ape,
            norm_weight=norm_weight,
            cos=cos,
            sin=sin,
            attention_sink=attention_sink,
            fused_weight=fused_weight,
            softmax_scale=getattr(layer, "scaling", None),
            norm_eps=norm_eps,
        )

    @staticmethod
    def pack_pool_updates(layer_updates, token_to_kv_pool=None, compressor_state_pool=None) -> dict:
        """Pack complete family replacements; only the runtime commits the updates."""
        if token_to_kv_pool is not None or compressor_state_pool is not None:
            if token_to_kv_pool is None or compressor_state_pool is None:
                raise ValueError("V4 updates require both KV and compressor-state pools")
            kv_updates, state_updates = {}, {}
            for layer_id, updates in layer_updates.items():
                if set(updates) != {"state", "swa", "compressed"}:
                    raise ValueError(
                        "each HCA layer must return complete swa, c128 and state updates"
                    )
                kv_updates[layer_id] = {key: updates[key] for key in ("swa", "compressed")}
                state_updates[layer_id] = {"compressor": updates["state"]}
            return {
                "token_to_kv_pool": token_to_kv_pool.build_buffer_updates(kv_updates),
                "compressor_state_pool": compressor_state_pool.build_buffer_updates(state_updates),
            }
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
