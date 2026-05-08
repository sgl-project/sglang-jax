"""LightningAttnBackend — GLA (Gated Linear Attention) backend for BailingMoeV2.5.

Extends LinearRecurrentAttnBackend to provide:
- Chunked prefill via simple_gla_fwd (Pallas kernel) with scatter/gather packing
- Decode via fused_recurrent_simple_gla (jax.lax.scan)
- Recurrent state management through RecurrentStatePool (no conv state)

Aligns with upstream sglang's LightningAttentionBackend(MambaAttnBackendBase) pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.linear.gla_metadata import (
    gather_from_packed,
    scatter_to_packed,
)
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackend,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

try:
    from sgl_jax.srt.kernels.simple_gla.simple_gla import (
        fused_recurrent_simple_gla,
        simple_gla_fwd,
    )
except ModuleNotFoundError:
    simple_gla_fwd = None
    fused_recurrent_simple_gla = None

if TYPE_CHECKING:
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

_CHUNK_SIZE = 64


class LightningAttnBackend(LinearRecurrentAttnBackend):
    """Attention backend for GLA (Gated Linear Attention) used by BailingMoeV2.5."""

    def __init__(self, mesh: jax.sharding.Mesh = None, chunk_size: int = _CHUNK_SIZE):
        super().__init__(mesh=mesh)
        self.chunk_size = chunk_size
        self.T_packed_bucket: int = 0
        self.scatter_idx = nnx.data(None)
        self.cu_seqlens_aligned = nnx.data(None)

    def get_forward_metadata(self, batch):
        metadata = super().get_forward_metadata(batch)

        if batch.forward_mode == ForwardMode.EXTEND:
            self._compute_scatter_metadata(batch)
        else:
            self.scatter_idx = nnx.data(None)
            self.cu_seqlens_aligned = nnx.data(None)

        return metadata

    def _compute_scatter_metadata(self, batch):
        extend_seq_lens = np.asarray(batch.extend_seq_lens, dtype=np.int32)
        cs = self.chunk_size

        aligned_lens = np.where(
            extend_seq_lens == 0,
            0,
            ((extend_seq_lens + cs - 1) // cs) * cs,
        ).astype(np.int32)

        cu_seqlens = np.concatenate(
            [np.array([0], dtype=np.int32), np.cumsum(aligned_lens, dtype=np.int32)]
        )

        T_pb = int(cu_seqlens[-1])
        T_outer = len(batch.input_ids)

        scatter_idx = np.full(T_outer, T_pb, dtype=np.int32)

        offset_tight = 0
        for i in range(len(extend_seq_lens)):
            seq_len = int(extend_seq_lens[i])
            if seq_len == 0:
                continue
            scatter_idx[offset_tight : offset_tight + seq_len] = np.arange(
                cu_seqlens[i], cu_seqlens[i] + seq_len, dtype=np.int32
            )
            offset_tight += seq_len

        self.T_packed_bucket = T_pb

        sharding = (
            NamedSharding(self.mesh, P())
            if self.mesh is not None and jax.process_count() == 1
            else None
        )
        from sgl_jax.srt.utils.jax_utils import device_array

        cu_dev, scatter_dev = device_array((cu_seqlens, scatter_idx), sharding=sharding)
        self.cu_seqlens_aligned = nnx.data(cu_dev)
        self.scatter_idx = nnx.data(scatter_dev)

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer,
        forward_batch: ForwardBatch,
        recurrent_state_pool,
        **kwargs,
    ) -> tuple[jax.Array, tuple]:
        recurrent_indices = self.forward_metadata.recurrent_indices
        ssm_states = self.get_state(recurrent_state_pool, layer.layer_id, recurrent_indices)

        if forward_batch.forward_mode.is_decode():
            output, new_recurrent = self._forward_decode(q, k, v, ssm_states, layer)
        elif forward_batch.forward_mode == ForwardMode.EXTEND:
            output, new_recurrent = self._forward_extend(q, k, v, ssm_states, layer)
        else:
            raise NotImplementedError(
                f"LightningAttnBackend does not support {forward_batch.forward_mode}"
            )

        new_ssm_full = self.set_ssm_state(
            recurrent_state_pool, layer.layer_id, recurrent_indices, new_recurrent
        )
        return output.reshape(output.shape[0], -1), (new_ssm_full, [])

    def get_state(self, recurrent_state_pool, layer_id, recurrent_indices):
        recurrent_buffer, _ = self.get_layer_cache(recurrent_state_pool, layer_id)
        return recurrent_buffer[recurrent_indices]

    def set_ssm_state(self, recurrent_state_pool, layer_id, recurrent_indices, new_recurrent):
        recurrent_buffer, _ = self.get_layer_cache(recurrent_state_pool, layer_id)
        return recurrent_buffer.at[recurrent_indices].set(new_recurrent)

    def _forward_decode(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        ssm_states: jax.Array,
        layer,
    ) -> tuple[jax.Array, jax.Array]:
        if fused_recurrent_simple_gla is None:
            raise ImportError("simple_gla kernel is required for GLA decode")

        ssm_states = ssm_states.astype(jnp.float32)
        ssm_states = jax.sharding.reshard(
            ssm_states,
            NamedSharding(layer.mesh, P(None, "tensor", None, None)),
        )

        q_d = q[:, None, :, :]
        k_d = k[:, None, :, :]
        v_d = v[:, None, :, :]
        output_d, new_state = fused_recurrent_simple_gla(
            q_d,
            k_d,
            v_d,
            g_gamma=layer.slope,
            initial_state=ssm_states,
            output_final_state=True,
            scale=None,
        )
        return output_d[:, 0, :, :], new_state

    def _forward_extend(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        ssm_states: jax.Array,
        layer,
    ) -> tuple[jax.Array, jax.Array]:
        if simple_gla_fwd is None:
            raise ImportError("simple_gla kernel is required for GLA prefill")

        T_pb = self.T_packed_bucket
        scatter_idx = self.scatter_idx
        cu_seqlens = self.cu_seqlens_aligned

        ssm_states = ssm_states.astype(jnp.float32)

        slope_sm = jax.sharding.reshard(
            layer.slope, NamedSharding(layer.mesh, P("tensor"))
        )
        h0_sm = jax.sharding.reshard(
            ssm_states,
            NamedSharding(layer.mesh, P(None, "tensor", None, None)),
        )

        chunk_size = self.chunk_size

        def _prefill_fn(q_local, k_local, v_local, gamma, h0, scatter_idx_p, cu_seqlens_p):
            q_p = scatter_to_packed(q_local, scatter_idx_p, T_pb)
            k_p = scatter_to_packed(k_local, scatter_idx_p, T_pb)
            v_p = scatter_to_packed(v_local, scatter_idx_p, T_pb)
            return simple_gla_fwd(
                q_p,
                k_p,
                v_p,
                g_gamma=gamma,
                h0=h0,
                cu_seqlens_dev=cu_seqlens_p,
                scale=None,
                use_ht=True,
                chunk_size=chunk_size,
            )

        output_packed, new_state = jax.shard_map(
            _prefill_fn,
            mesh=layer.mesh,
            in_specs=(
                P(None, "tensor", None),
                P(None, "tensor", None),
                P(None, "tensor", None),
                P("tensor"),
                P(None, "tensor", None, None),
                P(),
                P(),
            ),
            out_specs=(
                P(None, None, "tensor", None),
                P(None, "tensor", None, None),
            ),
            check_vma=False,
        )(q, k, v, slope_sm, h0_sm, scatter_idx, cu_seqlens)

        attn_output = gather_from_packed(output_packed, scatter_idx)
        return attn_output, new_state


__all__ = ["LightningAttnBackend"]
