from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.multimodal.kernels.flash_attention_v2 import (
    TuningConfig,
    flash_attention,
)


class VisionFlashAttentionV2Backend(AttentionBackend):
    """One complete lane per token shard, with optional head tensor parallelism.

    Boundaries are lane-local cumulative lengths; unused slots repeat the final
    valid end. The kernel skips empty segments and this wrapper zeros bucket
    padding. No host compaction or dense segment-id expansion is required.
    """

    def __init__(
        self,
        mesh,
        sm_scale: float = 1.0,
        causal: bool = False,
        head_tp: bool = False,
        tuning: TuningConfig | None = None,
    ):
        device = mesh.devices.flat[0]
        if device.platform != "tpu" or device.device_kind != "TPU7x":
            raise ValueError("VisionFlashAttentionV2Backend requires a TPU v7x mesh")
        if head_tp and "tensor" not in mesh.axis_names:
            raise ValueError("head_tp requires a tensor mesh axis")
        token_axis = "data" if head_tp or "tensor" not in mesh.axis_names else ("data", "tensor")
        qkv_spec = P(token_axis, "tensor" if head_tp else None, None)
        cu_spec = P(token_axis)

        def per_lane(q, k, v, cu_seqlens):
            output = flash_attention(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                causal=causal,
                sm_scale=sm_scale,
                tuning=tuning,
            )
            valid = jnp.arange(q.shape[0]) < cu_seqlens[-1]
            return jnp.where(valid[:, None, None], output, 0)

        self.jit_flash_attention = jax.jit(
            jax.shard_map(
                per_lane,
                mesh=mesh,
                in_specs=(qkv_spec, qkv_spec, qkv_spec, cu_spec),
                out_specs=qkv_spec,
                check_vma=False,
            )
        )

    def __call__(self, q, k, v, cu_seqlens, *, max_seq_len: int | None = None):
        """Attend within segments; tuning uses the static packed token capacity."""
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or cu_seqlens.ndim != 1:
            raise ValueError("vision attention requires [tokens, heads, dim] and flat cu_seqlens")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("a single vision cu_seqlens requires equal q and kv lengths")
        return self.jit_flash_attention(q, k, v, cu_seqlens)

    def get_forward_metadata(self, batch):
        return None
