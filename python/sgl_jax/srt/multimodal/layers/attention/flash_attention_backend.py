from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.multimodal.kernels.flash_attention import SegmentIds, flash_attention
from sgl_jax.srt.multimodal.kernels.varlen_attention import varlen_attention

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch


@dataclasses.dataclass
class VisionAttentionMetadata:
    """Block-diagonal (packed) self-attention layout for the vision tower.

    ``cu_seqlens`` are bucket-shaped cumulative segment boundaries with shape
    ``[num_lanes, K + 1]``: each row starts with zero, holds a lane's cumulative
    segment ends, then repeats the final valid end through the padding slots.
    ``max_seq_len`` is a host-computed upper bound on every positive boundary
    difference. It is static JAX metadata so the varlen backend can select
    compile-time block sizes without inspecting traced boundary values. The
    bound should be stable for an input-shape bucket to avoid recompilation for
    different segment values with the same shapes.
    """

    cu_seqlens: Any
    max_seq_len: int | None = None


jax.tree_util.register_dataclass(
    VisionAttentionMetadata,
    data_fields=["cu_seqlens"],
    meta_fields=["max_seq_len"],
)


def vision_segment_ids_from_cu_seqlens(
    cu_seqlens: jax.Array,
    sequence_length: int,
    *,
    search_method: str = "compare_all",
) -> SegmentIds:
    """Expand bucket-shaped vision boundaries to dense self-attention ids.

    ``cu_seqlens`` has shape ``[B, K + 1]``.  Every row starts with zero,
    contains the cumulative exclusive ends of its real segments, and repeats
    its final valid end through the remaining bucket slots.  An empty lane is
    therefore all zeros.  Tokens at or beyond the final end receive ``-1`` so
    bucket padding cannot attend to real tokens.

    The conversion is deliberately shared by all vision attention backends and
    runs on each batch shard; no cross-device collective is needed.
    """
    if cu_seqlens.ndim != 2 or cu_seqlens.shape[1] < 1:
        raise ValueError(
            "vision cu_seqlens must have shape [batch, boundary_capacity + 1], "
            f"got {cu_seqlens.shape}"
        )
    if not jnp.issubdtype(cu_seqlens.dtype, jnp.integer):
        raise ValueError(f"vision cu_seqlens must be integer, got {cu_seqlens.dtype}")

    positions = jnp.arange(sequence_length, dtype=cu_seqlens.dtype)

    def lane_segment_ids(boundaries):
        # ``side='right'`` assigns a token exactly at a boundary to the next
        # segment.  Repeated tail boundaries only affect padded positions,
        # which are overwritten with -1 below.
        ids = (
            jnp.searchsorted(
                boundaries,
                positions,
                side="right",
                method=search_method,
            )
            - 1
        )
        return jnp.where(positions < boundaries[-1], ids, -1).astype(jnp.int32)

    dense = jax.vmap(lane_segment_ids)(cu_seqlens)
    return SegmentIds(q=dense, kv=dense)


class FlashAttentionBackend(AttentionBackend):
    def __init__(self, mesh, sm_scale=1.0, causal=False, vmem_limit_bytes=128 * 1024 * 1024):
        in_specs = (
            P("data", "tensor", None, None),  # q
            P("data", "tensor", None, None),  # k
            P("data", "tensor", None, None),  # v
            P(),  # segment_ids
        )
        out_specs = P("data", "tensor", None, None)

        def _flash_attention(q, k, v, segment_ids):
            return flash_attention(
                q,
                k,
                v,
                segment_ids=segment_ids,
                sm_scale=sm_scale,
                causal=causal,
                vmem_limit_bytes=vmem_limit_bytes,
            )

        self.jit_flash_attention = jax.jit(
            jax.shard_map(
                _flash_attention, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
            )
        )

    def __call__(
        self,
        q,  # [batch_size, head_nums, req_len, head_dim]
        k,  # [batch_size, head_nums, kv_len, head_dim]
        v,  # [batch_size, head_nums, kv_len, head_dim]
        segment_ids,
    ):
        output = self.jit_flash_attention(q, k, v, segment_ids)
        return output

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        """Init the metadata for a forward pass and return it"""
        return None


class VisionFlashAttentionBackend(AttentionBackend):
    """Batch-sharded segment-flash attention for the in-model VLM ViT.

    Kept SEPARATE from ``FlashAttentionBackend`` (which is head-TP, used by
    ``USPAttention`` for Flux / Wan / Qwen3-Omni audio) so that class stays
    untouched. With replicated ViT weights, batch lanes span both ``data`` and
    ``tensor``. With ``head_tp=True``, batch is sharded on ``data`` and heads on
    ``tensor``. Bucket-shaped cumulative lengths follow the batch sharding and
    are expanded to dense ids locally inside the shard map. On CPU meshes the
    same Pallas kernel runs in interpret mode, so this is the only vision
    backend.
    """

    def __init__(
        self,
        mesh,
        sm_scale=1.0,
        causal=False,
        vmem_limit_bytes: int | None = None,
        head_tp: bool = False,
    ):
        interpret = mesh.devices.flat[0].platform == "cpu"
        if vmem_limit_bytes is None:
            if mesh.devices.flat[0].platform == "tpu":
                from jax.experimental.pallas import tpu as pltpu

                # Keep the Pallas program below the physical VMEM capacity.
                # The old 128 MiB default lets the compiler produce programs
                # that exceed v7x's 64 MiB per-core limit.
                vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
            else:
                vmem_limit_bytes = 128 * 1024 * 1024
        self.vmem_limit_bytes = vmem_limit_bytes
        if head_tp:
            if "tensor" not in mesh.axis_names:
                raise ValueError("head_tp requires a tensor mesh axis")
            batch_axis = "data"
            head_axis = "tensor"
        else:
            batch_axis = ("data", "tensor") if "tensor" in mesh.axis_names else "data"
            head_axis = None
        qkv_spec = P(batch_axis, head_axis, None, None)
        metadata_spec = P(batch_axis, None)
        in_specs = (qkv_spec, qkv_spec, qkv_spec, metadata_spec)
        out_specs = qkv_spec

        def _flash_attention(q, k, v, cu_seqlens):
            segment_ids = vision_segment_ids_from_cu_seqlens(
                cu_seqlens,
                q.shape[2],
                search_method="scan",
            )
            return flash_attention(
                q,
                k,
                v,
                segment_ids=segment_ids,
                sm_scale=sm_scale,
                causal=causal,
                vmem_limit_bytes=self.vmem_limit_bytes,
                interpret=interpret,
            )

        self.jit_flash_attention = jax.jit(
            jax.shard_map(
                _flash_attention, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
            )
        )

    def __call__(self, q, k, v, metadata: VisionAttentionMetadata):
        """Segment-flash attention over batch-leading ``[B, T, heads, head_dim]``.

        Adapts to the kernel's head-leading layout and pads the sequence to the
        tile its block-size path needs, then restores ``[B, T, heads, head_dim]``
        so every vision backend shares one THD in/out contract.
        """
        cu_seqlens = metadata.cu_seqlens
        if q.shape[0] != cu_seqlens.shape[0]:
            raise ValueError(
                f"vision cu_seqlens batch must match q/k/v: {cu_seqlens.shape[0]} != {q.shape[0]}"
            )
        seq_len = q.shape[1]
        if seq_len != k.shape[1]:
            raise ValueError("a single vision cu_seqlens requires equal q and kv lengths")

        # [B, T, H, D] -> [B, H, T, D] for the kernel.
        q, k, v = (jnp.transpose(x, (0, 2, 1, 3)) for x in (q, k, v))

        # The dense kernel's default query tile is 256 tokens.
        alignment = 256
        aligned = max(256, ((seq_len + alignment - 1) // alignment) * alignment)
        pad = aligned - seq_len
        if pad:
            q, k, v = (jnp.pad(x, ((0, 0), (0, 0), (0, pad), (0, 0))) for x in (q, k, v))

        out = self.jit_flash_attention(q, k, v, cu_seqlens)  # [B, H, aligned, D]
        return jnp.transpose(out[:, :, :seq_len], (0, 2, 1, 3))  # -> [B, T, H, D]

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        """Init the metadata for a forward pass and return it"""
        return None


class VisionVarlenAttentionBackend(AttentionBackend):
    """Batch-sharded packed variable-length vision attention.

    Shares the ``[B, T, heads, head_dim]`` + bucket-shaped ``cu_seqlens``
    contract of :class:`VisionFlashAttentionBackend`, but each batch lane maps
    *directly* to the ``varlen_attention`` kernel's packed ``[tokens, heads,
    head_dim]`` layout: no head-major transpose and no sequence padding (the
    kernel reserves its own DMA tail slack). A lane's ``num_seqs`` is the count
    of positive-length segments in its bucket row -- the repeated tail ends
    become zero-length segments the kernel skips. Unlike the flash backend this
    kernel supports GQA, per-call local ``window_size`` and per-head attention
    sinks, so it backs models like MiMo-V2. It is TPU-only.
    """

    def __init__(
        self,
        mesh,
        sm_scale: float = 1.0,
        head_tp: bool = False,
        vmem_limit_bytes: int = 128 * 1024 * 1024,
    ):
        self.mesh = mesh
        self.sm_scale = sm_scale
        self.vmem_limit_bytes = vmem_limit_bytes
        if mesh.devices.flat[0].platform != "tpu":
            raise ValueError("VisionVarlenAttentionBackend requires a TPU mesh")
        if head_tp:
            if "tensor" not in mesh.axis_names:
                raise ValueError("head_tp requires a tensor mesh axis")
            batch_axis = "data"
            self.head_axis = "tensor"
        else:
            batch_axis = ("data", "tensor") if "tensor" in mesh.axis_names else "data"
            self.head_axis = None
        # q/k/v are token-major [B, T, heads, head_dim]: batch on dim 0, heads on dim 2.
        self.qkv_spec = P(batch_axis, None, self.head_axis, None)
        self.cu_spec = P(batch_axis, None)

    def __call__(
        self,
        q,  # [B, T, heads, head_dim]
        k,  # [B, T, kv_heads, head_dim]
        v,  # [B, T, kv_heads, head_dim]
        cu_seqlens,  # int32[B, boundary_capacity + 1] or VisionAttentionMetadata
        attention_sink=None,  # float[heads] or None
        *,
        window_size: tuple[int, int] = (-1, -1),
    ):
        # Accept either a raw cu_seqlens array (MiMo/Omni) or the shared
        # VisionAttentionMetadata (Qwen VL), so this backend is a drop-in for
        # VisionFlashAttentionBackend's (q, k, v, metadata) contract too.
        max_seq_len = None
        if isinstance(cu_seqlens, VisionAttentionMetadata):
            max_seq_len = cu_seqlens.max_seq_len
            cu_seqlens = cu_seqlens.cu_seqlens
        if q.shape[0] != cu_seqlens.shape[0]:
            raise ValueError(
                f"vision cu_seqlens batch must match q/k/v: {cu_seqlens.shape[0]} != {q.shape[0]}"
            )
        if q.shape[1] != k.shape[1]:
            raise ValueError("a single vision cu_seqlens requires equal q and kv lengths")

        def per_lane(lane_q, lane_k, lane_v, lane_cu, lane_sink):
            # A lane's num_seqs is the count of positive-length bucket segments;
            # the repeated tail ends collapse to zero-length segments the kernel skips.
            num_seqs = jnp.sum(jnp.diff(lane_cu) > 0, dtype=jnp.int32).reshape(1)
            return varlen_attention(
                lane_q,
                lane_k,
                lane_v,
                lane_cu,
                num_seqs,
                sm_scale=self.sm_scale,
                window_size=window_size,
                attention_sink=lane_sink,
                max_seq_len=max_seq_len,
                vmem_limit_bytes=self.vmem_limit_bytes,
            )

        # A sink is a shard_map input so it follows the head sharding; without one
        # it is broadcast (in_axes=None) as a plain Python ``None`` per lane.
        over_batch = (0, 0, 0, 0, None)
        if attention_sink is None:

            def sharded(bq, bk, bv, bcu):
                return jax.vmap(per_lane, in_axes=over_batch)(bq, bk, bv, bcu, None)

            in_specs = (self.qkv_spec, self.qkv_spec, self.qkv_spec, self.cu_spec)
            args = (q, k, v, cu_seqlens)
        else:

            def sharded(bq, bk, bv, bcu, sink):
                return jax.vmap(per_lane, in_axes=over_batch)(bq, bk, bv, bcu, sink)

            in_specs = (*(self.qkv_spec,) * 3, self.cu_spec, P(self.head_axis))
            args = (q, k, v, cu_seqlens, attention_sink)

        return jax.shard_map(
            sharded, mesh=self.mesh, in_specs=in_specs, out_specs=self.qkv_spec, check_vma=False
        )(*args)

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        """Init the metadata for a forward pass and return it"""
        return None


def make_vision_attention_backend(
    mesh,
    *,
    sm_scale,
    causal: bool = False,
    head_tp: bool = False,
    use_varlen: bool = False,
) -> AttentionBackend:
    """Build the batch-sharded vision attention backend.

    On TPU, ``use_varlen`` routes the tower through the packed
    ``varlen_attention`` kernel instead of the dense ``flash_attention``
    kernel. Varlen walks each cu_seqlens segment, so window
    layers cost O(sum segment^2) rather than the dense O(T^2). The host-computed
    maximum segment length in :class:`VisionAttentionMetadata` selects the
    v7x-tuned block sizes. CPU meshes use the flash backend's test-only
    interpreter path.
    """
    if use_varlen and mesh.devices.flat[0].platform == "tpu":
        return VisionVarlenAttentionBackend(
            mesh,
            sm_scale=sm_scale,
            head_tp=head_tp,
        )
    return VisionFlashAttentionBackend(
        mesh,
        sm_scale=sm_scale,
        causal=causal,
        head_tp=head_tp,
    )
