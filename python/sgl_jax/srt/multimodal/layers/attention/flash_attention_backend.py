from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.multimodal.kernels.flash_attention import SegmentIds, flash_attention
from sgl_jax.srt.multimodal.kernels.varlen_attention import varlen_attention

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch


def _resolve_vision_vmem_limit_bytes(mesh, vmem_limit_bytes: int | None) -> int:
    if vmem_limit_bytes is not None:
        return vmem_limit_bytes
    if mesh.devices.flat[0].platform == "tpu":
        from jax.experimental.pallas import tpu as pltpu

        # Keep Pallas programs below the physical per-core VMEM capacity.
        # A fixed 128 MiB limit exceeds the 64 MiB capacity of v7x.
        return int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    return 128 * 1024 * 1024


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
    """Segment-flash attention over flat ``[tokens, heads, head_dim]`` inputs.

    Each token shard contains one complete lane. Its flat cumulative lengths
    are expanded to segment ids locally. The dense kernel's singleton batch
    and head-major layout are introduced only inside the shard map. CPU meshes
    run the Pallas kernel in interpret mode.
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
        self.vmem_limit_bytes = _resolve_vision_vmem_limit_bytes(mesh, vmem_limit_bytes)
        if head_tp:
            if "tensor" not in mesh.axis_names:
                raise ValueError("head_tp requires a tensor mesh axis")
            batch_axis = "data"
            head_axis = "tensor"
        else:
            batch_axis = ("data", "tensor") if "tensor" in mesh.axis_names else "data"
            head_axis = None
        qkv_spec = P(batch_axis, head_axis, None)
        metadata_spec = P(batch_axis)
        in_specs = (qkv_spec, qkv_spec, qkv_spec, metadata_spec)
        out_specs = qkv_spec

        def _flash_attention(q, k, v, cu_seqlens):
            seq_len = q.shape[0]
            # The dense kernel requires [1, heads, tokens, head_dim] and a
            # sequence capacity aligned to its query tile.
            q, k, v = (jnp.transpose(x, (1, 0, 2))[None] for x in (q, k, v))
            aligned = max(256, ((seq_len + 255) // 256) * 256)
            pad = aligned - seq_len
            if pad:
                q, k, v = (jnp.pad(x, ((0, 0), (0, 0), (0, pad), (0, 0))) for x in (q, k, v))
            segment_ids = vision_segment_ids_from_cu_seqlens(
                cu_seqlens[None],
                aligned,
                search_method="scan",
            )
            out = flash_attention(
                q,
                k,
                v,
                segment_ids=segment_ids,
                sm_scale=sm_scale,
                causal=causal,
                vmem_limit_bytes=self.vmem_limit_bytes,
                interpret=interpret,
            )
            return jnp.transpose(out[0, :, :seq_len], (1, 0, 2))

        self.jit_flash_attention = jax.jit(
            jax.shard_map(
                _flash_attention, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
            )
        )

    def __call__(self, q, k, v, cu_seqlens: jax.Array, *, max_seq_len: int | None = None):
        """Attend within lane-local segments; ``max_seq_len`` is varlen-only."""
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or cu_seqlens.ndim != 1:
            raise ValueError("vision attention requires [tokens, heads, dim] and flat cu_seqlens")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("a single vision cu_seqlens requires equal q and kv lengths")
        return self.jit_flash_attention(q, k, v, cu_seqlens)

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        """Init the metadata for a forward pass and return it"""
        return None


class VisionVarlenAttentionBackend(AttentionBackend):
    """TPU variable-length attention over flat token and boundary buffers.

    Each token shard maps directly to the kernel's ``[tokens, heads, head_dim]``
    layout. Lane-local cumulative lengths retain repeated tail ends; those
    zero-length segments are skipped. No batch axis or vmap is needed.
    """

    def __init__(
        self,
        mesh,
        sm_scale: float = 1.0,
        head_tp: bool = False,
        vmem_limit_bytes: int | None = None,
    ):
        self.mesh = mesh
        self.sm_scale = sm_scale
        if mesh.devices.flat[0].platform != "tpu":
            raise ValueError("VisionVarlenAttentionBackend requires a TPU mesh")
        self.vmem_limit_bytes = _resolve_vision_vmem_limit_bytes(mesh, vmem_limit_bytes)
        if head_tp:
            if "tensor" not in mesh.axis_names:
                raise ValueError("head_tp requires a tensor mesh axis")
            batch_axis = "data"
            self.head_axis = "tensor"
        else:
            batch_axis = ("data", "tensor") if "tensor" in mesh.axis_names else "data"
            self.head_axis = None
        self.qkv_spec = P(batch_axis, self.head_axis, None)
        self.cu_spec = P(batch_axis)

    def __call__(
        self,
        q,  # [tokens, heads, head_dim]
        k,  # [tokens, kv_heads, head_dim]
        v,  # [tokens, kv_heads, head_dim]
        cu_seqlens: jax.Array,  # Flat lane-local cumulative lengths.
        attention_sink=None,  # float[heads] or None
        *,
        window_size: tuple[int, int] = (-1, -1),
        max_seq_len: int | None = None,
    ):
        """Attend within segments, using a static length bound for kernel tuning.

        ``max_seq_len`` should be a Python int stable within each input-shape
        bucket. If omitted, the kernel uses the packed Q capacity as its bound.
        """
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or cu_seqlens.ndim != 1:
            raise ValueError("vision attention requires [tokens, heads, dim] and flat cu_seqlens")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
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

        # A sink follows the head sharding; each shard already contains one lane.
        if attention_sink is None:

            def sharded(q, k, v, cu):
                return per_lane(q, k, v, cu, None)

            in_specs = (self.qkv_spec, self.qkv_spec, self.qkv_spec, self.cu_spec)
            args = (q, k, v, cu_seqlens)
        else:

            def sharded(q, k, v, cu, sink):
                return per_lane(q, k, v, cu, sink)

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
    """Build vision attention over flat tokens sharded into complete lanes.

    On TPU, ``use_varlen`` routes the tower through the packed
    ``varlen_attention`` kernel instead of the dense ``flash_attention``
    kernel. Varlen walks each cu_seqlens segment, so window
    layers cost O(sum segment^2) rather than the dense O(T^2). The static
    ``max_seq_len`` upper bound passed to the backend selects the
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
