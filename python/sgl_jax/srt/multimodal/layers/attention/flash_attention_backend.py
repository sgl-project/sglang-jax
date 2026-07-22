import jax
import jax.experimental.pallas as pl
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.multimodal.kernels.flash_attention import flash_attention


def align_to(x, a):
    return pl.cdiv(x, a) * a


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
    untouched. In the default (replicated) mode the ViT weights are replicated
    while the vision batch is sharded over both mesh axes; heads / T / head_dim
    remain unsharded. With ``head_tp=True`` the batch is sharded over ``"data"``
    only and the head axis over ``"tensor"`` (the weight-TP layout, matching the
    sibling ``FlashAttentionBackend`` specs). Wraps the SAME pallas segment-flash
    kernel -- only the shard specs differ. Reusable across in-model VLM ViTs
    (Qwen2.5-VL, future Qwen3-Omni, ...).
    """

    def __init__(
        self,
        mesh,
        sm_scale=1.0,
        causal=False,
        vmem_limit_bytes=128 * 1024 * 1024,
        head_tp: bool = False,
    ):
        if head_tp:
            batch_axis = "data"
            head_axis = "tensor"
        else:
            batch_axis = ("data", "tensor") if "tensor" in mesh.axis_names else "data"
            head_axis = None
        qkv_spec = P(batch_axis, head_axis, None, None)  # [batch, heads, T, head_dim]
        seg_spec = P(batch_axis, None)
        in_specs = (qkv_spec, qkv_spec, qkv_spec, seg_spec)
        out_specs = qkv_spec

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

    def __call__(self, q, k, v, segment_ids):
        return self.jit_flash_attention(q, k, v, segment_ids)

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        """Init the metadata for a forward pass and return it"""
        return None
