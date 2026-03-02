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
