"""Official grouped top-k Pallas kernel (stable lowest-index tie-break)."""

from sgl_jax.srt.kernels.grouped_topk.v1.kernel import grouped_topk_pallas

__all__ = ["grouped_topk_pallas"]
