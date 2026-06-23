"""Grouped top-k Pallas kernels.

`grouped_topk_pallas` is the official (v1) stable-tiebreak kernel; older variants (the original
argmax inference kernel and the ids-only training kernel) live under `legacy/`.
"""

from sgl_jax.srt.kernels.grouped_topk.v1 import grouped_topk_pallas

__all__ = ["grouped_topk_pallas"]
