"""In-place Pallas kernels for recurrent-state copy-on-write."""

from sgl_jax.srt.kernels.h0_clone.kernel import clone_slots_inplace

__all__ = ["clone_slots_inplace"]
