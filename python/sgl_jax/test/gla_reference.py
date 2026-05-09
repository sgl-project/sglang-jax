"""Compatibility wrapper for the native GLA reference implementation."""

from sgl_jax.srt.kernels.simple_gla.native import naive_gla_decode, naive_gla_prefill

__all__ = ["naive_gla_decode", "naive_gla_prefill"]
