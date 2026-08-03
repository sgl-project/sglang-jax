"""Gated DeltaNet (GDN) kernels.

Public entry points:

* :func:`chunked_gated_delta_rule_jax` — chunkwise-parallel gated delta-rule
  recurrence in pure JAX (extend / chunked-prefill).
* :func:`ragged_gated_delta_rule_ref` — token-by-token ``lax.scan`` over a
  packed ragged batch (reference oracle).
* :func:`decode_gated_delta_rule_ref` — parallel single-step recurrence
  across the batch (decode fast path).
* :func:`jax_causal_conv1d_prefill` / :func:`jax_causal_conv1d_update` —
  depthwise causal conv1d helpers (ragged prefill + single-token decode).
"""

from sgl_jax.srt.kernels.gdn.gated_delta import (
    chunked_gated_delta_rule_jax,
    decode_gated_delta_rule_ref,
    jax_causal_conv1d_prefill,
    jax_causal_conv1d_update,
    ragged_gated_delta_rule_ref,
)

__all__ = [
    "chunked_gated_delta_rule_jax",
    "decode_gated_delta_rule_ref",
    "jax_causal_conv1d_prefill",
    "jax_causal_conv1d_update",
    "ragged_gated_delta_rule_ref",
]
