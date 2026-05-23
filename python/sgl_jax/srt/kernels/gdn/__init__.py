"""Gated DeltaNet (GDN) reference kernels.

Public entry points:

* :func:`ragged_gated_delta_rule_ref` — token-by-token ``lax.scan`` over a
  packed ragged batch (extend / chunked-prefill).
* :func:`decode_gated_delta_rule_ref` — parallel single-step recurrence
  across the batch (decode fast path).
* :func:`jax_causal_conv1d_prefill` / :func:`jax_causal_conv1d_update` —
  depthwise causal conv1d helpers (ragged prefill + single-token decode).
"""

from sgl_jax.srt.kernels.gdn.gated_delta import (
    decode_gated_delta_rule_ref,
    jax_causal_conv1d_prefill,
    jax_causal_conv1d_update,
    ragged_gated_delta_rule_ref,
)

__all__ = [
    "decode_gated_delta_rule_ref",
    "jax_causal_conv1d_prefill",
    "jax_causal_conv1d_update",
    "ragged_gated_delta_rule_ref",
]
