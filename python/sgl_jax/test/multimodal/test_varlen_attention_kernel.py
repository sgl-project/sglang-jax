"""TPU accuracy tests for the packed variable-length attention kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.multimodal.kernels.varlen_attention import (
    ref_varlen_attention,
    varlen_attention,
)

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="varlen_attention is a TPU-only Pallas kernel",
)


def test_varlen_attention_matches_jax_reference():
    """Cover the BF16 MHA fast path and the GQA fallback on ragged inputs."""
    cases = (
        ("mha_full", (63, 129, 257), 16, 16, (-1, -1)),
        ("gqa_causal", (33, 129), 8, 2, (-1, 0)),
    )

    for case_index, (name, seq_lens, q_heads, kv_heads, window_size) in enumerate(cases):
        total_tokens = sum(seq_lens)
        keys = jax.random.split(jax.random.key(case_index), 3)
        q = jax.random.normal(keys[0], (total_tokens, q_heads, 128), dtype=jnp.bfloat16)
        k = jax.random.normal(keys[1], (total_tokens, kv_heads, 128), dtype=jnp.bfloat16)
        v = jax.random.normal(keys[2], (total_tokens, kv_heads, 128), dtype=jnp.bfloat16)
        cu_seqlens = jnp.asarray((0, *np.cumsum(seq_lens)), dtype=jnp.int32)
        num_seqs = jnp.asarray([len(seq_lens)], dtype=jnp.int32)
        kwargs = {
            "window_size": window_size,
            "sm_scale": 128**-0.5,
        }

        expected = ref_varlen_attention(q, k, v, cu_seqlens, num_seqs, **kwargs)
        actual = varlen_attention(
            q,
            k,
            v,
            cu_seqlens,
            num_seqs,
            max_seq_len=max(seq_lens),
            **kwargs,
        )
        jax.block_until_ready(actual)

        actual_np = np.asarray(actual, dtype=np.float32)
        expected_np = np.asarray(expected, dtype=np.float32)
        max_abs_error = float(np.max(np.abs(actual_np - expected_np)))
        print(f"{name}: max_abs_error={max_abs_error:.8f}")
        np.testing.assert_allclose(
            actual_np,
            expected_np,
            rtol=2e-2,
            atol=1e-2,
            err_msg=name,
        )
