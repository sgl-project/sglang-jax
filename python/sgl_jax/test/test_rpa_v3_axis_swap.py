import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    prepare_inputs,
    prepare_outputs,
)


@pytest.mark.parametrize("max_tokens", [32, 64])
@pytest.mark.parametrize("actual_num_kv_heads", [2, 4])
@pytest.mark.parametrize("num_heads_per_kv", [1, 4])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_rpa_v3_prepare_inputs_and_outputs_axis_swap(
    max_tokens: int,
    actual_num_kv_heads: int,
    num_heads_per_kv: int,
    head_dim: int,
):
    actual_num_q_heads = actual_num_kv_heads * num_heads_per_kv
    actual_num_q_heads_per_kv_head = num_heads_per_kv

    rng = np.random.default_rng(42)
    q_np = rng.standard_normal((max_tokens, actual_num_q_heads, head_dim)).astype(np.float32)
    k_np = rng.standard_normal((max_tokens, actual_num_kv_heads, head_dim)).astype(np.float32)
    v_np = rng.standard_normal((max_tokens, actual_num_kv_heads, head_dim)).astype(np.float32)

    q = jnp.asarray(q_np, dtype=jnp.bfloat16)
    k = jnp.asarray(k_np, dtype=jnp.bfloat16)
    v = jnp.asarray(v_np, dtype=jnp.bfloat16)

    # 1. Default / Legacy path: axis swap emitted
    q_legacy, kv_legacy, sink_legacy = prepare_inputs(q, k, v, fuse_non_tiling_axis_swap=False)
    assert q_legacy.shape[0] == actual_num_kv_heads
    assert q_legacy.shape[1] == max_tokens

    # 2. Fused layout path: axis swap absorbed in strided DMA layout
    q_fused, kv_fused, sink_fused = prepare_inputs(q, k, v, fuse_non_tiling_axis_swap=True)
    assert q_fused.shape[0] == max_tokens
    assert q_fused.shape[1] == actual_num_kv_heads

    # Permuted layout equivalence
    np.testing.assert_allclose(
        np.asarray(q_legacy),
        np.asarray(jnp.swapaxes(q_fused, 0, 1)),
        rtol=1e-5,
        atol=1e-5,
    )

    # 3. Output preparation roundtrip
    out_legacy = prepare_outputs(
        q_legacy,
        actual_num_q_heads_per_kv_head,
        head_dim,
        fuse_non_tiling_axis_swap=False,
    )
    out_fused = prepare_outputs(
        q_fused,
        actual_num_q_heads_per_kv_head,
        head_dim,
        fuse_non_tiling_axis_swap=True,
    )

    assert out_legacy.shape == (max_tokens, actual_num_q_heads, head_dim)
    assert out_fused.shape == (max_tokens, actual_num_q_heads, head_dim)
    np.testing.assert_allclose(
        np.asarray(out_legacy),
        np.asarray(out_fused),
        rtol=1e-5,
        atol=1e-5,
    )
