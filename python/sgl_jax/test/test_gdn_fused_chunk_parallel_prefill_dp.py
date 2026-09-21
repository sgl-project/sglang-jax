"""DP sharding contract for fused chunk-parallel GDN prefill."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sgl_jax.srt.kernels.gdn.fused_chunk_parallel_adapter as adapter
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackendMetadata,
)

DP = 4
N_KQ = 1
N_V = 1
D_K = 128
D_V = 128
KERNEL_SIZE = 3
DIM = 2 * N_KQ * D_K + N_V * D_V
TOKENS_PER_RANK = 2
LOCAL_POOL_SIZE = 2

pytestmark = pytest.mark.skipif(
    not any(device.platform == "tpu" for device in jax.local_devices()),
    reason="the fused chunk-parallel DP contract requires real TPU hardware",
)


def _fake_fused_kernel(
    qkv,
    b,
    a,
    conv_state,
    recurrent_state,
    conv_weight,
    conv_bias,
    a_log,
    dt_bias,
    cu_seqlens,
    state_indices,
    distribution,
    seq_lens,
    **kwargs,
):
    del b, a, conv_weight, conv_bias, a_log, dt_bias, distribution, seq_lens, kwargs
    assert qkv.shape[0] == TOKENS_PER_RANK
    assert cu_seqlens.shape == (2,)
    assert state_indices.shape == (1,)

    rank_value = qkv[0, 0].astype(conv_state.dtype)
    new_conv = conv_state.at[state_indices].set(rank_value)
    new_recurrent = recurrent_state.at[state_indices].set(rank_value)
    output = jnp.full((qkv.shape[0], N_V * D_V), rank_value, dtype=qkv.dtype)
    return (new_conv, new_recurrent), output


def test_adapter_executes_with_four_data_parallel_shards(monkeypatch):
    assert jax.device_count() == DP
    devices = np.asarray(jax.devices()).reshape(DP, 1)
    mesh = jax.sharding.Mesh(devices, ("data", "tensor"))
    monkeypatch.setattr(adapter, "_fused_chunk_parallel_kernel", _fake_fused_kernel)

    rank_values = jnp.repeat(jnp.arange(1, DP + 1, dtype=jnp.bfloat16), TOKENS_PER_RANK)
    mixed_qkv = jnp.broadcast_to(rank_values[:, None], (DP * TOKENS_PER_RANK, DIM))
    metadata = LinearRecurrentAttnBackendMetadata(
        cu_q_lens=jnp.tile(jnp.asarray([0, TOKENS_PER_RANK], dtype=jnp.int32), DP),
        recurrent_indices=jnp.ones((DP,), dtype=jnp.int32),
        has_initial_state=jnp.zeros((DP,), dtype=jnp.bool_),
    )
    backend = SimpleNamespace(
        forward_metadata=metadata,
        mesh=mesh,
        num_k_heads=N_KQ,
        num_v_heads=N_V,
        head_k_dim=D_K,
        head_v_dim=D_V,
        conv_kernel_size=KERNEL_SIZE,
    )

    with jax.sharding.set_mesh(mesh):
        output, new_conv, new_recurrent = adapter.fused_chunk_parallel_prefill(
            backend,
            mixed_qkv,
            jnp.zeros((DP * LOCAL_POOL_SIZE, DIM, KERNEL_SIZE - 1), dtype=jnp.bfloat16),
            jnp.zeros((DP * LOCAL_POOL_SIZE, N_V, D_K, D_V), dtype=jnp.float32),
            jnp.zeros((DP * TOKENS_PER_RANK, N_V), dtype=jnp.bfloat16),
            jnp.zeros((DP * TOKENS_PER_RANK, N_V), dtype=jnp.bfloat16),
            jnp.zeros((DIM, KERNEL_SIZE), dtype=jnp.bfloat16),
            jnp.zeros((N_V,), dtype=jnp.float32),
            jnp.zeros((N_V,), dtype=jnp.float32),
            jnp.full((DP,), TOKENS_PER_RANK, dtype=jnp.int32),
        )
        jax.block_until_ready((output, new_conv, new_recurrent))

    output = np.asarray(output)
    new_conv = np.asarray(new_conv)
    new_recurrent = np.asarray(new_recurrent)
    for rank in range(DP):
        expected = rank + 1
        token_slice = slice(rank * TOKENS_PER_RANK, (rank + 1) * TOKENS_PER_RANK)
        np.testing.assert_array_equal(output[token_slice], expected)

        dummy_slot = rank * LOCAL_POOL_SIZE
        running_slot = dummy_slot + 1
        np.testing.assert_array_equal(new_conv[dummy_slot], 0)
        np.testing.assert_array_equal(new_recurrent[dummy_slot], 0)
        np.testing.assert_array_equal(new_conv[running_slot], expected)
        np.testing.assert_array_equal(new_recurrent[running_slot], expected)
