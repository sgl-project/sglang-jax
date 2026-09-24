"""TT block verification and noncausal drafting against a dense reference."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from sgl_jax.srt.hardware_backend.tt.attention.tt_backend import (
    TTAttention,
    TTTokenToKVPool,
)
from sgl_jax.srt.layers.radix_attention import AttentionType, RadixAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

PAGE = 32
BLOCK = 16
# The third block stays inside one page. The fourth request is padding. With
# 64 query rows, KV writes and attention both span two launches.
PREFIXES = [30, 65, 3, 0]
TABLES = [[5, 2], [9, 7, 11], [4], []]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("trace", [False, True])
def test_block_attention(causal, trace):
    if jax.default_backend() != "tt":
        pytest.skip("requires the TT plugin")
    device = jax.devices()[0]
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        seq_lens=np.array(PREFIXES, np.int32),
        logits_indices_selector=np.array([0, 1, 2], np.int32),
        spec_info_padded=SimpleNamespace(draft_token_num=BLOCK, custom_mask=None),
    )
    backend = TTAttention(PAGE, mesh)
    pages = np.pad(np.concatenate(TABLES).astype(np.int32), (0, 10))
    backend.forward_metadata = backend.get_eagle_forward_metadata(batch, page_indices=pages)
    pool = TTTokenToKVPool(16 * PAGE, PAGE, jnp.bfloat16, 8, 128, 1, mesh)
    layer = RadixAttention(
        32,
        128,
        128**-0.5,
        8,
        0,
        attn_type=AttentionType.DECODER if causal else AttentionType.ENCODER_ONLY,
    )
    positions = np.full(len(PREFIXES) * BLOCK, -1, np.int32)
    locations = np.full(len(PREFIXES) * BLOCK, -1, np.int32)
    for seq, (prefix, table) in enumerate(zip(PREFIXES[:3], TABLES)):
        rows = slice(seq * BLOCK, (seq + 1) * BLOCK)
        positions[rows] = prefix + np.arange(BLOCK)
        locations[rows] = np.array(table)[positions[rows] // PAGE] * PAGE + positions[rows] % PAGE

    rng = np.random.default_rng(2)
    rows = len(positions)
    q = np.asarray(rng.normal(size=(rows, 32, 128)), dtype=jnp.bfloat16)
    key_cache, value_cache = (
        np.asarray(rng.normal(size=(17, 8, PAGE, 128)), dtype=jnp.bfloat16) for _ in range(2)
    )
    pool.kv_buffer[0] = tuple(jax.device_put(x, device) for x in (key_cache, value_cache))

    def forward(q, k, v, pool, backend, positions, locations):
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY, positions=positions, out_cache_loc=locations
        )
        return backend(q, k, v, layer, forward_batch, pool)

    run = jax.jit(
        forward,
        donate_argnums=(3,),
        compiler_options={"optimization_level": "O1", "enable_trace": str(trace).lower()},
    )
    # The second iteration replays the trace with new K/V values.
    for _ in range(2):
        k, v = (np.asarray(rng.normal(size=(rows, 8, 128)), dtype=jnp.bfloat16) for _ in range(2))
        output, caches = run(
            *[jax.device_put(x, device) for x in (q, k, v)],
            pool,
            backend,
            jax.device_put(positions, device),
            jax.device_put(locations, device),
        )
        for row, location in enumerate(locations):
            if location >= 0:
                key_cache[location // PAGE, :, location % PAGE] = k[row]
                value_cache[location // PAGE, :, location % PAGE] = v[row]
        for actual, expected in zip(caches, (key_cache, value_cache)):
            np.testing.assert_array_equal(np.asarray(actual), expected)

        expected = []
        for row in range(3 * BLOCK):
            seq = row // BLOCK
            end = positions[row] + 1 if causal else PREFIXES[seq] + BLOCK
            keys, values = (
                np.repeat(
                    cache[TABLES[seq]].transpose(0, 2, 1, 3).reshape(-1, 8, 128)[:end],
                    4,
                    axis=1,
                ).astype(np.float32)
                for cache in (key_cache, value_cache)
            )
            scores = np.einsum("hd,thd->ht", q[row].astype(np.float32), keys) * 128**-0.5
            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights /= weights.sum(axis=-1, keepdims=True)
            expected.append(np.einsum("ht,thd->hd", weights, values).reshape(-1))
        # BF16 kernels leave isolated errors near 0.1 on short contexts. A
        # wrong attention extent moves every row, so also bound the mean.
        error = np.abs(np.asarray(output)[: 3 * BLOCK].astype(np.float32) - expected)
        assert error.max() < 0.15 and error.mean() < 0.01, (error.max(), error.mean())
        pool.kv_buffer[0] = caches
