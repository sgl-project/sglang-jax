"""CPU checks for the SWA compute guard; DMA/kernel accuracy needs a TPU."""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.ragged_paged_attention import ragged_paged_attention_v3 as rpa


@pytest.mark.parametrize(
    "q_start,q_count,kv_start,kv_count,window,visible",
    [
        (5664, 32, 0, 1024, 4096, False),
        (5664, 32, 1024, 1024, 4096, True),
        (4223, 32, 0, 128, 4096, False),  # Last key == q0 - window.
        (4222, 32, 0, 128, 4096, True),  # Only earliest query sees last key.
        (4224, 1, 0, 128, 4096, False),
        (384, 9, 0, 256, 128, False),  # Last, partial query block.
        (384, 9, 256, 256, 128, True),
        (40960, 32, 36864, 1024, 4096, True),  # No int16 wrap.
        (40960, 32, 35840, 1024, 4096, False),
        (31, 1, 0, 32, 1, True),
    ],
)
def test_swa_subblock_visibility(q_start, q_count, kv_start, kv_count, window, visible):
    # Enumerate token pairs, independently of the block-boundary inequality.
    queries = np.arange(q_start, q_start + q_count)[:, None]
    keys = np.arange(kv_start, kv_start + kv_count)[None, :]
    token_mask = (keys <= queries) & (queries - keys < window)
    assert bool(token_mask.any()) == visible
    predicate = jax.jit(rpa._swa_subblock_has_visible_keys, static_argnums=(2, 3))
    assert bool(predicate(jnp.int32(q_start), jnp.int32(kv_start), kv_count, window)) == visible


@pytest.mark.parametrize("bkv_sz,bkv_csz", [(128, 128), (256, 128), (2048, 1024)])
@pytest.mark.parametrize("window", [1, 128, 4096])
def test_swa_guard_never_discards_a_visible_pair(bkv_sz, bkv_csz, window):
    # Sweep both sides of compute/DMA boundaries, with short query/KV tails.
    starts = np.array([0, 1, 127, 128, 129, 255, 256, 1023, 1024, 40960])
    for q_start in starts:
        q_count = 9
        kv_len = q_start + q_count
        first_dma = max(q_start - window, 0) // bkv_sz
        for dma in range(first_dma, (kv_len + bkv_sz - 1) // bkv_sz):
            for offset in range(0, bkv_sz, bkv_csz):
                k_start = dma * bkv_sz + offset
                if k_start >= kv_len:
                    continue  # Existing right-edge guard.
                keys = np.arange(k_start, min(k_start + bkv_csz, kv_len))[None, :]
                queries = np.arange(q_start, q_start + q_count)[:, None]
                visible = ((keys <= queries) & (queries - keys < window)).any()
                keep = rpa._swa_subblock_has_visible_keys(q_start, k_start, bkv_csz, window)
                assert keep or not visible
                # Full subblocks have an exact, rather than conservative, decision.
                if k_start + bkv_csz <= kv_len:
                    assert bool(keep) == bool(visible)


@pytest.mark.parametrize(
    "window,causal,custom_mask,mask_value,sink,uses_guard",
    [
        (128, 1, False, rpa.DEFAULT_MASK_VALUE, None, True),
        (128, 1, False, rpa.DEFAULT_MASK_VALUE, 2.0, True),
        (None, 1, False, rpa.DEFAULT_MASK_VALUE, None, False),
        (128, 0, True, rpa.DEFAULT_MASK_VALUE, None, False),
        (128, 0, False, rpa.DEFAULT_MASK_VALUE, None, False),
        (128, 1, False, 0.0, None, False),
    ],
    ids=["swa", "sink", "full", "custom-mask", "noncausal", "finite-mask-value"],
)
def test_kernel_traces_swa_guard_only_on_supported_paths(
    window, causal, custom_mask, mask_value, sink, uses_guard
):
    # Trace the actual Pallas body, without compiling/executing TPU operations.
    # This checks wiring and static fallbacks in addition to scalar arithmetic.
    q = jax.ShapeDtypeStruct((32, 4, 128), jnp.bfloat16)
    kv = jax.ShapeDtypeStruct((32, 1, 128), jnp.bfloat16)
    cache = jax.ShapeDtypeStruct((4, 128, 1, 2, 128), jnp.bfloat16)

    def run(q, k, v, cache):
        return rpa.ragged_paged_attention.__wrapped__(
            q,
            k,
            v,
            cache,
            jnp.array([393], jnp.int32),
            jnp.arange(4, dtype=jnp.int32),
            jnp.array([0, 9], jnp.int32),
            jnp.array([0, 512], jnp.int32),
            jnp.array([0, 0, 1], jnp.int32),
            jnp.zeros((32, 1, 512), jnp.int32) if custom_mask else None,
            sliding_window=window,
            causal=causal,
            mask_value=mask_value,
            attention_sink=sink,
            d_block_sizes=(1, 512, 1, 128),
            p_block_sizes=(32, 512, 16, 128),
            m_block_sizes=(32, 512, 16, 128),
            vmem_limit_bytes=64 * 1024 * 1024,
        )

    with (
        mock.patch.object(rpa, "get_tpu_version", return_value=7),
        mock.patch.object(
            rpa, "_swa_subblock_has_visible_keys", wraps=rpa._swa_subblock_has_visible_keys
        ) as guard,
    ):
        jax.make_jaxpr(run)(q, kv, kv, cache)
        assert bool(guard.call_count) == uses_guard
