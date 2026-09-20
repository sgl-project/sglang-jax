"""New KV must survive SWA prefill even outside the last query block's window."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    ragged_paged_attention,
    ref_ragged_paged_attention,
)


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="Requires TPU DMA support")
@pytest.mark.parametrize(
    "sequences,page_size,window,chunk_size,sink,bkv_sz,bkv_csz,reuse_prefix",
    [
        ([(393, 0)], 256, 128, None, None, 256, 256, False),
        ([(393, 273), (137, 512)], 128, 128, None, None, 256, 256, False),
        ([(416, 256)], 256, 128, 416, None, 256, 256, False),
        ([(1, 512), (1, 273)], 256, 128, None, None, 256, 256, False),
        ([(393, 0)], 256, None, None, None, 256, 256, False),
        ([(393, 0)], 256, 128, None, 2.0, 256, 256, False),
        ([(393, 0)], 128, 128, None, None, 256, 128, True),
        ([(393, 273), (137, 512)], 128, 128, None, None, 256, 128, True),
        ([(416, 256)], 128, 128, 416, None, 256, 128, False),
        ([(1, 512), (1, 273)], 128, 128, None, None, 256, 128, False),
        ([(393, 0)], 128, None, None, None, 256, 128, False),
        ([(393, 0)], 128, 128, None, 2.0, 256, 128, False),
        # Key 127 equals q0 - W, so [0, 128) is entirely invisible.
        ([(33, 255)], 128, 128, None, None, 256, 128, False),
        # Key 127 is visible only to query 254, not queries 255 onward.
        ([(33, 254)], 128, 128, None, None, 256, 128, False),
        ([(32, 5664)], 128, 4096, None, None, 2048, 1024, False),
        ([(33, 33023)], 128, 128, None, None, 256, 128, False),
    ],
    ids=[
        "cold",
        "warm-ragged",
        "chunk-prefill",
        "decode",
        "full",
        "sink",
        "cold-split",
        "warm-ragged-split",
        "chunk-prefill-split",
        "decode-split",
        "full-split",
        "sink-split",
        "swa-boundary",
        "first-query-only",
        "v7-tiles",
        "long-position",
    ],
)
def test_kv_writeback(
    sequences, page_size, window, chunk_size, sink, bkv_sz, bkv_csz, reuse_prefix
):
    # Each sequence is (new token count, cached prefix length).
    q_lens, prefixes = np.array(sequences).T
    kv_lens = q_lens + prefixes
    cu_q_lens = np.r_[0, np.cumsum(q_lens)]
    page_counts = (kv_lens + page_size - 1) // page_size
    page_offsets = np.r_[0, np.cumsum(page_counts)]
    num_tokens = cu_q_lens[-1]
    padded_tokens = (num_tokens + 31) // 32 * 32
    num_heads, head_dim = 4, 128
    rng = np.random.default_rng(42)
    page_indices = rng.permutation(np.arange(1, page_offsets[-1] + 1))
    # The first/last pages and unused slots within mapped pages must stay intact.
    cache = np.full((page_offsets[-1] + 2, page_size, 1, 2, head_dim), -7, np.float32)
    expected_cache = cache.copy()
    new_kv = np.zeros((padded_tokens, 2, head_dim), np.float32)
    reference_pages = np.zeros((len(sequences), max(page_counts)), np.int32)
    for i, prefix in enumerate(prefixes):
        kv = rng.integers(-4, 5, size=(kv_lens[i], 2, head_dim)).astype(np.float32)
        if sequences == [(33, 254)]:
            # Make dropping the only visible key in [0, 128) exceed the
            # numerical tolerance, even though it affects only the first query.
            kv[127, 1] = 128
        pages = page_indices[page_offsets[i] : page_offsets[i + 1]]
        reference_pages[i, : len(pages)] = pages
        # Token-level scatter is independent of the kernel's DMA/window tiling.
        positions = np.arange(kv_lens[i])
        physical_pages, slots = pages[positions // page_size], positions % page_size
        expected_cache[physical_pages, slots, 0] = kv
        cache[physical_pages[:prefix], slots[:prefix], 0] = kv[:prefix]
        new_kv[cu_q_lens[i] : cu_q_lens[i + 1]] = kv[prefix:]

    queries = jnp.asarray(
        rng.uniform(-0.25, 0.25, (padded_tokens, num_heads, head_dim)), jnp.bfloat16
    )
    cu_q_lens = jnp.asarray(cu_q_lens, jnp.int32)
    kv_lens = jnp.asarray(kv_lens, jnp.int32)
    attention_args = dict(sm_scale=head_dim**-0.5, sliding_window=window, attention_sink=sink)
    expected_output = ref_ragged_paged_attention(
        queries,
        jnp.asarray(expected_cache[:, :, :, 0, :], jnp.bfloat16),
        jnp.asarray(expected_cache[:, :, :, 1, :], jnp.bfloat16),
        kv_lens,
        jnp.asarray(reference_pages),
        cu_q_lens,
        jnp.array([len(sequences)], jnp.int32),
        **attention_args,
    )
    count = len(sequences)
    decode_count = count if np.all(q_lens == 1) else 0
    prefill_count = count if chunk_size is not None else decode_count
    output, updated_cache = ragged_paged_attention(
        queries,
        jnp.asarray(new_kv[:, 0:1], jnp.bfloat16),
        jnp.asarray(new_kv[:, 1:2], jnp.bfloat16),
        jnp.asarray(cache, jnp.bfloat16),
        kv_lens,
        jnp.asarray(page_indices, jnp.int32),
        cu_q_lens,
        jnp.asarray(page_offsets * page_size, jnp.int32),
        jnp.array([decode_count, prefill_count, count], jnp.int32),
        None,
        chunk_prefill_size=chunk_size,
        d_block_sizes=(1, bkv_sz, 1, bkv_csz),
        p_block_sizes=(32, bkv_sz, 32, bkv_csz),
        m_block_sizes=(32, bkv_sz, 32, bkv_csz),
        **attention_args,
    )
    output, updated_cache_host = jax.device_get((output, updated_cache))

    np.testing.assert_allclose(
        output[:num_tokens].astype(np.float32),
        np.asarray(expected_output, np.float32),
        atol=0.03,
        rtol=0.03,
    )
    # Includes all new KV, the unchanged prefix, page tails and guard pages.
    np.testing.assert_array_equal(updated_cache_host.astype(np.float32), expected_cache)

    if not reuse_prefix:
        return

    # Full attention must now consume the returned cache, including new KV
    # outside the previous SWA window. These cases have room in each last page.
    decode_kv = rng.integers(-4, 5, size=(32, 2, head_dim)).astype(np.float32)
    for i, old_len in enumerate(np.asarray(kv_lens)):
        assert old_len % page_size != 0
        page = reference_pages[i, old_len // page_size]
        expected_cache[page, old_len % page_size, 0] = decode_kv[i]
    decode_queries = jnp.asarray(rng.uniform(-0.25, 0.25, (32, num_heads, head_dim)), jnp.bfloat16)
    decode_q_lens = jnp.arange(count + 1, dtype=jnp.int32)
    expected_decode = ref_ragged_paged_attention(
        decode_queries,
        jnp.asarray(expected_cache[:, :, :, 0, :], jnp.bfloat16),
        jnp.asarray(expected_cache[:, :, :, 1, :], jnp.bfloat16),
        kv_lens + 1,
        jnp.asarray(reference_pages),
        decode_q_lens,
        jnp.array([count], jnp.int32),
        sm_scale=head_dim**-0.5,
    )
    decode_output, decoded_cache = ragged_paged_attention(
        decode_queries,
        jnp.asarray(decode_kv[:, 0:1], jnp.bfloat16),
        jnp.asarray(decode_kv[:, 1:2], jnp.bfloat16),
        updated_cache,
        kv_lens + 1,
        jnp.asarray(page_indices, jnp.int32),
        decode_q_lens,
        jnp.asarray(page_offsets * page_size, jnp.int32),
        jnp.array([count, count, count], jnp.int32),
        None,
        d_block_sizes=(1, bkv_sz, 1, bkv_csz),
        sm_scale=head_dim**-0.5,
    )
    decode_output, decoded_cache = jax.device_get((decode_output, decoded_cache))
    np.testing.assert_allclose(
        decode_output[:count].astype(np.float32),
        np.asarray(expected_decode, np.float32),
        atol=0.03,
        rtol=0.03,
    )
    np.testing.assert_array_equal(decoded_cache.astype(np.float32), expected_cache)
