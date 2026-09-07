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
    "sequences,page_size,window,chunk_size,sink",
    [
        ([(393, 0)], 256, 128, None, None),
        ([(393, 273), (137, 512)], 128, 128, None, None),
        ([(416, 256)], 256, 128, 416, None),
        ([(1, 512), (1, 273)], 256, 128, None, None),
        ([(393, 0)], 256, None, None, None),
        ([(393, 0)], 256, 128, None, 2.0),
    ],
    ids=["cold", "warm-ragged", "chunk-prefill", "decode", "full", "sink"],
)
def test_kv_writeback(sequences, page_size, window, chunk_size, sink):
    # Each sequence is (new token count, cached prefix length). Use multiple
    # query blocks, unaligned boundaries and a non-contiguous physical page map.
    q_lens = [q_len for q_len, _ in sequences]
    kv_lens = [q_len + prefix for q_len, prefix in sequences]
    page_counts = [(length + page_size - 1) // page_size for length in kv_lens]
    capacities = np.array(page_counts) * page_size
    num_pages = sum(page_counts)
    num_tokens = sum(q_lens)
    padded_tokens = (num_tokens + 31) // 32 * 32
    num_heads, head_dim = 4, 128
    rng = np.random.default_rng(42)
    page_indices = rng.permutation(np.arange(1, num_pages + 1))
    # The first/last pages and unused slots within mapped pages must stay intact.
    cache = np.full((num_pages + 2, page_size, 1, 2, head_dim), -7, np.float32)
    expected_cache = cache.copy()
    keys = np.zeros((padded_tokens, 1, head_dim), np.float32)
    values = np.zeros_like(keys)
    reference_pages = np.zeros((len(sequences), max(page_counts)), np.int32)
    token_offset = page_offset = 0
    for seq_idx, ((q_len, prefix), kv_len, page_count) in enumerate(
        zip(sequences, kv_lens, page_counts)
    ):
        kv = rng.integers(-4, 5, size=(kv_len, 2, head_dim)).astype(np.float32)
        pages = page_indices[page_offset : page_offset + page_count]
        reference_pages[seq_idx, :page_count] = pages
        # Token-level scatter is independent of the kernel's DMA/window tiling.
        for position in range(kv_len):
            physical_page = pages[position // page_size]
            slot = position % page_size
            expected_cache[physical_page, slot, 0] = kv[position]
            if position < prefix:
                cache[physical_page, slot, 0] = kv[position]
        keys[token_offset : token_offset + q_len, 0] = kv[prefix:, 0]
        values[token_offset : token_offset + q_len, 0] = kv[prefix:, 1]
        token_offset += q_len
        page_offset += page_count

    queries = jnp.asarray(
        rng.uniform(-0.25, 0.25, (padded_tokens, num_heads, head_dim)), jnp.bfloat16
    )
    cu_q_lens = jnp.asarray(np.cumsum([0] + q_lens), jnp.int32)
    kv_lens = jnp.asarray(kv_lens, jnp.int32)
    expected_output = ref_ragged_paged_attention(
        queries,
        jnp.asarray(expected_cache[:, :, :, 0, :], jnp.bfloat16),
        jnp.asarray(expected_cache[:, :, :, 1, :], jnp.bfloat16),
        kv_lens,
        jnp.asarray(reference_pages),
        cu_q_lens,
        jnp.array([len(sequences)], jnp.int32),
        sm_scale=head_dim**-0.5,
        sliding_window=window,
        attention_sink=sink,
    )
    count = len(sequences)
    decode_count = count if all(q_len == 1 for q_len in q_lens) else 0
    prefill_count = count if chunk_size is not None else decode_count
    output, updated_cache = ragged_paged_attention(
        queries,
        jnp.asarray(keys, jnp.bfloat16),
        jnp.asarray(values, jnp.bfloat16),
        jnp.asarray(cache, jnp.bfloat16),
        kv_lens,
        jnp.asarray(page_indices, jnp.int32),
        cu_q_lens,
        jnp.asarray(np.cumsum(np.r_[0, capacities]), jnp.int32),
        jnp.array([decode_count, prefill_count, count], jnp.int32),
        None,
        sliding_window=window,
        sm_scale=head_dim**-0.5,
        chunk_prefill_size=chunk_size,
        attention_sink=sink,
        d_block_sizes=(1, 256, 1, 256),
        p_block_sizes=(32, 256, 32, 256),
        m_block_sizes=(32, 256, 32, 256),
    )
    output, updated_cache = jax.device_get((output, updated_cache))

    np.testing.assert_allclose(
        output[:num_tokens].astype(np.float32),
        np.asarray(expected_output, np.float32),
        atol=0.03,
        rtol=0.03,
    )
    # Includes all new KV, the unchanged prefix, page tails and guard pages.
    np.testing.assert_array_equal(updated_cache.astype(np.float32), expected_cache)
