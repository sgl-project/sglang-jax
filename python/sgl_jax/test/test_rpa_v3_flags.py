"""RPA v3 kernel switches and stage gating.

predicate_cache_dma, elide_single_tile_loops and compact_span_mask default to
on; each case runs with them on and off and checks both against the reference
and the KV cache writeback. decode_only / skip_decode drop the stages that a
decode-only or decode-free batch does not need and must not change results.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    ragged_paged_attention,
    ref_ragged_paged_attention,
)

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Requires TPU DMA support")

NUM_HEADS, HEAD_DIM = 4, 128
FLAGS_OFF = dict(predicate_cache_dma=False, elide_single_tile_loops=False, compact_span_mask=False)


def _tree_mask(rng, q_len, prefix):
    """Speculative-tree-like mask: the prefix, the token itself and an ancestor run."""
    mask = np.zeros((q_len, q_len + prefix), np.int32)
    mask[:, :prefix] = 1
    for j in range(q_len):
        mask[j, prefix : prefix + rng.integers(0, j + 1)] = 1
        mask[j, prefix + j] = 1
    return mask


def _make_batch(sequences, page_size, *, tree_mask=False, seed=0):
    """Each sequence is (new token count, cached prefix length)."""
    q_lens, prefixes = (np.asarray(x) for x in zip(*sequences))
    kv_lens = q_lens + prefixes
    cu_q_lens = np.r_[0, np.cumsum(q_lens)]
    page_counts = (kv_lens + page_size - 1) // page_size
    page_offsets = np.r_[0, np.cumsum(page_counts)]
    num_tokens = int(cu_q_lens[-1])
    padded_tokens = (num_tokens + 31) // 32 * 32
    rng = np.random.default_rng(seed)
    page_indices = rng.permutation(np.arange(1, page_offsets[-1] + 1))
    cache = np.full((page_offsets[-1] + 2, page_size, 1, 2, HEAD_DIM), -7, np.float32)
    expected_cache = cache.copy()
    new_kv = np.zeros((padded_tokens, 2, HEAD_DIM), np.float32)
    ref_pages = np.zeros((len(sequences), max(page_counts)), np.int32)
    # The host pads custom-mask rows to a power-of-two width >= every padded kv length.
    mask_width = 1 << int(np.ceil(np.log2(max(128, max(page_counts) * page_size))))
    custom_mask = np.zeros((padded_tokens, 1, mask_width), np.int32) if tree_mask else None
    ref_mask = []
    for i, prefix in enumerate(prefixes):
        kv = rng.integers(-4, 5, size=(kv_lens[i], 2, HEAD_DIM)).astype(np.float32)
        pages = page_indices[page_offsets[i] : page_offsets[i + 1]]
        ref_pages[i, : len(pages)] = pages
        positions = np.arange(kv_lens[i])
        physical_pages, slots = pages[positions // page_size], positions % page_size
        expected_cache[physical_pages, slots, 0] = kv
        cache[physical_pages[:prefix], slots[:prefix], 0] = kv[:prefix]
        new_kv[cu_q_lens[i] : cu_q_lens[i + 1]] = kv[prefix:]
        if tree_mask:
            mask = _tree_mask(rng, q_lens[i], prefix)
            custom_mask[cu_q_lens[i] : cu_q_lens[i + 1], 0, : kv_lens[i]] = mask
            ref_mask.append(mask.reshape(-1))
    queries = rng.uniform(-0.25, 0.25, (padded_tokens, NUM_HEADS, HEAD_DIM)).astype(np.float32)
    return dict(
        queries=queries,
        new_kv=new_kv,
        cache=cache,
        expected_cache=expected_cache,
        kv_lens=kv_lens.astype(np.int32),
        page_indices=page_indices.astype(np.int32),
        cu_q_lens=cu_q_lens.astype(np.int32),
        cu_kv_lens=(page_offsets * page_size).astype(np.int32),
        ref_pages=ref_pages,
        custom_mask=custom_mask,
        ref_mask=np.concatenate(ref_mask) if tree_mask else None,
        num_seqs=len(sequences),
        num_tokens=num_tokens,
        all_decode=bool(np.all(q_lens == 1)),
    )


def _reference(batch, *, causal, sliding_window=None):
    out = ref_ragged_paged_attention(
        jnp.asarray(batch["queries"], jnp.bfloat16),
        jnp.asarray(batch["expected_cache"][:, :, :, 0, :], jnp.bfloat16),
        jnp.asarray(batch["expected_cache"][:, :, :, 1, :], jnp.bfloat16),
        jnp.asarray(batch["kv_lens"]),
        jnp.asarray(batch["ref_pages"]),
        jnp.asarray(batch["cu_q_lens"]),
        jnp.array([batch["num_seqs"]], jnp.int32),
        custom_mask=None if batch["ref_mask"] is None else jnp.asarray(batch["ref_mask"]),
        causal=bool(causal),
        sm_scale=HEAD_DIM**-0.5,
        sliding_window=sliding_window,
    )
    return np.asarray(out, np.float32)


def _run(batch, *, causal=1, chunk_prefill_size=None, **kwargs):
    n = batch["num_seqs"]
    if batch["all_decode"]:
        distribution = [n, n, n]
    else:
        distribution = [0, n if chunk_prefill_size is not None else 0, n]
    custom_mask = batch["custom_mask"]
    # Fresh device arrays per call: queries/keys/values/cache are donated.
    output, cache = ragged_paged_attention(
        jnp.asarray(batch["queries"], jnp.bfloat16),
        jnp.asarray(batch["new_kv"][:, 0:1], jnp.bfloat16),
        jnp.asarray(batch["new_kv"][:, 1:2], jnp.bfloat16),
        jnp.asarray(batch["cache"], jnp.bfloat16),
        jnp.asarray(batch["kv_lens"]),
        jnp.asarray(batch["page_indices"]),
        jnp.asarray(batch["cu_q_lens"]),
        jnp.asarray(batch["cu_kv_lens"]),
        jnp.asarray(distribution, jnp.int32),
        None if custom_mask is None else jnp.asarray(custom_mask),
        causal=causal,
        sm_scale=HEAD_DIM**-0.5,
        chunk_prefill_size=chunk_prefill_size,
        **kwargs,
    )
    output, cache = jax.device_get((output, cache))
    return output[: batch["num_tokens"]].astype(np.float32), cache.astype(np.float32)


def _check(batch, output, cache, expected_output):
    assert np.isfinite(output).all(), f"{np.isnan(output).sum()} NaN in the attention output"
    np.testing.assert_allclose(output, expected_output, atol=0.03, rtol=0.03)
    np.testing.assert_array_equal(cache, batch["expected_cache"])


def test_skip_kv_mask_single_sequence_multiple_q_blocks():
    """One sequence, one KV block, two Q blocks: the second Q block reads bkv buffer 1.

    With skip_kv_mask=True the V rows past kv_len are not zeroed by the kernel,
    so buffer 1 must be zero-initialized even though there is only one sequence
    and kv_len <= bkv_sz (0 * NaN poisons the P @ V product otherwise).
    """
    batch = _make_batch([(40, 60)], page_size=128, tree_mask=True)
    expected = _reference(batch, causal=0)
    for skip_kv_mask in (False, True):
        output, cache = _run(
            batch, causal=0, m_block_sizes=(32, 256, 32, 256), skip_kv_mask=skip_kv_mask
        )
        _check(batch, output, cache, expected)


CASES = {
    "decode": dict(
        sequences=[(1, 0), (1, 127), (1, 128), (1, 300), (1, 513)],
        d_block_sizes=(1, 256, 1, 256),
    ),
    "decode-split": dict(
        sequences=[(1, 0), (1, 127), (1, 300), (1, 513)],
        d_block_sizes=(1, 256, 1, 128),
    ),
    "decode-single": dict(sequences=[(1, 100)], d_block_sizes=(1, 256, 1, 256)),
    "mixed": dict(
        sequences=[(40, 0), (1, 200), (17, 300), (70, 5)],
        m_block_sizes=(32, 256, 32, 256),
    ),
    "mixed-split": dict(
        sequences=[(40, 0), (1, 200), (17, 300), (70, 5)],
        m_block_sizes=(32, 256, 32, 128),
    ),
    "mixed-single-multi-bq": dict(sequences=[(70, 30)], m_block_sizes=(32, 256, 32, 256)),
    "noncausal": dict(sequences=[(40, 0), (17, 300)], m_block_sizes=(32, 256, 32, 128), causal=0),
    "sliding-window": dict(
        sequences=[(40, 300), (70, 0)],
        m_block_sizes=(32, 256, 32, 128),
        sliding_window=128,
    ),
    "chunk-prefill": dict(
        sequences=[(32, 0), (32, 200)],
        chunk_prefill_size=32,
        p_block_sizes=(32, 256, 32, 256),
        m_block_sizes=(32, 256, 32, 256),
    ),
    "custom-mask": dict(
        sequences=[(8, 130), (8, 3), (8, 250)],
        m_block_sizes=(32, 256, 32, 256),
        causal=0,
        tree_mask=True,
    ),
}


@pytest.mark.parametrize("flags", [{}, FLAGS_OFF], ids=["default", "flags-off"])
@pytest.mark.parametrize("name", list(CASES))
def test_kernel_flags(name, flags):
    case = dict(CASES[name])
    batch = _make_batch(case.pop("sequences"), 128, tree_mask=case.pop("tree_mask", False))
    causal = case.get("causal", 1)
    expected = _reference(batch, causal=causal, sliding_window=case.get("sliding_window"))
    output, cache = _run(batch, **case, **flags)
    _check(batch, output, cache, expected)


@pytest.mark.parametrize(
    "name,stage_flags",
    [
        ("decode", dict(decode_only=True)),
        ("decode-single", dict(decode_only=True)),
        ("mixed", dict(skip_decode=True)),
        ("chunk-prefill", dict(skip_decode=True)),
        ("custom-mask", dict(skip_decode=True)),
    ],
    ids=lambda x: x if isinstance(x, str) else ",".join(x),
)
def test_stage_gating_matches_default(name, stage_flags):
    """decode_only / skip_decode (set by FlashAttention from forward_mode) skip empty stages."""
    case = dict(CASES[name])
    batch = _make_batch(case.pop("sequences"), 128, tree_mask=case.pop("tree_mask", False))
    default_output, default_cache = _run(batch, **case)
    output, cache = _run(batch, **case, **stage_flags)
    _check(batch, output, cache, default_output)
    np.testing.assert_array_equal(output, default_output)
    np.testing.assert_array_equal(cache, default_cache)
