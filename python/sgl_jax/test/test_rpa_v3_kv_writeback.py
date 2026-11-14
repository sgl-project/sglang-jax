"""New KV must survive SWA prefill even outside the last query block's window."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.ragged_paged_attention import ragged_paged_attention_v3
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
        ([(393, 273), (137, 512)], 128, 128, None, None, 256, 128, False),
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

    # Only cold-split checks subsequent consumption of the returned cache,
    # including new KV outside the previous SWA window; this is not a
    # service-level radix-cache reuse test. Its last page has room for decode.
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


_RAGGED_PAGE_TABLE_CASES = [
    ([(393, 273), (137, 512), (65, 191)], 128, 512, 256, "shuffled", 8, 2),
    ([(1, 37), (16, 48), (33, 60), (7, 100)], 128, 256, 128, "shuffled", 8, 2),
    ([(256, 640), (192, 384), (320, 512)], 128, 512, 256, "reverse_interleaved", 8, 2),
    ([(1, 512), (1, 273), (129, 384), (256, 128)], 128, 512, 256, "shuffled", 8, 2),
    # bkv_sz == bkv_csz elides the attention sub-block loop while KV heads
    # stay pipelined; mixed sits exactly at the 32 * 4 * 512 = 65536 gate.
    ([(1, 512), (1, 273), (129, 300)], 128, 512, 512, "shuffled", 8, 2),
    # bkv_sz == page_size (bkv_p == 1): multi-token prefill writeback uses
    # the single-DMA path, together with loop elision and pipelining.
    ([(1, 300), (100, 200), (300, 213)], 256, 256, 256, "shuffled", 8, 2),
    # 32 * 8 * 512 > 65536, so mixed runs QK/PV for each KV head in turn.
    ([(1, 512), (393, 273), (137, 512)], 128, 1024, 512, "shuffled", 16, 2),
    # Four KV heads: step1 of each head overlaps step2 of the previous one.
    ([(1, 700), (64, 350)], 128, 256, 256, "reverse_interleaved", 16, 4),
    # bkv_csz > 1024, as in the v6e tuned decode and prefill tiles, also
    # runs KV heads in turn, here with the sub-block loop elided.
    ([(1, 2100), (1, 1500), (200, 1900)], 128, 2048, 2048, "shuffled", 8, 2),
    # Reported TPU v7x benchmark shapes (Hq/Hkv=8/2, head_dim=128, page_size=128).
    ([(512, 4096), (512, 4096)], 128, 1024, 512, "contiguous", 8, 2),
    ([(512, 4096), (512, 4096)], 128, 1024, 512, "shuffled", 8, 2),
    ([(1, 8191)] * 8, 128, 1024, 1024, "contiguous", 8, 2),
    ([(1, 8191)] * 8, 128, 1024, 1024, "shuffled", 8, 2),
]

_RAGGED_PAGE_TABLE_IDS = [
    "partial-last-page",
    "single-page-sequences",
    "non-contiguous-strided-pages",
    "mixed-decode-prefill-ragged",
    "one-subblock-pipelined",
    "single-page-blocks",
    "unpipelined-kv-heads",
    "four-kv-heads",
    "wide-subblock-unpipelined",
    "bench-prefill-contiguous",
    "bench-prefill-shuffled",
    "bench-decode-contiguous",
    "bench-decode-shuffled",
]


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="Requires TPU DMA support")
@pytest.mark.parametrize(
    "sequences,page_size,bkv_sz,bkv_csz,perm_mode,num_q_heads,num_kv_heads",
    _RAGGED_PAGE_TABLE_CASES,
    ids=_RAGGED_PAGE_TABLE_IDS,
)
def test_ragged_page_table_parity(
    sequences, page_size, bkv_sz, bkv_csz, perm_mode, num_q_heads, num_kv_heads
):
    q_lens, prefixes = np.array(sequences).T
    kv_lens = q_lens + prefixes
    cu_q_lens = np.r_[0, np.cumsum(q_lens)]
    page_counts = (kv_lens + page_size - 1) // page_size
    page_offsets = np.r_[0, np.cumsum(page_counts)]
    num_tokens = int(cu_q_lens[-1])
    padded_tokens = (num_tokens + 31) // 32 * 32
    head_dim = 128
    rng = np.random.default_rng(42)
    total_mapped_pages = int(page_offsets[-1])
    if perm_mode == "contiguous":
        page_indices = np.arange(1, total_mapped_pages + 1, dtype=np.int32)
    elif perm_mode == "shuffled":
        page_indices = rng.permutation(np.arange(1, total_mapped_pages + 1, dtype=np.int32))
    else:
        all_p = np.arange(1, total_mapped_pages * 2 + 1, 2, dtype=np.int32)[::-1]
        page_indices = all_p[:total_mapped_pages]

    max_phys_page = int(np.max(page_indices)) + 2
    cache = np.full((max_phys_page, page_size, num_kv_heads, 2, head_dim), -7.0, np.float32)
    expected_cache = cache.copy()
    new_k = np.zeros((padded_tokens, num_kv_heads, head_dim), np.float32)
    new_v = np.zeros((padded_tokens, num_kv_heads, head_dim), np.float32)
    reference_pages = np.zeros((len(sequences), int(max(page_counts))), np.int32)

    for i, prefix in enumerate(prefixes):
        k_seq = rng.integers(-4, 5, size=(kv_lens[i], num_kv_heads, head_dim)).astype(np.float32)
        v_seq = rng.integers(-4, 5, size=(kv_lens[i], num_kv_heads, head_dim)).astype(np.float32)
        pages = page_indices[page_offsets[i] : page_offsets[i + 1]]
        reference_pages[i, : len(pages)] = pages
        positions = np.arange(kv_lens[i])
        physical_pages, slots = pages[positions // page_size], positions % page_size
        kv_interleaved = np.stack([k_seq, v_seq], axis=2).reshape(
            kv_lens[i], num_kv_heads, 2, head_dim
        )
        expected_cache[physical_pages, slots] = kv_interleaved
        cache[physical_pages[:prefix], slots[:prefix]] = kv_interleaved[:prefix]
        new_k[cu_q_lens[i] : cu_q_lens[i + 1]] = k_seq[prefix:]
        new_v[cu_q_lens[i] : cu_q_lens[i + 1]] = v_seq[prefix:]

    queries = jnp.asarray(
        rng.uniform(-0.25, 0.25, (padded_tokens, num_q_heads, head_dim)), jnp.bfloat16
    )
    cu_q_lens_j = jnp.asarray(cu_q_lens, jnp.int32)
    kv_lens_j = jnp.asarray(kv_lens, jnp.int32)
    expected_output = ref_ragged_paged_attention(
        queries,
        jnp.asarray(expected_cache[:, :, :, 0, :], jnp.bfloat16),
        jnp.asarray(expected_cache[:, :, :, 1, :], jnp.bfloat16),
        kv_lens_j,
        jnp.asarray(reference_pages, jnp.int32),
        cu_q_lens_j,
        jnp.array([len(sequences)], jnp.int32),
        sm_scale=head_dim**-0.5,
    )

    count = len(sequences)
    num_leading_decode = int(np.sum(q_lens == 1))
    decode_count = num_leading_decode if np.all(q_lens[:num_leading_decode] == 1) else 0
    output, updated_cache = ragged_paged_attention(
        queries,
        jnp.asarray(new_k, jnp.bfloat16),
        jnp.asarray(new_v, jnp.bfloat16),
        jnp.asarray(cache, jnp.bfloat16),
        kv_lens_j,
        jnp.asarray(page_indices, jnp.int32),
        cu_q_lens_j,
        jnp.asarray(page_offsets * page_size, jnp.int32),
        jnp.array([decode_count, decode_count, count], jnp.int32),
        None,
        d_block_sizes=(1, bkv_sz, 1, bkv_csz),
        p_block_sizes=(32, bkv_sz, 32, bkv_csz),
        m_block_sizes=(32, bkv_sz, 32, bkv_csz),
        sm_scale=head_dim**-0.5,
    )
    output, updated_cache_host = jax.device_get((output, updated_cache))

    np.testing.assert_allclose(
        output[:num_tokens].astype(np.float32),
        np.asarray(expected_output, np.float32),
        atol=0.03,
        rtol=0.03,
    )
    np.testing.assert_array_equal(updated_cache_host.astype(np.float32), expected_cache)


def _pallas_calls(jaxpr):
    """All pallas_call equations in `jaxpr`, including those inside nested jits.

    On TPU v7 each pallas_call is wrapped in one more jax.jit (for the HBM
    memory-space constraints), so it is not a top-level equation there.
    """
    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
            found.append(eqn)
            continue
        for val in eqn.params.values():
            for sub in val if isinstance(val, (tuple, list)) else (val,):
                if hasattr(sub, "eqns"):  # a (closed) jaxpr: jit, cond, while, ...
                    found.extend(_pallas_calls(sub))
    return found


@pytest.fixture(params=[None, 7], ids=["host-tpu-generation", "tpu-v7"])
def tpu_generation(request, monkeypatch):
    """Trace for the host's TPU generation and, faked, for TPU v7.

    The trace cache is not keyed on the TPU generation, so it is cleared on
    both sides to keep a faked trace from leaking into other tests.
    """
    if request.param is not None:
        monkeypatch.setattr(ragged_paged_attention_v3, "get_tpu_version", lambda: request.param)
    jax.clear_caches()
    yield request.param
    jax.clear_caches()


@pytest.mark.parametrize(
    "sequences,page_size,bkv_sz,bkv_csz,perm_mode,num_q_heads,num_kv_heads",
    _RAGGED_PAGE_TABLE_CASES,
    ids=_RAGGED_PAGE_TABLE_IDS,
)
def test_ragged_page_table_trace_cpu(
    tpu_generation, sequences, page_size, bkv_sz, bkv_csz, perm_mode, num_q_heads, num_kv_heads
):
    """Verify Pallas kernel jaxpr tracing for every ragged and benchmark shape on CPU."""
    del perm_mode
    q_lens, prefixes = np.array(sequences).T
    kv_lens = q_lens + prefixes
    cu_q_lens = np.r_[0, np.cumsum(q_lens)]
    page_counts = (kv_lens + page_size - 1) // page_size
    page_offsets = np.r_[0, np.cumsum(page_counts)]
    padded_tokens = (int(cu_q_lens[-1]) + 31) // 32 * 32
    head_dim = 128
    total_mapped_pages = int(page_offsets[-1])

    queries = jax.ShapeDtypeStruct((padded_tokens, num_q_heads, head_dim), jnp.bfloat16)
    keys = jax.ShapeDtypeStruct((padded_tokens, num_kv_heads, head_dim), jnp.bfloat16)
    values = jax.ShapeDtypeStruct((padded_tokens, num_kv_heads, head_dim), jnp.bfloat16)
    cache = jax.ShapeDtypeStruct(
        (total_mapped_pages + 2, page_size, num_kv_heads, 2, head_dim), jnp.bfloat16
    )
    kv_lens_s = jax.ShapeDtypeStruct((len(sequences),), jnp.int32)
    page_indices_s = jax.ShapeDtypeStruct((total_mapped_pages,), jnp.int32)
    cu_q_lens_s = jax.ShapeDtypeStruct((len(sequences) + 1,), jnp.int32)
    cu_kv_lens_s = jax.ShapeDtypeStruct((len(sequences) + 1,), jnp.int32)
    distribution_s = jax.ShapeDtypeStruct((3,), jnp.int32)

    def _run(q, k, v, c, kvl, pi, cql, ckvl, dist):
        return ragged_paged_attention(
            q,
            k,
            v,
            c,
            kvl,
            pi,
            cql,
            ckvl,
            dist,
            None,
            d_block_sizes=(1, bkv_sz, 1, bkv_csz),
            p_block_sizes=(32, bkv_sz, 32, bkv_csz),
            m_block_sizes=(32, bkv_sz, 32, bkv_csz),
            sm_scale=head_dim**-0.5,
        )

    traced = jax.make_jaxpr(_run)(
        queries,
        keys,
        values,
        cache,
        kv_lens_s,
        page_indices_s,
        cu_q_lens_s,
        cu_kv_lens_s,
        distribution_s,
    )
    assert len(_pallas_calls(traced.jaxpr)) >= 2


def test_prologue_skips_bkv_zeroing_unless_skip_kv_mask_cpu(tpu_generation):
    """Verify prologue only emits VMEM bkv_x2_ref zero-fill when skip_kv_mask=True."""
    page_size, hq, hkv, hd = 128, 8, 2, 128
    queries = jax.ShapeDtypeStruct((1024, hq, hd), jnp.bfloat16)
    kv = jax.ShapeDtypeStruct((1024, hkv, hd), jnp.bfloat16)
    cache = jax.ShapeDtypeStruct((74, page_size, hkv, 2, hd), jnp.bfloat16)
    kv_lens_s = jax.ShapeDtypeStruct((2,), jnp.int32)
    page_indices_s = jax.ShapeDtypeStruct((72,), jnp.int32)
    cu_lens_s = jax.ShapeDtypeStruct((3,), jnp.int32)
    dist_s = jax.ShapeDtypeStruct((3,), jnp.int32)

    def _prologue_zero_prims(skip_kv_mask: bool) -> list[str]:
        def _fn(q, k, v, c, kvl, pi, cql, ckvl, dist):
            return ragged_paged_attention(
                q,
                k,
                v,
                c,
                kvl,
                pi,
                cql,
                ckvl,
                dist,
                None,
                d_block_sizes=(1, 1024, 1, 1024),
                p_block_sizes=(32, 1024, 32, 512),
                m_block_sizes=(32, 1024, 32, 512),
                sm_scale=hd**-0.5,
                causal=0,
                skip_kv_mask=skip_kv_mask,
            )

        traced = jax.make_jaxpr(_fn)(
            queries, kv, kv, cache, kv_lens_s, page_indices_s, cu_lens_s, cu_lens_s, dist_s
        )
        pallas_calls = _pallas_calls(traced.jaxpr)
        assert len(pallas_calls) == 2
        zero_prims = []
        for pc in pallas_calls:
            body = pc.params["jaxpr"].eqns[-1].params["body_jaxpr"].jaxpr
            conds = [e for e in body.eqns if e.primitive.name == "cond"]
            prologue_branch = conds[0].params["branches"][1].jaxpr
            zero_prims.extend(
                e.primitive.name
                for e in prologue_branch.eqns
                if e.primitive.name in ("broadcast_in_dim", "swap")
            )
        return zero_prims

    assert _prologue_zero_prims(False) == []
    assert _prologue_zero_prims(True) == ["broadcast_in_dim", "swap", "swap"] * 2
