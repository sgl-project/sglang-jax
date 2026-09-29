from test.srt.kernels.csa_attention.ref import reference as attention_reference
from test.srt.kernels.csa_compressor import ref as compressor_ref
from test.srt.kernels.csa_compressor.test_compressor import make_case as compressor_case

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from sgl_jax.srt.kernels.csa import CSACache, csa_attention, csa_topk
from sgl_jax.srt.kernels.csa_attention.tune import get_csa_attention_schedule
from sgl_jax.srt.kernels.csa_compressor.tune import get_compressor_schedule
from sgl_jax.srt.layers.attention.csa_metadata import prepare_csa_metadata

from .ref import reference, topk

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires TPU")


def make_case(lengths, prefixes, *, pad=3, hidden=128, heads=8, index_heads=8, seed=10):
    rng = np.random.default_rng(seed)
    operands, _ = compressor_case(lengths, prefixes, hidden=hidden, pad=pad, seed=seed)
    batch, tokens = len(lengths), sum(lengths) + pad
    maxend = max(p + n for p, n in zip(prefixes, lengths, strict=True))
    pages_per_request = max(1, (maxend // 4 + 127) // 128)
    cp = (
        rng.permutation(np.arange(1, batch * pages_per_request + 1))
        .astype(np.int32)
        .reshape(batch, -1)
    )
    wp = rng.permutation(np.arange(1, batch + 1)).astype(np.int32).reshape(batch, 1)
    slots = np.arange(batch, 0, -1, dtype=np.int32)
    meta = prepare_csa_metadata(
        lengths,
        prefixes,
        slots,
        wp,
        cp,
        num_tokens=tokens,
        window_size=128,
        window_page_size=128,
        compressed_page_size=128,
    )

    def bf16(shape, scale=0.3):
        return rng.normal(0, scale, shape).astype(ml_dtypes.bfloat16)

    pages = batch * pages_per_request + 1
    nope, rope = compressor_ref.pack_main(rng.normal(0, 0.3, (pages * 128, 512)).astype(np.float32))
    index = compressor_ref.pack_index(rng.normal(0, 0.3, (pages * 128, 128)).astype(np.float32))
    window = bf16((batch + 1, 64, 2, 512))
    window[0] = np.nan
    cache = CSACache(
        *operands[8:10],
        nope.reshape(pages, 128, 4, 128),
        rope.reshape(pages, 32, 4, 128),
        index.reshape(pages, 32, 4, 256),
        window,
    )
    iq, iw = (
        bf16((tokens, index_heads, 128)),
        rng.uniform(0, 0.1, (tokens, index_heads)).astype(np.float32),
    )
    q, new = bf16((tokens, heads, 512)), bf16((tokens, 512))
    sink = np.linspace(-2, 2, heads, dtype=np.float32)
    return (*operands[:8], iq, iw, q, new, sink), cache, meta


def options(inputs, metadata):
    device = jax.devices()[0].device_kind
    decode = np.diff(np.asarray(metadata.attention.cu_q_lens)).max() <= 1
    return dict(
        compressor_schedule=get_compressor_schedule(inputs[0].shape[1], device_kind=device),
        attention_schedule=get_csa_attention_schedule(device, decode=decode),
        scale=512**-0.5,
        window_size=128,
        top_k=512,
        kv_pages_per_block=1,
        queries_per_block=1 if decode else 32,
    )


def assert_close(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_allclose(a, b, rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize(
    "lengths,prefixes",
    [
        ((1,), (0,)),
        ((1,), (3,)),
        ((1, 1, 1, 1), (127, 2047, 4095, 8191)),
        ((128,), (0,)),
        ((140,), (0,)),
        ((5, 0, 1, 9), (127, 0, 2047, 4095)),
        ((8,), (4088,)),
        ((0, 0), (0, 0)),
    ],
)
@pytest.mark.parametrize("index_heads", [8, 64])
def test_full_csa(lengths, prefixes, index_heads):
    inputs, cache, meta = make_case(lengths, prefixes, index_heads=index_heads)
    opts = options(inputs, meta)
    schedule = opts["compressor_schedule"]
    expected, expected_cache, _ = reference(
        inputs,
        cache,
        meta,
        projection_k_tile=schedule.projection_k_tile,
        query_tile=schedule.query_tile,
    )
    actual, updated = jax.block_until_ready(
        csa_attention(
            *jax.tree.map(jnp.asarray, inputs),
            jax.tree.map(jnp.asarray, cache),
            meta,
            **opts,
        )
    )
    updated = jax.tree.map(np.asarray, updated)
    assert_close(actual, expected)
    np.testing.assert_array_equal(actual[sum(lengths) :], 0)
    np.testing.assert_array_equal(
        updated.window.view(np.uint16), expected_cache[-1].view(np.uint16)
    )
    for got, want in zip(updated[:2], expected_cache[:2], strict=True):
        np.testing.assert_array_equal(np.isneginf(got), np.isneginf(want))
        finite = np.isfinite(want)
        assert_close(got[finite], want[finite])
    # The composition must preserve every compressed record it did not write.
    locations = np.asarray(meta.compressor.cache_locations)
    written = np.unique(locations[locations >= 0])
    for got, old, width in zip(updated[2:5], cache[2:5], (512, 128, 256), strict=True):
        got, old = got.reshape(-1, width), old.reshape(-1, width)
        keep = np.ones(len(old), dtype=bool)
        keep[written] = False
        np.testing.assert_array_equal(got[keep], old[keep])
    main_values = compressor_ref.decode_main(
        updated.main_nope.reshape(-1, 512), updated.main_rope.reshape(-1, 128)
    )
    index_values = compressor_ref.decode_index(updated.index.reshape(-1, 256))
    assert np.isfinite(main_values).all() and np.isfinite(index_values).all()
    # Isolate retrieval/attention from the separately tested quantization boundary.
    idx = csa_topk(
        jnp.asarray(inputs[8]),
        jnp.asarray(inputs[9]),
        jnp.asarray(updated.index),
        meta,
        top_k=512,
        kv_pages_per_block=1,
        queries_per_block=opts["queries_per_block"],
    )
    want_idx = topk(inputs[8], inputs[9], updated.index, meta, 512)
    np.testing.assert_array_equal(np.sort(idx, axis=1), np.sort(want_idx, axis=1))
    staged = attention_reference(
        *inputs[10:12],
        cache.window,
        updated.main_nope,
        updated.main_rope,
        want_idx,
        inputs[12],
        meta.attention,
        scale=512**-0.5,
    )
    assert_close(actual, staged)


@pytest.mark.parametrize("prefix,total,first", [(0, 140, 127), (2040, 24, 7), (4088, 24, 7)])
def test_stateful_decode_matches_prefill(prefix, total, first):
    inputs, initial, whole_meta = make_case((total,), (prefix,), pad=0)
    opts = options(inputs, whole_meta)
    whole, whole_cache = jax.block_until_ready(
        csa_attention(
            *jax.tree.map(jnp.asarray, inputs),
            jax.tree.map(jnp.asarray, initial),
            whole_meta,
            **opts,
        )
    )
    cache = jax.tree.map(jnp.asarray, initial)
    outputs = []
    offset = 0
    # Prefix fill then individual steps across SWA wrap and two compression groups.
    for length in (first, *([1] * (total - first))):
        meta = prepare_csa_metadata(
            (length,),
            (prefix + offset,),
            (1,),
            np.asarray(whole_meta.attention.window_page_indices).reshape(1, -1),
            np.asarray(whole_meta.attention.compressed_page_indices).reshape(1, -1),
            num_tokens=length,
            window_size=128,
            window_page_size=128,
            compressed_page_size=128,
        )
        chunk = tuple(
            v[offset : offset + length] if i in (0, 8, 9, 10, 11) else v
            for i, v in enumerate(inputs)
        )
        output, cache = jax.block_until_ready(
            csa_attention(
                *jax.tree.map(jnp.asarray, chunk),
                cache,
                meta,
                **options(chunk, meta),
            )
        )
        outputs.append(np.asarray(output))
        offset += length
    assert_close(np.concatenate(outputs), whole)
    np.testing.assert_array_equal(
        np.asarray(cache.window).view(np.uint16), np.asarray(whole_cache.window).view(np.uint16)
    )
    for actual, expected in zip(cache[:2], whole_cache[:2], strict=True):
        actual, expected = np.asarray(actual), np.asarray(expected)
        np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
        active = ~np.isneginf(expected)
        assert_close(actual[active], expected[active])


def test_reject_shared_write_ownership():
    with pytest.raises(ValueError, match="distinct positive"):
        prepare_csa_metadata(
            (1, 1),
            (3, 3),
            (1, 2),
            [[1], [1]],
            [[1], [2]],
            num_tokens=2,
            window_size=128,
            window_page_size=128,
            compressed_page_size=128,
        )


@pytest.mark.parametrize("future", [3, 7])
def test_future_compressor_input_does_not_change_past_output(future):
    inputs, cache, meta = make_case((8,), (0,), pad=0)
    opts = options(inputs, meta)

    def run(values):
        return np.asarray(
            csa_attention(
                *jax.tree.map(jnp.asarray, values),
                jax.tree.map(jnp.asarray, cache),
                meta,
                **opts,
            )[0]
        )

    before = run(inputs)
    changed = list(inputs)
    changed[0] = inputs[0].copy()
    changed[0][future] = (-3 * changed[0][future].astype(np.float32)).astype(ml_dtypes.bfloat16)
    after = run(tuple(changed))
    assert np.isfinite(before).all() and np.isfinite(after).all()
    np.testing.assert_array_equal(before[:future], after[:future])
    assert np.any(
        before[future:] != after[future:]
    ), "future-token probe must affect its own output"


@pytest.mark.parametrize("lengths,pages", [(1, [[1]]), ([0], [[]])])
def test_reject_malformed_metadata(lengths, pages):
    with pytest.raises(ValueError):
        prepare_csa_metadata(
            lengths,
            [0],
            [0],
            [[1]],
            pages,
            num_tokens=1,
            window_size=128,
            window_page_size=128,
            compressed_page_size=128,
        )


def test_stale_compressed_cache_is_detected(monkeypatch):
    from sgl_jax.srt.kernels.csa import csa as module

    inputs, cache, meta = make_case((4,), (0,), pad=0)
    opts = options(inputs, meta)
    schedule = opts["compressor_schedule"]
    expected, _, _ = reference(
        inputs,
        cache,
        meta,
        projection_k_tile=schedule.projection_k_tile,
        query_tile=schedule.query_tile,
    )
    # Deliberately omit compressor/state writes: the integrated output must fail.
    monkeypatch.setattr(module, "csa_compressor", lambda *args, **kwargs: args[8:13])
    actual, _ = jax.block_until_ready(
        module.csa_attention.__wrapped__(
            *jax.tree.map(jnp.asarray, inputs),
            jax.tree.map(jnp.asarray, cache),
            meta,
            **opts,
        )
    )
    with pytest.raises(AssertionError):
        assert_close(actual, expected)


@pytest.mark.parametrize("hidden", [4096, 7168])
def test_deployment_width(hidden):
    inputs, cache, meta = make_case((8,), (0,), pad=0, hidden=hidden, heads=64)
    opts = options(inputs, meta)
    schedule = opts["compressor_schedule"]
    expected, _, _ = reference(
        inputs,
        cache,
        meta,
        projection_k_tile=schedule.projection_k_tile,
        query_tile=schedule.query_tile,
    )
    actual, _ = jax.block_until_ready(
        csa_attention(
            *jax.tree.map(jnp.asarray, inputs),
            jax.tree.map(jnp.asarray, cache),
            meta,
            **opts,
        )
    )
    assert_close(actual, expected)
