import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.csa_attention import CSAAttentionMetadata, csa_joint_attention
from sgl_jax.srt.kernels.csa_attention.tune import CSAAttentionSchedule

from .ref import reference, update_window

requires_tpu = pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires TPU")


def assert_close(actual, expected):
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=1e-2)


def make_case(
    lengths=(1,),
    prefixes=(511,),
    *,
    heads=8,
    pad=0,
    seed=0,
    page_size=128,
    window_size=128,
    compression_ratio=4,
    top_k=512,
    fp8_scale_block=64,
    rows_per_group=4,
):
    rng = np.random.default_rng(seed)

    def bf16(x):
        return np.asarray(x, np.float32).astype(ml_dtypes.bfloat16)

    batch, real = len(lengths), sum(lengths)
    tokens = real + pad
    cu = np.asarray((0, *np.cumsum(lengths)), np.int32)
    ends = np.asarray(lengths, np.int32) + np.asarray(prefixes, np.int32)
    reqs = np.pad(
        np.repeat(np.arange(batch, dtype=np.int32), lengths), (0, pad), constant_values=-1
    )
    q = bf16(rng.normal(0, 2, (tokens, heads, 512)))
    new = bf16(rng.normal(0, 0.7, (tokens, 512)))
    q[real:], new[real:] = np.nan, np.nan
    wp = rng.permutation(np.arange(1, batch + 1)).astype(np.int32)
    wc = np.arange(batch + 1, dtype=np.int32) * window_size
    window = bf16(rng.normal(0, 0.7, (batch + 1, window_size // 2, 2, 512)))
    window[0] = np.nan
    per_request = max(1, (max(ends, default=0) // compression_ratio + page_size - 1) // page_size)
    cp = rng.permutation(np.arange(1, batch * per_request + 1)).astype(np.int32)
    cc = np.arange(batch + 1, dtype=np.int32) * per_request * page_size
    compressed_lens = ends // compression_ratio
    pages = batch * per_request + 1
    values = rng.normal(0, 0.7, (pages, page_size, 512)).astype(np.float32)
    nope = np.zeros((pages, page_size, 512), np.uint8)
    blocks = values[..., :448].reshape(pages, page_size, 448 // fp8_scale_block, fp8_scale_block)
    scales = np.exp2(np.ceil(np.log2(np.maximum(np.max(np.abs(blocks), axis=-1), 1e-4) / 448)))
    nope[..., :448] = (
        (blocks / scales[..., None])
        .astype(ml_dtypes.float8_e4m3fn)
        .view(np.uint8)
        .reshape(pages, page_size, 448)
    )
    nope[..., 448 : 448 + 448 // fp8_scale_block] = scales.astype(ml_dtypes.float8_e8m0fnu).view(
        np.uint8
    )
    bits = bf16(values[..., 448:]).view(np.uint16)
    rope = np.concatenate(((bits >> 8).astype(np.uint8), (bits & 255).astype(np.uint8)), axis=-1)
    nope[0], rope[0] = 127, 255
    indices = np.full((tokens, top_k), -1, np.int32)
    for t, r in enumerate(reqs[:real]):
        visible = min(top_k, (prefixes[r] + t - cu[r] + 1) // compression_ratio)
        indices[t, :visible] = rng.permutation((prefixes[r] + t - cu[r] + 1) // compression_ratio)[
            :visible
        ]
    sink = np.linspace(-2, 2, heads, dtype=np.float32)
    locations = np.full(tokens, -1, np.int32)
    for r, length in enumerate(lengths):
        local = np.arange(max(0, length - window_size), length)
        locations[cu[r] + local] = wp[r] * window_size + (prefixes[r] + local) % window_size
    meta = CSAAttentionMetadata(reqs, cu, ends, wp, wc, cp, cc, compressed_lens, locations)
    return (
        q,
        new,
        window,
        nope.reshape(pages, page_size, 4, 128),
        rope.reshape(pages, page_size // rows_per_group, rows_per_group, 128),
        indices,
        sink,
        meta,
    )


def run(
    args,
    query_tile=1,
    selected_tile=128,
    *,
    window_size=128,
    compression_ratio=4,
    fp8_scale_block=64,
    rows_per_group=4,
):
    arrays = jax.tree.map(jnp.asarray, args)
    result = csa_joint_attention(
        *arrays,
        fp8_scale_block=fp8_scale_block,
        rows_per_group=rows_per_group,
        scale=512**-0.5,
        window_size=window_size,
        compression_ratio=compression_ratio,
        schedule=CSAAttentionSchedule(query_tile=query_tile, selected_tile=selected_tile),
    )
    output, window = jax.tree.map(np.asarray, jax.block_until_ready(result))
    expected_window = update_window(args[1], args[2], args[-1], window_size=window_size)
    np.testing.assert_array_equal(window.view(np.uint16), expected_window.view(np.uint16))
    return output.astype(np.float32)


@pytest.mark.parametrize(
    "lengths,prefixes",
    [
        ((1,), (511,)),
        ((1, 1, 1, 1), (0, 3, 128, 4095)),
        ((16,), (0,)),
        ((132,), (12,)),
        ((1, 8, 0, 5), (511, 12, 0, 3)),
    ],
)
@pytest.mark.parametrize("query_tile", [1, 8, 32])
@requires_tpu
def test_attention(lengths, prefixes, query_tile):
    args = make_case(lengths, prefixes, pad=3)
    actual = run(args, query_tile)
    expected = reference(*args, scale=512**-0.5)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    assert_close(actual, expected)
    np.testing.assert_array_equal(actual[sum(lengths) :], 0)


@requires_tpu
def test_masks_and_sink():
    args = list(make_case((3, 1), (4, 7), pad=2))
    args[5][:] = -1
    args[5][0, :4] = [0, 1, -2, 99999]  # Entry 1 is not yet causally visible.
    args[6] = np.linspace(-80, 80, 8, dtype=np.float32)
    actual = run(tuple(args), 8)
    expected = reference(*args, scale=512**-0.5)
    assert np.isfinite(actual).all()
    assert_close(actual, expected)


@pytest.mark.parametrize("page_size", [16, 32, 128])
@pytest.mark.parametrize("selected_tile", [128, 256])
@requires_tpu
def test_page_layout(page_size, selected_tile):
    args = make_case((1, 1), (2047, 4095), page_size=page_size, heads=64)
    assert_close(run(args, selected_tile=selected_tile), reference(*args, scale=512**-0.5))


@pytest.mark.parametrize("lengths,prefixes", [((16,), (0,)), ((1, 9), (4095, 119))])
@pytest.mark.parametrize("query_tile", [8, 32])
@pytest.mark.parametrize("selected_tile", [128, 256])
@requires_tpu
def test_full_heads_query_blocks(lengths, prefixes, query_tile, selected_tile):
    args = make_case(lengths, prefixes, heads=64, pad=3)
    assert_close(run(args, query_tile, selected_tile), reference(*args, scale=512**-0.5))


@requires_tpu
def test_wide_ragged_block_boundaries():
    args = make_case((33, 5, 0, 35), (4095, 127, 0, 2048), heads=64, pad=3)
    args[-1].compressed_page_indices[[1, 2]] = 0
    assert_close(run(args, 32, 256), reference(*args, scale=512**-0.5))


@pytest.mark.parametrize("query_tile", [1, 32])
@pytest.mark.parametrize("selected_tile", [128, 256])
@requires_tpu
def test_topk_bitset_boundaries(query_tile, selected_tile):
    args = make_case((3, 1), (2047, 4095), heads=8, pad=2)
    indices = args[5]
    indices[:] = -1
    # Include sign bits, adjacent words/tiles, and a future (masked) entry.
    selected = [0, 30, 31, 32, 63, 64, 95, 96, 127, 128, 255, 256, 511, 512, 1023]
    for token in range(4):
        indices[token, : len(selected)] = np.roll(selected, token)
    args[-1].compressed_page_indices[1] = 0
    assert_close(run(args, query_tile, selected_tile), reference(*args, scale=512**-0.5))


@pytest.mark.parametrize("query_tile", [1, 32])
@pytest.mark.parametrize("selected_tile", [128, 256, 512])
@requires_tpu
def test_causal_tile_tails(query_tile, selected_tile):
    args = make_case((1, 3, 2, 1, 3, 2), (507, 511, 515, 1019, 1023, 1027), pad=3, page_size=32)
    args[-1].compressed_page_indices[1] = 0
    assert_close(run(args, query_tile, selected_tile), reference(*args, scale=512**-0.5))


@pytest.mark.parametrize("query_tile", [1, 8])
@requires_tpu
def test_dma_flags_reused_across_tiles_and_requests(query_tile):
    args = make_case((1, 1, 1), (8191, 8191, 8191), pad=2)
    args[5][:] = -1
    # Alternate active panels when each double-buffer slot is reused.
    chosen = [0, 384, 640, 768, 1024, 1408, 1664, 1792]
    args[5][0, : len(chosen)] = chosen
    args[5][2, : len(chosen)] = chosen[::-1]
    third_request_page = args[-1].compressed_cu_kv_lens[2] // 128
    args[-1].compressed_page_indices[third_request_page + 5] = 0
    assert_close(run(args, query_tile, 256), reference(*args, scale=512**-0.5))


@pytest.mark.parametrize("lengths,prefixes,pad", [((), (), 0), ((0,), (0,), 0), ((0,), (0,), 3)])
@requires_tpu
def test_empty_requests(lengths, prefixes, pad):
    args = make_case(lengths, prefixes, pad=pad)
    np.testing.assert_array_equal(run(args, 8), np.zeros(args[0].shape, np.float32))


@requires_tpu
def test_missing_pages_and_readonly_compressed_cache():
    args = make_case((1, 1), (511, 4095), pad=1)
    args[-1].window_page_indices[:] = 0
    args[-1].window_write_locations[:] = -1
    args[-1].compressed_page_indices[:] = 0
    arrays = jax.tree.map(jnp.asarray, args)
    before = [np.asarray(x).copy() for x in arrays[2:5]]
    actual = np.asarray(
        csa_joint_attention(
            *arrays,
            fp8_scale_block=64,
            rows_per_group=4,
            scale=512**-0.5,
            window_size=128,
            compression_ratio=4,
            schedule=CSAAttentionSchedule(query_tile=1),
        )[0]
    ).astype(np.float32)
    assert_close(actual, reference(*args, scale=512**-0.5))
    for old, new in zip(before, arrays[2:5], strict=True):
        np.testing.assert_array_equal(old.view(np.uint8), np.asarray(new).view(np.uint8))


@requires_tpu
def test_causal_mask_is_observable():
    args = make_case((1,), (3,))
    actual = run(args)
    corrupted = list(args)
    corrupted[5] = np.full_like(args[5], -1)  # Deliberately drop the visible compressed entry.
    wrong = reference(*corrupted, scale=512**-0.5)
    with pytest.raises(AssertionError):
        assert_close(actual, wrong)


@pytest.mark.parametrize("tile", [0, -128, 64, 384])
@requires_tpu
def test_invalid_tile(tile):
    args = jax.tree.map(jnp.asarray, make_case())
    with pytest.raises(ValueError, match="selected_tile"):
        csa_joint_attention(
            *args,
            fp8_scale_block=64,
            rows_per_group=4,
            scale=512**-0.5,
            window_size=128,
            compression_ratio=4,
            schedule=CSAAttentionSchedule(selected_tile=tile),
        )


@pytest.mark.parametrize("tile", [256, 512])
@requires_tpu
def test_selected_tiles(tile):
    args = make_case((1,), (4095,))
    assert_close(run(args, selected_tile=tile), reference(*args, scale=512**-0.5))


@pytest.mark.parametrize("prefix", [8191, 65535])
@pytest.mark.parametrize("selected_tile", [128, 256])
@requires_tpu
def test_sparse_page_union(prefix, selected_tile):
    args = make_case((2,), (prefix,), heads=64, pad=2)
    args[5][:] = -1
    last = (prefix + 1) // 4 - 1
    args[5][0, [0, 127, 255, 511]] = [0, 127, 896, last]
    args[5][1, [1, 257]] = [0, 1280]
    actual = run(args, 8, selected_tile)
    assert np.isfinite(actual).all()
    assert_close(actual, reference(*args, scale=512**-0.5))


@requires_tpu
def test_unselected_rows_are_not_used():
    args = make_case()
    page = int(args[-1].compressed_page_indices[0])
    nope = args[3][page, 31].copy()
    rope = args[4][page, 31 // 4, 31 % 4].copy()
    args[3][:] = 127  # Poison even unselected rows on an otherwise valid page.
    args[4][:] = 255
    args[3][page, 31] = nope
    args[4][page, 31 // 4, 31 % 4] = rope
    args[5][:] = -1
    args[5][0, 257] = 31
    actual = run(args)
    assert np.isfinite(actual).all()
    assert_close(actual, reference(*args, scale=512**-0.5))


@pytest.mark.parametrize("query_tile,prefix", [(1, 511), (32, 0)])
@pytest.mark.parametrize("interpret", [True, pltpu.InterpretParams()])
def test_interpret_and_partial_tile(query_tile, prefix, interpret):
    args = make_case((1,), (prefix,))
    with jax.default_device(jax.devices("cpu")[0]):
        arrays = jax.tree.map(jnp.asarray, args)
        actual = np.asarray(
            csa_joint_attention(
                *arrays,
                fp8_scale_block=64,
                rows_per_group=4,
                scale=512**-0.5,
                window_size=128,
                compression_ratio=4,
                schedule=CSAAttentionSchedule(query_tile=query_tile),
                interpret=interpret,
            )[0]
        ).astype(np.float32)
    assert_close(actual, reference(*args, scale=512**-0.5))


@pytest.mark.parametrize("count", [0, 1, 127, 128, 129, 255, 256, 257, 511, 512])
@requires_tpu
def test_compact_fill_and_flush(count):
    args = make_case((1,), (8191,), heads=64, seed=count)
    args[5][:] = -1
    args[5][0, :count] = np.random.default_rng(count).permutation(2048)[:count]
    assert_close(run(args), reference(*args, scale=512**-0.5))


@pytest.mark.parametrize("page_size", [4, 64, 96, 256])
@requires_tpu
def test_compact_page_boundaries(page_size):
    args = make_case((1, 1), (2047, 8191), heads=64, page_size=page_size, pad=1)
    args[-1].compressed_page_indices[[1, 2]] = 0
    assert_close(run(args), reference(*args, scale=512**-0.5))


@requires_tpu
def test_missing_compressed_mutation(monkeypatch):
    original = csa_joint_attention

    def omit_compressed(*args, **kwargs):
        broken = list(args)
        broken[5] = jnp.full_like(broken[5], -1)
        return original(*broken, **kwargs)

    monkeypatch.setitem(run.__globals__, "csa_joint_attention", omit_compressed)
    with pytest.raises(AssertionError):
        test_attention((1,), (511,), 1)


@requires_tpu
def test_missing_window_write_is_detected(monkeypatch):
    original = csa_joint_attention

    def omit_write(*args, **kwargs):
        output, _ = original(*args, **kwargs)
        return output, args[2]

    monkeypatch.setitem(run.__globals__, "csa_joint_attention", omit_write)
    with pytest.raises(AssertionError):
        run(make_case((1,), (3,)))


@requires_tpu
def test_decode_writeback_multiple_window_pages():
    args = list(make_case((1, 0, 1), (0, 0, 255), pad=3, window_size=256))
    # Split each request's 256-token window into two physical 128-token pages.
    args[2] = args[2].reshape(-1, 64, 2, 512)
    pages = args[-1].window_page_indices
    args[-1] = args[-1]._replace(
        window_page_indices=np.stack((pages * 2, pages * 2 + 1), axis=1).reshape(-1)
    )
    args[1][0] = np.float32(-0.0)
    actual = run(tuple(args), window_size=256)
    assert_close(actual, reference(*args, scale=512**-0.5, window_size=256))


@requires_tpu
def test_stateful_decode_ring_wrap():
    args = list(make_case((1, 1), (126, 254), heads=64))
    window = args[2].copy()
    for step in range(4):
        meta = args[-1]
        positions = np.asarray((126, 254), np.int32) + step
        args[-1] = meta._replace(
            seq_lens=positions + 1,
            compressed_kv_lens=(positions + 1) // 4,
            window_write_locations=meta.window_page_indices * 128 + positions % 128,
        )
        args[2] = window
        # Includes signed zero while preserving the rest of the physical page.
        args[1][0, step] = np.float32(-0.0)
        expected = reference(*args, scale=512**-0.5)
        expected_window = update_window(args[1], window, args[-1], window_size=128)
        output, updated = csa_joint_attention(
            *jax.tree.map(jnp.asarray, args),
            scale=512**-0.5,
            schedule=CSAAttentionSchedule(query_tile=1),
            window_size=128,
            compression_ratio=4,
            fp8_scale_block=64,
            rows_per_group=4,
        )
        output, window = jax.tree.map(np.asarray, jax.block_until_ready((output, updated)))
        assert_close(output.astype(np.float32), expected)
        np.testing.assert_array_equal(window.view(np.uint16), expected_window.view(np.uint16))


@pytest.mark.parametrize("query_tile", [1, 32])
@pytest.mark.parametrize("window_size,compression_ratio,top_k", [(256, 8, 256), (128, 4, 1024)])
@requires_tpu
def test_model_parameters(query_tile, window_size, compression_ratio, top_k):
    args = make_case(
        (3, 5),
        (8191, 127),
        pad=1,
        window_size=window_size,
        compression_ratio=compression_ratio,
        top_k=top_k,
    )
    actual = run(args, query_tile, window_size=window_size, compression_ratio=compression_ratio)
    expected = reference(
        *args, scale=512**-0.5, window_size=window_size, compression_ratio=compression_ratio
    )
    assert_close(actual, expected)


@pytest.mark.parametrize("window_size,compression_ratio", [(0, 4), (127, 4), (128, 0), (128, -1)])
@requires_tpu
def test_invalid_model_parameters(window_size, compression_ratio):
    args = jax.tree.map(jnp.asarray, make_case())
    with pytest.raises(ValueError, match="window_size|compression_ratio"):
        csa_joint_attention(
            *args,
            fp8_scale_block=64,
            rows_per_group=4,
            scale=512**-0.5,
            schedule=CSAAttentionSchedule(),
            window_size=window_size,
            compression_ratio=compression_ratio,
        )


@pytest.mark.parametrize("top_k", [0, 128, 384])
@requires_tpu
def test_invalid_topk_width(top_k):
    args = jax.tree.map(jnp.asarray, make_case(top_k=top_k))
    with pytest.raises(ValueError, match="Top-K"):
        csa_joint_attention(
            *args,
            fp8_scale_block=64,
            rows_per_group=4,
            scale=512**-0.5,
            schedule=CSAAttentionSchedule(),
            window_size=128,
            compression_ratio=4,
        )


@pytest.mark.parametrize("query_tile", [1, 32])
@requires_tpu
def test_cache_format_parameters(query_tile):
    args = make_case((3,), (4095,), fp8_scale_block=32, rows_per_group=8)
    actual = run(args, query_tile, fp8_scale_block=32, rows_per_group=8)
    expected = reference(*args, scale=512**-0.5, fp8_scale_block=32, rows_per_group=8)
    assert_close(actual, expected)


@pytest.mark.parametrize(
    "fp8_scale_block,rows_per_group", [(0, 4), (128, 4), (1, 4), (64, 0), (64, 3)]
)
@requires_tpu
def test_invalid_cache_format(fp8_scale_block, rows_per_group):
    args = jax.tree.map(jnp.asarray, make_case())
    with pytest.raises(ValueError, match="fp8_scale_block|rows_per_group"):
        csa_joint_attention(
            *args,
            scale=512**-0.5,
            schedule=CSAAttentionSchedule(),
            window_size=128,
            compression_ratio=4,
            fp8_scale_block=fp8_scale_block,
            rows_per_group=rows_per_group,
        )


@requires_tpu
def test_large_prefill_writeback():
    # This token count overflowed the old SMEM run table despite a small final ring.
    batch, sequence, heads, width, page_size = 32, 2048, 8, 512, 128
    tokens = batch * sequence
    pages = batch * (sequence // 4 // page_size) + 1
    position = np.tile(np.arange(sequence, dtype=np.int32), batch)
    requests = np.repeat(np.arange(batch, dtype=np.int32), sequence)
    locations = np.where(
        position >= sequence - page_size,
        (requests + 1) * page_size + position % page_size,
        -1,
    )
    meta = CSAAttentionMetadata(
        requests,
        np.arange(batch + 1, dtype=np.int32) * sequence,
        np.full(batch, sequence, np.int32),
        np.arange(1, batch + 1, dtype=np.int32),
        np.arange(batch + 1, dtype=np.int32) * page_size,
        np.arange(1, pages, dtype=np.int32),
        np.arange(batch + 1, dtype=np.int32) * (sequence // 4),
        np.full(batch, sequence // 4, np.int32),
        locations,
    )
    nope = np.zeros((pages, page_size, width), np.uint8)
    nope[..., 448:455] = 127
    count = (position + 1) // 4
    indices = np.where(
        np.arange(512)[None, :] < count[:, None], np.arange(512)[None, :], -1
    ).astype(np.int32)
    output, window = jax.tree.map(
        np.asarray,
        jax.block_until_ready(
            csa_joint_attention(
                jnp.zeros((tokens, heads, width), jnp.bfloat16),
                jnp.ones((tokens, width), jnp.bfloat16),
                jnp.zeros((batch + 1, page_size // 2, 2, width), jnp.bfloat16),
                jnp.asarray(nope.reshape(pages, page_size, 4, 128)),
                jnp.zeros((pages, page_size // 4, 4, 128), jnp.uint8),
                jnp.asarray(indices),
                jnp.zeros(heads, jnp.float32),
                jax.tree.map(jax.device_put, meta),
                scale=width**-0.5,
                schedule=CSAAttentionSchedule(query_tile=32),
                window_size=page_size,
                compression_ratio=4,
                fp8_scale_block=64,
                rows_per_group=4,
            )
        ),
    )
    # Zero logits give uniform weights, including one zero-valued sink.
    window_count = np.minimum(position + 1, page_size).astype(np.float32)
    expected = window_count / (window_count + count.astype(np.float32) + 1)
    for start in range(0, tokens, sequence):
        assert_close(
            output[start : start + sequence].astype(np.float32),
            np.broadcast_to(
                expected[start : start + sequence, None, None], (sequence, heads, width)
            ),
        )
    np.testing.assert_array_equal(window[0], 0)
    np.testing.assert_array_equal(window[1:], 1)
