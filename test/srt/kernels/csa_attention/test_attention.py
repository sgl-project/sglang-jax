"""Native resource buffers, independent joint-softmax oracle and write ownership."""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from sgl_jax.srt.kernels.csa_attention import CSAAttentionMetadata, csa_joint_attention
from sgl_jax.srt.kernels.csa_attention.tune import (
    CSAAttentionSchedule,
    get_csa_attention_schedule,
)
from sgl_jax.srt.mem_cache.deepseek_v4.pool import (
    DeepseekV4CacheSpec,
    DeepseekV4TokenToKVPool,
)

from .ref import reference, update_window


def make_case(lengths, prefixes, *, heads=8, pad=2, page_size=128):
    batch, dim, window = len(lengths), 512, 128
    rng = np.random.default_rng(71)
    ends = np.asarray(lengths, np.int32) + np.asarray(prefixes, np.int32)
    cu = np.cumsum([0, *lengths], dtype=np.int32)
    tokens = int(cu[-1]) + pad
    pages_per_request = max(1, (int(max(ends)) + page_size - 1) // page_size)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))
    with jax.set_mesh(mesh):
        pool = DeepseekV4TokenToKVPool(
            batch * pages_per_request * page_size,
            batch * page_size,
            page_size,
            DeepseekV4CacheSpec((4,)),
            mesh,
        )

    def values(shape):
        return rng.normal(0, 0.2, shape).astype(ml_dtypes.bfloat16)

    swa = values(pool.get_swa_buffer(0).shape)
    c4 = values(pool.get_compressed_buffer(0).shape)
    swa[:page_size] = np.nan
    c4[0] = np.nan
    q, new = values((tokens, heads, dim)), values((tokens, dim))
    req = np.r_[np.repeat(np.arange(batch), lengths), np.full(pad, -1)].astype(np.int32)
    wp = np.arange(batch, 0, -1, dtype=np.int32)
    wc = np.arange(batch + 1, dtype=np.int32) * page_size
    cp = np.arange(batch * pages_per_request, 0, -1, dtype=np.int32)
    cc = np.arange(batch + 1, dtype=np.int32) * pages_per_request * (page_size // 4)
    selected = np.full((tokens, 512), -1, np.int32)
    loc = np.full(tokens, -1, np.int32)
    for r, (length, prefix) in enumerate(zip(lengths, prefixes, strict=True)):
        available = min(512, int(ends[r] // 4))
        for t in range(cu[r], cu[r + 1]):
            # Unsorted and partly future entries exercise causal filtering.
            selected[t, :available] = rng.choice(
                max(1, int(ends[r] // 4)), available, replace=False
            )
            loc[t] = wp[r] * page_size + (prefix + t - cu[r]) % window
    md = CSAAttentionMetadata(req, cu, ends, wp, wc, cp, cc, ends // 4, loc)
    return (q, new, swa, c4, selected, rng.normal(0, 0.2, heads).astype(np.float32), md)


def check(args, tile=4, page_size=128, schedule=None):
    expected = reference(*args, scale=512**-0.5, window_page_size=page_size)
    expected_swa = update_window(args[1], args[2], args[-1], window_size=128, page_size=page_size)
    device = jax.tree.map(jnp.asarray, args)
    output, updated = jax.block_until_ready(
        csa_joint_attention(
            *device,
            scale=512**-0.5,
            schedule=schedule
            or (
                get_csa_attention_schedule("TPU v6e", decode=True)
                if np.all(np.diff(args[-1].cu_q_lens) <= 1)
                else CSAAttentionSchedule(query_tile=tile)
            ),
            window_page_size=page_size,
            interpret=jax.default_backend() != "tpu",
        )
    )
    actual = np.asarray(output).astype(np.float32)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=1e-2, equal_nan=False)
    np.testing.assert_array_equal(np.asarray(updated).view(np.uint16), expected_swa.view(np.uint16))
    # No in-place mutation, including the unowned compressed-cache family.
    for index in (2, 3):
        np.testing.assert_array_equal(
            np.asarray(device[index]).view(np.uint16), args[index].view(np.uint16)
        )
    return output, updated


@pytest.mark.parametrize(
    "lengths,prefixes,tile",
    [
        ((1,), (0,), 1),
        ((1,), (8191,), 1),
        ((1, 1, 1, 1), (0, 3, 127, 2047), 1),
        ((0, 1, 0, 1), (512, 127, 2048, 4095), 4),
        ((0, 0), (512, 8192), 4),
        ((33,), (0,), 4),
        ((133,), (127,), 4),
        ((19, 3, 0, 37), (127, 130, 1024, 8191), 4),
    ],
)
def test_native_attention(lengths, prefixes, tile):
    check(make_case(lengths, prefixes), tile)


def test_model_heads():
    check(make_case((5, 3), (2047, 4095), heads=64), 4)


@pytest.mark.parametrize("lengths", [(32, 32), (33, 7, 65, 0)])
def test_request_group_dispatch(lengths):
    check(make_case(lengths, (1023,) * len(lengths), heads=64), 32)


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32])
def test_decode_schedule(batch):
    args = make_case((1,) * batch, (8191,) * batch, heads=64)
    check(
        args,
        schedule=get_csa_attention_schedule("TPU v6e", decode=True),
    )


@pytest.mark.parametrize("prefixes", [(0, 2, 3, 127), (511, 1023, 2047, 4095)])
def test_decode_boundaries(prefixes):
    check(
        make_case((1,) * len(prefixes), prefixes, heads=64),
        schedule=get_csa_attention_schedule("TPU v6e", decode=True),
    )


def test_invalid_decode_schedule():
    with pytest.raises(ValueError, match="query_tile=1"):
        check(make_case((1,), (127,)), schedule=CSAAttentionSchedule(decode=True))


@pytest.mark.parametrize("page_size", [128, 256])
def test_decode_empty_and_missing_pages(page_size):
    args = make_case((1, 0, 1, 1), (0, 512, 127, 8191), page_size=page_size)
    args[-1].compressed_page_indices[::3] = 0
    args[-1].window_page_indices[2] = 0
    args[-1].window_write_locations[1] = -1
    args[4][2] = -1
    check(
        args,
        page_size=page_size,
        schedule=get_csa_attention_schedule("TPU v6e", decode=True),
    )


@pytest.mark.parametrize("tile", [1, 32])
def test_page_256_and_full_query_tile(tile):
    check(make_case((17, 3), (511, 1023), page_size=256), tile, page_size=256)


def test_suppressed_write():
    args = make_case((9,), (127,))
    args[-1].window_write_locations[::2] = -1
    check(args)


def test_empty_selection():
    args = make_case((7, 1), (0, 8191))
    args[4][:] = -1
    args[3][:] = np.nan
    check(args)


def test_zero_tokens():
    check(make_case((0, 0), (0, 0), pad=0))


def test_peaked_scores():
    args = list(make_case((19, 3), (8191, 1023), heads=64))
    for index in range(4):
        args[index] = (args[index].astype(np.float32) * 5).astype(ml_dtypes.bfloat16)
    check(tuple(args))


def test_missing_pages():
    args = make_case((9, 5), (8191, 2047))
    args[-1].compressed_page_indices[::2] = 0
    args[-1].window_page_indices[0] = 0
    args[-1].window_write_locations[:9] = -1
    check(args)


def test_ring_wrap_and_decode():
    args = make_case((257,), (127,))
    _, updated = check(args)
    following = list(make_case((1,), (384,)))
    following[2] = np.asarray(updated)
    check(
        tuple(following),
        schedule=get_csa_attention_schedule("TPU v6e", decode=True),
    )


def test_mask_mutation_is_detected():
    args = make_case((1,), (8191,))
    output, _ = check(args, 1)
    changed = list(args)
    changed[4] = np.full_like(args[4], -1)
    expected = reference(*changed, scale=512**-0.5)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            np.asarray(output).astype(np.float32), expected, rtol=2e-2, atol=1e-2
        )
