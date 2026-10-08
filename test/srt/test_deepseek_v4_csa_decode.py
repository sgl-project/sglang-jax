"""CSA long/short decode correctness and exact page-address lookup."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsv4.csa_decode_attention import gathered_decode_attention
from sgl_jax.srt.layers.attention.dsv4 import decode as m
from sgl_jax.srt.layers.attention.dsv4.decode import csa_decode_attention
from sgl_jax.srt.layers.attention.dsv4.ref.decode_attention import (
    gathered_decode_attention_ref,
)

NEG = float(jnp.finfo(jnp.float32).min)


def _reference(q, window, compressed, window_valid, selected_valid, sink, scale):
    q = np.asarray(q.astype(jnp.float32))
    keys = np.concatenate(
        (np.asarray(window.astype(jnp.float32)), np.asarray(compressed.astype(jnp.float32))), 1
    )
    mask = np.concatenate((np.asarray(window_valid), np.asarray(selected_valid)), 1)
    sink = np.asarray(sink, np.float32)
    out = np.zeros(q.shape, np.float32)
    for t in range(q.shape[0]):
        s = (q[t] @ keys[t].T) * scale  # [H, K]
        s = np.where(mask[t][None], s, NEG)
        shift = np.maximum(s.max(-1, keepdims=True), sink[:, None])
        p = np.where(mask[t][None], np.exp(s - shift), 0.0)
        den = p.sum(-1, keepdims=True) + np.exp(sink[:, None] - shift)
        out[t] = (p @ keys[t]) / den
    return out


@pytest.mark.parametrize("tokens,rows", [(3, 4), (9, 4), (5, 1)])
def test_kernel_matches_numpy_reference(tokens, rows):
    rng = np.random.default_rng(tokens)
    H, D, W, S = 8, 512, 128, 512
    q = jnp.asarray(rng.standard_normal((tokens, H, D)), jnp.bfloat16)
    window = jnp.asarray(rng.standard_normal((tokens, W, D)), jnp.bfloat16)
    compressed = jnp.asarray(rng.standard_normal((tokens, S, D)), jnp.bfloat16)
    window_valid = jnp.asarray(rng.random((tokens, W)) > 0.3)
    selected_valid = jnp.asarray(rng.random((tokens, S)) > 0.5)
    window_valid = window_valid.at[0].set(False)  # a padded row: sink only -> zeros
    selected_valid = selected_valid.at[0].set(False)
    sink = jnp.asarray(rng.standard_normal((H,)), jnp.float32)
    out = np.asarray(
        gathered_decode_attention(
            q,
            window,
            compressed,
            window_valid,
            selected_valid,
            sink,
            softmax_scale=D**-0.5,
            rows_per_step=rows,
            interpret=True,
        )
    )
    ref = _reference(q, window, compressed, window_valid, selected_valid, sink, D**-0.5)
    assert out.shape == (tokens, H, D)
    np.testing.assert_allclose(out, ref, rtol=2e-4, atol=2e-4)
    assert np.all(out[0] == 0)


def _long_path_inputs(seed=0):
    B, H, D, DIDX, PAGE, NPAGES, W, RATIO, TOPK = 5, 8, 512, 128, 8, 128, 128, 4, 512
    k = jax.random.split(jax.random.PRNGKey(seed), 8)
    total_pages = 1 + B * NPAGES
    rng = np.random.default_rng(seed)
    pages = np.zeros((B, NPAGES), np.int32)
    for b in range(B):
        pages[b] = 1 + b * NPAGES + np.arange(NPAGES)
    positions = jnp.asarray([4095, 3000, 0, 2049, 700], jnp.int32)
    valid = jnp.asarray([True, True, False, True, True])
    return dict(
        q=jax.random.normal(k[0], (B, H, D), jnp.bfloat16),
        index_q=jax.random.normal(k[1], (B, 64, DIDX), jnp.bfloat16),
        index_weights=jax.nn.softmax(jax.random.normal(k[2], (B, 64), jnp.float32), axis=-1),
        index_cache=jax.random.normal(k[3], (total_pages * PAGE, DIDX), jnp.bfloat16),
        compressed_cache=jax.random.normal(k[4], (total_pages * PAGE, D), jnp.bfloat16),
        window_cache=jax.random.normal(k[5], (4096, D), jnp.bfloat16),
        pages=jnp.asarray(pages),
        window_rows=jnp.asarray(rng.permutation(4096)[: B * W].reshape(B, W).astype(np.int32)),
        query_positions=positions,
        valid_token_mask=valid,
        attention_sink=jax.random.normal(k[6], (H,), jnp.float32),
        softmax_scale=D**-0.5,
        compressed_page_size=PAGE,
        index_topk=TOPK,
        ratio=RATIO,
    )


def test_decode_long_path_kernel_matches_xla(monkeypatch):
    monkeypatch.setenv("DSV4_DECODE_INDEXER_BACKEND", "p370")
    inputs = _long_path_inputs()
    assert inputs["pages"].shape[1] * inputs["compressed_page_size"] > inputs["index_topk"]
    with monkeypatch.context() as patch:
        patch.setattr(m, "gathered_decode_attention", gathered_decode_attention_ref)
        ref = np.asarray(csa_decode_attention(**inputs))
    out = np.asarray(csa_decode_attention(**inputs))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, ref, rtol=2e-4, atol=2e-4)
    assert np.all(out[2] == 0)


B, H, D, DIDX, PAGE, NPAGES, W, RATIO, TOPK = 3, 4, 128, 128, 8, 16, 16, 4, 512


def _short_path_inputs(seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 8)
    cap = NPAGES * PAGE  # 128 <= TOPK: the shortcut applies
    total_pages = 64
    q = jax.random.normal(k[0], (B, H, D), jnp.bfloat16)
    index_q = jax.random.normal(k[1], (B, H, DIDX), jnp.bfloat16)
    index_weights = jax.nn.softmax(jax.random.normal(k[2], (B, H), jnp.float32), axis=-1)
    index_cache = jax.random.normal(k[3], (total_pages, PAGE, DIDX), jnp.bfloat16)
    compressed_cache = jax.random.normal(k[4], (total_pages * PAGE, D), jnp.bfloat16)
    window_cache = jax.random.normal(k[5], (1024, D), jnp.bfloat16)
    rng = np.random.default_rng(seed)
    pages = jnp.asarray(
        rng.permutation(total_pages)[: B * NPAGES].reshape(B, NPAGES).astype(np.int32)
    )
    window_rows = jnp.asarray(rng.permutation(1024)[: B * W].reshape(B, W).astype(np.int32))
    positions = jnp.asarray([300, 40, 0], jnp.int32)  # lengths 75, 10, (padded)
    valid = jnp.asarray([True, True, False])
    sink = jax.random.normal(k[6], (H,), jnp.float32)
    assert cap <= TOPK
    return dict(
        q=q,
        index_q=index_q,
        index_weights=index_weights,
        index_cache=index_cache,
        compressed_cache=compressed_cache,
        window_cache=window_cache,
        pages=pages,
        window_rows=window_rows,
        query_positions=positions,
        valid_token_mask=valid,
        attention_sink=sink,
        softmax_scale=D**-0.5,
        compressed_page_size=PAGE,
        index_topk=TOPK,
        ratio=RATIO,
    )


def _run_short_path():
    return np.asarray(csa_decode_attention(**_short_path_inputs()).astype(jnp.float32))


def test_short_kv_kernel_matches_xla_bypass(monkeypatch):
    monkeypatch.setenv("DSV4_DECODE_SHORT_KV_KERNEL", "0")
    ref = _run_short_path()
    monkeypatch.setenv("DSV4_DECODE_SHORT_KV_KERNEL", "1")
    out = _run_short_path()
    assert np.isfinite(out).all()
    assert np.all(out[2] == 0)
    np.testing.assert_allclose(out, ref, rtol=2e-2, atol=2e-2)


def _take_pages(pages, idx, monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv("DSV4_DECODE_PAGE_TAKE", raising=False)
    else:
        monkeypatch.setenv("DSV4_DECODE_PAGE_TAKE", mode)
    return np.asarray(m._take_pages(jnp.asarray(pages), jnp.asarray(idx)))


@pytest.mark.parametrize(
    "rows,table,k,hi",
    [(3, 72, 1, 300), (8, 512, 640, 70000), (2, 130, 5, (1 << 24) - 1)],
)
def test_take_pages_onehot_exact_for_awkward_shapes_and_large_ids(monkeypatch, rows, table, k, hi):
    rng = np.random.default_rng(1)
    pages = rng.integers(0, hi + 1, size=(rows, table), dtype=np.int32)
    pages[0, 0] = hi  # the largest id must survive the byte split
    idx = rng.integers(0, table, size=(rows, k), dtype=np.int32)
    idx[0, 0] = 0
    got = _take_pages(pages, idx, monkeypatch, "onehot")
    np.testing.assert_array_equal(got, np.take_along_axis(pages, idx, axis=1))
