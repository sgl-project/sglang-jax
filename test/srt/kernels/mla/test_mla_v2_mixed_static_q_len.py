"""``mla_ragged_paged_attention(mixed_static_q_len=G)`` must size the MIXED
kernel's query block by G (scope name ``MLA-m-bq_G-...``) while leaving the
decode kernels untouched. Traces the dispatch on CPU with ``pallas_call``
replaced by a shape-only stub."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

import sgl_jax.srt.kernels.mla.v2.kernel as K

H, LKV, R, PAGE = 64, 512, 64, 128


def _dispatch_names(**kw):
    names = []

    def fake_pallas_call(kernel, *, out_shape, name=None, **_):
        names.append(name)

        def run(*args):
            return [jnp.zeros(s.shape, s.dtype) for s in out_shape]

        return run

    S, G = 2, 4
    T = S * G
    ql = jnp.zeros((T, H, LKV), jnp.bfloat16)
    qpe = jnp.zeros((T, H, R), jnp.bfloat16)
    nkv = jnp.zeros((T, LKV), jnp.bfloat16)
    nkpe = jnp.zeros((T, R), jnp.bfloat16)
    cache = jnp.zeros((8, PAGE // 2, 2, LKV + 128), jnp.bfloat16)
    kv_lens = jnp.array([20, 20], jnp.int32)
    page_indices = jnp.arange(S * 2, dtype=jnp.int32)
    cu_q = jnp.arange(S + 1, dtype=jnp.int32) * G
    cu_kv = jnp.arange(S + 1, dtype=jnp.int32) * (2 * PAGE)
    dist = jnp.array([0, 0, S], jnp.int32)
    with jax.disable_jit(), mock.patch.object(K.pl, "pallas_call", fake_pallas_call):
        K.mla_ragged_paged_attention(
            ql,
            qpe,
            nkv,
            nkpe,
            cache,
            kv_lens,
            page_indices,
            cu_q,
            cu_kv,
            dist,
            num_kv_pages_per_block=(16, 1, 16),
            num_queries_per_block=(1, 1, 256),
            **kw,
        )
    return names


def _bq(name):
    return int(name.split("-bq_")[1].split("-")[0])


def test_default_mixed_block_comes_from_block_params():
    names = _dispatch_names()
    mixed = [n for n in names if n.startswith("MLA-m-")]
    assert len(mixed) == 1 and _bq(mixed[0]) == 256


def test_mixed_static_q_len_sizes_mixed_query_block():
    names = _dispatch_names(mixed_static_q_len=4)
    mixed = [n for n in names if n.startswith("MLA-m-")]
    decode = [n for n in names if not n.startswith("MLA-m-")]
    assert len(mixed) == 1 and _bq(mixed[0]) == 4, names
    assert decode and all(_bq(n) == 1 for n in decode), names


def test_mixed_static_q_len_kept_off_for_sub_tile_heads_when_illegal():
    # 4 bf16 heads/shard: a token spans 4 of the 16 tile rows, so bq*heads must
    # cover whole 16x16 tile groups (256 rows); G=4 -> 16 rows is not in the
    # validated family and the tuned block must stay.
    names = []

    def fake_pallas_call(kernel, *, out_shape, name=None, **_):
        names.append(name)
        return lambda *a: [jnp.zeros(s.shape, s.dtype) for s in out_shape]

    S, G, h = 2, 4, 4
    T = S * G
    args = (
        jnp.zeros((T, h, LKV), jnp.bfloat16),
        jnp.zeros((T, h, R), jnp.bfloat16),
        jnp.zeros((T, LKV), jnp.bfloat16),
        jnp.zeros((T, R), jnp.bfloat16),
        jnp.zeros((8, PAGE // 2, 2, LKV + 128), jnp.bfloat16),
        jnp.array([20, 20], jnp.int32),
        jnp.arange(S * 2, dtype=jnp.int32),
        jnp.arange(S + 1, dtype=jnp.int32) * G,
        jnp.arange(S + 1, dtype=jnp.int32) * (2 * PAGE),
        jnp.array([0, 0, S], jnp.int32),
    )
    with jax.disable_jit(), mock.patch.object(K.pl, "pallas_call", fake_pallas_call):
        K.mla_ragged_paged_attention(
            *args,
            num_kv_pages_per_block=(16, 1, 16),
            num_queries_per_block=(1, 1, 64),
            mixed_static_q_len=G,
        )
    mixed = [n for n in names if n.startswith("MLA-m-")]
    assert len(mixed) == 1 and _bq(mixed[0]) == 64, names
