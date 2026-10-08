"""DSV4_COMPRESSOR_ROW_SHARD: per-device row-block compression + all-gathered records equals the
full-chunk compress_chunk (CPU, 8 host devices, shard_map over 'tensor').

Runs in its own process: the XLA_FLAGS / JAX_PLATFORMS / DSV4_* settings below must be
in place before jax initialises, and they stay set for anything imported afterwards
(run_suite starts one process per file)."""

import os
import types

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["PALLAS_INTERPRET"] = "1"
os.environ["DSV4_COMPRESSOR_ROW_SHARD"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

if len(jax.devices()) < 8:
    pytest.skip(
        "needs 8 host devices (XLA_FLAGS set before jax initialised)", allow_module_level=True
    )

from sgl_jax.srt.layers.attention.dsv4 import dispatch
from sgl_jax.srt.layers.attention.dsv4.compressor import compress_chunk

HIDDEN = 256
RATIO = 4
WINDOW = 8
SLOTS = 3


def _rope_cache(max_pos, dim=64):
    half = dim // 2
    inv = 1.0 / (10000 ** (np.arange(half) / half))
    ang = np.arange(max_pos)[:, None] * inv[None, :]
    return jnp.asarray(np.concatenate((np.cos(ang), np.sin(ang)), axis=1), jnp.float32)


def _weights(key, head_dim, cache):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    width = 2 * head_dim
    wkv = (jax.random.normal(k1, (width, HIDDEN)) * 0.05).astype(jnp.bfloat16)
    wgate = (jax.random.normal(k2, (width, HIDDEN)) * 0.05).astype(jnp.bfloat16)
    ape = jax.random.normal(k3, (RATIO, width)) * 0.1
    norm = 1.0 + 0.1 * jax.random.normal(k4, (head_dim,))
    return dict(wkv=wkv, wgate=wgate, ape=ape, norm_weight=norm, cos_sin_cache=cache)


def _chunk(key, tokens, q_len, chunk_start):
    positions = jnp.where(jnp.arange(tokens) < q_len, chunk_start + jnp.arange(tokens), 0)
    groups = tokens // RATIO
    full = q_len // RATIO
    bidx = jnp.where(jnp.arange(groups) < full, RATIO * jnp.arange(groups) + RATIO - 1, tokens)
    bvalid = jnp.arange(groups) < full
    bpos = jnp.where(bvalid, chunk_start + RATIO * jnp.arange(groups), 0)
    x = (jax.random.normal(key, (tokens, HIDDEN))).astype(jnp.bfloat16)
    return x, positions, bidx, bvalid, bpos


@pytest.mark.parametrize(
    "tokens,q_len,chunk_start,head_dim,cap_pad",
    [(64, 64, 0, 512, 0), (64, 61, 64, 128, 1), (512, 300, 8192, 512, 1), (128, 128, 68, 128, 3)],
)
@pytest.mark.parametrize("local", [False, True])
def test_row_sharded_compress_matches_full(tokens, q_len, chunk_start, head_dim, cap_pad, local):
    key = jax.random.PRNGKey(tokens * 3 + q_len)
    kx, kw, ks = jax.random.split(key, 3)
    cache = _rope_cache(16384)
    w = _weights(kw, head_dim, cache)
    state = jax.random.normal(ks, (SLOTS, WINDOW, 2 * 2 * head_dim))
    x, positions, bidx, bvalid, bpos = _chunk(kx, tokens, q_len, chunk_start)
    if cap_pad:  # real metadata pads the slot axis to T // ratio + B
        bidx = jnp.concatenate([bidx, jnp.full((cap_pad,), tokens, jnp.int32)])
        bvalid = jnp.concatenate([bvalid, jnp.zeros((cap_pad,), bool)])
        bpos = jnp.concatenate([bpos, jnp.zeros((cap_pad,), jnp.int32)])
    slot = 1
    md = types.SimpleNamespace(
        cu_q_lens=jnp.asarray([0, q_len], jnp.int32),
        prefix_lens=jnp.asarray([chunk_start], jnp.int32),
        request_slots=jnp.asarray([slot], jnp.int32),
        query_positions=positions,
        query_request_ids=jnp.zeros((tokens,), jnp.int32),
    )
    rmd = types.SimpleNamespace(boundary_token_indices=bidx, boundary_valid_mask=bvalid)
    r, v, s = compress_chunk(
        x,
        state=state,
        positions=positions,
        query_request_ids=md.query_request_ids,
        prefix_lens=md.prefix_lens,
        cu_q_lens=md.cu_q_lens,
        state_slots=md.request_slots,
        boundary_token_indices=bidx,
        boundary_valid_mask=bvalid,
        boundary_compressed_pos=bpos,
        ratio=RATIO,
        head_dim=head_dim,
        **w,
    )
    mesh = Mesh(np.asarray(jax.devices()[:8]), axis_names=("tensor",))

    def body(x, state):
        plan = dispatch._compressor_row_shard_plan(x, md, rmd, RATIO)
        assert plan is not None and plan["local"] == local
        return dispatch._row_sharded_compress(
            x,
            state=state,
            weights=w,
            rope_positions=bpos,
            metadata=md,
            ratio_md=rmd,
            ratio=RATIO,
            head_dim=head_dim,
            rope_head_dim=64,
            norm_eps=1e-6,
            **plan,
        )

    # local mode (DSV4_LOWRANK_AG): each device receives only its own row block of x
    x_spec = P("tensor", None) if local else P()
    gr, gv, gs = jax.jit(
        jax.shard_map(body, mesh=mesh, in_specs=(x_spec, P()), out_specs=P(), check_vma=False)
    )(x, state)
    np.testing.assert_array_equal(np.asarray(gv), np.asarray(v))
    valid = np.asarray(v)
    np.testing.assert_allclose(np.asarray(gr)[valid], np.asarray(r)[valid], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(gs), np.asarray(s), rtol=1e-6, atol=1e-6)
