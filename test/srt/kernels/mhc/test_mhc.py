"""Correctness tests for the mHC kernels.

Correctness is against ``ref.py``, numpy with no JAX import, so agreement is
evidence about the semantics rather than about two programs sharing a lowering.

Tolerances follow provenance, not storage dtype.  ``Precision.DEFAULT``
truncates the gate projection operands to bf16 on TPU, so every downstream gate
carries bf16 error despite fp32 storage.  ``Precision.HIGHEST`` is covered as a
separate six-pass path.  A JAX reference would hide the default-path error by
truncating in the same place.

mHC is per-token, so a token count is the only shape that matters -- a batch is
just a longer packed sequence.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sgl_jax.srt.kernels.mhc import (
    mhc_gates,
    mhc_head_collapse_fused,
    mhc_post_fused,
    mhc_pre_fused,
)

from . import ref

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "tpu", reason="mHC kernels require real Mosaic lowering"
)

# DeepSeek-V4-Flash shipped configuration.
HC, HIDDEN, ITERS, EPS = 4, 4096, 20, 1e-6
# Anything downstream of the bf16 projection: the repo convention for bf16
# kernels, which these clear by 5x.
PROJECTED = {"rtol": 2e-2, "atol": 1e-2}
# The Sinkhorn kernel in isolation: fp32 in, fp32 out, nothing truncated.
EXACT = {"rtol": 1e-5, "atol": 1e-6}

# Powers of two spanning the tuned Pallas tile sizes, plus counts that are a
# multiple of no block size.  The latter exercise the pad-then-slice-back path,
# where an error would corrupt only the final block.
TOKENS = [128, 256, 512, 1024, 2048, 4096, 8192]
RAGGED_TOKENS = [1, 127, 1000]


def _inputs(n, hc=HC, hidden=HIDDEN, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    mix_hc = ref.mix_hc_width(hc)
    return {
        "x": (jax.random.normal(keys[0], (n, hc, hidden), jnp.float32) * 0.1).astype(
            jnp.bfloat16
        ),
        "fn": jax.random.normal(keys[1], (mix_hc, hc * hidden), jnp.float32) * 0.01,
        "head_fn": jax.random.normal(keys[2], (hc, hc * hidden), jnp.float32) * 0.01,
        "scale": jnp.asarray([0.7, 1.1, 0.9], jnp.float32),
        "base": jax.random.normal(keys[3], (mix_hc,), jnp.float32) * 0.05,
        "head_scale": jnp.asarray([0.8], jnp.float32),
        "head_base": jax.random.normal(keys[4], (hc,), jnp.float32) * 0.05,
        "block_out": (
            jax.random.normal(keys[5], (n, hidden), jnp.float32) * 0.1
        ).astype(jnp.bfloat16),
        "mixes": jax.random.normal(keys[0], (n, mix_hc), jnp.float32),
    }


def _close(got, want, label, tol=PROJECTED):
    for index, (a, b) in enumerate(zip(got, want)):
        np.testing.assert_allclose(
            np.asarray(a, np.float32),
            np.asarray(b, np.float32),
            err_msg=f"{label}[{index}]",
            **tol,
        )


@pytest.mark.parametrize("n", TOKENS + RAGGED_TOKENS)
def test_gates_match_reference(n):
    """Gate kernel: [n, mix_hc] -> post, comb."""
    d = _inputs(n)
    kw = {"hc_mult": HC, "sinkhorn_iters": ITERS, "eps": EPS}
    _close(
        mhc_gates(d["mixes"], d["scale"], d["base"], **kw),
        ref.sinkhorn_gates(d["mixes"], d["scale"], d["base"], **kw)[1:],
        f"gates n={n}",
        EXACT,
    )


@pytest.mark.parametrize("n", TOKENS + RAGGED_TOKENS)
def test_pre_matches_reference(n):
    """Pre-block mixing: collapse hc streams to one and emit the gates."""
    d = _inputs(n)
    args = (d["x"], d["fn"], d["scale"], d["base"])
    kw = {"hc_mult": HC, "sinkhorn_iters": ITERS, "norm_eps": EPS, "hc_eps": EPS}
    _close(mhc_pre_fused(*args, **kw), ref.pre(*args, **kw), f"pre n={n}")


@pytest.mark.parametrize("n", TOKENS + RAGGED_TOKENS)
def test_post_matches_reference(n):
    """Post-block mixing: expand one stream back to hc and remix the residual."""
    d = _inputs(n)
    kw = {"hc_mult": HC, "sinkhorn_iters": ITERS, "norm_eps": EPS, "hc_eps": EPS}
    _, post, comb = ref.pre(d["x"], d["fn"], d["scale"], d["base"], **kw)
    post = jnp.asarray(post, jnp.float32)
    comb = jnp.asarray(comb, jnp.float32)
    _close(
        [mhc_post_fused(d["block_out"], d["x"], post, comb)],
        [ref.post(d["block_out"], d["x"], post, comb)],
        f"post n={n}",
    )


@pytest.mark.parametrize("n", TOKENS + RAGGED_TOKENS)
def test_head_matches_reference(n):
    """Head collapse: the final hc -> 1 before the LM head."""
    d = _inputs(n)
    args = (d["x"], d["head_fn"], d["head_scale"], d["head_base"])
    _close(
        [mhc_head_collapse_fused(*args, hc_mult=HC, norm_eps=EPS, hc_eps=EPS)],
        [ref.head_collapse(*args, norm_eps=EPS, hc_eps=EPS)],
        f"head n={n}",
    )


@pytest.mark.parametrize("hidden", [4096, 7168])
def test_highest_precision_matches_reference(hidden):
    """The six-pass projection fits VMEM and preserves the head BF16 boundary."""
    d = _inputs(256, hidden=hidden)
    pre_args = (d["x"], d["fn"], d["scale"], d["base"])
    pre_kw = {
        "hc_mult": HC,
        "sinkhorn_iters": ITERS,
        "norm_eps": EPS,
        "hc_eps": EPS,
    }
    _close(
        mhc_pre_fused(*pre_args, **pre_kw, dot_precision=jax.lax.Precision.HIGHEST),
        ref.pre(*pre_args, **pre_kw),
        f"pre highest hidden={hidden}",
    )

    _, post, comb = ref.pre(*pre_args, **pre_kw)
    for backend in ("xla", "pallas"):
        _close(
            [
                mhc_post_fused(
                    d["block_out"],
                    d["x"],
                    jnp.asarray(post, jnp.float32),
                    jnp.asarray(comb, jnp.float32),
                    backend=backend,
                    precision=jax.lax.Precision.HIGHEST,
                )
            ],
            [ref.post(d["block_out"], d["x"], post, comb)],
            f"post highest backend={backend} hidden={hidden}",
        )

    head_args = (d["x"], d["head_fn"], d["head_scale"], d["head_base"])
    _close(
        [
            mhc_head_collapse_fused(
                *head_args,
                hc_mult=HC,
                norm_eps=EPS,
                hc_eps=EPS,
                dot_precision=jax.lax.Precision.HIGHEST,
            )
        ],
        [ref.head_collapse(*head_args, norm_eps=EPS, hc_eps=EPS)],
        f"head highest hidden={hidden}",
    )


@pytest.mark.parametrize("hidden", [4096, 7168])
@pytest.mark.parametrize("hc", [2, 4, 8])
def test_shapes_beyond_the_shipped_config(hc, hidden):
    """Nothing is wired to hc_mult=4 or hidden=4096: a wider model must select a
    smaller block rather than overflow scoped VMEM."""
    d = _inputs(512, hc=hc, hidden=hidden)
    args = (d["x"], d["fn"], d["scale"], d["base"])
    kw = {"hc_mult": hc, "sinkhorn_iters": ITERS, "norm_eps": EPS, "hc_eps": EPS}
    _close(
        mhc_pre_fused(*args, **kw), ref.pre(*args, **kw), f"pre hc={hc} hidden={hidden}"
    )


def test_f32_activations_fit_vmem():
    d = _inputs(512)
    x = d["x"].astype(jnp.float32)
    args = (x, d["fn"], d["scale"], d["base"])
    kw = {"hc_mult": HC, "sinkhorn_iters": ITERS, "norm_eps": EPS, "hc_eps": EPS}
    _close(mhc_pre_fused(*args, **kw), ref.pre(*args, **kw), "pre f32")


@pytest.mark.parametrize("iters", [1, 2, 40])
def test_iteration_counts_other_than_the_shipped_twenty(iters):
    d = _inputs(512)
    kw = {"hc_mult": HC, "sinkhorn_iters": iters, "eps": EPS}
    _close(
        mhc_gates(d["mixes"], d["scale"], d["base"], **kw),
        ref.sinkhorn_gates(d["mixes"], d["scale"], d["base"], **kw)[1:],
        f"gates iters={iters}",
        EXACT,
    )


def test_comb_is_a_near_doubly_stochastic_mixing_matrix():
    """The property the Sinkhorn exists to establish.

    The schedule ends on a column pass, so columns are exact to ~1e-6 while rows
    stay approximate.  Entries must stay positive, or streams could cancel.
    """
    d = _inputs(2048)
    _, comb = mhc_gates(
        d["mixes"], d["scale"], d["base"], hc_mult=HC, sinkhorn_iters=ITERS, eps=EPS
    )
    comb = np.asarray(comb, np.float64)
    assert comb.min() > 0.0
    np.testing.assert_allclose(comb.sum(axis=-2), 1.0, atol=1e-5)
    np.testing.assert_allclose(comb.sum(axis=-1), 1.0, atol=0.2)
