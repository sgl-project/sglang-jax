"""Fused masked attention for the DSv4 CSA prefill path (Pallas TPU).

The reference (`layers/attention/dsv4/attention.py::dsv4_dense_attention`) scores every
query against every gathered key (`[T, H, N]` f32), masks, takes a softmax with the
attention sink and multiplies back: three full passes over a `[T, H, N]` tensor
per layer. This kernel streams key tiles past query blocks with an online softmax,
never materialising the scores, and skips key tiles that no query of the block
may attend (the sliding-window band and unselected history).

Semantics, per query ``t`` and head ``h`` (``m`` = admissible mask, ``s`` = sink):

    p_e = exp(scale * q.k_e - M) * m[t, e]      M = max(max_e scale*q.k_e, s_h)
    out = sum_e p_e k_e / (sum_e p_e + exp(s_h - M))

which is the reference with the sink folded in as a pseudo-key without a value.
The running maximum is initialised to the sink, so nothing ever underflows to a
NaN and a query that may attend nothing returns exactly zero.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(
    q_ref,
    k_ref,
    mask_ref,
    sink_ref,
    out_ref,
    m_sc,
    l_sc,
    acc_sc,
    *,
    heads,
    scale,
):
    j = pl.program_id(1)
    tq_h, tk = q_ref.shape[0], k_ref.shape[0]
    tq = tq_h // heads

    @pl.when(j == 0)
    def _init():
        m_sc[...] = sink_ref[...]  # [tq*H, 1]: the sink is the first "key"
        l_sc[...] = jnp.ones_like(l_sc)
        acc_sc[...] = jnp.zeros_like(acc_sc)

    mask = mask_ref[...] != 0  # [tq, tk]

    @pl.when(jnp.any(mask))
    def _tile():
        q = q_ref[...]
        k = k_ref[...]
        s = jax.lax.dot_general(
            q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )  # [tq*H, tk]
        s = s * scale
        s3 = s.reshape(tq, heads, tk)
        s3 = jnp.where(mask[:, None, :], s3, -jnp.inf)
        m_prev = m_sc[...].reshape(tq, heads, 1)
        m_new = jnp.maximum(m_prev, jnp.max(s3, axis=-1, keepdims=True))
        alpha = jnp.exp(m_prev - m_new)  # m_new >= sink > -inf: always finite
        p3 = jnp.exp(s3 - m_new)  # masked entries: exp(-inf) == 0
        l_new = alpha * l_sc[...].reshape(tq, heads, 1) + jnp.sum(p3, axis=-1, keepdims=True)
        # f32 probabilities against f32-upcast keys: the reference einsum promotes the
        # bf16 keys to f32 here too, so the two agree to f32 rounding.
        p = p3.reshape(tq_h, tk)
        pv = jax.lax.dot_general(
            p,
            k.astype(jnp.float32),
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )  # [tq*H, D]
        acc_sc[...] = acc_sc[...] * alpha.reshape(tq_h, 1) + pv
        m_sc[...] = m_new.reshape(tq_h, 1)
        l_sc[...] = l_new.reshape(tq_h, 1)

    @pl.when(j == pl.num_programs(1) - 1)
    def _finish():
        out_ref[...] = (acc_sc[...] / l_sc[...]).astype(out_ref.dtype)


def csa_flash_attention(
    q,
    keys,
    mask,
    sink,
    *,
    sm_scale: float,
    block_q: int = 256,
    block_k: int = 512,
    interpret: bool = False,
):
    """``q`` [T, H, D], ``keys`` [N, D], ``mask`` [T, N] (nonzero = admissible),
    ``sink`` [H] f32 -> [T, H, D] f32. ``T`` is padded to ``block_q`` and ``N`` to
    ``block_k`` internally (padded keys are masked)."""
    q = jnp.asarray(q)
    keys = jnp.asarray(keys)
    T, H, D = q.shape
    N = keys.shape[0]
    if keys.shape[1] != D or mask.shape != (T, N) or sink.shape != (H,):
        raise ValueError(
            f"shape mismatch: q{q.shape} keys{keys.shape} mask{mask.shape} sink{sink.shape}"
        )
    kdt = keys.dtype if keys.dtype in (jnp.bfloat16, jnp.float32) else jnp.bfloat16
    tq = min(block_q, -(-T // 8) * 8)
    tk = min(block_k, -(-N // 128) * 128)
    Tp = -(-T // tq) * tq
    Np = -(-N // tk) * tk
    q2 = jnp.pad(q.astype(kdt), ((0, Tp - T), (0, 0), (0, 0))).reshape(Tp * H, D)
    k2 = jnp.pad(keys.astype(kdt), ((0, Np - N), (0, 0)))
    m2 = jnp.pad(jnp.asarray(mask).astype(jnp.int8), ((0, Tp - T), (0, Np - N)))
    sink_rows = jnp.tile(jnp.asarray(sink, jnp.float32), tq)[:, None]  # [tq*H, 1]
    kernel = functools.partial(_kernel, heads=H, scale=float(sm_scale))
    out = pl.pallas_call(
        kernel,
        grid=(Tp // tq, Np // tk),
        in_specs=[
            pl.BlockSpec((tq * H, D), lambda i, j: (i, 0)),
            pl.BlockSpec((tk, D), lambda i, j: (j, 0)),
            pl.BlockSpec((tq, tk), lambda i, j: (i, j)),
            pl.BlockSpec((tq * H, 1), lambda i, j: (0, 0)),
        ],
        out_specs=pl.BlockSpec((tq * H, D), lambda i, j: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((Tp * H, D), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((tq * H, 1), jnp.float32),
            pltpu.VMEM((tq * H, 1), jnp.float32),
            pltpu.VMEM((tq * H, D), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
        interpret=interpret,
    )(q2, k2, m2, sink_rows)
    return out.reshape(Tp, H, D)[:T]
