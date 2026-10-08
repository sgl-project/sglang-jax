"""CSA decode attention over gathered keys (Pallas TPU).

After the indexer selected a decode query's compressed entries, the query attends
the sliding-window rows plus those entries (keys double as values) and the
attention sink. The XLA form cast both gathered key blocks to f32, masked them
(``[64, 640, 512]`` f32 = 84 MB per CSA layer at bs=64) and ran two f32 einsums:
about 1.5 ms of a 36 ms decode step over the 21 CSA layers on v7x. This kernel
keeps the keys bf16 in VMEM, forms the scores on the MXU with f32 accumulation and
multiplies the f32 probabilities against the keys upcast in VMEM, so the maths
matches the XLA path up to accumulation order.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_NEG_INF = jnp.finfo(jnp.float32).min
ROWS_ENV = "DSV4_CSA_DECODE_ATTN_ROWS"
DEFAULT_ROWS = 4


def _kernel(q_ref, window_ref, comp_ref, bias_ref, sink_ref, out_ref, *, softmax_scale, rows):
    sink = sink_ref[...].astype(jnp.float32)[:, None]  # [H, 1]
    window_len = window_ref.shape[1]
    for r in range(rows):
        q = q_ref[r]  # [H, D] bf16
        bias = bias_ref[r]  # [1, K] f32: 0 valid, finite minimum invalid
        window = window_ref[r]  # [W, D] bf16
        comp = comp_ref[r]  # [S, D] bf16

        def scores(keys, q=q):
            return lax.dot_general(
                q, keys, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )

        window_scores = scores(window) * jnp.float32(softmax_scale)
        comp_scores = scores(comp) * jnp.float32(softmax_scale)
        window_bias = bias[:, :window_len]
        comp_bias = bias[:, window_len:]
        window_scores = jnp.where(window_bias > _NEG_INF, window_scores, _NEG_INF)
        comp_scores = jnp.where(comp_bias > _NEG_INF, comp_scores, _NEG_INF)
        shift = jnp.maximum(
            jnp.maximum(
                jnp.max(window_scores, axis=-1, keepdims=True),
                jnp.max(comp_scores, axis=-1, keepdims=True),
            ),
            sink,
        )
        window_probs = jnp.where(window_bias > _NEG_INF, jnp.exp(window_scores - shift), 0.0)
        comp_probs = jnp.where(comp_bias > _NEG_INF, jnp.exp(comp_scores - shift), 0.0)
        denominator = (
            jnp.sum(window_probs, axis=-1, keepdims=True)
            + jnp.sum(comp_probs, axis=-1, keepdims=True)
            + jnp.exp(sink - shift)
        )
        value = lax.dot_general(
            window_probs,
            window.astype(jnp.float32),
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ) + lax.dot_general(
            comp_probs,
            comp.astype(jnp.float32),
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        out_ref[r] = value / denominator


def gathered_decode_attention(
    q,
    window,
    compressed,
    window_valid,
    selected_valid,
    attention_sink,
    *,
    softmax_scale,
    rows_per_step=None,
    interpret=None,
):
    """Softmax attention of ``q`` [T,H,D] over ``window`` [T,W,D] ++ ``compressed`` [T,S,D]
    (bf16 keys, also the values) with per-row validity masks and the sink logit;
    returns f32 [T,H,D]. Rows whose masks are all false attend only the sink (zero output).
    """
    tokens, heads, dim = q.shape
    window_len, selected = window.shape[1], compressed.shape[1]
    if window.shape[0] != tokens or compressed.shape[0] != tokens:
        raise ValueError("q, window and compressed must share T")
    if window.shape[2] != dim or compressed.shape[2] != dim:
        raise ValueError("keys must share the query head dimension")
    if window_valid.shape != (tokens, window_len) or selected_valid.shape != (tokens, selected):
        raise ValueError("validity masks must be [T,W] and [T,S]")
    if attention_sink.shape != (heads,):
        raise ValueError("attention_sink must be [H]")
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    rows = int(os.environ.get(ROWS_ENV, DEFAULT_ROWS)) if rows_per_step is None else rows_per_step
    rows = max(1, min(rows, tokens))
    padded = -(-tokens // rows) * rows
    pad = (0, padded - tokens)
    mask = jnp.concatenate((window_valid, selected_valid), axis=1)
    bias = jnp.where(mask, 0.0, _NEG_INF).astype(jnp.float32)[:, None, :]
    q = jnp.pad(q.astype(jnp.bfloat16), (pad, (0, 0), (0, 0)))
    window = jnp.pad(window.astype(jnp.bfloat16), (pad, (0, 0), (0, 0)))
    compressed = jnp.pad(compressed.astype(jnp.bfloat16), (pad, (0, 0), (0, 0)))
    bias = jnp.pad(bias, (pad, (0, 0), (0, 0)), constant_values=_NEG_INF)
    total = window_len + selected
    out = pl.pallas_call(
        functools.partial(_kernel, softmax_scale=float(softmax_scale), rows=rows),
        grid=(padded // rows,),
        in_specs=[
            pl.BlockSpec((rows, heads, dim), lambda s: (s, 0, 0)),
            pl.BlockSpec((rows, window_len, dim), lambda s: (s, 0, 0)),
            pl.BlockSpec((rows, selected, dim), lambda s: (s, 0, 0)),
            pl.BlockSpec((rows, 1, total), lambda s: (s, 0, 0)),
            pl.BlockSpec((heads,), lambda s: (0,)),
        ],
        out_specs=pl.BlockSpec((rows, heads, dim), lambda s: (s, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((padded, heads, dim), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=48 * 1024 * 1024
        ),
        interpret=interpret,
        name=f"csa-decode-gathered-attention-h{heads}-d{dim}-w{window_len}-s{selected}-r{rows}",
    )(q, window, compressed, bias, attention_sink.astype(jnp.float32))
    return out[:tokens]
