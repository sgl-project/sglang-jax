"""MXFP4 group32 routed FFN with BF16 tile decoding and explicit MXU ACC."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .fp4_common import activation, decode_tile_layout
from .fp4_common import start_panel_dma as _dma
from .fp4_common import wait_panel_dma

BK = BN = 256
FP4_VARIANT = "g32"
GROUP_SIZE = 32
start_panel_dma = partial(_dma, group_size=GROUP_SIZE)


def require_acc_support():
    """Keep the optional MXFP4 route from failing deep inside TPU lowering."""
    required = ("ACC", "matmul_push_rhs", "matmul_acc_lhs", "matmul_pop")
    if any(not hasattr(pltpu, name) for name in required):
        raise RuntimeError(
            "MXFP4 group32 requires the Pallas TPU ACC primitives available in "
            "JAX/jaxlib 0.11.2 (validated version). The existing BF16/FP8 paths "
            "do not require these primitives."
        )


def validate_inputs(tokens, w1, w2, w3, s1, s2, s3, weights, ids, ep_size, top_k):
    if tokens.ndim != 2 or tokens.dtype != jnp.bfloat16:
        raise TypeError("FP4 v2 requires rank-2 BF16 tokens (W4A16)")
    if any(w.dtype != jnp.uint8 or w.ndim != 3 for w in (w1, w2, w3)):
        raise TypeError("FP4 weights require rank-3 uint8 [E,K/2,N]")
    e, h2, intermediate = w1.shape
    hidden = tokens.shape[1]
    if hidden != 2 * h2 or w3.shape != w1.shape or w2.shape != (e, intermediate // 2, hidden):
        raise ValueError("Inconsistent logical Gate/Up/Down dimensions")
    if hidden % BK or intermediate % BK or e % ep_size or tokens.shape[0] % ep_size:
        raise ValueError("H/I must align to 256; E/T must divide the EP group")
    for scale, shape in [
        (s1, (e, hidden // 32, intermediate)),
        (s3, (e, hidden // 32, intermediate)),
        (s2, (e, intermediate // 32, hidden)),
    ]:
        if scale is None or scale.dtype != jnp.uint8 or scale.shape != shape:
            raise ValueError(f"Expected E8M0 uint8 scale {shape}")
    if not 0 < top_k <= e or ids.shape != (tokens.shape[0], top_k) or weights.shape != ids.shape:
        raise ValueError("Expected token-aligned [T,top_k] routing arrays")
    if ids.dtype != jnp.int32 or weights.dtype != jnp.float32:
        raise TypeError("Routing IDs/weights must be int32/float32")
    return e, intermediate, hidden


def _issue(lhs, rhs, acc, *, interpret, reuse_rhs=False):
    if interpret:
        acc[...] = acc[...] + jnp.dot(lhs, rhs, preferred_element_type=jnp.float32)
    else:
        if not reuse_rhs:
            pltpu.matmul_push_rhs(rhs, staging_register=0, mxu_index=0)
        pltpu.matmul_acc_lhs(acc, lhs, load_staged_rhs=None if reuse_rhs else 0)


def _pop(acc, *, interpret):
    if interpret:
        value = acc[...]
        acc[...] = jnp.zeros(acc.shape, jnp.float32)
        return value
    return pltpu.matmul_pop(acc)


def gate_up_k256(
    x,
    gate,
    up,
    gate_packed,
    gate_scale,
    up_packed,
    up_scale,
    expanded,
    unused_scale_scratch,
    *,
    hidden,
    bts,
    btc,
    valid_f=None,
    interpret=False,
):
    """Projection -> N -> K -> rows; independent MxN accumulators retained across full H.

    `interpret` emulates only the ACC primitives for local Pallas tests. The
    distributed entrypoint never enables it. No per-K result is materialized.
    """
    del unused_scale_scratch
    valid_f = gate.shape[1] if valid_f is None else valid_f

    def run(*accs):
        if interpret:
            for acc in accs:
                acc[...] = jnp.zeros(acc.shape, jnp.float32)
        for packed, scales, output in (
            (gate_packed, gate_scale, gate),
            (up_packed, up_scale, up),
        ):
            for n in range(0, valid_f, BN):

                def decode(k, slot, packed=packed, scales=scales, n=n):
                    with jax.named_scope("g32_decode_8groups"):
                        expanded.at[slot][...] = decode_tile_layout(
                            packed[pl.ds(k // 2, 128), pl.ds(n, BN)],
                            scales[pl.ds(k // 32, 8), pl.ds(n, BN)],
                        )

                decode(0, 0)
                for k in range(0, hidden, BK):
                    slot = (k // BK) % 2
                    for m in range(0, bts, btc):
                        with jax.named_scope("g32_accumulate_k256"):
                            _issue(
                                x[
                                    pl.ds(m, btc),
                                    k // (hidden // 2),
                                    pl.ds(k % (hidden // 2), BK),
                                ],
                                expanded[slot],
                                accs[m // btc],
                                interpret=interpret,
                                reuse_rhs=m > 0,
                            )
                    if k + BK < hidden:
                        decode(k + BK, 1 - slot)
                for m in range(0, bts, btc):
                    with jax.named_scope("g32_pop_full_h"):
                        output.at[pl.ds(m, btc), pl.ds(n, BN)][...] = _pop(
                            accs[m // btc], interpret=interpret
                        )

    memory = pltpu.VMEM if interpret else pltpu.ACC(0)
    pl.run_scoped(run, *(memory((btc, BN), jnp.float32) for _ in range(bts // btc)))


def activate_down_k256(
    gate,
    up,
    route,
    y_acc,
    down_packed,
    down_scale,
    expanded,
    *,
    hidden,
    bts,
    btc,
    bf_id,
    limit,
    valid_f=None,
    interpret=False,
):
    """Keep the original btc/activation/panel boundary; pop once per panel."""
    valid_f = gate.shape[1] if valid_f is None else valid_f

    def run(acc):
        if interpret:
            acc[...] = jnp.zeros(acc.shape, jnp.float32)
        for m in range(0, bts, btc):
            z = activation(
                gate[pl.ds(m, btc), pl.ds(0, valid_f)],
                up[pl.ds(m, btc), pl.ds(0, valid_f)],
                route[pl.ds(m, btc), 0:1],
                limit,
            )
            for n in range(0, hidden, BN):

                def decode(k, slot, n=n):
                    with jax.named_scope("g32_down_decode_8groups"):
                        expanded.at[slot][...] = decode_tile_layout(
                            down_packed[pl.ds(k // 2, 128), pl.ds(n, BN)],
                            down_scale[pl.ds(k // 32, 8), pl.ds(n, BN)],
                        )

                decode(0, 0)
                for k in range(0, valid_f, BK):
                    slot = (k // BK) % 2
                    _issue(z[:, pl.ds(k, BK)], expanded[slot], acc, interpret=interpret)
                    if k + BK < valid_f:
                        decode(k + BK, 1 - slot)
                value = _pop(acc, interpret=interpret)
                target = y_acc.at[pl.ds(m, btc), n // (hidden // 2), pl.ds(n % (hidden // 2), BN)]
                target[...] = value if bf_id == 0 else target[...] + value

    memory = pltpu.VMEM if interpret else pltpu.ACC(0)
    pl.run_scoped(run, memory((btc, BN), jnp.float32))


__all__ = [
    "activate_down_k256",
    "gate_up_k256",
    "start_panel_dma",
    "validate_inputs",
    "wait_panel_dma",
]
