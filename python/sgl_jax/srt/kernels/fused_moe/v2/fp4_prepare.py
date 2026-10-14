"""One-time physical panel padding. Call outside repeated MoE timing/execution."""

import jax.numpy as jnp


def prepare_fp4_weights(w1, w2, w3, s1, s2, s3, *, bf, group_size=32):
    """Return positional weights and keyword metadata; no numerical requantization.

    Input ABI is the unpadded canonical layout. Cache per bf configuration; use
    the returned logical length so padded channels never become valid work.
    """
    if group_size != 32:
        raise ValueError("Only E8M0 group32 weights are supported")
    if any(w.ndim != 3 for w in (w1, w2, w3)):
        raise ValueError("Packed weights must have shape [E,K/2,N]")
    if bf <= 0 or bf % 256:
        raise ValueError("bf does not align with the selected FP4 route")
    e, h2, i = w1.shape
    if i % 256 or w3.shape != w1.shape or w2.shape != (e, i // 2, 2 * h2):
        raise ValueError("Invalid canonical FP4 weight shapes")
    if any(w.dtype != jnp.uint8 for w in (w1, w2, w3)):
        raise TypeError("Packed weights must be uint8")
    hidden = h2 * 2
    gs = (e, hidden // 32, i)
    ds = (e, i // 32, hidden)
    dtype = jnp.uint8
    if hidden % 512:
        raise ValueError("Hidden dimension must align with the X packing and dot K")
    if any(s.shape != shape or s.dtype != dtype for s, shape in ((s1, gs), (s3, gs), (s2, ds))):
        raise ValueError("Invalid canonical scale shape or dtype")
    padded_i = (i + bf - 1) // bf * bf
    pad_i = padded_i - i
    if pad_i:
        w1 = jnp.pad(w1, ((0, 0), (0, 0), (0, pad_i)))
        w3 = jnp.pad(w3, ((0, 0), (0, 0), (0, pad_i)))
        w2 = jnp.pad(w2, ((0, 0), (0, pad_i // 2), (0, 0)))
        pads = [(0, 0)] * s1.ndim
        pads[-1] = (0, pad_i)
        s1, s3 = jnp.pad(s1, pads), jnp.pad(s3, pads)
        pads = [(0, 0)] * s2.ndim
        pads[1] = (0, padded_i // group_size - s2.shape[1])
        s2 = jnp.pad(s2, pads)
    return (w1, w2, w3), {
        "w1_scale": s1,
        "w2_scale": s2,
        "w3_scale": s3,
        "logical_intermediate_size": i,
        "mxfp4": True,
        "fp4_group_size": group_size,
    }
