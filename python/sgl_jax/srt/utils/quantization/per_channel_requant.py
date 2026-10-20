"""Post-load requantization of block-quant fp8 linear weights to per-output-channel fp8.

``quantization.linear.per_channel_requant: a16|a8`` in the quantization YAML (null = unchanged;
``per_channel_requant_skip`` is a regex on the module path for layers that keep block scales).
For every ``QuantizedLinear`` whose scale is the kernel-ready block layout ``[in_blocks, 1, n_out]``:
dequantize the 128x128-block fp8 weight, requantize each output row with one e4m3fn scale,
and switch the layer to the per-channel path of ``xla_quantized_matmul_local`` (plain
fp8 x fp8 ``dot_general`` on the MXU when ``a8``; bf16 x bf16 with the fp8 weight converted
in the dot's operand read when ``a16``). Motivation (measured on v7x): the blockwise
Pallas kernel pops and rescales its accumulator once per 128-wide K block and runs at
1.8-2.4x the bf16 dot floor and 2.5-3.9x the e4m3fn dot floor at M=4096; per-channel
scales need one pop and one scale pass.
"""

from __future__ import annotations

import functools
import logging
import re

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

logger = logging.getLogger(__name__)

_FP8_MAX = 448.0


def per_channel_mode(quantization_config) -> str | None:
    """``a16`` / ``a8`` from the quantization config, or None (no config / not requested)."""
    if quantization_config is None:
        return None
    v = getattr(quantization_config, "linear_per_channel_requant", None)
    if v in (None, ""):
        return None
    if v not in ("a16", "a8"):
        raise ValueError(f"quantization.linear.per_channel_requant must be a16 or a8, got {v!r}")
    return v


def _requant(w_q: jax.Array, ws3: jax.Array, block_k: int) -> tuple[jax.Array, jax.Array]:
    # ws3: [in_blocks, 1, n_out] kernel-ready block scale; w_q: [n_out, n_in] fp8
    s = jnp.repeat(ws3[:, 0, :].T, block_k, axis=1)  # [n_out, n_in]
    w = w_q.astype(jnp.float32) * s
    amax = jnp.max(jnp.abs(w), axis=1, keepdims=True)
    sc = jnp.where(amax == 0, 1.0, amax / _FP8_MAX)
    return (w / sc).astype(w_q.dtype), sc[:, 0].astype(jnp.float32)


def skip_matches(skip_re: str | None, path_str: str, member_paths=()) -> bool:
    """``skip_re`` matches the module path, or -- for a module fused out of several linears
    (``fused_member_paths``) -- the path any of those members had before fusion."""
    if not skip_re:
        return False
    if re.search(skip_re, path_str):
        return True
    parent = path_str.rsplit("/", 1)[0] if "/" in path_str else ""
    return any(re.search(skip_re, f"{parent}/{m}" if parent else m) for m in member_paths)


def requantize_linears_per_channel(model: nnx.Module, mode: str, skip: str | None = None) -> int:
    """Rewrite every block-quant QuantizedLinear in ``model`` in place; returns the count.
    ``skip``: regex on the module path; matching layers keep their block scales."""
    from sgl_jax.srt.layers.linear import QuantizedLinear

    act_dtype = jnp.float8_e4m3fn if mode == "a8" else None
    skip_re = (skip or "").strip() or None
    n = 0
    for path, m in nnx.iter_graph(model):
        if not isinstance(m, QuantizedLinear):
            continue
        path_str = "/".join(str(k) for k in path)
        if skip_matches(skip_re, path_str, getattr(m, "fused_member_paths", ())):
            logger.info("per-channel requant: skipping %s (matches %r)", path_str, skip_re)
            continue
        ws = m.weight_scale.value
        if getattr(ws, "ndim", 0) != 3:
            continue
        w_q = m.weight_q.value
        n_out, n_in = w_q.shape
        in_blocks = ws.shape[0]
        if n_in % in_blocks != 0 or ws.shape[2] != n_out:
            logger.warning(
                "per-channel requant: skipping %s (scale %s vs weight %s)",
                path,
                ws.shape,
                w_q.shape,
            )
            continue
        block_k = n_in // in_blocks
        out_axis = m.kernel_axes[1] if m.kernel_axes is not None else None
        scale_sharding = NamedSharding(m.mesh, P(out_axis))
        # Explicit-axis meshes (production) reject repeat/reshape on sharded axes without
        # an out_sharding; run the math in Auto mode and pin only the outputs.
        w_sharding = NamedSharding(m.mesh, getattr(w_q.sharding, "spec", P()))
        fn = jax.jit(
            jax.sharding.auto_axes(
                functools.partial(_requant, block_k=block_k),
                axes=m.mesh.axis_names,
                out_sharding=(w_sharding, scale_sharding),
            )
        )
        new_w, new_s = fn(w_q, ws)
        m.weight_q.value = new_w
        m.weight_scale.value = new_s
        m.weight_block_size = None
        m.activation_dtype = act_dtype
        n += 1
    logger.info("per-channel fp8 requant (%s): %d QuantizedLinear layers rewritten", mode, n)
    return n
