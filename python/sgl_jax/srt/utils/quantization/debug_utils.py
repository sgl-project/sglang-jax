import logging
import os

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def _maybe_log_quant_stats(tag: str, x: jax.Array, x_q: jax.Array, scale: jax.Array) -> None:
    """Log a few scalar stats to diagnose quantization collapse/outliers.

    Enable with: SGLANG_MOE_QUANT_STATS=1
    """
    if os.getenv("SGLANG_MOE_QUANT_STATS", "0") != "1":
        return

    x_f32 = x.astype(jnp.float32)
    x_has_nan = jnp.any(jnp.isnan(x_f32))
    x_has_inf = jnp.any(jnp.isinf(x_f32))
    absmax = jnp.max(jnp.abs(x_f32))
    rms = jnp.sqrt(jnp.mean(jnp.square(x_f32)))
    mean_abs = jnp.mean(jnp.abs(x_f32))

    q_f32 = x_q.astype(jnp.float32)
    q_has_nan = jnp.any(jnp.isnan(q_f32))
    q_absmax = jnp.nanmax(jnp.abs(q_f32))
    q_zero_frac = jnp.nanmean(q_f32 == 0)

    scale_f32 = scale.astype(jnp.float32)
    scale_has_nan = jnp.any(jnp.isnan(scale_f32))
    scale_zero_frac = jnp.mean(scale_f32 == 0)
    scale_zero_dim0_max = None
    scale_zero_dim0_over_10pct = None
    scale_zero_dim1_max = None
    scale_zero_dim1_over_10pct = None
    if scale_f32.ndim == 2:
        scale_zero_mask = scale_f32 == 0
        # For EPMoE: scale is (num_experts, out_features). This helps distinguish
        # "a few dead channels" from "whole experts are zero/padded/unloaded".
        scale_zero_dim0 = jnp.mean(scale_zero_mask, axis=1)  # per expert
        scale_zero_dim1 = jnp.mean(scale_zero_mask, axis=0)  # per output channel
        scale_zero_dim0_max = jnp.max(scale_zero_dim0)
        scale_zero_dim0_over_10pct = jnp.mean(scale_zero_dim0 > 0.10)
        scale_zero_dim1_max = jnp.max(scale_zero_dim1)
        scale_zero_dim1_over_10pct = jnp.mean(scale_zero_dim1 > 0.10)
    scale_max = jnp.max(scale_f32)
    scale_mean = jnp.mean(scale_f32)
    scale_min_nz = jnp.min(jnp.where(scale_f32 > 0, scale_f32, jnp.inf))

    # Per-channel diagnostics: if this is a simple per-axis quantization where
    # `scale.shape == x.shape with exactly one axis removed`, compute how many
    # channels are collapsing to many zeros (a common FP8 failure mode when
    # outliers dominate absmax).
    ch_zero_max = None
    ch_zero_over_1pct = None
    if x.ndim == scale.ndim + 1:
        reduced_axis = None
        for ax in range(x.ndim):
            if x.shape[:ax] + x.shape[ax + 1 :] == scale.shape:
                reduced_axis = ax
                break
        if reduced_axis is not None:
            ch_zero = jnp.mean(q_f32 == 0, axis=reduced_axis)
            ch_zero_max = jnp.max(ch_zero)
            ch_zero_over_1pct = jnp.mean(ch_zero > 0.01)

    absmax_v = float(absmax)
    rms_v = float(rms)
    mean_abs_v = float(mean_abs)
    x_has_nan_v = bool(x_has_nan)
    x_has_inf_v = bool(x_has_inf)
    q_absmax_v = float(q_absmax)
    q_zero_frac_v = float(q_zero_frac)
    q_has_nan_v = bool(q_has_nan)
    scale_max_v = float(scale_max)
    scale_mean_v = float(scale_mean)
    scale_min_nz_v = float(scale_min_nz)
    scale_has_nan_v = bool(scale_has_nan)
    scale_zero_frac_v = float(scale_zero_frac)
    scale_zero_dim0_max_v = None if scale_zero_dim0_max is None else float(scale_zero_dim0_max)
    scale_zero_dim0_over_10pct_v = (
        None if scale_zero_dim0_over_10pct is None else float(scale_zero_dim0_over_10pct)
    )
    scale_zero_dim1_max_v = None if scale_zero_dim1_max is None else float(scale_zero_dim1_max)
    scale_zero_dim1_over_10pct_v = (
        None if scale_zero_dim1_over_10pct is None else float(scale_zero_dim1_over_10pct)
    )
    if scale_min_nz_v == float("inf"):
        scale_min_nz_v = 0.0

    msg = (
        "MoE quant stats [%s]: absmax=%g rms=%g absmax/rms=%g mean_abs=%g x_nan=%s x_inf=%s | "
        "q_absmax=%g q_zero=%g q_nan=%s | "
        "scale[min_nz,mean,max]=[%g,%g,%g] scale_nan=%s scale_zero=%g"
    )
    args = [
        tag,
        absmax_v,
        rms_v,
        absmax_v / (rms_v + 1e-12),
        mean_abs_v,
        x_has_nan_v,
        x_has_inf_v,
        q_absmax_v,
        q_zero_frac_v,
        q_has_nan_v,
        scale_min_nz_v,
        scale_mean_v,
        scale_max_v,
        scale_has_nan_v,
        scale_zero_frac_v,
    ]
    if (
        scale_zero_dim0_max_v is not None
        and scale_zero_dim0_over_10pct_v is not None
        and scale_zero_dim1_max_v is not None
        and scale_zero_dim1_over_10pct_v is not None
    ):
        msg += " | scale0_dim0[max,>10%%]=[%g,%g] scale0_dim1[max,>10%%]=[%g,%g]"
        args.extend(
            [
                scale_zero_dim0_max_v,
                scale_zero_dim0_over_10pct_v,
                scale_zero_dim1_max_v,
                scale_zero_dim1_over_10pct_v,
            ]
        )
    if ch_zero_max is not None and ch_zero_over_1pct is not None:
        # NOTE: logger uses %-formatting; escape literal '%' as '%%'.
        msg += " | ch_zero[max,>1%%]=[%g,%g]"
        args.extend([float(ch_zero_max), float(ch_zero_over_1pct)])

    logger.info(msg, *args)
