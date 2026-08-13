# Adapted from https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/quantized_matmul/kernel.py
# SPDX-License-Identifier: Apache-2.0
"""Quantized matmul kernel."""

import functools
from typing import Protocol

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import util
from .tuned_block_sizes import get_device_vmem_limit
from .util import get_kernel_name, next_multiple, unfold_args

quantize_tensor = util.quantize_tensor


class TunedValueLike(Protocol):
    """Structural tile contract shared by registry and benchmark values."""

    batch_block_size: int
    out_block_size: int
    in_block_size: int
    n_lane_multiplier: int


def matmul_kernel(
    x_ref: jax.Array,  # (batch_block_size, in_block_size)
    w_q_ref: jax.Array,  # (out_block_size, in_block_size)
    w_scale_ref: jax.Array,  # (1, out_block_size)
    x_abs_max_ref: jax.Array | None,  # (1, batch_block_size)
    out_ref: jax.Array,  # (batch_block_size, out_block_size)
    acc_scratch: jax.Array | None,  # (batch_block_size, out_block_size)
    x_q_scratch: jax.Array | None,  # (batch_block_size, in_block_size)
    x_scale_scratch: jax.Array | None,  # (batch_block_size, 1)
    *,
    x_q_dtype: jnp.dtype,
    save_acc: bool,
    save_x_q: bool,
):
    out_idx, in_idx = pl.program_id(1), pl.program_id(2)
    n_in = pl.num_programs(2)
    x_ref_dtype = x_ref.dtype

    quantize_activation = x_q_dtype != x_ref_dtype
    assert quantize_activation == (x_abs_max_ref is not None)

    # Initialize conditional logic.
    if save_x_q:
        assert quantize_activation
        assert x_q_scratch is not None
        assert x_scale_scratch is not None
        quant = out_idx == 0
    else:
        assert x_q_scratch is None
        assert x_scale_scratch is None
        quant = quantize_activation

    if save_acc:
        assert acc_scratch is not None
        is_first_step = in_idx == 0
        is_last_step = in_idx == (n_in - 1)
    else:
        assert acc_scratch is None
        is_first_step = True
        is_last_step = True

    acc_dtype = jnp.float32
    if quantize_activation and jnp.issubdtype(w_q_ref.dtype, jnp.integer):
        acc_dtype = jnp.int32

    # Start of actual computation logic.
    def matmul_body(quant: bool, is_first_step: bool, is_last_step: bool):
        if quantize_activation:
            assert x_abs_max_ref is not None
            if quant:
                x_q_tmp, x_scale_tmp = util.quantize_array(
                    x_ref[...],
                    x_abs_max_ref[...],
                    x_q_dtype,
                )

                if save_x_q:
                    assert x_q_scratch is not None
                    assert x_scale_scratch is not None
                    x_q_scratch[...] = x_q_tmp
                    x_scale_scratch[...] = x_scale_tmp

            else:
                assert save_x_q
                assert x_q_scratch is not None
                assert x_scale_scratch is not None
                x_q_tmp = x_q_scratch[...]
                if is_last_step:
                    x_scale_tmp = x_scale_scratch[...]

            acc = jax.lax.dot_general(
                x_q_tmp,
                w_q_ref[...],
                (((1,), (1,)), ((), ())),
                preferred_element_type=acc_dtype,
            )
        else:
            acc = jax.lax.dot_general(
                x_ref[...],
                w_q_ref[...],
                (((1,), (1,)), ((), ())),
                preferred_element_type=acc_dtype,
            )

        if not is_first_step:
            assert acc_scratch is not None
            acc += acc_scratch[...]

        if is_last_step:
            acc *= w_scale_ref[...]
            if quantize_activation:
                # TODO(kyuyeunk): Investigate caching broadcast.
                acc *= x_scale_tmp
            out_ref[...] = acc.astype(x_ref_dtype)
        else:
            assert save_acc
            assert acc_scratch is not None
            acc_scratch[...] = acc

    unfold_args((quant, is_first_step, is_last_step), (), matmul_body)


@jax.jit(
    static_argnames=[
        "x_q_dtype",
        "tuned_value",
    ]
)
def quantized_matmul_kernel(
    x: jax.Array,  # [bs, n_in]
    w_q: jax.Array,  # [n_out, n_in]
    w_scale: jax.Array,  # [n_out]
    w_zp: jax.Array | None = None,  # [n_out]
    block_size: int | None = None,
    x_q_dtype: jnp.dtype | None = None,
    *,
    tuned_value: TunedValueLike | None = None,
) -> jax.Array:
    """Quantized matmul kernel.

    Args:
      x: Input unquantized array.
      w_q: Weight quantized array. [n_output_features, n_input_features]
      w_scale: Weight quantization scale. [n_output_features]
      w_zp: Weight zero point for asymmetric quantization.
      block_size: Block size for subchannel quantization.
      x_q_dtype: Quantization type of the input. If None or if the value is the
        same as x.dtype, then no quantization is applied.
      tuned_value: Exact kernel tuned values for this workload. The caller must
        resolve this from the per-channel registry or pass an explicit
        benchmark candidate; there is no implicit fallback tile.

    Returns:
      Quantized matmul result.
    """

    if w_zp is not None:
        raise NotImplementedError("zero_point is not supported.")
    if block_size is not None:
        raise NotImplementedError("block_size is not supported.")

    if x_q_dtype is None:
        x_q_dtype = x.dtype
    quantize_activation = x_q_dtype != x.dtype

    x_abs_max = None
    if quantize_activation:
        # Pallas only sees one K block at a time, so full-K per-token absmax is
        # computed outside the kernel when activation quantization is enabled.
        x_abs_max = jnp.max(jnp.abs(x), axis=-1, keepdims=False)  # [bs]
        # Pallas requires the minormost dimension to be a multiple of the
        # sublane size 128. Use [1, bs] instead of [bs, 1].
        x_abs_max = jnp.expand_dims(x_abs_max, axis=0)  # [1, bs]
        assert x_abs_max.shape == (1, x.shape[0])

    orig_n_batch, orig_n_in = x.shape
    orig_n_out, _ = w_q.shape

    if tuned_value is None:
        raise ValueError(
            "Per-channel Pallas matmul requires an explicit tuned_value; "
            "unknown shapes must use the DOT fallback."
        )
    batch_block_size = tuned_value.batch_block_size
    out_block_size = tuned_value.out_block_size
    in_block_size = tuned_value.in_block_size

    # Pad the inputs to be multiple of block size.
    padded_n_batch = next_multiple(orig_n_batch, batch_block_size)
    if orig_n_batch < padded_n_batch:
        x = jnp.pad(x, ((0, padded_n_batch - orig_n_batch), (0, 0)))
        if x_abs_max is not None:
            x_abs_max = jnp.pad(
                x_abs_max,
                ((0, 0), (0, padded_n_batch - orig_n_batch)),
            )
    padded_n_out = next_multiple(orig_n_out, out_block_size)
    if orig_n_out < padded_n_out:
        w_q = jnp.pad(w_q, ((0, padded_n_out - orig_n_out), (0, 0)))
        w_scale = jnp.pad(w_scale, (0, padded_n_out - orig_n_out))
    padded_n_in = next_multiple(orig_n_in, in_block_size)
    if orig_n_in < padded_n_in:
        x = jnp.pad(x, ((0, 0), (0, padded_n_in - orig_n_in)))
        w_q = jnp.pad(w_q, ((0, 0), (0, padded_n_in - orig_n_in)))

    if w_scale.dtype != jnp.float32:
        w_scale = w_scale.astype(jnp.float32)
    w_scale = jnp.expand_dims(w_scale, axis=0)  # [1, n_output_features]

    n_batch = padded_n_batch // batch_block_size
    n_out = padded_n_out // out_block_size
    n_in = padded_n_in // in_block_size

    save_acc = n_in > 1
    # Remove redundant input quantization logic by caching quantized input. For
    # best performance, only enable this behavior when single input block is
    # used per batch.
    save_x_q = quantize_activation and n_in == 1 and n_out > 1

    acc_dtype = jnp.float32
    if quantize_activation and jnp.issubdtype(w_q.dtype, jnp.integer):
        acc_dtype = jnp.int32

    vmem_limit_bytes = util.get_vmem_limit(
        n_batch=n_batch,
        n_out=n_out,
        n_in=n_in,
        batch_block_size=batch_block_size,
        out_block_size=out_block_size,
        in_block_size=in_block_size,
        x_dtype=x.dtype,
        x_q_dtype=x_q_dtype,
        w_q_dtype=w_q.dtype,
        scale_dtype=jnp.float32,
        out_dtype=x.dtype,
        acc_dtype=acc_dtype,
        save_acc=save_acc,
        save_x_q=save_x_q,
        upper_limit_bytes=get_device_vmem_limit(),
        has_x_abs_max=quantize_activation,
    )

    x_abs_max_spec = (
        pl.BlockSpec((1, batch_block_size), lambda b, o, i: (0, b)) if quantize_activation else None
    )
    input_specs = [
        pl.BlockSpec((batch_block_size, in_block_size), lambda b, o, i: (b, i)),  # x
        pl.BlockSpec((out_block_size, in_block_size), lambda b, o, i: (o, i)),  # w_q
        pl.BlockSpec((1, out_block_size), lambda b, o, i: (0, o)),  # w_scale
        x_abs_max_spec,
    ]
    kernel_args = [x, w_q, w_scale, x_abs_max]
    assert (x_abs_max is not None) == quantize_activation

    acc_scratch_shape = (
        pltpu.VMEM((batch_block_size, out_block_size), acc_dtype) if save_acc else None
    )
    scratch_shapes = [
        acc_scratch_shape,
        pltpu.VMEM((batch_block_size, in_block_size), x_q_dtype) if save_x_q else None,
        pltpu.VMEM((batch_block_size, 1), jnp.float32) if save_x_q else None,
    ]

    kernel = pl.pallas_call(
        functools.partial(
            matmul_kernel,
            x_q_dtype=x_q_dtype,
            save_acc=save_acc,
            save_x_q=save_x_q,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=input_specs,
            out_specs=pl.BlockSpec((batch_block_size, out_block_size), lambda b, o, i: (b, o)),
            scratch_shapes=scratch_shapes,
            grid=(n_batch, n_out, n_in),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_n_batch, padded_n_out), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
    )

    util.validate_inputs(
        x=x,
        w_q=w_q,
        w_scale=w_scale,
        x_abs_max=x_abs_max,
        x_q_dtype=x_q_dtype,
        batch_block_size=batch_block_size,
        out_block_size=out_block_size,
        in_block_size=in_block_size,
    )

    # The named_scope is used for autotune.
    kernel_name = get_kernel_name(tuned_value)
    with jax.named_scope(kernel_name):
        out = kernel(*kernel_args)

    return out[:orig_n_batch, :orig_n_out]
