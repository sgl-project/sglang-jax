# SPDX-License-Identifier: Apache-2.0

import functools

import jax
import jax.numpy as jnp
import pytest

from sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels import kernel
from sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels.tuned_block_sizes import (
    TunedValue,
)


def _run_wrapper_without_tpu(
    monkeypatch,
    *,
    n_batch: int,
    x_q_dtype: jnp.dtype | None,
):
    """Trace wrapper preparation while replacing the TPU-only Pallas launch."""
    captured = {}

    real_jnp_max = kernel.jnp.max

    def capture_jnp_max(*args, **kwargs):
        captured["jnp_max_calls"] = captured.get("jnp_max_calls", 0) + 1
        return real_jnp_max(*args, **kwargs)

    real_get_vmem_limit = kernel.util.get_vmem_limit

    def capture_get_vmem_limit(**kwargs):
        captured["vmem_kwargs"] = kwargs
        return real_get_vmem_limit(**kwargs)

    def fake_pallas_call(kernel_fn, *, grid_spec, out_shape, compiler_params):
        captured["kernel_fn"] = kernel_fn
        captured["grid_spec"] = grid_spec
        captured["out_shape"] = out_shape
        captured["compiler_params"] = compiler_params

        def fake_kernel_call(*args):
            captured["args"] = args
            return jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

        return fake_kernel_call

    monkeypatch.setattr(kernel.pl, "pallas_call", fake_pallas_call)
    monkeypatch.setattr(kernel, "get_device_vmem_limit", lambda: 96 * 1024 * 1024)
    monkeypatch.setattr(kernel.jnp, "max", capture_jnp_max)
    monkeypatch.setattr(kernel.util, "get_vmem_limit", capture_get_vmem_limit)

    # Keep K in one tile so W8A8 can exercise activation-cache selection. N is
    # deliberately not tile-aligned to cover rank-1 scale padding.
    n_in, n_out = 128, 129
    x = jnp.ones((n_batch, n_in), dtype=jnp.bfloat16)
    w_q = jnp.ones((n_out, n_in), dtype=jnp.float8_e4m3fn)
    w_scale = jnp.ones((n_out,), dtype=jnp.float32)
    tuned_value = TunedValue(
        # BM=8 is only a test tile used to exercise decode padding. It is not
        # intended to represent a tuned M=2 configuration.
        batch_block_size=8 if n_batch == 2 else 1024,
        out_block_size=128,
        in_block_size=128,
    )

    # Call the Python wrapper rather than compiling the TPU Pallas primitive.
    out = kernel.quantized_matmul_kernel.__wrapped__(
        x,
        w_q,
        w_scale,
        x_q_dtype=x_q_dtype,
        tuned_value=tuned_value,
    )
    return out, captured


@pytest.mark.parametrize("n_batch", [2, 1024], ids=["decode-m2", "prefill-m1024"])
@pytest.mark.parametrize(
    (
        "x_q_dtype",
        "expected_x_q_dtype",
        "expected_save_x_q",
        "expected_array_operand_count",
        "expected_active_scratch_count",
        "expected_max_calls",
    ),
    [
        pytest.param(
            None,
            jnp.bfloat16,
            False,
            3,
            0,
            0,
            id="w8a16-default",
        ),
        pytest.param(
            jnp.bfloat16,
            jnp.bfloat16,
            False,
            3,
            0,
            0,
            id="w8a16-explicit",
        ),
        pytest.param(
            jnp.float8_e4m3fn,
            jnp.float8_e4m3fn,
            True,
            4,
            2,
            1,
            id="w8a8",
        ),
    ],
)
def test_per_channel_wrapper_uses_dtype_equality_for_activation_quantization(
    monkeypatch,
    n_batch,
    x_q_dtype,
    expected_x_q_dtype,
    expected_save_x_q,
    expected_array_operand_count,
    expected_active_scratch_count,
    expected_max_calls,
):
    out, captured = _run_wrapper_without_tpu(
        monkeypatch,
        n_batch=n_batch,
        x_q_dtype=x_q_dtype,
    )

    kernel_fn = captured["kernel_fn"]
    assert isinstance(kernel_fn, functools.partial)
    assert kernel_fn.func is kernel.matmul_kernel
    assert kernel_fn.keywords["x_q_dtype"] == expected_x_q_dtype
    assert kernel_fn.keywords["save_x_q"] is expected_save_x_q
    assert len(captured["args"]) == 4
    assert len(captured["grid_spec"].in_specs) == 4
    assert len(jax.tree_util.tree_leaves(captured["args"])) == expected_array_operand_count
    assert len(captured["grid_spec"].scratch_shapes) == 3
    active_scratches = sum(shape is not None for shape in captured["grid_spec"].scratch_shapes)
    assert active_scratches == expected_active_scratch_count
    assert captured.get("jnp_max_calls", 0) == expected_max_calls
    assert captured["vmem_kwargs"]["has_x_abs_max"] is bool(expected_max_calls)
    assert out.shape == (n_batch, 129)
    assert out.dtype == jnp.bfloat16


@pytest.mark.parametrize("n_batch", [2, 1024], ids=["decode-m2", "prefill-m1024"])
def test_per_channel_wrapper_pads_rank_one_scale_and_crops_output(monkeypatch, n_batch):
    out, captured = _run_wrapper_without_tpu(
        monkeypatch,
        n_batch=n_batch,
        x_q_dtype=jnp.bfloat16,
    )

    x, w_q, w_scale, x_abs_max = captured["args"]
    padded_n_batch = 8 if n_batch == 2 else 1024
    assert x.shape == (padded_n_batch, 128)
    assert w_q.shape == (256, 128)
    assert w_scale.shape == (1, 256)
    assert x_abs_max is None
    assert out.shape == (n_batch, 129)


@pytest.mark.parametrize("n_batch", [2, 1024], ids=["decode-m2", "prefill-m1024"])
def test_per_channel_w8a8_pads_absmax(monkeypatch, n_batch):
    out, captured = _run_wrapper_without_tpu(
        monkeypatch,
        n_batch=n_batch,
        x_q_dtype=jnp.float8_e4m3fn,
    )

    x, w_q, w_scale, x_abs_max = captured["args"]
    padded_n_batch = 8 if n_batch == 2 else 1024
    assert x.shape == (padded_n_batch, 128)
    assert w_q.shape == (256, 128)
    assert w_scale.shape == (1, 256)
    assert x_abs_max.shape == (1, padded_n_batch)
    assert out.shape == (n_batch, 129)


@pytest.mark.parametrize("n_batch_blocks", [1, 2])
def test_vmem_accounting_only_includes_absmax_for_matching_abi(n_batch_blocks):
    kwargs = dict(
        n_batch=n_batch_blocks,
        n_out=2,
        n_in=1,
        batch_block_size=8,
        out_block_size=128,
        in_block_size=128,
        x_dtype=jnp.bfloat16,
        x_q_dtype=jnp.bfloat16,
        w_q_dtype=jnp.float8_e4m3fn,
        scale_dtype=jnp.float32,
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        save_acc=False,
        save_x_q=False,
        upper_limit_bytes=1 << 60,
    )

    without_absmax = kernel.util.get_vmem_limit(**kwargs, has_x_abs_max=False)
    with_absmax = kernel.util.get_vmem_limit(**kwargs, has_x_abs_max=True)

    expected_absmax_bytes = (2 + int(n_batch_blocks > 1)) * 8 * 4
    assert with_absmax - without_absmax == expected_absmax_bytes


@pytest.mark.parametrize(
    ("x_q_dtype", "n_in", "n_out", "exact_match"),
    [
        pytest.param(jnp.bfloat16, 128, 128, True, id="w8a16-no-scratch"),
        pytest.param(jnp.bfloat16, 256, 128, True, id="w8a16-acc-scratch"),
        pytest.param(jnp.float8_e4m3fn, 128, 256, False, id="w8a8-xq-cache-scratch"),
        pytest.param(jnp.float8_e4m3fn, 256, 128, False, id="w8a8-acc-scratch"),
    ],
)
def test_per_channel_fixed_abi_executes_in_pallas_interpret_mode(
    monkeypatch,
    x_q_dtype,
    n_in,
    n_out,
    exact_match,
):
    real_pallas_call = kernel.pl.pallas_call

    def interpret_pallas_call(kernel_fn, *, grid_spec, out_shape, compiler_params):
        return real_pallas_call(
            kernel_fn,
            grid_spec=grid_spec,
            out_shape=out_shape,
            compiler_params=compiler_params,
            interpret=True,
        )

    monkeypatch.setattr(kernel.pl, "pallas_call", interpret_pallas_call)
    monkeypatch.setattr(kernel, "get_device_vmem_limit", lambda: 96 * 1024 * 1024)

    x = jnp.arange(2 * n_in, dtype=jnp.bfloat16).reshape(2, n_in) / 128
    weight_indices = jnp.arange(n_out * n_in, dtype=jnp.int32).reshape(n_out, n_in)
    w_q = (((weight_indices % 7) - 3).astype(jnp.float32) / 4).astype(jnp.float8_e4m3fn)
    w_scale = 0.5 + (jnp.arange(n_out, dtype=jnp.float32) % 7) / 8
    tuned_value = TunedValue(
        batch_block_size=8,
        out_block_size=128,
        in_block_size=128,
    )

    out = kernel.quantized_matmul_kernel.__wrapped__(
        x,
        w_q,
        w_scale,
        x_q_dtype=x_q_dtype,
        tuned_value=tuned_value,
    )
    reference = kernel.util.xla_quantized_matmul(
        x,
        w_q,
        w_scale,
        quantize_activation=x_q_dtype != x.dtype,
    )

    assert out.shape == reference.shape
    assert out.dtype == reference.dtype
    if exact_match:
        assert bool(jnp.array_equal(out, reference))
    else:
        assert bool(jnp.allclose(out, reference, rtol=0.02, atol=1.0))
