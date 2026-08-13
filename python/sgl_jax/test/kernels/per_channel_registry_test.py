# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import pytest

import sgl_jax.srt.kernels.quantized_matmul.kernel as dispatch
from sgl_jax.srt.kernels.quantized_matmul import per_channel_utils
from sgl_jax.srt.kernels.quantized_matmul.per_channel_utils import (
    PerChannelTunedEntry,
    PerChannelTunedKey,
    PerChannelTunedValue,
)
from sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels import kernel as pallas_kernel


def _entry():
    return PerChannelTunedEntry(tuned_value=PerChannelTunedValue(8, 128, 128))


@pytest.mark.parametrize(
    ("m", "n", "k", "expected_tile"),
    [
        (2, 2048, 6144, (2, 2048, 3072, 1)),
        (2, 16384, 2048, (8, 4096, 2048, 1)),
        (2, 576, 6144, (2, 576, 3072, 1)),
        (2, 6144, 16384, (2, 2048, 8192, 1)),
        (2, 12288, 6144, (2, 4096, 3072, 1)),
        (2, 6144, 12288, (2, 2048, 6144, 1)),
        (2, 4096, 2048, (2, 4096, 1024, 1)),
        (2, 128, 6144, (2, 128, 3072, 1)),
        (8, 2048, 6144, (8, 512, 6144, 1)),
        (8, 16384, 2048, (8, 2048, 2048, 1)),
        (8, 576, 6144, (8, 256, 6144, 1)),
        (8, 6144, 16384, (8, 512, 16384, 1)),
        (8, 12288, 6144, (8, 4096, 3072, 1)),
        (8, 6144, 12288, (8, 512, 12288, 1)),
        (8, 4096, 2048, (8, 4096, 2048, 1)),
        (8, 128, 6144, (8, 128, 6144, 1)),
        (1024, 2048, 6144, (1024, 1024, 3072, 1)),
        (1024, 16384, 2048, (1024, 2048, 2048, 1)),
        (1024, 576, 6144, (1024, 576, 1536, 1)),
        (1024, 6144, 16384, (1024, 1536, 4096, 1)),
        (1024, 12288, 6144, (1024, 2048, 3072, 1)),
        (1024, 6144, 12288, (1024, 2048, 3072, 1)),
        (1024, 4096, 2048, (1024, 2048, 2048, 1)),
        (1024, 128, 6144, (1024, 128, 1536, 1)),
        (2048, 2048, 6144, (1024, 2048, 3072, 1)),
        (2048, 16384, 2048, (1024, 4096, 2048, 1)),
        (2048, 576, 6144, (1024, 576, 2048, 1)),
        (2048, 6144, 16384, (1024, 1536, 4096, 1)),
        (2048, 12288, 6144, (1024, 2048, 3072, 1)),
        (2048, 6144, 12288, (1024, 2048, 3072, 1)),
        (2048, 4096, 2048, (1024, 2048, 2048, 1)),
        (2048, 128, 6144, (512, 128, 6144, 1)),
    ],
)
def test_glm52_tpu7_w8a16_registry_entries(
    m,
    n,
    k,
    expected_tile,
):
    entry = per_channel_utils.get_exact_per_channel_tuned_entry(
        n_batch=m,
        n_out=n,
        n_in=k,
        x_dtype=jnp.bfloat16,
        x_q_dtype=jnp.bfloat16,
        w_q_dtype=jnp.float8_e4m3fn,
        tpu_version=7,
    )

    assert entry is not None
    assert tuple(entry.tuned_value) == expected_tile


def test_glm52_registry_does_not_contain_untuned_w8a8():
    assert (
        per_channel_utils.get_exact_per_channel_tuned_entry(
            n_batch=2,
            n_out=2048,
            n_in=6144,
            x_dtype=jnp.bfloat16,
            x_q_dtype=jnp.float8_e4m3fn,
            w_q_dtype=jnp.float8_e4m3fn,
            tpu_version=7,
        )
        is None
    )


def test_exact_registry_has_no_shape_or_dtype_fallback(monkeypatch):
    key = PerChannelTunedKey(
        7,
        2,
        128,
        256,
        "bfloat16",
        "bfloat16",
        "float8_e4m3fn",
    )
    entry = _entry()
    monkeypatch.setitem(per_channel_utils.PER_CHANNEL_TUNED_ENTRIES, key, entry)

    kwargs = dict(
        n_batch=2,
        n_out=128,
        n_in=256,
        x_dtype=jnp.bfloat16,
        x_q_dtype=jnp.bfloat16,
        w_q_dtype=jnp.float8_e4m3fn,
        tpu_version=7,
    )
    assert per_channel_utils.get_exact_per_channel_tuned_entry(**kwargs) is entry
    assert per_channel_utils.get_exact_per_channel_tuned_entry(**(kwargs | {"n_batch": 3})) is None
    assert (
        per_channel_utils.get_exact_per_channel_tuned_entry(
            **(kwargs | {"x_q_dtype": jnp.float8_e4m3fn})
        )
        is None
    )


def test_low_level_pallas_kernel_rejects_implicit_default_tile():
    x = jnp.ones((2, 128), dtype=jnp.bfloat16)
    w_q = jnp.ones((128, 128), dtype=jnp.float8_e4m3fn)
    w_scale = jnp.ones((128,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="requires an explicit tuned_value"):
        pallas_kernel.quantized_matmul_kernel.__wrapped__(
            x,
            w_q,
            w_scale,
            x_q_dtype=jnp.bfloat16,
        )


@pytest.mark.parametrize("compute_dtype", [None, jnp.bfloat16])
def test_dot_backend_never_loads_registry_or_pallas(monkeypatch, compute_dtype):
    monkeypatch.setattr(
        dispatch,
        "get_exact_per_channel_tuned_entry",
        lambda **_: (_ for _ in ()).throw(AssertionError("registry must not be queried")),
    )
    monkeypatch.setattr(
        dispatch,
        "get_per_channel_kernel",
        lambda: (_ for _ in ()).throw(AssertionError("Pallas must not be loaded")),
    )

    x = jnp.arange(8, dtype=jnp.bfloat16).reshape(2, 4)
    w_q = jnp.ones((3, 4), dtype=jnp.float8_e4m3fn)
    w_scale = jnp.arange(1, 4, dtype=jnp.float32)
    out = dispatch.xla_quantized_matmul_local(
        x,
        w_q,
        w_scale,
        quantize_activation=False,
        compute_dtype=compute_dtype,
        per_channel_matmul_backend="dot",
    )

    expected_compute_dtype = jnp.float32 if compute_dtype is None else compute_dtype
    expected = jnp.dot(x, w_q.T, preferred_element_type=expected_compute_dtype)
    expected = (expected.astype(expected_compute_dtype) * w_scale).astype(x.dtype)
    assert bool(jnp.array_equal(out, expected))


def test_pallas_backend_rejects_registry_miss(monkeypatch):
    monkeypatch.setattr(dispatch, "get_exact_per_channel_tuned_entry", lambda **_: None)
    x = jnp.ones((2, 4), dtype=jnp.bfloat16)
    w_q = jnp.ones((3, 4), dtype=jnp.float8_e4m3fn)
    w_scale = jnp.ones((3,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="requires an exact tuned entry"):
        dispatch.xla_quantized_matmul_local(
            x,
            w_q,
            w_scale,
            quantize_activation=False,
            per_channel_matmul_backend="pallas",
        )


def test_pallas_backend_rejects_non_fp32_accumulation(monkeypatch):
    monkeypatch.setattr(
        dispatch,
        "get_exact_per_channel_tuned_entry",
        lambda **_: (_ for _ in ()).throw(AssertionError("registry must not be queried")),
    )
    x = jnp.ones((2, 4), dtype=jnp.bfloat16)
    w_q = jnp.ones((3, 4), dtype=jnp.float8_e4m3fn)
    w_scale = jnp.ones((3,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="requires float32 accumulation"):
        dispatch.xla_quantized_matmul_local(
            x,
            w_q,
            w_scale,
            quantize_activation=False,
            compute_dtype=jnp.bfloat16,
            per_channel_matmul_backend="pallas",
        )


@pytest.mark.parametrize(
    ("quantize_activation", "activation_quant_dtype", "expected_x_q_dtype"),
    [
        pytest.param(False, None, jnp.bfloat16, id="w8a16"),
        pytest.param(True, jnp.float8_e4m3fn, jnp.float8_e4m3fn, id="w8a8"),
    ],
)
def test_production_dispatches_exact_validated_entry_to_common_kernel(
    monkeypatch,
    quantize_activation,
    activation_quant_dtype,
    expected_x_q_dtype,
):
    entry = _entry()
    calls = {}

    def fake_lookup(**kwargs):
        calls["lookup"] = kwargs
        return entry

    def fake_kernel(**kwargs):
        calls["kernel"] = kwargs
        return jnp.full(
            (kwargs["x"].shape[0], kwargs["w_q"].shape[0]),
            7,
            dtype=kwargs["x"].dtype,
        )

    monkeypatch.setattr(dispatch, "get_exact_per_channel_tuned_entry", fake_lookup)
    monkeypatch.setattr(dispatch, "get_per_channel_kernel", lambda: fake_kernel)

    x = jnp.ones((2, 4), dtype=jnp.bfloat16)
    w_q = jnp.ones((3, 4), dtype=jnp.float8_e4m3fn)
    w_scale = jnp.ones((3,), dtype=jnp.float32)
    out = dispatch.xla_quantized_matmul_local(
        x,
        w_q,
        w_scale,
        quantize_activation=quantize_activation,
        activation_quant_dtype=activation_quant_dtype,
        per_channel_matmul_backend="pallas",
    )

    assert calls["lookup"]["x_q_dtype"] == expected_x_q_dtype
    assert calls["kernel"]["x_q_dtype"] == expected_x_q_dtype
    assert calls["kernel"]["tuned_value"] is entry.tuned_value
    assert bool(jnp.all(out == 7))
