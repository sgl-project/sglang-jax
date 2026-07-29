"""Startup dispatch contract for the opt-in TPU-Inference v3 prefill path."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from sgl_jax.srt.kernels.gdn import decode_gated_delta_rule_ref
from sgl_jax.srt.kernels.gdn.tpu_inference_adapter import (
    GDNPrefillCapabilityError,
    tpu_inference_v3_prefill,
)
from sgl_jax.srt.layers.attention.linear.gdn_backend import GDNAttnBackend


def _mesh():
    return jax.sharding.Mesh(jax.devices()[:1], ("tensor",))


def _backend(monkeypatch, *, impl=None, dtype=None, **overrides):
    if impl is None:
        monkeypatch.delenv("SGLANG_JAX_GDN_PREFILL_IMPL", raising=False)
    else:
        monkeypatch.setenv("SGLANG_JAX_GDN_PREFILL_IMPL", impl)
    config = dict(
        num_k_heads=2,
        num_v_heads=4,
        head_k_dim=64,
        head_v_dim=64,
        conv_kernel_size=4,
        mesh=_mesh(),
        dtype=dtype,
    )
    config.update(overrides)
    return GDNAttnBackend(**config)


def test_unset_selector_uses_frozen_reference_prefill(monkeypatch, caplog):
    caplog.set_level("INFO", logger="sgl_jax.srt.layers.attention.linear.gdn_backend")
    backend = _backend(monkeypatch)

    assert backend.requested_prefill_impl == "reference"
    assert backend.effective_prefill_impl == "reference"
    assert backend.fallback_reason is None
    assert backend._prefill_callable == backend._forward_extend_reference
    assert backend._decode_callable is decode_gated_delta_rule_ref
    assert (
        sum(
            "requested=reference effective=reference fallback_reason=None" in record.message
            for record in caplog.records
        )
        == 1
    )


def test_explicit_reference_uses_existing_prefill(monkeypatch):
    backend = _backend(monkeypatch, impl="reference")

    assert backend.effective_prefill_impl == "reference"
    assert backend._prefill_callable == backend._forward_extend_reference
    assert backend._decode_callable is decode_gated_delta_rule_ref


def test_tpu_inference_v3_uses_distinct_frozen_adapter(monkeypatch):
    monkeypatch.setenv("PALLAS_INTERPRET", "1")
    backend = _backend(monkeypatch, impl="tpu_inference_v3", dtype=jnp.bfloat16)
    monkeypatch.setenv("SGLANG_JAX_GDN_PREFILL_IMPL", "reference")

    assert backend.requested_prefill_impl == "tpu_inference_v3"
    assert backend.effective_prefill_impl == "tpu_inference_v3"
    assert backend.fallback_reason is None
    assert backend._prefill_callable.func is tpu_inference_v3_prefill
    assert backend._decode_callable is decode_gated_delta_rule_ref


def test_invalid_selector_fails_during_initialization(monkeypatch):
    with pytest.raises(ValueError, match="SGLANG_JAX_GDN_PREFILL_IMPL"):
        _backend(monkeypatch, impl="chunkwise")


@pytest.mark.parametrize(
    ("dtype", "kwargs", "match"),
    [
        (None, {}, "dtype"),
        (jnp.float32, {}, "BF16"),
        (jnp.bfloat16, {"conv_kernel_size": 1}, "conv_kernel_size"),
    ],
)
def test_tpu_inference_v3_rejects_unsupported_startup_capability(monkeypatch, dtype, kwargs, match):
    monkeypatch.setenv("PALLAS_INTERPRET", "1")
    with pytest.raises(GDNPrefillCapabilityError, match=match):
        _backend(monkeypatch, impl="tpu_inference_v3", dtype=dtype, **kwargs)


def test_tpu_inference_v3_requires_tpu_mesh_without_interpret(monkeypatch):
    monkeypatch.delenv("PALLAS_INTERPRET", raising=False)

    with pytest.raises(GDNPrefillCapabilityError, match="TPU"):
        _backend(monkeypatch, impl="tpu_inference_v3", dtype=jnp.bfloat16)


def test_production_constructor_passes_model_dtype_to_backend():
    source = (Path(__file__).parents[1] / "srt" / "layers" / "attention" / "hybrid_linear_attn_backend.py").read_text()

    assert "dtype=runner.model_config.dtype" in source
