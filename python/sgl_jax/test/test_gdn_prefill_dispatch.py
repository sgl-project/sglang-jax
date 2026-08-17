"""Startup dispatch contract for the opt-in fused chunk-parallel prefill path."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from sgl_jax.srt.kernels.gdn import decode_gated_delta_rule_ref
from sgl_jax.srt.kernels.gdn.fused_chunk_parallel_adapter import (
    GDNPrefillCapabilityError,
    fused_chunk_parallel_prefill,
)
from sgl_jax.srt.layers.attention import hybrid_linear_attn_backend
from sgl_jax.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sgl_jax.srt.server_args import ServerArgs


def _mesh():
    return jax.sharding.Mesh(jax.devices()[:1], ("tensor",))


def _backend(monkeypatch, *, impl=None, dtype=None, **overrides):
    config = dict(
        num_k_heads=2,
        num_v_heads=4,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        mesh=_mesh(),
        dtype=dtype,
    )
    if impl is not None:
        config["prefill_impl"] = impl
    config.update(overrides)
    return GDNAttnBackend(**config)


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ([], "fused_chunk_parallel"),
        (["--gdn-prefill-impl", "chunked_jax"], "chunked_jax"),
    ],
)
def test_server_args_parses_gdn_prefill_impl(extra_args, expected):
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)

    namespace = parser.parse_args(["--model-path", "dummy", *extra_args])
    server_args = ServerArgs.from_cli_args(namespace)

    assert server_args.gdn_prefill_impl == expected


def test_default_selector_uses_frozen_fused_chunk_parallel_prefill(monkeypatch, caplog):
    caplog.set_level("INFO", logger="sgl_jax.srt.layers.attention.linear.gdn_backend")
    monkeypatch.setattr(
        "sgl_jax.srt.kernels.gdn.fused_chunk_parallel_adapter._mesh_devices",
        lambda _: (SimpleNamespace(platform="tpu"),),
    )
    backend = _backend(monkeypatch, dtype=jnp.bfloat16)

    assert backend.prefill_impl == "fused_chunk_parallel"
    assert backend._prefill_callable is fused_chunk_parallel_prefill
    assert backend._decode_callable is decode_gated_delta_rule_ref
    assert not hasattr(backend, "requested_prefill_impl")
    assert not hasattr(backend, "effective_prefill_impl")
    assert not hasattr(backend, "fallback_reason")
    assert (
        sum(
            "GDN prefill implementation=fused_chunk_parallel" in record.message
            for record in caplog.records
        )
        == 1
    )


def test_explicit_chunked_jax_uses_existing_prefill(monkeypatch):
    backend = _backend(monkeypatch, impl="chunked_jax")

    assert backend.prefill_impl == "chunked_jax"
    assert backend._prefill_callable is GDNAttnBackend._forward_extend_chunked_jax
    assert backend._decode_callable is decode_gated_delta_rule_ref


def test_fused_chunk_parallel_uses_distinct_frozen_adapter(monkeypatch):
    monkeypatch.setattr(
        "sgl_jax.srt.kernels.gdn.fused_chunk_parallel_adapter._mesh_devices",
        lambda _: (SimpleNamespace(platform="tpu"),),
    )
    backend = _backend(monkeypatch, impl="fused_chunk_parallel", dtype=jnp.bfloat16)

    assert backend.prefill_impl == "fused_chunk_parallel"
    assert backend._prefill_callable is fused_chunk_parallel_prefill
    assert backend._decode_callable is decode_gated_delta_rule_ref


@pytest.mark.parametrize("impl", ["reference", "separate", "chunkwise", "token_scan"])
def test_invalid_selector_fails_during_initialization(monkeypatch, impl):
    with pytest.raises(ValueError, match="gdn_prefill_impl"):
        _backend(monkeypatch, impl=impl)


@pytest.mark.parametrize(
    ("dtype", "kwargs", "match"),
    [
        (None, {}, "dtype"),
        (jnp.float32, {}, "BF16"),
        (jnp.bfloat16, {"conv_kernel_size": 1}, "conv_kernel_size"),
        (jnp.bfloat16, {"head_k_dim": 64}, "head_k_dim"),
        (jnp.bfloat16, {"head_v_dim": 192}, "head_v_dim"),
    ],
)
def test_fused_chunk_parallel_rejects_unsupported_startup_capability(
    monkeypatch, dtype, kwargs, match
):
    monkeypatch.setattr(
        "sgl_jax.srt.kernels.gdn.fused_chunk_parallel_adapter._mesh_devices",
        lambda _: (SimpleNamespace(platform="tpu"),),
    )
    with pytest.raises(GDNPrefillCapabilityError, match=match):
        _backend(monkeypatch, impl="fused_chunk_parallel", dtype=dtype, **kwargs)


def test_fused_chunk_parallel_requires_tpu_mesh_even_with_interpret(monkeypatch):
    monkeypatch.setenv("PALLAS_INTERPRET", "1")

    with pytest.raises(GDNPrefillCapabilityError, match="TPU"):
        _backend(monkeypatch, impl="fused_chunk_parallel", dtype=jnp.bfloat16)


def test_production_constructor_passes_model_dtype_and_prefill_impl_to_backend(monkeypatch):
    captured = {}

    def fake_gdn_backend(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        "sgl_jax.srt.layers.attention.linear.gdn_backend.GDNAttnBackend",
        fake_gdn_backend,
    )
    text_config = SimpleNamespace(
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    runner = SimpleNamespace(
        linear_recurrent_config=SimpleNamespace(full_attention_layer_ids=[1]),
        kimi_linear_config=None,
        qwen3_5_hybrid_config=SimpleNamespace(text_config=text_config),
        lightning_config=None,
        mesh=_mesh(),
        model_config=SimpleNamespace(dtype=jnp.bfloat16),
        server_args=SimpleNamespace(gdn_prefill_impl="chunked_jax"),
    )

    hybrid_linear_attn_backend.attn_backend_wrapper(runner, SimpleNamespace())

    assert captured["dtype"] == jnp.bfloat16
    assert captured["prefill_impl"] == "chunked_jax"
