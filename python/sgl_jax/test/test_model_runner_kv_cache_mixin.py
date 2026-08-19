import types

import jax.numpy as jnp
import pytest

from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
    _enforce_recurrent_state_server_constraints,
)
from sgl_jax.srt.server_args import ServerArgs


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("bf16", jnp.bfloat16),
        ("fp8_e4m3", jnp.float8_e4m3fn),
        ("fp8_e5m2", jnp.float8_e5m2),
    ],
)
def test_init_kv_cache_dtype(name, expected):
    runner = types.SimpleNamespace(
        dtype=jnp.float32,
        server_args=types.SimpleNamespace(kv_cache_dtype=name),
    )
    ModelRunnerKVCacheMixin._init_kv_cache_dtype(runner)
    assert runner.kv_cache_dtype == expected


def test_recurrent_state_legacy_disable_radix_cache_passes():
    sa = ServerArgs(model_path="dummy", disable_radix_cache=True)
    _enforce_recurrent_state_server_constraints(sa)  # no raise


def test_recurrent_state_radix_cache_requires_unified_radix_tree():
    sa = ServerArgs(
        model_path="dummy",
        disable_radix_cache=False,
        enable_unified_radix_tree=False,
    )
    with pytest.raises(AssertionError, match="unified-radix-tree"):
        _enforce_recurrent_state_server_constraints(sa)


def _fake_runner(max_recurrent_state_size, dp_size):
    sa = ServerArgs(
        model_path="dummy",
        enable_unified_radix_tree=True,
        disable_radix_cache=False,
        dp_size=dp_size,
        max_recurrent_state_size=max_recurrent_state_size,
    )
    return types.SimpleNamespace(
        is_draft_worker=False,
        spec_algorithm=None,
        linear_recurrent_config=object(),
        server_args=sa,
        dp_size=dp_size,
    )


def test_recurrent_admission_cap_is_dp_aligned():
    # factor=2: 12 // 2 = 6 is off the dp_size=4 grid, so the cap rounds down to 4.
    cap = ModelRunnerKVCacheMixin._resolve_max_num_reqs(_fake_runner(12, 4), 1000)
    assert cap == 4
    assert cap % 4 == 0


def test_recurrent_admission_cap_already_aligned_unchanged():
    # 16 // 2 = 8 is already a dp_size=4 multiple.
    cap = ModelRunnerKVCacheMixin._resolve_max_num_reqs(_fake_runner(16, 4), 1000)
    assert cap == 8


@pytest.mark.parametrize(("embedding_pool_bytes", "expected"), [(0, 700), (100, 600)])
def test_profile_available_bytes_reserves_static_and_embedding_pool(embedding_pool_bytes, expected):
    runner = types.SimpleNamespace(
        get_available_device_memory=lambda: 900,
        mem_fraction_static=0.8,
        embedding_pool_bytes=embedding_pool_bytes,
        linear_recurrent_config=None,
    )
    assert ModelRunnerKVCacheMixin._profile_available_bytes(runner, 1000) == expected


def test_embedding_pool_bytes_only_for_in_model_prefill():
    from sgl_jax.srt.model_executor.model_runner import _embedding_pool_bytes

    config = types.SimpleNamespace(
        is_multimodal=True,
        hidden_size=8,
        dtype=jnp.bfloat16,
        hf_config=types.SimpleNamespace(architectures=["Qwen2_5_VLForConditionalGeneration"]),
    )
    args = types.SimpleNamespace(
        mm_embedding_cache_size_mb=None,
        mm_embedding_page_size=64,
        max_prefill_tokens=100,
        multimodal=False,
        enable_lora=False,
        disaggregation_mode="null",
    )
    assert _embedding_pool_bytes(config, args) == 128 * 8 * 2
    args.mm_embedding_cache_size_mb = 128
    assert _embedding_pool_bytes(config, args) == 128 * 1024**2
    args.multimodal = True
    assert _embedding_pool_bytes(config, args) == 0


def test_deepstack_embedding_pool_uses_packed_feature_width():
    from sgl_jax.srt.model_executor.model_runner import (
        ModelRunner,
        _embedding_pool_bytes,
    )

    config = types.SimpleNamespace(
        is_multimodal=True,
        hidden_size=8,
        dtype=jnp.bfloat16,
        hf_config=types.SimpleNamespace(architectures=["Qwen2_5_VLForConditionalGeneration"]),
    )
    args = types.SimpleNamespace(
        mm_embedding_cache_size_mb=None,
        mm_embedding_page_size=64,
        max_prefill_tokens=100,
        multimodal=False,
        enable_lora=False,
        disaggregation_mode="null",
    )
    model = types.SimpleNamespace(deepstack_visual_layers=3)
    budget = _embedding_pool_bytes(config, args, multimodal_model=model)
    assert budget == 128 * 32 * 2

    runner = types.SimpleNamespace(
        embedding_pool_bytes=budget,
        server_args=args,
        model_config=config,
        model=model,
        dtype=jnp.bfloat16,
        mesh=None,
        embedding_pool=None,
    )
    ModelRunner._build_embedding_pool(runner)

    assert runner.embedding_pool.hidden == 32
    assert runner.embedding_pool.num_pages == 2
    assert runner.embedding_pool.pages.shape == (2, 64, 32)


def test_explicit_embedding_pool_budget_stays_fixed_for_deepstack():
    from sgl_jax.srt.model_executor.model_runner import (
        ModelRunner,
        _embedding_pool_bytes,
    )

    config = types.SimpleNamespace(
        is_multimodal=True,
        hidden_size=8,
        dtype=jnp.bfloat16,
        hf_config=types.SimpleNamespace(architectures=["Qwen2_5_VLForConditionalGeneration"]),
    )
    args = types.SimpleNamespace(
        mm_embedding_cache_size_mb=1,
        mm_embedding_page_size=64,
        max_prefill_tokens=100,
        multimodal=False,
        enable_lora=False,
        disaggregation_mode="null",
    )
    model = types.SimpleNamespace(deepstack_visual_layers=3)
    budget = _embedding_pool_bytes(config, args, multimodal_model=model)
    assert budget == 1024**2

    runner = types.SimpleNamespace(
        embedding_pool_bytes=budget,
        server_args=args,
        model_config=config,
        model=model,
        dtype=jnp.bfloat16,
        mesh=None,
        embedding_pool=None,
    )
    ModelRunner._build_embedding_pool(runner)

    assert runner.embedding_pool.hidden == 32
    assert runner.embedding_pool.num_pages == 256
