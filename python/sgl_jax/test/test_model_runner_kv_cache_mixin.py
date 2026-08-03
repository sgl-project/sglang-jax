import types

import pytest

from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
    _enforce_recurrent_state_server_constraints,
)
from sgl_jax.srt.server_args import ServerArgs


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


def test_profile_available_bytes_reserves_only_static_fraction():
    runner = types.SimpleNamespace(
        get_available_device_memory=lambda: 900,
        mem_fraction_static=0.8,
        embedding_pool_bytes=0,
        linear_recurrent_config=None,
    )
    assert ModelRunnerKVCacheMixin._profile_available_bytes(runner, 1000) == 700


def test_embedding_pool_bytes_reserved_from_kv_budget():
    runner = types.SimpleNamespace(
        get_available_device_memory=lambda: 900,
        mem_fraction_static=0.8,
        embedding_pool_bytes=100,
        linear_recurrent_config=None,
    )
    assert ModelRunnerKVCacheMixin._profile_available_bytes(runner, 1000) == 600


def test_embedding_pool_bytes_only_for_in_model_prefill():
    from sgl_jax.srt.model_executor.model_runner import _embedding_pool_bytes

    config = types.SimpleNamespace(
        is_multimodal=True,
        hf_config=types.SimpleNamespace(architectures=["Qwen2_5_VLForConditionalGeneration"]),
    )
    args = types.SimpleNamespace(
        mm_embedding_cache_size_mb=128,
        multimodal=False,
        enable_lora=False,
        disaggregation_mode="null",
    )
    assert _embedding_pool_bytes(config, args) == 128 * 1024**2
    args.multimodal = True
    assert _embedding_pool_bytes(config, args) == 0
