"""init_memory_pool has_recurrent_state branch.

Uses SimpleNamespace mock for hf_config / server_args / mesh-related state to
exercise the hybrid construction path without loading a real model.
"""

import unittest
from types import SimpleNamespace

import jax


def _mock_hf_config(kda_layers=None, num_heads=2, head_dim=4, conv_kernel_size=4):
    """Mock hf_config with linear_attn_config sub-dict.

    NOTE: linear_attn_config is a `dict | None` in real Kimi-Linear HF config,
    not a nested config object (see Decision #12). Field access in the
    implementation uses dict[key] subscript, not obj.attr.
    """
    if kda_layers is None:
        return SimpleNamespace(linear_attn_config=None)
    return SimpleNamespace(
        linear_attn_config={
            "kda_layers": kda_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "short_conv_kernel_size": conv_kernel_size,
        }
    )


class TestHasRecurrentStateDetection(unittest.TestCase):
    """Detection logic: has_recurrent_state iff hf_config.linear_attn_config.kda_layers non-empty."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_detects_kimi_linear_kda_layers(self):
        from sgl_jax.srt.model_executor.model_runner import _has_recurrent_state

        hf = _mock_hf_config(kda_layers=[0, 1, 2])
        self.assertTrue(_has_recurrent_state(hf))

    def test_no_linear_attn_config_returns_false(self):
        from sgl_jax.srt.model_executor.model_runner import _has_recurrent_state

        hf = SimpleNamespace()  # no linear_attn_config attr
        self.assertFalse(_has_recurrent_state(hf))

    def test_linear_attn_config_none_returns_false(self):
        from sgl_jax.srt.model_executor.model_runner import _has_recurrent_state

        hf = _mock_hf_config(kda_layers=None)
        self.assertFalse(_has_recurrent_state(hf))

    def test_empty_kda_layers_returns_false(self):
        from sgl_jax.srt.model_executor.model_runner import _has_recurrent_state

        hf = _mock_hf_config(kda_layers=[])
        self.assertFalse(_has_recurrent_state(hf))

    def test_kda_layers_key_missing_returns_false(self):
        """linear_attn_config exists but lacks "kda_layers" key (corrupt config)."""
        from sgl_jax.srt.model_executor.model_runner import _has_recurrent_state

        hf = SimpleNamespace(linear_attn_config={"num_heads": 2})  # no kda_layers key
        self.assertFalse(_has_recurrent_state(hf))


class TestInitMemoryPoolHybridBranch(unittest.TestCase):
    """D3 decision: 3 integration tests covering the full has_recurrent_state branch
    × KV pool type matrix (MHA / MLA). No 4-way over-coverage; MHA and MLA share
    the same MemoryPools wrapping path and only differ in the KV pool object.
    """

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_standard_model_no_recurrent_state(self):
        """has_recurrent_state=False -> MemoryPools contains only token_to_kv_pool;
        req_to_token_pool is plain ReqToTokenPool."""
        from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
        from sgl_jax.srt.model_executor.model_runner import (
            _build_non_hybrid_memory_pools,
            _has_recurrent_state,
        )

        hf = _mock_hf_config(kda_layers=None)
        self.assertFalse(_has_recurrent_state(hf))

        kv_stub = SimpleNamespace(replace_buffer=lambda v: None)
        mp = _build_non_hybrid_memory_pools(token_to_kv_pool=kv_stub)
        self.assertIsInstance(mp, MemoryPools)
        self.assertEqual(set(mp._pools.keys()), {"token_to_kv_pool"})
        # Plain ReqToTokenPool path is unchanged (caller handles construction).

    def test_hybrid_with_mha_pool(self):
        """has_recurrent_state=True with MHA-like KV pool stub -> MemoryPools
        contains both pools; req_to_token_pool is HybridReqToTokenPool."""
        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool, MemoryPools
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
        from sgl_jax.srt.model_executor.model_runner import _build_hybrid_pools

        hf = _mock_hf_config(kda_layers=[0, 1], num_heads=2, head_dim=4)
        # Stub mimics MHATokenToKVPool externally (any object with replace_buffer).
        mha_like_stub = SimpleNamespace(replace_buffer=lambda v: None)
        rsp, hybrid_pool, mp = _build_hybrid_pools(
            hf_config=hf,
            max_num_reqs=4,
            max_context_len=16,
            tp_size=1,
            token_to_kv_pool=mha_like_stub,
        )
        self.assertIsInstance(rsp, RecurrentStatePool)
        self.assertIsInstance(hybrid_pool, HybridReqToTokenPool)
        self.assertIsInstance(mp, MemoryPools)
        self.assertEqual(set(mp._pools.keys()), {"token_to_kv_pool", "recurrent_state_pool"})
        self.assertIs(mp.token_to_kv_pool, mha_like_stub)
        self.assertIs(mp.recurrent_state_pool, rsp)
        self.assertIs(hybrid_pool.recurrent_state_pool, rsp)
        # linear_attn_config dict subscript path produced expected layer ids.
        self.assertEqual(rsp.linear_recurrent_layer_ids, [0, 1])

    def test_hybrid_with_mla_pool(self):
        """Kimi-Linear real-world scenario: has_recurrent_state=True with MLA-like
        KV pool. Verifies MLA pool enters MemoryPools without special handling
        (transparent to wrapping layer). Same shape as MHA case; only KV pool
        type differs - this IS the proof that MHA/MLA pool is transparent to
        the MemoryPools wrapper."""
        from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
        from sgl_jax.srt.model_executor.model_runner import _build_hybrid_pools

        hf = _mock_hf_config(kda_layers=[0, 1], num_heads=2, head_dim=4)
        # Stub mimics MLATokenToKVPool externally (different shape/dim
        # internally, but MemoryPools sees only `replace_buffer`).
        mla_like_stub = SimpleNamespace(
            replace_buffer=lambda v: None,
            kv_lora_rank=512,  # MLA-specific attr; MemoryPools doesn't touch it
        )
        rsp, hybrid_pool, mp = _build_hybrid_pools(
            hf_config=hf,
            max_num_reqs=4,
            max_context_len=16,
            tp_size=1,
            token_to_kv_pool=mla_like_stub,
        )
        self.assertIsInstance(mp, MemoryPools)
        self.assertEqual(set(mp._pools.keys()), {"token_to_kv_pool", "recurrent_state_pool"})
        # MLA-specific attribute survived (MemoryPools doesn't strip it).
        self.assertEqual(mp.token_to_kv_pool.kv_lora_rank, 512)
        # Both pools wired correctly.
        self.assertIs(hybrid_pool.recurrent_state_pool, rsp)


class TestServerArgsForcedConstraints(unittest.TestCase):
    """has_recurrent_state model must force disable_radix_cache=True
    and assert disable_overlap_schedule=True."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_force_disable_radix_cache_logs_and_sets_true(self):
        from sgl_jax.srt.model_executor.model_runner import (
            _enforce_recurrent_state_server_constraints,
        )

        sa = SimpleNamespace(disable_radix_cache=False, disable_overlap_schedule=True)
        _enforce_recurrent_state_server_constraints(sa)
        self.assertTrue(sa.disable_radix_cache)

    def test_disable_radix_cache_already_true_is_idempotent(self):
        from sgl_jax.srt.model_executor.model_runner import (
            _enforce_recurrent_state_server_constraints,
        )

        sa = SimpleNamespace(disable_radix_cache=True, disable_overlap_schedule=True)
        _enforce_recurrent_state_server_constraints(sa)
        self.assertTrue(sa.disable_radix_cache)

    def test_assert_overlap_schedule_disabled(self):
        from sgl_jax.srt.model_executor.model_runner import (
            _enforce_recurrent_state_server_constraints,
        )

        sa = SimpleNamespace(disable_radix_cache=False, disable_overlap_schedule=False)
        with self.assertRaises(AssertionError):
            _enforce_recurrent_state_server_constraints(sa)


if __name__ == "__main__":
    unittest.main()
