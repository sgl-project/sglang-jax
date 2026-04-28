"""Test that ModelRunner._kv_pool_layer_count and _compute_cell_size correctly
exclude KDA layers for hybrid linear-recurrent models, and that
init_memory_pool wraps the pool in HybridLinearKVPool.

Run:
  cd python && USE_DEVICE_TYPE=cpu python -m pytest \
    sgl_jax/test/model_executor/test_kv_pool_layer_count.py -v
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.memory_pool import HybridLinearKVPool, MLATokenToKVPool
from sgl_jax.srt.model_executor.model_runner import ModelRunner
from sgl_jax.test.test_utils import CustomTestCase


def _mesh_1():
    devices = np.array(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, axis_names=("data", "tensor"))


class TestKVPoolLayerCount(CustomTestCase):
    """Unit tests for ModelRunner._kv_pool_layer_count helper.

    `linear_recurrent_config` is a @property on ModelRunner; we monkey-patch
    the class temporarily to return a fixture during the test, then restore.
    """

    def test_kv_pool_layer_count_hybrid_returns_full_attn_count(self):
        """Hybrid recurrent runner returns L_full = 4, not L = 16."""
        cfg = SimpleNamespace(full_attention_layer_ids=[1, 5, 9, 13])
        runner = ModelRunner.__new__(ModelRunner)
        runner.model_config = SimpleNamespace(num_hidden_layers=16)
        runner.is_hybrid = False  # not SWA-hybrid

        original = ModelRunner.linear_recurrent_config
        try:
            ModelRunner.linear_recurrent_config = property(lambda self: cfg)
            self.assertEqual(runner._kv_pool_layer_count(), 4)
        finally:
            ModelRunner.linear_recurrent_config = original

    def test_kv_pool_layer_count_non_hybrid_returns_adjust_layer_num(self):
        """Non-hybrid runner falls back to adjust_layer_num() which returns
        num_hidden_layers when not is_hybrid."""
        runner = ModelRunner.__new__(ModelRunner)
        runner.model_config = SimpleNamespace(num_hidden_layers=32, hf_config=SimpleNamespace())
        runner.is_hybrid = False

        original = ModelRunner.linear_recurrent_config
        try:
            ModelRunner.linear_recurrent_config = property(lambda self: None)
            self.assertEqual(runner._kv_pool_layer_count(), 32)
        finally:
            ModelRunner.linear_recurrent_config = original


class TestInitMemoryPoolHybridRecurrent(CustomTestCase):
    """End-to-end check: when linear_recurrent_config is set on the runner,
    the constructed token_to_kv_pool is HybridLinearKVPool with inner pool
    sized to L_full (= len(full_attention_layer_ids)).

    Heavy hybrid_recurrent_utils helpers (RecurrentStatePool construction
    in _build_hybrid_pools) and HBM profiling (profile_max_num_token) are
    short-circuited via patch / stub — this test targets ONLY the KV pool
    construction branch in init_memory_pool.
    """

    def _make_runner(self, mesh):
        runner = ModelRunner.__new__(ModelRunner)
        hf_text_config = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        runner.model_config = SimpleNamespace(
            num_hidden_layers=16,
            head_dim=192,
            hf_config=SimpleNamespace(),
            hf_text_config=hf_text_config,
            context_len=2048,
            sliding_window=None,
            dtype=jnp.bfloat16,
            get_num_kv_heads=lambda tp: 1,
            get_total_num_kv_heads_with_replication=lambda tp: 1,
        )
        runner.dtype = jnp.bfloat16
        runner.server_args = SimpleNamespace(
            attention_backend="fa",
            kv_cache_dtype="auto",
            page_size=1,
            disable_radix_cache=True,
            disable_overlap_schedule=True,
            state_to_kv_ratio=0.9,
            speculative_algorithm=None,
            speculative_num_steps=0,
            speculative_eagle_topk=0,
            speculative_num_draft_tokens=0,
            max_num_reqs=None,
            draft_runner_cache_size=None,
        )
        runner.tp_size = 1
        runner.use_mla_backend = True
        runner.is_hybrid = False
        runner.is_draft_worker = False
        runner.spec_algorithm = None
        runner.mesh = mesh
        runner.page_size = 1
        runner.mem_fraction_static = 0.7
        runner.req_to_token_pool = None
        runner.token_to_kv_pool_allocator = None
        runner.attn_backend = MagicMock()
        # Stub profile_max_num_token (avoids get_available_device_memory).
        runner.profile_max_num_token = lambda total_device_memory: 4096
        return runner

    def test_init_memory_pool_constructs_hybrid_linear_kv_pool(self):
        linear_attn_config = {
            "kda_layers": [i for i in range(16) if i not in [1, 5, 9, 13]],
            "full_attn_layers": [1, 5, 9, 13],
            "num_heads": 8,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
        }
        cfg = SimpleNamespace(
            full_attention_layer_ids=[1, 5, 9, 13],
            linear_attn_config=linear_attn_config,
            is_linear_attn=True,
        )

        mesh = _mesh_1()
        runner = self._make_runner(mesh)

        original_property = ModelRunner.linear_recurrent_config
        try:
            ModelRunner.linear_recurrent_config = property(lambda self: cfg)

            with patch(
                "sgl_jax.srt.model_executor.model_runner._build_hybrid_pools",
                return_value=(MagicMock(), MagicMock(), MagicMock()),
            ):
                runner.init_memory_pool(
                    max_num_reqs=8,
                    max_total_tokens=4096,
                    total_device_memory=16 * 1024**3,
                )

            self.assertIsInstance(runner.token_to_kv_pool, HybridLinearKVPool)
            self.assertIsInstance(runner.token_to_kv_pool.full_kv_pool, MLATokenToKVPool)
            self.assertEqual(runner.token_to_kv_pool.full_kv_pool.layer_num, 4)
            self.assertEqual(
                runner.token_to_kv_pool.full_attention_layer_id_mapping,
                {1: 0, 5: 1, 9: 2, 13: 3},
            )
        finally:
            ModelRunner.linear_recurrent_config = original_property


if __name__ == "__main__":
    unittest.main()
