import unittest
import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig
from sgl_jax.srt.models.gemma4 import Gemma4Router, Gemma4DecoderLayer
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MHATokenToKVPool
from sgl_jax.test.test_utils import CustomTestCase, is_in_ci, write_github_step_summary

class TestGemma4MultiTP(CustomTestCase):
    def setUp(self):
        self.moe_config = PretrainedConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_attention_heads=8,
            num_key_value_heads=8,
            vocab_size=32000,
            num_hidden_layers=2,
            enable_moe_block=True,
            num_experts=8,
            top_k_experts=2,
            moe_intermediate_size=512,
            ep_size=1,
            layer_types=["full_attention", "sliding_attention"],
        )
        self.dense_config = PretrainedConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_attention_heads=8,
            num_key_value_heads=8,
            vocab_size=32000,
            num_hidden_layers=2,
            enable_moe_block=False,
            layer_types=["full_attention", "sliding_attention"],
        )

    def get_mesh(self, num_devices):
        devs = jax.devices()[:min(num_devices, len(jax.devices()))]
        return jax.sharding.Mesh(devs, ("tensor",), axis_types=(jax.sharding.AxisType.Explicit,))

    def test_gemma4_router_forward(self):
        mesh = self.get_mesh(1)
        with jax.set_mesh(mesh):
            router = Gemma4Router(self.moe_config, dtype=jnp.float32, mesh=mesh)
            hidden_states = jnp.ones((4, 512), dtype=jnp.float32)
            logits = router(hidden_states)
            self.assertEqual(logits.shape, (4, 8))
            self.assertTrue(jnp.all(jnp.isfinite(logits)))

    def test_gemma4_moe_tp_1(self):
        mesh = self.get_mesh(1)
        with jax.set_mesh(mesh):
            layer = Gemma4DecoderLayer(self.moe_config, mesh=mesh, layer_id=0, dtype=jnp.float32)
            self.assertIsNotNone(layer.router)
            self.assertIsNotNone(layer.topk)
            self.assertIsNotNone(layer.experts)

    def test_gemma4_moe_tp_2(self):
        mesh = self.get_mesh(2)
        with jax.set_mesh(mesh):
            layer = Gemma4DecoderLayer(self.moe_config, mesh=mesh, layer_id=0, dtype=jnp.float32)
            self.assertIsNotNone(layer.router)

    def test_gemma4_moe_tp_4(self):
        mesh = self.get_mesh(4)
        with jax.set_mesh(mesh):
            layer = Gemma4DecoderLayer(self.moe_config, mesh=mesh, layer_id=0, dtype=jnp.float32)
            self.assertIsNotNone(layer.router)

    def test_gemma4_moe_tp_8(self):
        mesh = self.get_mesh(8)
        with jax.set_mesh(mesh):
            layer = Gemma4DecoderLayer(self.moe_config, mesh=mesh, layer_id=0, dtype=jnp.float32)
            self.assertIsNotNone(layer.router)

    def test_gemma4_moe_ep_8(self):
        mesh = self.get_mesh(8)
        config = self.moe_config
        config.ep_size = 8
        with jax.set_mesh(mesh):
            layer = Gemma4DecoderLayer(config, mesh=mesh, layer_id=0, dtype=jnp.float32)
            self.assertIsNotNone(layer.router)
            self.assertEqual(layer.experts.ep_size, 8)
        config.ep_size = 1

    def test_gemma4_dense_tp_1(self):
        mesh = self.get_mesh(1)
        with jax.set_mesh(mesh):
            layer = Gemma4DecoderLayer(self.dense_config, mesh=mesh, layer_id=0, dtype=jnp.float32)
            self.assertIsNone(layer.router)

    def test_gemma4_dense_tp_2(self):
        mesh = self.get_mesh(2)
        with jax.set_mesh(mesh):
            layer = Gemma4DecoderLayer(self.dense_config, mesh=mesh, layer_id=0, dtype=jnp.float32)
            self.assertIsNone(layer.router)

    def test_gemma4_dense_tp_4(self):
        mesh = self.get_mesh(4)
        with jax.set_mesh(mesh):
            layer = Gemma4DecoderLayer(self.dense_config, mesh=mesh, layer_id=0, dtype=jnp.float32)
            self.assertIsNone(layer.router)

    def test_gemma4_dense_tp_8(self):
        mesh = self.get_mesh(8)
        with jax.set_mesh(mesh):
            layer = Gemma4DecoderLayer(self.dense_config, mesh=mesh, layer_id=0, dtype=jnp.float32)
            self.assertIsNone(layer.router)

if __name__ == "__main__":
    unittest.main()
