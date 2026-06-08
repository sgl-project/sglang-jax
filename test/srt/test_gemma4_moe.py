import unittest
import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig
from sgl_jax.srt.models.gemma4 import Gemma4Router, Gemma4DecoderLayer
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MHATokenToKVPool
from sgl_jax.test.test_utils import CustomTestCase, is_in_ci, write_github_step_summary

class TestGemma4MoE(CustomTestCase):
    def setUp(self):
        self.mesh = jax.sharding.Mesh(jax.devices()[:1], ("tensor",))
        self.config = PretrainedConfig(
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

    def test_gemma4_router_forward(self):
        router = Gemma4Router(self.config, dtype=jnp.float32)
        hidden_states = jnp.ones((4, 512), dtype=jnp.float32)
        logits = router(hidden_states)
        self.assertEqual(logits.shape, (4, 8))
        self.assertTrue(jnp.all(jnp.isfinite(logits)))
        if is_in_ci():
            write_github_step_summary("### test_gemma4_router_forward\nRouter forward pass completed successfully.\n")

    def test_gemma4_moe_decoder_layer_init(self):
        layer = Gemma4DecoderLayer(self.config, mesh=self.mesh, layer_id=0, dtype=jnp.float32)
        self.assertIsNotNone(layer.router)
        self.assertIsNotNone(layer.topk)
        self.assertIsNotNone(layer.experts)
        if is_in_ci():
            write_github_step_summary("### test_gemma4_moe_decoder_layer_init\nMoE decoder layer structure verified successfully.\n")

if __name__ == "__main__":
    unittest.main()
