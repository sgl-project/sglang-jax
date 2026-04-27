import unittest

import jax
import jax.numpy as jnp
from transformers import LlamaConfig

from sgl_jax.srt.configs.dtype_config import DtypeConfig
from sgl_jax.srt.models.llama import LlamaForCausalLM
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase


class TestDtypeConfigLlama(CustomTestCase):
    def test_llama_topology_dtype(self):
        config = LlamaConfig(
            hidden_size=256,
            intermediate_size=512,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_hidden_layers=1,
            vocab_size=32000,
        )
        mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
        jax.sharding.set_mesh(mesh)

        # Configure granular dtypes
        dtype_config = DtypeConfig(
            default_dtype=jnp.bfloat16,
            config_dict={
                "model": {
                    "layers": {
                        "self_attn": {
                            "q_proj": "bfloat16",
                            "k_proj": "float32",
                            "softmax": "float32",
                        },
                        "mlp": {
                            "gate_proj": "float32",
                        },
                    }
                },
                "lm_head": "float32",
            },
        )

        model = LlamaForCausalLM(
            config=config,
            mesh=mesh,
            dtype=jnp.bfloat16,
            dtype_config=dtype_config,
        )

        layer = model.model.layers[0]

        # Use assertions instead of prints to make it a real test
        self.assertEqual(model.dtype, jnp.bfloat16)
        self.assertEqual(model.lm_head.embedding[...].dtype, jnp.float32)
        self.assertEqual(layer.self_attn.q_proj.params_dtype, jnp.bfloat16)
        self.assertEqual(layer.self_attn.k_proj.params_dtype, jnp.float32)
        self.assertEqual(layer.self_attn.attn.softmax_dtype, jnp.float32)
        self.assertEqual(layer.mlp.gate_proj.params_dtype, jnp.float32)
        self.assertEqual(layer.mlp.up_proj.params_dtype, jnp.bfloat16)


if __name__ == "__main__":
    unittest.main()
