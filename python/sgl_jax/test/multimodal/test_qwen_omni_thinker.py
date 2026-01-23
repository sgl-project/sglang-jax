import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import unittest

import jax
import jax.numpy as jnp
from flax import nnx
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTextConfig,
)

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


class TestQwen3OmniMoePrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = {
            "_name_or_path": "",
            "add_cross_attention": False,
            "architectures": ["Qwen3OmniMoeThinkerForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bad_words_ids": None,
            "begin_suppress_tokens": None,
            "bos_token_id": None,
            "chunk_size_feed_forward": 0,
            "cross_attention_hidden_size": None,
            "decoder_sparse_step": 1,
            "decoder_start_token_id": None,
            "diversity_penalty": 0.0,
            "do_sample": False,
            "dtype": None,
            "early_stopping": False,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": None,
            "exponential_decay_length_penalty": None,
            "finetuning_task": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
            "initializer_range": 0.02,
            "intermediate_size": 768,
            "is_decoder": False,
            "is_encoder_decoder": False,
            "label2id": {"LABEL_0": 0, "LABEL_1": 1},
            "length_penalty": 1.0,
            "max_length": 20,
            "max_position_embeddings": 65536,
            "min_length": 0,
            "mlp_only_layers": [],
            "model_type": "qwen3_omni_moe_text",
            "moe_intermediate_size": 768,
            "no_repeat_ngram_size": 0,
            "norm_topk_prob": True,
            "num_attention_heads": 32,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 48,
            "num_key_value_heads": 4,
            "num_return_sequences": 1,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_router_logits": False,
            "output_scores": False,
            "pad_token_id": None,
            "prefix": None,
            "problem_type": None,
            "pruned_heads": {},
            "remove_invalid_values": False,
            "repetition_penalty": 1.0,
            "return_dict": True,
            "return_dict_in_generate": False,
            "rms_norm_eps": 1e-06,
            "rope_scaling": {
                "interleaved": True,
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
                "rope_type": "default",
                "type": "default",
            },
            "rope_theta": 1000000,
            "router_aux_loss_coef": 0.001,
            "sep_token_id": None,
            "shared_expert_intermediate_size": 0,
            "sliding_window": None,
            "suppress_tokens": None,
            "task_specific_params": None,
            "temperature": 1.0,
            "tf_legacy_loss": False,
            "tie_encoder_decoder": False,
            "tie_word_embeddings": False,
            "tokenizer_class": None,
            "top_k": 50,
            "top_p": 1.0,
            "torchscript": False,
            "typical_p": 1.0,
            "use_bfloat16": False,
            "use_cache": True,
            "use_qk_norm": True,
            "use_sliding_window": False,
            "vocab_size": 152064,
        }
        cls.config = Qwen3OmniMoeTextConfig(**config)
        cls.config.ep_size = 1
        cls.mesh = create_device_mesh(
            ici_parallelism=[-1, 1],
            dcn_parallelism=[1, 1],
        )
        # print(cls.mesh)
        with jax.set_mesh(cls.mesh):
            cls.model = Qwen3OmniMoeForConditionalGeneration(
                config=cls.config, mesh=cls.mesh, dtype=jnp.float32
            )
        model_def, state = nnx.split(cls.model)
        # for key, value in state.flat_state():
        # print('.'.join([str(k) for k in key]), value.shape)
        model_config = ModelConfig(model_path="/models/Qwen/Qwen3-Omni-30B-A3B-Instruct")
        cls.model.load_weights(model_config=model_config)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("JAX_PLATFORMS", None)

    def test_config(self):
        self.assertEqual(self.config.hidden_size, 2048)

    def test_forward(self):
        pass
        # arr = jnp.array(
        #     [
        #         [
        #             1474,
        #             25,
        #             220,
        #             14880,
        #             11622,
        #             105321,
        #             104136,
        #             106582,
        #             109539,
        #             115822,
        #             1773,
        #             71703,
        #             25,
        #             220,
        #         ]
        #     ]
        # )
        # y = self.model.model.embed_tokens(arr)
        # y = y.reshape(-1, y.shape[-1])
        # np.savetxt("jax_output_before.txt", jax.device_get(y[-1,:].astype(np.float32)), fmt="%.15f")
        # y = self.model.model.norm(y)
        # print(self.model.lm_head.embedding.value.T.dtype, y.dtype)
        # lm_head_out = jnp.dot(y, self.model.lm_head.embedding.value.T)
        # np.savetxt("jax_output.txt", jax.device_get(lm_head_out[-1,:].astype(np.float32)), fmt="%.15f")
        # np.savetxt("jax_weight.txt", jax.device_get(self.model.lm_head.embedding.value.T[0,:].astype(np.float32)), fmt="%.15f")


if __name__ == "__main__":
    unittest.main()
