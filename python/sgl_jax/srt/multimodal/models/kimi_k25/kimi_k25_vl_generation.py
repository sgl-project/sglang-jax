import logging

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.layers.moe import create_moe_weights_mapping
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.deepseek_v3 import DeepseekV3ForCausalLM
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class KimiK25ForConditionalGeneration(DeepseekV3ForCausalLM):

    def __init__(
        self,
        config=None,
        dtype=None,
        mesh=None,
    ):
        self.config = config
        self.text_config = get_hf_text_config(config) or config
        self.dtype = dtype or jnp.bfloat16
        self.mesh = mesh

        # text_config.quantization_config is a raw dict from the JSON.
        # ModelConfig already handles quantization at the top-level hf_config, but
        # due to quantization config being nested in text_config, it still in JSON
        # Clear it here so FusedMoE doesn't receive a raw dict as the config contains 'pack-quantized'
        # format which isn't yet supported.
        if isinstance(getattr(self.text_config, "quantization_config", None), dict):
            self.text_config.quantization_config = None

        super().__init__(
            config=self.text_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        self.hf_weight_prefix = "language_model."

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        token_to_kv_pool = memory_pools.token_to_kv_pool
        hidden_states, layers_kv_fused, layers_topk_ids = self.model(
            forward_batch, token_to_kv_pool
        )

        output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        return output, {"token_to_kv_pool": layers_kv_fused}, True, layers_topk_ids

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings(model_config)
        loader.load_weights_from_safetensors(weight_mappings)

        for layer in self.model.layers:
            layer.self_attn.post_load_weights()
        logger.info("Kimi K2.5 Language model weights loaded successfully!")


EntryClass = KimiK25ForConditionalGeneration
