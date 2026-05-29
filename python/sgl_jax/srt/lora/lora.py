# Copyright 2023-2024 SGLang Team
# Modifications copyright 2025 SGLang-JAX Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Integrates "S-LoRA: Serving Thousands of Concurrent LoRA Adapters"
# and "Punica: Multi-Tenant LoRA Serving"

# LoRA layers class inheritance adapted from:
# https://github.com/vllm-project/vllm/blob/4abf6336ec65c270343eb895e7b18786e9274176/vllm/lora/layers.py

import logging
import re

import jax
from flax import nnx

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.hf_transformers_utils import AutoConfig
from sgl_jax.srt.lora.backend.base_backend import BaseLoRABackend
from sgl_jax.srt.lora.lora_config import LoRAConfig
from sgl_jax.srt.model_loader.loader import DefaultModelLoader

logger = logging.getLogger(__name__)


class LoRALayer(nnx.Module):
    def __init__(self, config: LoRAConfig, base_hf_config: AutoConfig):
        super().__init__()
        self.config: LoRAConfig = config
        self.base_hf_config: AutoConfig = base_hf_config

        # lora weights in cpu. The weights are loaded from checkpoint.
        self.weights: dict[str, jax.Array] = {}


class LoRAAdapter(nnx.Module):

    def __init__(
        self,
        uid: str,
        config: LoRAConfig,
        base_hf_config: AutoConfig,
        load_config: LoadConfig,
        lora_backend: BaseLoRABackend,  # note: Currently, only BgmvLoRABackend is supported.
    ):
        super().__init__()
        self.uid: str = uid
        self.config: LoRAConfig = config
        assert self.config.hf_config["peft_type"].lower() == "lora"
        self.base_hf_config: AutoConfig = base_hf_config
        self.load_config: LoadConfig = load_config
        self.lora_backend: BaseLoRABackend = lora_backend
        self.scaling: float = self.config.lora_alpha / self.config.r

        self.layers: list[LoRALayer] = nnx.data(
            [LoRALayer(config, base_hf_config) for _ in range(base_hf_config.num_hidden_layers)]
        )

        self.weights: dict[str, jax.Array] = {}

    # initialize the LoRA weights to cpu
    def initialize_weights(self):
        model_path = self.config.path
        loader = DefaultModelLoader(self.load_config)
        revision = getattr(self.config.hf_config, "revision", None)

        logger.info(
            "Loading LoRA weights from %s (revision=%s)",
            model_path,
            revision if revision else "default",
        )

        weight_count = 0
        layer_weight_count = {}
        skipped_layers = set()
        num_layers = len(self.layers)

        for name, loaded_weight in loader._get_weights_iterator(
            DefaultModelLoader.Source(model_path, revision=revision, fall_back_to_pt=True)
        ):
            match = re.search(r"layers\.(\d+)\.", name)
            if match is not None:
                layer_id = int(match.group(1))
                # Skip layers that are out of range (when using --model-layer-nums)
                if layer_id >= num_layers:
                    if layer_id not in skipped_layers:
                        skipped_layers.add(layer_id)
                    continue
                self.layers[layer_id].weights[name] = loaded_weight
                layer_weight_count[layer_id] = layer_weight_count.get(layer_id, 0) + 1
                weight_count += 1

                # Log first few weights for debugging
                if weight_count <= 5:
                    logger.debug(
                        "Loaded weight: %s, shape=%s, dtype=%s",
                        name,
                        loaded_weight.shape,
                        loaded_weight.dtype,
                    )
            else:
                self.weights[name] = loaded_weight
                weight_count += 1

        if skipped_layers:
            logger.info(
                "Skipped LoRA weights for layers %s (model only has %d layers)",
                sorted(skipped_layers),
                num_layers,
            )

        logger.info(
            "LoRA weights loaded: total=%d weights, layer_distribution=%s",
            weight_count,
            dict(sorted(layer_weight_count.items())),
        )
