# Copyright 2023-2024 SGLang Team
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
from sgl_jax.srt.lora.lora_config import LoRAConfig
from sgl_jax.srt.model_loader.loader import DefaultModelLoader

logger = logging.getLogger(__name__)


class BaseLoRABackend:
    pass


class ChunkedSgmvLoRABackend(BaseLoRABackend):
    pass


SUPPORTED_BACKENDS = ChunkedSgmvLoRABackend


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
        lora_backend: BaseLoRABackend,
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
        for name, loaded_weight in loader._get_weights_iterator(
            DefaultModelLoader.Source(model_path, revision=revision, fall_back_to_pt=True)
        ):
            match = re.search(r"layers\.(\d+)\.", name)
            if match is not None:
                layer_id = int(match.group(1))
                self.layers[layer_id].weights[name] = loaded_weight
            else:
                self.weights[name] = loaded_weight
