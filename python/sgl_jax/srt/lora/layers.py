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
"""LoRA layer wrappers using Flax Model Surgery."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from flax import nnx

from python.sgl_jax.srt.layers.linear import LinearBase

if TYPE_CHECKING:
    from sgl_jax.srt.lora.backend.base_backend import BaseLoRABackend


class BaseLayerWithLoRA(nnx.Module):
    def __init__(
        self,
        base_layer: nnx.Module,
        lora_backend: BaseLoRABackend,
    ):
        super().__init__()
        self.base_layer: nnx.Module = base_layer
        self.set_lora: bool = False
        self.lora_backend: BaseLoRABackend = lora_backend
        if hasattr(self.base_layer, "weight"):
            self.weight = self.base_layer.weight

    def forward(self, x: jax.Array):
        return self.base_layer.forward(x)

    def set_lora_info(self, *args):
        pass


class LoRALinear(BaseLayerWithLoRA):
    """
    LoRA wrapper for Linear layers using Flax NNX.
    This wraps an existing Linear layer and adds LoRA (Low-Rank Adaptation)
    computation. Uses Model Surgery to preserve the original weights and sharding.

    Attributes:
        base_layer: Original Linear layer (preserves weights and sharding)
        backend: LoRA backend for efficient computation
    """

    def __init__(
        self,
        base_layer: LinearBase | None = None,
        lora_backend: BaseLoRABackend | None = None,
    ):
        """
        Initialize LoRA Linear layer.

        Args:
            base_layer: Existing Linear layer to wrap
            backend: LoRA backend for computation
        """
        super().__init__(base_layer, lora_backend)
        self.lora_backend = lora_backend

    def set_lora_info(
        self,
        A_buffer: jax.Array,
        B_buffer: jax.Array,
    ):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(self, base_output: jax.Array, x: jax.Array) -> jax.Array:
        lora_a_output = self.lora_backend.run_lora_a_gemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_gemm(
            x=lora_a_output,
            weights=self.B_buffer,
            base_output=base_output,
        )
        return lora_output

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array | None]:
        """
        Forward pass with optional LoRA computation using backend.

        Args:
            x: Input tensor (shape: [seq_len, in_features])

        Returns:
            Output tensor with LoRA delta added (if enabled) and bias from base_model
        """
        # Base layer computation (preserves original behavior)
        output_bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        base_output = self.base_layer(x)

        output = self.apply_lora(base_output, x) if self.set_lora else base_output

        return output, output_bias


class LoRAEmbedding(BaseLayerWithLoRA):
    """
    LoRA wrapper for Embedding layers.

    Similar to LoRALinear but for embedding layers.
    V1 implementation uses backend for computation.
    """

    def __init__(
        self,
        base_layer: LinearBase | None = None,
        lora_backend: BaseLoRABackend | None = None,
    ):
        """
        Initialize LoRA Embedding layer.

        Args:
            base_layer: Existing Embed layer to wrap
            backend: LoRA backend for computation
        """
        super().__init__(base_layer, lora_backend)
        self.weight = base_layer.weight
