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
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.lora.utils import (
    get_lora_a_output_sharding,
    get_lora_b_output_sharding,
)

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

    def __call__(self, x: jax.Array):
        return self.base_layer(x)

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
        module_name: str,
        mesh: Mesh,
    ):
        self.set_lora = True
        self.lora_a_output_sharding = get_lora_a_output_sharding(module_name, mesh)
        self.lora_b_output_sharding = get_lora_b_output_sharding(module_name, mesh)

        current_A = getattr(self, "A_buffer", None)
        if (
            current_A is not None
            and isinstance(self.A_buffer, nnx.Param)
            and self.A_buffer.value.shape == A_buffer.shape
        ):

            self.A_buffer.value = A_buffer
            self.B_buffer.value = B_buffer
        else:
            # first initialize
            self.A_buffer = nnx.Param(A_buffer)
            self.B_buffer = nnx.Param(B_buffer)

    def apply_lora(self, operands) -> jax.Array:
        base_output, x = operands
        jax.debug.print("A_buffer row means: {}", jnp.mean(self.A_buffer.value, axis=1))
        lora_a_output = self.lora_backend.run_lora_a_gemm(
            x=x, weights=self.A_buffer.value, sharding=self.lora_a_output_sharding
        )
        lora_output = self.lora_backend.run_lora_b_gemm(
            x=lora_a_output,
            weights=self.B_buffer.value,
            base_output=base_output,
            sharding=self.lora_b_output_sharding,
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
        base_output, output_bias = self.base_layer(x)

        output = jax.lax.cond(
            self.set_lora, self.apply_lora, lambda operands: operands[0], (base_output, x)
        )

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
