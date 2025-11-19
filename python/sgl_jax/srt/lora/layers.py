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

if TYPE_CHECKING:
    from sgl_jax.srt.lora.backend.base_backend import BaseLoRABackend


class LoRALinear(nnx.Module):
    """
    LoRA wrapper for Linear layers using Flax NNX.

    This wraps an existing Linear layer and adds LoRA (Low-Rank Adaptation)
    computation. Uses Model Surgery to preserve the original weights and sharding.

    V1 implementation uses backend to perform LoRA computation:
        output = base_layer(x)
        if enabled:
            lora_output = backend.run_lora_a_gemm(x, lora_A_weights)
            output = backend.run_lora_b_gemm(lora_output, lora_B_weights, output)

    Attributes:
        base_layer: Original Linear layer (preserves weights and sharding)
        lora_rank: LoRA rank dimension
        backend: LoRA backend for efficient computation
        enabled: Whether LoRA computation is active
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int,
        base_layer: nnx.Linear | None = None,
        backend: BaseLoRABackend | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        """
        Initialize LoRA Linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            lora_rank: Rank of LoRA matrices
            base_layer: Existing Linear layer to wrap (optional)
            backend: LoRA backend for computation (optional)
            rngs: Random number generators for initialization
        """
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.backend = backend

        # Base layer - will be populated via nnx.update() during surgery
        if base_layer is not None:
            self.base_layer = base_layer
        else:
            # Create placeholder base layer
            if rngs is None:
                rngs = nnx.Rngs(0)
            self.base_layer = nnx.Linear(
                in_features,
                out_features,
                use_bias=True,
                rngs=rngs,
            )

        # Control variable (not trainable)
        self.enabled = nnx.Variable(False)  # Whether LoRA is active

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass with optional LoRA computation using backend.

        Args:
            x: Input tensor (shape: [seq_len, in_features])

        Returns:
            Output tensor with LoRA delta added (if enabled)
        """
        # Base layer computation (preserves original behavior)
        output = self.base_layer(x)

        # Add LoRA delta if enabled and backend is available
        if self.enabled.value and self.backend is not None:
            # Get LoRA weights from memory pool via backend
            # Backend handles batched LoRA computation for multiple adapters

            # Step 1: Shrink - project to low-rank space
            # lora_A_weights fetched from memory pool based on batch_info
            lora_a_output = self.backend.run_lora_a_gemm(
                x, None
            )  # Backend manages weights internally

            # Step 2: Expand - project back to output space and add to base output
            output = self.backend.run_lora_b_gemm(lora_a_output, None, output)

        return output


class LoRAEmbedding(nnx.Module):
    """
    LoRA wrapper for Embedding layers.

    Similar to LoRALinear but for embedding layers.
    V1 implementation uses backend for computation.
    """

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        lora_rank: int,
        base_layer: nnx.Embed | None = None,
        backend: BaseLoRABackend | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        """
        Initialize LoRA Embedding layer.

        Args:
            num_embeddings: Size of vocabulary
            features: Embedding dimension
            lora_rank: Rank of LoRA matrices
            base_layer: Existing Embed layer to wrap (optional)
            backend: LoRA backend for computation (optional)
            rngs: Random number generators
        """
        self.num_embeddings = num_embeddings
        self.features = features
        self.lora_rank = lora_rank
        self.backend = backend

        # Base layer
        if base_layer is not None:
            self.base_layer = base_layer
        else:
            if rngs is None:
                rngs = nnx.Rngs(0)
            self.base_layer = nnx.Embed(
                num_embeddings,
                features,
                rngs=rngs,
            )

        # Control variable
        self.enabled = nnx.Variable(False)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass for embedding with LoRA using backend.

        Args:
            x: Input token indices

        Returns:
            Embedded output with LoRA delta (if enabled)
        """
        output = self.base_layer(x)

        # V1: Embedding LoRA computation via backend
        # TODO: Implement embedding-specific backend methods if needed
        # For now, embeddings use simple pass-through
        if self.enabled.value and self.backend is not None:
            # Backend handles embedding LoRA computation
            pass

        return output
