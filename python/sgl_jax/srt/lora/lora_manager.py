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
"""LoRA manager implementation for JAX - Phase 3 placeholder."""

from __future__ import annotations

import logging

import jax.numpy as jnp
from jax.sharding import Mesh

from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.lora.layers import BaseLayerWithLoRA
from sgl_jax.srt.lora.lora import ChunkedSgmvLoRABackend, LoRAAdapter
from sgl_jax.srt.lora.lora_config import LoRAConfig
from sgl_jax.srt.lora.lora_memory_pool import LoRAMemoryPool
from sgl_jax.srt.lora.lora_registry import LoRARef
from sgl_jax.srt.lora.utils import get_target_module_name
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch

logger = logging.getLogger(__name__)


class LoRAManager:
    """
    LoRA manager for JAX-based inference.

    V1 implementation: Simplified version with static adapter loading.
    - All LoRA adapters are loaded at initialization time
    - No dynamic loading/unloading during inference
    - No eviction policy (all adapters stay in CPU memory)
    - Adapters are transferred to device memory pool on-demand per batch

    Key differences from PyTorch/SGLang version:
    - Uses JAX arrays instead of PyTorch tensors
    - Memory pool uses JAX sharding for multi-device support
    - Layer wrapping uses Flax NNX Model Surgery
    - No kernel backend (uses JAX/XLA compilation instead)

    Future enhancements (V2+):
    - Dynamic adapter loading/unloading
    - LRU/FIFO eviction policies
    - Adapter registry with async loading
    - Support for larger number of adapters than memory pool slots

    Attributes:
        max_loras_per_batch: Maximum number of LoRA adapters per batch
        max_lora_rank: Maximum LoRA rank supported
        num_layers: Number of transformer layers
        target_modules: Set of target module names
        mesh: JAX device mesh for sharding
        dtype: Data type for LoRA weights
        configs: Dict mapping lora_id -> LoRAConfig
        loras: Dict mapping lora_id -> LoRAAdapter (CPU-side weights)
        lora_refs: Dict mapping lora_id -> LoRARef
        memory_pool: LoRAMemoryPool instance
    """

    def __init__(
        self,
        base_model,
        base_hf_config,
        max_loras_per_batch: int,
        dtype: jnp.dtype,
        mesh: Mesh,
        max_lora_rank: int | None = None,
        target_modules: set[str] | None = None,
        lora_paths: list[LoRARef] | None = None,
        server_args=None,
        model_config=None,
    ):
        """
        Initialize LoRA manager.

        Args:
            base_model: The base model to apply LoRA to
            base_hf_config: HuggingFace config of the base model
            max_loras_per_batch: Maximum number of LoRA adapters in a batch
            dtype: Data type for LoRA weights
            mesh: JAX device mesh for sharding
            max_lora_rank: Maximum LoRA rank to support (or None to infer)
            target_modules: Set of target module names (or None to infer)
            lora_paths: Optional list of LoRARef to preload
            server_args: Server arguments (for future use)
            model_config: ModelConfig instance (for accessing original_num_kv_heads)
        """
        self.base_model = base_model
        self.base_hf_config = base_hf_config
        self.max_loras_per_batch = max_loras_per_batch
        self.dtype = dtype
        self.mesh = mesh
        self.server_args = server_args
        self.model_config = model_config

        # Extract model architecture from hf_config
        self.num_layers = base_hf_config.num_hidden_layers
        self.hidden_size = base_hf_config.hidden_size
        self.intermediate_size = getattr(base_hf_config, "intermediate_size", self.hidden_size * 4)
        self.num_attention_heads = base_hf_config.num_attention_heads
        self.num_kv_heads = getattr(base_hf_config, "num_key_value_heads", self.num_attention_heads)
        self.head_dim = getattr(base_hf_config, "head_dim", None)

        # Get original num_kv_heads and tp_size for replication
        if model_config is not None:
            self.original_num_kv_heads = getattr(
                model_config, "_original_num_key_value_heads", self.num_kv_heads
            )
            self.tp_size = mesh.shape.get("tensor", 1) if hasattr(mesh, "shape") else 1
        else:
            # Fallback: assume no replication
            self.original_num_kv_heads = self.num_kv_heads
            self.tp_size = 1

        # Initialize mutable state
        self.init_state(
            max_lora_rank=max_lora_rank,
            target_modules=target_modules,
            lora_paths=lora_paths,
        )

    def init_state(
        self,
        max_lora_rank: int | None = None,
        target_modules: set[str] | None = None,
        lora_paths: list[LoRARef] | None = None,
    ):
        """
        Initialize internal state of LoRAManager.

        Args:
            max_lora_rank: Maximum LoRA rank (or None to infer from lora_paths)
            target_modules: Target module names (or None to infer from lora_paths)
            lora_paths: Optional list of LoRARef to preload
        """
        # Validate arguments
        if not lora_paths and (max_lora_rank is None or target_modules is None):
            raise ValueError(
                "When no lora_paths provided, must specify both max_lora_rank and target_modules"
            )

        # Initialize adapter storage
        self.init_lora_adapters(lora_paths)

        # Infer or validate shapes
        self.init_lora_shapes(
            max_lora_rank=max_lora_rank,
            target_modules=target_modules,
        )

        # Apply Model Surgery to add LoRA layers (if base_model provided)
        if self.base_model is not None:
            self.apply_lora_surgery()

        # Initialize memory pool
        self.init_memory_pool()

        self.update_lora_info()

        logger.info(
            "LoRA manager initialized: max_rank=%d, target_modules=%s, max_loras=%d",
            self.max_lora_rank,
            self.target_modules,
            self.max_loras_per_batch,
        )

    def init_lora_adapters(self, lora_paths: list[LoRARef] | None = None):
        """
        Initialize adapter storage and optionally load adapters.

        Args:
            lora_paths: Optional list of LoRARef to preload
        """
        # Configs of all active LoRA adapters, indexed by LoRA ID
        self.configs: dict[str, LoRAConfig] = {}

        # LoRA adapter weights cached in CPU memory, indexed by LoRA ID
        self.loras: dict[str, LoRAAdapter] = {}

        # Mapping from LoRA ID to LoRARef object
        self.lora_refs: dict[str, LoRARef] = {}

        # Count of pinned LoRA adapters
        self.num_pinned_loras: int = 0

        if lora_paths:
            for lora_ref in lora_paths:
                self.load_lora_adapter(lora_ref)

    def init_lora_shapes(
        self,
        max_lora_rank: int | None = None,
        target_modules: set[str] | None = None,
    ):
        """
        Infer LoRA target modules and max_lora_rank from loaded adapters if not provided.

        Args:
            max_lora_rank: Maximum LoRA rank (or None to infer)
            target_modules: Target module names (or None to infer)
        """
        # Initialize target_modules
        if target_modules is not None:
            self.target_modules = target_modules
        else:
            self.target_modules = set()

        # Infer from loaded adapters
        for lora_id, config in self.configs.items():
            adapter_target_modules = set(config.target_modules)

            if target_modules is not None:
                # Validate adapter is compatible
                if not adapter_target_modules.issubset(self.target_modules):
                    unsupported = adapter_target_modules - self.target_modules
                    lora_name = self.lora_refs[lora_id].lora_name
                    raise ValueError(
                        "LoRA adapter '%s' contains unsupported modules: %s. "
                        "Specified target_modules: %s",
                        lora_name,
                        unsupported,
                        self.target_modules,
                    )
            else:
                # Infer target_modules from adapter
                self.target_modules.update(adapter_target_modules)

        # Infer or use max_lora_rank
        if max_lora_rank is not None:
            self.max_lora_rank = max_lora_rank
        else:
            self.max_lora_rank = max(
                [config.r for config in self.configs.values()],
                default=8,  # Default rank if no adapters loaded
            )

    def init_memory_pool(self):
        """Initialize the LoRA memory pool with proper sharding."""
        self.memory_pool = LoRAMemoryPool(
            max_loras_per_batch=self.max_loras_per_batch,
            max_lora_rank=self.max_lora_rank,
            num_layers=self.num_layers,
            target_modules=self.target_modules,
            mesh=self.mesh,
            dtype=self.dtype,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            original_num_kv_heads=self.original_num_kv_heads,
            tp_size=self.tp_size,
        )
        self.memory_pool.init_buffers()

    def update_lora_info(self):
        """
        Update all LoRA modules to associate them with the latest memory buffer.
        """
        for layer_id, layer_modules in enumerate(self.lora_modules):
            for module_name, module in layer_modules.items():
                target_module = get_target_module_name(module_name, self.memory_pool.target_modules)
                module.set_lora_info(
                    self.memory_pool.get_tensor(
                        module_name=target_module,
                        layer_id=layer_id,
                        is_lora_a=True,
                    ),
                    self.memory_pool.get_tensor(
                        module_name=target_module,
                        layer_id=layer_id,
                        is_lora_a=False,
                    ),
                    target_module,
                    self.mesh,
                )

    def load_lora_adapter(self, lora_ref: LoRARef):
        """
        Load a single LoRA adapter.

        V1 implementation: Loads config and weights from disk to CPU memory once.
        No dynamic loading/unloading.

        Args:
            lora_ref: LoRARef object with lora_id, lora_name, lora_path

        Raises:
            ValueError: If adapter already loaded or incompatible
        """
        if lora_ref.lora_id in self.loras:
            raise ValueError(f"LoRA adapter {lora_ref.lora_id} already loaded")

        if lora_ref.pinned and self.num_pinned_loras >= self.max_loras_per_batch - 1:
            raise ValueError(
                f"Cannot pin adapter {lora_ref.lora_name}: already have {self.num_pinned_loras} "
                f"pinned adapters (max {self.max_loras_per_batch - 1}, reserving 1 slot for dynamic use)"
            )

        # Load config
        config = LoRAConfig(lora_ref.lora_path)
        self.configs[lora_ref.lora_id] = config

        # Load adapter weights to CPU
        self.load_lora_weights(lora_ref)

        # Store metadata
        self.lora_refs[lora_ref.lora_id] = lora_ref
        if lora_ref.pinned:
            self.num_pinned_loras += 1

        logger.info(
            "Loaded LoRA adapter: %s (id=%s, rank=%d, pinned=%s)",
            lora_ref.lora_name,
            lora_ref.lora_id,
            config.r,
            lora_ref.pinned,
        )

    def load_lora_weights(self, lora_ref: LoRARef):
        """
        Load LoRA weights from disk to CPU memory.

        V1 implementation: Creates LoRAAdapter and calls initialize_weights()
        to load weights from checkpoint files.

        Args:
            lora_ref: LoRARef object with lora_id and lora_path
        """
        from sgl_jax.srt.configs.load_config import LoadConfig

        # Get load config (TODO: get from server_args if available)
        load_config = LoadConfig()

        # Create LoRA backend (placeholder for v1)
        lora_backend = ChunkedSgmvLoRABackend()

        # Create adapter
        adapter = LoRAAdapter(
            uid=lora_ref.lora_id,
            config=self.configs[lora_ref.lora_id],
            base_hf_config=self.base_hf_config,
            load_config=load_config,
            lora_backend=lora_backend,
        )

        # Load weights from disk to CPU
        adapter.initialize_weights()

        # Store adapter
        self.loras[lora_ref.lora_id] = adapter

        logger.info(
            "Loaded weights for LoRA adapter: %s (%d layers)",
            lora_ref.lora_name,
            len(adapter.layers),
        )

    def can_support(self, config: LoRAConfig) -> bool:
        """Check if memory pool can support the given LoRA config."""
        return self.memory_pool.can_support(config)

    def prepare_lora_batch(self, model_worker_batch: ModelWorkerBatch):
        """
        Prepare LoRA batch for inference.

        V1 implementation: Transfers required adapter weights from CPU to device memory pool.
        All adapters are pre-loaded at initialization, no dynamic loading.

        Args:
            forward_batch: ForwardBatch containing requests with lora_ids

        Raises:
            ValueError: If batch exceeds max_loras_per_batch or adapter not loaded
        """
        # Load active loras into lora memory pool
        cur_uids = set(model_worker_batch.lora_ids)

        assert len(cur_uids) <= self.max_loras_per_batch

        # Load adapters into device memory pool (CPU -> device transfer)
        self.memory_pool.prepare_lora_batch(
            cur_uids=cur_uids,
            lora_adapters=self.loras,
        )

        weight_indices = [0] * len(model_worker_batch.lora_ids)
        lora_ranks = [0] * self.max_loras_per_batch
        scalings = [0] * self.max_loras_per_batch
        # print(f"{self.loras=}")
        for i, uid in enumerate(model_worker_batch.lora_ids):
            weight_indices[i] = self.memory_pool.get_buffer_id(uid)
            if uid is not None and uid in self.loras:
                # print(f"{uid=}")
                lora = self.loras[uid]
                lora_ranks[weight_indices[i]] = lora.config.r
                scalings[weight_indices[i]] = lora.scaling

        self.lora_backend.prepare_lora_batch(
            model_worker_batch=model_worker_batch,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
        )

        self.verify_sharding_preserved()

        logger.debug("Prepared LoRA batch: %d unique adapters", len(cur_uids))

    def get_buffer_id(self, lora_id: str | None) -> int:
        """Get buffer slot ID for a given LoRA adapter ID."""
        return self.memory_pool.get_buffer_id(lora_id)

    def apply_lora_surgery(self):
        """
        Apply Flax Model Surgery to add LoRA layers to the base model.

        This method uses Flax NNX's Model Surgery technique to dynamically
        replace Linear layers with LoRALinear wrappers without modifying
        the original model definition.

        Steps:
        1. Save original model state (including sharding information)
        2. Replace target Linear layers with LoRALinear wrappers
        3. Restore original weights via nnx.update() (preserves sharding)

        The surgery preserves:
        - Original model weights
        - Sharding specifications
        - Model structure compatibility with JIT compilation
        """
        from flax import nnx

        if self.base_model is None:
            logger.warning("No base_model provided, skipping LoRA surgery")
            return

        logger.info("Applying LoRA surgery to base model...")

        # Step 1: Save original state (with sharding!)
        original_state = nnx.state(self.base_model)

        # Step 2: Track replaced modules
        self.lora_modules: list[dict[str, BaseLayerWithLoRA]] = [
            {} for _ in range(self.base_hf_config.num_hidden_layers)
        ]

        # Step 3: Replace Linear layers with LoRALinear
        # We need to iterate through the model and find target modules
        # For now, use a simple approach: check common layer names
        try:
            # Try to access model.layers (common structure)
            if hasattr(self.base_model, "layers"):
                layers = self.base_model.layers
            elif hasattr(self.base_model, "model") and hasattr(self.base_model.model, "layers"):
                layers = self.base_model.model.layers
            else:
                logger.warning("Could not find model.layers, skipping surgery")
                return

            # Iterate through layers
            for layer_idx in range(len(layers)):
                layer = layers[layer_idx]

                # Check for attention layers
                if hasattr(layer, "self_attn"):
                    attn = layer.self_attn
                    for module_name in self.target_modules:
                        if hasattr(attn, module_name):
                            self._replace_with_lora(
                                attn,
                                module_name,
                                f"layers.{layer_idx}.self_attn.{module_name}",
                                layer_idx,
                            )

                # Check for MLP layers
                if hasattr(layer, "mlp"):
                    mlp = layer.mlp
                    for module_name in self.target_modules:
                        if hasattr(mlp, module_name):
                            self._replace_with_lora(
                                mlp,
                                module_name,
                                f"layers.{layer_idx}.mlp.{module_name}",
                                layer_idx,
                            )

        except Exception as e:
            logger.error("Error during LoRA surgery: %s", e)
            logger.warning("LoRA surgery failed, continuing without LoRA layers")
            return

        # Step 4: Restore original weights (preserves sharding)
        try:
            nnx.update(self.base_model, original_state)
            logger.info(
                "LoRA surgery completed: replaced %d modules",
                len(self.lora_modules),
            )
        except Exception as e:
            logger.error("Error restoring original state: %s", e)
            raise

    def _replace_with_lora(
        self,
        parent_module,
        attr_name: str,
        full_path: str,
        layer_idx: int,
    ):
        """
        Replace a Linear layer with LoRALinear wrapper.

        Args:
            parent_module: Parent module containing the layer
            attr_name: Attribute name of the layer (e.g., "q_proj")
            full_path: Full path for logging (e.g., "layers.0.self_attn.q_proj")
            layer_idx: layer index
        """

        from sgl_jax.srt.lora.layers import LoRALinear

        original_layer = getattr(parent_module, attr_name, None)
        if original_layer is None:
            return

        # Check if it's a Linear layer
        if not isinstance(original_layer, LinearBase):
            return

        # Get or create backend
        if not hasattr(self, "lora_backend"):
            from sgl_jax.srt.lora.backend.bgmv_backend import BgmvLoRABackend

            self.lora_backend = BgmvLoRABackend(
                max_loras_per_batch=self.max_loras_per_batch,
                max_lora_rank=self.max_lora_rank,
            )

        # Create LoRALinear wrapper with backend
        lora_layer = LoRALinear(
            base_layer=original_layer,
            lora_backend=self.lora_backend,
        )

        # Replace the layer
        setattr(parent_module, attr_name, lora_layer)

        # Track the replacement
        self.lora_modules[layer_idx][attr_name] = lora_layer

        logger.debug("Replaced %s with LoRALinear", full_path)

    def _get_nested_attr(self, obj, attr_path: str):
        """
        Get nested attribute using dot notation.

        Args:
            obj: Object to traverse
            attr_path: Dot-separated path (e.g., "layers.0.self_attn.q_proj")

        Returns:
            The nested attribute
        """
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj

    def _set_nested_attr(self, obj, attr_path: str, value):
        """
        Set nested attribute using dot notation.

        Args:
            obj: Object to traverse
            attr_path: Dot-separated path
            value: Value to set
        """
        parts = attr_path.split(".")
        for attr in parts[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, parts[-1], value)

    def verify_sharding_preserved(self):
        """
        Verify that model surgery preserved sharding information.

        Checks that base layer weights still have their original sharding specs.
        """
        if not hasattr(self, "lora_modules") or not self.lora_modules:
            logger.warning("No LoRA modules to verify")
            return

        for layer_idx, layer_modules in enumerate(self.lora_modules):
            for module_name, module in layer_modules.items():
                try:
                    # Check if base layer kernel has sharding
                    if hasattr(module.base_layer, "kernel"):
                        kernel = module.base_layer.kernel
                        if hasattr(kernel, "value"):
                            kernel_value = kernel.value
                            if hasattr(kernel_value, "sharding"):
                                sharding = kernel_value.sharding
                                logger.info(
                                    "%s base_layer.kernel sharding: %s",
                                    f"layer{layer_idx}.{module_name}",
                                    sharding,
                                )
                            else:
                                logger.warning(
                                    "%s base_layer.kernel has no sharding attribute",
                                    f"layer{layer_idx}.{module_name}",
                                )
                # TODO: add the check of sharding for lora_a and lora_b
                except Exception as e:
                    logger.warning(
                        "Error checking sharding for %s: %s",
                        f"layer{layer_idx}.{module_name}",
                        e,
                    )
