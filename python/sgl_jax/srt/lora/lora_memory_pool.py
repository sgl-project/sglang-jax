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
"""LoRA memory pool implementation for JAX."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.lora.utils import get_lora_a_sharding, get_lora_b_sharding

if TYPE_CHECKING:
    from sgl_jax.srt.lora.lora import LoRAAdapter
    from sgl_jax.srt.lora.lora_config import LoRAConfig

logger = logging.getLogger(__name__)


class EmptySlot:
    """
    Singleton class to represent an empty slot in the memory pool.
    This improves readability by not using special str as a placeholder.
    """

    __slots__ = ()

    def __repr__(self):
        return "|EMPTY|"

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance


EMPTY_SLOT = EmptySlot()


@register_pytree_node_class
class LoRAMemoryPool:
    """
    JAX-based memory pool for LoRA adapters.

    Unlike PyTorch version, this uses functional updates and pytree registration
    for JAX jit compatibility. No eviction policy is implemented - uses simple
    incremental buffer allocation.

    Key differences from PyTorch version:
    - Pytree-compatible for JAX jit
    - Functional updates with .at[].set() instead of in-place mutations
    - Sharding specs for distributed inference
    - Simple buffer_id allocation without eviction
    - CPU-side tracking (uid mappings) separate from JAX arrays

    Attributes:
        max_loras_per_batch: Maximum number of LoRA adapters per batch
        max_lora_rank: Maximum LoRA rank supported
        num_layers: Number of transformer layers
        target_modules: Set of target module names (e.g., {"qkv_proj", "o_proj"})
        mesh: JAX device mesh for sharding
        dtype: Data type for LoRA weights
        A_buffer: Dict[module_name, List[jax.Array]] - A matrices per layer
        B_buffer: Dict[module_name, List[jax.Array]] - B matrices per layer
        uid_to_buffer_id: Mapping from lora_id to buffer slot (CPU-side)
        buffer_id_to_uid: Mapping from buffer slot to lora_id (CPU-side)
    """

    def __init__(
        self,
        max_loras_per_batch: int,
        max_lora_rank: int,
        num_layers: int,
        target_modules: set[str],
        mesh: Mesh,
        dtype: jnp.dtype = jnp.float16,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_attention_heads: int = 32,
        num_kv_heads: int = 32,
        head_dim: int | None = None,
        original_num_kv_heads: int | None = None,
        tp_size: int = 1,
    ):
        """
        Initialize LoRA memory pool.

        Args:
            max_loras_per_batch: Maximum number of LoRA adapters in a batch
            max_lora_rank: Maximum LoRA rank to support
            num_layers: Number of transformer layers
            target_modules: Set of target module names
            mesh: JAX device mesh for sharding
            dtype: Data type for LoRA weights
            hidden_size: Model hidden dimension
            intermediate_size: FFN intermediate dimension
            num_attention_heads: Number of attention heads
            num_kv_heads: Number of KV heads AFTER replication (for GQA)
            head_dim: Head dimension (optional)
            original_num_kv_heads: Original number of KV heads BEFORE replication (optional)
            tp_size: Tensor parallel size (for calculating replication)
        """
        self.max_loras_per_batch = max_loras_per_batch
        self.max_lora_rank = max_lora_rank
        self.num_layers = num_layers
        self.target_modules = target_modules
        self.mesh = mesh
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim is not None else (hidden_size // num_attention_heads)

        # Store original count and calculate replication info
        self.original_num_kv_heads = (
            original_num_kv_heads if original_num_kv_heads is not None else num_kv_heads
        )
        self.tp_size = tp_size

        # Calculate if replication is needed and how many replicas
        if tp_size > self.original_num_kv_heads:
            self.needs_kv_replication = True
            self.num_kv_replicas = (
                tp_size + self.original_num_kv_heads - 1
            ) // self.original_num_kv_heads
        else:
            self.needs_kv_replication = False
            self.num_kv_replicas = 1

        # CPU-side tracking (not in pytree)
        # These are mutable Python objects used for bookkeeping
        self.uid_to_buffer_id: dict[str | None, int] = {}
        self.buffer_id_to_uid: list[str | None | EmptySlot] = [EMPTY_SLOT] * max_loras_per_batch

        # GPU buffers (in pytree) - initialized in init_buffers()
        self.A_buffer: dict[str, list[jax.Array]] = {}
        self.B_buffer: dict[str, list[jax.Array]] = {}

    def tree_flatten(self):
        """Flatten for pytree registration - only JAX arrays are children."""
        # Flatten A_buffer and B_buffer into lists
        a_buffer_flat = []
        b_buffer_flat = []
        module_names = sorted(self.A_buffer.keys())

        for module_name in module_names:
            a_buffer_flat.extend(self.A_buffer[module_name])
            b_buffer_flat.extend(self.B_buffer[module_name])

        children = (a_buffer_flat, b_buffer_flat)
        aux_data = {
            "max_loras_per_batch": self.max_loras_per_batch,
            "max_lora_rank": self.max_lora_rank,
            "num_layers": self.num_layers,
            "target_modules": self.target_modules,
            "mesh": self.mesh,
            "dtype": self.dtype,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "original_num_kv_heads": self.original_num_kv_heads,
            "tp_size": self.tp_size,
            "needs_kv_replication": self.needs_kv_replication,
            "num_kv_replicas": self.num_kv_replicas,
            "module_names": module_names,
            "uid_to_buffer_id": self.uid_to_buffer_id,
            "buffer_id_to_uid": self.buffer_id_to_uid,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten from pytree."""
        obj = object.__new__(cls)

        # Restore attributes
        obj.max_loras_per_batch = aux_data["max_loras_per_batch"]
        obj.max_lora_rank = aux_data["max_lora_rank"]
        obj.num_layers = aux_data["num_layers"]
        obj.target_modules = aux_data["target_modules"]
        obj.mesh = aux_data["mesh"]
        obj.dtype = aux_data["dtype"]
        obj.hidden_size = aux_data["hidden_size"]
        obj.intermediate_size = aux_data["intermediate_size"]
        obj.num_attention_heads = aux_data["num_attention_heads"]
        obj.num_kv_heads = aux_data["num_kv_heads"]
        obj.head_dim = aux_data.get("head_dim", obj.hidden_size // obj.num_attention_heads)
        obj.original_num_kv_heads = aux_data.get("original_num_kv_heads", aux_data["num_kv_heads"])
        obj.tp_size = aux_data.get("tp_size", 1)
        obj.needs_kv_replication = aux_data.get("needs_kv_replication", False)
        obj.num_kv_replicas = aux_data.get("num_kv_replicas", 1)
        obj.uid_to_buffer_id = aux_data["uid_to_buffer_id"]
        obj.buffer_id_to_uid = aux_data["buffer_id_to_uid"]

        # Reconstruct A_buffer and B_buffer
        a_buffer_flat, b_buffer_flat = children
        module_names = aux_data["module_names"]

        obj.A_buffer = {}
        obj.B_buffer = {}

        a_idx = 0
        b_idx = 0
        for module_name in module_names:
            obj.A_buffer[module_name] = a_buffer_flat[a_idx : a_idx + obj.num_layers]
            obj.B_buffer[module_name] = b_buffer_flat[b_idx : b_idx + obj.num_layers]
            a_idx += obj.num_layers
            b_idx += obj.num_layers

        return obj

    def can_support(self, config: LoRAConfig) -> bool:
        """Check if the memory pool can support the given LoRA config."""
        if config.r > self.max_lora_rank:
            return False
        # Check if target modules are supported
        config_modules = set(config.target_modules)
        return config_modules.issubset(self.target_modules)

    def _get_lora_a_shape(self, module_name: str) -> tuple[int, int, int]:
        """
        Get shape for LoRA A matrix.

        Returns: (max_loras_per_batch, max_lora_rank, input_dim)

        Note: Shape is global in JAX. Sharding is controlled by sharding spec,
        not by dividing dimensions in the shape.
        """
        if module_name == "qkv_proj" or module_name in ["q_proj", "k_proj", "v_proj"]:
            # Input: hidden_size (column-parallel, input not sharded)
            input_dim = self.hidden_size
        elif module_name == "o_proj":
            # Input: num_attention_heads * head_dim (row-parallel, input sharded by TP)
            input_dim = self.num_attention_heads * self.head_dim
        elif module_name == "gate_up_proj":
            # Input: hidden_size (column-parallel, input not sharded)
            input_dim = self.hidden_size
        elif module_name == "down_proj":
            # Input: intermediate_size (row-parallel, input sharded by TP)
            input_dim = self.intermediate_size
        else:
            # Default: hidden_size
            input_dim = self.hidden_size

        return (self.max_loras_per_batch, self.max_lora_rank, input_dim)

    def _get_lora_b_shape(self, module_name: str) -> tuple[int, int, int]:
        """
        Get shape for LoRA B matrix.

        Returns: (max_loras_per_batch, output_dim, max_lora_rank)

        Note: Shape is global in JAX. Sharding is controlled by sharding spec,
        not by dividing dimensions in the shape.
        """
        if module_name == "qkv_proj":
            # Output: (num_heads * head_dim) for Q, K, V combined
            # Q heads + KV heads (column-parallel, output sharded by TP)
            output_dim = (self.num_attention_heads + 2 * self.num_kv_heads) * self.head_dim
        elif module_name == "q_proj":
            # Output: num_attention_heads * head_dim (column-parallel, output sharded by TP)
            output_dim = self.num_attention_heads * self.head_dim
        elif module_name in ["k_proj", "v_proj"]:
            # Output: num_kv_heads * head_dim (column-parallel, output sharded by TP)
            output_dim = self.num_kv_heads * self.head_dim
        elif module_name == "o_proj":
            # Output: hidden_size (row-parallel, output not sharded)
            output_dim = self.hidden_size
        elif module_name == "gate_up_proj":
            # Output: intermediate_size * 2 (column-parallel, output sharded by TP)
            output_dim = self.intermediate_size * 2
        elif module_name == "down_proj":
            # Output: hidden_size (row-parallel, output not sharded)
            output_dim = self.hidden_size
        elif module_name in ["gate_proj", "up_proj"]:
            # Output: intermediate_size (column-parallel, output sharded by TP)
            output_dim = self.intermediate_size
        else:
            # Default: hidden_size
            output_dim = self.hidden_size

        return (self.max_loras_per_batch, output_dim, self.max_lora_rank)

    def _get_lora_a_sharding(self, module_name: str) -> NamedSharding:
        return get_lora_a_sharding(module_name, self.mesh)

    def _get_lora_b_sharding(self, module_name: str) -> NamedSharding:
        return get_lora_b_sharding(module_name, self.mesh)

    def init_buffers(self):
        """
        Initialize GPU buffers for LoRA weights.

        Creates A_buffer and B_buffer with proper sharding.
        """
        logger.info(
            "Initializing LoRA memory pool buffers: num_layers=%d, max_loras_per_batch=%d, max_lora_rank=%d, dtype=%s",
            self.num_layers,
            self.max_loras_per_batch,
            self.max_lora_rank,
            self.dtype,
        )
        logger.info(
            "Target modules to initialize: %s",
            sorted(self.target_modules),
        )

        with self.mesh:
            for module_name in self.target_modules:
                a_shape = self._get_lora_a_shape(module_name)
                b_shape = self._get_lora_b_shape(module_name)
                a_sharding = self._get_lora_a_sharding(module_name)
                b_sharding = self._get_lora_b_sharding(module_name)

                self.A_buffer[module_name] = []
                self.B_buffer[module_name] = []

                for _ in range(self.num_layers):
                    # Create sharded A buffer
                    a_buf = jax.jit(
                        lambda shape=a_shape, dt=self.dtype: jnp.zeros(shape, dtype=dt),
                        out_shardings=a_sharding,
                    )()
                    self.A_buffer[module_name].append(a_buf)

                    # Create sharded B buffer
                    b_buf = jax.jit(
                        lambda shape=b_shape, dt=self.dtype: jnp.zeros(shape, dtype=dt),
                        out_shardings=b_sharding,
                    )()
                    self.B_buffer[module_name].append(b_buf)

                logger.info(
                    "Created LoRA buffers for %s: A_shape=%s (sharding=%s), B_shape=%s (sharding=%s)",
                    module_name,
                    a_shape,
                    a_sharding,
                    b_shape,
                    b_sharding,
                )

        logger.info("LoRA memory pool initialization complete")

    def prepare_lora_batch(
        self,
        cur_uids: set[str | None],
        lora_adapters: dict[str | None, LoRAAdapter],
    ) -> bool:
        """
        Prepare LoRA batch by loading adapters into buffer slots.

        Simplified version without eviction policy - uses incremental allocation.

        Args:
            cur_uids: Set of lora_ids needed for current batch
            lora_adapters: Dict mapping lora_id to LoRAAdapter

        Returns:
            bool: True if new weights were loaded (requires updating references), False otherwise.

        Raises:
            ValueError: If no buffer slots available
        """

        def get_available_buffer_slot() -> int:
            """Find next available buffer slot (simple incremental allocation)."""
            for buffer_id in range(self.max_loras_per_batch):
                if self.buffer_id_to_uid[buffer_id] == EMPTY_SLOT:
                    return buffer_id

            raise ValueError(
                "No available buffer slots. Max %d LoRA adapters per batch exceeded.",
                self.max_loras_per_batch,
            )

        has_new_weights = False

        # Load each adapter that's not already loaded
        for uid in cur_uids:
            if uid not in self.uid_to_buffer_id:
                buffer_id = get_available_buffer_slot()
                lora_adapter = lora_adapters.get(uid)
                self.load_lora_weight_to_buffer(uid, buffer_id, lora_adapter)
                self.uid_to_buffer_id[uid] = buffer_id
                self.buffer_id_to_uid[buffer_id] = uid
                has_new_weights = True
                logger.info("Loaded LoRA %s into buffer slot %d", uid, buffer_id)
            else:
                logger.debug("LoRA %s already in buffer slot %d", uid, self.uid_to_buffer_id[uid])

        return has_new_weights

    def load_lora_weight_to_buffer(
        self,
        uid: str | None,
        buffer_id: int,
        lora_adapter: LoRAAdapter | None,
    ):
        """
        Load LoRA weights into buffer slot.

        Args:
            uid: LoRA adapter ID (None for base model)
            buffer_id: Buffer slot index
            lora_adapter: LoRA adapter object (None for base model)
        """

        def get_ab_zero_matrix_shape(module_name: str):
            _, max_lora_rank, input_dim = self._get_lora_a_shape(module_name)
            a_zero_matrix_shape = (max_lora_rank, input_dim)
            _, output_dim, max_lora_rank = self._get_lora_b_shape(module_name)
            b_zero_matrix_shape = (output_dim, max_lora_rank)
            # jax.device_put(jnp.zeros(a_zero_matrix_shape,dtype=self.dtype), self._get_lora_a_sharding)
            return a_zero_matrix_shape, b_zero_matrix_shape

        with jax.set_mesh(self.mesh):
            if uid is None:
                # Base model: zero out the buffer slot
                logger.debug("Loading base model (zeros) into buffer slot %d", buffer_id)
                for module_name in self.target_modules:
                    for layer_id in range(self.num_layers):
                        a_shape, b_shape = get_ab_zero_matrix_shape(module_name)
                        # Zero out A buffer
                        self.A_buffer[module_name][layer_id] = (
                            self.A_buffer[module_name][layer_id]
                            .at[buffer_id]
                            .set(jnp.zeros(a_shape, dtype=self.dtype))
                        )
                        # Zero out B buffer
                        self.B_buffer[module_name][layer_id] = (
                            self.B_buffer[module_name][layer_id]
                            .at[buffer_id]
                            .set(jnp.zeros(b_shape, dtype=self.dtype))
                        )
                return

            if lora_adapter is None:
                logger.warning("LoRA adapter %s is None, loading zeros", uid)
                # Treat as base model if adapter is None
                for module_name in self.target_modules:
                    a_shape, b_shape = get_ab_zero_matrix_shape(module_name)
                    for layer_id in range(self.num_layers):
                        self.A_buffer[module_name][layer_id] = (
                            self.A_buffer[module_name][layer_id]
                            .at[buffer_id]
                            .set(jnp.zeros(a_shape, dtype=self.dtype))
                        )
                        self.B_buffer[module_name][layer_id] = (
                            self.B_buffer[module_name][layer_id]
                            .at[buffer_id]
                            .set(jnp.zeros(b_shape, dtype=self.dtype))
                        )
                return

            logger.info(
                "Loading LoRA adapter %s into buffer slot %d (num_layers=%d, target_modules=%s)",
                uid,
                buffer_id,
                self.num_layers,
                sorted(self.target_modules),
            )

            # Track loaded weights for debugging
            loaded_modules_count = {module: 0 for module in self.target_modules}

            # Process each layer
            for layer_id in range(self.num_layers):
                layer_weights = lora_adapter.layers[layer_id].weights

                # Process each target module
                for module_name in self.target_modules:
                    a_shape, b_shape = get_ab_zero_matrix_shape(module_name)
                    # Extract and load weights for this module
                    lora_a, lora_b = self._extract_module_weights(
                        layer_weights, layer_id, module_name
                    )

                    if lora_a is not None and lora_b is not None:
                        # Track successful load
                        loaded_modules_count[module_name] += 1

                        # Handle rank padding/slicing
                        lora_a = self._handle_rank_mismatch(lora_a, is_lora_a=True)
                        lora_b = self._handle_rank_mismatch(lora_b, is_lora_a=False)

                        # Handle KV head replication for k_proj/v_proj
                        lora_a = self._replicate_kv_lora_weight(
                            lora_a, is_lora_a=True, module_name=module_name
                        )
                        lora_b = self._replicate_kv_lora_weight(
                            lora_b, is_lora_a=False, module_name=module_name
                        )

                        # Load into buffer
                        self.A_buffer[module_name][layer_id] = (
                            self.A_buffer[module_name][layer_id].at[buffer_id].set(lora_a)
                        )
                        self.B_buffer[module_name][layer_id] = (
                            self.B_buffer[module_name][layer_id].at[buffer_id].set(lora_b)
                        )

                        if layer_id == 0:  # Log details for first layer only
                            logger.debug(
                                "Loaded %s layer %d: lora_a.shape=%s, lora_b.shape=%s",
                                module_name,
                                layer_id,
                                lora_a.shape,
                                lora_b.shape,
                            )
                    else:
                        # Module not found in adapter weights, zero out
                        logger.debug(
                            "Module %s not found in layer %d weights, zeroing buffer",
                            module_name,
                            layer_id,
                        )
                        self.A_buffer[module_name][layer_id] = (
                            self.A_buffer[module_name][layer_id]
                            .at[buffer_id]
                            .set(jnp.zeros(a_shape, dtype=self.dtype))
                        )
                        self.B_buffer[module_name][layer_id] = (
                            self.B_buffer[module_name][layer_id]
                            .at[buffer_id]
                            .set(jnp.zeros(b_shape, dtype=self.dtype))
                        )

            # Log summary of loaded weights
            logger.info(
                "Completed loading LoRA adapter %s: loaded_modules=%s (count per module across all layers)",
                uid,
                loaded_modules_count,
            )

    def _extract_module_weights(
        self,
        layer_weights: dict[str, jax.Array],
        layer_id: int,
        module_name: str,
    ) -> tuple[jax.Array | None, jax.Array | None]:
        """
        Extract LoRA A and B weights for a specific module from layer weights.

        Args:
            layer_weights: Dictionary of weight tensors for this layer
            layer_id: Layer index
            module_name: Target module name (qkv_proj, o_proj, gate_up_proj, down_proj)

        Returns:
            Tuple of (lora_a, lora_b) tensors, or (None, None) if not found

        Weight naming convention in LoRA adapters:
            base_model.model.layers.{layer_id}.{module_path}.lora_A.weight
            base_model.model.layers.{layer_id}.{module_path}.lora_B.weight

        Module name mapping:
            - qkv_proj: Concatenate q_proj, k_proj, v_proj
            - gate_up_proj: Concatenate gate_proj, up_proj
            - o_proj: Direct mapping to o_proj
            - down_proj: Direct mapping to down_proj
        """
        # Handle composite modules (qkv_proj, gate_up_proj)
        if module_name == "qkv_proj":
            # Need to concatenate q_proj, k_proj, v_proj
            return self._extract_and_concat_qkv(layer_weights, layer_id)
        elif module_name == "gate_up_proj":
            # Need to concatenate gate_proj, up_proj
            return self._extract_and_concat_gate_up(layer_weights, layer_id)
        else:
            # Direct mapping (o_proj, down_proj)
            return self._extract_single_module(layer_weights, layer_id, module_name)

    def _extract_single_module(
        self,
        layer_weights: dict[str, jax.Array],
        layer_id: int,
        module_name: str,
    ) -> tuple[jax.Array | None, jax.Array | None]:
        """Extract weights for a single module (o_proj, down_proj)."""
        lora_a = None
        lora_b = None

        # Search for matching weight keys
        for key, weight in layer_weights.items():
            # Match pattern: layers.{layer_id}.{path}.{module_name}.lora_{A|B}.weight
            if f"layers.{layer_id}." in key and module_name in key:
                if "lora_A.weight" in key:
                    lora_a = weight
                elif "lora_B.weight" in key:
                    lora_b = weight

        return lora_a, lora_b

    def _extract_and_concat_qkv(
        self,
        layer_weights: dict[str, jax.Array],
        layer_id: int,
    ) -> tuple[jax.Array | None, jax.Array | None]:
        """
        Extract and concatenate q_proj, k_proj, v_proj weights.

        For attention QKV projection, we need to concatenate:
        - lora_A: Concatenate along rank dimension (axis 0)
        - lora_B: Concatenate along output dimension (axis 0)

        NOTE: When KV head replication is needed, k_proj and v_proj weights
        are replicated BEFORE concatenation.

        Returns:
            Concatenated (lora_a_qkv, lora_b_qkv)
        """
        # Extract individual components
        q_a, q_b = self._extract_single_module(layer_weights, layer_id, "q_proj")
        k_a, k_b = self._extract_single_module(layer_weights, layer_id, "k_proj")
        v_a, v_b = self._extract_single_module(layer_weights, layer_id, "v_proj")

        # Check if all components are present
        if all(x is not None for x in [q_a, k_a, v_a, q_b, k_b, v_b]):
            # Replicate k_proj and v_proj if needed (before concatenation)
            k_a = self._replicate_kv_lora_weight(k_a, is_lora_a=True, module_name="k_proj")
            k_b = self._replicate_kv_lora_weight(k_b, is_lora_a=False, module_name="k_proj")
            v_a = self._replicate_kv_lora_weight(v_a, is_lora_a=True, module_name="v_proj")
            v_b = self._replicate_kv_lora_weight(v_b, is_lora_a=False, module_name="v_proj")

            # Concatenate A matrices along rank dimension (axis 0)
            # Shape: (rank, hidden_size) -> (3*rank, hidden_size) or similar
            # Note: k_a and v_a won't change with replication (input is hidden_size)
            lora_a_qkv = jnp.concatenate([q_a, k_a, v_a], axis=0)

            # Concatenate B matrices along output dimension (axis 0)
            # Shape: (head_dim, rank) -> (total_output_dim, rank)
            # After replication, k_b and v_b have more output dims
            lora_b_qkv = jnp.concatenate([q_b, k_b, v_b], axis=0)

            return lora_a_qkv, lora_b_qkv
        else:
            # Not all components found
            logger.warning(
                "Incomplete QKV weights in layer %d: q_proj=%s, k_proj=%s, v_proj=%s",
                layer_id,
                q_a is not None,
                k_a is not None,
                v_a is not None,
            )
            return None, None

    def _extract_and_concat_gate_up(
        self,
        layer_weights: dict[str, jax.Array],
        layer_id: int,
    ) -> tuple[jax.Array | None, jax.Array | None]:
        """
        Extract and concatenate gate_proj, up_proj weights.

        For FFN gate/up projection, we need to concatenate:
        - lora_A: Concatenate along rank dimension (axis 0)
        - lora_B: Concatenate along output dimension (axis 0)

        Returns:
            Concatenated (lora_a_gate_up, lora_b_gate_up)
        """
        # Extract individual components
        gate_a, gate_b = self._extract_single_module(layer_weights, layer_id, "gate_proj")
        up_a, up_b = self._extract_single_module(layer_weights, layer_id, "up_proj")

        # Check if both components are present
        if all(x is not None for x in [gate_a, up_a, gate_b, up_b]):
            # Concatenate A matrices along rank dimension (axis 0)
            lora_a_gate_up = jnp.concatenate([gate_a, up_a], axis=0)

            # Concatenate B matrices along output dimension (axis 0)
            lora_b_gate_up = jnp.concatenate([gate_b, up_b], axis=0)

            return lora_a_gate_up, lora_b_gate_up
        else:
            # Not all components found
            logger.warning(
                "Incomplete gate_up weights in layer %d: gate_proj=%s, up_proj=%s",
                layer_id,
                gate_a is not None,
                up_a is not None,
            )
            return None, None

    def _handle_rank_mismatch(
        self,
        weight: jax.Array,
        is_lora_a: bool,
    ) -> jax.Array:
        """
        Handle rank mismatch between adapter and buffer.

        For lora_A: shape is (rank, input_dim)
        For lora_B: shape is (output_dim, rank)

        If adapter rank < max_lora_rank: Pad with zeros
        If adapter rank > max_lora_rank: Slice (shouldn't happen with proper config)

        Args:
            weight: LoRA weight tensor
            is_lora_a: True for A matrix, False for B matrix

        Returns:
            Adjusted weight tensor
        """
        if is_lora_a:
            # lora_A shape: (rank, input_dim)
            current_rank = weight.shape[0]
            if current_rank < self.max_lora_rank:
                # Pad along rank dimension (axis 0)
                pad_size = self.max_lora_rank - current_rank
                weight = jnp.pad(
                    weight,
                    ((0, pad_size), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
            elif current_rank > self.max_lora_rank:
                # Slice to max_lora_rank (shouldn't happen normally)
                logger.warning(
                    "LoRA rank %d exceeds max_lora_rank %d, slicing",
                    current_rank,
                    self.max_lora_rank,
                )
                weight = weight[: self.max_lora_rank, :]
        else:
            # lora_B shape: (output_dim, rank)
            current_rank = weight.shape[1]
            if current_rank < self.max_lora_rank:
                # Pad along rank dimension (axis 1)
                pad_size = self.max_lora_rank - current_rank
                weight = jnp.pad(
                    weight,
                    ((0, 0), (0, pad_size)),
                    mode="constant",
                    constant_values=0,
                )
            elif current_rank > self.max_lora_rank:
                # Slice to max_lora_rank
                logger.warning(
                    "LoRA rank %d exceeds max_lora_rank %d, slicing",
                    current_rank,
                    self.max_lora_rank,
                )
                weight = weight[:, : self.max_lora_rank]

        return weight

    def _replicate_kv_lora_weight(
        self,
        weight: jax.Array,
        is_lora_a: bool,
        module_name: str,
    ) -> jax.Array:
        """
        Replicate LoRA weights for k_proj/v_proj when KV head replication is needed.

        This ensures LoRA weights match the base model's KV head replication pattern.
        When tp_size > original_num_kv_heads, each original KV head is replicated
        num_kv_replicas times.

        Args:
            weight: LoRA weight tensor to replicate
            is_lora_a: True for A matrix, False for B matrix
            module_name: Module name (should be k_proj or v_proj)

        Returns:
            Replicated weight tensor

        Example:
            If original_num_kv_heads=4, tp_size=8, num_kv_replicas=2:
            - Original lora_B shape: (output_dim, rank) where output_dim = 4 * head_dim
            - After replication: (output_dim, rank) where output_dim = 8 * head_dim
            - Each of the 4 original heads is replicated 2 times
        """
        # Only replicate if needed and for k_proj/v_proj
        if not self.needs_kv_replication:
            return weight

        if module_name not in ["k_proj", "v_proj"]:
            return weight

        if is_lora_a:
            # lora_A shape: (rank, input_dim)
            # Input is hidden_size, doesn't need replication
            return weight
        else:
            # lora_B shape: (output_dim, rank)
            # output_dim = original_num_kv_heads * head_dim
            # Need to replicate to: tp_size * head_dim (or num_kv_heads * head_dim)

            output_dim, rank = weight.shape

            # Validate expected shape
            expected_output_dim = self.original_num_kv_heads * self.head_dim
            if output_dim != expected_output_dim:
                logger.warning(
                    "Unexpected lora_B output_dim for %s: got %d, expected %d (original_kv_heads=%d * head_dim=%d)",
                    module_name,
                    output_dim,
                    expected_output_dim,
                    self.original_num_kv_heads,
                    self.head_dim,
                )
                # Continue anyway - weight might be from a different config

            # Replicate each head: split by head, then repeat each head num_kv_replicas times
            # Reshape to (original_num_kv_heads, head_dim, rank)
            weight_per_head = jnp.reshape(weight, (self.original_num_kv_heads, self.head_dim, rank))

            # Replicate each head
            replicated_parts = []
            for head_idx in range(self.original_num_kv_heads):
                head_weight = weight_per_head[head_idx]  # (head_dim, rank)
                # Repeat this head num_kv_replicas times
                for _ in range(self.num_kv_replicas):
                    replicated_parts.append(head_weight)

            # Concatenate all replicated heads
            # Shape: (original_num_kv_heads * num_kv_replicas, head_dim, rank)
            replicated_weight = jnp.stack(replicated_parts, axis=0)

            # Reshape back to (output_dim, rank)
            final_output_dim = len(replicated_parts) * self.head_dim
            replicated_weight = jnp.reshape(replicated_weight, (final_output_dim, rank))

            logger.debug(
                "Replicated %s lora_B: original shape=%s, replicated shape=%s (original_kv_heads=%d, replicas=%d)",
                module_name,
                weight.shape,
                replicated_weight.shape,
                self.original_num_kv_heads,
                self.num_kv_replicas,
            )

            return replicated_weight

    def get_buffer_id(self, lora_uid: str | None) -> int:
        """Get buffer slot ID for a given LoRA adapter ID."""
        return self.uid_to_buffer_id[lora_uid]

    def get_tensor(self, module_name: str, layer_id: int, is_lora_a: bool) -> jax.Array:
        """
        Get LoRA tensor for a specific module and layer.

        Args:
            module_name: Target module name (e.g., "qkv_proj")
            layer_id: Layer index
            is_lora_a: True for A matrix, False for B matrix

        Returns:
            JAX array with shape:
            - A: (max_loras_per_batch, max_lora_rank, input_dim)
            - B: (max_loras_per_batch, output_dim, max_lora_rank)
        """
        if is_lora_a:
            return self.A_buffer[module_name][layer_id]
        else:
            return self.B_buffer[module_name][layer_id]
