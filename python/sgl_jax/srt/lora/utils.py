from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


@dataclass
class LoRABatchInfo:
    # scaling of each lora adapter, in shape (num_tokens,)
    scalings: jax.Array

    # (num_tokens,)
    token_lora_indices: jax.Array

    # (num_tokens,)
    lora_ranks: jax.Array


class LoRAType(Enum):
    LORA_A = 0
    LORA_B = 1


def get_target_module_name(full_module_name: str, target_modules: set[str]) -> str:
    """
    Get the target module name in target_modules that can match full_module_name.

    If there is a target module name in target_modules that can match full_module_name, return this name
    Else raise ValueError.
    """
    for target_module in target_modules:
        if target_module in full_module_name:
            return target_module
    raise ValueError(f"Cannot find target module name for {full_module_name} in {target_modules}")


def get_lora_a_sharding(module_name: str, mesh: Mesh) -> NamedSharding:
    """Get sharding spec for LoRA A matrix."""
    # Row-parallel layers: shard input dimension
    if module_name in {"o_proj", "down_proj"}:
        # Shape: (batch, rank, input_dim)
        # Shard input_dim across tensor axis
        return NamedSharding(mesh, P(None, None, "tensor"))
    else:
        # Column-parallel: no sharding for A
        return NamedSharding(mesh, P(None, None, None))


def get_lora_b_sharding(module_name: str, mesh: Mesh) -> NamedSharding:
    """Get sharding spec for LoRA B matrix."""
    # Column-parallel layers: shard output dimension
    if module_name in {
        "qkv_proj",
        "gate_up_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
    }:
        # Shape: (batch, output_dim, rank)
        # Shard output_dim across tensor axis
        return NamedSharding(mesh, P(None, "tensor", None))
    else:
        # Row-parallel: no sharding for B
        return NamedSharding(mesh, P(None, None, None))


def get_lora_a_output_sharding(module_name: str, mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P(None, None))


def get_lora_b_output_sharding(module_name: str, mesh: Mesh) -> NamedSharding:
    """Get sharding spec for LoRA B matrix."""
    if module_name in {
        "qkv_proj",
        "gate_up_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
    }:
        # Shape: (num_tokens, input_dim)
        # Shard input_dim across tensor axis
        return NamedSharding(mesh, P(None, "tensor"))
    else:
        return NamedSharding(mesh, P(None, None))


def get_normalized_target_modules(
    target_modules: Iterable[str],
) -> set[str]:
    """
    Mapping a list of target module name to names of the normalized LoRA weights.
    Handles both base module names (e.g., "gate_proj") and prefixed module names (e.g., "feed_forward.gate_proj").
    """
    params_mapping = {
        "q_proj": "q_proj",
        "k_proj": "k_proj",
        "v_proj": "v_proj",
        "o_proj": "o_proj",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
    }

    result = set()
    for name in target_modules:
        base_name = name.split(".")[-1]
        normalized_name = params_mapping.get(base_name, base_name)
        result.add(normalized_name)
    return result
