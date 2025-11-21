from dataclasses import dataclass
from enum import Enum

import jax


@dataclass
class LoRABatchInfo:
    # Batch size
    bs: int

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
