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
