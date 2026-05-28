import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


@dataclass(frozen=True)
class LoRABatchPlan:
    """CPU-side LoRA assignment plan for one scheduler batch.

    weight_indices is request-granular and points at device-memory LoRA slots.
    ranks_by_slot and scalings_by_slot are slot-granular metadata. Keeping these
    together makes the indexing contract explicit before a backend expands it to
    token-granular arrays.
    """

    weight_indices: tuple[int, ...]
    ranks_by_slot: tuple[int, ...]
    scalings_by_slot: tuple[float, ...]

    def __init__(
        self,
        weight_indices: Sequence[int],
        ranks_by_slot: Sequence[int],
        scalings_by_slot: Sequence[float],
    ):
        object.__setattr__(self, "weight_indices", tuple(int(idx) for idx in weight_indices))
        object.__setattr__(self, "ranks_by_slot", tuple(int(rank) for rank in ranks_by_slot))
        object.__setattr__(
            self,
            "scalings_by_slot",
            tuple(float(scaling) for scaling in scalings_by_slot),
        )
        self._validate()

    def _validate(self):
        if len(self.ranks_by_slot) != len(self.scalings_by_slot):
            raise ValueError(
                "LoRA rank and scaling metadata must have the same slot count: "
                f"{len(self.ranks_by_slot)} != {len(self.scalings_by_slot)}"
            )

        num_slots = len(self.ranks_by_slot)
        for request_idx, slot in enumerate(self.weight_indices):
            if slot < 0 or slot >= num_slots:
                raise ValueError(
                    f"LoRA request {request_idx} references slot {slot}, "
                    f"but only {num_slots} slots are available"
                )

        for slot, rank in enumerate(self.ranks_by_slot):
            if rank < 0:
                raise ValueError(f"LoRA slot {slot} has negative rank {rank}")

        for slot, scaling in enumerate(self.scalings_by_slot):
            if not math.isfinite(scaling):
                raise ValueError(f"LoRA slot {slot} has non-finite scaling {scaling}")
            if scaling < 0:
                raise ValueError(f"LoRA slot {slot} has negative scaling {scaling}")

    @classmethod
    def for_static_lora(
        cls,
        batch_size: int,
        num_lora_slots: int,
        rank: int,
        scaling: float,
    ) -> "LoRABatchPlan":
        return cls(
            weight_indices=(0,) * batch_size,
            ranks_by_slot=(rank,) * num_lora_slots,
            scalings_by_slot=(scaling,) * num_lora_slots,
        )

    def ranks_for_requests(self) -> tuple[int, ...]:
        return tuple(self.ranks_by_slot[slot] for slot in self.weight_indices)

    def scalings_for_requests(self) -> tuple[float, ...]:
        return tuple(self.scalings_by_slot[slot] for slot in self.weight_indices)


class LoRAType(Enum):
    LORA_A = 0
    LORA_B = 1


def get_target_module_name(full_module_name: str, target_modules: set[str]) -> str:
    """
    Get the target module name in target_modules that can match full_module_name.

    If there is a target module name in target_modules that can match full_module_name, return this name
    Else raise ValueError.
    """
    ordered_target_modules = sorted(target_modules, key=lambda name: (-len(name), name))
    for target_module in ordered_target_modules:
        if full_module_name == target_module or full_module_name.endswith(f".{target_module}"):
            return target_module

    for target_module in ordered_target_modules:
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
