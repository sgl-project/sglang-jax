"""Shared LoRA sentinel values.

The BGMV kernels treat adapter indices as ordinary array indices.  The
serving stack reserves slot 0 for base-model requests by keeping its LoRA
weights zeroed; that policy belongs to the LoRA manager/memory-pool layer.
"""

BASE_LORA_ID = "0"
BASE_LORA_SLOT = 0


def is_base_lora_id(lora_id: str | None) -> bool:
    return lora_id is None or lora_id == BASE_LORA_ID


def normalize_lora_id(lora_id: str | None) -> str:
    return BASE_LORA_ID if lora_id is None else lora_id
