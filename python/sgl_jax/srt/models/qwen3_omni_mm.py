"""Top-level registration entry for the in-model Qwen3-Omni MoE.

``import_model_classes`` (registry.py) scans ``sgl_jax.srt.models`` one level deep
and only picks up non-package modules with an ``EntryClass``. The actual model
lives in the ``qwen3_omni_moe`` subpackage (not scanned), so this thin module
re-exports its entry class to register it via the standard package scan.
"""

from sgl_jax.srt.models.qwen3_omni_moe.qwen3_omni_inmodel import (
    Qwen3OmniMoeForConditionalGeneration,
)

EntryClass = [Qwen3OmniMoeForConditionalGeneration]
