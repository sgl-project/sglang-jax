"""Top-level registration entry for the in-model Qwen2.5-VL.

``import_model_classes`` (registry.py) scans ``sgl_jax.srt.models`` one level deep
and only picks up non-package modules with an ``EntryClass``. The actual model
lives in the ``qwen2_5VL`` subpackage (not scanned), so this thin module re-exports
its entry class to register it via the standard package scan.
"""

from sgl_jax.srt.models.qwen2_5VL.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

EntryClass = [Qwen2_5_VLForConditionalGeneration]
