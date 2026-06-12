"""Top-level registration entry for the in-model MiMo-V2.5.

``import_model_classes`` (registry.py) scans ``sgl_jax.srt.models`` one level deep
and only picks up non-package modules with an ``EntryClass``. The actual model
lives in the ``mimo_v2_5`` subpackage (not scanned), so this thin module
re-exports its entry class to register it via the standard package scan.
"""

from sgl_jax.srt.models.mimo_v2_5.mimo_v2_5_inmodel import (
    MiMoV2_5ForConditionalGeneration,
)

EntryClass = [MiMoV2_5ForConditionalGeneration]
