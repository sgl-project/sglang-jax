"""Resolve a model-specific host processor that wraps the HF AutoProcessor.

P1/2b design (design doc §4.3.5): some omni models need a composed host processor
(e.g. MiMo-V2.5 wraps Qwen2.5-VL vision + a host RVQ audio codec). This resolver lets
the generic tokenizer pick one by capability without importing any concrete model;
when none matches it returns the original HF processor unchanged (generic path).
"""

from __future__ import annotations

import importlib

# (module_path, class_name) candidates, imported lazily so the generic tokenizer never
# imports a specific model at module load. Each class exposes matches()/from_hf_processor().
_PROCESSOR_SPECS: list[tuple[str, str]] = [
    ("sgl_jax.srt.multimodal.models.mimo_v2_5.processor", "MiMoV25Processor"),
]


def resolve_host_processor(mm_config, model_path, hf_processor, *, trust_remote_code=True):
    """Return a wrapping host processor for this model, or ``hf_processor`` unchanged."""
    if mm_config is None:
        return hf_processor
    for module_path, class_name in _PROCESSOR_SPECS:
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue
        cls = getattr(module, class_name, None)
        if cls is None:
            continue
        if not cls.matches(mm_config):
            continue
        return cls.from_hf_processor(
            mm_config, model_path, hf_processor, trust_remote_code=trust_remote_code
        )
    return hf_processor
