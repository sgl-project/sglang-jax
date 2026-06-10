"""Multimodal input processors (refactor M3, design §3.5.1).

Processor subclasses of mm_core.BaseMultimodalProcessor declare the HF architectures they
serve via ``models = [...]`` and self-register (arch-keyed) into the mm_core
ProcessorRegistry on package scan (mm_core.import_processor_classes). The standard
TokenizerManager resolves the processor for understanding requests by hf_config.architectures.
"""
