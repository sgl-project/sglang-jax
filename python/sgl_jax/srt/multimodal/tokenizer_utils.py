# Relocated to sgl_jax.srt.utils.common_utils for srt<->multimodal decoupling (refactor M1).
# Kept as a re-export shim for back-compat with existing multimodal imports.
from sgl_jax.srt.utils.common_utils import resolve_tokenizer_subdir

__all__ = ["resolve_tokenizer_subdir"]
