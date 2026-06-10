"""Neutral multimodal CORE (refactor M2).

srt-visible layer shared by the understanding plane (in-model, reuses the standard srt
LLM control plane) and — later — the generation plane. It must NOT depend on the staged
`sgl_jax.srt.multimodal` runtime package, so the dependency graph stays a DAG:
    understanding-model (srt/models) -> mm_core -> (jax only)
    generation-plane               -> mm_core
See design doc §3.2 / §3.7.
"""

from sgl_jax.srt.mm_core.merge import FusedEmbed, merge

__all__ = ["merge", "FusedEmbed"]
