"""Multimodal embed routine plumbing.

Provides the gate predicate that decides whether the model runner should
invoke the vision-encode / embedding-merge routine before the backbone jit,
and the routine itself: for each round in ``mm_embed_plan``, run the
per-arch vision encode across DP shards and merge the resulting features
into the running text embedding at the plan-specified mask positions.

The final merged embedding is written to ``forward_batch.input_embedding``
so the LM backbone consumes it in place of a fresh ``embed_tokens`` call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax

from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.multimodal.embed_encode import jitted_mm_encode
from sgl_jax.srt.multimodal.embed_merge import jitted_mm_merge
from sgl_jax.srt.multimodal.embed_plan import MultimodalEmbedPlan


def should_run_mm_embed_routine(model_config: Any, forward_batch: ForwardBatch) -> bool:
    if not getattr(model_config, "is_multimodal", False):
        return False
    if forward_batch.forward_mode != ForwardMode.EXTEND:
        return False
    return forward_batch.contains_mm_inputs()


def general_mm_embed_routine(
    *,
    input_ids,
    forward_batch: ForwardBatch,
    language_model: Any,
    vision_encode_fn: Callable[[Any, Any, Any], jax.Array],
    mm_embed_plan: MultimodalEmbedPlan | None,
    mesh: Any,
) -> None:
    """Encode-merge loop; writes the merged embedding to ``forward_batch``.

    ``vision_encode_fn(pixels_kd, valid_scalar, meta_no_dp) -> features_fh`` is
    the arch-specific per-rank encoder (e.g. a bound method on the vision
    tower). Callers must have already device-placed ``mm_embed_plan`` arrays
    on ``mesh`` via ``device_put_plan``.
    """
    if mm_embed_plan is None:
        raise ValueError("general_mm_embed_routine requires a non-None mm_embed_plan")

    running = language_model.embed_tokens(input_ids)
    for round_ in mm_embed_plan.image_rounds:
        features = jitted_mm_encode(vision_encode_fn, round_.encode_inputs, mesh)
        running = jitted_mm_merge(
            running=running,
            features=features,
            src_idx=round_.src_idx,
            mask=round_.mask,
            mesh=mesh,
        )
    forward_batch.input_embedding = running
