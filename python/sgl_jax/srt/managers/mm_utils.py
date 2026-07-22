"""Common host orchestration for in-model multimodal embedding."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.multimodal.common.in_model_plan_builder import (
    resolve_in_model_plan_builder,
)
from sgl_jax.srt.multimodal.common.mm_plan import MultimodalEmbedPlan
from sgl_jax.srt.multimodal.common.modality_enum import MultimodalInputs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.managers.schedule_batch import ScheduleReqsInfo


def _has_in_model_multimodal_inputs(reqs_info: list[ScheduleReqsInfo] | None) -> bool:
    for info in reqs_info or []:
        for req in info.reqs or []:
            mm_inputs = req.mm_inputs
            if isinstance(mm_inputs, MultimodalInputs):
                if mm_inputs.mm_items:
                    return True
            elif mm_inputs is not None and not (
                isinstance(mm_inputs, dict) and "mm_items" not in mm_inputs
            ):
                return True
    return False


def build_mm_embed_plan(
    reqs_info: list[ScheduleReqsInfo] | None,
    dp_size: int,
    model_config: ModelConfig,
    per_dp_token: int,
    tp_size: int = 1,
    patch_buckets: Sequence[int] | None = None,
    merge_buckets: Sequence[int] | None = None,
) -> MultimodalEmbedPlan | None:
    """Build an multimodal encode/merge plan.

    ``patch_buckets`` pads encoder inputs; ``merge_buckets`` is retained as the
    configured encoder-output capacity list. Routing itself is padded to
    ``per_dp_token`` and has no independent shape bucket.
    """
    if not _has_in_model_multimodal_inputs(reqs_info):
        return None
    builder = resolve_in_model_plan_builder(
        model_config, patch_buckets=patch_buckets, merge_buckets=merge_buckets
    )
    if builder is None:
        return None
    return builder.build(reqs_info, dp_size, per_dp_token, tp_size)


def _merge_in_specs(encoder_tp: bool):
    """Merge in-specs. DP-Encoder shards lanes over ``"tensor"``; TP-Encoder
    replicates them (one collaborative lane per DP rank)."""
    tp = None if encoder_tp else "tensor"
    return (
        PartitionSpec("data", None),  # running   [total_tok, H]
        PartitionSpec("data", tp, None, None),  # features [dp,tp,source_capacity,H]
        PartitionSpec("data", tp, None),  # src_idx [dp,tp,per_dp_token]
        PartitionSpec("data", tp, None),  # mask [dp,tp,per_dp_token]
    )


@functools.partial(jax.jit, static_argnames=("mesh", "encoder_tp"))
def merge_jit(mesh, running, features, src_idx, mask, encoder_tp=False):
    """Gather per-lane encoder features directly into DP token embedding rows.

    Routing is destination-driven: its final axis is the local token axis, so
    there is no independent merge shape. DP-Encoder fans requests across tensor
    lanes and combines their token-aligned updates with ``psum``. TP-Encoder has
    one tensor-replicated lane per DP rank and needs no final combine.
    """

    def merge_local(
        running: jax.Array,
        features: jax.Array,
        src_idx: jax.Array,
        mask: jax.Array,
    ) -> jax.Array:
        lane_features = features[0, 0]
        lane_src = src_idx[0, 0]
        lane_mask = mask[0, 0]

        safe_src = jnp.where(lane_mask, lane_src, 0)
        updates = jnp.where(lane_mask[:, None], lane_features[safe_src], 0)
        modality_mask = lane_mask.astype(jnp.int32)
        if not encoder_tp:
            updates = jnp.asarray(jax.lax.psum(updates, "tensor"))
            modality_mask = jnp.asarray(jax.lax.psum(modality_mask, "tensor"))
        return jnp.where((modality_mask > 0)[:, None], updates, running)

    return jax.shard_map(
        merge_local,
        mesh=mesh,
        in_specs=_merge_in_specs(encoder_tp),
        out_specs=PartitionSpec("data", None),
        check_vma=False,
    )(running, features, src_idx, mask)


def _flatten_device_batch(value, *, dp_size: int, tp_size: int):
    return value.reshape(dp_size * tp_size, *value.shape[2:])


def _encode_inputs_lane_shape(encode_inputs) -> tuple[int, int]:
    leaves = jax.tree.leaves(encode_inputs)
    if not leaves:
        raise ValueError("Multimodal encode inputs must contain at least one array leaf.")
    if any(value.ndim < 2 for value in leaves):
        raise ValueError("Multimodal encode input leaves must have leading [dp,tp] axes.")
    lane_shape = tuple(leaves[0].shape[:2])
    if any(tuple(value.shape[:2]) != lane_shape for value in leaves[1:]):
        raise ValueError("Multimodal encode input leaves must share leading [dp,tp] axes.")
    return int(lane_shape[0]), int(lane_shape[1])


@functools.partial(jax.jit, static_argnames=("capacity",))
def _pad_features_jit(features, capacity: int):
    """Canonicalize encoder rows so downstream merge JITs do not depend on them."""
    rows = features.shape[2]
    if rows > capacity:
        raise ValueError(f"Encoder output rows ({rows}) exceed source capacity ({capacity}).")
    return jnp.pad(features, ((0, 0), (0, 0), (0, capacity - rows), (0, 0)))


def embed_mm_inputs(
    mm_embed_plan,
    input_ids,
    input_embedding,
    multimodal_model,
):
    """Encode each fixed-shape modality batch and merge it into token embeddings.

    ``running`` starts as the plain text embedding. Each modality batch encodes
    and merges directly. Returns ``[total_token, H]``.

    This first migration keeps one replicated encoder lane per DP rank. Encoder
    lane fan-out and weight tensor parallelism are added independently.
    """
    mesh = multimodal_model.mesh
    encoder_tp = True
    tp_axis = None if encoder_tp else "tensor"
    running = input_embedding(input_ids)
    for modality, batch in mm_embed_plan.items():
        embedder = getattr(multimodal_model, f"get_{modality.name.lower()}_feature", None)
        assert embedder is not None, f"no embedding method for {modality}"
        device_inputs = batch.encode_inputs
        dp_size, tp_size = _encode_inputs_lane_shape(device_inputs)
        flatten_device_batch = functools.partial(
            _flatten_device_batch,
            dp_size=dp_size,
            tp_size=tp_size,
        )

        # [dp,tp,...] lanes -> flat [dp*tp,...] batch for the ViT
        model_inputs = jax.tree.map(flatten_device_batch, device_inputs)
        features = embedder(model_inputs)
        feature_shape = (dp_size, tp_size, *features.shape[1:])
        features = jax.lax.reshape(
            features,
            feature_shape,
            out_sharding=NamedSharding(
                mesh,
                PartitionSpec("data", tp_axis, *([None] * (len(feature_shape) - 2))),
            ),
        )

        if batch.source_capacity is not None:
            features = _pad_features_jit(features, capacity=batch.source_capacity)
        running = merge_jit(
            mesh,
            running,
            features,
            batch.merge.src_idx,
            batch.merge.mask,
            encoder_tp=encoder_tp,
        )
    return running


def general_mm_embed_routine(
    input_ids,
    forward_batch,
    language_model,
    multimodal_model,
    mm_embed_plan,
):
    """Populate ``forward_batch.input_embedding`` for multimodal prefill.

    The language backbone consumes this fused embedding instead of re-embedding
    ``input_ids``.
    """
    embed_tokens = language_model.get_input_embeddings()
    input_embeds = embed_mm_inputs(
        mm_embed_plan,
        input_ids,
        embed_tokens,
        multimodal_model,
    )
    forward_batch.input_embedding = input_embeds
