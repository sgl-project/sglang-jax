from __future__ import annotations

import dataclasses
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.multimodal.common.modality_enum import (
    MultimodalDataItem,
    MultimodalInputs,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class VisionEncodeInputs:
    pixels: Any  # [dp, patch_k, patch_dim]
    valid: Any  # [dp], int32; real patch rows before padding
    meta: Any | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class EmbedRound:
    encode_inputs: VisionEncodeInputs
    src_idx: Any  # [total_token], int32
    mask: Any  # [total_token], bool


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MultimodalEmbedPlan:
    image_rounds: list[EmbedRound]


def _collect_image_items(reqs_info: Any, dp_size: int) -> list[list[MultimodalDataItem]]:
    items_by_rank: list[list[MultimodalDataItem]] = [[] for _ in range(dp_size)]
    for dp_rank in range(dp_size):
        for req in reqs_info[dp_rank].reqs or []:
            mm_inputs: MultimodalInputs | None = getattr(req, "mm_inputs", None)
            if mm_inputs is None:
                continue
            for item in mm_inputs.mm_items or []:
                if item.is_image():
                    items_by_rank[dp_rank].append(item)
    return items_by_rank


def build_mm_embed_plan(
    reqs_info: Any,
    dp_size: int,
    builder: Any,
    per_dp_token: int,
    total_token: int,
) -> MultimodalEmbedPlan | None:
    """Build the host-side image forward plan.

    ``builder`` is a per-arch ``VisionMetadataBuilderProtocol`` instance. It
    owns arch-specific geometry (``get_metadata``) and cross-rank pad-by-role
    stacking (``stack_metadata``). Returns ``None`` if there are no image items
    across any DP rank.
    """
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    if total_token < dp_size * per_dp_token:
        raise ValueError(
            f"total_token {total_token} smaller than dp_size*per_dp_token "
            f"{dp_size * per_dp_token}"
        )

    items_by_rank = _collect_image_items(reqs_info, dp_size)
    n_rounds = max((len(x) for x in items_by_rank), default=0)
    if n_rounds == 0:
        return None

    rounds: list[EmbedRound] = []
    for r in range(n_rounds):
        round_items: list[MultimodalDataItem | None] = [
            items_by_rank[rank][r] if r < len(items_by_rank[rank]) else None
            for rank in range(dp_size)
        ]

        # Determine patch bucket for this round.
        max_patches = 0
        patch_dim: int | None = None
        feature_dtype = np.float32
        for it in round_items:
            if it is None:
                continue
            feat = np.asarray(it.feature)
            if feat.ndim != 2:
                raise ValueError(
                    f"image feature must be 2D [num_patches, patch_dim], got shape {feat.shape}"
                )
            if feat.shape[0] > max_patches:
                max_patches = feat.shape[0]
            if patch_dim is None:
                patch_dim = feat.shape[1]
                feature_dtype = feat.dtype
            elif feat.shape[1] != patch_dim:
                raise ValueError(
                    f"inconsistent patch_dim across ranks in round {r}: "
                    f"{patch_dim} vs {feat.shape[1]}"
                )

        assert patch_dim is not None  # n_rounds > 0 guarantees >=1 real item

        pixels = np.zeros((dp_size, max_patches, patch_dim), dtype=feature_dtype)
        valid = np.zeros((dp_size,), dtype=np.int32)
        mask = np.zeros((total_token,), dtype=bool)
        src_idx = np.zeros((total_token,), dtype=np.int32)
        metas: list[Any] = [None] * dp_size

        for rank, it in enumerate(round_items):
            if it is None:
                continue
            meta = builder.get_metadata(it)
            metas[rank] = meta

            feat = np.asarray(it.feature)
            n_patch = feat.shape[0]
            pixels[rank, :n_patch, :] = feat.astype(feature_dtype, copy=False)
            valid[rank] = n_patch

            feature_rows = int(np.asarray(meta.window_index).shape[0])

            offsets = it.offsets
            if not offsets:
                raise ValueError(f"image item on rank {rank} round {r} has empty offsets")
            if len(offsets) != 1:
                raise ValueError(
                    f"image item on rank {rank} round {r} has {len(offsets)} "
                    "offset spans; exactly one is required (see module docstring)"
                )
            start, end = offsets[0]
            if start < 0 or start > end:
                raise ValueError(
                    f"invalid offset on rank {rank} round {r}: "
                    f"({start},{end}); require 0 <= start <= end"
                )
            if end >= per_dp_token:
                raise ValueError(
                    f"offset ({start},{end}) on rank {rank} escapes "
                    f"per_dp_token slice of length {per_dp_token}"
                )
            span_len = end - start + 1
            if span_len != feature_rows:
                raise ValueError(
                    f"placeholder span length mismatch on rank {rank} round "
                    f"{r}: offsets=({start},{end}) span_len={span_len} but "
                    f"feature_rows={feature_rows} (num_patches={n_patch})"
                )

            base = rank * per_dp_token
            g0 = base + start
            g1 = base + end
            mask[g0 : g1 + 1] = True
            src_idx[g0 : g1 + 1] = np.arange(feature_rows, dtype=np.int32)

        stacked_meta = builder.stack_metadata(metas, patch_k=max_patches)

        rounds.append(
            EmbedRound(
                encode_inputs=VisionEncodeInputs(
                    pixels=pixels,
                    valid=valid,
                    meta=stacked_meta,
                ),
                src_idx=src_idx,
                mask=mask,
            )
        )

    return MultimodalEmbedPlan(image_rounds=rounds)


def device_put_plan(
    plan: MultimodalEmbedPlan | None,
    mesh: Mesh,
) -> MultimodalEmbedPlan | None:
    """Move every array leaf of ``plan`` to ``NamedSharding(mesh, P("data"))``.

    Returns ``None`` unchanged so callers can pipe optional plans directly.
    Non-array meta leaves are left untouched.
    """
    if plan is None:
        return None
    sharding = NamedSharding(mesh, PartitionSpec("data"))

    def _put(x):
        if isinstance(x, (np.ndarray, jax.Array)):
            return jax.device_put(x, sharding)
        return x

    return jax.tree_util.tree_map(_put, plan)
