"""Host orchestration for in-model multimodal embedding.

Vision uses owning-rank data-parallel: each DP rank encodes its own images and
merges into its own token slice on the ``data`` mesh axis. This module provides
the token merge and host loop that call model-specific encoders before the
language backbone forward.
"""

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec

from sgl_jax.srt.multimodal.common.mm_plan import (
    EmbedRound,
    MultimodalEmbedPlan,
    VisionEncodeInputs,
)
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.common.vision_metadata import VisionMetadataBuilderProtocol

_MERGE_IN_SPECS = (
    PartitionSpec("data", None),  # running   [total_tok, H]
    PartitionSpec("data", None, None),  # features  [dp, out_rows, H]
    PartitionSpec("data"),  # src_idx   [total_tok]
    PartitionSpec("data"),  # mask      [total_tok]
)


@dataclasses.dataclass(frozen=True)
class ReqEncodeUnit:
    """One request's vision payload owned by one DP rank for one encode round."""

    images: list[MultimodalDataItem]
    req_base: int


def _collect_image_requests(
    reqs_info: list[Any] | None,
    dp_size: int,
) -> list[list[ReqEncodeUnit]]:
    """Collect vision-bearing requests as per-rank encode units."""
    per_rank_units: list[list[ReqEncodeUnit]] = [[] for _ in range(dp_size)]
    for dp_rank in range(dp_size):
        if not reqs_info or dp_rank >= len(reqs_info):
            continue
        info = reqs_info[dp_rank]
        if not info.reqs:
            continue
        req_base = 0
        for req in info.reqs:
            if req.mm_inputs is None:
                mm_items = []
            elif isinstance(req.mm_inputs, MultimodalInputs):
                mm_items = req.mm_inputs.mm_items
            elif isinstance(req.mm_inputs, dict) and "mm_items" not in req.mm_inputs:
                mm_items = []
            else:
                raise TypeError(
                    "build_mm_embed_plan expects req.mm_inputs to be MultimodalInputs, "
                    f"got {type(req.mm_inputs).__name__}."
                )
            image_items = []
            for item in mm_items:
                if not isinstance(item, MultimodalDataItem):
                    raise TypeError(
                        "build_mm_embed_plan expects mm_items to contain "
                        f"MultimodalDataItem, got {type(item).__name__}."
                    )
                if item.is_image() or item.is_video():
                    image_items.append(item)
            if image_items:
                per_rank_units[dp_rank].append(ReqEncodeUnit(images=image_items, req_base=req_base))
            req_base += int(getattr(req, "extend_input_len", 0) or 0)
    return per_rank_units


def _build_embed_round(
    round_units: list[ReqEncodeUnit | None],
    builder: VisionMetadataBuilderProtocol,
    dp_size: int,
    per_dp_token: int,
) -> EmbedRound:
    """Build one DP encode round from a single image order source.

    ``unit.images`` is the only ordering source for this round. The raw patch
    concat order, request-metadata pack order, and rank-local ``src_idx`` rows
    must all be derived from this same traversal; splitting those traversals can
    silently scatter the wrong vision row into a placeholder token.
    """
    total_token = dp_size * per_dp_token
    src_idx = np.zeros(total_token, dtype=np.int32)
    mask = np.zeros(total_token, dtype=np.bool_)
    patch_features_by_rank = [None] * dp_size
    metas = [None] * dp_size

    real_lane = False
    for dp_rank in range(dp_size):
        unit = round_units[dp_rank] if dp_rank < len(round_units) else None
        if unit is None:
            continue
        real_lane = True
        rank_base = dp_rank * per_dp_token
        merge_row = 0
        patch_features = []
        for item in unit.images:
            feat = item.feature
            if feat is None:
                raise ValueError(f"IMAGE item in dp_rank {dp_rank} is missing feature.")
            feat = np.asarray(feat)
            if feat.ndim != 2 or feat.shape[0] <= 0:
                raise ValueError(
                    "IMAGE item feature must be a non-empty 2D patch array, "
                    f"got shape={feat.shape} in dp_rank {dp_rank}."
                )

            patch_features.append(feat)
            for start, end in item.offsets or []:
                for o in range(int(start), int(end) + 1):
                    tok = rank_base + unit.req_base + o
                    if tok < rank_base or tok >= rank_base + per_dp_token:
                        raise ValueError(
                            "IMAGE placeholder offset is outside its packed rank slot: "
                            f"dp_rank={dp_rank}, req_base={unit.req_base}, offset={o}, "
                            f"per_dp_token={per_dp_token}."
                        )
                    if mask[tok]:
                        raise ValueError(
                            "IMAGE placeholder token is assigned more than once: "
                            f"dp_rank={dp_rank}, token={tok}, offset={o}."
                        )
                    src_idx[tok] = merge_row
                    mask[tok] = True
                    merge_row += 1

        actual_rows = int(mask[rank_base : rank_base + per_dp_token].sum())
        if actual_rows != merge_row:
            raise ValueError(
                "IMAGE placeholder mask rows must match emitted merge rows: "
                f"dp_rank={dp_rank}, mask rows={actual_rows}, expected={merge_row}."
            )
        patch_cat = np.concatenate(patch_features, axis=0)
        patch_features_by_rank[dp_rank] = patch_cat
        metas[dp_rank] = builder.get_metadata(unit.images)

    if not real_lane:
        raise ValueError("Multimodal embed round requires at least one real image lane.")

    present = [f for f in patch_features_by_rank if f is not None]
    patch_k = max(int(f.shape[0]) for f in present)
    patch_dim = int(present[0].shape[1])
    pixels = np.zeros((dp_size, patch_k, patch_dim), dtype=np.float32)
    valid = np.zeros((dp_size,), dtype=np.int32)
    for dp_rank, feat in enumerate(patch_features_by_rank):
        if feat is None:
            continue
        rows = int(feat.shape[0])
        pixels[dp_rank, :rows, :] = feat
        valid[dp_rank] = rows

    meta = builder.stack_metadata(metas, patch_k)
    return EmbedRound(
        encode_inputs=VisionEncodeInputs(pixels=pixels, valid=valid, meta=meta),
        src_idx=src_idx,
        mask=mask,
    )


def build_mm_embed_plan(
    reqs_info: list[Any] | None,
    dp_size: int,
    model_config: Any,
    per_dp_token: int,
) -> MultimodalEmbedPlan | None:
    """Build the owning-rank DP multimodal embed plan (host-side, numpy).

    Round-loop: process one request's images per rank per round. ``n_rounds`` =
    max over ranks of (#image-bearing requests that rank owns). Ranks with fewer
    requests use a dummy lane that contributes nothing (grid/pixels zero, mask
    all False).

    Returns ``None`` for batches with no image items.
    """
    per_rank_units = _collect_image_requests(reqs_info, dp_size)
    n_rounds = max((len(units) for units in per_rank_units), default=0)
    if n_rounds == 0:
        return None

    # Resolve the per-arch metadata builder only after confirming this batch has
    # image items, so non-image multimodal batches do not require a vision builder.
    from sgl_jax.srt.multimodal.common.vision_metadata import (
        resolve_vision_metadata_builder,
    )

    builder = resolve_vision_metadata_builder(model_config)

    rounds = []
    for round_idx in range(n_rounds):
        round_units = [
            per_rank_units[rank][round_idx] if round_idx < len(per_rank_units[rank]) else None
            for rank in range(dp_size)
        ]
        rounds.append(_build_embed_round(round_units, builder, dp_size, per_dp_token))

    return MultimodalEmbedPlan(rounds_by_modality={Modality.IMAGE: rounds})


@functools.partial(jax.jit, static_argnames=("mesh",))
def merge_jit(mesh, running, features, src_idx, mask):
    """Merge encoded multimodal rows into the token embedding stream.

    ``running`` is ``P("data", None)``; ``features`` is
    ``P("data", None, None)``. ``src_idx``/``mask`` are ``P("data")``. Returns
    ``P("data", None)``.
    """

    def merge_local(running, features, src_idx, mask):
        rank_features = features[0]
        return jnp.where(mask[:, None], rank_features[src_idx], running)

    return jax.shard_map(
        merge_local,
        mesh=mesh,
        in_specs=_MERGE_IN_SPECS,
        out_specs=PartitionSpec("data", None),
        check_vma=False,
    )(running, features, src_idx, mask)


def embed_mm_inputs(
    mm_embed_plan,
    input_ids,
    input_embedding,
    multimodal_model,
):
    """Encode each multimodal round and merge it into token embeddings.

    ``running`` starts as the plain text embedding (``embed_tokens`` once); each
    round encodes the image-bearing requests assigned to one DP rank via the
    model's ``get_{modality}_feature`` embedder, which consumes scheduler-built
    ``enc.meta``. Returns the merged embedding ``[total_token, H]``.
    """
    mesh = multimodal_model.mesh
    running = input_embedding(input_ids)
    for modality, rounds in mm_embed_plan.rounds_by_modality.items():
        embedder = getattr(multimodal_model, f"get_{modality.name.lower()}_feature", None)
        assert embedder is not None, f"no embedding method for {modality}"
        for rnd in rounds:
            features = embedder(rnd.encode_inputs)
            running = merge_jit(mesh, running, features, rnd.src_idx, rnd.mask)
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
