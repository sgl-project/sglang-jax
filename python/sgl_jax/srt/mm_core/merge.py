"""Canonical multimodal embedding merge (refactor M2).

The single, model-agnostic, placement-agnostic merge: scatter per-item modality
embeddings into their placeholder rows in the text-embedding sequence. This one
implementation supersedes the as-is divergent copies (design doc §1.7 / §1.12 / §2.4):

  - MiMo-V2.5            `embedding.py:_scatter_modality`     (model-internal scatter)
  - Qwen3-Omni           `get_placeholder_mask` + `.at[].set` (model-internal scatter)
  - Qwen2.5-VL           `vit_model_runner._merge_*`          (runner cumsum-gather)
  - srt ScheduleBatch    `_merge_multimodal`                  (batch assembly consumer)

Three contract rules (must NOT drift into per-model code) — design doc §3.3.2:
  1. Single primitive = per-modality scatter-by-mask: for each modality, locate its placeholder
     rows with `input_ids == its placeholder token id`, write with `.at[idx].set(..., mode="drop")`.
     Per-modality (not one global `isin` mask over concat'd features) so an interleaved prompt
     `<image>…<audio>…<image>` can't cross-wire a placeholder to another modality's feature
     (review K-1). (The discarded cumsum-gather trick existed only to dodge a sharding issue that
     rule 3 removes.)
  2. Key by the raw modality placeholder token id (image/video/audio_token_id) — NOT a
     pad_value. Under Scheme B (design §5.1.2) input_ids stays clean (placeholder rows hold
     the real in-vocab token id), so merge keys directly on it; the per-image radix cache key
     is decoupled and lives in `Req.cache_input_ids`. (The clamp contract rule the review
     proposed is therefore unnecessary: forward input never holds an out-of-vocab id. It
     would only re-apply if one ever switched to Scheme A.)
  3. Full-replication before scatter: under a sharded embed mesh, constrain operands to
     `PartitionSpec()` or the scatter cannot resolve an output sharding (the real bug
     that forced `with_sharding_constraint` in the as-is code).

Pure function: output depends only on inputs; no model state, no mesh read beyond the
optional `mesh` arg, no token-id config. Unit-testable without a model or TPU.

NOTE: requires jax to run (uses jnp scatter). The repo targets Python >=3.12.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec


@dataclasses.dataclass
class FusedEmbed:
    """Output of :func:`merge`.

    `embed` is the per-token input embedding handed to the LLM (placeholder rows replaced by
    modality features). (Deepstack visual embeddings, for Qwen3-VL/Omni, are NOT densified here:
    under C-1 (design §5.2) the encoder returns the SPARSE per-level features + a visual mask,
    and ScheduleBatch._merge_multimodal densifies them per chunk on the host -- a single densify
    implementation, not duplicated in this device-side merge.)
    """

    embed: jax.Array


def merge(
    text_embed: jax.Array,
    mod_embeds: list[jax.Array],
    placeholder_ids: list[int],
    input_ids: jax.Array,
    *,
    mesh: jax.sharding.Mesh | None = None,
) -> FusedEmbed:
    """Scatter encoded modality features into their placeholder rows, PER MODALITY.

    Args:
      text_embed: ``[seq, hidden]`` text-token embeddings (already computed by the model).
      mod_embeds: per-MODALITY encoded features, each ``[N_i, hidden]``. 1:1 with
        ``placeholder_ids``: ``mod_embeds[i]`` are the features for the modality whose placeholder
        token id is ``placeholder_ids[i]`` (and only for modalities present in this request). Within
        one modality the rows must be in the order its placeholders appear in ``input_ids`` (the
        processor already concatenates per-item features in order).
      placeholder_ids: the raw modality placeholder token ids, 1:1 with ``mod_embeds``. Each
        modality's features are scattered into ONLY the rows where ``input_ids`` equals its own id.
      input_ids: ``[seq]`` int array; placeholder rows hold their modality token id.
      mesh: optional embed mesh; when set, operands are resharded to full replication
        (jax.sharding.reshard) before the scatter (contract rule 3).

    Returns:
      :class:`FusedEmbed` whose ``embed`` is ``[seq, hidden]`` with each modality's placeholder
      rows replaced by that modality's features in order. A per-modality count mismatch is safe:
      surplus placeholder slots / surplus feature rows map to an out-of-bounds index and
      ``mode="drop"`` discards them (no silent overwrite of row 0).

    Why per modality (review K-1): a single ``concat(mod_embeds)`` + global placeholder mask pairs
    the i-th sequence placeholder with the i-th *block-order* feature. Models build ``mod_embeds``
    in modality-block order (image block, then video, then audio), so an INTERLEAVED prompt
    (``<image>…<audio>…<image>``) would cross-wire the audio placeholder to an image feature. Keying
    each modality to its own token id makes the scatter order-independent across modalities.

    Deepstack (Qwen3-VL/Omni) is NOT handled here: the encoder returns the sparse per-level visual
    features + a visual placeholder mask, and ScheduleBatch._merge_multimodal densifies them per
    chunk on the host (C-1, design §5.2). Keeping the densify out of merge avoids a second,
    device-side densify implementation.
    """
    # No multimodal items -> pure-text passthrough.
    if not mod_embeds:
        return FusedEmbed(embed=text_embed)
    if len(mod_embeds) != len(placeholder_ids):
        raise ValueError(
            "merge expects 1:1 mod_embeds<->placeholder_ids (per-modality scatter, review K-1); "
            f"got {len(mod_embeds)} feature blocks and {len(placeholder_ids)} placeholder ids"
        )

    # Contract rule 3: full replication before scatter on a sharded embed mesh. Use
    # jax.sharding.reshard (NOT with_sharding_constraint): under the standard AR mesh whose
    # axes are all type Explicit, with_sharding_constraint acts as an *assert* on the input's
    # existing sharding rather than re-sharding it, so a 'data'-sharded text_embed (the real
    # in-model case) would fail the assert. reshard actually replicates and works under both
    # Auto and Explicit axis types. input_ids must be replicated too (it feeds the placeholder
    # mask; jnp.nonzero on a 'data'-sharded mask can't resolve the spec under this mesh).
    if mesh is not None:
        repl = NamedSharding(mesh, PartitionSpec())
        text_embed = jax.sharding.reshard(text_embed, repl)
        input_ids = jax.sharding.reshard(input_ids, repl)
        mod_embeds = [jax.sharding.reshard(f, repl) for f in mod_embeds]

    seq_len = text_embed.shape[0]
    fused = text_embed
    for feats, tok_id in zip(mod_embeds, placeholder_ids):
        # Contract rules 1+2: scatter THIS modality's features into ONLY its own placeholder rows
        # (input_ids == its token id), in sequence order. Static size = #features (a traced shape);
        # surplus/short slots map to OOB (seq_len) and are dropped by mode="drop", so a per-modality
        # count mismatch never overwrites a text row.
        mask = input_ids == tok_id  # [seq] bool, this modality only
        positions = jnp.nonzero(mask, size=feats.shape[0], fill_value=seq_len)[0]
        fused = fused.at[positions, :].set(feats, mode="drop")

    return FusedEmbed(embed=fused)
