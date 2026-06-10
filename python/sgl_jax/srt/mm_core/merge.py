"""Canonical multimodal embedding merge (refactor M2).

The single, model-agnostic, placement-agnostic merge: scatter per-item modality
embeddings into their placeholder rows in the text-embedding sequence. This one
implementation supersedes the as-is divergent copies (design doc §1.7 / §1.12 / §2.4):

  - MiMo-V2.5            `embedding.py:_scatter_modality`     (model-internal scatter)
  - Qwen3-Omni           `get_placeholder_mask` + `.at[].set` (model-internal scatter)
  - Qwen2.5-VL           `vit_model_runner._merge_*`          (runner cumsum-gather)
  - srt ScheduleBatch    `_merge_multimodal`                  (batch assembly consumer)

Three contract rules (must NOT drift into per-model code) — design doc §3.3.2:
  1. Single primitive = scatter-by-mask: locate placeholder rows with
     `jnp.isin(input_ids, pad_values)`, write with `.at[idx].set(..., mode="drop")`.
     (The discarded cumsum-gather trick existed only to dodge a sharding issue that
     rule 3 removes.)
  2. Key by per-item `pad_value` (NOT raw token id), matching the RadixAttention cache
     key so per-image cache reuse works.
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

    `embed` is the per-token input embedding handed to the LLM (placeholder rows
    replaced by modality features). `deepstack_*` is an optional side-channel for
    models with deepstack visual embeddings (Qwen3-VL/Omni); None otherwise.
    """

    embed: jax.Array
    deepstack_embed: jax.Array | None = None
    deepstack_pos_mask: jax.Array | None = None


def merge(
    text_embed: jax.Array,
    mod_embeds: list[jax.Array],
    pad_values: list[int],
    input_ids: jax.Array,
    deepstack: jax.Array | None = None,
    *,
    mesh: jax.sharding.Mesh | None = None,
) -> FusedEmbed:
    """Scatter encoded modality features into their placeholder rows.

    Args:
      text_embed: ``[seq, hidden]`` text-token embeddings (already computed by the model).
      mod_embeds: per-item encoded features, each ``[N_i, hidden]``. Must be ordered the
        same as their placeholder rows appear in ``input_ids`` (sequence order); they are
        concatenated and assigned to placeholder rows in that order.
      pad_values: per-item pad_value ints; rows where ``input_ids == pad_values[i]`` are
        item i's placeholders. Used to build the placeholder membership mask.
      input_ids: ``[seq]`` int array; placeholder rows hold their item's pad_value.
      deepstack: optional precomputed deepstack side-channel, carried through to the result.
      mesh: optional embed mesh; when set, operands are resharded to full replication
        (jax.sharding.reshard) before the scatter (contract rule 3).

    Returns:
      :class:`FusedEmbed` whose ``embed`` is ``[seq, hidden]`` with placeholder rows
      replaced by ``concat(mod_embeds)`` in order. A count mismatch is safe: surplus
      placeholder slots map to an out-of-bounds index and ``mode="drop"`` discards them
      (no silent overwrite of row 0).
    """
    # No multimodal items -> pure-text passthrough.
    if not mod_embeds:
        return FusedEmbed(embed=text_embed, deepstack_embed=deepstack)

    all_features = jnp.concatenate(mod_embeds, axis=0)  # [sum_i N_i, hidden], item order
    placeholder = jnp.asarray(list(pad_values), dtype=input_ids.dtype)

    # Contract rule 3: full replication before scatter on a sharded embed mesh. Use
    # jax.sharding.reshard (NOT with_sharding_constraint): under the standard AR mesh whose
    # axes are all type Explicit, with_sharding_constraint acts as an *assert* on the input's
    # existing sharding rather than re-sharding it, so a 'data'-sharded text_embed (the real
    # in-model case) would fail the assert. reshard actually replicates and works under both
    # Auto and Explicit axis types.
    if mesh is not None:
        repl = NamedSharding(mesh, PartitionSpec())
        text_embed = jax.sharding.reshard(text_embed, repl)
        all_features = jax.sharding.reshard(all_features, repl)
        # input_ids must be replicated too: it feeds the placeholder mask, and jnp.nonzero
        # on a 'data'-sharded mask fails (its bincount/scatter internals can't resolve the
        # 'data' spec under this mesh). Replicating here keeps mask/positions/scatter aligned.
        input_ids = jax.sharding.reshard(input_ids, repl)

    seq_len = text_embed.shape[0]
    # Contract rules 1+2: placeholder mask by pad_value membership, ordered positions.
    mask = jnp.isin(input_ids, placeholder)  # [seq] bool
    # Static size = #features (a traced shape). Surplus/short slots map to OOB (seq_len)
    # and are dropped by mode="drop" -> a count mismatch never overwrites a text row.
    positions = jnp.nonzero(mask, size=all_features.shape[0], fill_value=seq_len)[0]
    fused = text_embed.at[positions, :].set(all_features, mode="drop")
    return FusedEmbed(embed=fused, deepstack_embed=deepstack)
