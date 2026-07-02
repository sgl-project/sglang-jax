"""Host orchestration for in-model multimodal embedding.

Vision uses owning-rank data-parallel: each DP rank encodes its own images and
merges into its own token slice on the ``data`` mesh axis. This module provides
the token merge and host loop that call model-specific encoders before the
language backbone forward.
"""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

_MERGE_IN_SPECS = (
    PartitionSpec("data", None),  # running   [total_tok, H]
    PartitionSpec("data", None, None),  # features  [dp, out_rows, H]
    PartitionSpec("data"),  # src_idx   [total_tok]
    PartitionSpec("data"),  # mask      [total_tok]
)


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
    round encodes one image per rank via the model's ``get_{modality}_feature``
    embedder, which consumes scheduler-built ``enc.meta``. Returns the merged
    embedding ``[total_token, H]``.
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
