"""In-model multimodal embed routine: the host orchestration that strings the
three independently-compiled JIT segments per forward.

Target forward (extend only) for the in-model VLM path::

    JIT(1) vision encode (shard_map)
    JIT(2) multimodal merge (shard_map)
    JIT(3) language backbone  (existing, unchanged)

Vision uses owning-rank data-parallel: each DP rank encodes its own images and
merges into its own tokens with ZERO cross-rank collective. The ``data`` mesh
axis == dp size; everything vision flows as ``P("data")``.

This module owns JIT(1) and JIT(2) plus the host loop that drives them
(``general_mm_embed_routine`` -> ``embed_mm_inputs``). ``mm_encode``/``mm_merge``
are jit-wrapped kernel entries (``mesh`` as a static arg) with the shard_map
inlined -- the ``@jax.jit`` handles caching, matching the repo convention
(``kernels/fused_moe``, ``layers/attention/.../paged_attention``).
"""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

_MERGE_IN_SPECS = (
    PartitionSpec("data", None),  # running   [total_tok, H]
    PartitionSpec("data", None),  # features  [dp*out_rows, H]
    PartitionSpec("data"),  # src_idx   [total_tok]
    PartitionSpec("data"),  # mask      [total_tok]
)


def _merge_local(running, features, src_idx, mask):
    """Per shard (one DP rank's slice): ``where(mask, features[src_idx], running)``.

    No cumsum, no clip, no host sync -- ``src_idx``/``mask`` are precomputed by
    the scheduler.
    """
    return jnp.where(mask[:, None], features[src_idx], running)


@functools.partial(jax.jit, static_argnames=("mesh",))
def mm_merge(mesh, running, features, src_idx, mask):
    """JIT(2): src_idx merge, ``P("data")`` flat layout.

    The shard_map is inlined and ``@jax.jit`` (``mesh`` static) caches it --
    matching the repo's jit-wrapped kernel-entry convention (``kernels/fused_moe``,
    ``layers/attention/.../paged_attention``); no separate cached builder.
    ``running``/``features`` are ``P("data", None)``; ``src_idx``/``mask`` are
    ``P("data")``. Returns ``P("data", None)``.
    """
    return jax.shard_map(
        _merge_local,
        mesh=mesh,
        in_specs=_MERGE_IN_SPECS,
        out_specs=PartitionSpec("data", None),
        check_vma=False,
    )(running, features, src_idx, mask)


@functools.partial(jax.jit, static_argnames=("mesh", "body"))
def mm_encode(mesh, body, pixels, aux, valid):
    """JIT(1): single-image ViT ``body`` wrapped in a shard_map.

    ``aux`` is the 4-tuple already computed host-side by the model's
    ``get_image_feature`` (Design X: aux 在 embedder 内算). The shard_map is
    inlined and ``@jax.jit`` (``mesh``/``body`` static) caches it -- matching the
    repo's jit-wrapped kernel-entry convention; no separate cached builder.

    Flat leaf order: ``pixels``, ``*aux``, ``valid`` (``grid`` is host-only, NOT a
    shard_map operand). The leading ``dp`` axis is fully sharded on ``data`` and
    shard_map strips it, so each shard's body sees one rank's single image and
    returns ``[out_rows, H]``; ``out_specs`` assembles them into
    ``[dp*out_rows, H]``. Returns ``P("data", None)``.
    """
    aux = tuple(aux) if aux is not None else ()
    leaves = (pixels, *aux, valid)
    n_aux = len(aux)
    # Every leaf: P("data", *[None]*(ndim-1)) -- dp axis sharded, rest replicated.
    in_specs = tuple(
        PartitionSpec("data", *([None] * (getattr(leaf, "ndim", 1) - 1))) for leaf in leaves
    )

    def encode(enc_leaves):
        a = tuple(enc_leaves[1 : 1 + n_aux]) if n_aux else None
        return body(enc_leaves[0], a, enc_leaves[-1])

    return jax.shard_map(
        encode,
        mesh=mesh,
        in_specs=(in_specs,),
        out_specs=PartitionSpec("data", None),
        check_vma=False,
    )(leaves)


def embed_mm_inputs(
    mm_embed_plan,
    input_ids,
    input_embedding,
    multimodal_model,
    data_embedding_func_mapping=None,
):
    """Per-modality embed + merge (= upstream ``embed_mm_inputs``, mm_utils.py:782).

    ``running`` starts as the plain text embedding (``embed_tokens`` once); each
    round encodes one image per rank via the model's ``get_{modality}_feature``
    embedder (Design X: aux computed inside it) and merges its features in. No
    ``clamp`` -- sglang-jax placeholders are in-vocab ``im_token_id``, not
    upstream's out-of-range hash ``pad_value``. Returns the merged embedding
    ``[total_token, H]`` (``P("data", None)``).
    """
    mesh = multimodal_model.mesh
    running = input_embedding(input_ids)
    mapping = data_embedding_func_mapping or {}
    for modality, rounds in mm_embed_plan.rounds_by_modality.items():
        embedder = mapping.get(modality) or getattr(
            multimodal_model, f"get_{modality.name.lower()}_feature", None
        )
        assert embedder is not None, f"no embedding method for {modality}"
        for rnd in rounds:
            features = embedder(rnd.encode_inputs)  # JIT(1) via get_image_feature
            running = mm_merge(mesh, running, features, rnd.src_idx, rnd.mask)  # JIT(2)
    return running


def general_mm_embed_routine(
    input_ids,
    forward_batch,
    language_model,
    multimodal_model,
    mm_embed_plan,
    data_embedding_funcs=None,
):
    """HOST segment (= upstream ``general_mm_embed_routine``, mm_utils.py:1023 --
    minus the LM call, since our backbone is a separate JIT).

    Strings ``embed_tokens`` -> per-round encode->merge, landing the merged
    embedding on ``forward_batch.input_embedding`` for the backbone JIT to
    consume (= upstream's ``forward_batch.mm_input_embeds`` field set).
    """
    embed_tokens = language_model.get_input_embeddings()
    input_embeds = embed_mm_inputs(
        mm_embed_plan,
        input_ids,
        embed_tokens,
        multimodal_model,
        data_embedding_funcs,
    )
    forward_batch.input_embedding = input_embeds
