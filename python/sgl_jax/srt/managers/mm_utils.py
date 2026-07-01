"""In-model multimodal embed routine: the host orchestration that strings the
three independently-compiled JIT segments per forward.

Target forward (extend only) for the in-model VLM path::

    JIT(1) vision encode (pure jit, GSPMD batched)
    JIT(2) multimodal merge (shard_map)
    JIT(3) language backbone  (existing, unchanged)

Vision uses owning-rank data-parallel: each DP rank encodes its own images and
merges into its own tokens with ZERO cross-rank collective. The ``data`` mesh
axis == dp size; everything vision flows as ``P("data")``.

This module owns JIT(1) and JIT(2) plus the host loop that drives them
(``general_mm_embed_routine`` -> ``embed_mm_inputs``).
``jitted_mm_encode`` is a pure ``@jax.jit`` (GSPMD batched ViT, dp-leading);
``jitted_mm_merge`` is ``@jax.jit`` wrapping a ``shard_map`` (rank-local gather).
Both are forward JIT segments (the same layer as ``jitted_run_model``), with
``mesh`` as a static arg for caching.
"""

import functools

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec

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
def jitted_mm_merge(mesh, running, features, src_idx, mask):
    """Forward JIT(2): src_idx merge, ``P("data")`` flat layout.

    A forward JIT segment (same layer as ``jitted_run_model``). The shard_map is
    inlined and ``@jax.jit`` (``mesh`` static) caches it -- matching the repo's
    jit-wrapped kernel-entry convention (``kernels/fused_moe``,
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


@functools.partial(jax.jit, static_argnames=("mesh", "graphdef", "body"))
def jitted_mm_encode(mesh, body, graphdef, state, pixels, meta, valid):
    """Forward JIT(1): GSPMD batched ViT (pure jit, dp-leading; spec §2.4 / §3.10).

    Pure ``@jax.jit`` (NO outer shard_map): the ViT runs as a GSPMD batched
    computation with the leading ``dp`` axis as the batch. Inputs are already
    device_put on ``P("data", ...)`` by ``init_new``; the ViT body operates on
    ``[dp, ...]`` and GSPMD partitions the matmuls/norms along ``data`` (weights
    replicated -> per-image independent -> zero cross-rank collective). The two
    ops GSPMD can't auto-partition keep their OWN local shard_map: attention
    (inside the body, via ``VisionFlashAttentionBackend``) and merge
    (``jitted_mm_merge``, JIT(2)).

    ViT weights flow in as an EXPLICIT operand via ``nnx.split`` (``graphdef``
    static + ``state`` traced), replicated (no TP); ``nnx.merge`` rebuilds inside.

    ``meta`` is an opaque scheduler-computed per-arch registered pytree; common
    code never names its fields -- the body interprets them.

    The body returns batched ``[dp, out_rows, H]``; we flatten to
    ``[dp*out_rows, H]`` and ANCHOR the sharding with
    ``with_sharding_constraint(P("data", None))`` so the encode->merge seam lines
    up with ``jitted_mm_merge``'s ``_MERGE_IN_SPECS`` (the dp-leading flatten is
    naturally aligned -- sharded-major ``dp`` + replicated ``out_rows`` -- and the
    anchor pins "inferred" to "contractual", spec §3.10 #13). Returns
    ``P("data", None)``.
    """
    visual = nnx.merge(graphdef, state)
    features = body(visual, pixels, meta, valid)  # [dp, out_rows, H]
    dp, out_rows, h = features.shape
    features = features.reshape(dp * out_rows, h)  # [dp*out_rows, H]
    return jax.lax.with_sharding_constraint(
        features, NamedSharding(mesh, PartitionSpec("data", None))
    )


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
    embedder, which consumes scheduler-built ``enc.meta``, and merges its
    features in. No ``clamp`` -- sglang-jax placeholders are in-vocab
    ``im_token_id``, not upstream's out-of-range hash ``pad_value``. Returns the
    merged embedding ``[total_token, H]`` (``P("data", None)``).
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
            running = jitted_mm_merge(mesh, running, features, rnd.src_idx, rnd.mask)  # JIT(2)
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
