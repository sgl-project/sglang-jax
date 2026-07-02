"""Device-side vision encode (jitted, shard_map on 'data').

Wraps an arch-specific per-rank vision encode function into a jitted callable
that dispatches across DP ranks using ``jax.shard_map`` on the ``"data"`` axis:

  in : pixels [dp, patch_k, patch_dim]  P("data", None, None)
       valid  [dp]                       P("data")
       meta   pytree, lead dim = dp      P("data", None, ...)
  out: features [dp, feature_k, hidden]  P("data", None, None)

Each rank calls ``encode_local_fn(pixels_kd, valid_scalar, meta_pytree)`` and
returns its features; dummy ranks (``valid == 0``) still run — their outputs
are ignored downstream because ``jitted_mm_merge`` masks them out.

The DP price of the round-loop model: every rank does one encode per round,
padded to the round's max patches, even if it has no image this round.

The vision weights are expected to be **DP-replicated** (no tensor sharding
inside the vision tower); ``encode_local_fn`` may capture them via closure.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.multimodal.embed_plan import VisionEncodeInputs


def _meta_pspec(x: jax.Array) -> P:
    ndim = getattr(x, "ndim", 1)
    return P("data", *((None,) * (ndim - 1)))


def make_jitted_mm_encode(
    encode_local_fn: Callable[[jax.Array, jax.Array, Any], jax.Array],
    mesh: Mesh,
) -> Callable[[VisionEncodeInputs], jax.Array]:
    """Return a jitted callable that runs vision encode across DP ranks.

    ``encode_local_fn`` runs on **one rank's slice** already stripped of the
    lead ``dp`` axis:

        encode_local_fn(pixels_kd, valid_scalar, meta_pytree_no_dp) -> features_fh
    """

    def _local(pixels_local, valid_local, meta_local):
        # shard_map keeps the rank axis (size 1 per shard) — drop it, encode,
        # add it back so the stacked output regains its dp dimension.
        pixels = pixels_local[0]
        valid = valid_local[0]
        meta = jax.tree_util.tree_map(lambda x: x[0], meta_local)
        features = encode_local_fn(pixels, valid, meta)
        return features[None]

    def _encode(pixels, valid, meta):
        meta_specs = jax.tree_util.tree_map(_meta_pspec, meta)
        return jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=(P("data", None, None), P("data"), meta_specs),
            out_specs=P("data", None, None),
        )(pixels, valid, meta)

    jitted = jax.jit(_encode)

    def encode(encode_inputs: VisionEncodeInputs) -> jax.Array:
        return jitted(encode_inputs.pixels, encode_inputs.valid, encode_inputs.meta)

    return encode


def jitted_mm_encode(
    encode_local_fn: Callable[[jax.Array, jax.Array, Any], jax.Array],
    encode_inputs: VisionEncodeInputs,
    mesh: Mesh,
) -> jax.Array:
    """One-shot convenience wrapper around :func:`make_jitted_mm_encode`."""
    return make_jitted_mm_encode(encode_local_fn, mesh)(encode_inputs)
