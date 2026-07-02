from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


def _merge_local(running_local, features_local, src_idx_local, mask_local):
    # features_local: [1, patch_k, hidden]  (rank axis kept by shard_map)
    feats = features_local[0]
    gathered = feats[src_idx_local]  # [per_dp_token, hidden]
    return jnp.where(mask_local[:, None], gathered.astype(running_local.dtype), running_local)


@partial(jax.jit, static_argnames=("mesh",))
def jitted_mm_merge(
    running: jax.Array,
    features: jax.Array,
    src_idx: jax.Array,
    mask: jax.Array,
    mesh: Mesh,
) -> jax.Array:
    return jax.shard_map(
        _merge_local,
        mesh=mesh,
        in_specs=(P("data", None), P("data", None, None), P("data"), P("data")),
        out_specs=P("data", None),
    )(running, features, src_idx, mask)
