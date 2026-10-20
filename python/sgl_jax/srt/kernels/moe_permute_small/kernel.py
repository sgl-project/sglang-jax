"""Small-batch MoE routing-index kernel (one pallas_call replaces argsort / bincount / inverse-argsort).

At decode / verify batch sizes (N = tokens x topk <= 256) XLA lowers the three index ops of
``EPMoE._permute`` / ``_unpermute`` (stable argsort of the expert ids, bincount, inverse argsort)
to separate sort / reduce / scatter programs, 1.7-2.4 us each per layer (bs1 census 10-05).  All of
them are integer functions of the [tokens, topk] id matrix, so one O(N^2) comparison in VMEM gives
them bit-exactly:

  rank[i]   = #{ j : id_j < id_i  or (id_j == id_i and j < i) }   (stable ascending sort position)
  sorted[p] = i  with rank[i] == p                                 (== jnp.argsort(ids, stable=True))
  inverse   = rank                                                 (== jnp.argsort(sorted))
  group_sizes[e] = #{ i : id_i == e }                              (== jnp.bincount(ids, length=E))
  token_idx[p]   = sorted[p] // topk
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# EPMoE uses this kernel for the routing indices when tokens * topk <= PERMUTE_KERNEL_MAX_ROWS
# (decode / verify batches); larger batches take the XLA sort / bincount path.
PERMUTE_KERNEL_MAX_ROWS = 256


def _kernel(ids_r_ref, ids_c_ref, sorted_ref, tok_ref, inv_ref, gs_ref, *, n: int, k: int, e: int):
    ids_r = ids_r_ref[...]  # [1, N]  int32 expert id per routing slot
    ids_c = ids_c_ref[...]  # [N, 1]  same data, slot on sublanes
    j = jax.lax.broadcasted_iota(jnp.int32, (n, n), 1)  # column (j) index
    i = jax.lax.broadcasted_iota(jnp.int32, (n, n), 0)  # row (i) index
    before = (ids_r < ids_c) | ((ids_r == ids_c) & (j < i))  # [N(i), N(j)]
    rank = jnp.sum(before.astype(jnp.int32), axis=1, keepdims=True)  # [N, 1]
    inv_ref[...] = jnp.sum(
        jnp.where(j == i, rank, 0), axis=0, keepdims=True
    )  # [1, N]: inverse[i] = rank[i]
    # sorted[p] = i where rank[i] == p   (one-hot over p on lanes, reduce over i on sublanes)
    p = j
    sorted_row = jnp.sum(jnp.where(rank == p, i, 0), axis=0, keepdims=True)  # [1, N]
    sorted_ref[...] = sorted_row
    tok_ref[...] = sorted_row // k
    # group sizes: one-hot [N(i), E] reduced over i
    ee = jax.lax.broadcasted_iota(jnp.int32, (n, e), 1)
    gs_ref[...] = jnp.sum((ee == ids_c).astype(jnp.int32), axis=0, keepdims=True)  # [1, E]


def permute_indices_small(top_k_ids: jax.Array, num_experts: int, *, interpret: bool | None = None):
    """top_k_ids: int32 [T, k] -> (sorted_selected_experts[N], token_indices[N], group_sizes[E],
    inverse[N]) with N = T * k. Bit-exact with the jnp reference (stable argsort / bincount)."""
    t, k = top_k_ids.shape
    n = t * k
    if interpret is None:
        interpret = jax.default_backend() == "cpu"
    ids = top_k_ids.astype(jnp.int32).reshape(n)
    ids_r = ids.reshape(1, n)
    ids_c = ids.reshape(n, 1)
    kern = lambda *refs: _kernel(*refs, n=n, k=k, e=num_experts)  # noqa: E731
    vmem = pltpu.MemorySpace.VMEM if hasattr(pltpu, "MemorySpace") else pltpu.TPUMemorySpace.VMEM
    sorted_row, tok_row, inv_row, gs_row = pl.pallas_call(
        kern,
        in_specs=[pl.BlockSpec(memory_space=vmem), pl.BlockSpec(memory_space=vmem)],
        out_specs=[pl.BlockSpec(memory_space=vmem)] * 4,
        out_shape=[
            jax.ShapeDtypeStruct((1, n), jnp.int32),
            jax.ShapeDtypeStruct((1, n), jnp.int32),
            jax.ShapeDtypeStruct((1, n), jnp.int32),
            jax.ShapeDtypeStruct((1, num_experts), jnp.int32),
        ],
        interpret=interpret,
        name="moe-permute-small",
    )(ids_r, ids_c)
    return (
        sorted_row.reshape(n),
        tok_row.reshape(n),
        gs_row.reshape(num_experts),
        inv_row.reshape(n),
    )


def permute_indices_reference(top_k_ids: jax.Array, num_experts: int):
    t, k = top_k_ids.shape
    flat = jnp.ravel(top_k_ids.astype(jnp.int32))
    sorted_sel = jnp.argsort(flat, stable=True)
    return (
        sorted_sel,
        sorted_sel // k,
        jnp.bincount(flat, length=num_experts),
        jnp.argsort(sorted_sel),
    )
