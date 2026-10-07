"""Global indexer top-k over a DCP-sharded key cache.

Each rank scores **local** keys (physical slots), maps them to virtual ids,
all-gathers scores, then takes a global top-k. The returned ids for rank ``r``
are the global top-k with non-owned entries set to -1 (local attend uses
``owned ∩ topk``).

Two layouts (``interleave`` == ``I``, see ``layout.py``):

* ``I = 1`` — token-striped. A DSA page's ``page_size`` tokens are spread over
  all ranks, so a page's score needs a cross-rank max and the attend read is
  ``page_size/dcp_size`` slots wide.
* ``I = page_size`` — block-interleaved, the production layout. Rank ``r``'s
  physical page ``P`` **is** global DSA page ``P * dcp_size + r``, so each page's
  score is computed entirely by its owner: all-gather the per-page maxes and
  interleave them, with no cross-rank max, and the attend reads a whole page.

``dcp_size=1`` is ordinary top-k on the single shard (virtual == physical) under
either ``I``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sgl_jax.srt.layers.dcp.layout import owner, virtual_index
from sgl_jax.srt.layers.dcp.write import owned_len_jax

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]


def global_topk_from_shards(
    shard_scores: FloatArray,
    k: int,
) -> tuple[IntArray, list[IntArray]]:
    """Gather local scores and take a global top-k.

    Parameters
    ----------
    shard_scores
        ``[dcp_size, T, local_kv]``. ``shard_scores[r, t, p]`` is the score of
        physical slot ``p`` on rank ``r`` (virtual id ``p * dcp_size + r``).
        Use ``-inf`` for padding.
    k
        Global top-k budget (e.g. 2048).

    Returns
    -------
    global_ids
        ``[T, k]`` virtual ids, highest score first.
    owned_ids
        length-``dcp_size`` list of ``[T, k]`` arrays: same ids with
        ``owner != rank`` replaced by -1.
    """
    scores = np.asarray(shard_scores, dtype=np.float32)
    if scores.ndim != 3:
        raise ValueError(f"shard_scores must be [R, T, L], got {scores.shape}")
    dcp_size, t, local_kv = scores.shape
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    k = min(k, dcp_size * local_kv)

    phys = np.arange(local_kv, dtype=np.int32)
    virt = np.stack(
        [virtual_index(phys, dcp_size, r).astype(np.int32) for r in range(dcp_size)],
        axis=0,
    )
    virt = np.broadcast_to(virt[:, None, :], scores.shape)

    flat_s = np.transpose(scores, (1, 0, 2)).reshape(t, dcp_size * local_kv)
    flat_v = np.transpose(virt, (1, 0, 2)).reshape(t, dcp_size * local_kv)

    # Stable: sort all, take k. Fine for the CPU I1 sizes.
    order = np.argsort(-flat_s, axis=-1, kind="stable")
    sel = order[:, :k]
    global_ids = np.take_along_axis(flat_v, sel, axis=-1).astype(np.int32)
    # Drop -inf padded slots.
    top_s = np.take_along_axis(flat_s, sel, axis=-1)
    global_ids = np.where(np.isfinite(top_s), global_ids, np.int32(-1))

    owned_ids = []
    for r in range(dcp_size):
        keep = (global_ids >= 0) & (owner(np.where(global_ids < 0, 0, global_ids), dcp_size) == r)
        owned_ids.append(np.where(keep, global_ids, np.int32(-1)))
    return global_ids, owned_ids


def owned_topk_jax(
    local_scores,
    dcp_size: int,
    dcp_rank,
    k: int,
):
    """Device twin of ``global_topk_from_shards`` for one rank.

    ``local_scores`` is ``[T, local_kv]``. Must run inside a ``shard_map``
    that names the DCP axis ``tensor`` (v0: ``dcp_size == tp_size``).
    """
    import jax
    import jax.numpy as jnp

    scores = local_scores.astype(jnp.float32)
    t, local_kv = scores.shape
    phys = jnp.arange(local_kv, dtype=jnp.int32)
    virt = jnp.broadcast_to((phys * dcp_size + dcp_rank).astype(jnp.int32), (t, local_kv))
    gathered_s = jax.lax.all_gather(scores, "tensor", axis=0)
    gathered_v = jax.lax.all_gather(virt, "tensor", axis=0)
    flat_s = gathered_s.transpose(1, 0, 2).reshape(t, dcp_size * local_kv)
    flat_v = gathered_v.transpose(1, 0, 2).reshape(t, dcp_size * local_kv)
    kk = min(k, dcp_size * local_kv)
    top_s, sel = jax.lax.top_k(flat_s, kk)
    top_v = jnp.take_along_axis(flat_v, sel, axis=-1)
    valid = jnp.isfinite(top_s)
    top_v = jnp.where(valid, top_v, jnp.int32(-1))
    owned = jnp.where((top_v >= 0) & ((top_v % dcp_size) == dcp_rank), top_v, jnp.int32(-1))
    if kk < k:
        owned = jnp.pad(owned, ((0, 0), (0, k - kk)), constant_values=-1)
    return owned.astype(jnp.int32)


def score_decode_local_jax(
    q,
    weights,
    cache3d,
    seq_lens,
    page_indices,
    cu_kv_lens,
    pages_per_seq: int,
    page_size: int,
    dcp_size: int,
    dcp_rank,
    interleave: int = 1,
):
    """Per-seq local indexer scores for decode (``T == num_seqs``).

    Keys are read from this rank's pages; the causal bound is
    ``get_dcp_lens(seq_lens, dcp_size, rank, interleave)`` so we only score owned
    tokens.
    """
    import jax
    import jax.numpy as jnp

    t, _, dim = q.shape
    max_kv = pages_per_seq * page_size
    w = weights.astype(jnp.float32)
    local_len = owned_len_jax(seq_lens, dcp_size, dcp_rank, interleave)

    def body(seq_id, scores):
        seq_pages = jax.lax.dynamic_slice_in_dim(
            page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
        )
        keys = cache3d[seq_pages].reshape(max_kv, dim)
        q_i = jax.lax.dynamic_slice_in_dim(q, seq_id, 1, axis=0)
        w_i = jax.lax.dynamic_slice_in_dim(w, seq_id, 1, axis=0)
        dots = jnp.einsum("thd,kd->thk", q_i, keys, preferred_element_type=jnp.float32)
        row = jnp.einsum("th,thk->tk", w_i, jax.nn.relu(dots))
        mask = jnp.arange(max_kv) < local_len[seq_id]
        row = jnp.where(mask[None, :], row, jnp.float32(-jnp.inf))
        return scores.at[seq_id].set(row[0])

    scores = jnp.full((t, max_kv), -jnp.inf, dtype=jnp.float32)
    return jax.lax.fori_loop(0, t, body, scores)


def score_prefill_local_jax(
    q,
    weights,
    cache3d,
    seq_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    positions,
    pages_per_seq: int,
    page_size: int,
    dcp_size: int,
    dcp_rank,
    interleave: int = 1,
):
    """Per-query local indexer scores for prefill (``T`` tokens, ``S`` seqs).

    Same virtual-id layout as decode. Causal: score physical slot ``p`` only when
    ``virtual_index(p) <= positions[t]`` and ``p < local_len[seq]``.
    """
    import jax
    import jax.numpy as jnp

    t, _, dim = q.shape
    s = seq_lens.shape[0]
    max_kv = pages_per_seq * page_size
    w = weights.astype(jnp.float32)
    tok = jnp.arange(t, dtype=jnp.int32)
    seq_id = jnp.clip(jnp.searchsorted(cu_q_lens[1:], tok, side="right"), 0, s - 1)

    def keys_for_seq(sid):
        seq_pages = jax.lax.dynamic_slice_in_dim(
            page_indices, cu_kv_lens[sid] // page_size, pages_per_seq
        )
        return cache3d[seq_pages].reshape(max_kv, dim)

    keys_all = jax.vmap(keys_for_seq)(jnp.arange(s, dtype=jnp.int32))
    keys = keys_all[seq_id]
    dots = jnp.einsum("thd,tkd->thk", q, keys, preferred_element_type=jnp.float32)
    scores = jnp.einsum("th,thk->tk", w, jax.nn.relu(dots))
    phys = jnp.arange(max_kv, dtype=jnp.int32)
    virt = _virt_ids(phys, dcp_size, dcp_rank, interleave)
    local_len = owned_len_jax(seq_lens[seq_id], dcp_size, dcp_rank, interleave)
    tok_valid = seq_lens[seq_id] > 0
    mask = (
        (phys[None, :] < local_len[:, None])
        & (virt[None, :] <= positions[:, None])
        & tok_valid[:, None]
    )
    return jnp.where(mask, scores, jnp.float32(-jnp.inf))


def _virt_ids(phys, dcp_size: int, dcp_rank, interleave: int):
    """Virtual id of local physical slot ``phys`` (jnp twin of ``layout.virtual_index``)."""
    if interleave == 1:
        return phys * dcp_size + dcp_rank
    i = interleave
    return ((phys // i) * dcp_size + dcp_rank) * i + (phys % i)


def local_page_max_jax(local_scores, page_size: int):
    """Per-page maxima of this rank's slot scores: ``[T, local_kv] -> [T, n_local]``."""
    t, max_kv = local_scores.shape
    if max_kv % page_size:
        raise ValueError(f"local kv {max_kv} must be a multiple of page_size {page_size}")
    return local_scores.reshape(t, max_kv // page_size, page_size).max(axis=-1)


def page_topk_from_gathered(
    gathered,
    dcp_size: int,
    dcp_rank,
    page_size: int,
    k_pages: int,
    positions,
):
    """Global DSA-page top-k from all-gathered per-rank page maxima.

    ``gathered`` is ``[dcp_size, T, n_local]``; entry ``[r, t, P]`` is rank ``r``'s
    page ``P``, which under ``I == page_size`` **is** global DSA page
    ``P * dcp_size + r``. So the global page-score vector is just the interleave
    with rank as the fastest axis — no cross-rank max, because no page is split.

    Returns ``(global_pages, owned_phys_pages)``, both ``[T, k_pages]`` int32 and
    ``-1`` padded. ``global_pages`` is identical on every rank (it is the same
    top-k a ``dcp=1`` run would pick). ``owned_phys_pages`` holds this rank's
    **physical** page ids compacted to the front of each row, so the attend
    kernel's static ``K`` is the array width while its dynamic ``cnt`` walks only
    the real entries.
    """
    import jax
    import jax.numpy as jnp

    _, t, n_local = gathered.shape
    global_ps = gathered.transpose(1, 2, 0).reshape(t, n_local * dcp_size)

    n_pages = n_local * dcp_size
    page_start = jnp.arange(n_pages, dtype=jnp.int32) * page_size
    global_ps = jnp.where(
        page_start[None, :] <= positions[:, None], global_ps, jnp.float32(-jnp.inf)
    )

    kk = min(k_pages, n_pages)
    vals, top = jax.lax.top_k(global_ps, kk)
    top = jnp.where(jnp.isfinite(vals), top.astype(jnp.int32), jnp.int32(-1))
    if kk < k_pages:
        top = jnp.pad(top, ((0, 0), (0, k_pages - kk)), constant_values=-1)

    owned = (top >= 0) & ((top % dcp_size) == dcp_rank)
    phys = jnp.where(owned, top // dcp_size, jnp.int32(-1))
    # stable sort on ~owned keeps the score order among the kept pages
    order = jnp.argsort(~owned, axis=-1, stable=True)
    phys = jnp.take_along_axis(phys, order, axis=-1)
    return top.astype(jnp.int32), phys.astype(jnp.int32)


def page_topk_blocked_jax(
    local_scores,
    dcp_size: int,
    dcp_rank,
    page_size: int,
    k_pages: int,
    positions,
):
    """Block-interleaved global page top-k. Must run inside a ``tensor``-named mesh."""
    import jax

    local_max = local_page_max_jax(local_scores, page_size)
    gathered = jax.lax.all_gather(local_max, "tensor", axis=0)  # [dcp, T, n_local]
    return page_topk_from_gathered(gathered, dcp_size, dcp_rank, page_size, k_pages, positions)


def virtual_page_topk_jax(
    local_scores,
    dcp_size: int,
    dcp_rank,
    page_size: int,
    k_pages: int,
    positions,
):
    """Global page-topk matching dcp=1 ``streamindex_page_topk`` (max over page).

    Local physical slots are striped: rank ``r`` slot ``p`` is virtual
    ``p * N + r``. Because ``page_size % N == 0``, each 128-token DSA page
    is a contiguous block of ``page_size // N`` physical slots — the same
    reshape-max dcp=1 uses, then all-gathered and max-ed across ranks.
    """
    import jax
    import jax.numpy as jnp

    t, max_kv = local_scores.shape
    if page_size % dcp_size != 0:
        raise ValueError(f"page_size={page_size} must be divisible by dcp_size={dcp_size}")
    spp = page_size // dcp_size
    n_dsa = max_kv // spp
    local_max = local_scores.reshape(t, n_dsa, spp).max(axis=-1)
    gathered = jax.lax.all_gather(local_max, "tensor", axis=0)
    global_ps = jnp.max(gathered, axis=0)
    page_start = jnp.arange(n_dsa, dtype=jnp.int32) * page_size
    global_ps = jnp.where(
        page_start[None, :] <= positions[:, None],
        global_ps,
        jnp.float32(-jnp.inf),
    )
    kk = min(k_pages, n_dsa)
    vals, top = jax.lax.top_k(global_ps, kk)
    top = jnp.where(jnp.isfinite(vals), top, jnp.int32(-1))
    if kk < k_pages:
        top = jnp.pad(top, ((0, 0), (0, k_pages - kk)), constant_values=-1)
    return top.astype(jnp.int32)


def owned_phys_from_vpages(vpages, dcp_size: int, dcp_rank, page_size: int):
    """Expand virtual pages to this rank's physical slots. ``[T, K*page_size/N]``."""
    import jax.numpy as jnp

    offs = jnp.arange(page_size, dtype=jnp.int32)
    virt = vpages[..., None] * page_size + offs
    owned = (vpages[..., None] >= 0) & ((virt % dcp_size) == dcp_rank)
    phys = virt // dcp_size
    t = vpages.shape[0]
    return jnp.where(owned, phys, jnp.int32(-1)).reshape(t, -1)


def dcp_gather_attend(
    ql,
    qpe,
    cache,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    phys_ids,
    positions,
    sm_scale: float,
    page_size: int,
    dcp_size: int,
    dcp_rank,
    query_chunk: int = 256,
):
    """Dense JAX attend over gathered owned tokens. Returns ``(o, lse)``.

    Scans ``query_chunk`` rows at a time so 8k × all-heads × topk does not
    materialise a 100G+ HLO temp.
    """
    import jax
    import jax.numpy as jnp

    t = ql.shape[0]
    chunk = min(query_chunk, t)
    pad = (-t) % chunk
    if pad:
        ql = jnp.pad(ql, ((0, pad), (0, 0), (0, 0)))
        qpe = jnp.pad(qpe, ((0, pad), (0, 0), (0, 0)))
        phys_ids = jnp.pad(phys_ids, ((0, pad), (0, 0)), constant_values=-1)
        positions = jnp.pad(positions, ((0, pad),), constant_values=-1)
    n = ql.shape[0] // chunk
    pn, pspk, pk, dk = cache.shape
    ps = page_size
    flat = cache.reshape(pn * ps, dk)
    dv = ql.shape[-1]
    rope = qpe.shape[-1]

    def _chunk(ql_c, qpe_c, phys_c, pos_c):
        ct = ql_c.shape[0]
        # chunk is a slice of the packed token axis; seq_id from global pos
        # is recovered by the caller passing already-sliced rows of a single
        # request (D2: max_running=1). All rows share seq 0.
        seq_id = jnp.zeros((ct,), dtype=jnp.int32)
        page_local = jnp.maximum(phys_c // ps, 0)
        off = jnp.maximum(phys_c, 0) % ps
        base = cu_kv_lens[seq_id] // ps
        gpage = page_indices[jnp.clip(base[:, None] + page_local, 0, page_indices.shape[0] - 1)]
        idx = gpage * ps + off
        idx = jnp.where(phys_c >= 0, idx, 0)
        rows = flat[idx]
        q = jnp.concatenate([ql_c, qpe_c], axis=-1)
        k = rows[..., : dv + rope]
        v = rows[..., :dv]
        scores = jnp.einsum("thd,tkd->thk", q, k, preferred_element_type=jnp.float32) * sm_scale
        virt = jnp.where(phys_c >= 0, phys_c * dcp_size + dcp_rank, jnp.int32(-1))
        valid = (phys_c >= 0) & (virt <= pos_c[:, None])
        scores = jnp.where(valid[:, None, :], scores, jnp.float32(-jnp.inf))
        m = jnp.max(scores, axis=-1)
        p = jnp.exp(scores - jnp.where(jnp.isfinite(m), m, 0.0)[..., None])
        p = jnp.where(jnp.isfinite(m)[..., None] & valid[:, None, :], p, 0.0)
        lsum = jnp.sum(p, axis=-1)
        o = jnp.einsum("thk,tkd->thd", p, v.astype(jnp.float32))
        o = o / jnp.where(lsum[..., None] > 0, lsum[..., None], 1.0)
        lse = jnp.where((lsum > 0) & jnp.isfinite(m), m + jnp.log(lsum), jnp.float32(-jnp.inf))
        o = jnp.where(jnp.isfinite(m)[..., None], o, 0.0)
        return o, lse

    ql_b = ql.reshape(n, chunk, *ql.shape[1:])
    qpe_b = qpe.reshape(n, chunk, *qpe.shape[1:])
    phys_b = phys_ids.reshape(n, chunk, phys_ids.shape[1])
    pos_b = positions.reshape(n, chunk)

    def _scan(_, xs):
        return None, _chunk(*xs)

    _, (o, lse) = jax.lax.scan(_scan, None, (ql_b, qpe_b, phys_b, pos_b))
    o = o.reshape(-1, *o.shape[2:])[:t]
    lse = lse.reshape(-1, *lse.shape[2:])[:t]
    return o, lse
