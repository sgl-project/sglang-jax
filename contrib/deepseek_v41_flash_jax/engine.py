"""DeepSeek-V4.1 unified chunk step in JAX: static chunk size S, TRACED start position, fixed-capacity caches.

One function serves chunked prefill (S = 2, 4, ..., 512, start aligned to S) and decode (S = 1). Everything that
depends on the position is array arithmetic (no Python branching on traced values), so it jits once per S.

Rules that keep compression groups aligned (V4.1 uses ratios 0/1/2): a chunk with S > 1 must have S % ratio == 0
and start % ratio == 0 (any power-of-two chunk at a multiple of its size satisfies this); S == 1 is the decode path
with a partial-group state. pos0 may be a scalar (all rows aligned) or [B] (each row at its own position; the
alignment rule then applies per active row). `active` [B] bool masks every state write, so a step can advance a
subset of rows (e.g. prefill one request while the other rows hold their decode state).

Validated against the reference in test_engine.py (same weights/outputs as test_vs_ref.py).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from engram_jax import dequant_rows, engram_gate
from model_ref import (
    Args,
    P_sub,
    act_quant_fp4,
    act_quant_fp8,
    expert,
    hc_mixes,
    hc_post,
    hc_pre,
    moe,
    rms_norm,
    rotary,
)

NEG = -jnp.inf

# Optional explicit layouts for the attention tensors (V41_ATTN_CONSTRAINTS=1 and MESH set by the generator): rows over the data-parallel axis,
# heads over the tensor-parallel axis. Left to itself the compiler picks layouts that need big all-gathers inside attention.
MESH = None


def _c(x, *spec):
    import os

    if MESH is None or os.environ.get("V41_ATTN_CONSTRAINTS", "0") != "1":
        return x
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as PS

    return jax.lax.with_sharding_constraint(x, NamedSharding(MESH, PS(*spec)))


def _rot_tail(x, rd, cos, sin, inverse=False):
    return jnp.concatenate([x[..., :-rd], rotary(x[..., -rd:], cos, sin, inverse)], axis=-1)


def init_states(a: Args, bsz, dtype=jnp.float32):
    """Fixed-capacity caches. T = max_seq // ratio entries for compressed caches."""
    sts = []
    for layer in range(a.n_layers):
        r = a.compress_ratios[layer]
        st = {"window": jnp.zeros((bsz, a.window_size, a.head_dim), dtype)}
        if r and layer in a.kv_source_layers:
            t = a.max_seq_len // r
            st["ckv"] = jnp.zeros((bsz, t, a.head_dim), dtype)
            if r > 1:
                st["kv_state"] = jnp.zeros((bsz, r, a.head_dim), jnp.float32)
                st["score_state"] = jnp.full((bsz, r, a.head_dim), NEG, jnp.float32)
            st["index_k"] = jnp.zeros((bsz, t, a.index_head_dim), dtype)
        sts.append(st)
    return sts


def _pool_groups(P, x, st, pos0, S, r, a, active, nvalid=None):
    """Compressor. pos0/active [B]. Returns (latent [B,G,hd] pre-RoPE, group_positions [B,G], valid [B] or [B,G], new st fields).
    nvalid [B] (optional): only the first nvalid tokens of the chunk are real (the rest is padding): a group is valid only if all of
    its tokens are real (nvalid is even whenever it is < S, so no group is ever split)."""
    x32 = x.astype(jnp.float32)
    if r == 1:
        lat = rms_norm(x @ P["wkv.weight"].T, P["norm.weight"], a.norm_eps)
        valid = (
            active
            if nvalid is None
            else active[:, None] & (jnp.arange(S)[None, :] < nvalid[:, None])
        )
        return lat, pos0[:, None] + jnp.arange(S), valid, {}
    kv, score = x32 @ P["wkv.weight"].T, x32 @ P["wgate.weight"].T
    if S > 1:
        assert S % r == 0
        b = x.shape[0]
        kv = kv.reshape(b, S // r, r, -1)
        score = score.reshape(b, S // r, r, -1)
        pooled = jnp.sum(kv * jax.nn.softmax(score, axis=2), axis=2)
        lat = rms_norm(pooled.astype(x.dtype), P["norm.weight"], a.norm_eps)
        valid = (
            active
            if nvalid is None
            else active[:, None] & ((jnp.arange(S // r)[None, :] + 1) * r <= nvalid[:, None])
        )
        return lat, pos0[:, None] + jnp.arange(S // r) * r, valid, {}
    slot = pos0 % r
    # one-hot select instead of a scatter at a per-row dynamic slot: purely elementwise, so every device updates its own rows locally
    # (the scatter made the compiler all-gather the whole state across the data-parallel axis: ~28% of a decode step at 128 rows)
    hit = (jnp.arange(r)[None, :] == slot[:, None]) & active[:, None]  # [B, r]
    kv_state = jnp.where(hit[:, :, None], kv[:, 0][:, None, :], st["kv_state"])
    score_state = jnp.where(hit[:, :, None], score[:, 0][:, None, :], st["score_state"])
    pooled = jnp.sum(kv_state * jax.nn.softmax(score_state, axis=1), axis=1, keepdims=True)
    lat = rms_norm(pooled.astype(x.dtype), P["norm.weight"], a.norm_eps)
    valid = ((pos0 + 1) % r == 0) & active
    return lat, (pos0 + 1 - r)[:, None], valid, {"kv_state": kv_state, "score_state": score_state}


def _write_cache(cache, rows, g0, valid):
    """cache [B,T,d]; rows [B,G,d] written at per-row group index g0 [B] only where valid ([B] per row, or [B,G] per group)."""

    def one(c, r_, g, v):
        cur = jax.lax.dynamic_slice_in_dim(c, g, r_.shape[0], axis=0)
        vv = v if v.ndim == 0 else v[:, None]
        return jax.lax.dynamic_update_slice_in_dim(c, jnp.where(vv, r_, cur), g, axis=0)

    return jax.vmap(one)(cache, rows, g0, valid)


def _cand_blocks(score, clens, topk_blocks, block):
    t = score.shape[-1]
    pad = -t % block
    sc = jnp.pad(score, [(0, 0)] * (score.ndim - 1) + [(0, pad)], constant_values=NEG)
    sc = sc.reshape(*sc.shape[:-1], -1, block).max(axis=-1)
    nb = sc.shape[-1]
    last = (clens - 1) // block  # [B,S]
    sc = jnp.where(jnp.arange(nb)[None, None, :] == last[:, :, None], jnp.inf, sc)
    vals, idx = jax.lax.top_k(sc, min(topk_blocks, nb))
    keep = jnp.any((jnp.arange(nb) == idx[..., None]) & (vals > NEG)[..., None], axis=-2)
    return jnp.repeat(keep, block, axis=-1)[..., :t]


# ---------------------------------------------------------------- long-context indexer (tiled, exact running top-k)


def _tile_for(S, T, heads):
    """Key-tile length: bounds the [B,S,heads,tile] score intermediate to ~256 MB per row; T is a power of two."""
    import os

    if os.environ.get("V41_INDEX_TILE"):  # tests: force small tiles so the multi-tile loop runs
        return min(int(os.environ["V41_INDEX_TILE"]), T)
    t = 16384
    while t > 256 and S * heads * t * 4 > (1 << 28):
        t //= 2
    return min(t, T)


def _index_scores(qi, wts, Kt):
    """qi [B,S,H,d], wts [B,S,H], Kt [B,t,d] -> [B,S,t] (same expression as the full-matrix version)."""
    return (jax.nn.relu(jnp.einsum("bshd,btd->bsht", qi, Kt)) * wts[..., None]).sum(axis=2)


def _topk_merge(carry, v, i, k):
    cv, ci = carry
    allv, alli = jnp.concatenate([cv, v], -1), jnp.concatenate([ci, i], -1)
    nv, sel = jax.lax.top_k(allv, k)
    return nv, jnp.take_along_axis(alli, sel, axis=-1)


def _tiled_topk(score_tile, T, tile, k, lead, big, n_valid=None):
    """Exact top-k over T keys scored tile by tile. score_tile(start) -> [*lead, tile]. Returns (vals, idx) [*lead, k];
    unfilled slots hold idx=big. Ties resolve to the lower index (earlier tiles first), like a single top_k.
    """
    nt = T // tile
    kt = min(k, tile)
    # runtime trip count: tiles entirely past the longest visible length score NEG everywhere and cannot change the result
    nt_dyn = nt if n_valid is None else jnp.minimum(nt, (n_valid + tile - 1) // tile)

    def one(t, carry):
        sc = score_tile(t * tile)
        v, i = jax.lax.top_k(sc, kt)
        return _topk_merge(carry, v, (i + t * tile).astype(jnp.int32), k)

    init = (jnp.full((*lead, k), NEG, jnp.float32), jnp.full((*lead, k), big, jnp.int32))
    if nt == 1:
        return one(0, init)
    return jax.lax.fori_loop(0, nt_dyn, one, init)


def _select_entries(qi, wts, K, clens, a: Args, layer, shared):
    """Indexer selection for one index-source layer. K [B,T,d] = visible index keys (capacity view).
    Sets shared['topk'] ([B,S,k] sorted entry ids, -1 = none) and, at the candidate source layer, shared['cand'] ([B,S,Kb]
    block ids, -1 = none)."""
    b, S, H, _ = qi.shape
    T = K.shape[1]
    k = min(a.index_topk, T)
    tile = _tile_for(S, T, H)
    cs, blk = a.candidate_source_layer, a.candidate_block_size
    after_cand = 0 <= cs < layer
    n_valid = jnp.max(
        clens
    )  # longest visible compressed length over the batch (scalar; same on every device)
    kbc = min(a.candidate_topk_blocks, T // blk)

    def masked(t0, sc):
        return jnp.where(t0 + jnp.arange(sc.shape[-1])[None, None, :] >= clens[:, :, None], NEG, sc)

    def make_score_tile(mm):
        def score_tile(t0):
            sc = masked(
                t0, _index_scores(qi, wts, jax.lax.dynamic_slice_in_dim(K, t0, tile, axis=1))
            )
            if mm is not None:
                m = jnp.repeat(
                    jax.lax.dynamic_slice_in_dim(mm, t0 // blk, tile // blk, axis=2), blk, axis=-1
                )
                sc = jnp.where(m, sc, NEG)
            return sc

        return score_tile

    def mk_memb(
        _=None,
    ):  # per-query candidate-block membership, [B,S,nb] bool (only needed when the context exceeds the budget)
        nb = T // blk
        cand = shared["cand"]
        bi = jnp.arange(b)[:, None, None]
        si = jnp.arange(S)[None, :, None]
        return jnp.zeros((b, S, nb), bool).at[bi, si, jnp.clip(cand, 0, nb - 1)].max(cand >= 0)

    if (
        layer == cs
    ):  # candidate blocks (any S): top-kb blocks by max entry score, the newest block always included
        nb = T // blk
        kb = min(a.candidate_topk_blocks, nb)
        last = (clens - 1) // blk
        score_tile0 = make_score_tile(None)

        def block_tile(t0):
            bm = score_tile0(t0).reshape(b, S, tile // blk, blk).max(axis=-1)
            gb = t0 // blk + jnp.arange(tile // blk)
            return jnp.where(gb[None, None, :] == last[:, :, None], jnp.inf, bm)

        def full_select(_):
            bv, bidx = _tiled_topk_blocks(block_tile, T, tile, blk, kb, (b, S), nb, n_valid)
            return jnp.where(bv > NEG, bidx, -1).astype(jnp.int32)

        def all_blocks(
            _,
        ):  # <= kb visible blocks: every visible block is a candidate (identical result, no scoring)
            ids = jnp.arange(kb, dtype=jnp.int32)[None, None, :]
            return jnp.broadcast_to(jnp.where(ids <= last[:, :, None], ids, -1), (b, S, kb)).astype(
                jnp.int32
            )

        shared["cand"] = jax.lax.cond(n_valid <= kb * blk, all_blocks, full_select, None)

    def full_topk(
        _,
    ):  # exact top-k over the scored entries (restricted to the candidate blocks after the candidate layer)
        if after_cand and S == 1:
            # decode: score only the candidate entries (Kb*blk of them); when the visible context fits the candidate budget every
            # entry is a candidate, so score the visible entries directly (no fixed-size gather)
            def gather_path(_):
                cand = shared["cand"]  # [B,1,Kb]
                ent = (cand[..., None] * blk + jnp.arange(blk)).reshape(b, 1, -1)  # [B,1,E]
                ok = (jnp.repeat(cand, blk, axis=-1) >= 0) & (ent < clens[:, :, None])
                Kc = jnp.take_along_axis(
                    K, jnp.clip(ent[:, 0], 0, T - 1)[..., None], axis=1
                )  # [B,E,d]
                sc = jnp.where(ok, _index_scores(qi, wts, Kc), NEG)
                kk = min(k, sc.shape[-1])
                _, sel = jax.lax.top_k(sc, kk)
                return jnp.take_along_axis(jnp.where(ok, ent, T), sel, axis=-1).astype(jnp.int32)

            if kbc * blk >= k:
                return jax.lax.cond(
                    n_valid <= kbc * blk,
                    lambda _: _tiled_topk(make_score_tile(None), T, tile, k, (b, S), T, n_valid)[1],
                    gather_path,
                    None,
                )
            return gather_path(None)
        if after_cand:
            # <= kbc*blk visible entries: every block is a candidate, no mask; longer contexts build the membership mask
            return jax.lax.cond(
                n_valid <= kbc * blk,
                lambda _: _tiled_topk(make_score_tile(None), T, tile, k, (b, S), T, n_valid)[1],
                lambda _: _tiled_topk(make_score_tile(mk_memb()), T, tile, k, (b, S), T, n_valid)[
                    1
                ],
                None,
            )
        return _tiled_topk(make_score_tile(None), T, tile, k, (b, S), T, n_valid)[1]

    def select_all(
        _,
    ):  # at most k entries visible to ANY query: the top-k is simply every visible entry (no scoring, no sort)
        ids = jnp.arange(k, dtype=jnp.int32)[None, None, :]
        return jnp.broadcast_to(jnp.where(ids < clens[:, :, None], ids, T), (b, S, k)).astype(
            jnp.int32
        )

    ti = jax.lax.cond(n_valid <= k, select_all, full_topk, None)
    ti = jnp.sort(ti, axis=-1)
    shared["topk"] = jnp.where(ti < clens[:, :, None], ti, -1).astype(jnp.int32)


def _tiled_topk_blocks(block_tile, T, tile, blk, kb, lead, big, n_valid=None):
    """Top-kb blocks: block_tile(t0) -> [*lead, tile//blk] block scores for key tile starting at t0."""
    nt = T // tile
    kt = min(kb, tile // blk)
    nt_dyn = nt if n_valid is None else jnp.minimum(nt, (n_valid + tile - 1) // tile)

    def one(t, carry):
        v, i = jax.lax.top_k(block_tile(t * tile), kt)
        return _topk_merge(carry, v, (i + t * (tile // blk)).astype(jnp.int32), kb)

    init = (jnp.full((*lead, kb), NEG, jnp.float32), jnp.full((*lead, kb), big, jnp.int32))
    if nt == 1:
        return one(0, init)
    return jax.lax.fori_loop(0, nt_dyn, one, init)


def _win_blocked(S, win):
    import os

    return os.environ.get("V41_WIN_BLOCK", "0") == "1" and S >= win and S % win == 0


def _use_attn_kernel(S):
    import os

    return os.environ.get("V41_ATTN_KERNEL", "0") == "1" and S >= int(
        os.environ.get("V41_ATTN_KERNEL_MIN_S", "64")
    )


def _attention_kernel(q, K, pos0, ckv, tk, sink, win, scale):
    """Window + selected-compressed-entry attention with the SGL-JAX fused flash kernel (csa_flash_attention), run per device on its own rows
    (data-parallel axis) and heads (tensor-parallel axis) inside a shard_map, so no attention tensor is ever all-gathered.
    q [B,S,H,hd]; K [B,win+S,hd] window keys (previous ring + this chunk); ckv [B,Nc,hd] compressed cache; tk [B,S,k] selected entry ids
    (-1 = none); sink [H]. Returns [B,S,H,hd] float32."""
    import jax
    from csa_flash_attention import csa_flash_attention

    interp = jax.default_backend() != "tpu"
    S = q.shape[1]

    def local(q, K, pos0, ckv, tk, sink):
        def one(args):
            qr, Kr, p0, ckr, tkr = args
            kk = jnp.arange(win + S)[None, :]
            ss = jnp.arange(S)[:, None]
            mask = (
                (kk >= ss + 1) & (kk <= ss + win) & (kk >= win - p0)
            )  # query s sees K[s+1 .. s+win]; ring slots before pos0 are empty
            keys = Kr
            if ckr is not None:
                cm = (
                    jnp.zeros((S, ckr.shape[0]), jnp.int32)
                    .at[jnp.arange(S)[:, None], jnp.maximum(tkr, 0)]
                    .max((tkr >= 0).astype(jnp.int32))
                )
                keys = jnp.concatenate([Kr, ckr.astype(Kr.dtype)], axis=0)
                mask = jnp.concatenate([mask, cm > 0], axis=1)
            return csa_flash_attention(
                qr, keys, mask, sink.astype(jnp.float32), sm_scale=scale, interpret=interp
            )

        if ckv is None:
            return jax.lax.map(lambda t: one((t[0], t[1], t[2], None, None)), (q, K, pos0))
        return jax.lax.map(one, (q, K, pos0, ckv, tk))

    if MESH is None:
        return local(q, K, pos0, ckv, tk, sink)
    from jax.sharding import PartitionSpec as PS

    try:
        from jax import shard_map

        sm = lambda fn, **kw: shard_map(fn, check_vma=False, **kw)
    except ImportError:
        from jax.experimental.shard_map import shard_map

        sm = lambda fn, **kw: shard_map(fn, check_rep=False, **kw)
    if ckv is None:
        f = sm(
            lambda q, K, p0, sk: local(q, K, p0, None, None, sk),
            mesh=MESH,
            in_specs=(PS("dp", None, "tp", None), PS("dp", None, None), PS("dp"), PS("tp")),
            out_specs=PS("dp", None, "tp", None),
        )
        return f(q, K, pos0, sink)
    f = sm(
        local,
        mesh=MESH,
        in_specs=(
            PS("dp", None, "tp", None),
            PS("dp", None, None),
            PS("dp"),
            PS("dp", None, None),
            PS("dp", None, None),
            PS("tp"),
        ),
        out_specs=PS("dp", None, "tp", None),
    )
    return f(q, K, pos0, ckv, tk, sink)


def attention_chunk(P, x, st, pos0, layer, a: Args, shared, rope, active, cap=None, nvalid=None):
    b, S, _ = x.shape
    cos_all, sin_all = rope
    rd, win, r = a.rope_head_dim, a.window_size, a.compress_ratios[layer]
    pos = pos0[:, None] + jnp.arange(S)  # [B,S]
    cos, sin = jnp.take(cos_all, pos, axis=0), jnp.take(sin_all, pos, axis=0)
    x = _c(x, "dp", None, None)
    qr = _c(rms_norm(x @ P["wq_a.weight"].T, P["q_norm.weight"], a.norm_eps), "dp", None, None)
    q = _c(
        _rot_tail((qr @ P["wq_b.weight"].T).reshape(b, S, a.n_heads, a.head_dim), rd, cos, sin),
        "dp",
        None,
        "tp",
        None,
    )

    # ---- window keys: previous ring (ordered) + this chunk
    kv = act_quant_fp8(
        _rot_tail(rms_norm(x @ P["wkv.weight"].T, P["kv_norm.weight"], a.norm_eps), rd, cos, sin)
    )
    ring = st["window"]
    ar = jnp.arange(win)[None, :]
    prev = jnp.take_along_axis(
        ring, ((ar + pos0[:, None]) % win)[:, :, None], axis=1
    )  # per-row roll by -pos0
    K = _c(
        jnp.concatenate([prev, kv.astype(ring.dtype)], axis=1), "dp", None, None
    )  # [B, win+S, hd]
    if nvalid is None:
        new_ring = jnp.take_along_axis(
            K[:, -win:], ((ar - (pos0[:, None] + S)) % win)[:, :, None], axis=1
        )  # roll by +(pos0+S)
    else:
        # only the first nvalid tokens are real: slot j holds the newest real position p < pos0+nvalid with p % win == j,
        # which sits at index p - pos0 + win of K
        nv = nvalid[:, None]
        kidx_slot = nv - 1 + win - ((pos0[:, None] + nv - 1 - ar) % win)
        new_ring = jnp.take_along_axis(K, kidx_slot[:, :, None], axis=1)
    new_ring = jnp.where(active[:, None, None], new_ring, ring)
    blocked = _win_blocked(S, win)
    if blocked:
        # Blocked window: queries in groups of `win`; group j sees K[j*win+1 : j*win+1+2*win] (static slices, no per-query gather), and the
        # band mask picks each query's own window. Avoids materialising [B,S,win,hd] (0.5 GB/layer at S=512) and its all-gather.
        nb = S // win
        Kp = jnp.concatenate([K, jnp.zeros_like(K[:, :1])], axis=1)
        Kw = jnp.stack(
            [Kp[:, j * win + 1 : j * win + 1 + 2 * win] for j in range(nb)], axis=1
        )  # [B,nb,2win,hd]
        si = jnp.arange(S)
        qi = si % win
        jb = si // win
        kk = jnp.arange(2 * win)[None, :]
        band = (kk >= qi[:, None]) & (kk <= qi[:, None] + win - 1)  # [S,2win]
        kg = jb[:, None] * win + 1 + kk  # key index into K
        wvalid = band[None] & (kg[None] >= (win - pos0)[:, None, None])  # [B,S,2win]
    else:
        kidx = jnp.arange(S)[:, None] + 1 + jnp.arange(win)[None, :]  # [S, win] into K
        wvalid = kidx[None] >= (win - pos0)[:, None, None]  # [B,S,win]
        Kw = _c(K[:, kidx], "dp", None, None, None)  # [B,S,win,hd]
    new_st = dict(st, window=new_ring)

    # ---- compressed side
    comp = None
    if r:
        clens = (pos + 1) // r  # compressed entries visible to each query [B,S]
        latent = valid = None
        ist_k = None
        if layer in a.kv_source_layers:
            latent, gpos, valid, upd = _pool_groups(
                P_sub(P, "compressor."), x, st, pos0, S, r, a, active, nvalid
            )
            new_st.update(upd)
            g0 = pos0 // r
        # indexer (needs the pre-RoPE latent) -- or reuse the source's selection
        if layer in a.index_source_layers:
            I = P_sub(P, "indexer.")
            if layer in a.kv_source_layers:
                kk = rms_norm(latent @ I["wk.weight"].T, I["k_norm.weight"], a.norm_eps)
                fc, fs = jnp.take(cos_all, gpos, axis=0), jnp.take(sin_all, gpos, axis=0)
                kk = act_quant_fp4(_rot_tail(kk, rd, fc, fs), 32)
                new_st["index_k"] = _write_cache(
                    st["index_k"], kk.astype(st["index_k"].dtype), g0, valid
                )
                shared["index_k"] = new_st["index_k"]
            qi = _rot_tail(
                (qr @ I["wq_b.weight"].T).reshape(b, S, a.index_n_heads, a.index_head_dim),
                rd,
                cos,
                sin,
            )
            qi = act_quant_fp4(qi, 32)
            wts = (x @ I["weights_proj.weight"].T) * (
                a.index_head_dim**-0.5 * a.index_n_heads**-0.5
            )
            Kv = (
                shared["index_k"] if cap is None else shared["index_k"][:, : cap // r]
            )  # capacity view: entries of the first `cap` tokens
            _select_entries(qi, wts, Kv, clens, a, layer, shared)
        if layer in a.kv_source_layers:
            fc, fs = jnp.take(cos_all, gpos, axis=0), jnp.take(sin_all, gpos, axis=0)
            lat = act_quant_fp4(_rot_tail(latent, rd, fc, fs), 16, e4m3_scale=True)
            new_st["ckv"] = _write_cache(st["ckv"], lat.astype(st["ckv"].dtype), g0, valid)
            shared["ckv"] = new_st["ckv"]
        comp = (shared["ckv"], shared["topk"])

    # ---- attention: window + selected compressed entries, one softmax with a sink
    scale = a.head_dim**-0.5
    if _use_attn_kernel(S):
        ckv_, tk_ = (
            (comp[0] if cap is None else comp[0][:, : cap // r], comp[1])
            if comp is not None
            else (None, None)
        )
        o = _attention_kernel(q, K, pos0, ckv_, tk_, P["attn_sink"], win, scale).astype(x.dtype)
    else:
        qf = q.astype(jnp.float32)
        if blocked:
            sw = (
                jnp.einsum(
                    "bnqhd,bnkd->bnqhk",
                    qf.reshape(b, S // win, win, *qf.shape[2:]),
                    Kw.astype(jnp.float32),
                )
                * scale
            )
            sw = sw.reshape(b, S, qf.shape[2], 2 * win)
        else:
            sw = _c(
                jnp.einsum("bshd,bswd->bshw", qf, Kw.astype(jnp.float32)) * scale,
                "dp",
                None,
                "tp",
                None,
            )
        sw = jnp.where(wvalid[:, :, None, :], sw, NEG)
        scores, vals = [sw], [Kw.astype(jnp.float32)]
        if comp is not None:
            ckv, tk = comp
            cvalid = tk >= 0
            Kc = jnp.take_along_axis(
                ckv[:, None], jnp.maximum(tk, 0)[..., None], axis=2
            )  # [B,S,k,hd]
            Kc = _c(Kc, "dp", None, None, None)
            sc_ = _c(
                jnp.einsum("bshd,bskd->bshk", qf, Kc.astype(jnp.float32)) * scale,
                "dp",
                None,
                "tp",
                None,
            )
            scores.append(jnp.where(cvalid[:, :, None, :], sc_, NEG))
            vals.append(Kc.astype(jnp.float32))
        sc_all = jnp.concatenate(scores, axis=-1)
        any_valid = jnp.isfinite(sc_all).any(axis=-1, keepdims=True)
        mx = jnp.where(any_valid, jnp.max(sc_all, axis=-1, keepdims=True), 0.0)
        p = jnp.exp(sc_all - mx)
        den = p.sum(axis=-1, keepdims=True) + jnp.exp(
            P["attn_sink"].astype(jnp.float32)[None, None, :, None] - mx
        )
        o = jnp.zeros((b, S, a.n_heads, a.head_dim), jnp.float32)
        off = 0
        for vi, v in enumerate(vals):
            if blocked and vi == 0:  # blocked window values [B,nb,2win,hd]
                k = v.shape[2]
                pw = p[..., off : off + k].reshape(b, S // win, win, p.shape[2], k)
                o = o + jnp.einsum("bnqhk,bnkd->bnqhd", pw, v).reshape(b, S, p.shape[2], -1)
            else:
                k = v.shape[2]
                o = o + jnp.einsum("bshk,bskd->bshd", p[..., off : off + k], v)
            off += k
        o = _c(jnp.where(any_valid, o / den, 0.0).astype(x.dtype), "dp", None, "tp", None)
    o = _rot_tail(o, rd, cos, sin, inverse=True)
    g = a.o_groups
    # Quantize per head (blocks lie inside a head), then split the head axis into (group, heads-per-group) instead of merging heads into
    # the last dim: the merge made the compiler all-gather the whole attention output across all chips (~3 ms per layer in prefill).
    o = _c(
        act_quant_fp8(o).reshape(b, S, g, a.n_heads // g, a.head_dim), "dp", None, "tp", None, None
    )
    o = jnp.einsum(
        "bsgjd,grjd->bsgr",
        o,
        P["wo_a.weight"].reshape(g, a.o_lora_rank, a.n_heads // g, a.head_dim),
    )
    return _c(o.reshape(b, S, -1) @ P["wo_b.weight"].T, "dp", None, None), new_st


def block_chunk(
    P, x, st, pos0, layer, pre_mix, a: Args, shared, rope, active, cap=None, nvalid=None
):
    residual = x
    ap, apost, acomb = hc_mixes(x, P["hc_attn_fn"], P["hc_attn_scale"], P["hc_attn_base"], a)
    h = rms_norm(hc_pre(x, pre_mix), P["attn_norm.weight"], a.norm_eps)
    h, st = attention_chunk(
        P_sub(P, "attn."), h, st, pos0, layer, a, shared, rope, active, cap, nvalid
    )
    x = hc_post(h, residual, apost, acomb)
    residual = x
    fpre, fpost, fcomb = hc_mixes(x, P["hc_ffn_fn"], P["hc_ffn_scale"], P["hc_ffn_base"], a)
    h = rms_norm(hc_pre(x, ap), P["ffn_norm.weight"], a.norm_eps)
    if nvalid is None:
        tok_mask = jnp.repeat(active, h.shape[1])
    else:  # padding tokens route to no expert
        tok_mask = (active[:, None] & (jnp.arange(h.shape[1])[None, :] < nvalid[:, None])).reshape(
            -1
        )
    h = moe(P_sub(P, "ffn."), h, a, tok_mask)
    return hc_post(h, residual, fpost, fcomb), fpre, st


def chunk_forward(
    params,
    a: Args,
    tokens,
    pos0,
    states,
    ropes,
    engram_rows=None,
    want_logits=True,
    engram_fn=None,
    active=None,
    cap=None,
    nvalid=None,
):
    """tokens [B,S] int32; pos0 int32 scalar or [B] (traced); active optional [B] bool (rows whose state this step may change).
    engram_rows: optional {layer: [B,S,n_cols,head_dim]} rows gathered on the host.
    nvalid: optional [B] int32 = number of real tokens in each row's chunk (the rest is padding that writes no state and routes to no
    expert); logits are taken at each row's last real token. Requires nvalid even when < S (compression groups are never split).
    """
    bsz = tokens.shape[0]
    pos0 = jnp.broadcast_to(jnp.asarray(pos0, jnp.int32), (bsz,))
    active = jnp.ones((bsz,), bool) if active is None else jnp.asarray(active, bool)
    h = params["embed.weight"][tokens]
    h = jnp.repeat(h[:, :, None, :], a.hc_mult, axis=2)
    pre_mix = jnp.zeros((*h.shape[:2], a.hc_mult), jnp.float32).at[:, :, 0].set(1.0)
    shared, new_states = {}, []
    for layer in range(a.n_layers):
        Pl = P_sub(params, f"layers.{layer}.")
        if engram_rows is not None and layer in a.engram_layer_ids:
            er = engram_rows[layer]
            er = (
                dequant_rows(*er) if isinstance(er, tuple) else er.astype(jnp.float32)
            )  # (uint8 bytes, uint8 exponents) or float rows
            h = engram_gate(h, er, P_sub(Pl, "engram."), a.hc_mult, a.dim, a.norm_eps)
        # ropes: per-layer list (tests) or {"c": compressed-layer table, "b": base table} passed once (at 1M positions per-layer
        # arguments would be counted 40x by the compiler's memory check even when they share buffers)
        rp = (
            ropes[layer]
            if not isinstance(ropes, dict)
            else ropes["c" if a.compress_ratios[layer] else "b"]
        )
        h, pre_mix, st = block_chunk(
            Pl, h, states[layer], pos0, layer, pre_mix, a, shared, rp, active, cap, nvalid
        )
        new_states.append(st)
    if not want_logits:
        return None, new_states
    h = rms_norm(hc_pre(h, pre_mix), params["norm.weight"], a.norm_eps)
    if nvalid is None:
        last = h[:, -1]
    else:
        last = h[jnp.arange(bsz), jnp.clip(nvalid - 1, 0, h.shape[1] - 1)]
    return last.astype(jnp.float32) @ params["head.weight"].astype(jnp.float32).T, new_states
