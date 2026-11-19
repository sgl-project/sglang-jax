"""Generation loop for the V4.1 JAX engine: aligned-chunk prefill, decode, sampling, host Engram rows.

Batch rows are position-aligned and (for now) carry the same sequence replicated B times (B = data-parallel size), so one request
at a time; per-row positions / multiple requests come later. All hosts run this same code deterministically (SPMD).
"""

from __future__ import annotations

import time

import engine as EN
import engram_jax as E
import jax
import jax.numpy as jnp
import model_ref as J
import numpy as np
import sharding as SH
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def decompose(n, sizes):
    """Split n tokens into chunks from `sizes` (descending, each a multiple of the next) so every S>1 chunk starts aligned."""
    out = []
    for s in sizes:
        while n >= s and s > 1:
            out.append(s)
            n -= s
    out += [1] * n
    return out


def decompose_min(n, sizes, min_size):
    """Like decompose but only with chunk sizes >= min_size; returns (chunks, rest) with rest < min_size tokens left for the decode path."""
    out = []
    for s_ in sizes:
        while n >= s_ and s_ >= min_size:
            out.append(s_)
            n -= s_
    return out, n


class Generator:
    def __init__(
        self,
        params,
        a: J.Args,
        mesh,
        batch,
        engram=None,
        sizes=(128, 32, 8, 2),
        max_seq=None,
        eos_id=1,
        log=print,
    ):
        """engram: optional dict(hasher=NgramHasher, tables={layer: object with .rows(ids)->[...,D] float32})."""
        self.params, self.a, self.mesh, self.B, self.sizes, self.eos, self.log = (
            params,
            a,
            mesh,
            batch,
            sizes,
            eos_id,
            log,
        )
        self.engram = engram
        self.rep = NamedSharding(mesh, P())
        self.bsh = NamedSharding(mesh, P("dp", None))
        # only two distinct RoPE tables exist (compressed-layer theta vs base theta): build and upload each once
        # (per-layer copies would cost ~10 GB per chip at 1M positions)
        _rc = {}
        for l in range(a.n_layers):
            key = bool(a.compress_ratios[l])
            if key not in _rc:
                _rc[key] = tuple(jax.device_put(np.asarray(x), self.rep) for x in J.make_rope(a, l))
        self.ropes = (
            {"c": _rc[True], "b": _rc[False]}
            if len(_rc) == 2
            else {"c": next(iter(_rc.values())), "b": next(iter(_rc.values()))}
        )
        EN.MESH = mesh
        self.step = jax.jit(
            EN.chunk_forward,
            static_argnames=("a", "want_logits", "engram_fn", "cap"),
            donate_argnames=("states",),
        )
        self.max_seq = max_seq or a.max_seq_len
        self._sample = jax.jit(self._sample_fn, out_shardings=self.rep)
        self.states = None
        self.reset()
        # Every state-producing jit pins its output to the SAME sharding as freshly initialised states; otherwise XLA picks its own output
        # sharding, the next call sees a different input sharding and recompiles at request time.
        self.state_sh = jax.tree.map(
            lambda _: NamedSharding(mesh, P("dp", None, None)), self.states
        )
        self.step = jax.jit(
            EN.chunk_forward,
            static_argnames=("a", "want_logits", "engram_fn", "cap"),
            donate_argnames=("states",),
            out_shardings=(None, self.state_sh),
        )

    # ---- state / sampling
    def reset(self):
        self.states = None  # free the old caches first (two copies would not fit at 1M context)
        self.states = SH.init_states_sharded(
            lambda: EN.init_states(self.a, self.B, self._act_dtype()), self.mesh
        )
        if self.engram:
            self.engram["hasher"].cache[:] = 0

    def _act_dtype(self):
        return self.params["embed.weight"].dtype

    @staticmethod
    def _sample_fn(logits, key, temperature):
        greedy = jnp.argmax(logits, axis=-1)
        g = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape, minval=1e-9, maxval=1.0)))
        sampled = jnp.argmax(logits / jnp.maximum(temperature, 1e-6) + g, axis=-1)
        return jnp.where(temperature <= 0.0, greedy, sampled).astype(jnp.int32)

    def _dev(self, x, sharding=None):
        """Host numpy -> global device array. In a multi-process run jax.device_put(np, NamedSharding) also verifies, with a cross-host
        collective, that every process passed the same data; every host here computes the same arrays deterministically, so build the
        shards locally and assemble the global array without that check."""
        sharding = sharding or self.rep
        if jax.process_count() == 1:
            return jax.device_put(x, sharding)
        x = np.asarray(x)
        amap = sharding.addressable_devices_indices_map(x.shape)
        return jax.make_array_from_single_device_arrays(
            x.shape, sharding, [jax.device_put(x[idx], d) for d, idx in amap.items()]
        )

    def _rows(self, tokens_np, pos):
        if not self.engram:
            return None
        hid = self.engram["hasher"](tokens_np, pos)  # [B,S,n_layers,n_cols]
        return {
            layer: self._dev(self.engram["tables"][layer].rows(hid[:, :, i, :]).astype(np.float32))
            for i, layer in enumerate(self.a.engram_layer_ids)
        }

    def _run(self, tokens_np, pos):
        toks = self._dev(np.asarray(tokens_np, np.int32), self.bsh)
        logits, self.states = self.step(
            self.params,
            self.a,
            toks,
            self._dev(np.int32(pos)),
            self.states,
            self.ropes,
            self._rows(np.asarray(tokens_np), pos),
        )
        return logits

    # ---- generation
    def generate(self, prompt_ids, max_new, temperature=0.0, seed=0, stop_ids=None):
        """Returns the list of generated token ids (not including the prompt). Stops at EOS / stop ids / max_new."""
        stop = set(stop_ids or []) | {self.eos}
        n = len(prompt_ids)
        assert (
            n + max_new <= self.max_seq
        ), f"prompt {n} + max_new {max_new} exceeds max_seq {self.max_seq}"
        self.reset()
        prompt = np.asarray(prompt_ids, np.int32)
        pos, logits, t0 = 0, None, time.time()
        for S in decompose(n, self.sizes):
            logits = self._run(np.broadcast_to(prompt[pos : pos + S], (self.B, S)), pos)
            pos += S
        key = jax.random.PRNGKey(seed)
        out, t_first = [], None
        for i in range(max_new):
            key, sub = jax.random.split(key)
            tok = int(np.asarray(self._sample(logits, sub, self._dev(np.float32(temperature))))[0])
            if t_first is None:
                t_first = time.time() - t0
            out.append(tok)
            if tok in stop:
                break
            logits = self._run(np.full((self.B, 1), tok, np.int32), pos)
            pos += 1
        self.last_stats = {
            "prompt_tokens": n,
            "new_tokens": len(out),
            "ttft_s": t_first,
            "total_s": time.time() - t0,
        }
        return out

    def warmup(self, extra_sizes=()):
        """Compile every chunk shape once (prefill sizes + decode)."""
        t = time.time()
        for S in tuple(self.sizes) + (1,) + tuple(extra_sizes):
            self.reset()
            self._run(np.zeros((self.B, S), np.int32), 0)
            self.log(f"  warmup S={S} done at +{time.time()-t:.0f}s")
        jax.block_until_ready(
            self._sample(
                jnp.zeros((self.B, self.a.vocab_size), jnp.float32),
                jax.random.PRNGKey(0),
                jnp.float32(0.0),
            )
        )
        self.reset()


# ------------------------------------------------------------------------------------------ multi-request batching


TOPK_MAX = 64


def _sample_rows(logits, seeds, counters, temps, top_ps, top_ks):
    """Per-row sampling. logits [B,V]; seeds/counters int32 [B]; temps f32 [B] (<=0 -> greedy); top_ps f32 [B] (1 = off); top_ks int32 [B]
    (0 = off). Rows without top-p/top-k sample exactly from the full softmax (Gumbel-max over the whole vocabulary); rows with top-p or
    top-k sample inside the top-64 logits (cheap; top-p is applied within those 64, an approximation when the nucleus is wider). Each
    kind of noise is only generated if some row needs it, so an all-greedy batch pays for no random numbers.
    """
    B, V = logits.shape
    greedy = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    sampling = temps > 0.0
    restricted = (top_ps < 1.0) | (top_ks > 0)
    keys = jax.vmap(lambda s_, c_: jax.random.fold_in(jax.random.PRNGKey(s_), c_))(seeds, counters)
    inv_t = 1.0 / jnp.maximum(temps, 1e-6)

    def full_path(_):
        g = jax.vmap(
            lambda k: -jnp.log(-jnp.log(jax.random.uniform(k, (V,), minval=1e-9, maxval=1.0)))
        )(keys)
        return jnp.argmax(logits * inv_t[:, None] + g, axis=-1).astype(jnp.int32)

    def restricted_path(_):
        vals, idx = jax.lax.top_k(logits, TOPK_MAX)  # sorted descending
        scaled = vals * inv_t[:, None]
        probs = jax.nn.softmax(scaled, axis=-1)
        rank = jnp.arange(TOPK_MAX)[None, :]
        k_eff = jnp.where(top_ks > 0, jnp.minimum(top_ks, TOPK_MAX), TOPK_MAX)
        keep = (rank < k_eff[:, None]) & (
            ((jnp.cumsum(probs, axis=-1) - probs) < top_ps[:, None]) | (rank == 0)
        )
        g = jax.vmap(
            lambda k: -jnp.log(
                -jnp.log(
                    jax.random.uniform(
                        jax.random.fold_in(k, 7), (TOPK_MAX,), minval=1e-9, maxval=1.0
                    )
                )
            )
        )(keys)
        pick = jnp.argmax(jnp.where(keep, scaled, -jnp.inf) + g, axis=-1)
        return jnp.take_along_axis(idx, pick[:, None], axis=1)[:, 0].astype(jnp.int32)

    zeros = lambda _: jnp.zeros((B,), jnp.int32)
    full = jax.lax.cond(jnp.any(sampling & ~restricted), full_path, zeros, None)
    restr = jax.lax.cond(jnp.any(sampling & restricted), restricted_path, zeros, None)
    return jnp.where(~sampling, greedy, jnp.where(restricted, restr, full))


class PrefixCache:
    """Reuse of prompt prefixes across requests. A snapshot holds a row's attention state at a prompt position P (even, <= pmax): the
    window ring plus the first P/ratio entries of the compressed/indexer caches (partial compressor state is not needed at an even P).
    Snapshots live in a replicated pool of `slots` entries, keyed by a hash of the exact token prefix, LRU-evicted. All hosts take
    the same decisions (the scheduler is deterministic), so the pool stays identical everywhere."""

    def __init__(self, g, slots=32, pmax=8192):
        import hashlib
        from collections import OrderedDict

        self.g, self.slots, self.pmax, self._hl = g, slots, pmax, hashlib
        self.map = OrderedDict()  # digest -> (slot, P)
        self.free = list(range(slots))
        self.lens = {}  # P -> number of stored snapshots with that length
        self.hits = self.misses = self.saved_tokens = 0
        a, rep = g.a, g.rep
        tmpl = jax.eval_shape(lambda: EN.init_states(a, 1, g._act_dtype()))
        self._spec = []  # per layer: {key: (kind, Tp)}
        pool = []
        for l, st in enumerate(tmpl):
            r = a.compress_ratios[l]
            d, spec = {}, {}
            for k, v in st.items():
                if k == "window":
                    spec[k] = ("full", None)
                    d[k] = jnp.zeros((slots, *v.shape[1:]), v.dtype)
                elif k in ("ckv", "index_k"):
                    tp = min(pmax // r, v.shape[1])
                    spec[k] = ("head", tp)
                    d[k] = jnp.zeros((slots, tp, v.shape[2]), v.dtype)
            self._spec.append(spec)
            pool.append(d)
        self.pool = jax.jit(lambda: pool, out_shardings=jax.tree.map(lambda _: rep, pool))()
        spec = self._spec

        def snap(states, pool, row, slot):
            out = []
            for l, st in enumerate(states):
                d = {}
                for k, (kind, tp) in spec[l].items():
                    src = st[k][row] if kind == "full" else st[k][row, :tp]
                    d[k] = pool[l][k].at[slot].set(src.astype(pool[l][k].dtype))
                out.append(d)
            return out

        def rest(states, pool, row, slot):
            out = []
            for l, st in enumerate(states):
                new = dict(st)
                for k, (kind, tp) in spec[l].items():
                    new[k] = (
                        st[k].at[row].set(pool[l][k][slot])
                        if kind == "full"
                        else st[k].at[row, :tp].set(pool[l][k][slot])
                    )
                out.append(new)
            return out

        self._snap = jax.jit(
            snap, donate_argnums=(1,), out_shardings=jax.tree.map(lambda _: rep, pool)
        )
        self._rest = jax.jit(rest, donate_argnums=(0,), out_shardings=g.state_sh)

    def _digest(self, prompt, p):
        return self._hl.blake2b(np.ascontiguousarray(prompt[:p]).tobytes(), digest_size=16).digest()

    def lookup(self, prompt, limit):
        """Longest cached prefix length P <= limit of this prompt (or None)."""
        for p in sorted((x for x in self.lens if x <= limit), reverse=True)[:16]:
            dg = self._digest(prompt, p)
            if dg in self.map:
                self.map.move_to_end(dg)
                return self.map[dg]
        return None

    def restore(self, row, slot):
        self.g.states = self._rest(self.g.states, self.pool, np.int32(row), np.int32(slot))

    def insert(self, row, prompt, p):
        if p <= 0 or p > self.pmax or p % 2:
            return
        dg = self._digest(prompt, p)
        if dg in self.map:
            self.map.move_to_end(dg)
            return
        if not self.free:  # evict the least recently used snapshot
            old_dg, (slot, old_p) = self.map.popitem(last=False)
            self.lens[old_p] -= 1
            if not self.lens[old_p]:
                del self.lens[old_p]
            self.free.append(slot)
        slot = self.free.pop()
        self.pool = self._snap(self.g.states, self.pool, np.int32(row), np.int32(slot))
        self.map[dg] = (slot, p)
        self.lens[p] = self.lens.get(p, 0) + 1


class BatchScheduler:
    """Continuous batching, SGLang-style: `B` = max running requests (allocated once), decode padded to batch-size buckets.

    Rows are split into dp blocks of R = B/dp (block g owns rows g*R .. g*R+R-1, matching the dp sharding of the caches).
    Live requests are kept packed at the low slots of each block (a finished row is filled by moving that block's last live
    row into it), so a decode step only runs the first k slots of every block, k = smallest bucket >= max live per block.
    Like SGLang's --precompile-bs-paddings, each bucket (total batch dp*k) is compiled once at startup; `max_live` is a runtime
    cap (<= B) on concurrent requests.

    Deterministic given the same sequence of add()/set_max_live() calls, so every host runs its own copy in lockstep.
    A new request is prefilled alone (B=1 replicated call on its row's state when B > dp) while other rows hold.
    """

    def __init__(self, gen: Generator, bs_paddings=None, ladder=None):
        self.g = gen
        # context-capacity ladder (like SGL-JAX's SGLANG_JAX_PRECOMPILE_CONTEXT_LADDER): each step attends over only the first
        # `cap` cache entries, cap = smallest rung >= the longest live position + chunk. Rungs compile lazily on first use
        # (only the first rung is warmed at startup).
        mx = gen.max_seq
        rungs = sorted({r for r in (ladder or [16384, 65536, 262144, 1048576]) if r < mx} | {mx})
        self.ladder = rungs
        self._seen_caps = {rungs[0]}
        self.B = gen.B
        self.dp = gen.mesh.shape["dp"]
        assert (
            self.B % self.dp == 0
        ), f"max running requests {self.B} must be a multiple of dp {self.dp}"
        self.R = self.B // self.dp
        # per-dp bucket sizes from total-batch paddings (SGLang default: powers of two), always including R
        tot = bs_paddings or [1 << i for i in range(0, 12)]
        ks = sorted(
            {t // self.dp for t in tot if t % self.dp == 0 and 1 <= t // self.dp <= self.R}
            | {self.R}
        )
        self.buckets = ks
        self.rows = [None] * self.B
        self.nlive = [0] * self.dp
        self.pending = []
        self.max_live = self.B
        import os as _os

        self.tail_min_rows = int(
            _os.environ.get("V41_TAIL_MIN_ROWS", "1000000")
        )  # use tail-by-decode when at least this many rows are decoding
        self.stats_on = _os.environ.get("V41_STATS", "1") == "1"
        self.on_token = (
            None  # optional callback(rid, token_id) fired for every sampled token (streaming)
        )
        self.pcache = None
        if _os.environ.get("V41_PREFIX_CACHE", "0") == "1":
            self.pcache = PrefixCache(
                gen,
                slots=int(_os.environ.get("V41_PREFIX_SLOTS", "32")),
                pmax=int(_os.environ.get("V41_PREFIX_MAX", "8192")),
            )
        self.pc_granule = int(
            _os.environ.get("V41_PREFIX_GRANULE", "512")
        )  # also snapshot at every multiple of this prompt position
        # prefill accumulation (lockstep-deterministic: counted in scheduling rounds, never wall-clock): hold prompts back until enough are
        # waiting to fill the prefill call, or the oldest has waited pf_max_wait rounds, or nothing is decoding. 0 = prefill immediately.
        from concurrent.futures import ThreadPoolExecutor

        self._pool = ThreadPoolExecutor(8)
        self._host = dict(n=0, hash=0.0, gather=0.0, put=0.0)
        self.pf_min_rows = int(_os.environ.get("V41_PF_MIN_ROWS", "0"))
        self.pf_max_wait = int(_os.environ.get("V41_PF_MAX_WAIT", "0"))
        self.pf_m = sorted(
            {min(int(x), self.R) for x in _os.environ.get("V41_PREFILL_M", "1,4").split(",")}
        )  # rows per block per prefill call
        self._pfs = (
            {}
        )  # (chunk size, rows per block) -> [calls, seconds, requests advanced]; logged with the round stats
        self._st = {
            "start": 0.0,
            "prefill": 0.0,
            "dec_prep": 0.0,
            "dec_call": 0.0,
            "dec_wait": 0.0,
            "post": 0.0,
            "rounds": 0,
            "pf_rounds": 0,
            "rows": 0,
        }
        self.tail_min_size = int(
            _os.environ.get("V41_TAIL_MIN_SIZE", "32")
        )  # prompt remainders shorter than this go through decode steps
        self._sample = jax.jit(_sample_rows, out_shardings=gen.rep)
        self.steps = 0
        self.single = self.B > self.dp
        self._get_row = jax.jit(
            lambda sts, i: jax.tree.map(
                lambda x: jax.lax.dynamic_index_in_dim(x, i, 0, keepdims=True), sts
            ),
            out_shardings=gen.rep,
        )
        self._put_row = jax.jit(
            lambda sts, row, i: jax.tree.map(
                lambda x, r: jax.lax.dynamic_update_index_in_dim(x, r[0].astype(x.dtype), i, 0),
                sts,
                row,
            ),
            donate_argnums=(0,),
        )
        self._move = jax.jit(self._row_ops()[4], donate_argnums=(0,), out_shardings=gen.state_sh)
        self._dec = {k: self._make_decode(k) for k in self.buckets}
        self._pre = self._make_prefill()

    def _row_ops(self):
        """Row operations on the dp-sharded state tree as shard_map bodies: every data-parallel group indexes ITS OWN rows locally. With
        plain global indexing (vmapped gather/scatter, dynamic update on the sharded axis) the compiler partitioned the row axis over
        the tensor-parallel chips and all-gathered whole blocks of every cache (profiled: 67% of prefill time, ~12 ms per all-gather).
        """
        g, dp, R = self.g, self.dp, self.R
        from jax.sharding import PartitionSpec as PS

        try:
            from jax import shard_map
        except ImportError:  # older jax
            from jax.experimental.shard_map import shard_map
        mesh = g.mesh
        S3 = lambda tree: jax.tree.map(lambda _: PS("dp", None, None), tree)

        def sm(fn, in_specs, out_specs):
            kw = dict(mesh=mesh, in_specs=in_specs, out_specs=out_specs)
            try:
                return shard_map(fn, check_vma=False, **kw)
            except TypeError:
                return shard_map(fn, check_rep=False, **kw)

        def gather_rows(states, jidx):  # -> [dp*m, ...] (m rows per block, block-major)
            m = jidx.shape[1]

            def body(st, j):
                return jax.tree.map(
                    lambda x: jnp.stack(
                        [
                            jax.lax.dynamic_index_in_dim(x, j[0, i], 0, keepdims=False)
                            for i in range(m)
                        ]
                    ),
                    st,
                )

            return sm(body, (S3(states), PS("dp", None)), S3(states))(states, jidx)

        def scatter_rows(states, new, jidx):  # distinct indices within a block
            m = jidx.shape[1]

            def body(st, nw, j):
                def upd(x, n):
                    for i in range(m):
                        x = jax.lax.dynamic_update_index_in_dim(x, n[i].astype(x.dtype), j[0, i], 0)
                    return x

                return jax.tree.map(upd, st, nw)

            return sm(body, (S3(states), S3(states), PS("dp", None)), S3(states))(states, new, jidx)

        def slice_rows(states, k):  # first k rows of every block -> [dp*k, ...]
            return sm(lambda st: jax.tree.map(lambda x: x[:k], st), (S3(states),), S3(states))(
                states
            )

        def set_rows(states, new, k):
            return sm(
                lambda st, nw: jax.tree.map(lambda x, n: x.at[:k].set(n.astype(x.dtype)), st, nw),
                (S3(states), S3(states)),
                S3(states),
            )(states, new)

        def move(states, src, dst):  # copy global row src -> dst (same block by construction)
            def body(st, s_, d_):
                mine = jax.lax.axis_index("dp")
                ok = (s_ // R == mine) & (d_ // R == mine)

                def upd(x):
                    row = jax.lax.dynamic_index_in_dim(x, s_ % R, 0, keepdims=False)
                    cur = jax.lax.dynamic_index_in_dim(x, d_ % R, 0, keepdims=False)
                    return jax.lax.dynamic_update_index_in_dim(
                        x, jnp.where(ok, row, cur), d_ % R, 0
                    )

                return jax.tree.map(upd, st)

            return sm(body, (S3(states), PS(), PS()), S3(states))(states, src, dst)

        return gather_rows, scatter_rows, slice_rows, set_rows, move

    def _make_prefill(self):
        """Prefill step for m rows per data-parallel block at once (m = jidx.shape[1], each row its own request/position): gathers each
        block's rows (batched gather along the dp-sharded axis, no collective), runs the chunk for dp*m rows, scatters the states back.
        The m indices of a block MUST be distinct (inactive filler rows are written back unchanged).
        """
        g, dp, R = self.g, self.dp, self.R
        import engine as EN_

        gather_rows, scatter_rows, _, _, _ = self._row_ops()

        def f(params, toks, pos, states, ropes, rows, act, jidx, nvalid, cap):
            sub = gather_rows(states, jidx)
            lg, new = EN_.chunk_forward(
                params, g.a, toks, pos, sub, ropes, rows, active=act, cap=cap, nvalid=nvalid
            )
            return lg, scatter_rows(states, new, jidx)

        return jax.jit(
            f, donate_argnums=(3,), static_argnames=("cap",), out_shardings=(None, g.state_sh)
        )

    def _call_prefill(self, idx, toks_full, pos_full, act_full, nvalid_full, cap):
        """idx: flat list of dp*m global row ids, block-major (m per block). nvalid_full [B]: real tokens per row in this chunk."""
        g = self.g
        rows = self._engram_rows(toks_full, pos_full, act_full, idx)
        jidx = (np.array(idx, np.int32) % self.R).reshape(self.dp, -1)
        logits, g.states = self._pre(
            g.params,
            g._dev(np.asarray(toks_full[idx], np.int32), g.bsh),
            g._dev(np.asarray(pos_full[idx], np.int32)),
            g.states,
            g.ropes,
            rows,
            g._dev(np.asarray(act_full[idx], bool)),
            g._dev(jidx),
            g._dev(np.asarray(nvalid_full[idx], np.int32)),
            cap,
        )
        self.steps += 1
        return logits  # [dp*m, V]

    def _cap(self, need):
        c = next((r for r in self.ladder if r >= need), self.ladder[-1])
        if c not in self._seen_caps:
            self._seen_caps.add(c)
            self.g.log(
                f"context capacity {c} first used: its shapes compile lazily now (~1 min each)"
            )
        return c

    def _make_decode(self, k):
        g, dp, R = self.g, self.dp, self.R
        import engine as EN_

        if k == R:
            return lambda params, toks, pos, states, ropes, rows, act, cap: g.step(
                params, g.a, toks, pos, states, ropes, rows, active=act, cap=cap
            )

        _, _, slice_rows, set_rows, _ = self._row_ops()

        def f(params, toks, pos, states, ropes, rows, act, cap):
            sub = slice_rows(states, k)
            lg, new = EN_.chunk_forward(
                params, g.a, toks, pos, sub, ropes, rows, active=act, cap=cap
            )
            return lg, set_rows(states, new, k)

        return jax.jit(
            f, donate_argnums=(3,), static_argnames=("cap",), out_shardings=(None, g.state_sh)
        )

    def add(self, rid, prompt_ids, max_new, temperature=0.0, seed=0, top_p=1.0, top_k=0):
        self.pending.append(
            dict(
                rid=rid,
                ids=list(prompt_ids),
                max_new=int(max_new),
                temp=float(temperature),
                seed=int(seed),
                top_p=float(top_p),
                top_k=int(top_k),
                t0=time.time(),
            )
        )  # timings include queueing

    def set_max_live(self, n):
        """Change the concurrency cap at runtime (no recompile). Rows already running finish normally."""
        self.max_live = max(1, min(int(n), self.B))

    def set_pf_policy(self, min_rows, max_wait):
        self.pf_min_rows = max(0, int(min_rows))
        self.pf_max_wait = max(0, int(max_wait))

    def busy(self):
        return bool(self.pending) or any(r is not None for r in self.rows)

    def live_count(self):
        return sum(self.nlive)

    # ---- device calls
    def _engram_rows(self, tokens_full, pos_full, act_full, idx):
        g = self.g
        if not g.engram:
            return None
        T = time.perf_counter
        t0 = T()
        hid = g.engram["hasher"](np.asarray(tokens_full), pos_full, act_full)[idx]
        t1 = T()
        pool = self._pool
        if hid.size >= 4096:  # big prefill gathers: threaded; tiny ones (decode) are faster inline
            raws = [
                (layer, g.engram["tables"][layer].raw_threaded(hid[:, :, i, :], pool))
                for i, layer in enumerate(g.a.engram_layer_ids)
            ]
        else:
            raws = [
                (layer, g.engram["tables"][layer].raw(hid[:, :, i, :]))
                for i, layer in enumerate(g.a.engram_layer_ids)
            ]
        t2 = T()
        out = {
            layer: (g._dev(w8, g.bsh), g._dev(s8, g.bsh)) for layer, (w8, s8) in raws
        }  # rows sharded over dp: 4x less host->device traffic
        t3 = T()
        if self.stats_on:
            h = self._host
            h["n"] += 1
            h["hash"] += t1 - t0
            h["gather"] += t2 - t1
            h["put"] += t3 - t2
        return out

    def _call_single(self, i, tokens_full, pos_full, act, cap=None):
        """Prefill chunk for row i only, as a B=1 replicated call on row i's state."""
        g = self.g
        rows = self._engram_rows(tokens_full, pos_full, act, [i])
        st1 = self._get_row(g.states, np.int32(i))
        logits, st1 = g.step(
            g.params,
            g.a,
            g._dev(np.asarray(tokens_full[i : i + 1], np.int32)),
            g._dev(np.asarray(pos_full[i : i + 1], np.int32)),
            st1,
            g.ropes,
            rows,
            active=g._dev(np.ones(1, bool)),
            cap=cap,
        )
        g.states = self._put_row(g.states, st1, np.int32(i))
        self.steps += 1
        return logits  # [1, V]

    def _call_full(self, tokens, pos, active, cap=None):
        g = self.g
        rows = self._engram_rows(tokens, pos, active, slice(None))
        logits, g.states = g.step(
            g.params,
            g.a,
            g._dev(np.asarray(tokens, np.int32), g.bsh),
            g._dev(np.asarray(pos, np.int32)),
            g.states,
            g.ropes,
            rows,
            active=g._dev(np.asarray(active, bool)),
            cap=cap,
        )
        self.steps += 1
        return logits

    def _call_decode(self, k, tokens_full, pos_full, act_full, cap=None):
        g = self.g
        idx = [b * self.R + j for b in range(self.dp) for j in range(k)]
        rows = self._engram_rows(tokens_full, pos_full, act_full, idx)
        logits, g.states = self._dec[k](
            g.params,
            g._dev(np.asarray(tokens_full[idx], np.int32), g.bsh),
            g._dev(np.asarray(pos_full[idx], np.int32)),
            g.states,
            g.ropes,
            rows,
            g._dev(np.asarray(act_full[idx], bool)),
            cap,
        )
        self.steps += 1
        return logits, idx

    def _sample_rows(self, logits, idx):
        rs = [self.rows[i] for i in idx]
        seeds = np.array([r["seed"] if r else 0 for r in rs], np.int32)
        cnt = np.array([len(r["out"]) if r else 0 for r in rs], np.int32)
        temps = np.array([r["temp"] if r else 0.0 for r in rs], np.float32)
        tps = np.array([r["top_p"] if r else 1.0 for r in rs], np.float32)
        tks = np.array([r["top_k"] if r else 0 for r in rs], np.int32)
        return np.asarray(
            self._sample(
                logits,
                self.g._dev(seeds),
                self.g._dev(cnt),
                self.g._dev(temps),
                self.g._dev(tps),
                self.g._dev(tks),
            )
        )

    # ---- row bookkeeping (packed slots per dp block)
    def _free(self, i):
        b = i // self.R
        last = b * self.R + self.nlive[b] - 1
        if last != i:  # keep the block packed: move its last live row into the hole
            self.g.states = self._move(self.g.states, np.int32(last), np.int32(i))
            if self.g.engram:
                h = self.g.engram["hasher"].cache
                h[i] = h[last]
            self.rows[i] = self.rows[last]
        self.rows[last] = None
        self.nlive[b] -= 1

    def _finish(self, i, done):
        r = self.rows[i]
        done.append(
            (
                r["rid"],
                r["out"],
                {
                    "prompt_tokens": len(r["ids"]),
                    "new_tokens": len(r["out"]),
                    "ttft_s": r["ttft"],
                    "total_s": time.time() - r["t0"],
                },
            )
        )

    def _take(self, i, tok):
        """Append a sampled token; returns True if the request is finished."""
        r = self.rows[i]
        if r["ttft"] is None:
            r["ttft"] = time.time() - r["t0"]
        r["out"].append(int(tok))
        if self.on_token is not None:
            self.on_token(r["rid"], int(tok))
        return int(tok) == self.g.eos or len(r["out"]) >= r["max_new"]

    # ---- scheduling
    def _start(self, req):
        """Reserve a slot for a new request. Its prompt is prefilled in padded fixed-size chunks (all prefilling rows share each call);
        an odd final token goes through the decode path (compression groups are 2 tokens, a chunk never ends mid-group).
        """
        b = int(np.argmin(self.nlive))
        i = b * self.R + self.nlive[b]
        self.nlive[b] += 1
        n = len(req["ids"])
        assert n + req["max_new"] <= self.g.max_seq
        req.update(
            out=[],
            pos=0,
            wait=0,
            ttft=None,
            prompt=np.asarray(req["ids"], np.int32),
            pf_end=n - (n % 2),
        )
        self.rows[i] = req
        pc = self.pcache
        if pc is not None:
            hit = pc.lookup(
                req["prompt"], req["pf_end"] - 2
            )  # keep >= 2 prompt tokens to prefill: the first output token needs logits
            if hit is not None:
                slot, p = hit
                pc.restore(i, slot)
                req["pos"] = p
                pc.hits += 1
                pc.saved_tokens += p
                if self.g.engram:  # the n-gram hasher needs the history of this row
                    hs = self.g.engram["hasher"]
                    hs.cache[i, :p] = hs.token_map[req["prompt"][:p]]
            else:
                pc.misses += 1

    def _prefill_batch(self, done):
        """One prefill call advancing EVERY waiting prompt (up to m rows per data-parallel block): the call uses a single chunk size S
        (smallest configured size >= the largest remaining prompt part, capped at the biggest size) and each row advances by its own
        number of real tokens (padding writes no state). m is the smallest option that fits the waiting rows.
        """
        pre = [i for i, r in enumerate(self.rows) if r is not None and r["pos"] < r["pf_end"]]
        if not pre:
            return False
        per_block = [[] for _ in range(self.dp)]
        for i in pre:
            per_block[i // self.R].append(i)
        need = max(len(b) for b in per_block)
        if self.pf_min_rows > 0:
            decoding = any(r is not None and r["pos"] >= r["pf_end"] for r in self.rows)
            if (
                decoding
                and len(pre) < self.pf_min_rows
                and need < self.pf_m[-1]
                and max(self.rows[i]["wait"] for i in pre) < self.pf_max_wait
            ):
                for i in pre:
                    self.rows[i]["wait"] += 1
                return False
        m = next((x for x in self.pf_m if x >= need), self.pf_m[-1])
        chosen_by_block = [b[:m] for b in per_block]
        chosen = [i for b in chosen_by_block for i in b]
        sizes = sorted(self.g.sizes)
        max_rem = max(self.rows[i]["pf_end"] - self.rows[i]["pos"] for i in chosen)
        S = next((x for x in sizes if x >= max_rem), sizes[-1])
        toks_full = np.zeros((self.B, S), np.int32)
        pos_full = np.zeros(self.B, np.int32)
        act_full = np.zeros(self.B, bool)
        nvalid_full = np.ones(self.B, np.int32)
        idx, adv = [], {}
        for b in range(self.dp):
            take = chosen_by_block[b]
            fill = [
                b * self.R + j for j in range(self.R) if b * self.R + j not in take
            ]  # distinct inactive filler slots
            sel = take + fill[: m - len(take)]
            for i in take:
                r = self.rows[i]
                n_adv = min(S, r["pf_end"] - r["pos"])  # even: pos and pf_end are even, S is even
                toks_full[i, :n_adv] = r["prompt"][r["pos"] : r["pos"] + n_adv]
                pos_full[i] = r["pos"]
                act_full[i] = True
                nvalid_full[i] = n_adv
                adv[i] = n_adv
            idx.extend(sel)
        cap = self._cap(max(self.rows[i]["pos"] + S for i in chosen))
        t_pf = time.perf_counter()
        logits = self._call_prefill(idx, toks_full, pos_full, act_full, nvalid_full, cap)
        if self.stats_on:
            jax.block_until_ready(
                logits
            )  # stats mode: attribute device time of the prefill call to the prefill bucket
            ent = self._pfs.setdefault((S, m), [0, 0.0, 0])
            ent[0] += 1
            ent[1] += time.perf_counter() - t_pf
            ent[2] += len(chosen)
        where = {i: n for n, i in enumerate(idx)}
        fin_n, fin_i = [], []
        for i in chosen:
            r = self.rows[i]
            r["pos"] += adv[i]
            if (
                self.pcache is not None
                and r["pos"] % 2 == 0
                and (r["pos"] % self.pc_granule == 0 or r["pos"] == r["pf_end"])
            ):
                self.pcache.insert(i, r["prompt"], r["pos"])
            if r["pos"] == len(
                r["ids"]
            ):  # even-length prompt fully consumed by the chunk: its logits give the first token
                fin_n.append(where[i])
                fin_i.append(i)
        if fin_i:
            toks = self._sample_rows(logits[np.array(fin_n)], fin_i)
            gone = [i for i, t in zip(fin_i, toks) if self._take(i, t)]
            for i in sorted(gone, reverse=True):
                self._finish(i, done)
                self._free(i)
        return True

    def step(self):
        """One scheduling round: reserve slots for new requests, run at most ONE batched prefill call, then one decode step for every
        row whose prompt is complete (so decoding is never stalled for a whole prompt). Returns finished [(rid, tokens, stats)].
        """
        T = time.perf_counter
        st = self._st
        done = []
        t0 = T()
        while self.pending and self.live_count() < min(self.max_live, self.B):
            self._start(self.pending.pop(0))
        t1 = T()
        pf = self._prefill_batch(done)
        t2 = T()
        toks = np.zeros((self.B, 1), np.int32)
        pos = np.zeros(self.B, np.int32)
        act = np.zeros(self.B, bool)
        for i, r in enumerate(self.rows):
            if r is not None and r["pos"] >= r["pf_end"]:
                # odd final prompt token still to feed through the decode path, or the last sampled token
                toks[i, 0] = r["prompt"][r["pos"]] if r["pos"] < len(r["ids"]) else r["out"][-1]
                pos[i] = r["pos"]
                act[i] = True
        if not act.any():
            return done
        kmax = max(self.nlive)
        k = next(x for x in self.buckets if x >= kmax)
        t3 = T()
        logits, idx = self._call_decode(k, toks, pos, act, self._cap(int(pos.max()) + 1))
        t4 = T()
        tk = self._sample_rows(logits, idx)
        t5 = T()
        fin = []
        for j, i in enumerate(idx):
            if act[i]:
                r = self.rows[i]
                r["pos"] += 1
                if r["pos"] < len(
                    r["ids"]
                ):  # still feeding the prompt tail: this step's logits are not used
                    continue
                if self._take(i, tk[j]):
                    fin.append(i)
        for i in sorted(
            fin, reverse=True
        ):  # free high slots first so the row moved into a hole is always live
            self._finish(i, done)
            self._free(i)
        t6 = T()
        if self.stats_on:
            for key, v in (
                ("start", t1 - t0),
                ("prefill", t2 - t1),
                ("dec_prep", t3 - t2),
                ("dec_call", t4 - t3),
                ("dec_wait", t5 - t4),
                ("post", t6 - t5),
            ):
                st[key] += v
            st["rounds"] += 1
            st["pf_rounds"] += int(pf)
            st["rows"] += int(act.sum())
            if st["rounds"] >= 100:
                n = st["rounds"]
                self.g.log(
                    "sched stats /round(ms): "
                    + " ".join(
                        f"{k}={1000*st[k]/n:.1f}"
                        for k in ("start", "prefill", "dec_prep", "dec_call", "dec_wait", "post")
                    )
                    + f" | prefill rounds {100*st['pf_rounds']/n:.0f}% | avg decode rows {st['rows']/n:.1f} | bucket k={k}"
                )
                for key in st:
                    st[key] = 0 if key in ("rounds", "pf_rounds", "rows") else 0.0
                if self.pcache is not None:
                    pc = self.pcache
                    self.g.log(
                        f"prefix cache: {len(pc.map)}/{pc.slots} snapshots, hits {pc.hits} misses {pc.misses}, prompt tokens saved {pc.saved_tokens}"
                    )
                if self._pfs:
                    self.g.log(
                        "prefill calls (S,m): "
                        + " ".join(
                            f"({S_},{m_}) n={c} {1000*t_/c:.0f}ms/call {r_/c:.1f}req/call"
                            for (S_, m_), (c, t_, r_) in sorted(self._pfs.items(), reverse=True)
                        )
                    )
                    self._pfs = {}
                h = self._host
                if h["n"]:
                    self.g.log(
                        f"prefill host time per call (ms): hash={1000*h['hash']/h['n']:.1f} gather={1000*h['gather']/h['n']:.1f} put={1000*h['put']/h['n']:.1f} (n={h['n']})"
                    )
                    for key in h:
                        h[key] = 0 if key == "n" else 0.0
        return done

    def warmup_flow(self):
        """Run synthetic requests through the real scheduler path (prefill of every chunk size, every decode batch size, slot moves)
        so that nothing compiles at request time. Leaves the scheduler idle and the state cleared.
        """
        rng = np.random.default_rng(0)
        big = min(
            2 * sum(self.g.sizes) + 1, self.g.max_seq // 2
        )  # = 1023 for the default sizes: touches every chunk size + the 1-token tail
        self.add(-1, rng.integers(3, 1000, big).tolist(), 3)
        for i in range(self.B + 1):
            self.add(-2 - i, rng.integers(3, 1000, 10).tolist(), 2 + i % 3)
        t = time.time()
        while self.busy():
            self.step()
        self.g.log(f"  warm flow done at +{time.time()-t:.0f}s ({self.steps} device calls so far)")
        self.g.reset()

    def warmup(self):
        """Compile prefill shapes, every decode bucket, sampling and the row-move op."""
        t = time.time()
        zB = lambda S: (
            np.zeros((self.B, S), np.int32),
            np.zeros(self.B, np.int32),
            np.zeros(self.B, bool),
        )
        for m in self.pf_m:
            for S in tuple(self.g.sizes):
                t_, p_, a_ = zB(S)
                lg = self._call_prefill(
                    [b * self.R + jj for b in range(self.dp) for jj in range(m)],
                    t_,
                    p_,
                    a_,
                    np.full(self.B, S, np.int32),
                    cap=self.ladder[0],
                )
                jax.block_until_ready(lg)
                self.g.log(f"  warmup batched prefill m={m} S={S} done at +{time.time()-t:.0f}s")
        for k in self.buckets:
            lg, idx = self._call_decode(k, *zB(1), cap=self.ladder[0])
            self._sample_rows(lg, idx)
            self.g.log(f"  warmup decode bucket bs={k * self.dp} done at +{time.time()-t:.0f}s")
        self._sample_rows(jnp.zeros((1, self.g.a.vocab_size), jnp.float32), [0])
        if self.B > 1:
            self.g.states = self._move(self.g.states, np.int32(1), np.int32(0))
        self.g.reset()
