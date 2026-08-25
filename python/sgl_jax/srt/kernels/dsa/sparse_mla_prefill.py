"""TPU Pallas kernel for **sparse MLA-latent attention** (the DSA attention-consuming
path) — the fused sparse-MLA *prefill* kernel.

Given, per query, a set of selected past tokens (the lightning-indexer top-k), attend
**only** to those tokens over the MLA *latent* cache. Prefill (S queries) and decode
(S == 1) share the one kernel; this module is the prefill entry point.

Design (mirrors DeepSeek's ``refs/dsa/tilelang_sparse_mla_fwd.py`` *algorithm*, not
its GPU code):

* **Sparse == dense-MLA with a gathered KV iteration.** The only structural change
  from a dense paged-MLA kernel is that the inner loop walks the *selected* KV rows
  (named by ``indices``) instead of all of them, with the usual flash/online-softmax
  accumulation.
* **Head-sharing is what makes it viable on TPU.** MLA caches a single latent vector
  per token (``[Dk] = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576``) that is
  reused by *all* query heads, so each fetched row feeds a fat ``[H, Dk]`` matmul.
  The "value" is the first ``kv_lora_rank`` dims of the same latent.
* **One kernel, prefill + decode:** the grid is over ``(batch, query_token)``;
  decode is just ``S == 1``. Queries are *not* batched into a shared-K matmul because
  each query has its own selection — parallelism comes from the head dim per query.
* **Static counts, dynamic addresses.** The number of selected units is fixed
  (``K``); only the *addresses* (which rows) are data-dependent, read from ``indices``
  on-chip and used to drive per-unit DMAs from HBM.

Selection granularity is parameterised by ``read_block`` (``RB``): ``indices[b, s, :]``
are **unit ids**, one unit == ``RB`` contiguous tokens. Page-level selection sets
``RB == page_size`` and consumes the indexer's page-topk directly; effective attended
tokens per query == ``K * RB``.

Only the **head-minor chunked** kernel is deployed (few heads/device, i.e. head-TP):
``G`` selected units are gathered into one ``[CBR, Dk]`` tile with the abundant keys on
the 128-wide MXU lane axis and the few heads on the sublane axis, so one MXU-filling
matmul runs per chunk. (The v1 per-unit / non-head-minor / chunk-pipelined A/B
baselines from the perf-characterization sweep are kept in git history only; they are
not part of this deployed kernel.)
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _sparse_mla_kernel_chunked_hminor(
    q_ref,  # [1, 1, Hp, Dk]  Hp padded to the bf16 sublane tile (16), NOT to 128
    idx_ref,  # [1, 1, 1, K]    SMEM
    pos_ref,  # [1, 1, 1, 1]    SMEM
    kvlen_ref,  # [1, 1, 1, 1]  SMEM  per-query causal bound (seq_lens[rid]; == T single-seq)
    base_ref,  # [1, 1, 1, 1]   SMEM  per-query page-table base (cu_kv_lens[rid]//ps; 0 single-seq)
    kv_hbm,  # flat: [B, T(+RBF), Dk] HBM;  paged: [1, num_pages*page_size, Dk] HBM
    pt_ref,  # [1, 1, 1, PTW] SMEM  page table (per-request slice at base + logical page)
    o_ref,  # [1, 1, Hp, Dv]
    kv_scratch,  # [CBR, Dk] VMEM  one chunk of gathered latent
    sem,  # DMA semaphores (G,)
    *,
    sm_scale: float,
    Dv: int,
    RB: int,
    RBF: int,
    K: int,
    G: int,
    CBR: int,  # G*RBF rounded up to the 128-lane tile (score's key axis is the lane axis)
    paged: bool = False,  # read from the packed 4D paged latent cache via pt_ref
    page_size: int = 0,  # tokens per page (paged only; static)
    PTW: int = 1,  # page-table width (max_pages single-seq / total packed pages ragged)
):
    """Head-minor relayout of the chunked kernel for **few heads/device** (head-TP).

    A default chunked layout would put heads on the 128-wide MXU lane axis (score
    ``[CBR, Hp]``), wasting ~97% of the MXU when a device holds only ~4 heads. Here we
    swap the axes: the abundant **selected keys sit on the lane axis** (``CBR`` filling
    128) and the few **heads sit on the sublane axis** (padded to 16), so 4 heads cost
    a 4x sublane pad instead of a 32x lane pad. Score is ``[Hp, CBR]``, softmax reduces
    over the key (lane) axis, value matmul contracts ``CBR``.

    The validity mask lives on the *lane* axis (keys). A lane concat of ``RBF``-wide
    pieces is not Mosaic-lowerable, so we build the ``[CBR]`` lane mask **arithmetically**
    from an on-chip ``iota`` with per-unit **static-window** compares (no concat, no
    dynamic gather, no integer div/mod on a vector): each chunk selects every lane's
    unit id / row offset with ``G`` broadcasted ``where`` ops, recovers key positions,
    and turns the boolean into a ``0 / -inf`` additive bias."""
    b = pl.program_id(0)
    Hp = q_ref.shape[2]
    q = q_ref[0, 0]  # [Hp, Dk] bf16
    qpos = pos_ref[0, 0, 0, 0]
    kv_len = kvlen_ref[0, 0, 0, 0]  # per-query causal bound (== T in single-seq)
    base = base_ref[0, 0, 0, 0]  # per-query page-table base (== 0 in single-seq)
    NC = (K + G - 1) // G

    lane = jnp.arange(CBR, dtype=jnp.int32)  # [CBR] iota on the key (lane) axis

    def _src(u):
        """HBM source slice ([RBF, Dk]) for selected unit ``u`` (logical unit id)."""
        if paged:
            # unit u == RB contiguous LOGICAL tokens starting at u*RB. RB divides
            # page_size, so the whole unit lives in one physical page. The 4D cache
            # is flattened to [num_pages*page_size, Dk]; token (page P, offset o) is
            # row P*page_size + o. RB%16==0 => RBF==RB => no cross-page over-fetch.
            #
            # The token axis is sublane-tiled (8), so a *dynamic* slice offset must be
            # provably %8. row0 = pp*page_size + o0 is always a multiple of RB(>=16),
            # but pp (from the page table) is opaque to Mosaic. Express the offset as
            # an explicit `r8 * 8` (page_size and o0 are both %8) so divisibility is
            # provable — the same idiom the dense paged kernel uses.
            #
            # ``u`` is a SEQ-LOCAL logical page id; the request's pages start at
            # ``base`` in the packed page table (base==0 for the single-seq per-b
            # table). Clamp base+lp to the table width so a padded/-1 lane can't
            # gather out of range (the lane is masked out below anyway).
            lt = u * RB  # logical token start (within the request)
            lp = lt // page_size  # logical page (static divisor)
            o0 = lt - lp * page_size  # in-page offset (multiple of RB)
            pidx = jnp.minimum(base + lp, PTW - 1)
            pp = pt_ref[0, 0, 0, pidx]  # physical page id
            r8 = pp * (page_size // 8) + o0 // 8  # (P*page_size + o0) // 8, exact
            return kv_hbm.at[0, pl.ds(r8 * 8, RBF), :]
        return kv_hbm.at[b, pl.ds(u * RB, RBF), :]

    # padded lanes (>= G*RBF) are never DMA'd; zero them so a stale/NaN row can't
    # poison the (masked-out) score before the -inf bias is added.
    if CBR > G * RBF:
        pad = CBR - G * RBF
        kv_scratch[pl.ds(G * RBF, pad), :] = jnp.zeros((pad, kv_scratch.shape[1]), kv_scratch.dtype)

    m0 = jnp.full((Hp,), -jnp.inf, dtype=jnp.float32)
    l0 = jnp.zeros((Hp,), dtype=jnp.float32)
    acc0 = jnp.zeros((Hp, Dv), dtype=jnp.float32)

    def chunk_body(c, carry):
        m_i, l_i, acc = carry
        base = c * G

        # --- issue G gathers (overlap) into contiguous RBF-aligned lane slots ---
        for g in range(G):
            gidx = jnp.minimum(base + g, K - 1)
            # clamp negative unit ids (topk padding == -1) to 0 for the DMA so the
            # source offset stays in range; the lane is masked out below anyway.
            u = jnp.maximum(idx_ref[0, 0, 0, gidx], 0)
            dst = kv_scratch.at[pl.ds(g * RBF, RBF), :]
            pltpu.make_async_copy(_src(u), dst, sem.at[g]).start()

        # --- build the [CBR] lane validity bias arithmetically while DMAs run ---
        # Accumulators are int32 (0/1), NOT bool: Mosaic can't materialise an i1 vector
        # from a broadcast-scalar select (trunc i8->i1 is unsupported). The i1 predicate
        # is only formed transiently by the final comparisons feeding the select.
        u_vec = jnp.zeros((CBR,), jnp.int32)  # per-lane selected unit id
        row_vec = jnp.zeros((CBR,), jnp.int32)  # per-lane row-within-unit (0..RBF-1)
        ir_vec = jnp.zeros((CBR,), jnp.int32)  # 1 where the lane's unit id < K
        for g in range(G):
            lo = g * RBF  # static lane window for unit g
            sel = (lane >= lo) & (lane < lo + RBF)  # [CBR] const partition (transient i1)
            u_g = idx_ref[0, 0, 0, jnp.minimum(base + g, K - 1)]
            inr = ((base + g) < K).astype(jnp.int32)  # scalar 0/1
            u_vec = jnp.where(sel, u_g, u_vec)
            row_vec = jnp.where(sel, lane - lo, row_vec)
            ir_vec = jnp.where(sel, inr, ir_vec)
        kp = u_vec * RB + row_vec  # [CBR] seq-local key positions
        # (u_vec >= 0) drops topk padding lanes (unit id -1); their DMA was
        # clamped to unit 0, so the read is safe and only the mask excludes them.
        # ``kv_len`` is the per-request bound (seq_lens[rid]); in the single-seq
        # path it equals the static T, so this is a strict generalisation.
        valid = (u_vec >= 0) & (row_vec < RB) & (ir_vec > 0) & (kp <= qpos) & (kp < kv_len)
        bias = jnp.where(valid, 0.0, -jnp.inf)  # [CBR] fp32

        for g in range(G):
            gidx = jnp.minimum(base + g, K - 1)
            u = jnp.maximum(idx_ref[0, 0, 0, gidx], 0)
            dst = kv_scratch.at[pl.ds(g * RBF, RBF), :]
            pltpu.make_async_copy(_src(u), dst, sem.at[g]).wait()

        kv_blk = kv_scratch[pl.ds(0, CBR), :]  # [CBR, Dk] bf16

        # score: [Hp,Dk]·[CBR,Dk] -> [Hp, CBR]  (heads on sublane, keys on lanes)
        s = (
            jax.lax.dot_general(
                q, kv_blk, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )
            * sm_scale
        )
        s = s + bias[None, :]  # [Hp, CBR] fp32

        # --- flash-softmax over keys = axis 1 (the lane axis) ---
        m_cur = jnp.max(s, axis=1)  # [Hp]
        m_new = jnp.maximum(m_i, m_cur)
        corr = jnp.where(jnp.isneginf(m_new), 0.0, m_new)
        p = jnp.exp(s - corr[:, None])  # [Hp, CBR]
        alpha = jnp.where(jnp.isneginf(m_new), 1.0, jnp.exp(m_i - m_new))
        l_i = l_i * alpha + jnp.sum(p, axis=1)  # [Hp]
        v_blk = kv_blk[:, :Dv]  # [CBR, Dv]
        # acc[Hp,Dv] += sum_cbr p[hp,cbr] * v[cbr,dv]
        acc = acc * alpha[:, None] + jax.lax.dot_general(
            p.astype(kv_blk.dtype),
            v_blk,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        return (m_new, l_i, acc)

    m_i, l_i, acc = jax.lax.fori_loop(0, NC, chunk_body, (m0, l0, acc0))
    out = acc / jnp.where(l_i == 0.0, 1.0, l_i)[:, None]
    o_ref[0, 0] = out.astype(o_ref.dtype)


def sparse_mla_attention(
    q,  # [B, S, H, Dk]        Dk = kv_lora_rank + qk_rope_head_dim
    kv,  # [B, T, Dk]           MLA latent cache (value = first Dv cols)
    indices,  # [B, S, K] int32      selected unit ids (unit == RB tokens)
    positions,  # [B, S] int32         query positions (causal bound)
    *,
    kv_lora_rank: int = 512,
    read_block: int = 1,
    block_units: int | None = None,  # units gathered+matmul'd per chunk (G; parallelism knob)
    sm_scale: float | None = None,  # REQUIRED: 1/sqrt(qk_nope_head_dim + qk_rope_head_dim)
    interpret: bool = False,
    page_table=None,  # [B, max_pages] int32: logical page -> physical page. When set,
    # ``kv`` is the packed 4D paged cache and the kernel gathers from it.
    page_size: int | None = None,  # tokens/page (required with page_table)
    seq_len: int | None = None,  # logical T for the causal mask (required, non-ragged paged)
    # ── packed-ragged mode (multi-request extend) ────────────────────────────
    # When ``q_seq_id`` is set, ``q`` is packed as [1, total_tokens, H, Dk] and the
    # per-query causal bound / page-table base are resolved per token from the same
    # ragged metadata the dense path uses. ``page_indices`` (flat packed physical
    # pages) replaces the per-request ``page_table``; ``indices`` are seq-local page
    # ids relative to each request's window (page_indices[cu_kv_lens[rid]//ps + p]).
    q_seq_id=None,  # [total_tokens] int32  token -> request id (enables ragged mode)
    seq_lens=None,  # [num_seqs] int32       per-request kv length (causal bound)
    cu_kv_lens=None,  # [num_seqs+1] int32    page-aligned kv offsets (page_indices stride)
    page_indices=None,  # [total_pages] int32  packed physical page ids
):
    """Sparse MLA-latent attention (head-minor chunked kernel).
    Returns ``[B, S, H, kv_lora_rank]`` (float32).

    ``sm_scale`` is **required**: the score is the absorbed q·k over the latent
    ``Dk`` but the correct scale is the original ``1/sqrt(qk_nope_head_dim +
    qk_rope_head_dim)`` (= 1/√192 for DS/GLM), which the latent dim doesn't encode.

    ``indices`` are *unit ids* (one unit = ``read_block`` contiguous tokens); set
    ``read_block=1`` for token-granular (exact) DSA, or e.g. 128 for page-sized block
    reads. Effective attended tokens per query == ``K * read_block``. ``block_units=G``
    (>=1) sets how many units are gathered into one MXU-filling matmul per chunk
    (``G>=K`` ⇒ a single chunk, the fastest measured config).

    KV source:
    * default: ``kv`` is a flat ``[B, T, Dk]`` latent buffer.
    * paged (``page_table`` set): ``kv`` is the sglang-jax packed 4D MLA cache
      ``[num_pages, align(page_size,packing)//packing, packing, align(lkv,128)+align(rope,128)]``.
      Flattened to ``[num_pages*page_size, Dk_pad]``, token (physical page P, offset o)
      is row ``P*page_size + o``; the kernel resolves each selected unit through
      ``page_table`` and requires ``read_block % 16 == 0`` (so a unit never over-fetches
      across a page).
    """
    B, S, H, Dk = q.shape
    K = indices.shape[2]
    Dv = kv_lora_rank
    RB = read_block
    ragged = q_seq_id is not None
    paged = (page_table is not None) or ragged
    if ragged and B != 1:
        raise ValueError(f"ragged mode packs all requests into B=1 (got B={B})")
    if ragged and (seq_lens is None or cu_kv_lens is None or page_indices is None):
        raise ValueError("ragged mode requires seq_lens, cu_kv_lens and page_indices")
    if not block_units or block_units < 1:
        raise ValueError("sparse_mla_attention requires block_units >= 1")
    if sm_scale is None:
        # No safe default: the score is the absorbed q·k over the full latent Dk
        # (kv_lora_rank + qk_rope_head_dim), but the correct scale is the ORIGINAL
        # attention head dim 1/sqrt(qk_nope_head_dim + qk_rope_head_dim) (= 1/sqrt(192)
        # for DS/GLM), which cannot be recovered from Dk or kv_lora_rank alone.
        # Deriving it from the latent dim (the old 1/sqrt(Dk) default) silently
        # under-scales the logits, so require the caller to pass it explicitly.
        raise ValueError(
            "sparse_mla_attention requires an explicit sm_scale "
            "(1/sqrt(qk_nope_head_dim + qk_rope_head_dim)); the latent Dk is not the "
            "score head dim, so there is no safe default."
        )

    # TPU/Mosaic wants the lane (last) dim %128; the MLA latent is 576 = 512 + 64
    # rope, so pad the feature dim with zeros to the next multiple of 128 (640).
    # Padding the *contraction* dim with zeros leaves q·k unchanged, and the value
    # is only the first Dv (=512) dims, so the pad never enters the output.
    Dk_pad = ((Dk + 127) // 128) * 128
    q = jnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, Dk_pad - Dk))) if Dk_pad != Dk else q
    if paged:
        if RB % 16 != 0:
            raise ValueError("paged KV requires read_block % 16 == 0 (no cross-page over-fetch)")
        if page_size is None:
            raise ValueError("paged KV requires page_size")
        if not ragged and seq_len is None:
            raise ValueError("non-ragged paged KV requires seq_len")
        if kv.shape[-1] != Dk_pad:
            raise ValueError(f"paged cache last dim {kv.shape[-1]} != Dk_pad {Dk_pad}")
        # T is the causal bound only in the non-ragged path (a single static scalar);
        # ragged resolves the bound per token from seq_lens[q_seq_id] instead.
        T = seq_len if not ragged else S
        # flatten [num_pages, ps//packing, packing, Dk_pad] -> [1, num_pages*page_size, Dk_pad]
        num_pages = kv.shape[0]
        kv = kv.reshape(1, num_pages * page_size, Dk_pad)
    else:
        T = kv.shape[1]
        if Dk_pad != Dk:
            kv = jnp.pad(kv, ((0, 0), (0, 0), (0, Dk_pad - Dk)))

    # Add a singleton axis so the per-(b,s) block's trailing two dims are legal
    # (block over the S axis is 1, which is illegal as a *second-to-last* dim).
    indices4 = indices.reshape(B, S, 1, K)
    positions4 = positions.reshape(B, S, 1, 1)

    # head-minor relayout: keys on the lane axis (fills the MXU even with ~4 heads),
    # heads on the sublane axis (padded to the bf16 sublane tile of 16, not 128).
    G = max(1, min(block_units, K))
    RBF = ((RB + 15) // 16) * 16
    ps = page_size or 0
    if ragged:
        # packed page table: one flat physical-page list; each request's window
        # starts at cu_kv_lens[rid]//ps (the per-query ``base``). Per-query causal
        # bound is seq_lens[rid]. Both are resolved per token below.
        #
        # SMEM footprint: the whole flat table (PTW = sum of pages_per_seq over the
        # packed batch) lives resident in SMEM via ``pt_spec`` at 4*PTW bytes. The
        # block index map is constant in ``s`` (and ragged is B=1), so it is copied
        # in once, not re-DMA'd per grid step. PTW is bounded by what fits in HBM:
        # each page costs page_size*Dk_pad*2 HBM bytes vs 4 SMEM bytes (~1e4x), so
        # any batch whose KV fits in HBM keeps this block in the low-KB range (~8KB
        # at ctx 8k / page 128 / max_running 32). Reaching ~1MB here (~270k pages,
        # e.g. 128k ctx / page 64 / max_running 128) implies ~20GB of resident KV,
        # i.e. an HBM OOM would hit first.
        PTW = page_indices.shape[0]
        pt_arg = page_indices.reshape(1, 1, 1, PTW).astype(jnp.int32)
        qsid = jnp.clip(q_seq_id, 0, seq_lens.shape[0] - 1).astype(jnp.int32)
        kvlen_arg = seq_lens[qsid].reshape(B, S, 1, 1).astype(jnp.int32)
        base_arg = (cu_kv_lens[qsid] // ps).reshape(B, S, 1, 1).astype(jnp.int32)
    elif paged:
        PTW = (T + ps - 1) // ps  # max_pages
        # page table is per-REQUEST (same for all query tokens s) => index by b only.
        pt_arg = page_table.reshape(B, 1, 1, PTW).astype(jnp.int32)
        kvlen_arg = jnp.full((B, S, 1, 1), T, jnp.int32)  # static causal bound
        base_arg = jnp.zeros((B, S, 1, 1), jnp.int32)  # per-b table => base 0
    else:
        kv = jnp.pad(kv, ((0, 0), (0, RBF), (0, 0)))
        # dummy 1-wide page table so the kernel signature is uniform (unused when flat).
        PTW = 1
        pt_arg = jnp.zeros((B, 1, 1, 1), jnp.int32)
        kvlen_arg = jnp.full((B, S, 1, 1), T, jnp.int32)
        base_arg = jnp.zeros((B, S, 1, 1), jnp.int32)
    pt_spec = pl.BlockSpec((1, 1, 1, PTW), lambda b, s: (b, 0, 0, 0), memory_space=pltpu.SMEM)
    CBR = ((G * RBF + 127) // 128) * 128  # key (lane) axis: %128
    Hq = ((H + 15) // 16) * 16  # heads on sublane: %16 (bf16 tile)
    if Hq != H:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, Hq - H), (0, 0)))
    kernel = functools.partial(
        _sparse_mla_kernel_chunked_hminor,
        sm_scale=sm_scale,
        Dv=Dv,
        RB=RB,
        RBF=RBF,
        K=K,
        G=G,
        CBR=CBR,
        paged=paged,
        page_size=ps,
        PTW=PTW,
    )
    scratch_shapes = [
        pltpu.VMEM((CBR, Dk_pad), kv.dtype),  # one chunk of gathered latent
        pltpu.SemaphoreType.DMA((G,)),  # one semaphore per unit
    ]

    smem = pltpu.SMEM
    in_specs = [
        pl.BlockSpec((1, 1, Hq, Dk_pad), lambda b, s: (b, s, 0, 0)),  # q (VMEM)
        pl.BlockSpec((1, 1, 1, K), lambda b, s: (b, s, 0, 0), memory_space=smem),  # indices
        pl.BlockSpec((1, 1, 1, 1), lambda b, s: (b, s, 0, 0), memory_space=smem),  # positions
        pl.BlockSpec((1, 1, 1, 1), lambda b, s: (b, s, 0, 0), memory_space=smem),  # kv_len bound
        pl.BlockSpec((1, 1, 1, 1), lambda b, s: (b, s, 0, 0), memory_space=smem),  # page base
        pl.BlockSpec(memory_space=pltpu.HBM),  # kv (untiled HBM, full — DMA-gathered)
        pt_spec,  # page table (SMEM)
    ]
    call_args = [q, indices4, positions4, kvlen_arg, base_arg, kv, pt_arg]

    out = pl.pallas_call(
        kernel,
        grid=(B, S),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((1, 1, Hq, Dv), lambda b, s: (b, s, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((B, S, Hq, Dv), jnp.float32),
        scratch_shapes=scratch_shapes,
        interpret=interpret,
    )(*call_args)
    if Hq != H:
        out = out[:, :, :H, :]  # drop padded heads
    return out


def prefill_write_and_attend(
    ql,  # [T, H, kv_lora_rank]   absorbed latent query (nope)
    qpe,  # [T, H, rope]           rope query
    kvc,  # [T, kv_lora_rank]      new c_kv to write
    kpe,  # [T, rope]              new k_rope to write
    cache,  # [P, ps//pk, pk, Dk_pad]  paged fused latent cache
    topk_pages,  # [T, K] int32          seq-local page ids (-1 padded)
    positions,  # [T] int32             absolute query positions (causal bound)
    loc,  # [T] int32             physical flat slot per token (out_cache_loc)
    *,
    kv_lora_rank: int,
    page_size: int,
    sm_scale: float,
    read_block: int | None = None,  # defaults to page_size (page-level selection)
    interpret: bool = False,
):
    """Self-write the current chunk's latent into the paged cache, then attend
    only the indexer-selected pages via the fused sparse-MLA prefill kernel.

    **SINGLE-SEQUENCE, SINGLE-SHOT extend only** (the padded-bucket TTFT case). All
    ``T`` tokens are one contiguous request prefilled from position 0, so a single
    per-request page table ``pt = loc[::page_size] // page_size`` maps logical→physical
    (page p's first token slot // ps) and ``seq_len == T`` is the causal bound
    (per-token causality via ``positions``). This stride mapping is only valid under
    that scope — enforced upstream by the ``model_runner`` ``DSA_PREFILL_SPARSE``
    guards (radix cache OFF, ``max_running == 1``, ``chunked_prefill >= context``). A
    partial / batched / chunked extend would reference pages outside the current chunk
    and corrupt the table; support for that is a follow-up.

    Returns ``(o[T,H,kv_lora_rank], updated_cache)``.
    """
    T, H, Dv = ql.shape
    rope = qpe.shape[-1]
    ps = page_size
    RB = read_block if read_block is not None else ps
    # page-level selection: a selected unit (RB tokens) must sit within one page so the
    # paged gather never straddles a page boundary (kernel also asserts RB % 16 == 0).
    assert ps % RB == 0, f"read_block {RB} must divide page_size {ps} (page-aligned units)"
    Pn, pspk, pk, Dk_pad = cache.shape
    K = topk_pages.shape[-1]

    q_sparse = jnp.concatenate([ql, qpe], axis=-1)  # [T, H, Dv+rope]
    # per-request logical→physical page table (page p's first token slot // ps).
    # Valid ONLY for the single-shot contiguous extend above; clamp guards
    # padding/-1 slots (out-of-range → OOB gather / core halt).
    pt = jnp.clip(loc[::ps] // ps, 0, Pn - 1).astype(jnp.int32)

    # self-write: [c_kv | k_pe | pad] row per token at its physical slot. Padded
    # tokens carry out_cache_loc == -1 and MUST be skipped. NOTE: mode="drop" only
    # drops OUT-OF-RANGE indices — it still WRAPS negatives (-1 -> last slot), which
    # would silently corrupt the final physical KV slot on every prefill. Pass
    # wrap_negative_indices=False so the -1 rows are actually dropped.
    row = jnp.zeros((T, Dk_pad), cache.dtype)
    row = row.at[:, :Dv].set(kvc.astype(cache.dtype))
    row = row.at[:, Dv : Dv + rope].set(kpe.reshape(T, rope).astype(cache.dtype))
    flat = cache.reshape(Pn * ps, Dk_pad)
    flat = flat.at[loc].set(row, mode="drop", wrap_negative_indices=False)
    cache_new = flat.reshape(Pn, pspk, pk, Dk_pad)

    out = sparse_mla_attention(
        q_sparse.reshape(1, T, H, q_sparse.shape[2]),
        cache_new,
        topk_pages.reshape(1, T, -1),
        positions.reshape(1, T),
        kv_lora_rank=Dv,
        read_block=RB,
        block_units=K,
        sm_scale=float(sm_scale),
        page_table=pt.reshape(1, -1),
        page_size=ps,
        seq_len=T,
        interpret=interpret,
    )
    return out.reshape(T, H, Dv), cache_new


def prefill_write_and_attend_ragged(
    ql,  # [total_tokens, H, kv_lora_rank]   absorbed latent query (nope)
    qpe,  # [total_tokens, H, rope]           rope query
    kvc,  # [total_tokens, kv_lora_rank]      new c_kv to write
    kpe,  # [total_tokens, rope]              new k_rope to write
    cache,  # [P, ps//pk, pk, Dk_pad]         paged fused latent cache
    topk_pages,  # [total_tokens, K] int32    seq-local page ids (-1 padded)
    positions,  # [total_tokens] int32        absolute query positions (causal bound)
    loc,  # [total_tokens] int32              physical flat slot per token (out_cache_loc)
    seq_lens,  # [num_seqs] int32             per-request kv length
    cu_q_lens,  # [num_seqs+1] int32          per-request query offsets (ragged segments)
    cu_kv_lens,  # [num_seqs+1] int32         page-aligned kv offsets (page_indices stride)
    page_indices,  # [total_pages] int32      packed physical page ids
    *,
    kv_lora_rank: int,
    page_size: int,
    sm_scale: float,
    read_block: int | None = None,  # defaults to page_size (page-level selection)
    interpret: bool = False,
):
    """Packed-ragged self-write + sparse-MLA prefill for **multi-request** extend.

    Generalises :func:`prefill_write_and_attend` to a batch of ragged sequences
    packed along the token axis (the same layout the dense ``mla_ragged_paged_attention``
    and the indexer consume). Differences from the single-sequence wrapper:

    * The self-write is unchanged — ``loc`` (out_cache_loc) already names each token's
      physical slot, so ``flat.at[loc].set(...)`` is correct for any number of
      sequences and any prefix (padded ``-1`` slots are dropped).
    * The page table is **not** derived by the ``loc[::ps]`` stride (only valid for one
      contiguous request). Instead the kernel reads the packed ``page_indices`` at a
      per-request base ``cu_kv_lens[rid]//ps`` and uses ``seq_lens[rid]`` as the causal
      bound, with ``rid = q_seq_id[token]`` recovered from ``cu_q_lens``.

    Returns ``(o[total_tokens, H, kv_lora_rank], updated_cache)``.
    """
    T, H, Dv = ql.shape
    rope = qpe.shape[-1]
    ps = page_size
    RB = read_block if read_block is not None else ps
    assert ps % RB == 0, f"read_block {RB} must divide page_size {ps} (page-aligned units)"
    Pn, pspk, pk, Dk_pad = cache.shape
    K = topk_pages.shape[-1]
    S = seq_lens.shape[0]

    q_sparse = jnp.concatenate([ql, qpe], axis=-1)  # [T, H, Dv+rope]

    # self-write: per-token scatter to out_cache_loc (suffix-safe, prefix-agnostic).
    # mode="drop"+wrap_negative_indices=False drops padded -1 slots (see single-seq).
    row = jnp.zeros((T, Dk_pad), cache.dtype)
    row = row.at[:, :Dv].set(kvc.astype(cache.dtype))
    row = row.at[:, Dv : Dv + rope].set(kpe.reshape(T, rope).astype(cache.dtype))
    flat = cache.reshape(Pn * ps, Dk_pad)
    flat = flat.at[loc].set(row, mode="drop", wrap_negative_indices=False)
    cache_new = flat.reshape(Pn, pspk, pk, Dk_pad)

    # token -> request id (same convention as _scatter_paged / the ref oracle).
    t = jnp.arange(T, dtype=jnp.int32)
    q_seq_id = jnp.clip(jnp.searchsorted(cu_q_lens[1:], t, side="right"), 0, S - 1).astype(
        jnp.int32
    )

    out = sparse_mla_attention(
        q_sparse.reshape(1, T, H, q_sparse.shape[2]),
        cache_new,
        topk_pages.reshape(1, T, -1),
        positions.reshape(1, T),
        kv_lora_rank=Dv,
        read_block=RB,
        block_units=K,
        sm_scale=float(sm_scale),
        page_size=ps,
        q_seq_id=q_seq_id,
        seq_lens=seq_lens,
        cu_kv_lens=cu_kv_lens,
        page_indices=page_indices,
        interpret=interpret,
    )
    return out.reshape(T, H, Dv), cache_new


def units_to_token_ids(indices, read_block: int):
    """Expand unit ids ``[B, S, K]`` → token ids ``[B, S, K*read_block]``.

    Convenience for validating against the token-granular numpy oracle
    (``glm5_tpu.dsa.attention.dsa_attention``), which expects token indices.
    """
    starts = indices[..., None] * read_block  # [B,S,K,1]
    offs = jnp.arange(read_block, dtype=indices.dtype)  # [RB]
    return (starts + offs).reshape(indices.shape[:2] + (-1,))


def flat_to_paged_cache(kv, page_size: int, kv_packing: int = 2):
    """Pack a flat ``[B, T, Dk]`` latent into the sglang-jax 4D paged MLA cache
    plus a **contiguous** page table ``[B, max_pages]``. Test/bench scaffolding
    mirroring ``MLATokenToKVPool``'s layout
    ``[num_pages, page_size//kv_packing, kv_packing, Dk]`` where token (page P,
    within-page offset o) lives at ``[P, o//kv_packing, o%kv_packing, :]``.

    Sequence ``b`` is assigned pages ``[b*pages_per_seq : (b+1)*pages_per_seq)``.
    Returns ``(cache, page_table)`` as jnp arrays (cache dtype == kv dtype).
    """
    import numpy as _np

    kv_np = _np.asarray(kv)
    B, T, Dk = kv_np.shape
    if page_size % kv_packing != 0:
        raise ValueError(f"page_size {page_size} must be divisible by kv_packing {kv_packing}")
    pages_per_seq = (T + page_size - 1) // page_size
    num_pages = B * pages_per_seq
    ps_pack = page_size // kv_packing
    cache = _np.zeros((num_pages, ps_pack, kv_packing, Dk), dtype=kv_np.dtype)
    page_table = _np.zeros((B, pages_per_seq), dtype=_np.int32)
    for b in range(B):
        for lp in range(pages_per_seq):
            phys = b * pages_per_seq + lp
            page_table[b, lp] = phys
            for o in range(page_size):
                t = lp * page_size + o
                if t < T:
                    cache[phys, o // kv_packing, o % kv_packing, :] = kv_np[b, t]
    return jnp.asarray(cache), jnp.asarray(page_table)
