"""DeepSeek-V4.1 in JAX, first version: a line-by-line port of DeepSeek's reference `inference/model.py`.

Purpose: numerical ground truth on CPU against the reference (see test_vs_ref.py). Control flow uses Python ints for
`start_pos` (prefill = 0, decode = one token), caches are explicit functional state. Not jit/TPU-ready yet.
Dense (non-quantized) weights only; activation fake-quantization (fp8 / fp4 round trips) is reproduced exactly.
Engram, DSpark, MTP, vision: not implemented.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

NOQ = (
    os.environ.get("V41_NOQUANT") == "1"
)  # debug: disable activation fake-quant (matches the reference switch)
FP8_BLOCK = 32  # fp8_block_size in the reference
FP4_BLOCK = 32

_FP4_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
_FP4_MID = (_FP4_VALUES[1:] + _FP4_VALUES[:-1]) / 2


@dataclass(frozen=True)
class Args:
    dim: int
    n_layers: int
    n_heads: int
    head_dim: int
    rope_head_dim: int
    q_lora_rank: int
    o_groups: int
    o_lora_rank: int
    window_size: int
    compress_ratios: tuple
    kv_source_layers: tuple
    index_source_layers: tuple
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source_layer: int
    candidate_topk_blocks: int
    candidate_block_size: int
    n_routed_experts: int
    n_activated_experts: int
    moe_inter_dim: int
    swiglu_limit: float
    score_func: str
    gate_temp: float
    norm_topk_prob: bool
    route_scale: float
    norm_eps: float
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    rope_theta: float
    compress_rope_theta: float
    original_seq_len: int
    rope_factor: float
    beta_fast: int
    beta_slow: int
    max_seq_len: int
    vocab_size: int
    engram_layer_ids: tuple = ()
    engram_max_ngram_size: int = 1
    engram_n_heads: int = 0
    engram_head_dim: int = 0

    @staticmethod
    def from_torch(a) -> "Args":
        return Args(
            **{f: getattr(a, f) for f in Args.__dataclass_fields__ if f != "dim"}, dim=a.dim
        )


# ----------------------------------------------------------------------------- numerics helpers


def rms_norm(x, w, eps):
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return (w * (x32 * jax.lax.rsqrt(var + eps))).astype(x.dtype)


def _pow2_ceil_scale(v):
    """2 ** ceil(log2(v)) computed exactly (the reference kernel does this with IEEE bit tricks)."""
    m, ex = jnp.frexp(v)  # v = m * 2**ex, m in [0.5, 1)
    e = jnp.where(m == 0.5, ex - 1, ex)
    return jnp.ldexp(jnp.ones_like(v), e)


def act_quant_fp8(x, block=FP8_BLOCK):
    """act_quant(..., scale_fmt='ue8m0', inplace=True): e4m3 fake quant with power-of-two scales."""
    if NOQ:
        return x
    z = x.astype(jnp.float32)
    zb = z.reshape(*z.shape[:-1], -1, block)
    amax = jnp.maximum(jnp.max(jnp.abs(zb), axis=-1, keepdims=True), 1e-4)
    s = _pow2_ceil_scale(amax * (1 / 448.0))
    y = (jnp.clip(zb / s, -448.0, 448.0).astype(jnp.float8_e4m3fn).astype(jnp.float32)) * s
    return y.reshape(z.shape).astype(x.dtype)


def _fp4_round(v):
    a = jnp.abs(v)
    mid = jnp.asarray(_FP4_MID)
    idx = jnp.searchsorted(mid, a, side="left")  # == torch.bucketize(a, mid)
    on_mid = jnp.any(a[..., None] == mid, axis=-1)
    idx = jnp.where(on_mid & (idx % 2 == 1), idx + 1, idx)
    return jnp.sign(v) * jnp.asarray(_FP4_VALUES)[idx]


def act_quant_fp4(x, block, e4m3_scale=False):
    """fp4_act_quant(..., inplace=True): e2m1 fake quant; scales are e8m0 (pow2) or e4m3 (compressed KV)."""
    if NOQ:
        return x
    z = x.astype(jnp.float32)
    zb = z.reshape(*z.shape[:-1], -1, block)
    amax = jnp.max(jnp.abs(zb), axis=-1, keepdims=True)
    if e4m3_scale:
        amax = jnp.maximum(amax, 6 * 2.0**-9)
        s = (amax / 6.0).astype(jnp.float8_e4m3fn).astype(jnp.float32)
    else:
        amax = jnp.maximum(amax, 6 * 2.0**-126)
        s = _pow2_ceil_scale(amax * (1 / 6.0))
    y = _fp4_round(jnp.clip(zb / s, -6.0, 6.0)) * s
    return y.reshape(z.shape).astype(x.dtype)


def freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """(cos, sin) [seqlen, dim/2], float32; YaRN when original_seq_len > 0."""
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    if original_seq_len > 0:

        def corrected(rot):
            return dim * math.log(original_seq_len / (rot * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected(beta_fast)), 0)
        high = min(math.ceil(corrected(beta_slow)), dim - 1)
        ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / max(high - low, 1e-3), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    ang = np.outer(np.arange(seqlen, dtype=np.float32), freqs.astype(np.float32))
    return jnp.asarray(np.cos(ang), jnp.float32), jnp.asarray(np.sin(ang), jnp.float32)


def rotary(x, cos, sin, inverse=False):
    """Rotate adjacent pairs of the last dim. x [b,s,d] or [b,s,h,d]; cos/sin [s,d/2]."""
    xr = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 2)
    re, im = xr[..., 0], xr[..., 1]
    if inverse:
        sin = -sin
    if cos.ndim == 3:  # per-row positions: cos/sin [b,s,d/2]
        c, s = (cos, sin) if x.ndim == 3 else (cos[:, :, None], sin[:, :, None])
    elif x.ndim == 3:
        c, s = cos[None], sin[None]
    else:
        c, s = cos[None, :, None], sin[None, :, None]
    out = jnp.stack([re * c - im * s, re * s + im * c], axis=-1).reshape(x.shape)
    return out.astype(x.dtype)


def rotate_tail(x, rd, cos, sin, inverse=False):
    return jnp.concatenate([x[..., :-rd], rotary(x[..., -rd:], cos, sin, inverse)], axis=-1)


def window_topk_idxs(window, bsz, seqlen, start_pos):
    if start_pos == 0:
        end = np.arange(seqlen)[:, None]
        idxs = np.clip(end - window + 1, 0, None) + np.arange(min(seqlen, window))
        idxs = np.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window + 1
        idxs = np.concatenate([np.arange(oldest, window), np.arange(oldest)])
        idxs = np.where(idxs > start_pos, -1, idxs)
        idxs = idxs[None]
    return jnp.broadcast_to(jnp.asarray(idxs, jnp.int32)[None], (bsz, *idxs.shape))


def sparse_attn(q, kv, sink, idxs, scale):
    """q [b,m,h,d]; kv [b,n,d]; sink [h]; idxs [b,m,k] (-1 = none). Softmax with an extra sink logit."""
    b, m, h, d = q.shape
    valid = idxs >= 0
    safe = jnp.maximum(idxs, 0)
    g = jnp.take_along_axis(kv[:, None], safe[..., None], axis=2) * valid[..., None]  # [b,m,k,d]
    sc = jnp.einsum("bmhd,bmkd->bmhk", q.astype(jnp.float32), g.astype(jnp.float32)) * scale
    sc = jnp.where(valid[:, :, None, :], sc, -jnp.inf)
    mx = jnp.max(sc, axis=-1, keepdims=True)
    any_valid = jnp.any(valid, axis=-1)[:, :, None, None]
    mx = jnp.where(any_valid, mx, 0.0)
    p = jnp.exp(sc - mx)
    den = jnp.sum(p, axis=-1, keepdims=True) + jnp.exp(
        sink.astype(jnp.float32)[None, None, :, None] - mx
    )
    o = jnp.einsum("bmhk,bmkd->bmhd", p, g.astype(jnp.float32)) / den
    return jnp.where(any_valid, o, 0.0).astype(q.dtype)


def hc_split_sinkhorn(mixes, scale, base, hc, iters, eps):
    mixes = mixes.astype(jnp.float32)
    pre = jax.nn.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * jax.nn.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = mixes[..., 2 * hc :].reshape(*mixes.shape[:-1], hc, hc) * scale[2] + base[
        2 * hc :
    ].reshape(hc, hc)
    comb = jax.nn.softmax(comb, axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


# ----------------------------------------------------------------------------- attention pieces


def compressor(P, x, st, start_pos, ratio, a: Args):
    """Returns (latent pre-RoPE or None, new state). ratio 1 = plain projection."""
    if ratio == 1:
        return rms_norm(x @ P["wkv.weight"].T, P["norm.weight"], a.norm_eps), st
    b, s, _ = x.shape
    dtype = x.dtype
    x32 = x.astype(jnp.float32)
    kv, score = x32 @ P["wkv.weight"].T, x32 @ P["wgate.weight"].T
    kv_state, score_state = st["kv_state"], st["score_state"]
    if start_pos == 0:
        should = s >= ratio
        rem = s % ratio
        cut = s - rem
        if rem:
            kv_state = kv_state.at[:b, :rem].set(kv[:, cut:])
            score_state = score_state.at[:b, :rem].set(score[:, cut:])
            kv, score = kv[:, :cut], score[:, :cut]
        kv = kv.reshape(b, -1, ratio, kv.shape[-1])
        score = score.reshape(b, -1, ratio, score.shape[-1])
        kv = jnp.sum(kv * jax.nn.softmax(score, axis=2), axis=2)
    else:
        should = (start_pos + 1) % ratio == 0
        slot = start_pos % ratio
        kv_state = kv_state.at[:b, slot].set(kv[:, 0])
        score_state = score_state.at[:b, slot].set(score[:, 0])
        if should:
            kv = jnp.sum(
                kv_state[:b] * jax.nn.softmax(score_state[:b], axis=1), axis=1, keepdims=True
            )
    st = {"kv_state": kv_state, "score_state": score_state}
    if not should:
        return None, st
    return rms_norm(kv.astype(dtype), P["norm.weight"], a.norm_eps), st


def select_candidate_blocks(logits, compress_lens, topk_blocks, block):
    width = logits.shape[-1]
    pad = -width % block
    scores = jnp.pad(logits, [(0, 0)] * (logits.ndim - 1) + [(0, pad)], constant_values=-jnp.inf)
    scores = scores.reshape(*scores.shape[:-1], -1, block).max(axis=-1)
    nb = scores.shape[-1]
    last = (jnp.asarray(compress_lens) - 1) // block
    scores = jnp.where(jnp.arange(nb) == last, jnp.inf, scores)
    k = min(topk_blocks, nb)
    vals, idx = jax.lax.top_k(scores, k)
    keep = jnp.any(
        (jnp.arange(nb)[None, None, None, :] == idx[..., None]) & (vals > -jnp.inf)[..., None],
        axis=-2,
    )
    return jnp.repeat(keep, block, axis=-1)[..., :width]


def indexer(P, x, qr, latent, start_pos, offset, a: Args, layer, st, shared, cos, sin):
    b, s, _ = x.shape
    ratio, rd = a.compress_ratios[layer], a.rope_head_dim
    end_pos = start_pos + s
    owns_k = layer in a.kv_source_layers
    is_cand_src = layer == a.candidate_source_layer
    uses_cand = 0 <= a.candidate_source_layer < layer
    if owns_k and latent is not None:
        if start_pos == 0:
            fc, fs = cos[: s - s % ratio : ratio], sin[: s - s % ratio : ratio]
        else:
            fc, fs = cos[start_pos + 1 - ratio][None], sin[start_pos + 1 - ratio][None]
        k = rms_norm(latent @ P["wk.weight"].T, P["k_norm.weight"], a.norm_eps)
        k = rotate_tail(k, rd, fc, fs)
        k = act_quant_fp4(k, FP4_BLOCK)
        st = dict(st)
        st["k_cache"] = (
            st["k_cache"].at[:b, start_pos // ratio : start_pos // ratio + k.shape[1]].set(k)
        )
    if owns_k:
        shared["index_k"] = st["k_cache"]
    q = (qr @ P["wq_b.weight"].T).reshape(b, s, a.index_n_heads, a.index_head_dim)
    q = rotate_tail(q, rd, cos[start_pos:end_pos], sin[start_pos:end_pos])
    q = act_quant_fp4(q, FP4_BLOCK)
    index_k = shared["index_k"][:b, : end_pos // ratio]
    softmax_scale = a.index_head_dim**-0.5
    weights = (x @ P["weights_proj.weight"].T) * (softmax_scale * a.index_n_heads**-0.5)
    score = jnp.einsum("bshd,btd->bsht", q, index_k)
    score = (jax.nn.relu(score) * weights[..., None]).sum(axis=2)  # [b,s,t]
    if start_pos == 0:
        clens = (jnp.arange(1, s + 1) // ratio)[:, None]
        score = jnp.where(jnp.arange(s // ratio)[None, None, :] >= clens[None], -jnp.inf, score)
    else:
        clens = end_pos // ratio
    if is_cand_src:
        shared["candidates"] = select_candidate_blocks(
            score, clens, a.candidate_topk_blocks, a.candidate_block_size
        )
    elif uses_cand:
        score = jnp.where(~shared["candidates"], -jnp.inf, score)
    topk = min(a.index_topk, end_pos // ratio)
    _, idx = jax.lax.top_k(score, topk)
    idx = jnp.sort(idx, axis=-1)
    clens_b = clens[None] if start_pos == 0 else clens
    out = jnp.where(idx < clens_b, idx + offset, -1).astype(jnp.int32)
    return out, st


def attention(P, x, st, start_pos, layer, a: Args, shared, rope):
    """P: this layer's attn params (keys like 'wq_a.weight', 'compressor.wkv.weight'); st: this layer's cache state."""
    b, s, _ = x.shape
    cos_all, sin_all = rope
    cos, sin = cos_all[start_pos : start_pos + s], sin_all[start_pos : start_pos + s]
    rd, win, ratio = a.rope_head_dim, a.window_size, a.compress_ratios[layer]
    qr = rms_norm(x @ P["wq_a.weight"].T, P["q_norm.weight"], a.norm_eps)
    q = (qr @ P["wq_b.weight"].T).reshape(b, s, a.n_heads, a.head_dim)
    q = rotate_tail(q, rd, cos, sin)

    # sliding-window KV
    kv = rms_norm(x @ P["wkv.weight"].T, P["kv_norm.weight"], a.norm_eps)
    kv = act_quant_fp8(rotate_tail(kv, rd, cos, sin))
    wc = st["window_kv_cache"]
    if start_pos == 0:
        if s <= win:
            wc = wc.at[:b, :s].set(kv)
        else:
            cutoff = s % win
            last = kv[:, -win:]
            wc = wc.at[:b].set(
                jnp.concatenate([last[:, win - cutoff :], last[:, : win - cutoff]], axis=1)
            )
        window_kv = kv
    else:
        wc = wc.at[:b, start_pos % win].set(kv[:, 0])
        window_kv = wc[:b]
    st = dict(st, window_kv_cache=wc)
    topk = window_topk_idxs(win, b, s, start_pos)

    if ratio:
        offset = window_kv.shape[1]
        compress_len = (start_pos + s) // ratio
        latent = None
        if layer in a.kv_source_layers:
            latent, cst = compressor(
                P_sub(P, "compressor."), x, st["compressor"], start_pos, ratio, a
            )
            st = dict(st, compressor=cst)
        # indexer first (it needs the pre-RoPE latent)
        if layer not in a.index_source_layers:
            cidx = shared["topk_idxs"]
        else:
            if compress_len == 0:
                cidx = jnp.zeros((b, s, 0), jnp.int32)
            else:
                cidx, ist = indexer(
                    P_sub(P, "indexer."),
                    x,
                    qr,
                    latent,
                    start_pos,
                    offset,
                    a,
                    layer,
                    st["indexer"],
                    shared,
                    cos_all,
                    sin_all,
                )
                st = dict(st, indexer=ist)
            shared["topk_idxs"] = cidx
        if layer in a.kv_source_layers:
            ckv = st["compress_kv_cache"]
            if latent is not None:
                if start_pos == 0:
                    fc, fs = cos_all[: s - s % ratio : ratio], sin_all[: s - s % ratio : ratio]
                else:
                    fc, fs = (
                        cos_all[start_pos + 1 - ratio][None],
                        sin_all[start_pos + 1 - ratio][None],
                    )
                lat = act_quant_fp4(rotate_tail(latent, rd, fc, fs), 16, e4m3_scale=True)
                ckv = ckv.at[:b, start_pos // ratio : start_pos // ratio + lat.shape[1]].set(lat)
            st = dict(st, compress_kv_cache=ckv)
            shared["compress_kv"] = ckv
        kv_all = jnp.concatenate([window_kv, shared["compress_kv"][:b, :compress_len]], axis=1)
        topk = jnp.concatenate([topk, cidx], axis=-1)
    else:
        kv_all = window_kv

    o = sparse_attn(q, kv_all, P["attn_sink"], topk, a.head_dim**-0.5)
    o = rotate_tail(o, rd, cos, sin, inverse=True)
    g = a.o_groups
    o = act_quant_fp8(o.reshape(b, s, g, -1))
    wo_a = P["wo_a.weight"].reshape(g, a.o_lora_rank, -1)
    o = jnp.einsum("bsgd,grd->bsgr", o, wo_a)
    return o.reshape(b, s, -1) @ P["wo_b.weight"].T, st


def P_sub(P, prefix):
    return {k[len(prefix) :]: v for k, v in P.items() if k.startswith(prefix)}


# ----------------------------------------------------------------------------- FFN / block


def expert(w1, w2, w3, x, limit):
    gate = (x @ w1.T).astype(jnp.float32)
    up = (x @ w3.T).astype(jnp.float32)
    if limit > 0:
        up = jnp.clip(up, -limit, limit)
        gate = jnp.minimum(gate, limit)
    return (jax.nn.silu(gate) * up).astype(x.dtype) @ w2.T


def dequant_int8_experts(w, s, dtype):
    """int8 [E,N,K] (2*e2m1) * 2**(e-127) * 0.5 per 32-block along K -> dtype [E,N,K] (exact for FP4 checkpoints)."""
    e, n, k = w.shape
    scale = jnp.ldexp(jnp.float32(0.5), s.astype(jnp.int32) - 127)
    return (
        (w.reshape(e, n, k // 32, 32).astype(jnp.float32) * scale[..., None])
        .reshape(e, n, k)
        .astype(dtype)
    )


def _moe_fp8_kernel(P, xf, w, idx, a: Args, tok_mask):
    """Routed experts through SGL-JAX's fused expert-parallel Pallas kernel (EP over all mesh devices, fp8 weights with per-channel
    scales, dequantized in VMEM). Needs a multiple of ep_size tokens: pad with routing copies of a live token at weight 0 (no extra
    expert weight reads). Weights must be in kernel layout (sharding.to_kernel_layout)."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as PS

    from sgl_jax.srt.kernels.fused_moe.v2.kernel import (
        FusedMoEBlockConfig,
        fused_ep_moe_v2,
    )

    # Anchor the kernel's inputs as replicated: otherwise XLA propagates the kernel's 32-way token sharding backward into the
    # attention/cache updates (it then all-reduces every per-row cache across the tensor-parallel axis each layer).
    # Measured on v6e-32 (decode step, ms; B rows): anchored vs not -> B=4: 21.9 vs 67.6, B=16: 28.1 vs 84.3, B=64: 54.3 vs 36.7. For
    # large token counts the leaked sharding is a win (attention work gets split over the tensor-parallel axis), so only anchor small N.
    if xf.shape[0] <= int(os.environ.get("V41_KERNEL_ANCHOR_MAX_TOKENS", "32")):
        rep2 = NamedSharding(MOE_MESH, PS(None, None))
        xf, idx, w = (jax.lax.with_sharding_constraint(v, rep2) for v in (xf, idx, w))
        if tok_mask is not None:
            tok_mask = jax.lax.with_sharding_constraint(tok_mask, NamedSharding(MOE_MESH, PS(None)))
    n, k = idx.shape
    ep = MOE_MESH.shape["dp"] * MOE_MESH.shape["tp"]
    npad = -(-n // ep) * ep
    if tok_mask is None:
        tok_mask = jnp.ones((n,), bool)
    first = jnp.argmax(tok_mask)  # first live token (0 if none)
    # idle/padding tokens route to expert id -1 (nowhere): routing them all to one live token's experts overloaded those devices and made the
    # fused kernel 2-6x slower (measured: N=256, 3 live: 2.0 ms vs 0.35 ms per layer); outputs of the live tokens are identical
    ids = jnp.where(tok_mask[:, None], idx, -1).astype(jnp.int32)
    wts = jnp.where(tok_mask[:, None], w, 0.0).astype(jnp.float32)
    x = xf
    if npad > n:
        ids = jnp.concatenate([ids, jnp.full((npad - n, k), -1, jnp.int32)], 0)
        wts = jnp.concatenate([wts, jnp.zeros((npad - n, k), jnp.float32)], 0)
        x = jnp.concatenate([xf, jnp.zeros((npad - n, xf.shape[1]), xf.dtype)], 0)
    tok_sh = NamedSharding(MOE_MESH, PS(("dp", "tp"), None))
    x, wts, ids = (jax.lax.with_sharding_constraint(v, tok_sh) for v in (x, wts, ids))
    bt, bf, btc = (int(v) for v in os.environ.get("V41_MOE_BLOCK", "128,1152,128").split(","))
    out = fused_ep_moe_v2(
        MOE_MESH,
        x,
        P["experts.w1"],
        P["experts.w2"],
        P["experts.w3"],
        wts,
        ids,
        k,
        act_fn="silu",
        swiglu_limit=a.swiglu_limit if a.swiglu_limit > 0 else None,
        block_config=FusedMoEBlockConfig(bt=bt, bf=bf, btc=btc, bse=256),
        quant_block_k=None,
        w1_scale=P["experts.w1_rowscale"],
        w2_scale=P["experts.w2_rowscale"],
        w3_scale=P["experts.w3_rowscale"],
        direct_scaled_dot=True,
        dp_axis_name="dp",
        tp_axis_name="tp",
    )
    out = out[:n]
    if os.environ.get("V41_MOE_OUT", "dp") == "dp" and n % MOE_MESH.shape["dp"] == 0:
        # keep the batch rows sharded over the data-parallel axis like the residual stream (forcing full replication over all 32
        # devices made XLA all-reduce the per-row KV caches in the following attention)
        return jax.lax.with_sharding_constraint(out, NamedSharding(MOE_MESH, PS("dp", None)))
    return jax.lax.with_sharding_constraint(out, NamedSharding(MOE_MESH, PS(None, None)))


def _moe_fp8_rows(P, xf, a: Args, shape, tok_mask=None):
    scores = (xf.astype(jnp.float32) @ P["gate.weight"].T.astype(jnp.float32)) / a.gate_temp
    scores = (
        jnp.sqrt(jax.nn.softplus(scores))
        if a.score_func == "sqrtsoftplus"
        else (
            jax.nn.softmax(scores, axis=-1) if a.score_func == "softmax" else jax.nn.sigmoid(scores)
        )
    )
    _, idx = jax.lax.top_k(scores + P["gate.bias"], a.n_activated_experts)
    w = jnp.take_along_axis(scores, idx, axis=1)
    if a.norm_topk_prob and a.n_activated_experts > 1:
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
    w = w * a.route_scale
    if (
        tok_mask is not None
    ):  # idle batch rows route to no expert (keeps the sparse path from reading experts for them)
        w = w * tok_mask[:, None].astype(w.dtype)
    e = a.n_routed_experts
    if (
        P["experts.w1_rowscale"].ndim == 4
    ):  # kernel layout (V41_MOE=kernel + sharding.to_kernel_layout)
        if xf.shape[0] <= int(
            os.environ.get("V41_KERNEL_MIN_TOKENS", "8")
        ):  # decode-sized: per-used-expert loop (lower latency)
            onehot = (idx[..., None] == jnp.arange(e)).astype(jnp.float32)
            coef = jnp.einsum("nk,nke->ne", w, onehot)
            y = _moe_fp8_loop(P, xf, coef, a)
        else:
            y = _moe_fp8_kernel(P, xf, w, idx, a, tok_mask).astype(jnp.float32)
        y = y + expert(
            P["shared_experts.w1.weight"],
            P["shared_experts.w2.weight"],
            P["shared_experts.w3.weight"],
            xf,
            a.swiglu_limit,
        ).astype(jnp.float32)
        return y.astype(xf.dtype).reshape(shape)
    mode = os.environ.get(
        "V41_MOE", "loop"
    )  # loop: per-used-expert loop for small N, dense above; grouped: per-expert token groups; dense
    if MOE_MESH is not None and mode == "grouped":
        onehot = (idx[..., None] == jnp.arange(e)).astype(jnp.float32)
        coef = jnp.einsum("nk,nke->ne", w, onehot)
        sel = jnp.any(
            (onehot > 0) & (w[..., None] != 0), axis=1
        )  # [n,e]: expert chosen by a live token (idle rows have w == 0)
        y = _moe_fp8_grouped(P, xf, sel, coef, a)
        y = y + expert(
            P["shared_experts.w1.weight"],
            P["shared_experts.w2.weight"],
            P["shared_experts.w3.weight"],
            xf,
            a.swiglu_limit,
        ).astype(jnp.float32)
        return y.astype(xf.dtype).reshape(shape)
    if (
        MOE_MESH is not None
        and mode == "loop"
        and xf.shape[0] <= int(os.environ.get("V41_SPARSE_MAX_TOKENS", "64"))
    ):
        onehot = (idx[..., None] == jnp.arange(e)).astype(jnp.float32)
        coef = jnp.einsum("nk,nke->ne", w, onehot)
        y = _moe_fp8_loop(P, xf, coef, a)
        y = y + expert(
            P["shared_experts.w1.weight"],
            P["shared_experts.w2.weight"],
            P["shared_experts.w3.weight"],
            xf,
            a.swiglu_limit,
        ).astype(jnp.float32)
        return y.astype(xf.dtype).reshape(shape)
    xb = xf
    cdt = xf.dtype
    gate = (
        jnp.einsum(
            "nd,efd->enf", xb, P["experts.w1"].astype(cdt), preferred_element_type=jnp.float32
        )
        * P["experts.w1_rowscale"][:, None, :]
    )
    up = (
        jnp.einsum(
            "nd,efd->enf", xb, P["experts.w3"].astype(cdt), preferred_element_type=jnp.float32
        )
        * P["experts.w3_rowscale"][:, None, :]
    )
    if a.swiglu_limit > 0:
        up = jnp.clip(up, -a.swiglu_limit, a.swiglu_limit)
        gate = jnp.minimum(gate, a.swiglu_limit)
    act = (jax.nn.silu(gate) * up).astype(cdt)
    out = (
        jnp.einsum(
            "enf,edf->end", act, P["experts.w2"].astype(cdt), preferred_element_type=jnp.float32
        )
        * P["experts.w2_rowscale"][:, None, :]
    )
    onehot = (idx[..., None] == jnp.arange(e)).astype(jnp.float32)
    coef = jnp.einsum("nk,nke->ne", w, onehot)
    y = jnp.einsum("ne,end->nd", coef, out)
    y = y + expert(
        P["shared_experts.w1.weight"],
        P["shared_experts.w2.weight"],
        P["shared_experts.w3.weight"],
        xf,
        a.swiglu_limit,
    ).astype(jnp.float32)
    return y.astype(xf.dtype).reshape(shape)


# Sparse expert path (decode): set MOE_MESH to the device mesh to enable. Each device loops only over the local experts that
# at least one token routed to (exact; reads just those experts' weights instead of all local experts).
MOE_MESH = None
MOE_AXES = ("dp", "tp")


MOE_CHUNK = int(
    os.environ.get("V41_MOE_CHUNK", "128")
)  # rows per expert matmul (tokens per expert per weight read)


def _moe_fp8_grouped(P, xf, sel, coef, a: Args):
    """Grouped expert matmul (exact). Per device and per local expert: gather the tokens routed to it, run [C,d] matmuls in chunks
    of C rows (dynamic trip count, zero iterations if no token chose the expert), scatter-add the weighted outputs; psum over devices.
    Each local expert's weights are read at most once per C routed tokens, so a step costs ~one pass over the weights it needs.
    """
    try:
        from jax import shard_map
    except ImportError:  # older jax
        from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as PS

    ax = MOE_AXES
    cdt = xf.dtype
    C = MOE_CHUNK

    def local(x, cf, sl, w1, w3, w2, s1, s3, s2):
        n = x.shape[0]
        el = cf.shape[1]
        npad = -(-n // C) * C + C

        def per_expert(e, y):
            m = sl[:, e]
            cnt = jnp.sum(m.astype(jnp.int32))
            order = jnp.argsort(jnp.logical_not(m).astype(jnp.int32), stable=True).astype(
                jnp.int32
            )  # routed tokens first
            order = jnp.concatenate([order, jnp.zeros((npad - n,), jnp.int32)])

            def chunk(j, y):
                rows = jax.lax.dynamic_slice_in_dim(order, j * C, C)
                valid = (j * C + jnp.arange(C)) < cnt
                xr = x[rows]
                g = jnp.dot(xr, w1[e].astype(cdt).T, preferred_element_type=jnp.float32) * s1[e]
                u = jnp.dot(xr, w3[e].astype(cdt).T, preferred_element_type=jnp.float32) * s3[e]
                if a.swiglu_limit > 0:
                    u = jnp.clip(u, -a.swiglu_limit, a.swiglu_limit)
                    g = jnp.minimum(g, a.swiglu_limit)
                h = (jax.nn.silu(g) * u).astype(cdt)
                o = jnp.dot(h, w2[e].astype(cdt).T, preferred_element_type=jnp.float32) * s2[e]
                cw = jnp.where(valid, cf[rows, e], 0.0)
                return y.at[rows].add(o * cw[:, None])

            return jax.lax.fori_loop(0, (cnt + C - 1) // C, chunk, y)

        y = jax.lax.fori_loop(0, el, per_expert, jnp.zeros((n, a.dim), jnp.float32))
        return jax.lax.psum(y, ax)

    e_spec = PS(ax)
    specs = dict(
        mesh=MOE_MESH,
        in_specs=(PS(), PS(None, ax), PS(None, ax), e_spec, e_spec, e_spec, e_spec, e_spec, e_spec),
        out_specs=PS(),
    )
    try:
        fn = shard_map(local, check_vma=False, **specs)
    except TypeError:
        fn = shard_map(local, check_rep=False, **specs)
    return fn(
        xf,
        coef,
        sel,
        P["experts.w1"],
        P["experts.w3"],
        P["experts.w2"],
        P["experts.w1_rowscale"],
        P["experts.w3_rowscale"],
        P["experts.w2_rowscale"],
    )


def _moe_fp8_loop(P, xf, coef, a: Args):
    try:
        from jax import shard_map
    except ImportError:  # older jax
        from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as PS

    ax = MOE_AXES
    cdt = xf.dtype
    kl = (
        P["experts.w1_rowscale"].ndim == 4
    )  # kernel layout: w1,w3 [E,D,F], w2 [E,F,D], scales [E,1,1,X] (no transpose needed)

    def local(x, cf, w1, w3, w2, s1, s3, s2):
        used = jnp.any(cf != 0, axis=0)  # [E_local]
        mm = (
            (lambda h, w: jnp.dot(h, w.astype(cdt), preferred_element_type=jnp.float32))
            if kl
            else (lambda h, w: jnp.dot(h, w.astype(cdt).T, preferred_element_type=jnp.float32))
        )
        order = jnp.argsort(jnp.logical_not(used).astype(jnp.int32), stable=True)
        cnt = jnp.sum(used.astype(jnp.int32))

        def body(j, y):
            e = order[j]
            g = mm(x, w1[e]) * s1[e].reshape(-1)
            u = mm(x, w3[e]) * s3[e].reshape(-1)
            if a.swiglu_limit > 0:
                u = jnp.clip(u, -a.swiglu_limit, a.swiglu_limit)
                g = jnp.minimum(g, a.swiglu_limit)
            h = (jax.nn.silu(g) * u).astype(cdt)
            o = mm(h, w2[e]) * s2[e].reshape(-1)
            return y + jax.lax.dynamic_index_in_dim(cf, e, axis=1, keepdims=True) * o

        y = jax.lax.fori_loop(0, cnt, body, jnp.zeros((x.shape[0], a.dim), jnp.float32))
        return jax.lax.psum(y, ax)

    e_spec = PS(ax)
    specs = dict(
        mesh=MOE_MESH,
        in_specs=(PS(), PS(None, ax), e_spec, e_spec, e_spec, e_spec, e_spec, e_spec),
        out_specs=PS(),
    )
    try:
        fn = shard_map(local, check_vma=False, **specs)
    except TypeError:
        fn = shard_map(local, check_rep=False, **specs)
    return fn(
        xf,
        coef,
        P["experts.w1"],
        P["experts.w3"],
        P["experts.w2"],
        P["experts.w1_rowscale"],
        P["experts.w3_rowscale"],
        P["experts.w2_rowscale"],
    )


def moe(P, x, a: Args, tok_mask=None):
    shape = x.shape
    xf = x.reshape(-1, a.dim)
    scores = (xf.astype(jnp.float32) @ P["gate.weight"].T.astype(jnp.float32)) / a.gate_temp
    if a.score_func == "softmax":
        scores = jax.nn.softmax(scores, axis=-1)
    elif a.score_func == "sigmoid":
        scores = jax.nn.sigmoid(scores)
    else:
        scores = jnp.sqrt(jax.nn.softplus(scores))
    _, idx = jax.lax.top_k(scores + P["gate.bias"], a.n_activated_experts)
    w = jnp.take_along_axis(scores, idx, axis=1)
    if a.norm_topk_prob and a.n_activated_experts > 1:
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
    w = w * a.route_scale
    e = a.n_routed_experts
    if (
        "experts.w1_rowscale" in P
    ):  # TPU production: float8_e4m3fn [E,N,K] + float32 row scales [E,N] applied after the matmul
        return _moe_fp8_rows(P, xf, a, shape, tok_mask)
    if (
        "experts.w1_scale" in P
    ):  # int8 + block scales: int8 (2*e2m1) [E,N,K] + uint8 e8m0 exponents [E,N,K/32], dequantized on the fly
        w1, w2, w3 = (
            dequant_int8_experts(P[f"experts.{w}"], P[f"experts.{w}_scale"], xf.dtype)
            for w in ("w1", "w2", "w3")
        )
    elif "experts.w1" in P:  # pre-stacked [E, N, K] float weights
        w1, w2, w3 = P["experts.w1"], P["experts.w2"], P["experts.w3"]
    else:
        w1 = jnp.stack([P[f"experts.{i}.w1.weight"] for i in range(e)])
        w2 = jnp.stack([P[f"experts.{i}.w2.weight"] for i in range(e)])
        w3 = jnp.stack([P[f"experts.{i}.w3.weight"] for i in range(e)])
    gate = jnp.einsum("nd,efd->enf", xf, w1).astype(jnp.float32)
    up = jnp.einsum("nd,efd->enf", xf, w3).astype(jnp.float32)
    if a.swiglu_limit > 0:
        up = jnp.clip(up, -a.swiglu_limit, a.swiglu_limit)
        gate = jnp.minimum(gate, a.swiglu_limit)
    act = jax.nn.silu(gate) * up  # [e,n,f]
    out = jnp.einsum("enf,edf->end", act, w2.astype(jnp.float32))  # [e,n,d]
    onehot = (idx[..., None] == jnp.arange(e)).astype(jnp.float32)  # [n,k,e]
    coef = jnp.einsum("nk,nke->ne", w, onehot)  # [n,e]
    y = jnp.einsum("ne,end->nd", coef, out)
    y = y + expert(
        P["shared_experts.w1.weight"],
        P["shared_experts.w2.weight"],
        P["shared_experts.w3.weight"],
        xf,
        a.swiglu_limit,
    ).astype(jnp.float32)
    return y.astype(x.dtype).reshape(shape)


def hc_mixes(x, fn, scale, base, a: Args):
    xf = x.reshape(*x.shape[:2], -1).astype(jnp.float32)
    rs = jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + a.norm_eps)
    return hc_split_sinkhorn(
        (xf @ fn.T) * rs, scale, base, a.hc_mult, a.hc_sinkhorn_iters, a.hc_eps
    )


def hc_pre(x, pre):
    return jnp.sum(pre[..., None] * x.astype(jnp.float32), axis=2).astype(x.dtype)


def hc_post(x, residual, post, comb):
    y = post[..., None] * x[:, :, None, :] + jnp.sum(
        comb[..., None] * residual[:, :, :, None, :], axis=2
    )
    return y.astype(x.dtype)


def block(P, x, st, start_pos, layer, pre_mix, a: Args, shared, rope):
    residual = x
    ap, apost, acomb = hc_mixes(x, P["hc_attn_fn"], P["hc_attn_scale"], P["hc_attn_base"], a)
    h = hc_pre(x, pre_mix)
    h = rms_norm(h, P["attn_norm.weight"], a.norm_eps)
    h, st = attention(P_sub(P, "attn."), h, st, start_pos, layer, a, shared, rope)
    x = hc_post(h, residual, apost, acomb)
    residual = x
    fpre, fpost, fcomb = hc_mixes(x, P["hc_ffn_fn"], P["hc_ffn_scale"], P["hc_ffn_base"], a)
    h = hc_pre(x, ap)
    h = rms_norm(h, P["ffn_norm.weight"], a.norm_eps)
    h = moe(P_sub(P, "ffn."), h, a)
    return hc_post(h, residual, fpost, fcomb), fpre, st


def make_rope(a: Args, layer):
    ratio = a.compress_ratios[layer]
    if ratio:
        return freqs_cis(
            a.rope_head_dim,
            a.max_seq_len,
            a.original_seq_len,
            a.compress_rope_theta,
            a.rope_factor,
            a.beta_fast,
            a.beta_slow,
        )
    return freqs_cis(
        a.rope_head_dim, a.max_seq_len, 0, a.rope_theta, a.rope_factor, a.beta_fast, a.beta_slow
    )


def init_state(a: Args, bsz):
    sts = []
    for layer in range(a.n_layers):
        ratio = a.compress_ratios[layer]
        st = {"window_kv_cache": jnp.zeros((bsz, a.window_size, a.head_dim), jnp.float32)}
        if ratio and layer in a.kv_source_layers:
            st["compress_kv_cache"] = jnp.zeros(
                (bsz, a.max_seq_len // ratio, a.head_dim), jnp.float32
            )
            if ratio > 1:
                st["compressor"] = {
                    "kv_state": jnp.zeros((bsz, ratio, a.head_dim), jnp.float32),
                    "score_state": jnp.full((bsz, ratio, a.head_dim), -jnp.inf, jnp.float32),
                }
            else:
                st["compressor"] = {}
        if ratio and layer in a.index_source_layers:
            ist = {}
            if layer in a.kv_source_layers:
                ist["k_cache"] = jnp.zeros(
                    (bsz, a.max_seq_len // ratio, a.index_head_dim), jnp.float32
                )
            st["indexer"] = ist
        sts.append(st)
    return sts


def forward(params, a: Args, input_ids, start_pos, states, ropes, engram=None):
    """Returns (logits [b, vocab], new states). params: flat dict keyed like the reference state_dict.
    engram: optional dict(hasher=NgramHasher, round_bf16=bool); table rows are read from params (test) or mmap (prod).
    """
    import engram_jax as E

    hash_ids = engram["hasher"](np.asarray(input_ids), start_pos) if engram else None
    h = params["embed.weight"][input_ids]
    h = jnp.repeat(h[:, :, None, :], a.hc_mult, axis=2)
    pre_mix = jnp.zeros((*h.shape[:2], a.hc_mult), jnp.float32).at[:, :, 0].set(1.0)
    shared = {}
    new_states = []
    for layer in range(a.n_layers):
        Pl = P_sub(params, f"layers.{layer}.")
        if engram and layer in a.engram_layer_ids:
            idx = a.engram_layer_ids.index(layer)
            rows = E.gather_rows(
                Pl["engram.embed.weight"], Pl["engram.embed.scale"], hash_ids[:, :, idx, :]
            )
            if engram.get("round_bf16"):
                rows = np.asarray(jnp.asarray(rows).astype(jnp.bfloat16).astype(jnp.float32))
            h = E.engram_gate(
                h, jnp.asarray(rows), P_sub(Pl, "engram."), a.hc_mult, a.dim, a.norm_eps
            )
        h, pre_mix, st = block(
            Pl, h, states[layer], start_pos, layer, pre_mix, a, shared, ropes[layer]
        )
        new_states.append(st)
    h = hc_pre(h, pre_mix)
    h = rms_norm(h, params["norm.weight"], a.norm_eps)
    logits = h[:, -1].astype(jnp.float32) @ params["head.weight"].astype(jnp.float32).T
    return logits, new_states
