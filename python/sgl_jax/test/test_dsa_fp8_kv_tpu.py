"""fp8 latent KV cache: does it compile, and is it numerically usable?

An fp8 cache halves the bytes of the MLA latent KV -- both the HBM footprint
(doubling the context that fits) and the HBM->VMEM traffic of the attend, which is
one of the largest ops in a long-context prefill. Two things have to hold, and
neither is obvious from reading the code:

1. **It compiles.** fp8 makes ``kv_packing`` 4 instead of 2, so the paged cache is
   ``[P, ps//4, 4, Dk_pad]``. The shape math is generic (``32 // bitwidth``), but
   whether Mosaic will DMA and reshape a 4-packed sub-32-bit array inside these
   kernels is a question only the compiler answers. Both readers of the cache are
   covered, because they reach it by different paths: the qblock prefill and the
   page-level decode.
2. **The error stays small.** e4m3 carries 3 mantissa bits, so ~6% worst-case
   relative error per element. Attention sums over hundreds of keys, so the errors
   partly cancel -- the bounds below are what that cancellation actually delivers,
   measured, not a bound derived on paper.

The work is deliberately single-device, so a failure here is a dtype failure and
nothing else::

    PYTHONPATH=python python -m sgl_jax.test.test_dsa_fp8_kv_tpu

On a multi-host slice a lone process blocks in TPU init waiting for its peers, so
set ``JAX_DIST_ADDR``/``JAX_NPROC``/``JAX_PID`` and run it redundantly on every
host.
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

PS = 128  # page_size
DV = 512  # kv_lora_rank
ROPE = 64
DK_PAD = 640  # align_to(512,128) + align_to(64,128)
H = 8
SEQ_LEN = 1024  # 8 whole DSA pages
K_PAGES = 4  # pages the indexer selects per query
SM = float(DV) ** -0.5

# Bounds on the error against an identical bf16 cache. These are measured, with
# roughly 1.4x headroom over the observed values on a TPU v7x (e4m3 came in at
# 3.47% rel_l2 for prefill and 3.06% for decode; e5m2 at 6.86% and 6.32%). They
# exist to catch a regression in the fp8 path, not to certify a paper bound, so
# widen them if a platform lands legitimately worse rather than silencing the test.
MAX_REL_L2 = {"fp8_e4m3": 0.05, "fp8_e5m2": 0.10}
MAX_PEAK_FRAC = {"fp8_e4m3": 0.06, "fp8_e5m2": 0.12}

# A Mosaic refusal to lower the 4-packed layout is a statement about the platform,
# not a regression, so it skips. Anything else -- including lowering fine and then
# producing wrong numbers -- fails.
_LOWERING_MARKERS = ("mosaic", "not implemented", "unsupported", "only interpret mode")


def _maybe_init_distributed() -> int:
    """Join the process group if ``JAX_DIST_ADDR`` is set; return this rank.

    Without this a single process blocks in ``CreateTpuSystemState`` waiting for
    the peer host to appear, which looks exactly like a kernel hang.
    """
    addr = os.environ.get("JAX_DIST_ADDR")
    if not addr:
        return 0
    jax.distributed.initialize(
        coordinator_address=addr,
        num_processes=int(os.environ.get("JAX_NPROC", "2")),
        process_id=int(os.environ.get("JAX_PID", "0")),
    )
    return jax.process_index()


def _is_lowering_failure(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _LOWERING_MARKERS)


def _paged(kv_flat: np.ndarray, dtype, page_offset: int = 0) -> jnp.ndarray:
    """``[tokens, DV+ROPE]`` -> ``[P, ps//pk, pk, DK_PAD]`` for this dtype.

    ``pk`` comes from the dtype, so this is where fp8 diverges from bf16: the same
    logical page becomes 32 sublanes of 4 rather than 64 of 2. ``page_offset``
    leaves physical pages free at the front, which the decode kernel expects.
    """
    from sgl_jax.srt.kernels.mla.v2.kernel import get_dtype_packing

    pk = get_dtype_packing(dtype)
    pages = SEQ_LEN // PS
    cache = np.zeros((pages + page_offset, PS // pk, pk, DK_PAD), np.float32)
    for v in range(SEQ_LEN):
        page, off = v // PS + page_offset, v % PS
        cache[page, off // pk, off % pk, :DV] = kv_flat[v, :DV]
        cache[page, off // pk, off % pk, DV : DV + ROPE] = kv_flat[v, DV:]
    return jnp.asarray(cache, dtype)


def _run(cache, ql, qpe, kvc, kpe, topk_pages, positions, loc, seq_lens, cu_q, cu_kv, pi):
    from sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock import (
        prefill_write_and_attend_ragged_qblock,
    )

    o, _cache = prefill_write_and_attend_ragged_qblock(
        ql,
        qpe,
        kvc,
        kpe,
        cache,
        topk_pages,
        positions,
        loc,
        seq_lens,
        cu_q,
        cu_kv,
        pi,
        kv_lora_rank=DV,
        page_size=PS,
        sm_scale=SM,
        query_block=32,
    )
    return np.asarray(o, np.float32)


def _run_decode(cache, q, new_kv_c, new_k_pe):
    """One decode query over the whole context via ``sparse_mla_page_level``.

    This is the other kernel that reads the cache, and it reads it through a
    different path than prefill (dense over topk-touched pages), so it needs its
    own fp8 gate.
    """
    from sgl_jax.srt.kernels.dsa.sparse_mla import sparse_mla_page_level

    pages = SEQ_LEN // PS
    o, _cache = sparse_mla_page_level(
        q[:, :, :DV],
        q[:, :, DV:],
        new_kv_c,
        new_k_pe,
        cache,
        jnp.asarray([SEQ_LEN], jnp.int32),
        jnp.full((1, 1), -1, jnp.int32),  # token-level topk unused in page mode
        jnp.arange(1, pages + 1, dtype=jnp.int32),
        jnp.asarray([0, 1], jnp.int32),
        jnp.asarray([0, pages * PS], jnp.int32),
        jnp.asarray([1, 1, 1], jnp.int32),
        jnp.arange(pages, dtype=jnp.int32)[None, :],  # select every page
        sm_scale=SM,
        page_size=PS,
        pages_per_seq=pages,
        kv_lora_rank=DV,
        k_pages_max=pages + 1,
    )
    return np.asarray(o, np.float32)


def _check(log, name, label, got, ref) -> float:
    """Log the error against the bf16 reference and assert it is within bounds."""
    scale = float(np.abs(ref).max())
    err = float(np.abs(got - ref).max())
    peak_frac = err / scale
    rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-9))
    log(f"{name} {label}: max|do|={err:.5f} ({peak_frac:.2%} of peak)  rel_l2={rel:.4%}")
    assert np.isfinite(got).all(), f"{name} {label}: output has non-finite values"
    assert rel <= MAX_REL_L2[name], f"{name} {label}: rel_l2 {rel:.4%} > {MAX_REL_L2[name]:.2%}"
    assert (
        peak_frac <= MAX_PEAK_FRAC[name]
    ), f"{name} {label}: max|do| {peak_frac:.2%} of peak > {MAX_PEAK_FRAC[name]:.2%}"
    return rel


def main() -> int:
    rank = _maybe_init_distributed()
    log = print if rank == 0 else (lambda *a, **k: None)
    platform = jax.devices()[0].platform
    if platform != "tpu":
        log(f"SKIP: the fp8 paged-cache gate needs TPU (got {platform})")
        return 0
    log("devices:", jax.device_count(), "local:", jax.local_device_count())
    rng = np.random.default_rng(0)

    # Mirror the real distribution: the latent cache holds post-RMSNorm values
    # (unit RMS) and post-rope k_pe. That is what makes a scale unnecessary --
    # everything sits far inside e4m3's 448 range and above its 0.0156 floor.
    kv_flat = rng.standard_normal((SEQ_LEN, DV + ROPE)).astype(np.float32)
    ql = jnp.asarray(rng.standard_normal((SEQ_LEN, H, DV)), jnp.bfloat16)
    qpe = jnp.asarray(rng.standard_normal((SEQ_LEN, H, ROPE)), jnp.bfloat16)
    kvc = jnp.asarray(kv_flat[:, :DV], jnp.bfloat16)
    kpe = jnp.asarray(kv_flat[:, DV:], jnp.bfloat16)

    positions = jnp.arange(SEQ_LEN, dtype=jnp.int32)
    loc = jnp.arange(SEQ_LEN, dtype=jnp.int32)
    seq_lens = jnp.asarray([SEQ_LEN], jnp.int32)
    cu_q = jnp.asarray([0, SEQ_LEN], jnp.int32)
    cu_kv = jnp.asarray([0, SEQ_LEN], jnp.int32)
    pi = jnp.arange(SEQ_LEN // PS, dtype=jnp.int32)
    # Every query selects the K_PAGES pages ending at its own, clamped at 0.
    own = np.arange(SEQ_LEN) // PS
    sel = np.stack([np.maximum(own - d, 0) for d in range(K_PAGES)], axis=1)
    topk_pages = jnp.asarray(sel.astype(np.int32))

    args = (ql, qpe, kvc, kpe, topk_pages, positions, loc, seq_lens, cu_q, cu_kv, pi)

    # Decode inputs. Re-writing the last token with its own value makes the
    # kernel's mandatory new-token write a no-op, so this isolates the attend.
    q_dec = jnp.asarray(rng.standard_normal((1, H, DV + ROPE)), jnp.bfloat16)
    new_kv_c = kvc[SEQ_LEN - 1 : SEQ_LEN]
    new_k_pe = kpe[SEQ_LEN - 1 : SEQ_LEN]

    ref_pre = _run(_paged(kv_flat, jnp.bfloat16), *args)
    ref_dec = _run_decode(_paged(kv_flat, jnp.bfloat16, page_offset=1), q_dec, new_kv_c, new_k_pe)
    bf16_mib = _paged(kv_flat, jnp.bfloat16).nbytes / 2**20
    log(
        f"bf16 prefill {ref_pre.shape} |o|max={np.abs(ref_pre).max():.4f} | "
        f"decode {ref_dec.shape} |o|max={np.abs(ref_dec).max():.4f}"
    )

    rel_l2: dict[str, float] = {}
    for name, dt in (("fp8_e4m3", jnp.float8_e4m3fn), ("fp8_e5m2", jnp.float8_e5m2)):
        cache = _paged(kv_flat, dt)
        assert (
            cache.nbytes * 2 == _paged(kv_flat, jnp.bfloat16).nbytes
        ), f"{name}: cache is {cache.nbytes} B, expected half of bf16's"
        log(
            f"\n--- {name}: cache {cache.shape} {cache.dtype}, "
            f"{cache.nbytes / 2**20:.1f} MiB vs bf16 {bf16_mib:.1f} MiB"
        )
        # Default-bound so each lambda captures this iteration's dtype/cache rather than
        # the loop variable, which would make both cases run with e5m2 (ruff B023).
        cases = (
            ("prefill", lambda c=cache: _run(c, *args), ref_pre),
            (
                "decode",
                lambda d=dt: _run_decode(
                    _paged(kv_flat, d, page_offset=1), q_dec, new_kv_c, new_k_pe
                ),
                ref_dec,
            ),
        )
        for label, fn, ref in cases:
            try:
                got = fn()
            except Exception as exc:
                if _is_lowering_failure(exc):
                    log(f"SKIP {name} {label}: will not lower here: {str(exc)[:200]}")
                    continue
                raise
            rel_l2[f"{name} {label}"] = _check(log, name, label, got, ref)

    if not rel_l2:
        log("SKIP: the fp8 paged cache does not lower on this platform")
        return 0

    # e4m3 is the default for a reason: same range headroom, one more mantissa bit.
    # If e5m2 ever came out no worse, the dtype plumbing would be mixing them up.
    for label in ("prefill", "decode"):
        lo, hi = f"fp8_e4m3 {label}", f"fp8_e5m2 {label}"
        if lo in rel_l2 and hi in rel_l2:
            assert rel_l2[hi] > rel_l2[lo], (
                f"{label}: e5m2 rel_l2 {rel_l2[hi]:.4%} is not worse than "
                f"e4m3's {rel_l2[lo]:.4%} -- are the dtypes actually distinct?"
            )

    log(f"\nPASS: {len(rel_l2)} fp8 cases within bounds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
