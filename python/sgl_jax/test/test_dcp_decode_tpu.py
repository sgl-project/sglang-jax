"""Slice 7 decode isolation: ``sparse_mla_page_level`` at ``dcp=16`` + LSE merge.

Debugging the D2' decode divergence through full server boots costs ~9 minutes a
try. This runs only the suspect component on the 16-device slice and compares the
merged result against a dense numpy softmax over the whole context, so a dropped
shard / mis-listed page shows up directly.

The geometry deliberately mirrors the failing 8k case: a context whose last DSA
page is partially filled, ranks with unequal owned page counts, and ranks that own
nothing near the tail.

Run via the 2-host launcher (a single-host jax init hangs)::

    JAX_DIST_ADDR=host0:63104 JAX_NPROC=2 JAX_PID=0|1 \\
      python -m sgl_jax.test.test_dcp_decode_tpu
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

DCP = 16
PS = 128  # page_size; I == PS is the block interleave
DV = 512  # kv_lora_rank
ROPE = 64
DK_PAD = 640
H = 8
SEQ_LEN = 3000  # 24 DSA pages (last one partial: 3000 % 128 == 56)


def _init_distributed() -> int:
    addr = os.environ.get("JAX_DIST_ADDR")
    if not addr:
        raise SystemExit("JAX_DIST_ADDR is required (2-host TPU slice)")
    nproc = int(os.environ.get("JAX_NPROC", "2"))
    pid = int(os.environ.get("JAX_PID", "0"))
    jax.distributed.initialize(coordinator_address=addr, num_processes=nproc, process_id=pid)
    return pid


def owner(v: int) -> int:
    return (v // PS) % DCP


def phys_of(v: int) -> int:
    return (v // (PS * DCP)) * PS + (v % PS)


def owned_len(upto: int, rank: int) -> int:
    full, rem = upto // PS, upto % PS
    whole = (full - rank + DCP - 1) // DCP
    tail = rem if (full % DCP) == rank else 0
    return whole * PS + tail


def main() -> None:
    pid = _init_distributed()
    devices = jax.devices()
    if len(devices) != 16 or jax.local_device_count() != 8:
        raise SystemExit(
            f"need 16 global / 8 local, got {len(devices)} / {jax.local_device_count()}"
        )

    rng = np.random.default_rng(3)
    n_dsa = (SEQ_LEN + PS - 1) // PS  # 24 real DSA pages
    vpage = PS * DCP
    pages_per_seq = (SEQ_LEN + vpage - 1) // vpage  # physical pages per rank
    sm_scale = 1.0 / np.sqrt(DV + ROPE)

    # ── ground truth: dense softmax of one query over the whole context ──────
    # bf16-round the KV first so the reference sees exactly the kernel's values.
    # Scale so the scores actually SPREAD: at 0.05 every score is ~0 and the LSE
    # collapses to log(n_keys), which made this check blind to wrong keys.
    kv = jnp.asarray(rng.standard_normal((SEQ_LEN, DV + ROPE)) * 1.4, jnp.bfloat16)
    kv_f = np.asarray(kv, np.float32)
    q = jnp.asarray(rng.standard_normal((1, H, DV + ROPE)) * 1.4, jnp.bfloat16)
    q_f = np.asarray(q, np.float32)

    s = np.einsum("thd,kd->thk", q_f, kv_f) * sm_scale
    if pid == 0:
        print(f"score spread: std={s.std():.3f} min={s.min():.2f} max={s.max():.2f}")
        assert s.std() > 0.5, "scores too flat -> LSE check would be insensitive"
    m = s.max(-1, keepdims=True)
    e = np.exp(s - m)
    ref = np.einsum("thk,kd->thd", e, kv_f[:, :DV]) / e.sum(-1, keepdims=True)

    if pid == 0:
        print(f"seq_len={SEQ_LEN} dsa_pages={n_dsa} pages_per_seq={pages_per_seq} H={H}")
        cnt = [owned_len(SEQ_LEN, r) for r in range(DCP)]
        print(f"owned_len per rank: {cnt}  sum={sum(cnt)} (must equal {SEQ_LEN})")
        print(f"owner of last token = rank {owner(SEQ_LEN - 1)}")

    # ── per-rank shards, stacked so shard_map hands each rank its own ────────
    Pn = pages_per_seq + 2  # page 0 reserved, last page sentinel
    cache_all = np.zeros((DCP, Pn, PS // 2, 2, DK_PAD), np.float32)
    topk_all = np.full((DCP, 1, pages_per_seq), -1, np.int32)
    for r in range(DCP):
        for v in range(SEQ_LEN):
            if owner(v) != r:
                continue
            p = phys_of(v)
            # seq page p//PS is pool page (p//PS)+1; see page_indices below
            pool_page = p // PS + 1
            off = p % PS
            cache_all[r, pool_page, off // 2, off % 2, : DV + ROPE] = kv_f[v]
        owned = [d // DCP for d in range(n_dsa) if d % DCP == r]
        topk_all[r, 0, : len(owned)] = owned

    cache_j = jnp.asarray(cache_all, jnp.bfloat16)
    topk_j = jnp.asarray(topk_all)
    page_indices = jnp.asarray(np.arange(1, pages_per_seq + 1, dtype=np.int32))
    kv_lens = jnp.asarray(np.array([SEQ_LEN], np.int32))
    cu_q = jnp.asarray(np.array([0, 1], np.int32))
    cu_kv = jnp.asarray(np.array([0, pages_per_seq * PS], np.int32))
    dist = jnp.asarray(np.array([1, 1, 1], np.int32))
    # re-write the last token with its own value: makes the kernel's mandatory
    # new-token write a no-op so this test isolates the ATTEND, not the write.
    new_kv_c = jnp.asarray(kv_f[SEQ_LEN - 1 : SEQ_LEN, :DV], jnp.bfloat16)
    new_k_pe = jnp.asarray(kv_f[SEQ_LEN - 1 : SEQ_LEN, DV:], jnp.bfloat16)

    mesh = Mesh(np.array(devices).reshape(1, 16), ("data", "tensor"))

    def kernel(cache_r, topk_r):
        from sgl_jax.srt.kernels.dsa.sparse_mla import sparse_mla_page_level

        rank = jax.lax.axis_index("tensor")
        o, _cache, lse = sparse_mla_page_level(
            q[:, :, :DV],
            q[:, :, DV:],
            new_kv_c,
            new_k_pe,
            cache_r[0],
            kv_lens,
            jnp.full((1, 1), -1, jnp.int32),  # token-level topk unused
            page_indices,
            cu_q,
            cu_kv,
            dist,
            topk_r[0],
            sm_scale=float(sm_scale),
            page_size=PS,
            pages_per_seq=pages_per_seq,
            kv_lora_rank=DV,
            k_pages_max=pages_per_seq + 1,
            return_lse=True,
            dcp_size=DCP,
            dcp_rank=rank,
            dcp_interleave=PS,
        )
        # unmerged, so a bad shard is attributable to its rank
        return o, lse

    fn = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("tensor", None, None, None, None), P("tensor", None, None)),
            out_specs=(P("tensor", None, None), P("tensor", None)),
            check_vma=False,
        )
    )
    o_r, lse_r = jax.block_until_ready(fn(cache_j, topk_j))
    o_all = np.asarray(process_allgather(o_r, tiled=True)).reshape(DCP, H, DV)
    lse_all = np.asarray(process_allgather(lse_r, tiled=True)).reshape(DCP, H)
    if pid != 0:
        return

    # per-shard reference: dense softmax over this rank's owned keys, in physical order
    bad = []
    for r in range(DCP):
        keys = np.stack([kv_f[v] for v in range(SEQ_LEN) if owner(v) == r])
        assert keys.shape[0] == owned_len(SEQ_LEN, r)
        sr = np.einsum("hd,kd->hk", q_f[0], keys) * sm_scale
        mr = sr.max(-1, keepdims=True)
        er = np.exp(sr - mr)
        o_ref = np.einsum("hk,kd->hd", er, keys[:, :DV]) / er.sum(-1, keepdims=True)
        l_ref = (mr.squeeze(-1) + np.log(er.sum(-1))).astype(np.float32)
        de = np.abs(o_all[r].astype(np.float32) - o_ref).max()
        dl = np.abs(lse_all[r] - l_ref).max()
        rel = de / max(np.abs(o_ref).max(), 1e-9)
        flag = "OK " if rel < 3e-2 and dl < 5e-2 else "BAD"
        if flag == "BAD":
            bad.append(r)
        print(
            f"  rank {r:2d} keys={keys.shape[0]:4d} o_rel={rel:.4f} "
            f"dlse={dl:.4f} lse_got={lse_all[r][0]:.4f} lse_ref={l_ref[0]:.4f}  {flag}"
        )
    print("bad ranks:", bad if bad else "none")

    # merged check, from the gathered partials
    mx = lse_all.max(0)
    w = np.exp(lse_all - mx)
    merged = (o_all.astype(np.float32) * w[:, :, None]).sum(0) / w.sum(0)[:, None]
    err = np.abs(merged - ref[0]).max()
    rel = err / max(np.abs(ref[0]).max(), 1e-9)
    print(f"merged max|err|={err:.6f} rel={rel:.5f}")
    print("DECODE-DCP PASS" if rel < 3e-2 and not bad else "DECODE-DCP FAIL")
    if bad or rel >= 3e-2:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
