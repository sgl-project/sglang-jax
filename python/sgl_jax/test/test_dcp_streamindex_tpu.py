"""Slice 8 gate: the streaming Pallas indexer == the JAX reference, under DCP.

The DCP prefill indexer used ``score_prefill_local_jax``, which materializes
``[T, local_kv]`` f32 — 2.1 GiB/layer at 1M and the long-context OOM. Routing it
through ``streamindex_page_topk(return_page_scores=True)`` removes that, but the
kernel derives its own causal bound: ``k_span`` is a *physical* slot while query
positions are *virtual*, so the bound must be ``owned_len(pos+1) - 1``. This test
is what pins that, per rank, against the reference it replaces.

Run via the 2-host launcher (single-host jax init hangs)::

    JAX_DIST_ADDR=host0:63108 JAX_NPROC=2 JAX_PID=0|1 \\
      python -m sgl_jax.test.test_dcp_streamindex_tpu
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

DCP = 16
PS = 128  # page_size
DIM = 128  # indexer head_dim; kernel asserts head_dim % 128 == 0
HEADS = 8
SEQ_LEN = 3000  # virtual; chunk-0 prefill so q_len == seq_len
PACK = 2  # bf16 kv_packing
VPAGE = PS * DCP
PAGES_PER_SEQ = -(-SEQ_LEN // VPAGE)  # physical pages this rank holds
TOTAL_PAGES = PAGES_PER_SEQ + 2  # + a reserved page and some slack


def _init() -> int:
    addr = os.environ.get("JAX_DIST_ADDR")
    if not addr:
        raise SystemExit("JAX_DIST_ADDR is required (2-host TPU slice)")
    jax.distributed.initialize(
        coordinator_address=addr,
        num_processes=int(os.environ.get("JAX_NPROC", "2")),
        process_id=int(os.environ.get("JAX_PID", "0")),
    )
    return int(os.environ.get("JAX_PID", "0"))


def main() -> None:
    pid = _init()
    devices = jax.devices()
    if len(devices) != 16 or jax.local_device_count() != 8:
        raise SystemExit(
            f"need 16 global / 8 local, got {len(devices)} / {jax.local_device_count()}"
        )

    rng = np.random.default_rng(5)
    t = SEQ_LEN
    q = jnp.asarray(rng.standard_normal((t, HEADS, DIM)) * 0.7, jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((t, HEADS)) * 0.5, jnp.bfloat16)
    # Cache in the native paged layout; the same bytes feed both scorers.
    cache4d = jnp.asarray(
        rng.standard_normal((TOTAL_PAGES, PS // PACK, PACK, DIM)) * 1.1, jnp.bfloat16
    )
    seq_lens = jnp.asarray([SEQ_LEN], jnp.int32)
    # Skip page 0 the way the allocator reserves it, so a wrong page base shows up.
    page_indices = jnp.asarray(np.arange(1, PAGES_PER_SEQ + 1), jnp.int32)
    cu_q_lens = jnp.asarray([0, t], jnp.int32)
    cu_kv_lens = jnp.asarray([0, SEQ_LEN], jnp.int32)
    positions = jnp.asarray(np.arange(t), jnp.int32)
    num_seqs = jnp.asarray(1, jnp.int32)

    mesh = Mesh(np.array(devices).reshape(1, 16), ("data", "tensor"))

    def kernel(_):
        from sgl_jax.srt.kernels.dsa.streamindex_topk import streamindex_page_topk
        from sgl_jax.srt.layers.dcp.indexer import (
            local_page_max_jax,
            score_prefill_local_jax,
        )

        rank = jax.lax.axis_index("tensor")
        cache3d = cache4d.reshape(TOTAL_PAGES, PS, DIM)

        ref = local_page_max_jax(
            score_prefill_local_jax(
                q,
                w,
                cache3d,
                seq_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                positions,
                PAGES_PER_SEQ,
                PS,
                DCP,
                rank,
                interleave=PS,
            ),
            PS,
        )
        got = streamindex_page_topk(
            q,
            w,
            cache4d,
            seq_lens,
            page_indices,
            cu_q_lens,
            num_seqs,
            k_pages=PAGES_PER_SEQ,
            dcp_size=DCP,
            dcp_rank=rank,
            dcp_interleave=PS,
            return_page_scores=True,
        )
        # Stack so every rank's pair comes back separately: a bug that hits only
        # some ranks (the bound is rank-dependent) must not average away.
        return jnp.stack([ref, got])[None]

    fn = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=P("data"),
            # Shard the leading axis over `tensor` so all 16 ranks' pairs come
            # back; P("data", ...) would collapse them to one rank's copy.
            out_specs=P("tensor", None, None, None),
            check_vma=False,
        )
    )
    t0 = time.perf_counter()
    out = np.asarray(process_allgather(jax.block_until_ready(fn(jnp.zeros((1,)))), tiled=True))
    took = time.perf_counter() - t0
    if pid != 0:
        return

    print(f"compile+run {took:.2f}s  out={out.shape}")
    assert out.shape[0] == DCP, out.shape
    worst = 0.0
    for r in range(DCP):
        ref, got = out[r, 0], out[r, 1]
        finite = np.isfinite(ref)
        # -inf must agree exactly: those are the masked (unowned / non-causal)
        # pages, and a disagreement means the causal bound is wrong.
        assert np.array_equal(finite, np.isfinite(got)), f"rank {r}: -inf mask differs"
        d = float(np.abs(ref[finite] - got[finite]).max()) if finite.any() else 0.0
        worst = max(worst, d)
        print(f"  rank {r:2d}: live pages={int(finite.sum()):6d}  max|d|={d:.5f}")

    # Sensitivity guard: a flat score field would pass any comparison.
    spread = float(np.nanstd(np.where(np.isfinite(out[0, 0]), out[0, 0], np.nan)))
    print(f"score spread std={spread:.3f}  worst max|d|={worst:.5f} ({worst / spread:.2%})")
    assert spread > 0.5, "scores too flat -> comparison would be insensitive"
    # bf16 inputs with a different accumulation order: gate relatively, not on an
    # absolute epsilon. The mask equality above is the exact part.
    assert worst / spread < 0.05, f"max|d|={worst} vs spread {spread}"

    # The gate that actually matters: the same pages must be SELECTED. Rank r's
    # physical page P is global page P*DCP+r, so interleave with rank fastest.
    n_glob = PAGES_PER_SEQ * DCP
    glob_ref = out[:, 0].transpose(1, 2, 0).reshape(t, n_glob)
    glob_got = out[:, 1].transpose(1, 2, 0).reshape(t, n_glob)
    k = max(1, n_glob // 4)
    sel_ref = np.sort(np.argsort(-glob_ref, kind="stable")[:, :k], axis=1)
    sel_got = np.sort(np.argsort(-glob_got, kind="stable")[:, :k], axis=1)
    differ = (sel_ref != sel_got).any(axis=1)
    rows_differ = int(differ.sum())
    print(f"top-{k} of {n_glob} global pages: {rows_differ}/{t} query rows differ")

    # A flip is only benign if it is a near-tie at the k-th boundary: the pages
    # swapped must be within the score drift of each other. A flip across a WIDE
    # gap would mean the scores are actually wrong, and widening a tolerance
    # would hide exactly that. Check the gap on the reference's own scale.
    if rows_differ:
        gaps = []
        for row in np.flatnonzero(differ):
            order = np.argsort(-glob_ref[row], kind="stable")
            kth, next_ = glob_ref[row][order[k - 1]], glob_ref[row][order[k]]
            gaps.append(abs(kth - next_))
        worst_gap = float(max(gaps))
        print(f"  boundary gap on flipped rows: max={worst_gap:.5f} (drift {worst:.5f})")
        assert worst_gap <= 3 * worst, (
            f"a flip crossed a gap of {worst_gap} >> drift {worst}: scores are wrong, "
            "not merely reordered at a tie"
        )
    assert rows_differ / t < 0.01, f"{rows_differ}/{t} rows differ"
    print(
        f"streamindex-DCP PASS (masks exact; {worst / spread:.2%} score drift; "
        f"{rows_differ}/{t} near-tie selection flips)"
    )


if __name__ == "__main__":
    main()
