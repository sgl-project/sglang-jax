"""DP parity for the packed-ragged DSA sparse prefill path.

Run with: JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
    python -m pytest python/sgl_jax/test/test_dsa_sparse_prefill_dp.py -v

Drives ``DSASparseAttentionBackend``, not the kernel: under ``shard_map`` the
kernel only ever sees its own shard, so a kernel-level DP test would be
exercising ``shard_map`` rather than sglang-jax. What DP puts at risk is the
wiring above it — ``pages_per_seq`` derived from *local* shard shapes, seq-local
``topk_pages`` resolved against a per-request ``page_indices`` base, token-axis
arrays lining up with the ``ql``/``kvc`` sections, and page ids that are
**rank-local** into a page axis sharded ``P("data")``.

Attention TP is held fixed between each reference and its DP run (the meshes
differ only in how many devices sit on the ``data`` axis), so a difference can
only come from DP, never from a changed head split.

Most cases use ``attn_tp=2`` rather than 1. Sharding over a size-1 axis is a
no-op, so at ``attn_tp=1`` a spec that wrongly read ``P("data", None, None)``
would behave exactly like the correct ``P("data", "tensor", None)`` and the
tensor half of every spec would go unverified.
"""

from __future__ import annotations

import os

# Must precede the first jax import: the DP meshes below need >1 CPU device.
# Append rather than setdefault — XLA_FLAGS set for any unrelated reason
# (--xla_dump_to=..., a profiling flag) would otherwise drop the device count,
# leaving one device, and this file is the DP gate in unit-test-cpu.
if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
    ).strip()
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import functools
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention import dsa_sparse_backend as dsa_mod
from sgl_jax.srt.layers.attention.dsa_sparse_backend import DSASparseAttentionBackend
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import MLATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase

# Fail at import rather than skip: on CPU the device count is forced above, so
# too few devices means the flag did not take effect, not that the machine is
# small. Skipping there would report a green run with zero DP coverage.
if jax.default_backend() == "cpu" and jax.device_count() < 2:
    raise RuntimeError(
        f"expected >=2 simulated CPU devices, got {jax.device_count()}; "
        f"XLA_FLAGS={os.environ.get('XLA_FLAGS')!r} did not take effect"
    )

# Small but structurally faithful to GLM-5.2: kv_lora_rank and index_head_dim
# both land on a 128 boundary after align_to, and page_size is a multiple of 16
# (the paged sparse kernel requires read_block % 16 == 0, and read_block
# defaults to page_size).
NUM_HEADS = 4
KV_LORA_RANK = 128
QK_NOPE_DIM = 128
QK_ROPE_DIM = 64
V_HEAD_DIM = 128
PAGE_SIZE = 16
# The indexer key cache pads its feature dim to a multiple of 128 and the
# backend scatters k_idx into it unpadded, so this must already be 128-aligned.
INDEX_HEAD_DIM = 128
INDEX_N_HEADS = 4
INDEX_TOPK = 32  # -> k_pages = 2 selected pages per query
PAGES_PER_RANK = 16  # KV pages owned by each DP rank
LAYER_ID = 0
DTYPE = jnp.float32
# The backend queries the TPU for a VMEM budget when this is None, which fails
# on CPU. The value only bounds a Pallas scratch budget the interpreter ignores.
VMEM_LIMIT = 64 * 1024 * 1024


def _mesh(dp_size: int, attn_tp: int) -> jax.sharding.Mesh:
    """Mesh over the first ``dp_size * attn_tp`` devices."""
    n = dp_size * attn_tp
    if n > jax.device_count():
        raise unittest.SkipTest(f"need {n} devices, have {jax.device_count()}")
    return create_device_mesh(
        ici_parallelism=[dp_size, attn_tp],
        dcn_parallelism=[1, 1],
        device_indexes=list(range(n)),
    )


class _Workload:
    """A set of prefill requests, addressed by (dp_rank, request).

    Single-shot extend: every request starts at position 0, so ``kv_len ==
    q_len`` and the prefill kernel's self-write populates the whole cache. That
    is the case DP is actually about — one sequence per rank, prefilled
    concurrently — and it keeps the fixture free of a separate prefix-load step.
    """

    def __init__(self, q_lens_per_rank: list[list[int]], seed: int = 0, num_heads: int = NUM_HEADS):
        self.q_lens_per_rank = q_lens_per_rank
        self.num_heads = num_heads
        self.dp_size = len(q_lens_per_rank)
        self.per_dp_bs = max(len(r) for r in q_lens_per_rank)
        # Every request's KV is padded up to a page boundary, and every rank
        # holds the same number of pages, so `pages_per_seq` is uniform — the
        # layout `streamindex_page_topk_ref` slices with.
        self.pages_per_seq = PAGES_PER_RANK // self.per_dp_bs
        max_len = self.pages_per_seq * PAGE_SIZE
        for rank in q_lens_per_rank:
            for q_len in rank:
                assert q_len <= max_len, f"q_len {q_len} exceeds {max_len} per-request slots"
        self.per_dp_tokens = self.per_dp_bs * max_len
        self.max_len = max_len

        rng = np.random.default_rng(seed)
        # Per (rank, request) payloads, generated independently of dp_size so
        # the same request can be replayed in a different DP layout.
        self.payloads: dict[tuple[int, int], dict] = {}
        for r, rank in enumerate(q_lens_per_rank):
            for i, q_len in enumerate(rank):
                self.payloads[(r, i)] = {
                    "q_len": q_len,
                    "ql": rng.standard_normal((q_len, num_heads, KV_LORA_RANK)),
                    "qpe": rng.standard_normal((q_len, num_heads, QK_ROPE_DIM)),
                    "kvc": rng.standard_normal((q_len, KV_LORA_RANK)),
                    "kpe": rng.standard_normal((q_len, QK_ROPE_DIM)),
                    "q_idx": rng.standard_normal((q_len, INDEX_N_HEADS, INDEX_HEAD_DIM)),
                    "k_idx": rng.standard_normal((q_len, INDEX_HEAD_DIM)),
                    "idx_w": rng.standard_normal((q_len, INDEX_N_HEADS)),
                }

    def subset(self, ranks: list[int]) -> _Workload:
        """The same requests, re-laid-out over ``len(ranks)`` DP ranks."""
        out = _Workload([self.q_lens_per_rank[r] for r in ranks], seed=0, num_heads=self.num_heads)
        assert out.per_dp_bs == self.per_dp_bs, "subset must preserve per-rank batch size"
        for new_r, old_r in enumerate(ranks):
            for i in range(len(self.q_lens_per_rank[old_r])):
                out.payloads[(new_r, i)] = self.payloads[(old_r, i)]
        return out


def _build(workload: _Workload, mesh: jax.sharding.Mesh, cache_loc=None):
    """Materialise a workload as (backend, forward_batch, pool, sharded inputs).

    Token arrays are ``dp_size`` equal sections with each rank's valid rows
    packed at its section start — the layout ``tp_worker`` produces and that
    ``P("data")`` slices back apart. Padding rows carry ``out_cache_loc == -1``.
    """
    dp = workload.dp_size
    per_dp_bs = workload.per_dp_bs
    per_dp_tokens = workload.per_dp_tokens
    total_bs = dp * per_dp_bs
    total_tokens = dp * per_dp_tokens

    def _tok(shape_tail, dtype=np.float32):
        return np.zeros((total_tokens, *shape_tail), dtype=dtype)

    num_heads = workload.num_heads
    ql = _tok((num_heads, KV_LORA_RANK))
    qpe = _tok((num_heads, QK_ROPE_DIM))
    kvc = _tok((KV_LORA_RANK,))
    kpe = _tok((QK_ROPE_DIM,))
    q_idx = _tok((INDEX_N_HEADS, INDEX_HEAD_DIM))
    k_idx = _tok((INDEX_HEAD_DIM,))
    idx_w = _tok((INDEX_N_HEADS,))

    positions = np.zeros(total_tokens, dtype=np.int32)
    out_cache_loc = np.full(total_tokens, -1, dtype=np.int32)
    seq_lens = np.zeros(total_bs, dtype=np.int32)
    extend_seq_lens = np.zeros(total_bs, dtype=np.int32)
    extend_prefix_lens = np.zeros(total_bs, dtype=np.int32)
    # cache_loc is per-rank contiguous over every request's page-aligned slots;
    # MLA's get_forward_metadata strides it by page_size to get page_indices.
    per_dp_loc = PAGES_PER_RANK * PAGE_SIZE
    if cache_loc is None:
        cache_loc = np.zeros(dp * per_dp_loc, dtype=np.int32)
        _fill_default_cache_loc = True
    else:
        cache_loc = np.asarray(cache_loc, dtype=np.int32)
        _fill_default_cache_loc = False
    # Slot ids are RANK-LOCAL: the KV page axis is sharded P("data"), so each
    # rank indexes its own [PAGES_PER_RANK] window. This mirrors the allocator,
    # which hands out rank-local ids from a `size_per_rank` free list.
    for r in range(dp):
        if _fill_default_cache_loc:
            cache_loc[r * per_dp_loc : (r + 1) * per_dp_loc] = np.arange(per_dp_loc, dtype=np.int32)

    # Track where each request's rows landed so callers can slice outputs back
    # out per request rather than reasoning about the padded layout.
    row_spans: dict[tuple[int, int], tuple[int, int]] = {}

    for r in range(dp):
        tok_base = r * per_dp_tokens
        # Both offsets must match what `get_forward_metadata` derives, because
        # the kernel recovers a query's request from `cu_q_lens` and bases its
        # page table at `cu_kv_lens[rid] // page_size`:
        #   cu_q_lens  = cumsum of extend_seq_lens  -> queries PACKED, no gaps
        #   cu_kv_lens = cumsum of page-aligned seq_lens -> KV page-aligned
        # Spacing requests any other way puts their rows outside every segment,
        # so they silently produce zeros.
        cum_tok = 0
        cum_kv = 0
        for i in range(per_dp_bs):
            slot = r * per_dp_bs + i
            payload = workload.payloads.get((r, i))
            if payload is None:
                continue
            q_len = payload["q_len"]
            lo = tok_base + cum_tok
            hi = lo + q_len
            row_spans[(r, i)] = (lo, hi)

            ql[lo:hi] = payload["ql"]
            qpe[lo:hi] = payload["qpe"]
            kvc[lo:hi] = payload["kvc"]
            kpe[lo:hi] = payload["kpe"]
            q_idx[lo:hi] = payload["q_idx"]
            k_idx[lo:hi] = payload["k_idx"]
            idx_w[lo:hi] = payload["idx_w"]

            positions[lo:hi] = np.arange(q_len)
            out_cache_loc[lo:hi] = np.arange(cum_kv, cum_kv + q_len, dtype=np.int32)
            seq_lens[slot] = q_len
            extend_seq_lens[slot] = q_len

            cum_tok += q_len
            cum_kv += ((q_len + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE

    pool = MLATokenToKVPool(
        size=PAGE_SIZE * PAGES_PER_RANK * dp,
        page_size=PAGE_SIZE,
        dtype=DTYPE,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_DIM,
        layer_num=1,
        mesh=mesh,
        dp_size=dp,
        indexer_key_dim=INDEX_HEAD_DIM,
        num_indexer_layers=1,
    )

    backend = DSASparseAttentionBackend(
        index_topk=INDEX_TOPK,
        index_head_dim=INDEX_HEAD_DIM,
        index_n_heads=INDEX_N_HEADS,
        skip_offset=0,
        full_slot={LAYER_ID: 0},
        num_attn_heads=num_heads,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_DIM,
        qk_rope_head_dim=QK_ROPE_DIM,
        v_head_dim=V_HEAD_DIM,
        page_size=PAGE_SIZE,
        mesh=mesh,
        attention_data_partition_axis="data",
        vmem_limit_bytes=VMEM_LIMIT,
    )

    mwb = ModelWorkerBatch(
        bid=1,
        forward_mode=ForwardMode.EXTEND,
        input_ids=np.zeros(total_tokens, dtype=np.int32),
        real_input_ids_len=total_tokens,
        seq_lens=seq_lens,
        out_cache_loc=out_cache_loc,
        req_pool_indices=np.arange(total_bs, dtype=np.int32),
        sampling_info=None,
        positions=positions,
        cache_loc=cache_loc,
        extend_seq_lens=extend_seq_lens,
        extend_prefix_lens=extend_prefix_lens,
        return_logprob=False,
        return_output_logprob_only=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        logits_indices=np.zeros(total_bs, dtype=np.int32),
        real_bs=total_bs,
        real_bs_per_dp=[len(workload.q_lens_per_rank[r]) for r in range(dp)],
        dp_size=dp,
        per_dp_bs_size=per_dp_bs,
        spec_info_padded=None,
    )
    backend.forward_metadata = backend.get_forward_metadata(mwb)

    # The mesh runs in Explicit-axis mode, so shard_map requires each input's
    # committed sharding to match its in_spec exactly — device_put, don't rely
    # on inference.
    def put(x, spec):
        return jax.device_put(jnp.asarray(x, dtype=DTYPE), NamedSharding(mesh, spec))

    def put_i32(x, spec):
        return jax.device_put(jnp.asarray(x, dtype=jnp.int32), NamedSharding(mesh, spec))

    inputs = {
        "ql": put(ql, P("data", "tensor", None)),
        "qpe": put(qpe, P("data", "tensor", None)),
        "kvc": put(kvc, P("data", None)),
        "kpe": put(kpe, P("data", None)),
        "q_idx": put(q_idx, P("data", None, None)),
        "k_idx": put(k_idx, P("data", None)),
        "idx_w": put(idx_w, P("data", None)),
    }
    for field in ("seq_lens", "page_indices", "cu_q_lens", "cu_kv_lens", "distribution"):
        setattr(
            backend.forward_metadata,
            field,
            put_i32(getattr(backend.forward_metadata, field), P("data")),
        )

    # ForwardBatch.init_new is the production entrypoint (see test_flashattention_dp.py).
    # It device_puts every per-token/per-request array under
    # NamedSharding(mesh, P("data")) in one call -- exactly the DP token layout
    # this file is testing, so it would be self-defeating to hand-roll it here.
    dummy_model_config = type(
        "DummyModelConfig",
        (),
        {"is_embedding": False, "hf_config": type("DummyHFConfig", (), {"architectures": []})()},
    )()
    dummy_runner = type(
        "DummyRunner",
        (),
        {"mesh": mesh, "attn_backend": backend, "model_config": dummy_model_config},
    )()
    fb = ForwardBatch.init_new(mwb, dummy_runner)

    return backend, fb, pool, inputs, row_spans


def _run(workload: _Workload, mesh: jax.sharding.Mesh):
    """Run one sparse-prefill EXTEND pass; return {(rank, req): output rows}."""
    backend, fb, pool, inputs, row_spans = _build(workload, mesh)

    layer = RadixAttention(
        num_heads=workload.num_heads,
        head_dim=QK_NOPE_DIM + QK_ROPE_DIM,
        scaling=1.0 / np.sqrt(QK_NOPE_DIM + QK_ROPE_DIM),
        num_kv_heads=1,
        layer_id=LAYER_ID,
        v_head_dim=V_HEAD_DIM,
    )

    with jax.set_mesh(mesh):
        o, _ = backend(
            inputs["ql"],
            inputs["kvc"],
            inputs["kvc"],
            layer,
            fb,
            pool,
            q_rope=inputs["qpe"],
            k_rope=inputs["kpe"],
            indexer_type="full",
            q_idx=inputs["q_idx"],
            k_idx=inputs["k_idx"],
            idx_weights=inputs["idx_w"],
        )
    o = np.asarray(jax.device_get(o), dtype=np.float64)
    return {key: o[lo:hi] for key, (lo, hi) in row_spans.items()}


class TestDSASparsePrefillDP(CustomTestCase):
    """DP parity gates for the packed-ragged DSA sparse prefill."""

    def setUp(self):
        # CPU is covered by the import-time check; a real accelerator with too
        # few devices is a genuine skip rather than a misconfiguration.
        if jax.device_count() < 2:
            self.skipTest(f"needs >=2 devices, have {jax.device_count()}")
        # The sparse prefill path is opt-in behind DSA_PREFILL_SPARSE; the
        # module reads it at import time into a constant.
        # Both patches are torn down by addCleanup, so a failure part-way
        # through setUp cannot leak module state into the next test.
        self.enterContext(mock.patch.object(dsa_mod, "_PREFILL_SPARSE", 1))
        # Run the Pallas sparse kernel under the CPU interpreter. This is the
        # only stub in the test: the backend calls the kernel directly with no
        # hook, and `interpret` is a documented parameter of that same entry
        # point — so the wiring under test (specs, metadata, page ids) is
        # untouched. A wrapper class in the style of KDAAttnBackendForTest
        # cannot reach it, because the seam is a module-level function rather
        # than a method on the backend.
        self.enterContext(
            mock.patch.object(
                dsa_mod,
                "prefill_write_and_attend_ragged",
                functools.partial(dsa_mod.prefill_write_and_attend_ragged, interpret=True),
            )
        )

    def _assert_dp_invariant(self, q_lens_per_rank, attn_tp, num_heads=NUM_HEADS):
        """Each rank's requests must give the same answer alone at dp=1 as they
        do inside the dp=D batch. Attention TP is identical in both runs."""
        dp = len(q_lens_per_rank)
        full = _Workload(q_lens_per_rank, num_heads=num_heads)
        got = _run(full, _mesh(dp, attn_tp))

        solos = {}
        for r in range(dp):
            solo = full.subset([r])
            want = _run(solo, _mesh(1, attn_tp))
            solos[r] = want
            for i in range(len(q_lens_per_rank[r])):
                # A request whose rows land outside every cu_q_lens segment
                # produces all zeros, and zeros match zeros. Check every
                # compared tensor carries signal before believing the match.
                self.assertGreater(
                    np.abs(got[(r, i)]).max(),
                    1e-6,
                    f"dp={dp} rank {r} req {i} output is all zeros — the request "
                    "never reached the kernel, so the comparison is vacuous",
                )
                np.testing.assert_allclose(
                    got[(r, i)],
                    want[(0, i)],
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"dp={dp} rank {r} req {i} differs from its dp=1 reference",
                )

        # Sensitivity control: the comparison above must be able to fail, so
        # each reference has to be distinguishable from the others. Covers every
        # compared request, not just index 0 — a blind spot there is exactly how
        # an all-zero request survives.
        refs = [(r, i, solos[r][(0, i)]) for r in range(dp) for i in range(len(q_lens_per_rank[r]))]
        for a in range(len(refs)):
            for b in range(a + 1, len(refs)):
                r_a, i_a, x = refs[a]
                r_b, i_b, y = refs[b]
                if x.shape != y.shape:
                    continue
                self.assertGreater(
                    np.abs(x - y).max(),
                    1e-3,
                    f"references for rank {r_a} req {i_a} and rank {r_b} req {i_b} "
                    "are indistinguishable — the invariance assertion cannot fail",
                )

    def test_dp_invariance_dp2_one_request_per_rank(self):
        """The core DP case: one sequence per rank, prefilled concurrently."""
        self._assert_dp_invariant([[96], [96]], attn_tp=2)

    def test_dp_invariance_dp4(self):
        """Four ranks, all different lengths — cu_q_lens/cu_kv_lens diverge per shard."""
        self._assert_dp_invariant([[96], [64], [112], [80]], attn_tp=2)

    def test_dp_invariance_dp2_with_batching(self):
        """Several requests per rank, on top of the packed-ragged batching."""
        self._assert_dp_invariant([[64, 48], [80, 32]], attn_tp=2)

    def test_dp_invariance_dp2_tp2_wide_heads(self):
        """DP and TP together, with a head block wider than the default.

        ``attention_tp = tp_size // dp_size``, so raising ``dp`` *widens* each
        device's head block. 16 heads over 2-way TP gives 8 per device, versus 4
        in the tests above — enough to cover both sides of the kernel's sublane
        padding (``Hq = ceil(H/16)*16``) without a separate sweep.

        The coverage this gives is shape plumbing only: the Pallas interpreter
        does not enforce Mosaic's tiling, so a shape that passes here can still
        fail to compile on device.
        """
        self._assert_dp_invariant([[96], [64]], attn_tp=2, num_heads=16)

    def test_cross_rank_no_bleed(self):
        """Rank 0's output must not depend on rank 1's *tokens*.

        Perturbing the KV pool would prove nothing here: single-shot prefill
        self-writes every token into the cache before attending, so any
        pre-seeded page content is overwritten and unreachable. The live lever
        is rank 1's input. Rank 1's tokens are self-written into pages carrying
        the same rank-local ids as rank 0's, so a global-vs-local index confusion
        — or a shard reaching past its own page window — moves rank 0's output.
        """
        mesh = _mesh(2, 2)
        base = _Workload([[96], [96]], seed=0)
        # Same rank-0 payload, different rank-1 payload.
        other = _Workload([[96], [96]], seed=99)
        other.payloads[(0, 0)] = base.payloads[(0, 0)]

        a = _run(base, mesh)
        b = _run(other, mesh)

        # Positive control: the two runs must genuinely differ somewhere, else
        # the assertion below is comparing two identical computations.
        rank1_delta = np.abs(a[(1, 0)] - b[(1, 0)]).max()
        self.assertGreater(rank1_delta, 1e-3, "rank 1 payload swap had no effect — test is vacuous")

        np.testing.assert_allclose(
            b[(0, 0)],
            a[(0, 0)],
            rtol=1e-6,
            atol=1e-6,
            err_msg="rank 0 output changed when only rank 1's tokens changed",
        )

    def test_metadata_shards_evenly(self):
        """`pages_per_seq` is derived from LOCAL shard shapes inside shard_map
        (`page_indices.shape[0] // seq_lens.shape[0]`). If either array fails to
        divide by dp_size, or the quotient drifts from the dp=1 value, every
        page id the indexer emits is silently wrong rather than an error."""
        # attn_tp=1 on purpose: that is the production mesh shape at dp16
        # (attention_tp = tp_size // dp_size), and it is the one case where the
        # tensor axis is genuinely size 1 in a real deployment.
        workload = _Workload([[96], [64]])
        mesh = _mesh(2, 1)
        backend, _, _, _, _ = _build(workload, mesh)
        md = backend.forward_metadata

        dp = workload.dp_size
        self.assertEqual(len(md.page_indices) % dp, 0, "page_indices must shard evenly")
        self.assertEqual(len(md.seq_lens) % dp, 0, "seq_lens must shard evenly")

        local_pages = len(md.page_indices) // dp
        local_seqs = len(md.seq_lens) // dp
        self.assertEqual(
            local_pages // local_seqs,
            workload.pages_per_seq,
            "per-shard pages_per_seq differs from the layout the fixture built",
        )

    def test_page_indices_are_rank_local_and_unmixed(self):
        """`get_forward_metadata` must stride `cache_loc` into page ids without
        adding a rank offset and without mixing ranks.

        The allocator hands out **rank-local** slot ids (`size_per_rank`), and the
        KV page axis is sharded ``P("data")``, so each shard indexes its own
        window. Each rank is given a *different* rank-local layout here: with the
        same layout on every rank, "rank r matches rank 0" holds by construction
        and no implementation could fail it.
        """
        dp = 2
        per_dp_loc = PAGES_PER_RANK * PAGE_SIZE
        # Rank r's window rotated by r pages. Still entirely rank-local — every
        # id < per_dp_loc — but distinguishable between ranks.
        cache_loc = np.concatenate(
            [(np.arange(per_dp_loc) + r * PAGE_SIZE) % per_dp_loc for r in range(dp)]
        ).astype(np.int32)

        workload = _Workload([[96], [64]])
        backend, _, _, _, _ = _build(workload, _mesh(dp, 1), cache_loc=cache_loc)
        pages = np.asarray(backend.forward_metadata.page_indices).reshape(dp, -1)

        for r in range(dp):
            expected = cache_loc[r * per_dp_loc : (r + 1) * per_dp_loc][::PAGE_SIZE] // PAGE_SIZE
            np.testing.assert_array_equal(
                pages[r],
                expected,
                err_msg=f"rank {r}'s page ids do not match its own cache_loc — "
                "ranks are being mixed or offset",
            )
        # A global (rather than rank-local) id space would push ids past the
        # per-rank window.
        self.assertLess(
            int(pages.max()),
            PAGES_PER_RANK,
            "a page id reaches past its rank's own window — ids are not rank-local",
        )
        # Sensitivity control: the per-rank checks above must be able to fail.
        self.assertFalse(
            np.array_equal(pages[0], pages[1]),
            "the two ranks were given identical layouts — the per-rank assertions "
            "would hold for any implementation",
        )


if __name__ == "__main__":
    unittest.main()
