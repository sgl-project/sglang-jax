"""QSA sparse backend CPU tests: the backend's own assembly.

Runs in the ``unit-test-cpu`` suite. The indexer, the paging and the kernels
are pinned by their own tests; these pin what the backend adds around them --
which inputs a QSA layer requires, what each data-parallel rank reads, and the
cache layout, page table and batch segmentation each kernel is handed.

The selection goes through the real ``streamindex_topk`` dispatcher. Only its
Pallas launch, which cannot run on CPU, is replaced (``_scores_launch``).

Two CPU devices, so that a data-parallel batch can run on two ranks.

Run:
    python -m unittest python.sgl_jax.test.layers.test_qsa_sparse_backend
"""

from __future__ import annotations

import functools
import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_NUM_CPU_DEVICES", "2")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental import pallas as pl
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.dsa import streamindex_topk
from sgl_jax.srt.kernels.qsa.ref import sparse_gqa_attention_ref
from sgl_jax.srt.kernels.qsa.sparse_gqa_attention import sparse_gqa_attention
from sgl_jax.srt.layers.attention import qsa_sparse_backend
from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
from sgl_jax.srt.layers.attention.qsa_indexer import QSAIndexer
from sgl_jax.srt.layers.attention.qsa_sparse_backend import QSASparseAttentionBackend
from sgl_jax.srt.layers.embeddings import RotaryEmbedding
from sgl_jax.srt.layers.radix_attention import AttentionType
from sgl_jax.srt.mem_cache.memory_pool import QSATokenToKVPool

RATIO = 4

# Placement of every array the backend hands a shard_map, as the pools and the
# attention metadata lay them out.
SPECS = {
    "q": P("data", "tensor", None),
    "indexer_q": P("data", None, None),
    "indexer_k": P("data", None),
    "kv_new": P("data", "tensor", None),
    "compressed": P("data", None, None, None),
    "kv": P("data", None, "tensor", None, None),
    "ring": P(None, None, None),
    "per_token": P("data"),
}


def _mesh(dp=1):
    return Mesh(
        np.array(jax.devices())[:dp].reshape(dp, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _put(mesh, x, spec):
    return jax.device_put(jnp.asarray(x), NamedSharding(mesh, spec))


def _abstract(mesh, shape, spec, dtype=jnp.bfloat16):
    return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, spec))


def _rank_metadata(seq_lens, q_lens, page_size, pages_per_seq, rng):
    """One rank's FlashAttention metadata, and the table it encodes.

    The metadata's page list is packed: request ``i`` holds only the pages it
    needs, from ``cu_kv_lens[i] // page_size``. The returned ``i32[S,
    pages_per_seq]`` table is the fixed-stride view of the same pages. Page ids
    are the rank's own, starting after the reserved page 0. ``distribution`` is
    what FlashAttention builds: ``(n, n, n)`` for decode, ``(0, n, n)`` for an
    extend.
    """
    n_seqs = len(seq_lens)
    physical = rng.permutation(np.arange(1, 1 + n_seqs * pages_per_seq)).astype(np.int32)
    packed = np.zeros(n_seqs * pages_per_seq, np.int32)
    table = np.zeros((n_seqs, pages_per_seq), np.int32)
    cu_kv = np.zeros(n_seqs + 1, np.int32)
    for i, n_tokens in enumerate(seq_lens):
        start, n = cu_kv[i] // page_size, -(-n_tokens // page_size)
        packed[start : start + n] = table[i, :n] = physical[start : start + n]
        cu_kv[i + 1] = cu_kv[i] + n * page_size
    decode = all(n == 1 for n in q_lens)
    metadata = {
        "seq_lens": np.asarray(seq_lens, np.int32),
        "page_indices": packed,
        "cu_kv_lens": cu_kv,
        "cu_q_lens": np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32),
        "distribution": np.asarray([n_seqs if decode else 0, n_seqs, n_seqs], np.int32),
    }
    return metadata, table


def _metadata(mesh, ranks):
    """Concatenate per-rank metadata the way FlashAttention does under DP."""
    return SimpleNamespace(
        custom_mask=None,
        **{
            name: _put(mesh, np.concatenate([r[name] for r in ranks]), SPECS["per_token"])
            for name in ranks[0]
        },
    )


def _backend(mesh, *, num_heads, head_dim, page_size, block_topk):
    return QSASparseAttentionBackend(
        num_heads,
        1,
        head_dim,
        page_size,
        mesh=mesh,
        compress_ratio=RATIO,
        block_topk=block_topk,
        full_slot={0: 0},
    )


def _indexer(mesh):
    with jax.set_mesh(mesh):
        return QSAIndexer(
            hidden_size=32,
            indexer_n_heads=_Rank.IDX_HEADS,
            indexer_kv_heads=1,
            indexer_head_dim=_Rank.IDX_DIM,
            indexer_budget=2 * RATIO,
            indexer_compress_ratio=RATIO,
            rotary_dim=8,
            mesh=mesh,
            params_dtype=jnp.float32,
        )


def _rotary():
    return RotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=1024,
        base=10000,
        is_neox_style=True,
        dtype=jnp.float32,
    )


class _DeviceStateRotary:
    """A plain rotary that keeps some of its state on the device."""

    def __init__(self, mesh):
        self._host = _rotary()
        self.head_size = self._host.head_size
        with jax.set_mesh(mesh):
            self.scale = _put(mesh, jnp.ones(()), P())

    def __call__(self, positions, query, key):
        query, key = self._host(positions, query, key)
        return query * self.scale, key * self.scale


class _NnxRotary(nnx.Module):
    """A rotary built as an nnx module, its state a parameter."""

    def __init__(self):
        self.head_size = 8
        self.scale = nnx.Param(jnp.ones(()))

    def __call__(self, positions, query, key):
        query, key = _rotary()(positions, query, key)
        return query * self.scale[...], key * self.scale[...]


def _scores_launch(kernel, *, out_shape, **_):
    """What one ``_scores_kernel`` launch computes, in JAX.

    ``streamindex_topk`` splits the batch into segments and launches its
    kernel once per segment. A launch scores the requests in ``[start, end)``:
    the first ``static_q_len`` queries of each, or all of them when that is
    None. Each query sees the entries its position has closed, read through the
    fixed-stride page table. Rows the launch does not reach keep what the
    previous launch left, which starts as ``-inf``.
    """
    static_q_len = kernel.keywords["static_q_len"]
    ratio = kernel.keywords["compression_ratio"]

    def launch(
        seq_lens, page_indices, cu_q_lens, start_end, _sems, _rows, q, weights, cache, scores
    ):
        n_seqs = seq_lens.shape[0]
        n_pages, rows_per_page, packing, dim = cache.shape
        pages_per_seq = page_indices.shape[0] // n_seqs
        keys = cache.reshape(n_pages, rows_per_page * packing, dim)[
            page_indices.reshape(n_seqs, pages_per_seq)
        ].reshape(n_seqs, -1, dim)

        token = jnp.arange(q.shape[0])
        seq = jnp.clip(jnp.searchsorted(cu_q_lens[1:], token, side="right"), 0, n_seqs - 1)
        q_len = cu_q_lens[seq + 1] - cu_q_lens[seq]
        offset = token - cu_q_lens[seq]
        n_scored = q_len if static_q_len is None else jnp.minimum(q_len, static_q_len)
        launched = (seq >= start_end[0]) & (seq < start_end[1]) & (offset < n_scored)

        q_pos = seq_lens[seq] - q_len + offset
        s = jnp.einsum(
            "thd,ted->the",
            q.astype(jnp.float32),
            keys[seq].astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        )
        s = (jnp.maximum(s, 0.0) * weights[:, :, None]).sum(axis=1)
        entry = jnp.arange(s.shape[1])[None, :]
        visible = (entry < (seq_lens[seq] // ratio)[:, None]) & (
            entry < ((q_pos + 1) // ratio)[:, None]
        )
        s = jnp.where(visible, s, -jnp.inf)
        width = out_shape.shape[1] * out_shape.shape[2]
        s = jnp.pad(s, ((0, 0), (0, width - s.shape[1])), constant_values=-jnp.inf)
        return jnp.where(launched[:, None, None], s.reshape(out_shape.shape), scores)

    return launch


class _PallasWithoutLaunch:
    """``pallas`` as the selector module sees it, with the launch replaced."""

    pallas_call = staticmethod(_scores_launch)

    def __getattr__(self, name):
        return getattr(pl, name)


def _cpu_kernels():
    """The selector's dispatch runs for real; attention runs interpreted."""
    return (
        mock.patch.object(streamindex_topk, "pl", _PallasWithoutLaunch()),
        mock.patch.object(
            qsa_sparse_backend,
            "sparse_gqa_attention",
            functools.partial(sparse_gqa_attention, interpret=True),
        ),
    )


class _Rank:
    """One data-parallel rank's step: its requests, its pages, its tokens.

    ``q_lens`` defaults to one token per request, a decode step.
    """

    PAGE_SIZE, PAGES_PER_SEQ = 16, 5
    HEADS, HEAD_DIM, IDX_HEADS, IDX_DIM = 2, 128, 4, 128
    N_PAGES = 1 + 2 * PAGES_PER_SEQ

    def __init__(self, seq_lens, slots, rng, q_lens=None):
        q_lens = q_lens or [1] * len(seq_lens)
        self.metadata, self.table = _rank_metadata(
            seq_lens, q_lens, self.PAGE_SIZE, self.PAGES_PER_SEQ, rng
        )
        n = sum(q_lens)
        self.positions = np.concatenate(
            [np.arange(s - q, s, dtype=np.int32) for s, q in zip(seq_lens, q_lens)]
        )
        self.token_to_req = np.repeat(np.arange(len(q_lens), dtype=np.int32), q_lens)
        self.slots = np.asarray(slots, np.int32)
        self.q = rng.standard_normal((n, self.HEADS, self.HEAD_DIM), np.float32)
        self.indexer_q = rng.standard_normal((n, self.IDX_HEADS, self.IDX_DIM), np.float32)
        self.indexer_k = rng.standard_normal((n, self.IDX_DIM), np.float32)
        self.compressed = rng.standard_normal(
            (self.N_PAGES, self.PAGE_SIZE // RATIO, 1, self.IDX_DIM), np.float32
        )
        self.kv = rng.standard_normal(
            (self.N_PAGES, self.PAGE_SIZE, 2, 1, self.HEAD_DIM), np.float32
        )

    def slots_of_tokens(self):
        """Each token's KV cache slot, ``page * PAGE_SIZE + offset``."""
        page = self.table[self.token_to_req, self.positions // self.PAGE_SIZE]
        return page * self.PAGE_SIZE + self.positions % self.PAGE_SIZE

    def expected_selection(self, compressed, block_topk):
        """Each query's top blocks among those its position has closed."""
        keys = compressed.reshape(self.N_PAGES, -1, self.IDX_DIM)
        ids = np.full((len(self.positions), block_topk), -1, np.int32)
        for t, (req, pos) in enumerate(zip(self.token_to_req, self.positions)):
            visible = keys[self.table[req]].reshape(-1, self.IDX_DIM)[: (pos + 1) // RATIO]
            scores = np.maximum(self.indexer_q[t] @ visible.T, 0.0).sum(axis=0)
            top = np.argsort(-scores)[:block_topk]
            ids[t, : len(top)] = top
        return ids

    def expected_attention(self, block_ids, kv):
        return sparse_gqa_attention_ref(
            jnp.asarray(self.q),
            jnp.asarray(block_ids),
            jnp.asarray(self.positions),
            jnp.asarray(kv[:, :, 0]),
            jnp.asarray(kv[:, :, 1]),
            jnp.asarray(self.table),
            jnp.asarray(self.token_to_req),
            compress_ratio=RATIO,
            sm_scale=self.HEAD_DIM**-0.5,
        )


def _relative_error(got, want):
    return float(jnp.max(jnp.abs(got - want))) / float(jnp.max(jnp.abs(want)))


class TestIndexerInputs(unittest.TestCase):
    def _call(self, layer_id, **qsa_kwargs):
        backend = _backend(_mesh(), num_heads=2, head_dim=128, page_size=16, block_topk=2)
        layer = SimpleNamespace(layer_id=layer_id, head_dim=128, scaling=None)
        dense = mock.MagicMock(return_value="dense")
        with mock.patch.object(FlashAttention, "__call__", dense):
            result = backend(None, None, None, layer, SimpleNamespace(), None, **qsa_kwargs)
        return result, dense

    def test_a_qsa_layer_without_its_indexer_inputs_raises(self):
        """Falling back to dense would skip selection and leave the indexer
        cache stale, and look like it worked."""
        for given in ({}, {"indexer_q": 1}, {"indexer_q": 1, "indexer_k": 1}):
            with (
                self.subTest(given=sorted(given)),
                self.assertRaisesRegex(ValueError, "indexer_rotary_emb"),
            ):
                self._call(0, **given)

    def test_a_layer_without_an_indexer_stays_dense(self):
        result, dense = self._call(1)
        self.assertEqual(result, "dense")
        dense.assert_called_once()

    def test_a_qsa_layer_refuses_what_its_kernel_would_ignore(self):
        """The sparse kernel's only mask is the causal one, so inputs the dense
        path honours on top of it must fail rather than be dropped."""
        backend = _backend(_mesh(), num_heads=2, head_dim=128, page_size=16, block_topk=2)
        inputs = dict(indexer_q=1, indexer_k=1, indexer=1, indexer_rotary_emb=1)
        cases = {
            "non-causal": dict(causal=0),
            "encoder-only": dict(attn_type=AttentionType.ENCODER_ONLY),
            "sinks": dict(attention_sink=1),
            "custom mask": dict(custom_mask=1),
            "sliding window": dict(sliding_window_size=128),
            "logit cap": dict(logit_cap=30.0),
            "temperature": dict(xai_temperature_len=128),
        }
        for name, case in cases.items():
            backend.forward_metadata = SimpleNamespace(custom_mask=case.pop("custom_mask", None))
            causal = case.pop("causal", 1)
            attention_sink = case.pop("attention_sink", None)
            layer = SimpleNamespace(layer_id=0, head_dim=128, scaling=None, **case)
            with self.subTest(name), self.assertRaises(NotImplementedError):
                backend(
                    None,
                    None,
                    None,
                    layer,
                    SimpleNamespace(),
                    None,
                    causal,
                    attention_sink,
                    **inputs,
                )


class TestSparsePath(unittest.TestCase):
    def test_the_sparse_path_traces_through_both_kernels(self):
        """Flash-Next's decode shapes on one tp=8 shard, traced through the real
        Pallas calls.

        Tracing is where the kernels check their inputs -- the selector its
        tiling and its 4D cache, the attention its packing -- and nothing
        runs, so this holds on CPU.
        """
        mesh = _mesh()
        max_context, n_seqs = 32768, 16
        num_heads, head_dim, idx_heads, idx_dim = 3, 256, 4, 128
        seq_lens = np.random.default_rng(0).integers(1, max_context, n_seqs)
        layer = SimpleNamespace(layer_id=0, head_dim=head_dim, scaling=None)
        forward_batch = SimpleNamespace(
            positions=_put(mesh, np.asarray(seq_lens - 1, np.int32), SPECS["per_token"])
        )

        for page_size in (64, 128):
            with self.subTest(page_size=page_size):
                pages_per_seq = max_context // page_size
                n_pages = 1 + n_seqs * pages_per_seq
                backend = _backend(
                    mesh,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    page_size=page_size,
                    block_topk=512,
                )
                metadata, _ = _rank_metadata(
                    seq_lens, [1] * n_seqs, page_size, pages_per_seq, np.random.default_rng(0)
                )
                backend.forward_metadata = _metadata(mesh, [metadata])
                compressed_shape = QSATokenToKVPool._compressed_cache_shape(
                    total_num_pages=n_pages,
                    page_size=page_size,
                    compress_ratio=RATIO,
                    dtype=jnp.bfloat16,
                    indexer_key_dim=idx_dim,
                )
                with jax.set_mesh(mesh):
                    out = jax.eval_shape(
                        lambda *a, backend=backend: backend._run_sparse(*a, layer, forward_batch),
                        _abstract(mesh, (n_seqs, num_heads, head_dim), SPECS["q"]),
                        _abstract(mesh, (n_seqs, idx_heads, idx_dim), SPECS["indexer_q"]),
                        _abstract(mesh, compressed_shape, SPECS["compressed"]),
                        _abstract(mesh, (n_pages, page_size, 1, 2, head_dim), SPECS["kv"]),
                    )
                self.assertEqual(out.shape, (n_seqs, num_heads, head_dim))

    def test_each_kernel_reads_the_requests_own_pages(self):
        """A ragged decode batch against an oracle that reads each request's
        keys straight from its own pages.

        With lengths (40, 72) the packed list starts request 1 at slot 3 and
        the fixed stride at slot 5, so handing either kernel the wrong table
        reads another request's pages.
        """
        mesh = _mesh()
        rank = _Rank([40, 72], [0, 1], np.random.default_rng(0))
        block_topk = 2
        want = rank.expected_attention(
            rank.expected_selection(rank.compressed, block_topk), rank.kv
        )

        backend = _backend(
            mesh,
            num_heads=_Rank.HEADS,
            head_dim=_Rank.HEAD_DIM,
            page_size=_Rank.PAGE_SIZE,
            block_topk=block_topk,
        )
        backend.forward_metadata = _metadata(mesh, [rank.metadata])
        layer = SimpleNamespace(layer_id=0, head_dim=_Rank.HEAD_DIM, scaling=None)
        forward_batch = SimpleNamespace(positions=_put(mesh, rank.positions, SPECS["per_token"]))
        select_patch, attend_patch = _cpu_kernels()
        with select_patch, attend_patch, jax.set_mesh(mesh):
            got = backend._run_sparse(
                _put(mesh, rank.q, SPECS["q"]),
                _put(mesh, rank.indexer_q, SPECS["indexer_q"]),
                _put(mesh, rank.compressed, SPECS["compressed"]),
                _put(mesh, rank.kv, SPECS["kv"]),
                layer,
                forward_batch,
            )

        self.assertLess(_relative_error(got, want), 1e-5)

    def test_prefill_beyond_the_budget_attends_over_the_selection(self):
        """An extend batch whose requests see more blocks than they may select.

        Request 0 is a fresh 24-token prompt, request 1 extends a 20-token
        prefix by 9, and the budget is 2 blocks, so most queries have more
        visible blocks than that. Every query must attend over its own top 2,
        chosen from the compressed keys this step leaves in the cache, plus
        its open group; the KV cache the gather reads must already hold the
        step's own keys.
        """
        mesh = _mesh()
        block_topk = 2
        rank = _Rank([24, 29], [0, 1], np.random.default_rng(0), q_lens=[24, 9])
        self.assertGreater(int(np.max((rank.positions + 1) // RATIO)), block_topk)
        n_tokens = len(rank.positions)
        rng = np.random.default_rng(1)
        k = rng.standard_normal((n_tokens, 1, _Rank.HEAD_DIM), np.float32)
        v = rng.standard_normal((n_tokens, 1, _Rank.HEAD_DIM), np.float32)
        slots = rank.slots_of_tokens()

        backend = _backend(
            mesh,
            num_heads=_Rank.HEADS,
            head_dim=_Rank.HEAD_DIM,
            page_size=_Rank.PAGE_SIZE,
            block_topk=block_topk,
        )
        backend.forward_metadata = _metadata(mesh, [rank.metadata])
        kv = {"buffer": _put(mesh, rank.kv, SPECS["kv"])}

        def set_kv_buffer(layer_id, loc, k, v, is_decode):
            self.assertFalse(is_decode)
            buffer = np.array(kv["buffer"])
            loc = np.asarray(loc)
            page, offset = loc // _Rank.PAGE_SIZE, loc % _Rank.PAGE_SIZE
            buffer[page, offset, 0, 0] = np.asarray(k)[:, 0]
            buffer[page, offset, 1, 0] = np.asarray(v)[:, 0]
            kv["buffer"] = _put(mesh, buffer, SPECS["kv"])

        compressed = _put(mesh, rank.compressed, SPECS["compressed"])
        rings = _put(mesh, np.zeros((4, RATIO, _Rank.IDX_DIM), np.float32), SPECS["ring"])
        pool = SimpleNamespace(
            get_compressed_key_buffer=lambda slot: compressed,
            get_open_group_buffer=lambda slot: rings,
            set_kv_buffer=set_kv_buffer,
            get_fused_kv_buffer=lambda layer_id: kv["buffer"],
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: False),
            positions=_put(mesh, rank.positions, SPECS["per_token"]),
            req_pool_indices=_put(mesh, rank.slots, SPECS["per_token"]),
            out_cache_loc=_put(mesh, slots, SPECS["per_token"]),
        )
        layer = SimpleNamespace(layer_id=0, head_dim=_Rank.HEAD_DIM, scaling=None)
        select_patch, attend_patch = _cpu_kernels()
        with select_patch, attend_patch, jax.set_mesh(mesh):
            out, fused = backend(
                _put(mesh, rank.q, SPECS["q"]),
                _put(mesh, k, SPECS["kv_new"]),
                _put(mesh, v, SPECS["kv_new"]),
                layer,
                forward_batch,
                pool,
                indexer_q=_put(mesh, rank.indexer_q, SPECS["indexer_q"]),
                indexer_k=_put(mesh, rank.indexer_k, SPECS["indexer_k"]),
                indexer=_indexer(mesh),
                indexer_rotary_emb=_rotary(),
            )

        want_kv = rank.kv.copy()
        want_kv[slots // _Rank.PAGE_SIZE, slots % _Rank.PAGE_SIZE, 0, 0] = k[:, 0]
        want_kv[slots // _Rank.PAGE_SIZE, slots % _Rank.PAGE_SIZE, 1, 0] = v[:, 0]
        np.testing.assert_array_equal(np.asarray(fused.kv), want_kv)
        want_ids = rank.expected_selection(np.asarray(fused.compressed), block_topk)
        want = rank.expected_attention(want_ids, want_kv).reshape(n_tokens, -1)
        self.assertLess(_relative_error(out, want), 1e-5)


def _decode_step(ranks, rings, indexer, rotary):
    """Absorb then attend, for ``ranks`` laid out as one data-parallel batch."""
    mesh = _mesh(dp=len(ranks))

    def cat(name, spec):
        return _put(mesh, np.concatenate([getattr(r, name) for r in ranks]), spec)

    backend = _backend(
        mesh,
        num_heads=_Rank.HEADS,
        head_dim=_Rank.HEAD_DIM,
        page_size=_Rank.PAGE_SIZE,
        block_topk=2,
    )
    backend.forward_metadata = _metadata(mesh, [r.metadata for r in ranks])
    graphdef, params = nnx.split(indexer)
    indexer = nnx.merge(graphdef, jax.tree.map(lambda x: _put(mesh, x, P()), params))
    compressed = cat("compressed", SPECS["compressed"])
    ring = _put(mesh, rings, SPECS["ring"])
    pool = SimpleNamespace(
        get_compressed_key_buffer=lambda slot: compressed,
        get_open_group_buffer=lambda slot: ring,
    )
    forward_batch = SimpleNamespace(
        positions=cat("positions", SPECS["per_token"]),
        req_pool_indices=cat("slots", SPECS["per_token"]),
    )
    layer = SimpleNamespace(layer_id=0, head_dim=_Rank.HEAD_DIM, scaling=None)
    qsa_kwargs = {
        "indexer_q": cat("indexer_q", SPECS["indexer_q"]),
        "indexer_k": cat("indexer_k", SPECS["indexer_k"]),
        "indexer": indexer,
        "indexer_rotary_emb": rotary,
    }
    select_patch, attend_patch = _cpu_kernels()
    with select_patch, attend_patch, jax.set_mesh(mesh):
        cache, rings_out = backend._absorb_indexer_step(pool, 0, forward_batch, qsa_kwargs)
        out = backend._run_sparse(
            cat("q", SPECS["q"]),
            qsa_kwargs["indexer_q"],
            cache,
            cat("kv", SPECS["kv"]),
            layer,
            forward_batch,
        )
    return np.asarray(cache), np.asarray(rings_out), np.asarray(out)


class TestDataParallel(unittest.TestCase):
    def test_each_rank_computes_what_it_would_alone(self):
        """Under data parallelism each rank's cumulative lengths start from zero
        and its page ids are local, so read as one global batch they address the
        wrong requests and pages. A two-rank batch must give each rank exactly
        what that rank's batch gives on its own: the same attention output, the
        same compressed-cache half, and the same ring rows for its requests.

        Positions 39, 71 and 11 close a group this step and 22 does not, so the
        compression and the scatter are exercised, not only the selection.
        """
        if len(jax.devices()) < 2:
            self.skipTest("needs two devices; run this file in its own process")
        rng = np.random.default_rng(0)
        indexer, rotary = _indexer(_mesh(dp=2)), _rotary()
        ranks = [_Rank([40, 23], [3, 5], rng), _Rank([72, 12], [0, 6], rng)]
        rings = rng.standard_normal((8, RATIO, _Rank.IDX_DIM)).astype(np.float32)

        cache, rings_out, out = _decode_step(ranks, rings, indexer, rotary)

        n_pages, n_tokens = _Rank.N_PAGES, 2
        for r, rank in enumerate(ranks):
            with self.subTest(rank=r):
                cache_r, rings_r, out_r = _decode_step([rank], rings, indexer, rotary)
                np.testing.assert_array_equal(cache[r * n_pages : (r + 1) * n_pages], cache_r)
                np.testing.assert_array_equal(rings_out[rank.slots], rings_r[rank.slots])
                np.testing.assert_allclose(
                    out[r * n_tokens : (r + 1) * n_tokens], out_r, rtol=1e-6, atol=1e-6
                )
        untouched = np.setdiff1d(np.arange(len(rings)), np.concatenate([r.slots for r in ranks]))
        np.testing.assert_array_equal(rings_out[untouched], rings[untouched])

    def test_a_rotary_with_device_state_compresses_like_a_host_one(self):
        """The compression shard_map cannot close over arrays on explicit mesh
        axes, so a rotary's device state enters it as an argument. Position 39
        closes a group, so the rotation is exercised."""
        rng = np.random.default_rng(0)
        mesh = _mesh()
        indexer = _indexer(mesh)
        ranks = [_Rank([40, 23], [3, 5], rng)]
        rings = rng.standard_normal((8, RATIO, _Rank.IDX_DIM)).astype(np.float32)
        want = _decode_step(ranks, rings, indexer, _rotary())
        with jax.set_mesh(mesh):
            nnx_rotary = _NnxRotary()
        for name, rotary in (("plain", _DeviceStateRotary(mesh)), ("nnx", nnx_rotary)):
            with self.subTest(name):
                got = _decode_step(ranks, rings, indexer, rotary)
                for g, w in zip(got, want):
                    np.testing.assert_array_equal(g, w)


if __name__ == "__main__":
    unittest.main()
