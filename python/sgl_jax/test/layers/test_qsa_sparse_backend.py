"""QSA sparse backend CPU tests: the backend's own assembly.

Runs in the ``unit-test-cpu`` suite. The indexer, the paging and the kernels
are pinned by their own tests; these pin what the backend adds around them --
which inputs a QSA layer requires, what each data-parallel rank reads, and the
cache layout and page table each kernel is handed.

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
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.qsa.ref import sparse_gqa_attention_ref
from sgl_jax.srt.kernels.qsa.sparse_gqa_attention import sparse_gqa_attention
from sgl_jax.srt.layers.attention import qsa_sparse_backend
from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
from sgl_jax.srt.layers.attention.qsa_indexer import QSAIndexer, select_blocks
from sgl_jax.srt.layers.attention.qsa_sparse_backend import QSASparseAttentionBackend
from sgl_jax.srt.layers.embeddings import RotaryEmbedding
from sgl_jax.srt.mem_cache.memory_pool import QSATokenToKVPool

RATIO = 4

# Placement of every array the backend hands a shard_map, as the pools and the
# attention metadata lay them out.
SPECS = {
    "q": P("data", "tensor", None),
    "indexer_q": P("data", None, None),
    "indexer_k": P("data", None),
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


def _rank_metadata(seq_lens, page_size, pages_per_seq, rng):
    """One rank's FlashAttention decode metadata, and the table it encodes.

    The metadata's page list is packed: request ``i`` holds only the pages it
    needs, from ``cu_kv_lens[i] // page_size``. The returned ``i32[S,
    pages_per_seq]`` table is the fixed-stride view of the same pages. Page ids
    are the rank's own, starting after the reserved page 0.
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
    metadata = {
        "seq_lens": np.asarray(seq_lens, np.int32),
        "page_indices": packed,
        "cu_kv_lens": cu_kv,
        "cu_q_lens": np.arange(n_seqs + 1, dtype=np.int32),
        "distribution": np.full((3,), n_seqs, np.int32),
    }
    return metadata, table


def _metadata(mesh, ranks):
    """Concatenate per-rank metadata the way FlashAttention does under DP."""
    return SimpleNamespace(
        **{
            name: _put(mesh, np.concatenate([r[name] for r in ranks]), SPECS["per_token"])
            for name in ranks[0]
        }
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


def _reference_selection(*args, **kwargs):
    return select_blocks(*args, **{**kwargs, "use_kernel": False})


def _cpu_kernels():
    """The selection kernel cannot run on CPU, so it becomes the reference,
    which walks the same table the same way; attention runs interpreted."""
    return (
        mock.patch.object(qsa_sparse_backend, "select_blocks", _reference_selection),
        mock.patch.object(
            qsa_sparse_backend,
            "sparse_gqa_attention",
            functools.partial(sparse_gqa_attention, interpret=True),
        ),
    )


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

    def test_dense_prefill_still_writes_the_indexer_cache(self):
        """Prefill is dense by default, but the decode steps after it select
        from the compressed keys it leaves behind, so it must write them."""
        mesh = _mesh()
        page_size, idx_dim = 16, _Rank.IDX_DIM
        backend = _backend(mesh, num_heads=2, head_dim=128, page_size=page_size, block_topk=2)
        metadata = {
            "seq_lens": np.asarray([8], np.int32),
            "page_indices": np.asarray([1], np.int32),
            "cu_kv_lens": np.asarray([0, page_size], np.int32),
            "cu_q_lens": np.asarray([0, 8], np.int32),
            "distribution": np.asarray([0, 1, 1], np.int32),
        }
        backend.forward_metadata = _metadata(mesh, [metadata])
        compressed = _put(
            mesh, np.zeros((2, page_size // RATIO, 1, idx_dim), np.float32), SPECS["compressed"]
        )
        pool = SimpleNamespace(
            get_compressed_key_buffer=lambda slot: compressed,
            get_open_group_buffer=lambda slot: _put(
                mesh, np.zeros((4, RATIO, idx_dim), np.float32), SPECS["ring"]
            ),
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: False),
            positions=_put(mesh, np.arange(8, dtype=np.int32), SPECS["per_token"]),
            req_pool_indices=_put(mesh, np.asarray([2], np.int32), SPECS["per_token"]),
        )
        layer = SimpleNamespace(layer_id=0, head_dim=128, scaling=None)
        rng = np.random.default_rng(0)
        dense = mock.MagicMock(return_value=("out", "kv"))
        with mock.patch.object(FlashAttention, "__call__", dense), jax.set_mesh(mesh):
            out, fused = backend(
                None,
                None,
                None,
                layer,
                forward_batch,
                pool,
                indexer_q=_put(mesh, rng.standard_normal((8, 4, idx_dim)), SPECS["indexer_q"]),
                indexer_k=_put(mesh, rng.standard_normal((8, idx_dim)), SPECS["indexer_k"]),
                indexer=_indexer(mesh),
                indexer_rotary_emb=_rotary(),
            )

        self.assertEqual((out, fused.kv), ("out", "kv"))
        written = np.abs(np.asarray(fused.compressed)[1, :, 0]).sum(axis=-1) > 0
        # Eight tokens close groups 0 and 1, the first two entries of page 1.
        np.testing.assert_array_equal(written, [True, True, False, False])


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
                    seq_lens, page_size, pages_per_seq, np.random.default_rng(0)
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
        rng = np.random.default_rng(0)
        page_size, pages_per_seq, block_topk = 16, 5, 2
        num_heads, head_dim, idx_heads, idx_dim = 2, 128, 4, 16
        seq_lens = [40, 72]
        n_seqs, n_pages = len(seq_lens), 1 + len(seq_lens) * pages_per_seq
        metadata, table = _rank_metadata(seq_lens, page_size, pages_per_seq, rng)

        # fp32, so packing is 1 and K and V sit on separate head-axis entries.
        compressed = rng.standard_normal((n_pages, page_size // RATIO, 1, idx_dim), np.float32)
        kv = rng.standard_normal((n_pages, page_size, 2, 1, head_dim), np.float32)
        q = rng.standard_normal((n_seqs, num_heads, head_dim), np.float32)
        indexer_q = rng.standard_normal((n_seqs, idx_heads, idx_dim), np.float32)
        positions = np.asarray(seq_lens, np.int32) - 1

        want_ids = np.zeros((n_seqs, block_topk), np.int32)
        for r, n_tokens in enumerate(seq_lens):
            keys = compressed[table[r]].reshape(-1, idx_dim)[: n_tokens // RATIO]
            scores = np.maximum(indexer_q[r] @ keys.T, 0.0).sum(axis=0)
            want_ids[r] = np.argsort(-scores)[:block_topk]
        want = sparse_gqa_attention_ref(
            jnp.asarray(q),
            jnp.asarray(want_ids),
            jnp.asarray(positions),
            jnp.asarray(kv[:, :, 0]),
            jnp.asarray(kv[:, :, 1]),
            jnp.asarray(table),
            jnp.arange(n_seqs, dtype=jnp.int32),
            compress_ratio=RATIO,
            sm_scale=head_dim**-0.5,
        )

        backend = _backend(
            mesh,
            num_heads=num_heads,
            head_dim=head_dim,
            page_size=page_size,
            block_topk=block_topk,
        )
        backend.forward_metadata = _metadata(mesh, [metadata])
        layer = SimpleNamespace(layer_id=0, head_dim=head_dim, scaling=None)
        forward_batch = SimpleNamespace(positions=_put(mesh, positions, SPECS["per_token"]))
        select_patch, attend_patch = _cpu_kernels()
        with select_patch, attend_patch, jax.set_mesh(mesh):
            got = backend._run_sparse(
                _put(mesh, q, SPECS["q"]),
                _put(mesh, indexer_q, SPECS["indexer_q"]),
                _put(mesh, compressed, SPECS["compressed"]),
                _put(mesh, kv, SPECS["kv"]),
                layer,
                forward_batch,
            )

        err = float(jnp.max(jnp.abs(got - want))) / float(jnp.max(jnp.abs(want)))
        self.assertLess(err, 1e-5)


class _Rank:
    """One data-parallel rank's decode step: its requests, its pages, its tokens."""

    PAGE_SIZE, PAGES_PER_SEQ = 16, 5
    HEADS, HEAD_DIM, IDX_HEADS, IDX_DIM = 2, 128, 4, 16
    N_PAGES = 1 + 2 * PAGES_PER_SEQ

    def __init__(self, seq_lens, slots, rng):
        self.metadata, _ = _rank_metadata(seq_lens, self.PAGE_SIZE, self.PAGES_PER_SEQ, rng)
        n = len(seq_lens)
        self.positions = np.asarray(seq_lens, np.int32) - 1
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


if __name__ == "__main__":
    unittest.main()
