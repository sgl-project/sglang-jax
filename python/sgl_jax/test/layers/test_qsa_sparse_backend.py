"""QSA sparse backend CPU tests: what the sparse path hands each kernel.

Runs in the ``unit-test-cpu`` suite. The indexer, the paging and the kernels
are pinned by their own tests; these pin the backend's assembly -- the cache
layout and the page table ``_run_sparse`` passes to selection and attention.

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

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh

from sgl_jax.srt.kernels.qsa.ref import sparse_gqa_attention_ref
from sgl_jax.srt.kernels.qsa.sparse_gqa_attention import sparse_gqa_attention
from sgl_jax.srt.layers.attention import qsa_sparse_backend
from sgl_jax.srt.layers.attention.qsa_indexer import select_blocks
from sgl_jax.srt.layers.attention.qsa_sparse_backend import QSASparseAttentionBackend
from sgl_jax.srt.mem_cache.memory_pool import QSATokenToKVPool

RATIO = 4


def _mesh():
    return Mesh(
        np.array(jax.devices())[:1].reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Auto, AxisType.Auto),
    )


def _decode_metadata(seq_lens, page_size, pages_per_seq, rng):
    """FlashAttention decode metadata for one rank, and the table it encodes.

    The metadata's page list is packed: request ``i`` holds only the pages it
    needs, from ``cu_kv_lens[i] // page_size``. The returned ``i32[S,
    pages_per_seq]`` table is the fixed-stride view of the same pages.
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
    metadata = SimpleNamespace(
        seq_lens=jnp.asarray(seq_lens, jnp.int32),
        page_indices=jnp.asarray(packed),
        cu_kv_lens=jnp.asarray(cu_kv),
        cu_q_lens=jnp.arange(n_seqs + 1, dtype=jnp.int32),
        distribution=jnp.full((3,), n_seqs, jnp.int32),
    )
    return metadata, table


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


def _reference_selection(*args, **kwargs):
    return select_blocks(*args, **{**kwargs, "use_kernel": False})


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
        forward_batch = SimpleNamespace(positions=jnp.asarray(seq_lens - 1, jnp.int32))

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
                backend.forward_metadata, _ = _decode_metadata(
                    seq_lens, page_size, pages_per_seq, np.random.default_rng(0)
                )
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
                        jax.ShapeDtypeStruct((n_seqs, num_heads, head_dim), jnp.bfloat16),
                        jax.ShapeDtypeStruct((n_seqs, idx_heads, idx_dim), jnp.bfloat16),
                        jax.ShapeDtypeStruct(compressed_shape, jnp.bfloat16),
                        jax.ShapeDtypeStruct((n_pages, page_size, 1, 2, head_dim), jnp.bfloat16),
                    )
                self.assertEqual(out.shape, (n_seqs, num_heads, head_dim))

    def test_each_kernel_reads_the_requests_own_pages(self):
        """A ragged decode batch against an oracle that reads each request's
        keys straight from its own pages.

        With lengths (40, 72) the packed list starts request 1 at slot 3 and
        the fixed stride at slot 5, so handing either kernel the wrong table
        reads another request's pages. The selection kernel cannot run on CPU
        and is swapped for the reference, which walks the same table the same
        way; the attention kernel runs in interpret mode.
        """
        mesh = _mesh()
        rng = np.random.default_rng(0)
        page_size, pages_per_seq, block_topk = 16, 5, 2
        num_heads, head_dim, idx_heads, idx_dim = 2, 128, 4, 16
        seq_lens = [40, 72]
        n_seqs, n_pages = len(seq_lens), 1 + len(seq_lens) * pages_per_seq
        metadata, table = _decode_metadata(seq_lens, page_size, pages_per_seq, rng)

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
        backend.forward_metadata = metadata
        layer = SimpleNamespace(layer_id=0, head_dim=head_dim, scaling=None)
        forward_batch = SimpleNamespace(positions=jnp.asarray(positions))
        with (
            mock.patch.object(qsa_sparse_backend, "select_blocks", _reference_selection),
            mock.patch.object(
                qsa_sparse_backend,
                "sparse_gqa_attention",
                functools.partial(sparse_gqa_attention, interpret=True),
            ),
            jax.set_mesh(mesh),
        ):
            got = backend._run_sparse(
                jnp.asarray(q),
                jnp.asarray(indexer_q),
                jnp.asarray(compressed),
                jnp.asarray(kv),
                layer,
                forward_batch,
            )

        err = float(jnp.max(jnp.abs(got - want))) / float(jnp.max(jnp.abs(want)))
        self.assertLess(err, 1e-5)


if __name__ == "__main__":
    unittest.main()
