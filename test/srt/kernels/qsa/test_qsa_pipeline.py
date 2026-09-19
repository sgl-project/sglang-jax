"""End-to-end QSA on CPU: indexer -> compress -> scatter -> select -> attend.

This is the first point in the stack where the selection half and the attention
half meet, so it is where their shared conventions get checked: the compressed
cache's page geometry, the group-to-token expansion, the causal bound, and the
open group's tail.

The strongest assertion here needs no reference implementation at all. Give the
indexer a budget larger than the number of visible blocks and it must select
every one of them, at which point sparse attention degenerates to dense
attention over the causal prefix. Any disagreement about what a block means, or
about which page it lives on, breaks that equality.
"""

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.qsa.paging import scatter_compressed
from sgl_jax.srt.kernels.qsa.ref import sparse_gqa_attention_ref
from sgl_jax.srt.kernels.qsa.sparse_gqa_attention import sparse_gqa_attention
from sgl_jax.srt.layers.attention.qsa_indexer import QSAIndexer, select_blocks
from sgl_jax.srt.layers.embeddings import RotaryEmbedding
from sgl_jax.test.test_utils import CustomTestCase

HIDDEN = 32
IDX_HEADS = 4
IDX_DIM = 16
ROTARY_DIM = 8
RATIO = 4
PAGE_SIZE = 16
PAGES_PER_SEQ = 2
Q_HEADS = 2
HEAD_DIM = 128
HIGHEST = jax.lax.Precision.HIGHEST


def _mesh():
    return Mesh(
        np.array(jax.devices())[:1].reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _indexer(mesh, budget):
    with jax.set_mesh(mesh):
        return QSAIndexer(
            hidden_size=HIDDEN,
            indexer_n_heads=IDX_HEADS,
            indexer_kv_heads=1,
            indexer_head_dim=IDX_DIM,
            indexer_budget=budget,
            indexer_compress_ratio=RATIO,
            rotary_dim=ROTARY_DIM,
            mesh=mesh,
            rms_norm_eps=1e-6,
            params_dtype=jnp.float32,
        )


def _rotary():
    return RotaryEmbedding(
        head_size=ROTARY_DIM,
        rotary_dim=ROTARY_DIM,
        max_position_embeddings=1024,
        base=10000,
        is_neox_style=True,
        dtype=jnp.float32,
    )


def _dense_reference(q, k_cache, v_cache, page_table, positions, token_to_req):
    """Full causal attention over each query's whole prefix, in fp32."""
    out = []
    for t in range(q.shape[0]):
        req, pos = int(token_to_req[t]), int(positions[t])
        rows = [(int(page_table[req, tok // PAGE_SIZE]), tok % PAGE_SIZE) for tok in range(pos + 1)]
        pages = jnp.asarray([p for p, _ in rows])
        offs = jnp.asarray([o for _, o in rows])
        k = jnp.repeat(k_cache[pages, offs].astype(jnp.float32), Q_HEADS, axis=1)
        v = jnp.repeat(v_cache[pages, offs].astype(jnp.float32), Q_HEADS, axis=1)
        s = jnp.einsum("hd,nhd->hn", q[t].astype(jnp.float32), k, precision=HIGHEST)
        s = s * HEAD_DIM**-0.5
        out.append(jnp.einsum("hn,nhd->hd", jax.nn.softmax(s, axis=-1), v, precision=HIGHEST))
    return jnp.stack(out)


class _Fixture:
    """One prefill step over two requests of different lengths."""

    def __init__(self, budget, seq_lens=(24, 13), seed=0):
        rng = np.random.default_rng(seed)
        self.mesh = _mesh()
        self.indexer = _indexer(self.mesh, budget)
        self.rotary = _rotary()
        self.n_seqs = len(seq_lens)

        self.seq_lens = jnp.asarray(np.asarray(seq_lens, np.int32))
        self.cu_q_lens = jnp.asarray(np.concatenate([[0], np.cumsum(seq_lens)]).astype(np.int32))
        # Packed page table: every request gets exactly PAGES_PER_SEQ pages, so
        # cu_kv_lens strides by PAGES_PER_SEQ * PAGE_SIZE. Physical pages are
        # shuffled and page 0 is the reserved sentinel.
        self.cu_kv_lens = jnp.asarray(
            (np.arange(self.n_seqs + 1) * PAGES_PER_SEQ * PAGE_SIZE).astype(np.int32)
        )
        n_pages = self.n_seqs * PAGES_PER_SEQ + 1
        table = rng.permutation(np.arange(1, n_pages))[: self.n_seqs * PAGES_PER_SEQ]
        self.page_indices = jnp.asarray(table.astype(np.int32))
        self.page_table = self.page_indices.reshape(self.n_seqs, PAGES_PER_SEQ)
        self.distribution = jnp.asarray(np.asarray([0, 0, self.n_seqs], np.int32))

        t_count = int(self.cu_q_lens[-1])
        self.positions = jnp.asarray(
            np.concatenate([np.arange(n, dtype=np.int32) for n in seq_lens])
        )
        self.token_to_req = jnp.asarray(
            np.concatenate([np.full(n, i, np.int32) for i, n in enumerate(seq_lens)])
        )
        self.hidden = jnp.asarray(rng.standard_normal((t_count, HIDDEN), np.float32))
        self.req_slots = jnp.asarray(np.arange(self.n_seqs, dtype=np.int32))
        self.rings = jnp.zeros((4, RATIO, IDX_DIM), jnp.float32)

        # Main GQA cache, fp32 so K and V sit on separate head-axis entries.
        kv = rng.standard_normal((n_pages, PAGE_SIZE, 1, 2, HEAD_DIM), np.float32)
        self.cache = jnp.asarray(kv).reshape(n_pages, PAGE_SIZE, 2, 1, HEAD_DIM)
        self.k_cache = jnp.asarray(kv[:, :, 0, 0, :][:, :, None, :])
        self.v_cache = jnp.asarray(kv[:, :, 0, 1, :][:, :, None, :])
        self.q = jnp.asarray(rng.standard_normal((t_count, Q_HEADS, HEAD_DIM), np.float32))
        self.compressed_cache = jnp.zeros((n_pages, PAGE_SIZE // RATIO, IDX_DIM), jnp.float32)

    def run_selection(self):
        with jax.set_mesh(self.mesh):
            q_idx, raw_k = self.indexer.project(self.hidden, self.positions, self.rotary)
            # The indexer's arithmetic is token-local and assumes unsharded
            # operands; in production it runs inside one shard_map. The
            # projection's outputs still carry the linear layer's axis
            # annotation, so drop it here the way the layer tests do.
            q_idx = jax.sharding.reshard(q_idx, P(None, None, None))
            raw_k = jax.sharding.reshard(raw_k, P(None, None))
            compressed, groups, seq_ids, _ = self.indexer.compress_batch(
                raw_k,
                self.positions,
                self.cu_q_lens,
                self.req_slots,
                self.rings,
                self.rotary,
            )
            cache = scatter_compressed(
                self.compressed_cache,
                compressed,
                groups,
                seq_ids,
                self.page_indices,
                self.cu_kv_lens,
                compress_ratio=RATIO,
            )
            block_ids = select_blocks(
                q_idx,
                cache,
                self.seq_lens,
                self.page_indices,
                self.cu_q_lens,
                self.cu_kv_lens,
                self.distribution,
                pages_per_seq=PAGES_PER_SEQ,
                block_topk=self.indexer.block_topk,
                compress_ratio=self.indexer.compress_ratio,
                use_kernel=False,
            )
        return block_ids, groups

    def attend(self, block_ids):
        return sparse_gqa_attention(
            self.q,
            block_ids,
            self.positions,
            self.token_to_req,
            self.page_table,
            self.cache,
            sm_scale=HEAD_DIM**-0.5,
            ratio=RATIO,
            block_units=4,
            interpret=True,
        )


class TestQSAPipeline(CustomTestCase):
    def test_full_budget_degenerates_to_dense_attention(self):
        """The assertion that needs no oracle: select everything, get dense."""
        fx = _Fixture(budget=PAGES_PER_SEQ * PAGE_SIZE)  # block_topk == 8 == max blocks
        block_ids, _ = fx.run_selection()
        got = fx.attend(block_ids)
        want = _dense_reference(
            fx.q, fx.k_cache, fx.v_cache, fx.page_table, fx.positions, fx.token_to_req
        )
        err = float(jnp.max(jnp.abs(got - want))) / float(jnp.max(jnp.abs(want)))
        self.assertLess(err, 1e-5, "sparse-with-everything-selected != dense")

    def test_selection_covers_every_visible_block(self):
        """The same claim from the other side: nothing visible was dropped, and
        nothing past the causal bound was picked."""
        fx = _Fixture(budget=PAGES_PER_SEQ * PAGE_SIZE)
        block_ids, _ = fx.run_selection()
        for t in range(block_ids.shape[0]):
            pos = int(fx.positions[t])
            visible = set(range((pos + 1) // RATIO))
            picked = {int(b) for b in block_ids[t] if int(b) >= 0}
            self.assertEqual(picked, visible, f"token {t} at pos {pos}")

    def test_narrow_budget_matches_the_kernel_reference(self):
        """With a real selection in play, fall back to checking the kernel does
        what the reference says about those blocks."""
        fx = _Fixture(budget=8)  # block_topk == 2
        block_ids, _ = fx.run_selection()
        self.assertTrue(bool(jnp.any(block_ids < 0)) or block_ids.shape[1] == 2)
        got = fx.attend(block_ids)
        want = sparse_gqa_attention_ref(
            fx.q,
            block_ids,
            fx.positions,
            fx.k_cache,
            fx.v_cache,
            fx.page_table,
            fx.token_to_req,
            compress_ratio=RATIO,
            sm_scale=HEAD_DIM**-0.5,
        )
        err = float(jnp.max(jnp.abs(got - want))) / float(jnp.max(jnp.abs(want)))
        self.assertLess(err, 1e-5)


if __name__ == "__main__":
    unittest.main()
