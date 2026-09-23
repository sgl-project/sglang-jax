"""W7: speculative verify / draft-extend in decode form -- CPU contract and parity.

``DSA_SPEC_AS_DECODE=1`` routes TARGET_VERIFY / DRAFT_EXTEND batches through the
decode machinery: every spec token becomes a one-query pseudo-sequence
(``_spec_pseudo_decode_metadata``), the KV rows are pre-written at the slots
``_spec_token_slots`` derives, and the decode indexer + ``sparse_mla_page_level``
run as for decode. The Pallas attention kernel is TPU-only; on CPU this test pins:

* pseudo metadata invariants for T=4 (bs1) and T=8 (bs2, with / without a padded
  request): ``kv_len = own position + 1``, arange ``cu_q``, fixed-stride ``cu_kv``,
  page segments equal to the origin's, distribution;
* the indexer-cache write under pseudo metadata (``_scatter_paged``) lands each
  token on exactly the slot the prefill-form write uses (``_spec_token_slots``);
* page-selection parity: ``streamindex_page_topk_ref`` in prefill form
  (``one_token_per_seq=False``, original metadata) and in decode form
  (``one_token_per_seq=True``, pseudo metadata) pick the same page sets per token
  (the causal bounds ``pos <= abs_q`` and ``pos < kv_len`` coincide).

Attention-output parity (bf16) is a TPU gate: accept length equal to the IS1 arm
within +-0.05 and gsm8k 5/5 on the W7 arm.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    ).strip()

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.dsa.ref import streamindex_page_topk_ref
from sgl_jax.srt.layers.attention.dsa_sparse_backend import (
    _gather_cache_rows,
    _pad_topk_pages,
    _placeholder_topk_like,
    _scatter_paged,
    _spec_pseudo_decode_metadata,
    _spec_token_slots,
)
from sgl_jax.srt.utils.jax_utils import device_array

PAGE_SIZE = 4
IDX_DIM = 8
IDX_HEADS = 2
N_DRAFT = 4


def _verify_metadata(base_lens, pages_per_req, n=N_DRAFT):
    """One-rank verify-batch metadata as _make_target_verify_metadata lays it out."""
    S = len(base_lens)
    valid = np.asarray(base_lens) > 0
    ext = np.where(valid, n, 0).astype(np.int32)
    seq_lens = np.where(valid, np.asarray(base_lens) + n, 0).astype(np.int32)
    cu_q = np.concatenate([[0], np.cumsum(ext)]).astype(np.int32)
    cu_kv = (np.arange(S + 1) * pages_per_req * PAGE_SIZE).astype(np.int32)
    page_indices = (1 + np.arange(S * pages_per_req)).astype(np.int32)  # page 0 reserved
    dist = np.array([0, 0, int(valid.sum())], np.int32)
    return seq_lens, cu_q, cu_kv, page_indices, dist


class SpecAsDecodeTest(unittest.TestCase):
    def _check_case(self, base_lens, pages_per_req):
        n = N_DRAFT
        S = len(base_lens)
        T = S * n
        seq_lens, cu_q, cu_kv, pi, dist = _verify_metadata(base_lens, pages_per_req)
        args = (jnp.asarray(seq_lens), jnp.asarray(cu_q), jnp.asarray(cu_kv), jnp.asarray(pi))
        kv_len, pcu_q, pcu_kv, ppi, pdist = [
            np.asarray(x)
            for x in jax.jit(
                lambda a, b, c, d: _spec_pseudo_decode_metadata(a, b, c, d, T, PAGE_SIZE)
            )(*args)
        ]
        W = len(pi) // S
        want_kv = np.zeros((T,), np.int32)
        tok = 0
        for b in base_lens:
            if b > 0:
                for t in range(n):
                    want_kv[tok] = b + t + 1
                    tok += 1
        np.testing.assert_array_equal(kv_len, want_kv)
        np.testing.assert_array_equal(pcu_q, np.arange(T + 1))
        # Stride view: the page table is passed through unchanged and every pseudo
        # sequence points at its origin sequence's segment start. What the page-level
        # kernel gathers (page_indices[cu_kv[i] // ps + local]) must equal the pages
        # of the origin sequence, i.e. the same selection the old per-token copy gave.
        np.testing.assert_array_equal(ppi, pi)
        self.assertEqual(pcu_kv.shape, (T + 1,))
        for i in range(T):
            if kv_len[i] == 0:
                self.assertEqual(pcu_kv[i], 0)
                continue
            s = int(np.searchsorted(cu_q, i, side="right") - 1)
            self.assertEqual(pcu_kv[i], cu_kv[s])
            seg = pi[cu_kv[s] // PAGE_SIZE : cu_kv[s] // PAGE_SIZE + W]
            got = ppi[pcu_kv[i] // PAGE_SIZE : pcu_kv[i] // PAGE_SIZE + W]
            np.testing.assert_array_equal(got, seg)
        np.testing.assert_array_equal(pdist, [int((want_kv > 0).sum())] * 3)

        # indexer-cache write under pseudo metadata == prefill-form slots
        slots = np.asarray(
            jax.jit(lambda a, b, c, d: _spec_token_slots(a, b, c, d, T, PAGE_SIZE))(*args)
        )
        num_pages = int(pi.max()) + 2
        keys = jnp.asarray((1 + np.arange(T))[:, None] * np.ones((T, IDX_DIM), np.float32))
        pargs = (jnp.asarray(kv_len), jnp.asarray(ppi), jnp.asarray(pcu_q), jnp.asarray(pcu_kv), W)
        written = _scatter_paged(
            jnp.zeros((num_pages, PAGE_SIZE, IDX_DIM), jnp.float32), keys, *pargs
        )
        flat = np.asarray(written).reshape(-1, IDX_DIM)[:, 0]
        for i in range(T):
            if slots[i] >= 0:
                self.assertEqual(flat[slots[i]], i + 1, f"token {i} slot {slots[i]}")
        self.assertEqual(int((flat[PAGE_SIZE:] != 0).sum()), int((slots >= 0).sum()))

        # page-selection parity: prefill form vs decode form
        rng = np.random.default_rng(0)
        cache_np = rng.standard_normal((num_pages, PAGE_SIZE, IDX_DIM)).astype(np.float32)
        cache_np[0] = 0.0
        new_keys = jnp.asarray(rng.standard_normal((T, IDX_DIM)).astype(np.float32))
        cache = _scatter_paged(jnp.asarray(cache_np), new_keys, *pargs)
        q = jnp.asarray(rng.standard_normal((T, IDX_HEADS, IDX_DIM)).astype(np.float32))
        w = jnp.asarray(rng.random((T, IDX_HEADS)).astype(np.float32))
        k_pages = 2
        prefill_pages = np.asarray(
            streamindex_page_topk_ref(
                q,
                w,
                cache,
                jnp.asarray(seq_lens),
                jnp.asarray(pi),
                jnp.asarray(cu_q),
                jnp.asarray(cu_kv),
                jnp.asarray(dist),
                k_pages=k_pages,
                pages_per_seq=W,
                one_token_per_seq=False,
            )
        )
        decode_pages = np.asarray(
            streamindex_page_topk_ref(
                q,
                w,
                cache,
                jnp.asarray(kv_len),
                jnp.asarray(ppi),
                jnp.asarray(pcu_q),
                jnp.asarray(pcu_kv),
                jnp.asarray(pdist),
                k_pages=k_pages,
                pages_per_seq=W,
                one_token_per_seq=True,
            )
        )
        for i in range(T):
            if kv_len[i] == 0:
                np.testing.assert_array_equal(decode_pages[i], -1)
                continue
            self.assertEqual(
                sorted(int(x) for x in prefill_pages[i] if x >= 0),
                sorted(int(x) for x in decode_pages[i] if x >= 0),
                f"token {i}: prefill {prefill_pages[i]} vs decode {decode_pages[i]}",
            )

    def test_placeholder_topk_keeps_data_placement(self):
        # IndexShare reuse hands the backend page-topk only; the token-topk
        # placeholder must carry the same P("data", None) placement or the
        # _run_sparse shard_map rejects it (dpa39nap, 2026-09-22).
        for dp in (1, 2):
            mesh = jax.sharding.Mesh(
                np.array(jax.devices()[:4]).reshape(dp, 4 // dp),
                axis_names=("data", "tensor"),
                axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
            )
            pages_sh = NamedSharding(mesh, P("data", None))
            with jax.set_mesh(mesh):
                pages = device_array(np.zeros((4 * dp, 3), np.int32), sharding=pages_sh)

                @jax.jit
                def f(p):
                    ph = _placeholder_topk_like(p)
                    (ph,) = jax.shard_map(
                        lambda a: (a,),
                        in_specs=(P("data", None),),
                        out_specs=(P("data", None),),
                        check_vma=False,
                    )(ph)
                    return ph

                out = f(pages)
            self.assertEqual(out.sharding.spec, pages_sh.spec)
            self.assertEqual(out.shape, (4 * dp, 1))
            self.assertTrue(bool((np.asarray(out) == -1).all()))

    def test_pad_topk_pages_to_page_level_width(self):
        # prefill-form page-topk is index_topk/page_size (32) wide; page_level wants k_pages_max-1 (128)
        tp = jnp.arange(4 * 32, dtype=jnp.int32).reshape(4, 32)
        out = _pad_topk_pages(tp, 128)
        self.assertEqual(out.shape, (4, 128))
        np.testing.assert_array_equal(np.asarray(out)[:, :32], np.asarray(tp))
        self.assertTrue(bool((np.asarray(out)[:, 32:] == -1).all()))
        self.assertIs(_pad_topk_pages(out, 128), out)  # already wide enough: untouched

    def test_gather_cache_rows_matches_slots(self):
        # KVShare read-only steps hand the kernel the rows already at their slots.
        pk, pages, dv = 2, 6, 8
        cache = jnp.arange(pages * PAGE_SIZE * dv, dtype=jnp.float32).reshape(
            pages, PAGE_SIZE // pk, pk, dv
        )
        loc = jnp.array([3 * PAGE_SIZE + 1, 4 * PAGE_SIZE + 3, -1, 5 * PAGE_SIZE], jnp.int32)
        rows = np.asarray(_gather_cache_rows(cache, loc, PAGE_SIZE))
        flat = np.asarray(cache).reshape(-1, dv)
        for i, l in enumerate(np.asarray(loc)):
            if l < 0:
                self.assertTrue((rows[i] == 0).all())
            else:
                np.testing.assert_array_equal(rows[i], flat[l])

    def test_decode_form_verify_buckets_have_tuned_blocks(self):
        # Decode-form verify feeds bs * num_draft_tokens queries to the MLA decode
        # kernel; every bucket the server can hit (up to cc128 x 4) must resolve
        # without the LOOKUP MISS fallback on the GLM-5.2 tp16 shard shape.
        from sgl_jax.srt.kernels.mla.v2.tuned_block_sizes import TUNED_BLOCK_SIZES_MLA

        table = TUNED_BLOCK_SIZES_MLA["TPU v7"]
        for bs in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            key = ("decode", "bfloat16", "bfloat16", 4, 512, 64, 128, bs)
            self.assertIn(key, table, f"missing MLA decode tuned entry for bucket {bs}")
            self.assertEqual(len(table[key]), 3)

    def test_bs1_T4(self):
        self._check_case(base_lens=[9], pages_per_req=4)  # 9 + 4 tokens straddle pages

    def test_bs2_T8(self):
        self._check_case(base_lens=[9, 6], pages_per_req=4)

    def test_bs2_T8_with_padded_request(self):
        self._check_case(base_lens=[10, 0], pages_per_req=4)


class PageLevelWrapperTraceTest(unittest.TestCase):
    """The page-level wrapper must trace (make_jaxpr) for every page_share_group
    the backend can pass; the group is a Python int and has to stay static."""

    def _args(self, T=8, S=2, kp=8, ps=128, pages_per_seq=16, rank=16, rope=8, heads=4):
        cache = jnp.zeros((S * pages_per_seq + 1, ps, rank + rope), jnp.bfloat16)
        ql = jnp.zeros((T, heads, rank), jnp.bfloat16)
        qpe = jnp.zeros((T, heads, rope), jnp.bfloat16)
        kv = jnp.zeros((T, rank), jnp.bfloat16)
        kpe = jnp.zeros((T, rope), jnp.bfloat16)
        kv_lens = jnp.full((T,), 200, jnp.int32)
        topk = jnp.zeros((T, 4), jnp.int32)
        page_indices = jnp.arange(S * pages_per_seq * 4, dtype=jnp.int32) % (S * pages_per_seq)
        cu_q = jnp.arange(T + 1, dtype=jnp.int32)
        cu_kv = jnp.arange(T + 1, dtype=jnp.int32) * (pages_per_seq * ps)
        dist = jnp.array([T, T, T], jnp.int32)
        return (ql, qpe, kv, kpe, cache, kv_lens, topk, page_indices, cu_q, cu_kv, dist)

    def test_traces_for_each_page_share_group(self):
        from functools import partial

        from sgl_jax.srt.kernels.dsa.sparse_mla import sparse_mla_page_level

        for g in (1, 4):
            fn = partial(
                sparse_mla_page_level,
                sm_scale=1.0,
                page_size=128,
                pages_per_seq=16,
                kv_lora_rank=16,
                k_pages_max=8,
                page_share_group=g,
            )
            jaxpr = jax.make_jaxpr(fn)(*self._args())  # TracerBoolConversionError if g were traced
            self.assertTrue(str(jaxpr))


if __name__ == "__main__":
    unittest.main()
