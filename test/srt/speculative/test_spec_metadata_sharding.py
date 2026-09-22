"""Static sharding contract for the fused speculative path (CPU, explicit mesh).

The attention backends ``shard_map`` every token-/sequence-dim array with
``P(attention_data_partition_axis)`` (see ``dsa_sparse_backend._run_sparse_prefill``
and ``mla_backend``), and the host-built extend batch places them that way
(``_make_forward_batch`` / ``MLAAttentionBackend.get_forward_metadata``). The spec
path rebuilds some of these arrays *inside* JIT; under explicit sharding a
``(bs, n) -> (bs * n)`` reshape silently drops a size-1 ``data`` axis (dp=1 turns
``P("data")`` into ``P(None)``) and the first verify step then fails at trace time
with ``in_specs passed to shard_map: P('data',) does not match ... P(None,)``.

This test runs the normal extend metadata builder and the spec verify /
draft-extend / draft-decode builders on a 4-device CPU mesh (dp=1 and dp=2) and
requires every token-/sequence-dim array to carry the same NamedSharding as the
normal extend path, then feeds them through a ``shard_map`` with the backend's
``in_specs`` so a regression reproduces the production error on CPU.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    ).strip()

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.speculative.draft_extend_fused import (
    _build_chain_verify_arrays,
    _make_draft_extend_metadata,
    _make_eagle3_decode_metadata,
    _make_target_verify_metadata,
    _rotate_input_ids,
    _rotate_prefill_input_ids,
    _seed_topk_pages_from_step0,
)
from sgl_jax.srt.utils.jax_utils import device_array

PAGE_SIZE = 128
NUM_DRAFT_TOKENS = 4
META_FIELDS = ("cu_q_lens", "cu_kv_lens", "page_indices", "seq_lens", "distribution")


def _mesh(dp: int):
    devices = np.array(jax.devices()[:4]).reshape(dp, 4 // dp)
    return jax.sharding.Mesh(
        devices,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


def _reference_extend_metadata(mesh, seq_lens):
    """Normal (host-built) extend metadata: the placement every consumer expects."""
    bs = len(seq_lens)
    dp = mesh.shape["data"]
    pages = int(np.sum((seq_lens + PAGE_SIZE - 1) // PAGE_SIZE))
    backend = SimpleNamespace(mesh=mesh, page_size=PAGE_SIZE, attention_data_partition_axis="data")
    batch = SimpleNamespace(
        dp_size=dp,
        per_dp_bs_size=bs // dp,
        seq_lens=seq_lens,
        extend_seq_lens=seq_lens,
        cache_loc=np.arange(pages * PAGE_SIZE, dtype=np.int32),
        forward_mode=ForwardMode.EXTEND,
    )
    return MLAAttentionBackend.get_forward_metadata(backend, batch)


def _assert_same_placement(tc, name, got, ref):
    tc.assertIsInstance(got.sharding, NamedSharding, name)
    tc.assertEqual(got.sharding.spec, ref.sharding.spec, f"{name}: spec")
    tc.assertEqual(got.sharding.mesh, ref.sharding.mesh, f"{name}: mesh")


def _consume_like_backend(*arrays):
    """Mirror the backend contract: shard_map with P('data') on every array."""
    specs = tuple(P("data") for _ in arrays)
    return jax.shard_map(lambda *a: a, in_specs=specs, out_specs=specs, check_vma=False)(*arrays)


def _consume_like_backend_2d(*arrays):
    """Same contract for [T, k] arrays (topk pages): P('data', None)."""
    specs = tuple(P("data", None) for _ in arrays)
    return jax.shard_map(lambda *a: a, in_specs=specs, out_specs=specs, check_vma=False)(*arrays)


class SpecMetadataShardingTest(unittest.TestCase):
    def _run_case(self, dp: int):
        mesh = _mesh(dp)
        bs = dp  # dp=1 -> bs=1 is the production shape that failed
        n = NUM_DRAFT_TOKENS
        seq_np = np.full((bs,), 768, dtype=np.int32)
        alloc_np = seq_np + n
        ref_md = _reference_extend_metadata(mesh, alloc_np)
        data = NamedSharding(mesh, P("data"))
        # Token-dim placement of the host-built extend batch (_make_forward_batch).
        ref_tokens = device_array(np.zeros((bs * n,), np.int32), sharding=data)

        with jax.set_mesh(mesh):
            seq_lens = device_array(seq_np, sharding=data)
            alloc = device_array(alloc_np, sharding=data)
            verified_id = device_array(np.arange(bs, dtype=np.int32) + 10, sharding=data)
            token_list = device_array(
                np.arange(bs * (n - 1), dtype=np.int32).reshape(bs, n - 1),
                sharding=NamedSharding(mesh, P("data", None)),
            )

            @jax.jit
            def verify_arrays(vid, tl, sl):
                tokens, positions, *_ = _build_chain_verify_arrays(
                    verified_id=vid, token_list=tl, seq_lens=sl, num_verify_tokens=n, batch_size=bs
                )
                # Same consumer the target model hits on the first verify step.
                positions, tokens = _consume_like_backend(positions, tokens)
                return tokens, positions

            tokens, positions = verify_arrays(verified_id, token_list, seq_lens)
            _assert_same_placement(self, "verify.positions", positions, ref_tokens)
            _assert_same_placement(self, "verify.input_ids", tokens, ref_tokens)
            np.testing.assert_array_equal(
                np.asarray(positions).reshape(bs, n), seq_np[:, None] + np.arange(n)[None, :]
            )

            @jax.jit
            def verify_md(sl, al, md):
                out = _make_target_verify_metadata(
                    md, sl, al, speculative_num_draft_tokens=n, page_size=PAGE_SIZE, dp_size=dp
                )
                fields = _consume_like_backend(*(getattr(out, f) for f in META_FIELDS))
                return dict(zip(META_FIELDS, fields))

            for name, arr in verify_md(seq_lens, alloc, ref_md).items():
                _assert_same_placement(self, f"verify_md.{name}", arr, getattr(ref_md, name))

            @jax.jit
            def draft_extend_md(sl, al, md):
                out = _make_draft_extend_metadata(
                    md, sl, al, query_lens=sl, page_size=PAGE_SIZE, dp_size=dp
                )
                fields = _consume_like_backend(*(getattr(out, f) for f in META_FIELDS))
                return dict(zip(META_FIELDS, fields))

            for name, arr in draft_extend_md(seq_lens, alloc, ref_md).items():
                _assert_same_placement(self, f"draft_extend_md.{name}", arr, getattr(ref_md, name))

            @jax.jit
            def draft_decode_md(sl, al, md):
                old = SimpleNamespace(page_indices=md.page_indices, swa_page_indices=None)
                out = _make_eagle3_decode_metadata(old, sl, al, page_size=PAGE_SIZE, dp_size=dp)
                fields = _consume_like_backend(*(getattr(out, f) for f in META_FIELDS))
                return dict(zip(META_FIELDS, fields))

            for name, arr in draft_decode_md(seq_lens, alloc, ref_md).items():
                _assert_same_placement(self, f"draft_decode_md.{name}", arr, getattr(ref_md, name))

            # IndexShare seed (draft step 0 -> later steps): [T, k] page-topk in,
            # same placement out, since the backend shard_maps it with P("data", None).
            tpr, k = n, 3
            pages_np = np.arange(bs * tpr * k, dtype=np.int32).reshape(bs * tpr, k)
            pages = device_array(pages_np, sharding=NamedSharding(mesh, P("data", None)))
            ext = device_array(np.full((bs,), tpr, np.int32), sharding=data)
            sel = device_array(np.full((bs,), tpr - 1, np.int32), sharding=data)

            @jax.jit
            def seed_pages(p, e, s):
                (out,) = _consume_like_backend_2d(_seed_topk_pages_from_step0(p, e, s))
                return out

            seeded = seed_pages(pages, ext, sel)
            _assert_same_placement(self, "index_share.seed_topk_pages", seeded, pages)
            want = np.repeat(pages_np.reshape(bs, tpr, k)[:, tpr - 1 : tpr, :], tpr, axis=1)
            np.testing.assert_array_equal(np.asarray(seeded).reshape(bs, tpr, k), want)

            # Draft-step input-id rotation (every MTP layer after the first):
            # [bs*n] ids in, same placement out (feeds the next layer's embedding).
            ids_np = (100 + np.arange(bs * tpr)).astype(np.int32)
            ids = device_array(ids_np, sharding=data)
            new_np = (900 + np.arange(bs)).astype(np.int32)
            new_tok = device_array(new_np, sharding=data)
            ext_np = np.full((bs,), tpr, np.int32)
            if bs > 1:
                ext_np[-1] = 0  # a padding request keeps its ids
            ext_pad = device_array(ext_np, sharding=data)
            sel_np = np.clip(ext_np - 1, 0, None).astype(np.int32)
            sel_pad = device_array(sel_np, sharding=data)

            @jax.jit
            def rotate(i, e, s, t):
                (out,) = _consume_like_backend(_rotate_input_ids(i, e, s, t))
                return out

            rotated = rotate(ids, ext_pad, sel_pad, new_tok)
            _assert_same_placement(self, "draft.rotate_input_ids", rotated, ids)
            ids_2d = ids_np.reshape(bs, tpr)
            want_ids = np.concatenate([ids_2d[:, 1:], ids_2d[:, -1:]], axis=1)
            want_ids[np.arange(bs), sel_np] = new_np
            want_ids[ext_np == 0] = ids_2d[ext_np == 0]
            np.testing.assert_array_equal(np.asarray(rotated).reshape(bs, tpr), want_ids)

            # Prefill-time rotation (target extend -> first draft extend): packed
            # ragged ids per rank; each request's window shifts left by one and
            # its last slot takes the verified id. Already validated on TPU; this
            # keeps it covered under the dp=1 / dp=2 explicit meshes.
            per_dp_bs = bs // dp
            ext_full = device_array(np.full((bs,), tpr, np.int32), sharding=data)
            verified = device_array((700 + np.arange(bs)).astype(np.int32), sharding=data)

            @jax.jit
            def rotate_prefill(i, e, v):
                (out,) = _consume_like_backend(_rotate_prefill_input_ids(i, e, v, dp, per_dp_bs))
                return out

            prot = rotate_prefill(ids, ext_full, verified)
            _assert_same_placement(self, "prefill.rotate_input_ids", prot, ids)
            want_p = np.concatenate([ids_2d[:, 1:], (700 + np.arange(bs))[:, None]], axis=1)
            np.testing.assert_array_equal(np.asarray(prot).reshape(bs, tpr), want_p)

    def test_dp1_matches_extend_placement(self):
        self._run_case(dp=1)

    def test_dp2_matches_extend_placement(self):
        self._run_case(dp=2)


if __name__ == "__main__":
    unittest.main()
