"""Length / slot contract for the sparse-prefill self-write on speculative batches.

Speculative verify and draft-extend batches do not carry a per-token
``out_cache_loc`` (the scheduler hands a ``2 * draft_token_num`` allocation
extension list, -1 padded), while the sparse-prefill self-write needs exactly one
slot per query token. ``dsa_sparse_backend._spec_token_slots`` derives those
slots from the ragged metadata the way the dense MLA / FA kernels place new KV
(``seq_lens[s] - q_len[s] + i`` inside the packed page table).

Checks, for bs in {1, 2, 4} x steps 3 / draft 4 x {no padding, bucket padding}:
  * verify metadata -> slot count == bs * draft_tokens == positions count,
    slots equal ``req_to_token[r, seq_len_r + t]``, unique, in range, -1 only for
    padded (empty) sequences;
  * draft-extend metadata -> same contract against the ``seq - q_len + i`` rule;
  * ``paged_write_back`` (CPU scatter path) writes the derived slots and rejects
    a ``loc`` whose length differs from the row count.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    ).strip()

import unittest
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock import (
    _build_write_runs,
    default_run_capacity,
    paged_write_back,
    pallas_always_fits,
)
from sgl_jax.srt.layers.attention.dsa_sparse_backend import _spec_token_slots
from sgl_jax.srt.layers.attention.mla_backend import (
    MLAAttentionBackend,
    MLAAttentionMetadata,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.speculative.draft_extend_fused import (
    _make_draft_extend_metadata,
    _make_target_verify_metadata,
)
from sgl_jax.srt.utils.jax_utils import device_array

PAGE_SIZE = 128
NUM_STEPS = 3
NUM_DRAFT_TOKENS = 4


def _mesh(dp: int):
    devices = np.array(jax.devices()[:4]).reshape(dp, 4 // dp)
    return jax.sharding.Mesh(
        devices,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


def _page_table(alloc_lens, dp):
    """Contiguous per-request page allocation laid out per DP rank.

    Returns (req_to_token, page_indices, num_pages). Like the production
    ``padding_for_decode`` layout, rank r's pages occupy a fixed-size segment
    (padded with page 0) so ``page_indices.reshape(dp, -1)[r]`` is rank-local.
    """
    pages = [int(-(-int(a) // PAGE_SIZE)) for a in alloc_lens]
    next_page = 3  # leave a few pages unused so slot 0 is never a valid answer
    req_pages = []
    for n in pages:
        req_pages.append(list(range(next_page, next_page + n)))
        next_page += n
    max_len = int(max(alloc_lens)) if len(alloc_lens) else 0
    req_to_token = np.full((len(alloc_lens), max(max_len, 1)), -1, dtype=np.int32)
    for r, pg in enumerate(req_pages):
        for k, p in enumerate(pg):
            lo, hi = k * PAGE_SIZE, min((k + 1) * PAGE_SIZE, int(alloc_lens[r]))
            req_to_token[r, lo:hi] = p * PAGE_SIZE + np.arange(hi - lo)
    per_dp_bs = len(alloc_lens) // dp
    rank_segments = [
        [p for j in range(per_dp_bs) for p in req_pages[r * per_dp_bs + j]] for r in range(dp)
    ]
    per_rank = max(len(seg) for seg in rank_segments)
    page_indices = np.array(
        [p for seg in rank_segments for p in seg + [0] * (per_rank - len(seg))], dtype=np.int32
    )
    return req_to_token, page_indices, next_page


def _extend_metadata(mesh, seq_lens, page_indices):
    backend = SimpleNamespace(mesh=mesh, page_size=PAGE_SIZE, attention_data_partition_axis="data")
    dp = mesh.shape["data"]
    batch = SimpleNamespace(
        dp_size=dp,
        per_dp_bs_size=len(seq_lens) // dp,
        seq_lens=seq_lens,
        extend_seq_lens=seq_lens,
        # cache_loc only feeds page_indices (strided by page); rebuild it from ours.
        cache_loc=np.repeat(page_indices * PAGE_SIZE, PAGE_SIZE)
        + np.tile(np.arange(PAGE_SIZE), len(page_indices)),
        forward_mode=ForwardMode.EXTEND,
    )
    md = MLAAttentionBackend.get_forward_metadata(backend, batch)
    np.testing.assert_array_equal(np.asarray(md.page_indices), page_indices)
    return md


def _expected_slots_from_md(md, num_tokens):
    """Independent numpy re-derivation of the dense kernels' placement rule."""
    seq_lens = np.asarray(md.seq_lens)
    cuq = np.asarray(md.cu_q_lens)
    cukv = np.asarray(md.cu_kv_lens)
    pi = np.asarray(md.page_indices)
    out = np.full((num_tokens,), -1, dtype=np.int32)
    for s in range(len(seq_lens)):
        q0, q1 = int(cuq[s]), int(cuq[s + 1])
        if q1 <= q0 or seq_lens[s] <= 0:
            continue
        for i in range(q0, q1):
            kv_pos = int(seq_lens[s]) - (q1 - q0) + (i - q0)
            page = pi[int(cukv[s]) // PAGE_SIZE + kv_pos // PAGE_SIZE]
            out[i] = page * PAGE_SIZE + kv_pos % PAGE_SIZE
    return out


def _check_contract(tc, name, slots, expected, num_real_tokens):
    slots = np.asarray(slots)
    tc.assertEqual(slots.shape, expected.shape, f"{name}: slot count")
    np.testing.assert_array_equal(slots, expected, err_msg=f"{name}: slot values")
    valid = slots[slots >= 0]
    tc.assertEqual(len(valid), num_real_tokens, f"{name}: one slot per real token")
    tc.assertEqual(len(np.unique(valid)), len(valid), f"{name}: duplicate slots")


class SpecTokenSlotsTest(unittest.TestCase):
    def _case(self, dp, real_bs, pad_to):
        mesh = _mesh(dp)
        bs = pad_to
        n = NUM_DRAFT_TOKENS
        seq_np = np.zeros((bs,), np.int32)
        seq_np[:real_bs] = 700 + 37 * np.arange(real_bs)  # straddle page boundaries
        alloc_np = np.where(seq_np > 0, seq_np + 2 * n, 0).astype(np.int32)
        req_to_token, page_indices, num_pages = _page_table(alloc_np, dp)
        ref_md = _extend_metadata(mesh, alloc_np, page_indices)
        data = NamedSharding(mesh, P("data"))

        with jax.set_mesh(mesh):
            seq_lens = device_array(seq_np, sharding=data)
            alloc = device_array(alloc_np, sharding=data)

            @jax.jit
            def verify_md(sl, al, md):
                return _make_target_verify_metadata(
                    md, sl, al, speculative_num_draft_tokens=n, page_size=PAGE_SIZE, dp_size=dp
                )

            vmd = verify_md(seq_lens, alloc, ref_md)

            @jax.jit
            def draft_extend_md(sl, al, md):
                q = jnp.where(sl > 0, jnp.full_like(sl, n), jnp.zeros_like(sl))
                return _make_draft_extend_metadata(
                    md, sl + q, al, query_lens=q, page_size=PAGE_SIZE, dp_size=dp
                )

            dmd = draft_extend_md(seq_lens, alloc, ref_md)

        # Rank-local view (dp=1 covers the production shape; dp=2 checks the
        # per-rank offsets): slice each rank's segment like shard_map would.
        per_dp_bs = bs // dp
        for rank in range(dp):
            for name, md in (("verify", vmd), ("draft_extend", dmd)):
                cuq = np.asarray(md.cu_q_lens).reshape(dp, per_dp_bs + 1)[rank]
                cukv = np.asarray(md.cu_kv_lens).reshape(dp, per_dp_bs + 1)[rank]
                sl = np.asarray(md.seq_lens).reshape(dp, per_dp_bs)[rank]
                pi_all = np.asarray(md.page_indices)
                pi = pi_all.reshape(dp, -1)[rank]
                local = MLAAttentionMetadata(
                    cu_q_lens=cuq, cu_kv_lens=cukv, page_indices=pi, seq_lens=sl, distribution=None
                )
                t_local = per_dp_bs * n
                slots = jax.jit(
                    lambda a, b, c, d: _spec_token_slots(a, b, c, d, t_local, PAGE_SIZE)
                )(jnp.asarray(sl), jnp.asarray(cuq), jnp.asarray(cukv), jnp.asarray(pi))
                expected = _expected_slots_from_md(local, t_local)
                real_local = int(np.sum(sl > 0)) * n
                _check_contract(
                    self, f"{name}[dp{dp} bs{real_bs}/{bs} rank{rank}]", slots, expected, real_local
                )
                valid = np.asarray(slots)[np.asarray(slots) >= 0]
                self.assertTrue(np.all(valid < num_pages * PAGE_SIZE), f"{name}: slot out of range")
                if name == "verify":
                    # Truth from the request page table: verify token t of req r
                    # lives at req_to_token[r, seq_len_r + t] (positions == seq_len + t).
                    for j in range(per_dp_bs):
                        r = rank * per_dp_bs + j
                        got = np.asarray(slots)[j * n : (j + 1) * n]
                        if seq_np[r] <= 0:
                            self.assertTrue(np.all(got == -1))
                            continue
                        want = req_to_token[r, seq_np[r] : seq_np[r] + n]
                        np.testing.assert_array_equal(got, want)

    def test_dp1_bs1_no_padding(self):
        self._case(dp=1, real_bs=1, pad_to=1)

    def test_dp1_bs2_no_padding(self):
        self._case(dp=1, real_bs=2, pad_to=2)

    def test_dp1_bs4_no_padding(self):
        self._case(dp=1, real_bs=4, pad_to=4)

    def test_dp1_bs1_padded_to_2(self):
        self._case(dp=1, real_bs=1, pad_to=2)

    def test_dp1_bs2_padded_to_4(self):
        self._case(dp=1, real_bs=2, pad_to=4)

    def test_dp2_bs4(self):
        self._case(dp=2, real_bs=4, pad_to=4)

    def test_dp2_bs2_padded_to_4(self):
        self._case(dp=2, real_bs=2, pad_to=4)

    def test_paged_write_back_scatter_matches_and_rejects_length_mismatch(self):
        pk, pages, dv = 2, 6, 8
        cache = jnp.zeros((pages, PAGE_SIZE // pk, pk, dv), jnp.float32)
        rows = jnp.arange(4 * dv, dtype=jnp.float32).reshape(4, dv) + 1.0
        loc = jnp.array([3 * PAGE_SIZE + 5, 3 * PAGE_SIZE + 6, 4 * PAGE_SIZE + 127, -1], jnp.int32)
        out = paged_write_back(cache, rows, loc, page_size=PAGE_SIZE, interpret=True)
        ref = (
            cache.reshape(-1, dv)
            .at[loc]
            .set(rows, mode="drop", wrap_negative_indices=False)
            .reshape(cache.shape)
        )
        np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))
        # Pre-fix this surfaced as the production error ("Incompatible types for
        # broadcasting: ... [4,D] ... [8,D]"); the contract check must name the cause.
        with self.assertRaisesRegex(
            ValueError, "paged_write_back: loc has 8 entries but row has 4"
        ):
            paged_write_back(
                cache,
                rows,
                jnp.pad(loc, (0, 4), constant_values=-1),
                page_size=PAGE_SIZE,
                interpret=True,
            )

    def test_write_runs_never_exceed_rows(self):
        # Justifies the static dispatch: every row starts at most one run, so
        # n_raw <= number of valid rows <= T for any loc pattern.
        rng = np.random.default_rng(0)
        pk = 2
        for T in (1, 4, 8, 32, 128):
            for _ in range(20):
                loc = rng.integers(0, 6 * PAGE_SIZE, size=T).astype(np.int32)
                drop = rng.random(T) < 0.3
                loc[drop] = -1
                if rng.random() < 0.5:  # runs of consecutive slots
                    loc = np.where(drop, -1, 3 * PAGE_SIZE + np.arange(T)).astype(np.int32)
                _, n_raw = _build_write_runs(jnp.asarray(loc), kv_packing=pk, r_cap=4 * T + 130)
                self.assertLessEqual(int(n_raw), int((loc >= 0).sum()))
                self.assertLessEqual(int(n_raw), T)
        self.assertTrue(pallas_always_fits(4, 130))
        self.assertTrue(pallas_always_fits(130, 130))
        self.assertFalse(pallas_always_fits(131, 130))
        # Default capacity covers every decode-form spec batch up to 1024 rows
        # (64 requests x 4 draft tokens = 256 was the cc64 gap), while large
        # prefill row counts keep the runtime cond + scatter fallback.
        for rows in (4, 32, 256, 1024):
            self.assertTrue(pallas_always_fits(rows, default_run_capacity(rows, PAGE_SIZE)), rows)
        self.assertFalse(pallas_always_fits(8192, default_run_capacity(8192, PAGE_SIZE)))
        self.assertEqual(default_run_capacity(8192, PAGE_SIZE), 2 * (8192 // PAGE_SIZE) + 130)

    def test_spec_rows_skip_cond_large_T_keeps_cond(self):
        # Spec verify / draft-extend rows (T=4, r_cap=130 by default) must not go
        # through lax.cond (both branches hand back a whole pool: the branch_1_fun
        # pool copies seen in fused_verify / fused_draft_extend); a T that exceeds
        # r_cap must still reach the runtime cond + scatter fallback.
        import sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock as qb

        pk, pages, dv = 2, 6, 8
        cache = jnp.zeros((pages, PAGE_SIZE // pk, pk, dv), jnp.float32)
        rows = jnp.arange(4 * dv, dtype=jnp.float32).reshape(4, dv) + 1.0
        loc = jnp.array([3 * PAGE_SIZE + 5, 3 * PAGE_SIZE + 6, 4 * PAGE_SIZE + 127, -1], jnp.int32)
        ref = (
            cache.reshape(-1, dv)
            .at[loc]
            .set(rows, mode="drop", wrap_negative_indices=False)
            .reshape(cache.shape)
        )
        calls = {"cond": 0, "pallas": 0}

        def fake_pallas_call(*a, **k):
            calls["pallas"] += 1
            return lambda n_ent, table, row_w, cache_: ref  # stand-in for the TPU kernel

        real_cond = qb.jax.lax.cond

        def counting_cond(pred, tb, fb, *ops):
            calls["cond"] += 1
            return real_cond(pred, tb, fb, *ops)

        with (
            mock.patch.object(qb.pl, "pallas_call", fake_pallas_call),
            mock.patch.object(qb.jax.lax, "cond", counting_cond),
        ):
            out = paged_write_back(cache, rows, loc, page_size=PAGE_SIZE)
            self.assertEqual(calls, {"cond": 0, "pallas": 1})
            np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))
            # T=4 rows but r_cap=2: static guarantee gone -> runtime cond (scatter branch taken).
            out2 = paged_write_back(cache, rows, loc, page_size=PAGE_SIZE, r_cap=2)
            self.assertEqual(calls["cond"], 1)
            np.testing.assert_array_equal(np.asarray(out2), np.asarray(ref))

    def test_default_capacity_static_dispatch_by_rows(self):
        # Decode-form spec verify rows (T = requests x draft tokens: 256 at cc64,
        # up to 1024) must dispatch statically with the default r_cap even when
        # every row starts its own run (n_raw == T); 4096 rows keep the cond.
        import sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock as qb

        pk, dv = 2, 8
        real_cond = qb.jax.lax.cond
        for T, want_cond in ((256, 0), (512, 0), (1024, 0), (4096, 1)):
            pages = 2 * T // PAGE_SIZE + 2
            cache = jnp.zeros((pages, PAGE_SIZE // pk, pk, dv), jnp.float32)
            rows = jnp.arange(T * dv, dtype=jnp.float32).reshape(T, dv) + 1.0
            loc = jnp.arange(T, dtype=jnp.int32) * 2  # stride-2 slots: one run per row
            ref = (
                cache.reshape(-1, dv)
                .at[loc]
                .set(rows, mode="drop", wrap_negative_indices=False)
                .reshape(cache.shape)
            )
            calls = {"cond": 0, "pallas": 0}

            def fake_pallas_call(*a, _ref=ref, **k):
                calls["pallas"] += 1
                return lambda n_ent, table, row_w, cache_: _ref

            def counting_cond(pred, tb, fb, *ops):
                calls["cond"] += 1
                return real_cond(pred, tb, fb, *ops)

            with (
                mock.patch.object(qb.pl, "pallas_call", fake_pallas_call),
                mock.patch.object(qb.jax.lax, "cond", counting_cond),
            ):
                out = paged_write_back(cache, rows, loc, page_size=PAGE_SIZE)
            self.assertEqual(calls["cond"], want_cond, T)
            np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))
            _, n_raw = qb._build_write_runs(
                loc, kv_packing=pk, r_cap=default_run_capacity(T, PAGE_SIZE)
            )
            self.assertEqual(int(n_raw), T)  # worst case really is one run per row

    def test_small_rows_scatter_skips_kernel_and_matches_reference(self):
        # Spec pre-write opts into the 4D-native scatter for <= SCATTER_ROWS_MAX
        # rows: no pallas_call, no lax.cond, bit-identical to the flat reference.
        import sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock as qb

        pk, dv = 2, 8
        real_cond = qb.jax.lax.cond
        for T in (4, 256, 1024):
            pages = 2 * T // PAGE_SIZE + 2
            cache = jnp.zeros((pages, PAGE_SIZE // pk, pk, dv), jnp.float32)
            rows = jnp.arange(T * dv, dtype=jnp.float32).reshape(T, dv) + 1.0
            loc = jnp.arange(T, dtype=jnp.int32) * 2
            loc = loc.at[-1].set(-1)
            ref = (
                cache.reshape(-1, dv)
                .at[loc]
                .set(rows, mode="drop", wrap_negative_indices=False)
                .reshape(cache.shape)
            )
            calls = {"cond": 0, "pallas": 0}

            def fake_pallas_call(*a, **k):
                calls["pallas"] += 1
                raise AssertionError("pallas_call must not be used on the scatter path")

            def counting_cond(pred, tb, fb, *ops):
                calls["cond"] += 1
                return real_cond(pred, tb, fb, *ops)

            with (
                mock.patch.object(qb.pl, "pallas_call", fake_pallas_call),
                mock.patch.object(qb.jax.lax, "cond", counting_cond),
            ):
                out = paged_write_back(
                    cache, rows, loc, page_size=PAGE_SIZE, small_rows_scatter=True
                )
            self.assertEqual(calls, {"cond": 0, "pallas": 0}, T)
            np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))
        # above the bound the flag is inert (kernel/cond path as before)
        self.assertGreater(4096, qb.SCATTER_ROWS_MAX)


if __name__ == "__main__":
    unittest.main()
