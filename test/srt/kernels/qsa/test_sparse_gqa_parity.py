"""Parity: the sparse GQA Pallas kernel vs the reference.

Runs against the real Mosaic lowering on TPU and falls back to Pallas interpret
mode elsewhere, so it is developable on a laptop but only meaningful in CI on
v6e: the constraints the kernel is built around -- a 4-token dynamic slice of
the pool's token axis, the DMA-count ceiling, the K/V bit-field split -- are all
lowering properties that interpret mode does not model.

The packed pool cache and the reference's logical k/v are built from one source
array, so the K/V unpacking is checked rather than assumed.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.qsa.ref import sparse_gqa_attention_ref
from sgl_jax.srt.kernels.qsa.sparse_gqa_attention import sparse_gqa_attention
from sgl_jax.test.test_utils import CustomTestCase

INTERPRET = jax.default_backend() != "tpu"
RATIO = 4
PAGE_SIZE = 128
HEAD_DIM = 128


def _build(
    *,
    dtype=jnp.bfloat16,
    t_count=4,
    k_blocks=64,
    pages_per_seq=4,
    n_reqs=3,
    n_heads=8,
    seed=0,
):
    """One source array, two views: the pool's packed layout and logical k/v."""
    rng = np.random.default_rng(seed)
    n_pages = n_reqs * pages_per_seq
    kv = jnp.asarray(
        rng.standard_normal((n_pages, PAGE_SIZE, 1, 2, HEAD_DIM)).astype(np.float32),
        dtype,
    )
    k_log = kv[:, :, 0, 0, :][:, :, None, :]
    v_log = kv[:, :, 0, 1, :][:, :, None, :]
    # fp32 keeps K and V as separate head-axis entries; bf16 packs them into the
    # two halves of one 32-bit word.
    cache = kv.reshape(n_pages, PAGE_SIZE, 2, 1, HEAD_DIM) if dtype == jnp.float32 else kv

    page_table = jnp.asarray(rng.permutation(n_pages).reshape(n_reqs, pages_per_seq), jnp.int32)
    req = jnp.asarray(rng.integers(0, n_reqs, t_count), jnp.int32)
    pos = jnp.asarray(rng.integers(RATIO * k_blocks, pages_per_seq * PAGE_SIZE, t_count), jnp.int32)

    blocks = np.full((t_count, k_blocks), -1, np.int32)
    for t in range(t_count):
        # The indexer only ever sees blocks that closed before the query, so the
        # open group is never selectable.
        pick = rng.permutation(int(pos[t] + 1) // RATIO)[:k_blocks]
        blocks[t, : len(pick)] = pick
    blk = jnp.asarray(blocks, jnp.int32)

    q = jnp.asarray(rng.standard_normal((t_count, n_heads, HEAD_DIM)).astype(np.float32), dtype)
    return dict(
        q=q,
        blk=blk,
        pos=pos,
        req=req,
        page_table=page_table,
        cache=cache,
        k_log=k_log,
        v_log=v_log,
    )


def _run(case, *, block_units=16, interpret=INTERPRET):
    return sparse_gqa_attention(
        case["q"],
        case["blk"],
        case["pos"],
        case["req"],
        case["page_table"],
        case["cache"],
        sm_scale=HEAD_DIM**-0.5,
        ratio=RATIO,
        block_units=block_units,
        interpret=interpret,
    )


def _reference(case, *, k_log=None, v_log=None):
    return sparse_gqa_attention_ref(
        case["q"],
        case["blk"],
        case["pos"],
        case["k_log"] if k_log is None else k_log,
        case["v_log"] if v_log is None else v_log,
        case["page_table"],
        case["req"],
        compress_ratio=RATIO,
        sm_scale=HEAD_DIM**-0.5,
    )


def _repack(case, v_log):
    """Put a modified logical V back into the layout the kernel reads."""
    k = case["k_log"][:, :, 0, :]
    v = v_log[:, :, 0, :]
    kv = jnp.stack([k, v], axis=2)[:, :, None, :, :]  # [P, ps, 1, 2, D]
    if case["cache"].dtype == jnp.float32:
        return kv.reshape(kv.shape[0], kv.shape[1], 2, 1, kv.shape[-1])
    return kv


def _rel_err(got, want):
    want = jnp.asarray(want, jnp.float32)
    return float(jnp.max(jnp.abs(got - want))) / max(float(jnp.max(jnp.abs(want))), 1e-3)


class TestSparseGQAParity(CustomTestCase):
    def test_bf16_packed_cache(self):
        """The production layout: K and V share a 32-bit word and the kernel splits
        them, checked against the reference built from the same source array."""
        case = _build(dtype=jnp.bfloat16)
        self.assertLess(_rel_err(_run(case), _reference(case)), 2e-2)

    def test_fp32_unpacked_cache(self):
        """The threshold is tight because the fp32 path asks for the fp32
        contraction. Mosaic's default is a single bf16 pass on the MXU, which
        lands around 3e-3 and would make this check blind to everything but an
        outright wrong answer."""
        case = _build(dtype=jnp.float32)
        self.assertLess(_rel_err(_run(case), _reference(case)), 1e-5)

    def test_production_shape(self):
        """512 blocks at 128 per chunk: four chunks, 2048 selected tokens plus
        the open group -- the shape the DMA-count ceiling was measured against."""
        case = _build(dtype=jnp.bfloat16, t_count=2, k_blocks=512, pages_per_seq=24)
        self.assertLess(_rel_err(_run(case, block_units=128), _reference(case)), 2e-2)

    @unittest.skipIf(INTERPRET, "SMEM capacity is a property of the TPU core")
    def test_block_ids_beyond_smem_capacity(self):
        """1024 queries of 512 block ids are 2 MiB, twice a core's SMEM. The index
        tables reach SMEM one query at a time, so this compiles and every row
        still attends correctly. Three rows are checked against the reference,
        which materialises every selected key."""
        case = _build(dtype=jnp.bfloat16, t_count=1024, k_blocks=512, pages_per_seq=24)
        got = _run(case, block_units=128)
        rows = jnp.asarray([0, 511, 1023])
        sub = {**case, **{name: case[name][rows] for name in ("q", "blk", "pos", "req")}}
        self.assertLess(_rel_err(got[rows], _reference(sub)), 2e-2)

    def test_partial_selection_and_padding(self):
        """Short sequences leave -1 padding in block_ids; those lanes must not
        reach the softmax even though their DMA is clamped to a real address."""
        case = _build(dtype=jnp.float32, k_blocks=32, pages_per_seq=2, seed=3)
        case["pos"] = jnp.full_like(case["pos"], 70)
        blocks = np.full((case["blk"].shape[0], 32), -1, np.int32)
        blocks[:, :5] = np.arange(5, dtype=np.int32)
        case["blk"] = jnp.asarray(blocks)
        self.assertLess(_rel_err(_run(case), _reference(case)), 1e-5)

    def test_chunking_does_not_change_the_answer(self):
        """block_units only sets how many blocks one chunk gathers; 64 blocks in
        one chunk and in eight must agree."""
        case = _build(dtype=jnp.float32, k_blocks=64)
        base = _run(case, block_units=64)
        for units in (8, 16, 48):
            self.assertLess(_rel_err(_run(case, block_units=units), base), 1e-5)

    def test_open_group_tail_is_attended_and_the_future_is_not(self):
        """Perturbing a token in the query's open group changes the output;
        perturbing the token just past the query does not."""
        case = _build(dtype=jnp.float32, t_count=1, k_blocks=16, n_reqs=1, seed=11)
        pos = 70  # 70 = 17*4 + 2, so the group [68, 71) is open at 68..70
        case["pos"] = jnp.asarray([pos], jnp.int32)
        blocks = np.full((1, 16), -1, np.int32)
        blocks[0, :16] = np.arange(16, dtype=np.int32)  # tokens 0..63, no overlap
        case["blk"] = jnp.asarray(blocks)
        base = _run(case)

        def bump(token):
            page = int(case["page_table"][0, token // PAGE_SIZE])
            return case["v_log"].at[page, token % PAGE_SIZE].add(10.0)

        moved = _rel_err(_run({**case, "cache": _repack(case, bump(pos))}), base)
        self.assertGreater(moved, 1e-2)
        still = _rel_err(_run({**case, "cache": _repack(case, bump(pos + 1))}), base)
        self.assertLess(still, 1e-6)

    def test_a_closed_group_reads_nothing_past_the_query(self):
        """When the query closes its group there is no open tail, and the unit
        after it may not be written yet: NaN there leaves the output unchanged."""
        case = _build(dtype=jnp.float32, t_count=1, k_blocks=32, n_reqs=1, seed=13)
        pos = 71  # 71 = 17*4 + 3, so the group [68, 72) is closed
        case["pos"] = jnp.asarray([pos], jnp.int32)
        blocks = np.full((1, 32), -1, np.int32)
        blocks[0, :18] = np.arange(18, dtype=np.int32)  # tokens 0..71
        case["blk"] = jnp.asarray(blocks)
        base = _run(case)

        page = int(case["page_table"][0, (pos + 1) // PAGE_SIZE])
        row = (pos + 1) % PAGE_SIZE
        poisoned = case["v_log"].at[page, row : row + RATIO].set(jnp.nan)
        got = _run({**case, "cache": _repack(case, poisoned)})
        self.assertTrue(bool(jnp.all(jnp.isfinite(got))))
        self.assertLess(_rel_err(got, base), 1e-6)

    def test_padding_chunks_are_left_unfetched(self):
        """A query sees (pos + 1) // ratio blocks; ids past them are -1 at the
        tail, and the chunks holding only -1 are not fetched. Off TPU this runs
        the TPU interpreter with unwritten memory filled with NaN, so reading
        rows nothing wrote for this query turns its output into NaN. Positions
        1 and 2 see no block at all, 20 fills one chunk of four, and 71 closes
        its group with three chunks live."""
        case = _build(dtype=jnp.float32, t_count=4, k_blocks=32, n_reqs=1, seed=17)
        positions = np.array([1, 2, 20, 71], np.int32)
        blocks = np.full((4, 32), -1, np.int32)
        for t, p in enumerate(positions):
            visible = (p + 1) // RATIO
            blocks[t, :visible] = np.arange(visible, dtype=np.int32)[::-1]
        case["pos"] = jnp.asarray(positions)
        case["blk"] = jnp.asarray(blocks)
        interpret = pltpu.InterpretParams(uninitialized_memory="nan") if INTERPRET else False
        got = _run(case, block_units=8, interpret=interpret)
        self.assertTrue(bool(jnp.all(jnp.isfinite(got))))
        self.assertLess(_rel_err(got, _reference(case)), 1e-5)

    def test_swapped_kv_is_rejected(self):
        """Reading V where K lives is an O(1) error, well clear of the bf16
        rounding the parity threshold allows."""
        case = _build(dtype=jnp.bfloat16, seed=5)
        swapped = _reference(case, k_log=case["v_log"], v_log=case["k_log"])
        self.assertGreater(_rel_err(_run(case), swapped), 1e-1)

    def test_shape_gates(self):
        """Shapes the kernel refuses rather than mis-reading: a head_dim the cache
        disagrees with, and more than one KV head per device."""
        case = _build(dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "head_dim"):
            _run({**case, "cache": case["cache"][..., :64]})
        with self.assertRaisesRegex(ValueError, "one KV head"):
            _run({**case, "cache": jnp.concatenate([case["cache"]] * 2, axis=2)})


if __name__ == "__main__":
    unittest.main()
