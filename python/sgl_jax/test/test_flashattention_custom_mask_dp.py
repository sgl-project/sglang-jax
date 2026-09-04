"""DP=4 coverage for the rank-3 speculative custom mask (TPU, 4 chips).

What this pins down
-------------------
The verify mask is a rectangle ``[total_q_rows, 1, W]`` sharded with ``P("data")``,
so rank ``r`` receives a contiguous block of ``per_dp_bs * draft_token_num`` rows.
Inside ``shard_map`` the kernel addresses a row as
``cu_q_lens_ref[seq_idx] + bq_idx * bq_sz``, where ``cu_q_lens`` is the *rank-local*
cumsum produced by ``_per_dp_cumsum``.

If those two disagree, one rank reads another rank's mask. Nothing crashes: the
model simply accepts wrong draft tokens. ``test_verify_mask_packing`` pins the
packing arithmetic on the host, but only this test puts the real sharded array
through the kernel.

Method: run the same logical batch at dp=4 and at dp=1 with block sizes pinned, and
require the outputs to agree. Every sequence gets its own random mask, so any
cross-rank mix-up changes the result.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    ragged_paged_attention,
)
from sgl_jax.srt.layers.attention.flashattention_backend import (
    _pack_verify_mask,
    _per_dp_cumsum,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase

PAGE_SIZE = 128
HEAD_DIM = 128
DTYPE = jnp.bfloat16
GQA = 4
DRAFT_TOKENS = 8
BLOCKS = (16, 512, 16, 512)  # pinned so dp=1 and dp=4 make identical block choices


def _cdiv(a, b):
    return -(-a // b)


def _build(kv_lens, n_kv_heads, dp_size, seed=0):
    """Per-rank inputs for a target-verify step, plus the rank-3 mask.

    ``kv_lens`` is the whole logical batch; it is split evenly across ranks. Page
    indices are rank-local, because under ``shard_map`` each rank sees only its own
    slice of the KV cache.
    """
    bs = len(kv_lens)
    assert bs % dp_size == 0
    per_dp_bs = bs // dp_size
    q = DRAFT_TOKENS
    packing = 32 // (jnp.dtype(DTYPE).itemsize * 8)
    aligned = [_cdiv(kv, PAGE_SIZE) * PAGE_SIZE for kv in kv_lens]
    pages_per_seq = max(_cdiv(kv, PAGE_SIZE) for kv in kv_lens)
    pages_per_rank = per_dp_bs * pages_per_seq
    width = max(128, 1 << max(max(aligned) - 1, 1).bit_length())

    # Build the FLAT tree-kernel output, then hand it to the production packer.
    # Doing the rectangle by hand here would make the test agree with whatever
    # mental model wrote it -- the same trap as teaching an oracle about the
    # layout it is supposed to check.
    rng = np.random.default_rng(seed)
    blocks = []
    for kv in kv_lens:
        prefix = kv - q
        assert prefix > 0
        block = np.zeros((q, kv), np.int32)
        block[:, :prefix] = 1  # committed prefix: every draft token sees all of it
        corner = np.tril(np.ones((q, q), np.int32))
        corner *= (rng.random((q, q)) < 0.7) | np.eye(q, dtype=bool)
        block[:, prefix:] = corner
        blocks.append(block.reshape(-1))
    cm = np.concatenate(blocks)

    seq_lens_np = np.asarray(kv_lens, np.int32)
    aligned_np = np.asarray(aligned, np.int32)
    cm_kl = np.where(seq_lens_np > 0, seq_lens_np, q - 1).astype(np.int64)
    cm_off = np.concatenate([[0], np.cumsum(q * cm_kl)])
    mask = _pack_verify_mask(cm, seq_lens_np, aligned_np, cm_off, q, dp_size, per_dp_bs)
    assert mask.shape == (bs * q, 1, width), mask.shape

    extend = np.full((bs,), q, dtype=np.int32)
    cu_q = _per_dp_cumsum(extend, dp_size, per_dp_bs)
    cu_kv = _per_dp_cumsum(np.asarray(aligned, np.int32), dp_size, per_dp_bs)
    # Rank-local page indices, repeated per rank.
    page_indices = np.tile(np.arange(pages_per_rank, dtype=np.int32), dp_size)
    distribution = np.tile(np.array([0, per_dp_bs, per_dp_bs], np.int32), dp_size)

    total_q = bs * q
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), 4)
    n_q_heads = GQA * n_kv_heads
    # The cache must hold real values, not zeros. With k=0 every committed
    # position scores 0, contributing exp(0)=1 to the denominator and nothing to
    # the numerator, so (a) the output collapses to the eight new tokens diluted
    # by ~2000 ones, which puts the whole signal below any usable tolerance, and
    # (b) a wrong page index reads zeros just like a right one, so the rank-local
    # page addressing this test also exercises would go unchecked.
    #
    # Both arms allocate the same total page count (dp=4: 4*32, dp=1: 1*128) and
    # sequence s reads global pages [s*16, (s+1)*16) either way, so seeding from
    # the same key gives both arms identical per-sequence cache contents.
    return dict(
        q=jax.random.normal(keys[0], (total_q, n_q_heads, HEAD_DIM), DTYPE),
        k=jax.random.normal(keys[1], (total_q, n_kv_heads, HEAD_DIM), DTYPE),
        v=jax.random.normal(keys[2], (total_q, n_kv_heads, HEAD_DIM), DTYPE),
        cache=jax.random.normal(
            keys[3],
            (dp_size * pages_per_rank, PAGE_SIZE, n_kv_heads * 2 // packing, packing, HEAD_DIM),
            DTYPE,
        ),
        kv_lens=jnp.asarray(kv_lens, jnp.int32),
        page_indices=jnp.asarray(page_indices),
        cu_q_lens=jnp.asarray(cu_q),
        cu_kv_lens=jnp.asarray(cu_kv),
        distribution=jnp.asarray(distribution),
        custom_mask=jnp.asarray(mask),
    )


def _run(case, mesh):
    """Mirror the backend's shard_map wiring, tp=1 so only the data axis splits."""
    jax.sharding.set_mesh(mesh)
    in_specs = (
        P("data", "tensor"),  # queries
        P("data", "tensor"),  # keys
        P("data", "tensor"),  # values
        P("data", None, "tensor", None, None),  # kv_cache_fused
        P("data"),  # kv_lens
        P("data"),  # page_indices
        P("data"),  # cu_q_lens
        P("data"),  # cu_kv_lens
        P("data"),  # distribution
        P("data"),  # custom_mask: [total_q_rows, 1, W], dim 0 split
    )

    def body(q, k, v, cache, kv_lens, page_indices, cu_q, cu_kv, dist, mask):
        return ragged_paged_attention(
            q,
            k,
            v,
            cache,
            kv_lens,
            page_indices,
            cu_q,
            cu_kv,
            dist,
            mask,
            causal=0,
            sm_scale=HEAD_DIM**-0.5,
            m_block_sizes=BLOCKS,
        )

    # Same construction as flashattention_backend's call site.
    fn = jax.jit(
        jax.shard_map(
            body,
            in_specs=in_specs,
            out_specs=(P("data", "tensor"), P("data", None, "tensor", None, None)),
            check_vma=False,
        )
    )
    args = [
        jax.device_put(case[n], NamedSharding(mesh, s))
        for n, s in zip(
            (
                "q",
                "k",
                "v",
                "cache",
                "kv_lens",
                "page_indices",
                "cu_q_lens",
                "cu_kv_lens",
                "distribution",
                "custom_mask",
            ),
            in_specs,
            strict=True,
        )
    ]
    out, _ = jax.block_until_ready(fn(*args))
    return np.asarray(out)


class TestCustomMaskDP(CustomTestCase):
    """Requires 4 chips; registered in unit-test-tpu-v6e-4."""

    KV_LENS = [2048, 1024, 1536, 512, 2048, 768, 1280, 2048]  # ragged on purpose

    def _compare(self, n_kv_heads):
        self.assertEqual(jax.device_count(), 4, "this test needs exactly 4 chips")

        mesh4 = create_device_mesh(ici_parallelism=[4, 1], dcn_parallelism=[1, 1])
        out4 = _run(_build(self.KV_LENS, n_kv_heads, dp_size=4), mesh4)

        # The reference runs the whole batch on ONE chip. A [1, 4] mesh would not
        # do: `P("data", "tensor")` would then split the head axis four ways, so
        # the two runs would not be computing the same thing (and n_kv_heads=2
        # would not even divide).
        mesh1 = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))
        out1 = _run(_build(self.KV_LENS, n_kv_heads, dp_size=1), mesh1)

        # Sequences are laid out in the same order either way, attention is
        # per-sequence, and BLOCKS is pinned, so the two runs should agree to the
        # last bit; the tolerance is here only to absorb roughly one bf16 ULP
        # (2**-8 relative). It has to stay that tight to be worth anything: the
        # failure named below flips a sequence's eight draft columns, which moves
        # each output element by ~1e-3, so a tolerance of 1e-2 would pass on a
        # mask being read from the wrong rank entirely. Both arms build their
        # mask through _pack_verify_mask, so the row layout under test is the
        # production one, not the test's idea of it.
        np.testing.assert_allclose(
            out4.astype(np.float32),
            out1.astype(np.float32),
            rtol=4e-3,
            atol=1e-4,
            err_msg="dp=4 disagrees with dp=1: a rank is probably reading another "
            "rank's mask rows (check _pack_verify_mask against _per_dp_cumsum)",
        )

    def test_custom_mask_dp4_matches_dp1_gqa(self):
        self._compare(n_kv_heads=8)

    def test_custom_mask_dp4_matches_dp1_narrow_kv(self):
        # 2 kv heads: with tp=1 the tensor axis is trivial, but this keeps the
        # per-rank kv-head count away from the value the single-chip tests use.
        self._compare(n_kv_heads=2)


if __name__ == "__main__":
    import unittest

    unittest.main()
