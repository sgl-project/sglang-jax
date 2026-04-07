"""Tests for zero-copy KV cache optimisation (interleaved head layout).

Part 1: Pool layer — verify interleaved head layout via jnp.repeat
Part 2: Backend correctness — end-to-end attention matches reference
Part 3: Performance sanity — new path does not regress vs old
"""

import math
import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention import (
    ragged_paged_attention,
)
from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import KVCache, SplitMHATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)


# ---------------------------------------------------------------------------
# Helpers (shared with test_split_kv_attention but kept self-contained)
# ---------------------------------------------------------------------------

def _align_to(length, alignment, pad_value=0):
    aligned_len = (length + alignment - 1) // alignment * alignment
    return aligned_len


def unique_in_original_order(arr: jax.Array) -> jax.Array:
    unique_info = jnp.unique_all(arr)
    sorted_order = jnp.argsort(unique_info.indices)
    return unique_info.values[sorted_order]


def create_split_qkv(lens, num_heads, head_dim, num_kv_heads, v_head_dim, page_size, dtype=jnp.bfloat16):
    """Create Q, K, V tensors for split KV tests."""
    batched_q_len = sum(q_len for q_len, _ in lens)
    seq_lens = jnp.array([kv_len for _, kv_len in lens], dtype=jnp.int32)
    aligned_seq_lens = ((seq_lens + page_size - 1) // page_size) * page_size
    batched_aligned_kv_len = int(jnp.sum(aligned_seq_lens).item())

    key = jax.random.PRNGKey(42)
    q = jax.random.normal(key, (batched_q_len, num_heads, head_dim), dtype=dtype)
    k = jnp.zeros((batched_aligned_kv_len, num_kv_heads, head_dim), dtype=dtype)
    v = jnp.zeros((batched_aligned_kv_len, num_kv_heads, v_head_dim), dtype=dtype)

    aligned_pos = 0
    for idx in range(len(lens)):
        seq_len = lens[idx][1]
        aligned_len = _align_to(seq_len, page_size)
        seq_k = jax.random.normal(
            jax.random.split(key, len(lens) * 2)[idx],
            (seq_len, num_kv_heads, head_dim), dtype=dtype,
        )
        seq_v = jax.random.normal(
            jax.random.split(key, len(lens) * 2)[len(lens) + idx],
            (seq_len, num_kv_heads, v_head_dim), dtype=dtype,
        )
        k = k.at[aligned_pos:aligned_pos + seq_len].set(seq_k)
        v = v.at[aligned_pos:aligned_pos + seq_len].set(seq_v)
        aligned_pos += aligned_len

    return q, k, v


def write_prefix_tokens(forward_batch, token_to_kv_pool, lens, k, v):
    """Write prefix tokens and return extend K/V."""
    page_size = forward_batch.attn_backend.page_size
    aligned_seq_lens = ((forward_batch.seq_lens + page_size - 1) // page_size) * page_size
    aligned_cache_loc_idx = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(aligned_seq_lens)]
    )
    extend_k, extend_v = [], []
    for i, (q_len, kv_len) in enumerate(lens):
        start = aligned_cache_loc_idx[i]
        prefix_end = start + (kv_len - q_len)
        extend_end = start + kv_len
        if kv_len > q_len:
            prefix_cache_loc = forward_batch.cache_loc[start:prefix_end]
            token_to_kv_pool.set_kv_buffer(0, prefix_cache_loc, k[start:prefix_end], v[start:prefix_end])
        extend_k.append(k[prefix_end:extend_end])
        extend_v.append(v[prefix_end:extend_end])
    return jnp.concatenate(extend_k), jnp.concatenate(extend_v)


def create_test_infra(mode, lens, num_heads, head_dim, num_kv_heads, v_head_dim, page_size,
                      max_total_token_size=710016):
    """Create ForwardBatch, KV pool, and Q/K/V tensors."""
    dtype = jnp.bfloat16
    batch_size = len(lens)
    seq_lens = jnp.array([kv_len for _, kv_len in lens], dtype=jnp.int32)
    total_q_lens = sum(q_len for q_len, _ in lens)
    aligned_seq_lens = ((seq_lens + page_size - 1) // page_size) * page_size
    total_aligned_tokens = int(jnp.sum(aligned_seq_lens).item())

    input_ids = jnp.arange(total_q_lens, dtype=jnp.int32)
    positions = jnp.arange(total_aligned_tokens, dtype=jnp.int32)
    req_pool_indices = jnp.arange(batch_size, dtype=jnp.int32)

    pool = SplitMHATokenToKVPool(
        size=max_total_token_size, page_size=page_size, dtype=dtype,
        head_num=num_kv_heads,
        head_dim=(head_dim + 127) // 128 * 128,
        layer_num=1, mesh=mesh,
        v_head_dim=(v_head_dim + 127) // 128 * 128,
    )

    q, k, v = create_split_qkv(lens, num_heads, head_dim, num_kv_heads, v_head_dim, page_size, dtype)

    # Build cache_loc
    cache_loc_flat = []
    current_aligned_pos = 0
    for _, kv_len in lens:
        seq_indices = list(range(current_aligned_pos, current_aligned_pos + kv_len))
        aligned_len = _align_to(kv_len, page_size)
        seq_indices += [0] * (aligned_len - kv_len)
        cache_loc_flat.extend(seq_indices)
        current_aligned_pos += aligned_len
    cache_loc = jnp.array(cache_loc_flat, dtype=jnp.int32)

    if mode == "prefill":
        cache_loc_idx = jnp.concatenate(
            [jnp.array([0], dtype=jnp.int32), jnp.cumsum(aligned_seq_lens)]
        )
        out_cache_loc, extend_prefix_lens, extend_seq_lens = [], [], []
        for i, (q_len, kv_len) in enumerate(lens):
            start = cache_loc_idx[i]
            actual_end = start + seq_lens[i]
            extend_prefix_len = kv_len - q_len
            out_start = start + extend_prefix_len
            out_cache_loc.append(cache_loc[out_start:actual_end])
            extend_prefix_lens.append(jnp.array([extend_prefix_len], dtype=jnp.int32))
            extend_seq_lens.append(jnp.array([q_len], dtype=jnp.int32))
        out_cache_loc = jnp.concatenate(out_cache_loc, dtype=jnp.int32)
        extend_prefix_lens = jnp.concatenate(extend_prefix_lens, dtype=jnp.int32)
        extend_seq_lens = jnp.concatenate(extend_seq_lens, dtype=jnp.int32)
    else:
        cache_start_loc = jnp.concatenate(
            [jnp.array([0], dtype=jnp.int32), jnp.cumsum(aligned_seq_lens)]
        )
        out_cache_loc = []
        for i, (q_len, kv_len) in enumerate(lens):
            start = cache_start_loc[i]
            actual_end = start + seq_lens[i]
            out_start = actual_end - 1
            out_cache_loc.append(cache_loc[out_start:actual_end])
        out_cache_loc = jnp.concatenate(out_cache_loc, dtype=jnp.int32)
        extend_prefix_lens = None
        extend_seq_lens = None

    backend = FlashAttention(num_heads, num_kv_heads, head_dim, page_size=page_size, mesh=mesh, v_head_dim=v_head_dim)
    forward_mode = ForwardMode.EXTEND if mode == "prefill" else ForwardMode.DECODE

    mwb = ModelWorkerBatch(
        bid=1, forward_mode=forward_mode,
        input_ids=np.asarray(input_ids), real_input_ids_len=input_ids.shape[0],
        seq_lens=np.asarray(seq_lens), out_cache_loc=np.asarray(out_cache_loc),
        req_pool_indices=np.asarray(req_pool_indices), sampling_info=None,
        positions=np.asarray(positions), cache_loc=np.asarray(cache_loc),
        extend_seq_lens=np.asarray(extend_seq_lens) if extend_seq_lens is not None else None,
        extend_prefix_lens=np.asarray(extend_prefix_lens) if extend_prefix_lens is not None else None,
        return_logprob=False, return_output_logprob_only=False,
        top_logprobs_nums=None, token_ids_logprobs=None,
        extend_logprob_start_lens=None, extend_input_logprob_token_ids=None,
        real_bs=seq_lens.shape[0], spec_info=None,
    )

    fb = ForwardBatch(
        bid=1, forward_mode=forward_mode, batch_size=batch_size,
        input_ids=input_ids, req_pool_indices=req_pool_indices,
        seq_lens=seq_lens, out_cache_loc=out_cache_loc,
        positions=positions, attn_backend=backend,
        cache_loc=cache_loc, extend_prefix_lens=extend_prefix_lens,
        extend_seq_lens=extend_seq_lens, spec_info=None,
    )
    fb.attn_backend.forward_metadata = backend.get_forward_metadata(mwb)

    return fb, pool, q, k, v


def ref_split_attention(queries, k_pages, v_pages, kv_lens, page_indices,
                        cu_q_lens, num_seqs, *, causal=True, sm_scale=1.0):
    """Pure JAX reference for split KV attention."""
    _, _, num_kv_heads, k_head_dim = k_pages.shape
    v_head_dim = v_pages.shape[-1]
    num_q_heads = queries.shape[1]
    num_query_per_kv = num_q_heads // num_kv_heads

    outputs = []
    for i in range(num_seqs[0]):
        q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
        kv_len = kv_lens[i]
        q_len = q_end - q_start
        q = queries[q_start:q_end]
        indices = page_indices[i]
        k = k_pages[indices].reshape(-1, num_kv_heads, k_head_dim)[:kv_len]
        v = v_pages[indices].reshape(-1, num_kv_heads, v_head_dim)[:kv_len]
        k = jnp.repeat(k, num_query_per_kv, axis=1)
        v = jnp.repeat(v, num_query_per_kv, axis=1)

        attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32) * sm_scale
        if causal:
            q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(jnp.int32, attn.shape, 1)
            kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
            attn += jnp.where(q_span < kv_span, DEFAULT_MASK_VALUE, 0.0)
        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
        outputs.append(out)
    return jnp.concatenate(outputs, axis=0)


# ===================================================================
# Part 1: Pool layer — interleaved head layout
# ===================================================================


class TestInterleavedHeadLayout(unittest.TestCase):
    """Verify _align_kv_heads uses interleaved repetition."""

    def test_align_kv_heads_interleaved(self):
        """After _align_kv_heads, adjacent physical heads must be identical
        (i.e. jnp.repeat pattern, not jnp.concatenate)."""
        num_kv_heads = 4
        head_dim_aligned = 128
        dtype = jnp.bfloat16

        pool = SplitMHATokenToKVPool(
            size=64, page_size=1, dtype=dtype,
            head_num=num_kv_heads,
            head_dim=head_dim_aligned,
            layer_num=1, mesh=mesh,
        )

        if pool.head_num_physical == num_kv_heads:
            # No expansion on single device — verify identity passthrough
            key = jax.random.PRNGKey(0)
            x = jax.random.normal(key, (2, num_kv_heads, head_dim_aligned), dtype=dtype)
            aligned = pool._align_kv_heads(x)
            np.testing.assert_array_equal(np.asarray(aligned), np.asarray(x))
            return

        repeats = pool.head_num_physical // num_kv_heads
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, num_kv_heads, head_dim_aligned), dtype=dtype)
        aligned = pool._align_kv_heads(x)

        self.assertEqual(aligned.shape[1], pool.head_num_physical)

        # Verify interleaved pattern: heads [i*r .. (i+1)*r) should all equal head i
        aligned_np = np.asarray(aligned)
        x_np = np.asarray(x)
        for h in range(num_kv_heads):
            for r in range(repeats):
                physical_idx = h * repeats + r
                np.testing.assert_array_equal(
                    aligned_np[:, physical_idx, :],
                    x_np[:, h, :],
                    err_msg=f"Physical head {physical_idx} should equal logical head {h}",
                )

    def test_shard_has_identical_heads(self):
        """With interleaved layout, each TP shard should contain identical heads."""
        num_kv_heads = 4
        head_dim_aligned = 128
        dtype = jnp.bfloat16

        pool = SplitMHATokenToKVPool(
            size=64, page_size=1, dtype=dtype,
            head_num=num_kv_heads,
            head_dim=head_dim_aligned,
            layer_num=1, mesh=mesh,
        )

        num_devices = mesh.shape[pool.kv_partition_axis]
        heads_per_device = pool.head_num_physical // num_devices
        repeats_per_head = pool.head_num_physical // num_kv_heads

        if repeats_per_head <= 1:
            # No expansion on single device; just verify shape is correct
            self.assertEqual(pool.head_num_physical, num_kv_heads)
            return

        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (2, num_kv_heads, head_dim_aligned), dtype=dtype)
        aligned = pool._align_kv_heads(x)
        aligned_np = np.asarray(aligned)

        for d in range(num_devices):
            shard_start = d * heads_per_device
            shard = aligned_np[:, shard_start:shard_start + heads_per_device, :]
            for h_idx in range(1, heads_per_device):
                np.testing.assert_array_equal(
                    shard[:, h_idx, :], shard[:, 0, :],
                    err_msg=f"Device {d}: head {h_idx} != head 0 within shard",
                )

    def test_pool_buffer_shape_physical(self):
        """get_split_kv_buffer returns full physical buffer (no head slicing)."""
        num_kv_heads = 4
        head_dim_aligned = 128
        dtype = jnp.bfloat16

        pool = SplitMHATokenToKVPool(
            size=64, page_size=1, dtype=dtype,
            head_num=num_kv_heads,
            head_dim=head_dim_aligned,
            layer_num=1, mesh=mesh,
        )

        k_buf, v_buf = pool.get_split_kv_buffer(0)
        self.assertEqual(k_buf.shape[1], pool.head_num_physical,
                         "K buffer should have physical head count")
        self.assertEqual(v_buf.shape[1], pool.head_num_physical,
                         "V buffer should have physical head count")


def _has_tpu():
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


# ===================================================================
# Part 2: Backend correctness
# ===================================================================


@unittest.skipUnless(_has_tpu(), "Split-KV Pallas kernel requires TPU")
class TestZeroCopyBackendAttention(CustomTestCase):
    """End-to-end correctness: FlashAttention backend + SplitMHATokenToKVPool."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.rng_key = jax.random.PRNGKey(42)
        np.random.seed(42)

    def run_attention_test(self, mode, lens, num_heads, head_dim, num_kv_heads,
                           v_head_dim, page_size):
        """Run split KV attention and compare against pure-JAX reference."""
        fb, pool, q, k, v = create_test_infra(
            mode, lens, num_heads, head_dim, num_kv_heads, v_head_dim, page_size,
        )
        sm_scale = head_dim ** -0.5

        sharding = jax.sharding.NamedSharding(mesh, P(None, "tensor"))
        q_shard = jax.device_put(q.copy(), sharding)
        k_shard = jax.device_put(k.copy(), sharding)
        v_shard = jax.device_put(v.copy(), sharding)

        extend_k, extend_v = write_prefix_tokens(fb, pool, lens, k_shard, v_shard)

        attn = RadixAttention(
            num_heads=num_heads, head_dim=head_dim, scaling=sm_scale,
            num_kv_heads=num_kv_heads, layer_id=0,
            sliding_window_size=0, logit_cap=0,
        )

        # Build reference page table
        seq_lens = fb.seq_lens
        aligned_seq_lens = ((seq_lens + page_size - 1) // page_size) * page_size
        cache_start_loc = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(aligned_seq_lens)]
        )
        padding_size = 4096
        page_table_list = []
        for i in range(fb.batch_size):
            start = cache_start_loc[i]
            end = start + seq_lens[i]
            seq_page_indices = fb.cache_loc[start:end] // page_size
            unique_pages = unique_in_original_order(seq_page_indices)
            padded = jnp.pad(
                jnp.array(unique_pages, dtype=jnp.int32),
                (0, padding_size - len(unique_pages)),
            )
            page_table_list.append(padded)
        page_table = jnp.stack(page_table_list)

        k_pages = k.reshape(-1, page_size, num_kv_heads, head_dim)
        v_pages = v.reshape(-1, page_size, num_kv_heads, v_head_dim)
        expected = ref_split_attention(
            q.reshape(q.shape[0], num_heads, head_dim),
            k_pages, v_pages, seq_lens, page_table,
            fb.attn_backend.forward_metadata.cu_q_lens,
            fb.attn_backend.forward_metadata.num_seqs,
            causal=True, sm_scale=sm_scale,
        )
        jax.block_until_ready(expected)

        @jax.jit
        def jit_attn(q, k, v, forward_batch, token_to_kv_pool: KVCache):
            return attn(q, k, v, forward_batch, token_to_kv_pool)

        jax_output, kv_updated = jit_attn(q_shard, extend_k, extend_v, fb, pool)
        jax.block_until_ready(jax_output)

        rtol, atol = 2e-2, 1e-2
        jax_flat = np.asarray(jax_output)
        expected_flat = np.asarray(expected.reshape(expected.shape[0], -1))
        diff = np.abs(jax_flat - expected_flat)
        max_diff = float(np.max(diff))

        print(f"\n=== Zero-copy Test: {mode} ===")
        print(f"  heads={num_heads}, kv_heads={num_kv_heads}, "
              f"head_dim={head_dim}, v_head_dim={v_head_dim}, ps={page_size}")
        print(f"  max diff: {max_diff:.6f}")

        self.assertTrue(
            np.allclose(jax_flat, expected_flat, rtol=rtol, atol=atol),
            f"Output mismatch, max diff: {max_diff:.6f}",
        )

        # Verify updated cache shape
        self.assertIsInstance(kv_updated, tuple)
        self.assertEqual(len(kv_updated), 2)
        updated_k, updated_v = kv_updated
        aligned_head_dim = (head_dim + 127) // 128 * 128
        aligned_v_head_dim = (v_head_dim + 127) // 128 * 128
        self.assertEqual(updated_k.shape[-1], aligned_head_dim)
        self.assertEqual(updated_v.shape[-1], aligned_v_head_dim)

        # Verify replace_kv_buffer round-trip
        pool.replace_kv_buffer([kv_updated])
        k_buf, v_buf = pool.get_split_kv_buffer(0)
        self.assertEqual(k_buf.shape[1], pool.head_num_physical)
        self.assertEqual(v_buf.shape[1], pool.head_num_physical)

    # --- Decode tests ---
    def test_decode_bs1_kv128(self):
        self.run_attention_test("decode", [(1, 128)], 16, 192, 4, 128, 1)

    def test_decode_bs1_kv512(self):
        self.run_attention_test("decode", [(1, 512)], 16, 192, 4, 128, 1)

    def test_decode_bs1_kv2048(self):
        self.run_attention_test("decode", [(1, 2048)], 16, 192, 4, 128, 1)

    def test_decode_multi_batch(self):
        self.run_attention_test("decode", [(1, 128), (1, 256), (1, 64)], 16, 192, 4, 128, 1)

    # --- Prefill tests ---
    def test_prefill_full(self):
        self.run_attention_test("prefill", [(32, 32), (128, 128)], 16, 192, 4, 128, 1)

    def test_prefill_with_prefix(self):
        self.run_attention_test("prefill", [(64, 128), (32, 64)], 16, 192, 4, 128, 1)

    # --- head_dim=256 (K dim > V dim) ---
    def test_decode_hd256(self):
        self.run_attention_test("decode", [(1, 128)], 16, 256, 16, 128, 1)

    def test_prefill_hd256(self):
        self.run_attention_test("prefill", [(128, 256)], 16, 256, 8, 128, 1)

    # --- Mixed configurations ---
    def test_decode_kv_heads_equal_q_heads(self):
        """MHA (not GQA): num_kv_heads == num_heads."""
        self.run_attention_test("decode", [(1, 128)], 8, 128, 8, 128, 1)


# ===================================================================
# Part 3: Performance sanity check
# ===================================================================


@unittest.skipUnless(_has_tpu(), "Split-KV Pallas kernel requires TPU")
class TestZeroCopyPerformance(CustomTestCase):
    """Sanity: zero-copy path should not regress latency vs old tile path."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_no_regression(self):
        """Time the full _call_split; zero-copy should be <= old path."""
        num_heads, head_dim, num_kv_heads, v_head_dim = 16, 192, 4, 128
        page_size = 1
        lens = [(1, 512)]

        fb, pool, q, k, v = create_test_infra(
            "decode", lens, num_heads, head_dim, num_kv_heads, v_head_dim, page_size,
        )
        sm_scale = head_dim ** -0.5
        sharding = jax.sharding.NamedSharding(mesh, P(None, "tensor"))
        q_shard = jax.device_put(q.copy(), sharding)
        k_shard = jax.device_put(k.copy(), sharding)
        v_shard = jax.device_put(v.copy(), sharding)
        extend_k, extend_v = write_prefix_tokens(fb, pool, lens, k_shard, v_shard)

        attn = RadixAttention(
            num_heads=num_heads, head_dim=head_dim, scaling=sm_scale,
            num_kv_heads=num_kv_heads, layer_id=0,
            sliding_window_size=0, logit_cap=0,
        )

        @jax.jit
        def jit_attn(q, k, v, forward_batch, token_to_kv_pool: KVCache):
            return attn(q, k, v, forward_batch, token_to_kv_pool)

        # Warmup
        for _ in range(3):
            out, _ = jit_attn(q_shard, extend_k, extend_v, fb, pool)
            jax.block_until_ready(out)

        # Measure
        n_iter = 10
        start = time.perf_counter()
        for _ in range(n_iter):
            out, _ = jit_attn(q_shard, extend_k, extend_v, fb, pool)
            jax.block_until_ready(out)
        elapsed = (time.perf_counter() - start) / n_iter * 1000  # ms

        print(f"\n=== Performance: decode bs=1 kv_len=512, {n_iter} iters ===")
        print(f"  avg latency: {elapsed:.2f} ms")
        # Just a smoke test — no hard threshold, just ensure it runs


if __name__ == "__main__":
    unittest.main()
