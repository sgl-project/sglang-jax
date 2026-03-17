"""Tests for split KV cache attention (different K and V head dimensions).

Part 1: Kernel correctness — ragged_paged_attention with split k_cache/v_cache
Part 2: FlashAttention backend + MHATokenToKVPool with is_split=True
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention import (
    ragged_paged_attention,
    ref_ragged_paged_attention,
)
from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MHATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)


# ---------------------------------------------------------------------------
# Reference implementation supporting different K/V head dimensions
# ---------------------------------------------------------------------------
def ref_split_attention(
    queries: jax.Array,  # [padded_num_tokens, num_q_heads, head_dim]
    k_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, k_head_dim]
    v_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, v_head_dim]
    kv_lens: jax.Array,  # i32[padded_batch_size]
    page_indices: jax.Array,  # i32[padded_batch_size, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[padded_batch_size + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    causal: bool = True,
    sm_scale: float = 1.0,
    mask_value: float = DEFAULT_MASK_VALUE,
):
    """Pure JAX reference for split KV attention (K and V may have different head_dim)."""
    _, _, num_kv_heads, k_head_dim = k_pages.shape
    v_head_dim = v_pages.shape[-1]
    num_q_heads = queries.shape[1]
    assert num_q_heads % num_kv_heads == 0
    num_query_per_kv = num_q_heads // num_kv_heads

    outputs = []
    for i in range(num_seqs[0]):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]
        indices = page_indices[i]

        q = queries[q_start:q_end]  # [q_len, num_q_heads, head_dim]
        k = k_pages[indices, :, :, :].reshape(-1, num_kv_heads, k_head_dim)[:kv_len]
        v = v_pages[indices, :, :, :].reshape(-1, num_kv_heads, v_head_dim)[:kv_len]

        k = jnp.repeat(k, num_query_per_kv, axis=1)
        v = jnp.repeat(v, num_query_per_kv, axis=1)

        # Q*K: [num_q_heads, q_len, kv_len]
        attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
        attn *= sm_scale

        if causal:
            q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(jnp.int32, attn.shape, 1)
            kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
            mask = q_span < kv_span
            attn += jnp.where(mask, mask_value, 0.0)

        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
        # attn*V: output has v_head_dim
        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
        outputs.append(out)

    return jnp.concatenate(outputs, axis=0)


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------
def create_split_qkv_cache(
    lens,
    num_heads,
    head_dim,
    num_kv_heads,
    v_head_dim,
    page_size=1,
    dtype=jnp.bfloat16,
):
    """Create Q, K, V tensors where K has head_dim and V has v_head_dim."""
    batched_q_len = sum(q_len for q_len, _ in lens)

    seq_lens = jnp.array([kv_len for _, kv_len in lens], dtype=jnp.int32)
    aligned_seq_lens = ((seq_lens + page_size - 1) // page_size) * page_size
    batched_aligned_kv_len = jnp.sum(aligned_seq_lens).item()

    key = jax.random.PRNGKey(42)
    q = jax.random.normal(key, (batched_q_len, num_heads, head_dim), dtype=dtype)

    k = jnp.zeros((batched_aligned_kv_len, num_kv_heads, head_dim), dtype=dtype)
    v = jnp.zeros((batched_aligned_kv_len, num_kv_heads, v_head_dim), dtype=dtype)

    aligned_pos = 0
    for idx in range(len(lens)):
        seq_len = lens[idx][1]
        aligned_len = ((seq_len + page_size - 1) // page_size) * page_size

        seq_k = jax.random.normal(
            jax.random.split(key, len(lens) * 2)[idx],
            (seq_len, num_kv_heads, head_dim),
            dtype=dtype,
        )
        seq_v = jax.random.normal(
            jax.random.split(key, len(lens) * 2)[idx + len(lens)],
            (seq_len, num_kv_heads, v_head_dim),
            dtype=dtype,
        )

        k = k.at[aligned_pos : aligned_pos + seq_len].set(seq_k)
        v = v.at[aligned_pos : aligned_pos + seq_len].set(seq_v)
        aligned_pos += aligned_len

    return q, k, v


def unique_in_original_order(arr: jax.Array) -> jax.Array:
    unique_info = jnp.unique_all(arr)
    sorted_order = jnp.argsort(unique_info.indices)
    return unique_info.values[sorted_order]


def write_prefix_tokens_for_split_kv(forward_batch, token_to_kv_pool, lens, k, v):
    """Write prefix tokens for split KV cache and return extend K/V."""
    page_size = forward_batch.attn_backend.page_size
    aligned_seq_lens = ((forward_batch.seq_lens + page_size - 1) // page_size) * page_size
    aligned_cache_loc_idx = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(aligned_seq_lens)]
    )

    extend_k = []
    extend_v = []
    for i, (q_len, kv_len) in enumerate(lens):
        start = aligned_cache_loc_idx[i]
        prefix_end = start + (kv_len - q_len)
        extend_start = prefix_end
        extend_end = start + kv_len

        if kv_len > q_len:
            prefix_cache_loc = forward_batch.cache_loc[start:prefix_end]
            prefix_k = k[start:prefix_end]
            prefix_v = v[start:prefix_end]
            token_to_kv_pool.set_kv_buffer(0, prefix_cache_loc, prefix_k, prefix_v)

        extend_k.append(k[extend_start:extend_end])
        extend_v.append(v[extend_start:extend_end])

    return jnp.concatenate(extend_k), jnp.concatenate(extend_v)


def create_split_test_data(
    mode,
    lens,
    num_heads,
    head_dim,
    num_kv_heads,
    v_head_dim,
    page_size,
    max_total_token_size=710016,
):
    """Create test data with split KV cache (head_dim != v_head_dim)."""
    assert mode in ["prefill", "decode"]
    dtype = jnp.bfloat16
    batch_size = len(lens)

    seq_lens = jnp.array([kv_len for _, kv_len in lens], dtype=jnp.int32)
    total_q_lens = sum(q_len for q_len, _ in lens)

    aligned_seq_lens = ((seq_lens + page_size - 1) // page_size) * page_size
    total_aligned_tokens = jnp.sum(aligned_seq_lens).item()

    input_ids = jnp.arange(total_q_lens, dtype=jnp.int32)
    positions = jnp.arange(total_aligned_tokens, dtype=jnp.int32)
    req_pool_indices = jnp.arange(batch_size, dtype=jnp.int32)

    # Create split KV pool (pass 128-aligned dims, matching model_runner behavior)
    current_kv_cache = MHATokenToKVPool(
        size=max_total_token_size,
        page_size=page_size,
        dtype=dtype,
        head_num=num_kv_heads,
        head_dim=(head_dim + 127) // 128 * 128,
        layer_num=1,
        mesh=mesh,
        v_head_dim=(v_head_dim + 127) // 128 * 128,
    )
    assert current_kv_cache.is_split, "Expected split KV cache"

    q, k, v = create_split_qkv_cache(
        lens, num_heads, head_dim, num_kv_heads, v_head_dim, page_size, dtype=dtype
    )

    # Build cache_loc
    def align_to_size(lst, size, value=0):
        align_len = (len(lst) + size - 1) // size * size
        return lst + [value] * (align_len - len(lst))

    cache_loc_flat = []
    current_aligned_pos = 0
    for _, kv_len in lens:
        seq_token_indices = list(range(current_aligned_pos, current_aligned_pos + kv_len))
        aligned_seq_indices = align_to_size(seq_token_indices, page_size, 0)
        cache_loc_flat.extend(aligned_seq_indices)
        aligned_len = ((kv_len + page_size - 1) // page_size) * page_size
        current_aligned_pos += aligned_len

    cache_loc = jnp.array(cache_loc_flat, dtype=jnp.int32)

    if mode == "prefill":
        cache_loc_idx = jnp.concatenate(
            [jnp.array([0], dtype=jnp.int32), jnp.cumsum(aligned_seq_lens)]
        )
        out_cache_loc = []
        extend_prefix_lens = []
        extend_seq_lens = []
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

    # Create FlashAttention backend with v_head_dim
    attention_backend = FlashAttention(
        num_heads,
        num_kv_heads,
        head_dim,
        page_size=page_size,
        mesh=mesh,
        v_head_dim=v_head_dim,
    )

    forward_mode = ForwardMode.EXTEND if mode == "prefill" else ForwardMode.DECODE

    mwb = ModelWorkerBatch(
        bid=1,
        forward_mode=forward_mode,
        input_ids=np.asarray(input_ids),
        real_input_ids_len=input_ids.shape[0],
        seq_lens=np.asarray(seq_lens),
        out_cache_loc=np.asarray(out_cache_loc),
        req_pool_indices=np.asarray(req_pool_indices),
        sampling_info=None,
        positions=np.asarray(positions),
        cache_loc=np.asarray(cache_loc),
        extend_seq_lens=np.asarray(extend_seq_lens) if extend_seq_lens is not None else None,
        extend_prefix_lens=(
            np.asarray(extend_prefix_lens) if extend_prefix_lens is not None else None
        ),
        return_logprob=False,
        return_output_logprob_only=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        real_bs=seq_lens.shape[0],
        spec_info=None,
    )

    fb = ForwardBatch(
        bid=1,
        forward_mode=forward_mode,
        batch_size=batch_size,
        input_ids=input_ids,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        out_cache_loc=out_cache_loc,
        positions=positions,
        attn_backend=attention_backend,
        cache_loc=cache_loc,
        extend_prefix_lens=extend_prefix_lens,
        extend_seq_lens=extend_seq_lens,
        spec_info=None,
    )
    fb.attn_backend.forward_metadata = attention_backend.get_forward_metadata(mwb)

    return fb, current_kv_cache, q, k, v


# ===================================================================
# Part 1: Kernel correctness tests
# ===================================================================
class TestSplitKernelAttention(CustomTestCase):
    """Test ragged_paged_attention split kernel directly against reference."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.rng_key = jax.random.PRNGKey(42)
        np.random.seed(42)

    def test_ref_split_matches_ref_fused_when_same_dim(self):
        """Sanity: when head_dim == v_head_dim, ref_split_attention must produce
        identical results to the battle-tested ref_ragged_paged_attention."""
        num_heads = 16
        num_kv_heads = 4
        head_dim = 128  # same as v_head_dim
        page_size = 1
        lens = [(1, 128), (64, 64), (32, 128)]
        dtype = jnp.bfloat16

        q, k, v = create_split_qkv_cache(
            lens, num_heads, head_dim, num_kv_heads, head_dim, page_size, dtype=dtype
        )
        seq_lens = jnp.array([kv_len for _, kv_len in lens], dtype=jnp.int32)
        aligned_seq_lens = ((seq_lens + page_size - 1) // page_size) * page_size
        k_pages = k.reshape(-1, page_size, num_kv_heads, head_dim)
        v_pages = v.reshape(-1, page_size, num_kv_heads, head_dim)

        # Build 2D page_table [batch, pages_per_seq]
        cache_start = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(aligned_seq_lens)])
        padding_size = 4096
        cache_loc_flat = []
        current_pos = 0
        for _, kv_len in lens:
            aligned_len = ((kv_len + page_size - 1) // page_size) * page_size
            cache_loc_flat.extend(range(current_pos, current_pos + kv_len))
            cache_loc_flat.extend([0] * (aligned_len - kv_len))
            current_pos += aligned_len
        cache_loc_jnp = jnp.array(cache_loc_flat, dtype=jnp.int32)

        page_table_list = []
        for i in range(len(lens)):
            start = cache_start[i]
            end = start + seq_lens[i]
            pages = unique_in_original_order(cache_loc_jnp[start:end] // page_size)
            page_table_list.append(jnp.pad(pages.astype(jnp.int32), (0, padding_size - len(pages))))
        page_table = jnp.stack(page_table_list)

        q_lens = jnp.array([q_len for q_len, _ in lens], dtype=jnp.int32)
        cu_q_lens = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(q_lens)])
        num_seqs = jnp.array([len(lens)], dtype=jnp.int32)
        sm_scale = head_dim**-0.5

        out_split = ref_split_attention(
            q.reshape(q.shape[0], num_heads, head_dim),
            k_pages,
            v_pages,
            seq_lens,
            page_table,
            cu_q_lens,
            num_seqs,
            causal=True,
            sm_scale=sm_scale,
        )
        out_fused = ref_ragged_paged_attention(
            q.reshape(q.shape[0], num_heads, head_dim),
            k_pages,
            v_pages,
            seq_lens,
            page_table,
            cu_q_lens,
            num_seqs,
            causal=True,
            sm_scale=sm_scale,
        )

        np.testing.assert_allclose(
            np.asarray(out_split),
            np.asarray(out_fused),
            rtol=0,
            atol=0,
            err_msg="ref_split_attention should be bit-identical to ref_ragged_paged_attention "
            "when head_dim == v_head_dim",
        )

    def run_kernel_test(self, mode, lens, num_heads, head_dim, num_kv_heads, v_head_dim, page_size):
        """Run split kernel and compare against reference."""
        dtype = jnp.bfloat16
        sm_scale = head_dim**-0.5
        batch_size = len(lens)

        q, k, v = create_split_qkv_cache(
            lens, num_heads, head_dim, num_kv_heads, v_head_dim, page_size, dtype=dtype
        )

        seq_lens = jnp.array([kv_len for _, kv_len in lens], dtype=jnp.int32)
        aligned_seq_lens = ((seq_lens + page_size - 1) // page_size) * page_size

        # Build page_indices (flat, like FlashAttention does)
        cache_loc_flat = []
        current_aligned_pos = 0
        for _, kv_len in lens:
            seq_token_indices = list(range(current_aligned_pos, current_aligned_pos + kv_len))
            aligned_len = ((kv_len + page_size - 1) // page_size) * page_size
            seq_token_indices += [0] * (aligned_len - kv_len)
            cache_loc_flat.extend(seq_token_indices)
            current_aligned_pos += aligned_len

        cache_loc = np.array(cache_loc_flat, dtype=np.int32)
        indices = np.arange(0, len(cache_loc), page_size)
        page_indices_flat = (cache_loc[indices] // page_size).astype(np.int32)

        # Build cu_q_lens, cu_kv_lens
        if mode == "prefill":
            q_lens = np.array([q_len for q_len, _ in lens], dtype=np.int32)
        else:
            q_lens = np.ones(batch_size, dtype=np.int32)

        cu_q_lens = np.concatenate([np.array([0], dtype=np.int32), np.cumsum(q_lens)])
        cu_kv_lens = np.concatenate(
            [np.array([0], dtype=np.int32), np.cumsum(np.asarray(aligned_seq_lens))]
        )

        num_seqs = np.array([batch_size], dtype=np.int32)

        if mode == "decode":
            distribution = np.array([0, 0, batch_size], dtype=np.int32)
        else:
            distribution = np.array([0, batch_size, batch_size], dtype=np.int32)

        # Split K/V into extend tokens and cache
        # For prefill: extend tokens are the last q_len tokens of each sequence
        # For decode: extend tokens are the last 1 token of each sequence
        # Cache has all preceding tokens
        k_pages = k.reshape(-1, page_size, num_kv_heads, head_dim)
        v_pages = v.reshape(-1, page_size, num_kv_heads, v_head_dim)

        extend_k_list = []
        extend_v_list = []
        aligned_pos = 0
        for q_len, kv_len in lens:
            aligned_len = ((kv_len + page_size - 1) // page_size) * page_size
            extend_start = aligned_pos + (kv_len - q_len)
            extend_end = aligned_pos + kv_len
            extend_k_list.append(k[extend_start:extend_end])
            extend_v_list.append(v[extend_start:extend_end])
            aligned_pos += aligned_len

        extend_k = jnp.concatenate(extend_k_list)
        extend_v = jnp.concatenate(extend_v_list)

        # Build page_table for reference (2D: [batch_size, pages_per_seq])
        padding_size = 4096
        page_table_list = []
        cache_loc_jnp = jnp.array(cache_loc, dtype=jnp.int32)
        cache_start = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(aligned_seq_lens)])
        for i in range(batch_size):
            start = cache_start[i]
            end = start + seq_lens[i]
            seq_cache_loc = cache_loc_jnp[start:end]
            seq_page_indices = seq_cache_loc // page_size
            unique_pages = unique_in_original_order(seq_page_indices)
            padded = jnp.pad(
                jnp.array(unique_pages, dtype=jnp.int32),
                (0, padding_size - len(unique_pages)),
                constant_values=0,
            )
            page_table_list.append(padded)
        page_table = jnp.stack(page_table_list)

        # Reference output
        expected = ref_split_attention(
            q.reshape(q.shape[0], num_heads, head_dim),
            k_pages,
            v_pages,
            seq_lens,
            page_table,
            jnp.array(cu_q_lens),
            jnp.array(num_seqs),
            causal=True,
            sm_scale=sm_scale,
        )
        jax.block_until_ready(expected)

        # Kernel call via shard_map (mirroring FlashAttention._call_split)
        kv_part = "tensor"

        in_specs = (
            P(None, kv_part),
            P(None, kv_part),
            P(None, kv_part),
            P(None, None, kv_part, None),
            P(None, None, kv_part, None),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
        )
        out_specs = (
            P(None, kv_part),
            P(None, kv_part, None),
            P(None, kv_part, None),
        )

        cu_q_lens_jnp = jnp.array(cu_q_lens)
        cu_kv_lens_jnp = jnp.array(cu_kv_lens)
        page_indices_jnp = jnp.array(page_indices_flat)
        seq_lens_jnp = seq_lens
        distribution_jnp = jnp.array(distribution)

        def _split_kernel(*args):
            queries, keys_new, values_new, k_cache_arg, v_cache_arg = args[:5]
            other_args = args[5:]
            result, updated_k, updated_v = ragged_paged_attention(
                queries,
                keys_new,
                values_new,
                None,
                *other_args,
                k_cache=k_cache_arg,
                v_cache=v_cache_arg,
                causal=1,
                sm_scale=sm_scale,
            )
            return result, updated_k, updated_v

        @jax.jit
        def run_kernel(
            q,
            extend_k,
            extend_v,
            k_pages,
            v_pages,
            seq_lens_j,
            page_indices_j,
            cu_q_lens_j,
            cu_kv_lens_j,
            distribution_j,
        ):
            attn_output, updated_k, updated_v = jax.shard_map(
                _split_kernel,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            )(
                q.reshape(q.shape[0], -1, head_dim),
                extend_k.reshape(extend_k.shape[0], -1, head_dim),
                extend_v.reshape(extend_v.shape[0], -1, v_head_dim),
                k_pages,
                v_pages,
                seq_lens_j,
                page_indices_j,
                cu_q_lens_j,
                cu_kv_lens_j,
                distribution_j,
                None,  # custom_mask
            )
            return attn_output, updated_k, updated_v

        sharding = jax.sharding.NamedSharding(mesh, P(None, "tensor"))
        q_shard = jax.device_put(q, sharding)

        jax_output, _, _ = run_kernel(
            q_shard,
            extend_k,
            extend_v,
            k_pages,
            v_pages,
            seq_lens_jnp,
            page_indices_jnp,
            cu_q_lens_jnp,
            cu_kv_lens_jnp,
            distribution_jnp,
        )
        jax.block_until_ready(jax_output)

        # Compare
        rtol = 2e-2
        atol = 1e-2
        # Output has v_head_dim per head
        jax_flat = np.asarray(jax_output.reshape(q.shape[0], -1))
        expected_flat = np.asarray(expected.reshape(expected.shape[0], -1))

        diff = np.abs(jax_flat - expected_flat)
        max_diff = np.max(diff)

        print(f"\n=== Split Kernel Test: {mode} ===")
        print(
            f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}, v_head_dim={v_head_dim}, page_size={page_size}"
        )
        print(f"JAX output shape: {jax_flat.shape}, Expected shape: {expected_flat.shape}")
        print(f"Max difference: {float(max_diff):.6f}")

        are_close = np.allclose(jax_flat, expected_flat, rtol=rtol, atol=atol)
        self.assertTrue(
            are_close,
            f"Split kernel output mismatch, max diff: {float(max_diff):.6f}",
        )

    def test_split_kernel_prefill_ps1(self):
        """MHA prefill with split KV (head_dim=256, v_head_dim=128)."""
        self.run_kernel_test(
            mode="prefill",
            lens=[(1, 128), (64, 64), (128, 256)],
            num_heads=16,
            head_dim=256,
            num_kv_heads=16,
            v_head_dim=128,
            page_size=1,
        )

    def test_split_kernel_decode_ps1(self):
        """MHA decode with split KV (head_dim=256, v_head_dim=128)."""
        self.run_kernel_test(
            mode="decode",
            lens=[(1, 119), (1, 128), (1, 256)],
            num_heads=16,
            head_dim=256,
            num_kv_heads=16,
            v_head_dim=128,
            page_size=1,
        )

    def test_split_kernel_gqa_prefill_ps1(self):
        """GQA prefill with split KV (head_dim=256, v_head_dim=128)."""
        self.run_kernel_test(
            mode="prefill",
            lens=[(1, 128), (125, 125), (64, 256)],
            num_heads=16,
            head_dim=256,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=1,
        )

    def test_split_kernel_gqa_decode_ps1(self):
        """GQA decode with split KV (head_dim=256, v_head_dim=128)."""
        self.run_kernel_test(
            mode="decode",
            lens=[(1, 127), (1, 128), (1, 512)],
            num_heads=16,
            head_dim=256,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=1,
        )

    # --- page_size=16 tests ---
    def test_split_kernel_gqa_prefill_ps16(self):
        """GQA prefill with split KV, page_size=16."""
        self.run_kernel_test(
            mode="prefill",
            lens=[(1, 128), (125, 125), (64, 256)],
            num_heads=16,
            head_dim=256,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=16,
        )

    def test_split_kernel_gqa_decode_ps16(self):
        """GQA decode with split KV, page_size=16."""
        self.run_kernel_test(
            mode="decode",
            lens=[(1, 127), (1, 128), (1, 512)],
            num_heads=16,
            head_dim=256,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=16,
        )

    # --- head_dim=192 tests (MLA-style, non-128-aligned) ---
    def test_split_kernel_192_prefill_ps1(self):
        """MHA prefill with head_dim=192, v_head_dim=128 (DeepSeek-V2 MLA dims)."""
        self.run_kernel_test(
            mode="prefill",
            lens=[(1, 128), (64, 64), (128, 256)],
            num_heads=16,
            head_dim=192,
            num_kv_heads=16,
            v_head_dim=128,
            page_size=1,
        )

    def test_split_kernel_192_decode_ps1(self):
        """MHA decode with head_dim=192, v_head_dim=128."""
        self.run_kernel_test(
            mode="decode",
            lens=[(1, 119), (1, 128), (1, 256)],
            num_heads=16,
            head_dim=192,
            num_kv_heads=16,
            v_head_dim=128,
            page_size=1,
        )

    def test_split_kernel_192_gqa_prefill_ps1(self):
        """GQA prefill with head_dim=192, v_head_dim=128."""
        self.run_kernel_test(
            mode="prefill",
            lens=[(1, 128), (125, 125), (64, 256)],
            num_heads=16,
            head_dim=192,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=1,
        )

    def test_split_kernel_192_gqa_decode_ps1(self):
        """GQA decode with head_dim=192, v_head_dim=128."""
        self.run_kernel_test(
            mode="decode",
            lens=[(1, 127), (1, 128), (1, 512)],
            num_heads=16,
            head_dim=192,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=1,
        )

    def test_split_kernel_192_gqa_prefill_ps16(self):
        """GQA prefill with head_dim=192, v_head_dim=128, page_size=16."""
        self.run_kernel_test(
            mode="prefill",
            lens=[(1, 128), (125, 125), (64, 256)],
            num_heads=16,
            head_dim=192,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=16,
        )

    def test_split_kernel_192_gqa_decode_ps16(self):
        """GQA decode with head_dim=192, v_head_dim=128, page_size=16."""
        self.run_kernel_test(
            mode="decode",
            lens=[(1, 127), (1, 128), (1, 512)],
            num_heads=16,
            head_dim=192,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=16,
        )


# ===================================================================
# Part 2: FlashAttention backend + split KV cache tests
# ===================================================================
class TestSplitBackendAttention(CustomTestCase):
    """Test FlashAttention backend with split MHATokenToKVPool end-to-end."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.rng_key = jax.random.PRNGKey(42)
        np.random.seed(42)

    def run_backend_test(
        self,
        mode,
        lens,
        num_heads,
        head_dim,
        num_kv_heads,
        v_head_dim,
        page_size,
        check_cache=False,
    ):
        """Run split KV attention through FlashAttention backend and compare against reference."""
        forward_batch, token_to_kv_pool, q, k, v = create_split_test_data(
            mode,
            lens,
            num_heads,
            head_dim,
            num_kv_heads,
            v_head_dim,
            page_size,
        )

        sm_scale = head_dim**-0.5

        # Shard inputs
        sharding = jax.sharding.NamedSharding(mesh, P(None, "tensor"))
        q_shard = jax.device_put(q.copy(), sharding)
        k_shard = jax.device_put(k.copy(), sharding)
        v_shard = jax.device_put(v.copy(), sharding)

        # Write prefix tokens and get extend k/v
        extend_k, extend_v = write_prefix_tokens_for_split_kv(
            forward_batch, token_to_kv_pool, lens, k_shard, v_shard
        )

        # Create RadixAttention layer
        attn = RadixAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=sm_scale,
            num_kv_heads=num_kv_heads,
            layer_id=0,
            sliding_window_size=0,
            logit_cap=0,
        )

        # Build reference page table
        seq_lens = forward_batch.seq_lens
        aligned_seq_lens = ((seq_lens + page_size - 1) // page_size) * page_size
        cache_start_loc = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(aligned_seq_lens)]
        )
        padding_size = 4096
        page_table_list = []
        for i in range(forward_batch.batch_size):
            start = cache_start_loc[i]
            end = start + seq_lens[i]
            seq_cache_loc = forward_batch.cache_loc[start:end]
            seq_page_indices = seq_cache_loc // page_size
            unique_pages = unique_in_original_order(seq_page_indices)
            padded = jnp.pad(
                jnp.array(unique_pages, dtype=jnp.int32),
                (0, padding_size - len(unique_pages)),
                constant_values=0,
            )
            page_table_list.append(padded)
        page_table = jnp.stack(page_table_list)

        # Reference expected output
        k_pages = k.reshape(-1, page_size, num_kv_heads, head_dim)
        v_pages = v.reshape(-1, page_size, num_kv_heads, v_head_dim)
        expected = ref_split_attention(
            q.reshape(q.shape[0], num_heads, head_dim),
            k_pages,
            v_pages,
            seq_lens,
            page_table,
            forward_batch.attn_backend.forward_metadata.cu_q_lens,
            forward_batch.attn_backend.forward_metadata.num_seqs,
            causal=True,
            sm_scale=sm_scale,
        )
        jax.block_until_ready(expected)

        # Run through backend
        @jax.jit
        def jit_attn(q, k, v, forward_batch, token_to_kv_pool: KVCache):
            out = attn(q, k, v, forward_batch, token_to_kv_pool)
            return out

        jax_output, kv_updated = jit_attn(
            q_shard, extend_k, extend_v, forward_batch, token_to_kv_pool
        )
        jax.block_until_ready(jax_output)

        # Compare attention output
        rtol = 2e-2
        atol = 1e-2
        jax_flat = np.asarray(jax_output)
        expected_flat = np.asarray(expected.reshape(expected.shape[0], -1))

        diff = np.abs(jax_flat - expected_flat)
        max_diff = np.max(diff)

        print(f"\n=== Split Backend Test: {mode} ===")
        print(
            f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}, v_head_dim={v_head_dim}, page_size={page_size}"
        )
        print(f"JAX output shape: {jax_flat.shape}, Expected shape: {expected_flat.shape}")
        print(f"Max difference: {float(max_diff):.6f}")

        are_close = np.allclose(jax_flat, expected_flat, rtol=rtol, atol=atol)
        self.assertTrue(
            are_close,
            f"Split backend output mismatch, max diff: {float(max_diff):.6f}",
        )

        # Verify cache update returns (k, v) tuple
        self.assertIsInstance(kv_updated, tuple, "Expected (k, v) tuple from split path")
        self.assertEqual(len(kv_updated), 2, "Expected tuple of length 2")
        updated_k, updated_v = kv_updated
        # Pool stores 128-aligned head_dim; updated cache inherits that alignment
        aligned_head_dim = (head_dim + 127) // 128 * 128
        aligned_v_head_dim = (v_head_dim + 127) // 128 * 128
        self.assertEqual(
            updated_k.shape[-1],
            aligned_head_dim,
            f"Updated K head_dim should be {aligned_head_dim} (aligned from {head_dim})",
        )
        self.assertEqual(
            updated_v.shape[-1],
            aligned_v_head_dim,
            f"Updated V head_dim should be {aligned_v_head_dim} (aligned from {v_head_dim})",
        )

        if check_cache:
            # After replace_kv_buffer, verify pool data matches
            token_to_kv_pool.replace_kv_buffer([kv_updated])

            k_buf, v_buf = token_to_kv_pool.get_split_kv_buffer(0)

            # Check that written extend tokens are in the cache
            # Pool stores 128-aligned dims, so compare only the actual (unpadded) portion
            aligned_pos = 0
            for i, (q_len, kv_len) in enumerate(lens):
                aligned_len = ((kv_len + page_size - 1) // page_size) * page_size
                extend_start = aligned_pos + (kv_len - q_len)
                extend_end = aligned_pos + kv_len

                # K cache at extend positions should match extend_k data (trim to actual head_dim)
                cached_k = np.asarray(k_buf[extend_start:extend_end, :, :head_dim])
                original_k = np.asarray(k[extend_start:extend_end])
                k_close = np.allclose(cached_k, original_k, rtol=1e-3, atol=1e-3)
                self.assertTrue(
                    k_close,
                    f"K cache mismatch at seq {i}, max diff: {float(np.max(np.abs(cached_k - original_k))):.6f}",
                )

                cached_v = np.asarray(v_buf[extend_start:extend_end, :, :v_head_dim])
                original_v = np.asarray(v[extend_start:extend_end])
                v_close = np.allclose(cached_v, original_v, rtol=1e-3, atol=1e-3)
                self.assertTrue(
                    v_close,
                    f"V cache mismatch at seq {i}, max diff: {float(np.max(np.abs(cached_v - original_v))):.6f}",
                )

                # Also verify padding region is zero
                if aligned_head_dim > head_dim:
                    pad_k = np.asarray(k_buf[extend_start:extend_end, :, head_dim:])
                    self.assertTrue(
                        np.all(pad_k == 0),
                        f"K cache padding region not zero at seq {i}",
                    )

                aligned_pos += aligned_len

    def test_split_backend_prefill_ps1(self):
        """MHA prefill through FlashAttention backend with split KV cache."""
        self.run_backend_test(
            mode="prefill",
            lens=[(1, 128), (64, 64), (128, 256)],
            num_heads=16,
            head_dim=256,
            num_kv_heads=16,
            v_head_dim=128,
            page_size=1,
        )

    def test_split_backend_decode_ps1(self):
        """MHA decode through FlashAttention backend with split KV cache."""
        self.run_backend_test(
            mode="decode",
            lens=[(1, 119), (1, 128), (1, 256)],
            num_heads=16,
            head_dim=256,
            num_kv_heads=16,
            v_head_dim=128,
            page_size=1,
        )

    def test_split_backend_gqa_prefill_ps1(self):
        """GQA prefill through FlashAttention backend with split KV cache."""
        self.run_backend_test(
            mode="prefill",
            lens=[(1, 128), (125, 125), (64, 256)],
            num_heads=16,
            head_dim=256,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=1,
        )

    def test_split_backend_gqa_decode_ps1(self):
        """GQA decode through FlashAttention backend with split KV cache."""
        self.run_backend_test(
            mode="decode",
            lens=[(1, 127), (1, 128), (1, 512)],
            num_heads=16,
            head_dim=256,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=1,
        )

    # --- page_size=16 tests ---
    def test_split_backend_gqa_prefill_ps16(self):
        """GQA prefill through FlashAttention backend with split KV cache, page_size=16."""
        self.run_backend_test(
            mode="prefill",
            lens=[(1, 128), (125, 125), (64, 256)],
            num_heads=16,
            head_dim=256,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=16,
        )

    def test_split_backend_gqa_decode_ps16(self):
        """GQA decode through FlashAttention backend with split KV cache, page_size=16."""
        self.run_backend_test(
            mode="decode",
            lens=[(1, 127), (1, 128), (1, 512)],
            num_heads=16,
            head_dim=256,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=16,
        )

    def test_split_backend_cache_update(self):
        """Verify cache update correctness after split KV attention."""
        self.run_backend_test(
            mode="prefill",
            lens=[(64, 128), (32, 64)],
            num_heads=16,
            head_dim=256,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=1,
            check_cache=True,
        )

    # --- head_dim=192 backend tests ---
    def test_split_backend_192_prefill_no_prefix(self):
        """Backend prefill with head_dim=192, no prefix (q_len==kv_len avoids set_kv_buffer)."""
        self.run_backend_test(
            mode="prefill",
            lens=[(128, 128), (64, 64), (256, 256)],
            num_heads=16,
            head_dim=192,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=1,
        )

    def test_split_backend_192_prefill_no_prefix_mha(self):
        """Backend MHA prefill with head_dim=192, no prefix."""
        self.run_backend_test(
            mode="prefill",
            lens=[(64, 64), (128, 128)],
            num_heads=16,
            head_dim=192,
            num_kv_heads=16,
            v_head_dim=128,
            page_size=1,
        )

    def test_split_backend_192_decode(self):
        """MHA decode with head_dim=192 (prefix tokens go through kv_cache_update)."""
        self.run_backend_test(
            mode="decode",
            lens=[(1, 119), (1, 128), (1, 256)],
            num_heads=16,
            head_dim=192,
            num_kv_heads=16,
            v_head_dim=128,
            page_size=1,
        )

    def test_split_backend_192_gqa_decode(self):
        """GQA decode with head_dim=192 (prefix tokens go through kv_cache_update)."""
        self.run_backend_test(
            mode="decode",
            lens=[(1, 127), (1, 128), (1, 512)],
            num_heads=16,
            head_dim=192,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=1,
        )

    def test_split_backend_192_prefill_with_prefix(self):
        """GQA prefill with head_dim=192 and prefix (kv_len > q_len triggers set_kv_buffer)."""
        self.run_backend_test(
            mode="prefill",
            lens=[(1, 128), (125, 125), (64, 256)],
            num_heads=16,
            head_dim=192,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=1,
        )

    # --- head_dim=192 + page_size=16 backend tests ---
    def test_split_backend_192_gqa_prefill_ps16(self):
        """GQA prefill with head_dim=192, page_size=16."""
        self.run_backend_test(
            mode="prefill",
            lens=[(1, 128), (125, 125), (64, 256)],
            num_heads=16,
            head_dim=192,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=16,
        )

    def test_split_backend_192_gqa_decode_ps16(self):
        """GQA decode with head_dim=192, page_size=16."""
        self.run_backend_test(
            mode="decode",
            lens=[(1, 127), (1, 128), (1, 512)],
            num_heads=16,
            head_dim=192,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=16,
        )

    # --- head_dim=192 cache verification ---
    def test_split_backend_192_cache_update(self):
        """Verify cache update correctness with head_dim=192 (non-128-aligned, exercises padding)."""
        self.run_backend_test(
            mode="prefill",
            lens=[(64, 128), (32, 64)],
            num_heads=16,
            head_dim=192,
            num_kv_heads=8,
            v_head_dim=128,
            page_size=1,
            check_cache=True,
        )


if __name__ == "__main__":
    unittest.main()
