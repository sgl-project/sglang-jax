"""Accuracy tests for the absorbed-MLA backend (v2 Pallas kernel).

Mirrors the structure of ``test_flashattention.py``: builds a ``ForwardBatch``
with mock metadata, pre-populates the KV cache with prefix tokens, runs the
MLA backend for the extend tokens, and compares against
``ref_mla_ragged_paged_attention`` (the v1 reference implementation).

The MLA kernel consumes a 4-tuple payload
``(ql_nope, q_pe, new_kv_c, new_k_pe)`` instead of ``(q, k, v)``:

    ql_nope:  [T, num_heads, kv_lora_rank]
    q_pe:     [T, num_heads, qk_rope_head_dim]
    new_kv_c: [T, kv_lora_rank]            (shared across heads)
    new_k_pe: [T, qk_rope_head_dim]        (shared across heads)

and returns the latent output ``o_latent: [T, num_heads, kv_lora_rank]``.

Both the backend (ragged page_indices) and the v1 reference (padded
page_indices) read/write against the same underlying 4D paged cache buffer; we
build the two index layouts side by side and dispatch them to the right call.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.mla.v1.ref import ref_mla_ragged_paged_attention
from sgl_jax.srt.kernels.mla.v2.kernel import align_to, get_dtype_packing
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import MLATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)


def _make_random_inputs(
    lens,
    num_heads,
    kv_lora_rank,
    qk_rope_head_dim,
    page_size,
    dtype,
    seed=0,
):
    """Generate ql_nope / q_pe / kv_c / k_pe.

    Q is shaped by the *extend* token count; kv_c / k_pe are shaped by the
    *aligned* total token count (prefix + extend per seq, padded up to
    ``page_size``). Padding slots between sequences are left as zeros.
    """
    extend_total = sum(q_len for q_len, _ in lens)
    aligned_seq_lens = [((kv_len + page_size - 1) // page_size) * page_size for _, kv_len in lens]
    aligned_total = sum(aligned_seq_lens)

    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    ql_nope = jax.random.normal(k1, (extend_total, num_heads, kv_lora_rank), dtype=dtype)
    q_pe = jax.random.normal(k2, (extend_total, num_heads, qk_rope_head_dim), dtype=dtype)

    kv_c_full = np.zeros((aligned_total, kv_lora_rank), dtype=np.float32)
    k_pe_full = np.zeros((aligned_total, qk_rope_head_dim), dtype=np.float32)
    offset = 0
    for i, (_, kv_len) in enumerate(lens):
        seq_kv_c = np.asarray(
            jax.random.normal(jax.random.fold_in(k3, i), (kv_len, kv_lora_rank), dtype=dtype),
            dtype=np.float32,
        )
        seq_k_pe = np.asarray(
            jax.random.normal(jax.random.fold_in(k4, i), (kv_len, qk_rope_head_dim), dtype=dtype),
            dtype=np.float32,
        )
        kv_c_full[offset : offset + kv_len] = seq_kv_c
        k_pe_full[offset : offset + kv_len] = seq_k_pe
        offset += aligned_seq_lens[i]

    return ql_nope, q_pe, kv_c_full, k_pe_full, aligned_seq_lens


def _build_caches_and_indices(
    lens,
    kv_c_full,
    k_pe_full,
    aligned_seq_lens,
    page_size,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    total_num_pages,
):
    """Pre-populate the 4D KV cache with prefix tokens.

    Returns:
      cache_np: prefix-populated cache (numpy, cast-friendly dtype)
      ragged_page_indices: i32[sum_pages_per_seq] — layout consumed by the v2
          kernel / backend (each seq's pages tightly concatenated).
      padded_page_indices: i32[batch_size * pages_per_seq] — layout consumed
          by the v1 reference.
      pages_per_seq: int — padding stride used for the padded layout.
      cache_loc_flat: i32[sum(aligned_seq_lens)] — per-token cache locations
          (feeds ``batch.cache_loc`` / metadata building).
    """
    kv_packing = get_dtype_packing(dtype)
    nope_dim = align_to(kv_lora_rank, 128)
    rope_dim = align_to(qk_rope_head_dim, 128)
    kv_dim = nope_dim + rope_dim
    page_size_per_kv_packing = max(align_to(page_size, kv_packing) // kv_packing, 1)

    np_dtype = np.float32 if dtype == jnp.float32 else np.float32
    cache_np = np.zeros(
        (total_num_pages, page_size_per_kv_packing, kv_packing, kv_dim), dtype=np_dtype
    )

    # pad kv_c / k_pe to 128-aligned dims (mirrors v2 kernel alignment)
    kv_c_np = kv_c_full
    k_pe_np = k_pe_full
    if nope_dim > kv_lora_rank:
        kv_c_np = np.pad(kv_c_np, ((0, 0), (0, nope_dim - kv_lora_rank)))
    if rope_dim > qk_rope_head_dim:
        k_pe_np = np.pad(k_pe_np, ((0, 0), (0, rope_dim - qk_rope_head_dim)))

    # Sequential page layout: seq i gets pages [page_start[i], page_start[i]+num_pages_i)
    page_counts = [al // page_size for al in aligned_seq_lens]
    page_starts = np.concatenate([[0], np.cumsum(page_counts)]).astype(np.int32)
    pages_per_seq = int(max(page_counts)) if page_counts else 1

    ragged_page_indices = []
    padded_page_indices = np.zeros(len(lens) * pages_per_seq, dtype=np.int32)
    aligned_offsets = np.concatenate([[0], np.cumsum(aligned_seq_lens)]).astype(np.int32)
    cache_loc_flat = np.zeros(int(aligned_offsets[-1]), dtype=np.int32)

    for i, ((q_len, kv_len), aligned_len) in enumerate(zip(lens, aligned_seq_lens)):
        num_pages_i = aligned_len // page_size
        seq_page_ids = np.arange(num_pages_i, dtype=np.int32) + int(page_starts[i])
        ragged_page_indices.append(seq_page_ids)
        padded_page_indices[i * pages_per_seq : i * pages_per_seq + num_pages_i] = seq_page_ids

        off = int(aligned_offsets[i])
        positions = np.arange(aligned_len, dtype=np.int32)
        page_ids_per_tok = int(page_starts[i]) + positions // page_size
        cache_loc_flat[off : off + aligned_len] = page_ids_per_tok * page_size + (
            positions % page_size
        )

        prefix_end = kv_len - q_len
        if prefix_end > 0:
            pref_pos = np.arange(prefix_end, dtype=np.int32)
            pref_page_ids = int(page_starts[i]) + pref_pos // page_size
            pref_pos_in_page = pref_pos % page_size
            pref_rows = pref_pos_in_page // kv_packing
            pref_cols = pref_pos_in_page % kv_packing
            cache_np[pref_page_ids, pref_rows, pref_cols, :nope_dim] = kv_c_np[
                off : off + prefix_end
            ]
            cache_np[pref_page_ids, pref_rows, pref_cols, nope_dim : nope_dim + rope_dim] = k_pe_np[
                off : off + prefix_end
            ]

    ragged_page_indices = (
        np.concatenate(ragged_page_indices).astype(np.int32)
        if ragged_page_indices
        else np.zeros((0,), dtype=np.int32)
    )
    return (
        cache_np,
        ragged_page_indices,
        padded_page_indices,
        pages_per_seq,
        cache_loc_flat,
    )


def create_mla_forward_batch(
    mode,
    lens,
    num_heads,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    page_size,
    dtype,
    max_total_token_size,
):
    assert mode in ("prefill", "decode")
    ql_nope, q_pe, kv_c_full, k_pe_full, aligned_seq_lens = _make_random_inputs(
        lens,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
        dtype=dtype,
    )

    pool = MLATokenToKVPool(
        size=max_total_token_size,
        page_size=page_size,
        dtype=dtype,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        layer_num=1,
        mesh=mesh,
    )
    total_num_pages = pool.kv_buffer[0].shape[0]

    (
        cache_np,
        _ragged_page_indices,
        padded_page_indices,
        _pages_per_seq,
        cache_loc_flat,
    ) = _build_caches_and_indices(
        lens,
        kv_c_full,
        k_pe_full,
        aligned_seq_lens,
        page_size=page_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dtype=dtype,
        total_num_pages=total_num_pages,
    )

    cache_jnp = jnp.asarray(cache_np, dtype=dtype)
    cache_jnp = jax.device_put(cache_jnp, pool.kv_sharding)
    pool.replace_kv_buffer([cache_jnp])

    seq_lens = np.array([kv_len for _, kv_len in lens], dtype=np.int32)
    q_lens = np.array([q_len for q_len, _ in lens], dtype=np.int32)
    extend_prefix_lens = seq_lens - q_lens

    aligned_offsets = np.concatenate([[0], np.cumsum(aligned_seq_lens)]).astype(np.int32)
    new_kv_c_parts = []
    new_k_pe_parts = []
    for i, (q_len, kv_len) in enumerate(lens):
        start = int(aligned_offsets[i]) + (kv_len - q_len)
        end = int(aligned_offsets[i]) + kv_len
        new_kv_c_parts.append(kv_c_full[start:end])
        new_k_pe_parts.append(k_pe_full[start:end])
    new_kv_c = jnp.asarray(np.concatenate(new_kv_c_parts, axis=0), dtype=dtype)
    new_k_pe = jnp.asarray(np.concatenate(new_k_pe_parts, axis=0), dtype=dtype)

    req_pool_indices = np.arange(len(lens), dtype=np.int32)
    positions = np.arange(int(aligned_offsets[-1]), dtype=np.int32)
    input_ids = np.arange(int(q_lens.sum()), dtype=np.int32)

    if mode == "prefill":
        forward_mode = ForwardMode.EXTEND
        out_cache_loc_parts = []
        for i, (q_len, kv_len) in enumerate(lens):
            start = int(aligned_offsets[i]) + (kv_len - q_len)
            end = int(aligned_offsets[i]) + kv_len
            out_cache_loc_parts.append(cache_loc_flat[start:end])
        out_cache_loc = np.concatenate(out_cache_loc_parts).astype(np.int32)
    else:
        forward_mode = ForwardMode.DECODE
        out_cache_loc_parts = []
        for i, (q_len, kv_len) in enumerate(lens):
            assert q_len == 1, "decode expects q_len == 1"
            end = int(aligned_offsets[i]) + kv_len
            out_cache_loc_parts.append(cache_loc_flat[end - 1 : end])
        out_cache_loc = np.concatenate(out_cache_loc_parts).astype(np.int32)

    backend = MLAAttentionBackend(
        num_attn_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        page_size=page_size,
        mesh=mesh,
        attention_data_partition_axis="data",
    )

    mwb = ModelWorkerBatch(
        bid=1,
        forward_mode=forward_mode,
        input_ids=input_ids,
        real_input_ids_len=input_ids.shape[0],
        seq_lens=seq_lens,
        out_cache_loc=out_cache_loc,
        req_pool_indices=req_pool_indices,
        sampling_info=None,
        positions=positions,
        cache_loc=cache_loc_flat,
        extend_seq_lens=q_lens if mode == "prefill" else None,
        extend_prefix_lens=extend_prefix_lens if mode == "prefill" else None,
        return_logprob=False,
        return_output_logprob_only=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        logits_indices=np.cumsum(q_lens) - 1 if mode == "prefill" else None,
        real_bs=len(lens),
        real_bs_per_dp=[len(lens)],
        dp_size=1,
        per_dp_bs_size=len(lens),
        spec_info=None,
    )

    fb = ForwardBatch(
        bid=1,
        forward_mode=forward_mode,
        batch_size=len(lens),
        input_ids=jnp.asarray(input_ids),
        req_pool_indices=jnp.asarray(req_pool_indices),
        seq_lens=jnp.asarray(seq_lens),
        out_cache_loc=jnp.asarray(out_cache_loc),
        positions=jnp.asarray(positions),
        attn_backend=backend,
        cache_loc=jnp.asarray(cache_loc_flat),
        extend_prefix_lens=jnp.asarray(extend_prefix_lens) if mode == "prefill" else None,
        extend_seq_lens=jnp.asarray(q_lens) if mode == "prefill" else None,
        spec_info=None,
    )
    fb.attn_backend.forward_metadata = backend.get_forward_metadata(mwb)

    return (
        fb,
        pool,
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_jnp,
        padded_page_indices,
        seq_lens,
    )


class TestMLAAttention(CustomTestCase):
    """Accuracy tests for ``MLAAttentionBackend`` vs. the v1 reference."""

    # Shape constants inspired by DeepSeek-V2/V3 MLA.
    NUM_HEADS = 16
    KV_LORA_RANK = 512
    QK_NOPE_DIM = 128
    QK_ROPE_DIM = 64
    V_HEAD_DIM = 128

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.rng_key = jax.random.PRNGKey(0)
        np.random.seed(0)

    def run_test(
        self,
        mode,
        lens,
        num_heads=None,
        kv_lora_rank=None,
        qk_nope_head_dim=None,
        qk_rope_head_dim=None,
        v_head_dim=None,
        page_size=16,
        dtype=jnp.bfloat16,
        max_total_token_size=200000,
        sliding_window=None,
        soft_cap=None,
    ):
        num_heads = num_heads or self.NUM_HEADS
        kv_lora_rank = kv_lora_rank or self.KV_LORA_RANK
        qk_nope_head_dim = qk_nope_head_dim or self.QK_NOPE_DIM
        qk_rope_head_dim = qk_rope_head_dim or self.QK_ROPE_DIM
        v_head_dim = v_head_dim or self.V_HEAD_DIM

        (
            fb,
            pool,
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_jnp,
            padded_page_indices,
            seq_lens_np,
        ) = create_mla_forward_batch(
            mode,
            lens,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            page_size=page_size,
            dtype=dtype,
            max_total_token_size=max_total_token_size,
        )

        sm_scale = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5
        # attn_mqa layout mirrors sglang-gpu's MLA RadixAttention instance:
        # qk contraction dim = kv_lora_rank + qk_rope_head_dim; MQA (one KV
        # head); latent V width = kv_lora_rank.
        attn = RadixAttention(
            num_heads=num_heads,
            head_dim=kv_lora_rank + qk_rope_head_dim,
            scaling=sm_scale,
            num_kv_heads=1,
            v_head_dim=kv_lora_rank,
            layer_id=0,
            sliding_window_size=sliding_window or 0,
            logit_cap=soft_cap or 0,
        )

        # Reshape latent K/K-rope to the 3-D [T, 1, *] form the RadixAttention
        # contract expects (single KV head). The backend strips the axis back
        # off before handing to the Pallas kernel.
        new_kv_c_3d = new_kv_c.reshape(new_kv_c.shape[0], 1, new_kv_c.shape[1])
        new_k_pe_3d = new_k_pe.reshape(new_k_pe.shape[0], 1, new_k_pe.shape[1])

        # Match the backend's shard_map in_specs: Q tensors are head-sharded on
        # "tensor", latent K/V are replicated (single shared KV head).
        heads_sharded = NamedSharding(mesh, P(None, "tensor", None))
        replicated = NamedSharding(mesh, P(None, None, None))
        ql_nope_s = jax.device_put(ql_nope, heads_sharded)
        q_pe_s = jax.device_put(q_pe, heads_sharded)
        new_kv_c_3d_s = jax.device_put(new_kv_c_3d, replicated)
        new_k_pe_3d_s = jax.device_put(new_k_pe_3d, replicated)

        @jax.jit
        def jit_call(q_, k_, v_, q_rope_, k_rope_, forward_batch, pool_):
            return attn(
                q_,
                k_,
                v_,
                forward_batch,
                pool_,
                q_rope=q_rope_,
                k_rope=k_rope_,
            )

        jax_output, _ = jit_call(
            ql_nope_s, new_kv_c_3d_s, new_kv_c_3d_s, q_pe_s, new_k_pe_3d_s, fb, pool
        )
        jax.block_until_ready(jax_output)

        cu_q_lens_np = np.concatenate([[0], np.cumsum([q for q, _ in lens])]).astype(np.int32)
        num_seqs = int(np.sum(seq_lens_np > 0))
        # Match the backend's MLAAttentionMetadata convention: decode -> all
        # decode-only, extend -> all mixed (chunked-prefill-only is skipped by
        # the kernel). Only distribution[-1] matters to the v1 reference.
        if mode == "decode":
            distribution = np.array([num_seqs, num_seqs, num_seqs], dtype=np.int32)
        else:
            distribution = np.array([0, 0, num_seqs], dtype=np.int32)

        expected, _ = ref_mla_ragged_paged_attention(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            jax.device_put(cache_jnp, NamedSharding(mesh, P())),
            jnp.asarray(seq_lens_np, dtype=jnp.int32),
            jnp.asarray(padded_page_indices, dtype=jnp.int32),
            jnp.asarray(cu_q_lens_np, dtype=jnp.int32),
            jnp.asarray(distribution, dtype=jnp.int32),
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
        )
        expected_np = np.asarray(expected[..., :kv_lora_rank])
        jax_np = np.asarray(jax_output)

        rtol, atol = 2e-2, 1e-2
        diff = np.abs(jax_np.astype(np.float32) - expected_np.astype(np.float32))
        max_diff = float(np.max(diff))
        print(
            f"mode={mode} page={page_size} lens={lens}: "
            f"shape={jax_np.shape} max_diff={max_diff:.6f}"
        )
        self.assertTrue(
            np.allclose(jax_np, expected_np, rtol=rtol, atol=atol),
            f"MLA output differs from reference; max_diff={max_diff}",
        )

    # ---------------- prefill / extend -----------------
    # Each prefill test is a mixed extend batch (different q_len per seq,
    # including pure prefill where q_len == kv_len, extend cases where
    # q_len < kv_len, and decode-like single-token extends). Shapes mirror the
    # lens lists in test_flashattention.py.
    PREFILL_LENS = [
        (1, 128),
        (125, 125),
        (1024, 1024),
        (123, 522),
        (1, 511),
        (512, 1024),
    ]

    def test_prefill_page_size_8(self):
        self.run_test("prefill", self.PREFILL_LENS, page_size=8)

    def test_prefill_page_size_16(self):
        self.run_test("prefill", self.PREFILL_LENS, page_size=16)

    def test_prefill_page_size_32(self):
        self.run_test("prefill", self.PREFILL_LENS, page_size=32)

    def test_prefill_page_size_64(self):
        self.run_test("prefill", self.PREFILL_LENS, page_size=64)

    def test_prefill_single_seq_long(self):
        # Single-sequence prefill, pure q_len == kv_len, spans many pages.
        self.run_test("prefill", [(1024, 1024)], page_size=32)

    # ---------------- decode -----------------
    # Batch of single-token decodes with diverse kv_lens around page boundaries
    # (e.g. 127 vs 128 vs 129 for page_size=64 stresses aligned vs. partial
    # trailing pages). Same lens list pattern as test_flashattention.
    DECODE_LENS = [
        (1, 119),
        (1, 127),
        (1, 128),
        (1, 129),
        (1, 133),
        (1, 1001),
        (1, 1023),
        (1, 1024),
        (1, 1025),
    ]

    def test_decode_page_size_8(self):
        self.run_test("decode", self.DECODE_LENS, page_size=8)

    def test_decode_page_size_16(self):
        self.run_test("decode", self.DECODE_LENS, page_size=16)

    def test_decode_page_size_32(self):
        self.run_test("decode", self.DECODE_LENS, page_size=32)

    def test_decode_page_size_64(self):
        self.run_test("decode", self.DECODE_LENS, page_size=64)

    def test_decode_batched_decode_branch(self):
        # batch_size == decode_batch_size (=4), so the BATCHED_DECODE branch
        # in mla_ragged_paged_attention fires (the other decode tests above
        # have 9 seqs, which also exercises the batched + residual branches).
        self.run_test(
            "decode",
            [(1, 64), (1, 128), (1, 192), (1, 256)],
            page_size=16,
        )

    # ---------------- sliding window -----------------
    # sliding_window bounds the causal attention span: each query token only
    # attends to the previous `sliding_window` KV tokens (plus itself). Uses
    # lens with kv_lens crossing the window boundary to verify both inside
    # and outside the window are handled.
    def test_prefill_sliding_window(self):
        self.run_test(
            "prefill",
            [(1, 128), (64, 64), (128, 256), (100, 300), (1, 400)],
            page_size=16,
            sliding_window=128,
        )

    def test_decode_sliding_window(self):
        self.run_test(
            "decode",
            [(1, 256), (1, 400), (1, 512), (1, 1024)],
            page_size=16,
            sliding_window=128,
        )

    # ---------------- soft cap (logit soft-capping) -----------------
    # soft_cap applies `s ← soft_cap * tanh(s / soft_cap)` to attention logits
    # before masking + softmax, bounding magnitude smoothly. Gemma-2 / DeepSeek
    # style cap ≈ 30.
    def test_prefill_soft_cap(self):
        self.run_test(
            "prefill",
            [(1, 128), (64, 64), (128, 256), (100, 300), (1, 400)],
            page_size=16,
            soft_cap=30.0,
        )

    def test_decode_soft_cap(self):
        self.run_test(
            "decode",
            [(1, 256), (1, 400), (1, 512), (1, 1024)],
            page_size=16,
            soft_cap=30.0,
        )

    # Combined sliding_window + soft_cap.
    def test_prefill_sliding_window_and_soft_cap(self):
        self.run_test(
            "prefill",
            [(1, 128), (64, 64), (128, 256), (100, 300), (1, 400)],
            page_size=16,
            sliding_window=128,
            soft_cap=30.0,
        )

    def test_decode_sliding_window_and_soft_cap(self):
        self.run_test(
            "decode",
            [(1, 256), (1, 400), (1, 512), (1, 1024)],
            page_size=16,
            sliding_window=128,
            soft_cap=30.0,
        )


if __name__ == "__main__":
    unittest.main()
