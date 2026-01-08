import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention import (
    ref_ragged_paged_attention,
)
from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.managers.schedule_batch import (
    PADDING_BUCKETS,
    ModelWorkerBatch,
    find_padding_size,
)
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase


def set_mesh(tp_size: int, dp_size: int):
    if tp_size * dp_size != jax.device_count():
        raise RuntimeError(
            f"tp_size * dp_size must equal to available device count {jax.device_count()}, but got tp_size {tp_size}, dp_size {dp_size}"
        )
    mesh = create_device_mesh(ici_parallelism=[dp_size, tp_size], dcn_parallelism=[1, 1])
    jax.sharding.set_mesh(mesh)
    return mesh


def unique_in_original_order(arr: jax.Array) -> jax.Array:
    unique_info = jnp.unique_all(arr)
    unique_values = unique_info.values
    original_indices = unique_info.indices

    # Sort the original indices to get the correct order
    sorted_order = jnp.argsort(original_indices)

    # Reorder the unique values based on the sorted indices
    unique_in_original_order = unique_values[sorted_order]
    return unique_in_original_order


def generate_random_req_data(reqs, num_heads, num_kv_heads, head_dim, seed):
    """
    Generates raw Q/K/V data blocks using NumPy to ensure compatibility with
    CPU-based metadata construction and avoid device-to-host overhead.
    """
    rng = np.random.default_rng(seed)
    results = []
    for q_len, kv_len in reqs:
        results.append(
            {
                # Generating as float32 for CPU processing; will be cast to target dtype later
                "q": rng.standard_normal((q_len, num_heads, head_dim)).astype(np.float32),
                "k": rng.standard_normal((kv_len, num_kv_heads, head_dim)).astype(np.float32),
                "v": rng.standard_normal((kv_len, num_kv_heads, head_dim)).astype(np.float32),
                "q_len": q_len,
                "kv_len": kv_len,
            }
        )
    return results


def create_test_data(
    mode,
    lens_dict,  # {dp_rank: [(q_len, kv_len)]}
    num_heads,
    head_dim,
    num_kv_heads,
    page_size,
    mesh,
    dp_size=1,
    model_config=None,
    max_total_token_size=16384,
):
    """
    Creates a full ForwardBatch for testing with DP support.
    Data is generated on CPU via NumPy and transferred to device at the final step.
    """
    assert mode in ["prefill", "decode"]
    is_prefill = mode == "prefill"

    # 1. Calculate per-DP padding sizes based on input requirements
    max_tokens_per_dp = 0
    max_bs_per_dp = 0
    for dp_rank, reqs in lens_dict.items():
        dp_tokens = sum(r[0] for r in reqs) if is_prefill else len(reqs)
        max_tokens_per_dp = max(max_tokens_per_dp, dp_tokens)
        max_bs_per_dp = max(max_bs_per_dp, len(reqs))

    per_dp_token_padding, _ = find_padding_size(max_tokens_per_dp, PADDING_BUCKETS)
    per_dp_bs_padding, _ = find_padding_size(max_bs_per_dp, [1, 2, 4, 8, 16, 32, 64])

    total_token_size = per_dp_token_padding * dp_size
    total_bs = per_dp_bs_padding * dp_size

    # --- KEY FIX: Determine dimensions based on mode ---
    # In Prefill: Inputs are [Tokens], layout strided by token_padding
    # In Decode:  Inputs are [Batch],  layout strided by bs_padding
    global_input_size = total_token_size if is_prefill else total_bs

    # 2. Initialize global NumPy containers (CPU)
    # input_ids, positions, out_cache_loc MUST match the active input dimension
    input_ids_cpu = np.zeros(global_input_size, dtype=np.int32)
    positions_cpu = np.zeros(global_input_size, dtype=np.int32)
    out_cache_loc_cpu = np.full(global_input_size, -1, dtype=np.int32)

    # These are always (total_bs,) because they are per-request metadata
    req_pool_indices_cpu = np.full(total_bs, -1, dtype=np.int32)
    seq_lens_cpu = np.zeros(total_bs, dtype=np.int32)

    extend_prefix_lens_cpu = np.zeros(total_bs, dtype=np.int32) if is_prefill else None
    extend_seq_lens_cpu = np.zeros(total_bs, dtype=np.int32) if is_prefill else None

    # Intermediate CPU buffers for QKV
    # Fix: Use global_input_size instead of always total_token_size
    q_global = np.zeros((global_input_size, num_heads, head_dim), dtype=np.float32)
    k_global = np.zeros((global_input_size, num_kv_heads, head_dim), dtype=np.float32)
    v_global = np.zeros((global_input_size, num_kv_heads, head_dim), dtype=np.float32)

    # 3. Process each DP Rank locally
    dp_kv_data_to_write = []
    dp_cache_locs_flat = []
    per_dp_infos = {}

    global_req_idx = 0
    for dp_rank in range(dp_size):
        reqs = lens_dict.get(dp_rank, [])
        rank_raw_data = generate_random_req_data(
            reqs, num_heads, num_kv_heads, head_dim, seed=42 + dp_rank
        )
        per_dp_infos[dp_rank] = {"req_infos": rank_raw_data, "reqs": reqs}

        dp_bs_offset = dp_rank * per_dp_bs_padding
        dp_token_offset = dp_rank * per_dp_token_padding

        curr_token_idx = 0
        curr_cache_slot = 0
        rank_cache_locs = []
        rank_kv_indices, rank_k_blocks, rank_v_blocks = [], [], []

        for i, info in enumerate(rank_raw_data):
            q_len, kv_len = info["q_len"], info["kv_len"]

            # --- Metadata Assignment ---
            # Always use BS offset for request-level metadata
            req_pool_indices_cpu[dp_bs_offset + i] = global_req_idx
            seq_lens_cpu[dp_bs_offset + i] = kv_len
            global_req_idx += 1

            # --- KV Cache Slot Allocation ---
            aligned_kv_len = ((kv_len + page_size - 1) // page_size) * page_size
            slots = np.arange(curr_cache_slot, curr_cache_slot + aligned_kv_len)
            curr_cache_slot += aligned_kv_len

            rank_cache_locs.extend(slots[:kv_len])
            rank_cache_locs.extend([0] * (aligned_kv_len - kv_len))

            # KV Data for Pool (Initial load)
            k_aligned = np.zeros((aligned_kv_len, num_kv_heads, head_dim), dtype=np.float32)
            v_aligned = np.zeros((aligned_kv_len, num_kv_heads, head_dim), dtype=np.float32)
            k_aligned[:kv_len] = info["k"]
            v_aligned[:kv_len] = info["v"]
            rank_kv_indices.append(slots)
            rank_k_blocks.append(k_aligned)
            rank_v_blocks.append(v_aligned)

            # --- Fill Batch Tensors (NumPy Slicing) ---
            if is_prefill:
                # Prefill: Layout inputs continuously in token space
                # Stride logic: dp_token_offset + local_token_idx
                prefix_len = kv_len - q_len
                slc = slice(
                    dp_token_offset + curr_token_idx, dp_token_offset + curr_token_idx + q_len
                )

                q_global[slc] = info["q"]
                k_global[slc] = info["k"][prefix_len:kv_len]
                v_global[slc] = info["v"][prefix_len:kv_len]

                positions_cpu[slc] = np.arange(prefix_len, kv_len)
                input_ids_cpu[slc] = np.arange(q_len)
                out_cache_loc_cpu[slc] = slots[prefix_len:kv_len]

                extend_prefix_lens_cpu[dp_bs_offset + i] = prefix_len
                extend_seq_lens_cpu[dp_bs_offset + i] = q_len
                curr_token_idx += q_len
            else:
                # Decode: Layout inputs one-per-request in batch space
                # Stride logic: dp_bs_offset + local_req_idx (i)
                idx = dp_bs_offset + i

                q_global[idx] = info["q"][-1]
                k_global[idx] = info["k"][-1]
                v_global[idx] = info["v"][-1]

                positions_cpu[idx] = kv_len - 1
                input_ids_cpu[idx] = 1
                out_cache_loc_cpu[idx] = slots[kv_len - 1]
                # curr_token_idx not used for offset calculation in Decode, but tracked anyway
                curr_token_idx += 1

        dp_cache_locs_flat.append(rank_cache_locs)
        dp_kv_data_to_write.append((rank_kv_indices, rank_k_blocks, rank_v_blocks))

    # 4. Initialize and Populate KV Cache Pool
    target_dtype = jnp.bfloat16 if model_config.get("bf16") else jnp.float32
    current_kv_cache = MHATokenToKVPool(
        size=max_total_token_size,
        page_size=page_size,
        dtype=target_dtype,
        head_num=num_kv_heads,
        head_dim=head_dim,
        layer_num=model_config["num_hidden_layers"],
        mesh=mesh,
        dp_size=dp_size,
    )

    max_kv_slots = max(sum(len(idx) for idx in item[0]) for item in dp_kv_data_to_write if item[0])
    all_idx, all_k, all_v = [], [], []

    for rank_idx, k_blocks, v_blocks in dp_kv_data_to_write:
        r_idx = np.concatenate(rank_idx) if rank_idx else np.array([], dtype=np.int32)
        r_k = np.concatenate(k_blocks) if k_blocks else np.zeros((0, num_kv_heads, head_dim))
        r_v = np.concatenate(v_blocks) if v_blocks else np.zeros((0, num_kv_heads, head_dim))

        pad = max_kv_slots - len(r_idx)
        all_idx.append(np.pad(r_idx, (0, pad), constant_values=-1))
        all_k.append(np.pad(r_k, ((0, pad), (0, 0), (0, 0))))
        all_v.append(np.pad(r_v, ((0, pad), (0, 0), (0, 0))))

    sharding_idx = NamedSharding(mesh, P("data"))
    sharding_kv = NamedSharding(mesh, P("data", "tensor", None))

    current_kv_cache.set_kv_buffer(
        0,
        jax.device_put(np.concatenate(all_idx), sharding_idx),
        jax.device_put(jnp.array(np.concatenate(all_k), dtype=target_dtype), sharding_kv),
        jax.device_put(jnp.array(np.concatenate(all_v), dtype=target_dtype), sharding_kv),
    )

    # 5. Build Final Batch Objects
    max_loc_len = max(len(locs) for locs in dp_cache_locs_flat)
    per_dp_cache_loc_size, _ = find_padding_size(max_loc_len, PADDING_BUCKETS)
    cache_loc_cpu = np.zeros(per_dp_cache_loc_size * dp_size, dtype=np.int32)
    for r, locs in enumerate(dp_cache_locs_flat):
        cache_loc_cpu[r * per_dp_cache_loc_size : r * per_dp_cache_loc_size + len(locs)] = locs

    mwb = ModelWorkerBatch(
        bid=1,
        forward_mode=ForwardMode.EXTEND if is_prefill else ForwardMode.DECODE,
        input_ids=input_ids_cpu,
        real_input_ids_len=input_ids_cpu.shape[
            0
        ],  # Correctly reflects prefill tokens vs decode batch
        seq_lens=seq_lens_cpu,
        out_cache_loc=out_cache_loc_cpu,
        req_pool_indices=req_pool_indices_cpu,
        positions=positions_cpu,
        cache_loc=cache_loc_cpu,
        extend_seq_lens=extend_seq_lens_cpu,
        extend_prefix_lens=extend_prefix_lens_cpu,
        real_bs=total_bs,
        dp_size=dp_size,
        per_dp_bs_size=per_dp_bs_padding,
        lora_ids=["0"] * total_bs,
        sampling_info=None,
        return_logprob=False,
        return_output_logprob_only=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        logits_indices=np.zeros(total_bs, dtype=np.int32) if is_prefill else None,
    )

    attn_backend = FlashAttention(num_heads, num_kv_heads, head_dim, page_size=page_size, mesh=mesh)
    fb = ForwardBatch.init_new(
        mwb, type("DummyRunner", (), {"mesh": mesh, "attn_backend": attn_backend})()
    )
    fb.attn_backend.forward_metadata = attn_backend.get_forward_metadata(mwb)

    return (
        fb,
        current_kv_cache,
        jnp.array(q_global, dtype=target_dtype),
        jnp.array(k_global, dtype=target_dtype),
        jnp.array(v_global, dtype=target_dtype),
        per_dp_infos,
        per_dp_bs_padding,
        per_dp_token_padding,
    )


class TestAttention(CustomTestCase):
    """Test cases for the Attention layer."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

        # Initialize random seeds for reproducible results
        self.rng_key = jax.random.PRNGKey(42)
        np.random.seed(42)

    def run_test(
        self,
        mode,
        lens_dict,
        mode_args,
        mesh,
        dp_size=1,
        max_total_token_size=16384,
        sliding_window=None,
        logit_cap=None,
        xai_temperature_len=None,
    ):
        num_heads, head_dim, num_kv_heads, page_size, dtype = mode_args
        causal = True

        is_bf16 = dtype == jnp.bfloat16

        print(f"[DEBUG] Creating test data for mode={mode}, dp_size={dp_size}")
        (
            forward_batch,
            token_to_kv_pool,
            q,
            k,
            v,
            per_dp_infos,
            per_dp_bs_padding,
            per_dp_token_padding,
        ) = create_test_data(
            mode,
            lens_dict,
            num_heads,
            head_dim,
            num_kv_heads,
            page_size,
            mesh,
            dp_size=dp_size,
            model_config={
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "num_hidden_layers": 1,
                "bf16": is_bf16,
                "xai_temperature_len": xai_temperature_len,
            },
            max_total_token_size=max_total_token_size,
        )
        print(f"[DEBUG] Test data created, q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")

        expected_np = compute_dp_reference_attention(
            mode=mode,
            per_dp_infos=per_dp_infos,
            dp_size=dp_size,
            per_dp_bs=per_dp_bs_padding,
            per_dp_token_padding=per_dp_token_padding,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            causal=causal,
            sliding_window=sliding_window,
            logit_cap=logit_cap,
            xai_temperature_len=xai_temperature_len,
        )

        sharding = NamedSharding(mesh, P("data", "tensor"))
        q_shard = jax.device_put(q, sharding)
        k_shard = jax.device_put(k, sharding)
        v_shard = jax.device_put(v, sharding)
        print(
            f"[DEBUG] Q/K/V sharded, q_shard.shape={q_shard.shape}, k_shard.shape={k_shard.shape}, v_shard.shape={v_shard.shape}"
        )

        # JAX attention
        attn = RadixAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=head_dim**-0.5,
            num_kv_heads=num_kv_heads,
            layer_id=0,
            sliding_window_size=sliding_window or 0,
            logit_cap=logit_cap or 0,
        )

        if xai_temperature_len is not None and xai_temperature_len > 0:
            attn.xai_temperature_len = xai_temperature_len

        @jax.jit
        def jit_attn(q, k, v, forward_batch, token_to_kv_pool):
            out = attn(q, k, v, forward_batch, token_to_kv_pool)
            return out

        # run
        print("[DEBUG] Running jit_attn...")
        jax_output, _ = jit_attn(q_shard, k_shard, v_shard, forward_batch, token_to_kv_pool)
        print("[DEBUG] jit_attn completed")
        jax_output = jax.block_until_ready(jax_output)

        # Compare
        jax_output_np = np.array(jax_output)
        print(f"[DEBUG] jax_output.shape={jax_output_np.shape}, expected.shape={expected_np.shape}")

        np.testing.assert_allclose(
            jax_output_np,
            expected_np,
            rtol=2e-2,
            atol=2e-2,
            err_msg="Output mismatch",
        )

    def test_mha_prefill_accuracy_page_size_1_dp_4_tp_1(self):
        """Test JAX attention accuracy against reference with DP=4"""
        mesh = set_mesh(tp_size=1, dp_size=4)

        num_heads = 8
        num_kv_heads = 8
        head_dim = 128

        lens_dict = {
            0: [(1, 128), (125, 125)],
            1: [(1024, 1024)],
            2: [(123, 522), (1, 511)],
            3: [(512, 1024)],
        }

        self.run_test(
            "prefill", lens_dict, (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16), mesh, 4
        )

    def test_mha_decode_accuracy_page_size_1_dp_4_tp_1(self):
        """Test JAX attention accuracy against reference with DP=4"""
        mesh = set_mesh(tp_size=1, dp_size=4)

        num_heads = 8
        num_kv_heads = 8
        head_dim = 128

        lens_dict = {
            0: [(1, 119), (1, 127)],
            1: [(1, 128), (1, 129)],
            2: [(1, 133), (1, 1001)],
            3: [(1, 1023), (1, 1024), (1, 1025)],
        }

        self.run_test(
            "decode", lens_dict, (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16), mesh, 4
        )

    def test_minimal_single_request_prefill(self):
        """Minimal test with single request, q=4, kv=4"""
        mesh = set_mesh(tp_size=1, dp_size=4)

        num_heads = 8
        num_kv_heads = 8
        head_dim = 128

        # Single request with q=4, kv=4 on DP rank 0 only
        lens_dict = {
            0: [(4, 4)],
        }

        self.run_test(
            "prefill", lens_dict, (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16), mesh, 4
        )

    def test_minimal_single_request_decode(self):
        """Minimal test with single request, q=1, kv=4"""
        mesh = set_mesh(tp_size=1, dp_size=4)

        num_heads = 8
        num_kv_heads = 8
        head_dim = 128

        # Single request with q=4, kv=4 on DP rank 0 only
        lens_dict = {
            0: [(1, 4)],
        }

        self.run_test(
            "decode", lens_dict, (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16), mesh, 4
        )

    def test_sparse_ranks_decode_dp_4(self):
        """
        Critical Test: Sparse Ranks in Decode.
        Ranks 0 and 2 are empty, Ranks 1 and 3 have data.
        Verifies that:
        1. Empty ranks (0, 2) don't crash (pipeline deadlock fix).
        2. Non-empty ranks (1, 3) get correct distribution/metadata (sharding fix).
        """
        mesh = set_mesh(tp_size=1, dp_size=4)
        num_heads = 8
        num_kv_heads = 8
        head_dim = 128

        # DP=4: Rank 0=Empty, Rank 1=Data, Rank 2=Empty, Rank 3=Data
        lens_dict = {
            0: [],
            1: [(1, 128), (1, 256)],
            2: [],
            3: [(1, 10), (1, 1024)],
        }

        self.run_test(
            "decode", lens_dict, (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16), mesh, 4
        )

    def test_sparse_ranks_prefill_dp_4(self):
        """
        Test Sparse Ranks in Prefill.
        Ensures global buffer placement logic handles empty ranks correctly.
        """
        mesh = set_mesh(tp_size=1, dp_size=4)
        num_heads = 8
        num_kv_heads = 8
        head_dim = 128

        # DP=4: Rank 0=Data, Rank 1=Empty, Rank 2=Data, Rank 3=Empty
        lens_dict = {
            0: [(64, 64)],
            1: [],
            2: [(128, 128), (32, 32)],
            3: [],
        }

        self.run_test(
            "prefill", lens_dict, (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16), mesh, 4
        )

    def test_heavy_load_unbalanced_decode_dp_4(self):
        """
        Stress Test: Heavy and unbalanced load across DP ranks.
        Tests padding logic stability when one rank dominates the size.
        """
        mesh = set_mesh(tp_size=1, dp_size=4)
        num_heads = 8
        num_kv_heads = 8
        head_dim = 128

        lens_dict = {
            0: [(1, 10)],  # Very small
            1: [(1, 100), (1, 100), (1, 100)],  # Moderate batch
            2: [(1, 1024)],  # Large context
            3: [(1, 10), (1, 20), (1, 30), (1, 40)],  # Large batch size
        }

        self.run_test(
            "decode", lens_dict, (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16), mesh, 4
        )

    def test_sliding_window_prefill_dp_2(self):
        """
        Feature Test: Sliding Window Attention with DP.
        """
        mesh = set_mesh(tp_size=2, dp_size=2)
        num_heads = 8
        num_kv_heads = 8
        head_dim = 128
        window_size = 128

        # KV len > window_size to trigger masking logic
        lens_dict = {
            0: [(256, 256), (130, 130)],
            1: [(10, 10), (512, 512)],
        }

        self.run_test(
            "prefill",
            lens_dict,
            (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16),
            mesh,
            2,
            sliding_window=window_size,
        )

    def test_logit_soft_cap_decode_dp_4(self):
        """
        Feature Test: Logit Soft Cap (Gemma/Grok style) with DP.
        """
        mesh = set_mesh(tp_size=1, dp_size=4)
        num_heads = 8
        num_kv_heads = 8
        head_dim = 128
        soft_cap = 50.0

        lens_dict = {
            0: [(1, 128)],
            1: [(1, 256)],
            2: [(1, 512)],
            3: [(1, 1024)],
        }

        self.run_test(
            "decode",
            lens_dict,
            (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16),
            mesh,
            4,
            logit_cap=soft_cap,
        )

    def test_xai_temperature_decode_dp_4(self):
        """
        Feature Test: XAI Temperature scaling (Grok style).
        """
        mesh = set_mesh(tp_size=1, dp_size=4)
        num_heads = 8
        num_kv_heads = 8
        head_dim = 128
        temp_len = 32

        lens_dict = {
            0: [(1, 10)],  # < temp_len
            1: [(1, 32)],  # == temp_len
            2: [(1, 64)],  # > temp_len
            3: [(1, 128)],
        }

        self.run_test(
            "decode",
            lens_dict,
            (num_heads, head_dim, num_kv_heads, 1, jnp.bfloat16),
            mesh,
            4,
            xai_temperature_len=temp_len,
        )


def compute_dp_reference_attention(
    mode,
    per_dp_infos,
    dp_size,
    per_dp_bs,
    per_dp_token_padding,
    num_heads,
    num_kv_heads,
    head_dim,
    page_size,
    causal=True,
    sliding_window=None,
    logit_cap=None,
    xai_temperature_len=None,
):
    """
    Computes reference attention results using ref_ragged_paged_attention.
    Correctly handles global array shapes and strided placement for DP padding.
    """
    hidden_size = num_heads * head_dim
    is_prefill = mode == "prefill"

    # 1. Initialize Global Output Buffer with correct shape
    if is_prefill:
        # Prefill Mode: Shape is [Total DP Token Capacity, Hidden Size]
        # logic: dp_size * max_tokens_per_dp_padded
        global_ref_outputs = np.zeros(
            (dp_size * per_dp_token_padding, hidden_size), dtype=np.float32
        )
    else:
        # Decode Mode: Shape is [Total DP Batch Capacity, Hidden Size]
        # logic: dp_size * max_bs_per_dp_padded
        global_ref_outputs = np.zeros((dp_size * per_dp_bs, hidden_size), dtype=np.float32)

    for dp_rank in range(dp_size):
        rank_info = per_dp_infos.get(dp_rank)
        # Handle empty ranks (just leave them as zeros in global output)
        if not rank_info or not rank_info["req_infos"]:
            continue

        req_infos = rank_info["req_infos"]

        # --- 2. Prepare Rank-Local Data for Reference Kernel ---
        rank_q_list = []
        rank_k_blocks = []
        rank_v_blocks = []
        rank_seq_lens = []
        rank_cache_loc_list = []

        # Helper to track simulated linear memory for this rank
        current_local_slot = 0

        for req_idx, info in enumerate(req_infos):
            q_data = info["q"]
            k_data = info["k"]
            v_data = info["v"]
            kv_len = info["kv_len"]

            rank_q_list.append(q_data)
            rank_seq_lens.append(kv_len)

            # Simulate Paging (linear local space for reference)
            aligned_kv_len = ((kv_len + page_size - 1) // page_size) * page_size

            # Generate page table for this request
            slots = np.arange(current_local_slot, current_local_slot + aligned_kv_len)
            current_local_slot += aligned_kv_len  # Advance slot pointer

            page_indices = slots // page_size
            page_indices_unique = page_indices[::page_size]

            # Pad page table to a fixed large size (e.g. 4096 or just enough for test)
            # The reference implementation typically expects a ragged or padded table
            padded_pages = np.zeros(4096, dtype=np.int32)
            padded_pages[: len(page_indices_unique)] = page_indices_unique
            rank_cache_loc_list.append(padded_pages)

            # Prepare K/V Pages
            k_padded = np.zeros((aligned_kv_len, num_kv_heads, head_dim), dtype=np.float32)
            v_padded = np.zeros((aligned_kv_len, num_kv_heads, head_dim), dtype=np.float32)
            k_padded[:kv_len] = k_data
            v_padded[:kv_len] = v_data

            rank_k_blocks.append(k_padded.reshape(-1, page_size, num_kv_heads, head_dim))
            rank_v_blocks.append(v_padded.reshape(-1, page_size, num_kv_heads, head_dim))

        # --- 3. Run Reference Attention for this Rank ---
        q_rank = np.concatenate(rank_q_list, axis=0)
        k_pages_rank = np.concatenate(rank_k_blocks, axis=0)
        v_pages_rank = np.concatenate(rank_v_blocks, axis=0)
        seq_lens_rank = jnp.array(rank_seq_lens, dtype=jnp.int32)
        page_table_rank = jnp.array(np.stack(rank_cache_loc_list), dtype=jnp.int32)

        q_lens = [info["q_len"] for info in req_infos]
        cu_q_lens = jnp.array([0] + np.cumsum(q_lens).tolist(), dtype=np.int32)

        rank_out = ref_ragged_paged_attention(
            jnp.array(q_rank),
            jnp.array(k_pages_rank),
            jnp.array(v_pages_rank),
            seq_lens_rank,
            page_table_rank,
            cu_q_lens,
            jnp.array([len(req_infos)], dtype=jnp.int32),
            causal=causal,
            sm_scale=head_dim**-0.5,
            sliding_window=sliding_window,
            soft_cap=logit_cap,
            xai_temperature_len=xai_temperature_len,
        )
        rank_out = jax.block_until_ready(rank_out)
        # Ensure numpy array
        rank_out = np.array(rank_out).reshape(-1, hidden_size)

        # --- 4. Place Results into Global Buffer (The Fix) ---
        if is_prefill:
            # Prefill: Rank output shape is [sum(q_lens), Hidden]
            # Must write to: [Rank_Start, Rank_Start + valid_tokens]
            # Stride is per_dp_token_padding

            offset_start = dp_rank * per_dp_token_padding
            valid_token_count = rank_out.shape[0]

            # Check for overflow (debugging aid)
            if valid_token_count > per_dp_token_padding:
                raise ValueError(
                    f"Rank {dp_rank} output size {valid_token_count} exceeds padding {per_dp_token_padding}"
                )

            global_ref_outputs[offset_start : offset_start + valid_token_count] = rank_out

        else:
            # Decode: Rank output shape is [num_seqs, 1, Hidden] or [num_seqs, Hidden]
            # Must write to: [Rank_Start, Rank_Start + num_seqs]
            # Stride is per_dp_bs

            offset_start = dp_rank * per_dp_bs
            num_seqs = len(req_infos)

            # Flatten [Num_seqs, 1, Hidden] -> [Num_seqs, Hidden]
            rank_out_reshaped = rank_out.reshape(num_seqs, -1)

            if num_seqs > per_dp_bs:
                raise ValueError(
                    f"Rank {dp_rank} batch size {num_seqs} exceeds padding {per_dp_bs}"
                )

            global_ref_outputs[offset_start : offset_start + num_seqs] = rank_out_reshaped

    return global_ref_outputs


if __name__ == "__main__":
    unittest.main()
