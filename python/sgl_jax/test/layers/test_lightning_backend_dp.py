"""Data parallelism tests for LightningAttnBackend (GLA).

Tests DP=2×TP=2 configuration against DP=1×TP=4 reference on TPU v6e-4.
Verifies metadata correctness, decode/extend numerical consistency, state
isolation, and pool roundtrip under data parallelism.

Run with: pytest python/sgl_jax/test/layers/test_gla_backend_dp.py -v
"""

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    MockRecurrentStatePool,
)
from sgl_jax.srt.layers.attention.linear.lightning_backend import LightningAttnBackend
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

try:
    from sgl_jax.srt.kernels.simple_gla.simple_gla import fused_recurrent_simple_gla

    HAS_SIMPLE_GLA = True
except ImportError:
    HAS_SIMPLE_GLA = False

_HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
_N_DEVICES = jax.device_count()

requires_simple_gla = pytest.mark.skipif(
    not HAS_SIMPLE_GLA, reason="simple_gla kernel not available"
)
requires_tpu = pytest.mark.skipif(not _HAS_TPU, reason="DP tests require TPU")
requires_4_devices = pytest.mark.skipif(_N_DEVICES < 4, reason="DP=2×TP=2 requires 4 devices")

_H = 16
_K = 128
_LAYER_ID = 5


def _make_slopes(H, layer_id=5, num_layers=80):
    def _get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    def _build_slope_tensor(n):
        if math.log2(n).is_integer():
            return _get_slopes_power_of_2(n)
        closest = 2 ** math.floor(math.log2(n))
        return (
            _get_slopes_power_of_2(closest) + _build_slope_tensor(2 * closest)[0::2][: n - closest]
        )

    base = np.array(_build_slope_tensor(H), dtype=np.float32)
    slope = -base * (1 - (layer_id - 1) / (num_layers - 1) + 1e-5)
    return slope


def _make_mock_pool(layer_id, recurrent_state, recurrent_indices=None, dp_size=1, mesh=None):
    """Create a mock recurrent state pool with DP-local buffers.

    Args:
        layer_id: Layer ID for the cache
        recurrent_state: Initial state array [B, H, K, K]
        recurrent_indices: LOCAL indices for each request (default: 1..B_per_rank)
        dp_size: Data parallelism size
        mesh: JAX mesh for sharding the buffer

    Returns:
        (MockRecurrentStatePool, recurrent_indices)

    Note:
        Mimics production RecurrentStatePool behavior:
        - Buffer always has P("data", "tensor", None, None) sharding
        - For DP: each rank has local buffer with local indices [1, slots_per_rank]
        - For TP-only: "data" axis has size 1, indices are [1, 2, ..., B]
    """
    B = recurrent_state.shape[0]
    if recurrent_indices is None:
        # Default: local indices [1, 2, ..., B_per_rank] for each rank
        B_per_rank = B // dp_size if dp_size > 1 else B
        recurrent_indices = np.arange(1, B_per_rank + 1, dtype=np.int32)

    # For DP, each rank gets a local buffer of size (slots_per_rank + 1)
    if dp_size > 1:
        B_per_rank = B // dp_size
        # Create DP-local buffer: each rank has (B_per_rank + 1) slots
        local_buf_size = B_per_rank + 1
        # Stack local buffers for all ranks
        buf_shape = (dp_size * local_buf_size,) + recurrent_state.shape[1:]
        buf = jnp.zeros(buf_shape, dtype=recurrent_state.dtype)

        # Fill each rank's local buffer with its portion of recurrent_state
        for rank in range(dp_size):
            rank_start = rank * B_per_rank
            rank_end = rank_start + B_per_rank
            rank_state = recurrent_state[rank_start:rank_end]

            # Place in local buffer at indices [1, 2, ..., B_per_rank]
            local_buf_start = rank * local_buf_size
            for i, idx in enumerate(recurrent_indices):
                buf = buf.at[local_buf_start + idx].set(rank_state[i])
    else:
        # TP-only: single buffer with "data" axis size 1
        # Buffer size: (B + 1) to match production (slot 0 is dummy)
        N_plus_1 = int(max(recurrent_indices)) + 1
        buf = jnp.zeros((N_plus_1,) + recurrent_state.shape[1:], dtype=recurrent_state.dtype)
        buf = buf.at[jnp.array(recurrent_indices)].set(recurrent_state)

    # Always use P("data", "tensor", None, None) to match production
    buf = jax.device_put(buf, NamedSharding(mesh, P("data", "tensor", None, None)))

    return MockRecurrentStatePool(layer_caches={layer_id: (buf, [])}), recurrent_indices


def _extract_state(pool_updates, recurrent_indices, dp_size=1):
    """Extract recurrent state from pool updates.

    For DP-local buffers, extracts from each rank's local buffer using local indices.

    Args:
        pool_updates: (new_ssm_full, conv_list) tuple from backend
        recurrent_indices: LOCAL indices used for each rank
        dp_size: Data parallelism size

    Returns:
        Extracted states with shape [B, H, K, K] where B is total batch size
    """
    new_ssm_full, conv_list = pool_updates
    buffer_sharding = new_ssm_full.sharding

    if not isinstance(buffer_sharding, NamedSharding):
        # Fallback for non-NamedSharding
        indices = jnp.array(recurrent_indices)
        return new_ssm_full[indices]

    mesh = buffer_sharding.mesh

    if dp_size == 1:
        # TP-only: direct indexing with local indices
        indices = jnp.array(recurrent_indices)
        return new_ssm_full.at[indices].get(
            out_sharding=NamedSharding(mesh, P("data", "tensor", None, None))
        )
    else:
        # DP: use shard_map to extract from each rank's local buffer
        indices = jnp.array(recurrent_indices)

        def _gather_local(buf, idx):
            # Each rank extracts from its local buffer using local indices
            return buf[idx]

        return jax.shard_map(
            _gather_local,
            mesh=mesh,
            in_specs=(
                P("data", "tensor", None, None),  # buffer
                P(),  # indices are replicated (same local indices on each rank)
            ),
            out_specs=P("data", "tensor", None, None),
            check_vma=False,
        )(new_ssm_full, indices)


def _make_fake_layer(layer_id=_LAYER_ID, H=_H):
    return SimpleNamespace(layer_id=layer_id, num_heads=H, head_dim=_K)


# ===========================================================================
# TestDPMetadata
# ===========================================================================


@requires_simple_gla
@requires_tpu
@requires_4_devices
class TestDPMetadata:
    def test_dp_metadata_per_shard(self):
        """Each DP shard's cu_q_lens independently correct."""
        mesh_dp = create_device_mesh(ici_parallelism=[2, 2], dcn_parallelism=[1, 1])

        with jax.set_mesh(mesh_dp):
            backend = LightningAttnBackend(
                mesh=mesh_dp,
                linear_recurrent_layer_ids=[_LAYER_ID],
                num_hidden_layers=80,
                num_heads=_H,
            )

            # Simulate DP=2: each shard has 2 requests with LOCAL indices [1, 2]
            # DP shard 0: req 0,1 with seq_lens [10, 20], local indices [1, 2]
            # DP shard 1: req 2,3 with seq_lens [15, 25], local indices [1, 2]
            batch = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                seq_lens=np.array([10, 20, 15, 25], dtype=np.int32),
                recurrent_indices=np.array([1, 2, 1, 2], dtype=np.int32),  # LOCAL indices per rank
                has_initial_state=np.ones(4, dtype=np.bool_),
                dp_size=2,
                per_dp_bs_size=2,
            )

            metadata = backend.get_forward_metadata(batch)

            # cu_q_lens should be 1D array of length dp_size * (per_dp_bs_size+1) = 2 * 3 = 6
            # Logically [dp_size, per_dp_bs_size+1] = [2, 3] raveled
            # DP shard 0: [0, 1, 2] (2 requests, decode → each contributes 1 token)
            # DP shard 1: [0, 1, 2]
            cu_q_lens_np = np.array(metadata.cu_q_lens)

            # Should be 1D with length 6
            assert cu_q_lens_np.shape == (6,), f"Expected shape (6,), got {cu_q_lens_np.shape}"

            # Reshape to [2, 3] to verify per-shard values
            cu_q_2d = cu_q_lens_np.reshape(2, 3)

            # Each shard: decode mode → 1 token per request
            expected_shard0 = np.array([0, 1, 2], dtype=np.int32)
            expected_shard1 = np.array([0, 1, 2], dtype=np.int32)

            np.testing.assert_array_equal(
                cu_q_2d[0],
                expected_shard0,
                err_msg="DP shard 0 cu_q_lens incorrect",
            )
            np.testing.assert_array_equal(
                cu_q_2d[1],
                expected_shard1,
                err_msg="DP shard 1 cu_q_lens incorrect",
            )


# ===========================================================================
# TestDPDecode
# ===========================================================================


@requires_simple_gla
@requires_tpu
@requires_4_devices
class TestDPDecode:
    def test_decode_dp2_tp2_matches_tp4(self):
        """DP=2×TP=2 decode output/state matches DP=1×TP=4."""
        B, H, K = 4, _H, _K
        rng = np.random.default_rng(5001)

        q_np = rng.standard_normal((B, 1, H, K)).astype(np.float32) * 0.1
        k_np = rng.standard_normal((B, 1, H, K)).astype(np.float32) * 0.1
        v_np = rng.standard_normal((B, 1, H, K)).astype(np.float32) * 0.1
        h0_np = rng.standard_normal((B, H, K, K)).astype(np.float32) * 0.01

        # Reference: DP=1×TP=4
        mesh_tp4 = create_device_mesh(ici_parallelism=[1, 4], dcn_parallelism=[1, 1])
        with jax.set_mesh(mesh_tp4):
            backend_tp4 = LightningAttnBackend(
                mesh=mesh_tp4,
                linear_recurrent_layer_ids=[_LAYER_ID],
                num_hidden_layers=80,
                num_heads=H,
            )
            rec_indices = np.arange(1, B + 1, dtype=np.int32)
            pool_tp4, _ = _make_mock_pool(_LAYER_ID, jnp.array(h0_np), rec_indices, mesh=mesh_tp4)

            batch_tp4 = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                seq_lens=np.ones(B, dtype=np.int32),
                recurrent_indices=rec_indices,
                has_initial_state=np.ones(B, dtype=np.bool_),
                dp_size=1,
                per_dp_bs_size=B,
            )
            metadata_tp4 = backend_tp4.get_forward_metadata(batch_tp4)
            backend_tp4.forward_metadata = metadata_tp4

            layer = _make_fake_layer()
            fb = SimpleNamespace(forward_mode=ForwardMode.DECODE)

            q_tp4 = jnp.array(q_np).reshape(B, H, K)
            k_tp4 = jnp.array(k_np).reshape(B, H, K)
            v_tp4 = jnp.array(v_np).reshape(B, H, K)

            out_tp4, pu_tp4 = backend_tp4(
                q_tp4, k_tp4, v_tp4, layer=layer, forward_batch=fb, recurrent_state_pool=pool_tp4
            )
            state_tp4 = _extract_state(pu_tp4, rec_indices)

        # Test: DP=2×TP=2
        mesh_dp = create_device_mesh(ici_parallelism=[2, 2], dcn_parallelism=[1, 1])
        with jax.set_mesh(mesh_dp):
            backend_dp = LightningAttnBackend(
                mesh=mesh_dp,
                linear_recurrent_layer_ids=[_LAYER_ID],
                num_hidden_layers=80,
                num_heads=H,
            )
            # For DP: use LOCAL indices [1, 2] per rank
            local_indices = np.arange(1, B // 2 + 1, dtype=np.int32)
            pool_dp, _ = _make_mock_pool(
                _LAYER_ID, jnp.array(h0_np), local_indices, dp_size=2, mesh=mesh_dp
            )

            # batch.recurrent_indices: repeated local indices [1, 2, 1, 2]
            batch_rec_indices = np.tile(local_indices, 2)
            batch_dp = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                seq_lens=np.ones(B, dtype=np.int32),
                recurrent_indices=batch_rec_indices,
                has_initial_state=np.ones(B, dtype=np.bool_),
                dp_size=2,
                per_dp_bs_size=B // 2,
            )
            metadata_dp = backend_dp.get_forward_metadata(batch_dp)
            backend_dp.forward_metadata = metadata_dp

            q_dp = jnp.array(q_np).reshape(B, H, K)
            k_dp = jnp.array(k_np).reshape(B, H, K)
            v_dp = jnp.array(v_np).reshape(B, H, K)

            out_dp, pu_dp = backend_dp(
                q_dp, k_dp, v_dp, layer=layer, forward_batch=fb, recurrent_state_pool=pool_dp
            )
            state_dp = _extract_state(pu_dp, local_indices, dp_size=2)

        np.testing.assert_allclose(
            np.array(out_dp),
            np.array(out_tp4),
            atol=1e-3,
            err_msg="DP=2×TP=2 decode output != DP=1×TP=4",
        )
        np.testing.assert_allclose(
            np.array(state_dp),
            np.array(state_tp4),
            atol=1e-3,
            err_msg="DP=2×TP=2 decode state != DP=1×TP=4",
        )

    def test_decode_dp_state_isolation(self):
        """No cross-DP state leakage (distinct h0 per shard)."""
        B, H, K = 4, _H, _K
        rng = np.random.default_rng(5002)

        q_np = rng.standard_normal((B, 1, H, K)).astype(np.float32) * 0.1
        k_np = rng.standard_normal((B, 1, H, K)).astype(np.float32) * 0.1
        v_np = rng.standard_normal((B, 1, H, K)).astype(np.float32) * 0.1

        # Different initial states per DP shard
        h0_shard0 = rng.standard_normal((2, H, K, K)).astype(np.float32) * 0.01
        h0_shard1 = rng.standard_normal((2, H, K, K)).astype(np.float32) * 0.02
        h0_np = np.concatenate([h0_shard0, h0_shard1], axis=0)

        mesh_dp = create_device_mesh(ici_parallelism=[2, 2], dcn_parallelism=[1, 1])
        with jax.set_mesh(mesh_dp):
            backend = LightningAttnBackend(
                mesh=mesh_dp,
                linear_recurrent_layer_ids=[_LAYER_ID],
                num_hidden_layers=80,
                num_heads=H,
            )
            rec_indices = np.arange(1, B + 1, dtype=np.int32)
            pool, _ = _make_mock_pool(
                _LAYER_ID, jnp.array(h0_np), rec_indices, dp_size=2, mesh=mesh_dp
            )

            batch = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                seq_lens=np.ones(B, dtype=np.int32),
                recurrent_indices=rec_indices,
                has_initial_state=np.ones(B, dtype=np.bool_),
                dp_size=2,
                per_dp_bs_size=B // 2,
            )
            metadata = backend.get_forward_metadata(batch)
            backend.forward_metadata = metadata

            layer = _make_fake_layer()
            fb = SimpleNamespace(forward_mode=ForwardMode.DECODE)

            q = jnp.array(q_np).reshape(B, H, K)
            k = jnp.array(k_np).reshape(B, H, K)
            v = jnp.array(v_np).reshape(B, H, K)

            out, pu = backend(q, k, v, layer=layer, forward_batch=fb, recurrent_state_pool=pool)
            state = _extract_state(pu, rec_indices)

        # Verify each request's output reflects its distinct initial state
        # Requests 0,1 (shard 0) should differ from requests 2,3 (shard 1)
        state_np = np.array(state)

        # State magnitude should reflect initial state scale
        shard0_mag = np.abs(state_np[:2]).mean()
        shard1_mag = np.abs(state_np[2:]).mean()

        # Shard 1 had 2x larger h0 → should have noticeably different magnitude
        assert not np.allclose(
            shard0_mag, shard1_mag, rtol=0.1
        ), "DP shards should have distinct state magnitudes"


# ===========================================================================
# TestDPExtend
# ===========================================================================


@requires_simple_gla
@requires_tpu
@requires_4_devices
class TestDPExtend:
    def test_extend_dp2_tp2_matches_tp4(self):
        """DP=2×TP=2 extend output/state matches DP=1×TP=4."""
        seq_lens = [128, 256]
        H, K = _H, _K
        rng = np.random.default_rng(6001)

        # Pack for backend (varlen format)
        total_tokens = sum(seq_lens)
        q_packed = rng.standard_normal((total_tokens, H, K)).astype(np.float32) * 0.1
        k_packed = rng.standard_normal((total_tokens, H, K)).astype(np.float32) * 0.1
        v_packed = rng.standard_normal((total_tokens, H, K)).astype(np.float32) * 0.1

        B = len(seq_lens)
        h0_np = np.zeros((B, H, K, K), dtype=np.float32)

        # Reference: DP=1×TP=4
        mesh_tp4 = create_device_mesh(ici_parallelism=[1, 4], dcn_parallelism=[1, 1])
        with jax.set_mesh(mesh_tp4):
            backend_tp4 = LightningAttnBackend(
                mesh=mesh_tp4,
                linear_recurrent_layer_ids=[_LAYER_ID],
                num_hidden_layers=80,
                num_heads=H,
            )
            rec_indices = np.arange(1, B + 1, dtype=np.int32)
            pool_tp4, _ = _make_mock_pool(_LAYER_ID, jnp.array(h0_np), rec_indices, mesh=mesh_tp4)

            batch_tp4 = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=np.array(seq_lens, dtype=np.int32),
                seq_lens=np.array(seq_lens, dtype=np.int32),
                input_ids=np.zeros(total_tokens, dtype=np.int32),
                recurrent_indices=rec_indices,
                has_initial_state=np.ones(B, dtype=np.bool_),
                dp_size=1,
                per_dp_bs_size=B,
            )
            metadata_tp4 = backend_tp4.get_forward_metadata(batch_tp4)
            backend_tp4.forward_metadata = metadata_tp4

            layer = _make_fake_layer()
            fb = SimpleNamespace(forward_mode=ForwardMode.EXTEND)

            out_tp4, pu_tp4 = backend_tp4(
                jnp.array(q_packed),
                jnp.array(k_packed),
                jnp.array(v_packed),
                layer=layer,
                forward_batch=fb,
                recurrent_state_pool=pool_tp4,
            )
            state_tp4 = _extract_state(pu_tp4, rec_indices)

        # Test: DP=2×TP=2
        mesh_dp = create_device_mesh(ici_parallelism=[2, 2], dcn_parallelism=[1, 1])
        with jax.set_mesh(mesh_dp):
            backend_dp = LightningAttnBackend(
                mesh=mesh_dp,
                linear_recurrent_layer_ids=[_LAYER_ID],
                num_hidden_layers=80,
                num_heads=H,
            )
            # For DP: use LOCAL indices [1] per rank (each rank has 1 request)
            local_indices = np.arange(1, B // 2 + 1, dtype=np.int32)
            pool_dp, _ = _make_mock_pool(
                _LAYER_ID, jnp.array(h0_np), local_indices, dp_size=2, mesh=mesh_dp
            )

            # batch.recurrent_indices: repeated local indices [1, 1]
            batch_rec_indices = np.tile(local_indices, 2)
            batch_dp = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=np.array(seq_lens, dtype=np.int32),
                seq_lens=np.array(seq_lens, dtype=np.int32),
                input_ids=np.zeros(total_tokens, dtype=np.int32),
                recurrent_indices=batch_rec_indices,
                has_initial_state=np.ones(B, dtype=np.bool_),
                dp_size=2,
                per_dp_bs_size=B // 2,
            )
            metadata_dp = backend_dp.get_forward_metadata(batch_dp)
            backend_dp.forward_metadata = metadata_dp

            # 关键：按照 FlashAttention 的方式准备数据
            # 数据布局：每个 DP rank 的 tokens 连续排列
            # DP rank 0: request 0 的所有 tokens [0:128]
            # DP rank 1: request 1 的所有 tokens [0:256] (局部索引)
            #
            # 但是全局数组需要 padding 到统一大小
            # per_dp_token_padding = max(seq_lens) = 256
            per_dp_token_padding = max(seq_lens)
            global_token_size = per_dp_token_padding * 2  # 512

            # 构造全局数组，按 DP rank 分段
            q_global = np.zeros((global_token_size, H, K), dtype=np.float32)
            k_global = np.zeros((global_token_size, H, K), dtype=np.float32)
            v_global = np.zeros((global_token_size, H, K), dtype=np.float32)

            # DP rank 0: tokens [0:128] 放在全局 [0:128]
            q_global[0 : seq_lens[0]] = q_packed[: seq_lens[0]]
            k_global[0 : seq_lens[0]] = k_packed[: seq_lens[0]]
            v_global[0 : seq_lens[0]] = v_packed[: seq_lens[0]]

            # DP rank 1: tokens [128:384] 放在全局 [256:512]
            q_global[per_dp_token_padding : per_dp_token_padding + seq_lens[1]] = q_packed[
                seq_lens[0] :
            ]
            k_global[per_dp_token_padding : per_dp_token_padding + seq_lens[1]] = k_packed[
                seq_lens[0] :
            ]
            v_global[per_dp_token_padding : per_dp_token_padding + seq_lens[1]] = v_packed[
                seq_lens[0] :
            ]

            # 使用 device_put 配合 P("data", "tensor", None) 自动切分
            from jax.sharding import NamedSharding

            sharding_spec = NamedSharding(mesh_dp, P("data", "tensor", None))

            q_sharded = jax.device_put(jnp.array(q_global), sharding_spec)
            k_sharded = jax.device_put(jnp.array(k_global), sharding_spec)
            v_sharded = jax.device_put(jnp.array(v_global), sharding_spec)

            out_dp, pu_dp = backend_dp(
                q_sharded,
                k_sharded,
                v_sharded,
                layer=layer,
                forward_batch=fb,
                recurrent_state_pool=pool_dp,
            )
            state_dp = _extract_state(pu_dp, local_indices, dp_size=2)

            # 提取有效的 tokens（去除 padding）
            # 先转换为 numpy array，再切片
            out_dp_np = np.array(out_dp)
            # DP rank 0: [0:128]
            # DP rank 1: [256:512] 中的 [256:256+256]
            out_dp_valid = np.concatenate(
                [
                    out_dp_np[0 : seq_lens[0]],
                    out_dp_np[per_dp_token_padding : per_dp_token_padding + seq_lens[1]],
                ],
                axis=0,
            )

        np.testing.assert_allclose(
            out_dp_valid,
            np.array(out_tp4),
            atol=1e-2,
            rtol=1e-2,
            err_msg="DP=2×TP=2 extend output != DP=1×TP=4",
        )
        np.testing.assert_allclose(
            np.array(state_dp),
            np.array(state_tp4),
            atol=1e-2,
            rtol=1e-2,
            err_msg="DP=2×TP=2 extend state != DP=1×TP=4",
        )

    def test_extend_dp_multi_request(self):
        """Multi-request extend under DP."""
        # 4 requests: 2 per DP shard
        seq_lens = [64, 128, 96, 192]
        H, K = _H, _K
        rng = np.random.default_rng(6002)

        q_list = [rng.standard_normal((1, sl, H, K)).astype(np.float32) * 0.1 for sl in seq_lens]
        k_list = [rng.standard_normal((1, sl, H, K)).astype(np.float32) * 0.1 for sl in seq_lens]
        v_list = [rng.standard_normal((1, sl, H, K)).astype(np.float32) * 0.1 for sl in seq_lens]

        total_tokens = sum(seq_lens)
        q_packed = np.zeros((total_tokens, H, K), dtype=np.float32)
        k_packed = np.zeros((total_tokens, H, K), dtype=np.float32)
        v_packed = np.zeros((total_tokens, H, K), dtype=np.float32)
        offset = 0
        for i, sl in enumerate(seq_lens):
            q_packed[offset : offset + sl] = q_list[i][0]
            k_packed[offset : offset + sl] = k_list[i][0]
            v_packed[offset : offset + sl] = v_list[i][0]
            offset += sl

        B = len(seq_lens)
        h0_np = np.zeros((B, H, K, K), dtype=np.float32)

        mesh_dp = create_device_mesh(ici_parallelism=[2, 2], dcn_parallelism=[1, 1])
        with jax.set_mesh(mesh_dp):
            backend = LightningAttnBackend(
                mesh=mesh_dp,
                linear_recurrent_layer_ids=[_LAYER_ID],
                num_hidden_layers=80,
                num_heads=H,
            )
            rec_indices = np.arange(1, B + 1, dtype=np.int32)
            pool, _ = _make_mock_pool(
                _LAYER_ID, jnp.array(h0_np), rec_indices, dp_size=2, mesh=mesh_dp
            )

            batch = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=np.array(seq_lens, dtype=np.int32),
                seq_lens=np.array(seq_lens, dtype=np.int32),
                input_ids=np.zeros(total_tokens, dtype=np.int32),
                recurrent_indices=rec_indices,
                has_initial_state=np.ones(B, dtype=np.bool_),
                dp_size=2,
                per_dp_bs_size=B // 2,
            )
            metadata = backend.get_forward_metadata(batch)
            backend.forward_metadata = metadata

            layer = _make_fake_layer()
            fb = SimpleNamespace(forward_mode=ForwardMode.EXTEND)

            out, pu = backend(
                jnp.array(q_packed),
                jnp.array(k_packed),
                jnp.array(v_packed),
                layer=layer,
                forward_batch=fb,
                recurrent_state_pool=pool,
            )
            state = _extract_state(pu, rec_indices)

        # Basic sanity: output shape matches total tokens
        assert out.shape[0] == total_tokens, f"Expected {total_tokens} tokens, got {out.shape[0]}"
        # State shape: [B, H, K, K]
        assert state.shape == (
            B,
            H,
            K,
            K,
        ), f"Expected state shape ({B},{H},{K},{K}), got {state.shape}"


# ===========================================================================
# TestDPEndToEnd
# ===========================================================================


@requires_simple_gla
@requires_tpu
@requires_4_devices
class TestDPEndToEnd:
    def test_dp_extend_then_decode(self):
        """Full extend→decode flow under DP."""
        seq_lens = [256, 512]  # 2 requests for DP=2
        decode_steps = 16
        H, K = _H, _K
        rng = np.random.default_rng(7001)
        g_gamma = _make_slopes(H)

        # Two requests
        q_ext_list = [
            rng.standard_normal((1, sl, H, K)).astype(np.float32) * 0.1 for sl in seq_lens
        ]
        k_ext_list = [
            rng.standard_normal((1, sl, H, K)).astype(np.float32) * 0.1 for sl in seq_lens
        ]
        v_ext_list = [
            rng.standard_normal((1, sl, H, K)).astype(np.float32) * 0.1 for sl in seq_lens
        ]

        # Pack for varlen format
        total_tokens = sum(seq_lens)
        q_ext_packed = np.zeros((total_tokens, H, K), dtype=np.float32)
        k_ext_packed = np.zeros((total_tokens, H, K), dtype=np.float32)
        v_ext_packed = np.zeros((total_tokens, H, K), dtype=np.float32)
        offset = 0
        for i, sl in enumerate(seq_lens):
            q_ext_packed[offset : offset + sl] = q_ext_list[i][0]
            k_ext_packed[offset : offset + sl] = k_ext_list[i][0]
            v_ext_packed[offset : offset + sl] = v_ext_list[i][0]
            offset += sl

        B = len(seq_lens)
        h0_np = np.zeros((B, H, K, K), dtype=np.float32)

        mesh_dp = create_device_mesh(ici_parallelism=[2, 2], dcn_parallelism=[1, 1])
        with jax.set_mesh(mesh_dp):
            backend = LightningAttnBackend(
                mesh=mesh_dp,
                linear_recurrent_layer_ids=[_LAYER_ID],
                num_hidden_layers=80,
                num_heads=H,
            )

            # Extend
            local_indices = np.arange(1, B // 2 + 1, dtype=np.int32)
            pool, _ = _make_mock_pool(
                _LAYER_ID, jnp.array(h0_np), local_indices, dp_size=2, mesh=mesh_dp
            )

            batch_rec_indices = np.tile(local_indices, 2)
            batch_ext = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=np.array(seq_lens, dtype=np.int32),
                seq_lens=np.array(seq_lens, dtype=np.int32),
                input_ids=np.zeros(total_tokens, dtype=np.int32),
                recurrent_indices=batch_rec_indices,
                has_initial_state=np.ones(B, dtype=np.bool_),
                dp_size=2,
                per_dp_bs_size=B // 2,
            )
            metadata_ext = backend.get_forward_metadata(batch_ext)
            backend.forward_metadata = metadata_ext

            layer = _make_fake_layer()
            fb_ext = SimpleNamespace(forward_mode=ForwardMode.EXTEND)

            _, pu_ext = backend(
                jnp.array(q_ext_packed),
                jnp.array(k_ext_packed),
                jnp.array(v_ext_packed),
                layer=layer,
                forward_batch=fb_ext,
                recurrent_state_pool=pool,
            )
            state_after_ext = _extract_state(pu_ext, local_indices, dp_size=2)

            # Decode - test first request only
            # Convert to numpy first to avoid sharding issues with slicing
            state_after_ext_np = np.array(state_after_ext)
            h_jax = jnp.array(state_after_ext_np[0:1])
            for step in range(decode_steps):
                q_d = rng.standard_normal((1, 1, H, K)).astype(np.float32) * 0.1
                k_d = rng.standard_normal((1, 1, H, K)).astype(np.float32) * 0.1
                v_d = rng.standard_normal((1, 1, H, K)).astype(np.float32) * 0.1

                _, h_jax = fused_recurrent_simple_gla(
                    jnp.array(q_d),
                    jnp.array(k_d),
                    jnp.array(v_d),
                    g_gamma=jnp.array(g_gamma),
                    initial_state=h_jax,
                    output_final_state=True,
                    scale=None,
                )

        # Sanity: state shape preserved
        assert h_jax.shape == (
            1,
            H,
            K,
            K,
        ), f"Expected (1, {H}, {K}, {K}), got {h_jax.shape}"

    def test_dp_pool_state_roundtrip(self):
        """Each DP shard's pool state roundtrip."""
        B, H, K = 4, _H, _K
        rng = np.random.default_rng(7002)

        h0_np = rng.standard_normal((B, H, K, K)).astype(np.float32) * 0.01

        mesh_dp = create_device_mesh(ici_parallelism=[2, 2], dcn_parallelism=[1, 1])
        with jax.set_mesh(mesh_dp):
            local_indices = np.arange(1, B // 2 + 1, dtype=np.int32)
            pool, _ = _make_mock_pool(
                _LAYER_ID, jnp.array(h0_np), local_indices, dp_size=2, mesh=mesh_dp
            )

            # Extract state from pool
            extracted = _extract_state(
                (pool.layer_caches[_LAYER_ID][0], []), local_indices, dp_size=2
            )

        np.testing.assert_allclose(
            np.array(extracted),
            h0_np,
            atol=1e-3,
            err_msg="Pool state roundtrip failed",
        )
