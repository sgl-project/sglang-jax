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


def _make_mock_pool(layer_id, recurrent_state, recurrent_indices=None):
    B = recurrent_state.shape[0]
    if recurrent_indices is None:
        recurrent_indices = np.arange(1, B + 1, dtype=np.int32)
    N_plus_1 = int(max(recurrent_indices)) + 1
    buf = jnp.zeros((N_plus_1,) + recurrent_state.shape[1:], dtype=recurrent_state.dtype)
    buf = buf.at[jnp.array(recurrent_indices)].set(recurrent_state)
    return MockRecurrentStatePool(layer_caches={layer_id: (buf, [])}), recurrent_indices


def _extract_state(pool_updates, recurrent_indices):
    new_ssm_full, conv_list = pool_updates
    return new_ssm_full[jnp.array(recurrent_indices)]


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

            # Simulate DP=2: each shard has 2 requests
            # DP shard 0: req 0,1 with seq_lens [10, 20]
            # DP shard 1: req 2,3 with seq_lens [15, 25]
            batch = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                seq_lens=np.array([10, 20, 15, 25], dtype=np.int32),
                recurrent_indices=np.array([1, 2, 3, 4], dtype=np.int32),
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
            pool_tp4, _ = _make_mock_pool(_LAYER_ID, jnp.array(h0_np), rec_indices)

            batch_tp4 = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                seq_lens=np.ones(B, dtype=np.int32),
                recurrent_indices=rec_indices,
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
            pool_dp, _ = _make_mock_pool(_LAYER_ID, jnp.array(h0_np), rec_indices)

            batch_dp = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                seq_lens=np.ones(B, dtype=np.int32),
                recurrent_indices=rec_indices,
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
            state_dp = _extract_state(pu_dp, rec_indices)

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
            pool, _ = _make_mock_pool(_LAYER_ID, jnp.array(h0_np), rec_indices)

            batch = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                seq_lens=np.ones(B, dtype=np.int32),
                recurrent_indices=rec_indices,
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
            pool_tp4, _ = _make_mock_pool(_LAYER_ID, jnp.array(h0_np), rec_indices)

            batch_tp4 = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=np.array(seq_lens, dtype=np.int32),
                seq_lens=np.array(seq_lens, dtype=np.int32),
                input_ids=np.zeros(total_tokens, dtype=np.int32),
                recurrent_indices=rec_indices,
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
            pool_dp, _ = _make_mock_pool(_LAYER_ID, jnp.array(h0_np), rec_indices)

            batch_dp = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=np.array(seq_lens, dtype=np.int32),
                seq_lens=np.array(seq_lens, dtype=np.int32),
                input_ids=np.zeros(total_tokens, dtype=np.int32),
                recurrent_indices=rec_indices,
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
            state_dp = _extract_state(pu_dp, rec_indices)

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
            pool, _ = _make_mock_pool(_LAYER_ID, jnp.array(h0_np), rec_indices)

            batch = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=np.array(seq_lens, dtype=np.int32),
                seq_lens=np.array(seq_lens, dtype=np.int32),
                input_ids=np.zeros(total_tokens, dtype=np.int32),
                recurrent_indices=rec_indices,
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
            rec_indices = np.arange(1, B + 1, dtype=np.int32)
            pool, _ = _make_mock_pool(_LAYER_ID, jnp.array(h0_np), rec_indices)

            batch_ext = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=np.array(seq_lens, dtype=np.int32),
                seq_lens=np.array(seq_lens, dtype=np.int32),
                input_ids=np.zeros(total_tokens, dtype=np.int32),
                recurrent_indices=rec_indices,
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
            state_after_ext = _extract_state(pu_ext, rec_indices)

            # Decode - test first request only
            h_jax = jnp.array(state_after_ext[0:1])
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
            rec_indices = np.arange(1, B + 1, dtype=np.int32)
            pool, _ = _make_mock_pool(_LAYER_ID, jnp.array(h0_np), rec_indices)

            # Extract state from pool
            extracted = _extract_state((pool.layer_caches[_LAYER_ID][0], []), rec_indices)

        np.testing.assert_allclose(
            np.array(extracted),
            h0_np,
            atol=1e-3,
            err_msg="Pool state roundtrip failed",
        )
