"""Backend unit tests for LightningAttnBackend (GLA).

Tests hybrid routing, factory wire-up, and fail-fast error messages.

Run with: pytest python/sgl_jax/test/layers/test_gla_backend.py -v
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    LinearRecurrentAttnBackend,
    MockRecurrentStatePool,
    attn_backend_wrapper,
)
from sgl_jax.srt.layers.attention.linear.lightning_backend import LightningAttnBackend
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

_H = 4
_K = 128


class TestHybridIntegration:
    """Test HybridLinearAttnBackend routing to LightningAttnBackend."""

    def test_dispatch_routes_to_lightning(self):
        """HybridLinearAttnBackend routes non-full-attn layers to LightningAttnBackend."""
        try:
            from sgl_jax.srt.layers.attention.flashattention_backend import (
                FlashAttention,
            )
        except (ImportError, ModuleNotFoundError):
            pytest.skip("FlashAttention import chain requires full sglang install")

        with jax.set_mesh(mesh):
            full_backend = FlashAttention(
                num_attn_heads=_H,
                num_kv_heads=_H,
                head_dim=_K,
                page_size=1,
                mesh=mesh,
            )
            lightning = LightningAttnBackend(mesh=mesh)
            hybrid = HybridLinearAttnBackend(full_backend, lightning, full_attn_layers=[0, 1])

        assert 5 not in hybrid.full_attn_layers
        assert isinstance(hybrid.linear_attn_backend, LightningAttnBackend)

    def test_attn_backend_wrapper_lightning_path(self):
        """attn_backend_wrapper factory builds hybrid stack from runner.lightning_config."""
        try:
            from sgl_jax.srt.layers.attention.flashattention_backend import (
                FlashAttention,
            )
        except (ImportError, ModuleNotFoundError):
            pytest.skip("FlashAttention import chain requires full sglang install")

        # Mock runner with lightning_config
        mock_runner = SimpleNamespace(
            mesh=mesh,
            linear_recurrent_config=SimpleNamespace(
                full_attention_layer_ids=[0, 1, 2],
            ),
            lightning_config=SimpleNamespace(
                linear_layer_ids=[3, 4, 5, 6],
                num_hidden_layers=10,
                num_attention_heads=_H,
            ),
            kimi_linear_config=None,
        )

        with jax.set_mesh(mesh):
            full_backend = FlashAttention(
                num_attn_heads=_H,
                num_kv_heads=_H,
                head_dim=_K,
                page_size=1,
                mesh=mesh,
            )

            hybrid = attn_backend_wrapper(mock_runner, full_backend)

        # Check hybrid structure
        assert isinstance(hybrid, HybridLinearAttnBackend)
        assert isinstance(hybrid.linear_attn_backend, LightningAttnBackend)
        assert hybrid.full_attn_layers == frozenset([0, 1, 2])

        # Check tp_slope populated with one entry per Lightning layer
        lightning_backend = hybrid.linear_attn_backend
        assert len(lightning_backend.tp_slope) == 4
        for layer_id in [3, 4, 5, 6]:
            assert layer_id in lightning_backend.tp_slope
            assert lightning_backend.tp_slope[layer_id].shape == (_H,)


class TestTpSlopeFailFast:
    """Test that missing tp_slope[layer_id] raises contextual KeyError."""

    def test_keyerror_message_lists_registered_ids(self):
        """Missing tp_slope[layer_id] raises KeyError listing registered ids."""
        import jax.numpy as jnp
        import numpy as np

        from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
            MockRecurrentStatePool,
        )
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

        # Create backend with tp_slope for layers [1, 2, 3]
        backend = LightningAttnBackend(
            mesh=mesh,
            linear_recurrent_layer_ids=[1, 2, 3],
            num_hidden_layers=10,
            num_heads=_H,
        )

        # Try to call with layer_id=5 (not registered)
        layer_id = 5
        layer = SimpleNamespace(layer_id=layer_id, mesh=mesh, num_heads=_H, head_dim=_K)

        B = 1
        h0 = jnp.zeros((B, _H, _K, _K), dtype=jnp.float32)
        rec_indices = np.arange(1, B + 1, dtype=np.int32)
        N_plus_1 = int(max(rec_indices)) + 1
        buf = jnp.zeros((N_plus_1,) + h0.shape[1:], dtype=h0.dtype)
        buf = buf.at[jnp.array(rec_indices)].set(h0)
        pool = MockRecurrentStatePool(layer_caches={layer_id: (buf, [])})

        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            seq_lens=np.ones(B, dtype=np.int32),
            recurrent_indices=rec_indices,
            dp_size=1,
            per_dp_bs_size=B,
        )

        with jax.set_mesh(mesh):
            metadata = backend.get_forward_metadata(batch)
            backend.forward_metadata = metadata

            q = jnp.zeros((B, _H, _K), dtype=jnp.bfloat16)
            k = jnp.zeros((B, _H, _K), dtype=jnp.bfloat16)
            v = jnp.zeros((B, _H, _K), dtype=jnp.bfloat16)
            fb = SimpleNamespace(forward_mode=ForwardMode.DECODE)

            with pytest.raises(KeyError) as exc_info:
                backend(q, k, v, layer=layer, forward_batch=fb, recurrent_state_pool=pool)

            # Check error message lists registered ids
            error_msg = str(exc_info.value)
            assert "layer_id=5" in error_msg
            assert "registered ids" in error_msg or "[1, 2, 3]" in error_msg
            assert "attn_backend_wrapper" in error_msg


class TestDPShardingContracts:
    """Unit-level DP contracts that do not require TPU kernels."""

    def test_linear_metadata_dp_uses_data_sharding(self, monkeypatch):
        """DP metadata is laid out as per-rank sections and sharded on data."""
        from sgl_jax.srt.layers.attention import (
            hybrid_linear_attn_backend as hybrid_mod,
        )

        captured = {}

        def fake_device_array(data, sharding=None):
            captured["sharding"] = sharding
            return data

        monkeypatch.setattr(hybrid_mod, "device_array", fake_device_array)
        monkeypatch.setattr(hybrid_mod, "NamedSharding", lambda mesh, spec: spec)

        backend = LinearRecurrentAttnBackend(mesh=SimpleNamespace(shape={"data": 2}))
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=np.array([2, 3, 7, 0], dtype=np.int32),
            seq_lens=np.array([2, 3, 7, 0], dtype=np.int32),
            recurrent_indices=np.array([1, 2, 1, 0], dtype=np.int32),
            dp_size=2,
            per_dp_bs_size=2,
        )

        metadata = backend.get_forward_metadata(batch)

        assert captured["sharding"] == P("data")
        np.testing.assert_array_equal(
            metadata.cu_q_lens,
            np.array([0, 2, 5, 0, 7, 7], dtype=np.int32),
        )
        np.testing.assert_array_equal(metadata.recurrent_indices, batch.recurrent_indices)

    def test_extend_dp_shard_map_uses_local_token_and_state_specs(self, monkeypatch):
        """DP extend must shard q/k/v, state buffer, indices, and cu_lens together."""
        from sgl_jax.srt.layers.attention.linear import (
            lightning_backend as lightning_mod,
        )

        calls = []

        def fake_shard_map(fn, *, mesh, in_specs, out_specs, check_vma=False):
            calls.append((in_specs, out_specs))

            def runner(*args):
                # Note: fn, mesh, check_vma are captured from outer scope, not deleted
                if in_specs == (P("data", "tensor", None, None), P("data")):
                    buf, indices = args
                    return jnp.zeros((indices.shape[0],) + buf.shape[1:], dtype=buf.dtype)
                if in_specs == (
                    P("data", "tensor", None, None),
                    P("data"),
                    P("data", "tensor", None, None),
                ):
                    return jnp.zeros_like(args[0])
                return jnp.zeros_like(args[0]), jnp.zeros_like(args[4])

            return runner

        monkeypatch.setattr(lightning_mod, "NamedSharding", lambda mesh, spec: spec)
        monkeypatch.setattr(lightning_mod.jax.sharding, "reshard", lambda value, sharding: value)
        monkeypatch.setattr(lightning_mod.jax, "shard_map", fake_shard_map)
        monkeypatch.setattr(
            lightning_mod,
            "simple_gla_fwd",
            lambda q, k, v, **kwargs: (jnp.zeros_like(q), jnp.zeros_like(kwargs["h0"])),
        )

        H, K, layer_id = _H, _K, 5
        fake_mesh = SimpleNamespace(shape={"data": 2})
        backend = LightningAttnBackend(
            mesh=fake_mesh,
            linear_recurrent_layer_ids=[layer_id],
            num_hidden_layers=80,
            num_heads=H,
        )
        backend.forward_metadata = SimpleNamespace(
            cu_q_lens=jnp.array([0, 2, 0, 2], dtype=jnp.int32),
            recurrent_indices=jnp.array([1, 1], dtype=jnp.int32),
        )

        recurrent_buffer = jnp.zeros((4, H, K, K), dtype=jnp.float32)
        pool = MockRecurrentStatePool(layer_caches={layer_id: (recurrent_buffer, [])})
        layer = SimpleNamespace(layer_id=layer_id, num_heads=H, head_dim=K)
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        q = jnp.zeros((4, H, K), dtype=jnp.bfloat16)

        backend(q, q, q, layer=layer, forward_batch=forward_batch, recurrent_state_pool=pool)

        assert (
            (
                P("data", "tensor", None, None),
                P("data"),
            ),
            P("data", "tensor", None, None),
        ) in calls
        assert (
            (
                P("data", "tensor", None, None),
                P("data"),
                P("data", "tensor", None, None),
            ),
            P("data", "tensor", None, None),
        ) in calls
        assert (
            (
                P("data", "tensor", None),
                P("data", "tensor", None),
                P("data", "tensor", None),
                P("tensor"),
                P("data", "tensor", None, None),
                P("data"),
            ),
            (
                P("data", "tensor", None),
                P("data", "tensor", None, None),
            ),
        ) in calls
