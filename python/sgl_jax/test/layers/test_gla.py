"""Tests for BailingMoeV2_5LinearAttention -- GLA module.

Run with: pytest python/sgl_jax/test/layers/test_gla.py -v
"""

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.linear.lightning_backend import LightningAttnBackend
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.models.bailing_moe_v2_5_linear_attention import (
    BailingMoeV2_5LinearAttention,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.layers.mock_recurrent_state_pool import MockRecurrentStatePool

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from sgl_jax.srt.kernels.simple_gla.simple_gla import (  # noqa: F401
        fused_recurrent_simple_gla,
        simple_gla_fwd,
    )

    HAS_SIMPLE_GLA = True
except ImportError:
    HAS_SIMPLE_GLA = False

# Prefill (chunk) kernel uses Pallas TPU primitives and cannot run on CPU.
_HAS_TPU = any(d.platform == "tpu" for d in jax.devices())

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------
requires_simple_gla = pytest.mark.skipif(
    not HAS_SIMPLE_GLA, reason="simple_gla kernel not available"
)
requires_tpu = pytest.mark.skipif(not _HAS_TPU, reason="chunk kernel requires TPU")

# ===========================================================================
# Module test helpers (from test_linear_attention.py)
# ===========================================================================


def _make_config(
    hidden_size=8192,
    num_attention_heads=64,
    head_dim=128,
    partial_rotary_factor=0.5,
    use_qk_norm=True,
    group_norm_size=8,
    rms_norm_eps=1e-6,
    use_qkv_bias=False,
    use_bias=False,
    rope_theta=6_000_000,
    max_position_embeddings=131072,
    num_hidden_layers=80,
):
    return SimpleNamespace(**locals())


def _hf_build_slope_tensor(num_heads):
    """Ground-truth ALiBi slope computation from HuggingFace."""

    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + _hf_build_slope_tensor(2 * closest_power_of_2)[0::2][: num_heads - closest_power_of_2]
        )


def _expected_slope(num_heads, layer_id, num_hidden_layers):
    """Expected slope tensor matching the _compute_slope formula."""
    base = np.array(_hf_build_slope_tensor(num_heads), dtype=np.float32)
    return -base * (1 - (layer_id - 1) / (num_hidden_layers - 1) + 1e-5)


def _make_module(layer_id=1, config=None):
    """Construct a BailingMoeV2_5LinearAttention on CPU under the global mesh.

    Returns (module, backend). The backend is wired to dispatch via
    forward_batch.attn_backend; pass it through ``_make_forward_batch(...,
    backend=backend)`` so the module's RadixLightningAttention dispatcher can
    reach it.
    """
    if config is None:
        config = _make_config()
    backend = LightningAttnBackend(
        mesh=mesh,
        linear_recurrent_layer_ids=list(range(config.num_hidden_layers)),
        num_hidden_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
    )
    with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
        module = BailingMoeV2_5LinearAttention(
            config=config,
            layer_id=layer_id,
            mesh=mesh,
        )
    return module, backend


def _copy_weights_across_meshes(target_module, source_module):
    """Copy weights from source_module to target_module across different meshes.

    Extracts weights from source as numpy arrays, then places them on target's
    mesh using target's sharding. This tests numerical equivalence of TP
    computation without testing weight loading logic.
    """
    # Get all variables from both modules
    source_state = nnx.state(source_module)
    target_state = nnx.state(target_module)

    # Get flat state dictionaries
    source_flat = source_state.flat_state()
    target_flat = target_state.flat_state()

    # Copy each variable
    for path in source_flat:
        if path in target_flat:
            # Convert to numpy, then place on target device with target sharding
            source_var = source_flat[path]
            target_var = target_flat[path]
            source_np = np.array(source_var.value)
            # Use the target variable's existing sharding
            target_var.value = jax.device_put(jnp.array(source_np), target_var.value.sharding)

    # Update the target module with new state
    nnx.update(target_module, target_state)


def _make_tp_meshes():
    """Create meshes for TP consistency tests.

    Returns list of (tp_size, mesh) tuples for available device counts.
    """
    n_devices = len(jax.devices())
    meshes = []

    # Always include TP=1 (using only first device)
    meshes.append(
        (
            1,
            create_device_mesh(
                ici_parallelism=[1, 1], dcn_parallelism=[1, 1], devices=jax.devices()[:1]
            ),
        )
    )

    # Add TP=2 if we have at least 2 devices
    if n_devices >= 2:
        meshes.append(
            (
                2,
                create_device_mesh(
                    ici_parallelism=[1, 2], dcn_parallelism=[1, 1], devices=jax.devices()[:2]
                ),
            )
        )

    # Add TP=4 if we have 4 devices
    if n_devices >= 4:
        meshes.append(
            (
                4,
                create_device_mesh(
                    ici_parallelism=[1, 4], dcn_parallelism=[1, 1], devices=jax.devices()[:4]
                ),
            )
        )

    return meshes


def _make_mock_pool(layer_id, recurrent_state, recurrent_indices=None):
    """Create a MockRecurrentStatePool with the given recurrent state."""
    B = recurrent_state.shape[0]
    if recurrent_indices is None:
        recurrent_indices = np.arange(1, B + 1, dtype=np.int32)
    N_plus_1 = int(max(recurrent_indices)) + 1
    buf = jnp.zeros((N_plus_1,) + recurrent_state.shape[1:], dtype=recurrent_state.dtype)
    buf = buf.at[jnp.array(recurrent_indices)].set(recurrent_state)
    return MockRecurrentStatePool(layer_caches={layer_id: (buf, [])}), recurrent_indices


def _extract_state(pool_updates, recurrent_indices):
    """Extract recurrent state from pool_updates tuple with explicit output sharding."""
    new_ssm_full, conv_list = pool_updates
    indices = jnp.array(recurrent_indices)
    return new_ssm_full.at[indices].get(
        out_sharding=NamedSharding(mesh, P("data", "tensor", None, None))
    )


def _setup_backend_metadata(
    backend, forward_mode, recurrent_indices, extend_seq_lens=None, input_ids=None
):
    """Set up backend forward_metadata for the given forward mode."""
    batch = SimpleNamespace(
        forward_mode=forward_mode,
        recurrent_indices=recurrent_indices,
        has_initial_state=np.ones(len(recurrent_indices), dtype=np.bool_),
        dp_size=1,
        per_dp_bs_size=len(recurrent_indices),
    )
    if forward_mode == ForwardMode.DECODE:
        batch.seq_lens = np.ones(len(recurrent_indices), dtype=np.int32)
    elif forward_mode == ForwardMode.EXTEND:
        batch.extend_seq_lens = np.asarray(extend_seq_lens, dtype=np.int32)
        batch.seq_lens = np.asarray(extend_seq_lens, dtype=np.int32)
        batch.input_ids = np.asarray(input_ids, dtype=np.int32)
    metadata = backend.get_forward_metadata(batch)
    backend.forward_metadata = metadata
    return metadata


def _make_forward_batch(forward_mode, backend=None):
    """Build a fake forward_batch for module-level tests.

    The model layer dispatches via ``self.attn`` (RadixLightningAttention),
    which calls ``forward_batch.attn_backend(..., pool=...)``. In production
    that's HybridLinearAttnBackend, which routes to the linear sub-backend
    with ``recurrent_state_pool=pool``. Here we adapt LightningAttnBackend
    directly with the same keyword translation.
    """
    fb = SimpleNamespace(forward_mode=forward_mode)
    if backend is not None:

        def attn_backend_call(q, k, v, layer, forward_batch, pool, **kw):
            return backend(
                q,
                k,
                v,
                layer=layer,
                forward_batch=forward_batch,
                recurrent_state_pool=pool,
                **kw,
            )

        fb.attn_backend = attn_backend_call
    return fb


def _reshape_qkv(qkv, T, num_heads, head_dim, m=None):
    """Reshape fused QKV tensor respecting tensor-parallel sharding.

    When running under TP, qkv has sharding P(None, "tensor") on the last dim.
    A bare .reshape() fails because JAX cannot infer the output sharding.
    Use jax.lax.reshape with explicit out_sharding, matching what the model does.
    """
    if m is None:
        m = mesh
    return jax.lax.reshape(
        qkv,
        (T, 3, num_heads, head_dim),
        out_sharding=NamedSharding(m, P(None, None, "tensor", None)),
    )


_SMALL_CONFIG = _make_config(
    hidden_size=512,
    num_attention_heads=4,
    head_dim=128,
    num_hidden_layers=10,
    partial_rotary_factor=0.5,
    max_position_embeddings=1024,
    rope_theta=10000,
    group_norm_size=4,
)

_SMALL_H = 4
_SMALL_K = 128
_SMALL_HIDDEN = 512

# ===========================================================================
# Cross-framework helpers (from test_cross_framework_linear_attention.py)
# ===========================================================================

# TPU float32 matmul uses reduced precision (MXU accumulates in bf16),
# causing ~0.17 max diff vs PyTorch CPU. This is a hardware characteristic,
# not a code bug. Use platform-appropriate atol for matmul-based tests.
_CF_IS_TPU = any(d.platform == "tpu" for d in jax.devices())
_CF_MATMUL_ATOL = 0.2 if _CF_IS_TPU else 5e-5

_CF_H = 4
_CF_K = 64
_CF_HIDDEN = 256
_CF_NUM_LAYERS = 10
_CF_NUM_GROUPS = 4
_CF_EPS = 1e-6
_CF_ROPE_THETA = 10000
_CF_PARTIAL_ROTARY_FACTOR = 0.5
_CF_ROTARY_DIM = int(_CF_K * _CF_PARTIAL_ROTARY_FACTOR)  # 32


def _make_cf_config():
    return SimpleNamespace(
        hidden_size=_CF_HIDDEN,
        num_attention_heads=_CF_H,
        head_dim=_CF_K,
        num_hidden_layers=_CF_NUM_LAYERS,
        partial_rotary_factor=_CF_PARTIAL_ROTARY_FACTOR,
        use_qk_norm=True,
        group_norm_size=_CF_NUM_GROUPS,
        rms_norm_eps=_CF_EPS,
        use_qkv_bias=False,
        use_bias=False,
        rope_theta=_CF_ROPE_THETA,
        max_position_embeddings=1024,
    )


def _make_cf_module(layer_idx=5, dtype=jnp.float32):
    """Same shape as _make_module but with the cross-framework config; returns (module, backend)."""
    config = _make_cf_config()
    backend = LightningAttnBackend(
        mesh=mesh,
        linear_recurrent_layer_ids=list(range(config.num_hidden_layers)),
        num_hidden_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
    )
    with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
        module = BailingMoeV2_5LinearAttention(
            config=config, layer_id=layer_idx, mesh=mesh, dtype=dtype
        )
    return module, backend


# ---------------------------------------------------------------------------
# Pure-torch reference implementations
# ---------------------------------------------------------------------------


def jax_to_numpy(x):
    return np.array(x)


# ===========================================================================
# Test classes
# ===========================================================================

# ---------------------------------------------------------------------------
# ALiBi slope tensor tests (via module._compute_slope / module.slope)
# ---------------------------------------------------------------------------


class TestSlopes:
    def test_slopes_match_hf_formula(self):
        """Slope tensor must exactly match the reference formula for several layers.

        Slope is now owned by the backend (per upstream LightningAttention
        pattern), so we read it from ``backend.tp_slope[layer_id]`` rather
        than ``module.slope``.
        """
        config = _make_config()
        for layer_idx in [1, 10, 40, 79]:
            module, backend = _make_module(layer_id=layer_idx, config=config)
            actual = np.asarray(backend.tp_slope[layer_idx])
            expected = _expected_slope(
                config.num_attention_heads, layer_idx, config.num_hidden_layers
            )
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Slope mismatch at layer_id={layer_idx}",
            )


# ---------------------------------------------------------------------------
# Multi-request isolation tests (require simple_gla kernel)
# ---------------------------------------------------------------------------


class TestMultiRequestIsolation:
    """Verify multi-request state isolation in varlen format."""

    @requires_simple_gla
    @requires_tpu
    def test_multi_request_isolation(self):
        """Two requests in same batch should not interfere with each other's state."""
        pass  # Placeholder for future multi-request isolation tests
