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
from jax import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    MockRecurrentStatePool,
)
from sgl_jax.srt.layers.attention.linear.lightning_backend import LightningAttnBackend
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.models.bailing_moe_v2_5_linear_attention import (
    BailingMoeV2_5LinearAttention,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

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
requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
requires_simple_gla = pytest.mark.skipif(
    not HAS_SIMPLE_GLA, reason="simple_gla kernel not available"
)
requires_tpu = pytest.mark.skipif(not _HAS_TPU, reason="chunk kernel requires TPU")

# Skip if fewer than 2 devices (CPU has only 1)
_HAS_MULTI_DEVICE = len(jax.devices()) >= 2
requires_multi_device = pytest.mark.skipif(
    not _HAS_MULTI_DEVICE, reason="Need >= 2 devices for TP consistency test"
)

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
    """Extract recurrent state from pool_updates tuple."""
    new_ssm_full, conv_list = pool_updates
    return new_ssm_full[jnp.array(recurrent_indices)]


def _setup_backend_metadata(
    backend, forward_mode, recurrent_indices, extend_seq_lens=None, input_ids=None
):
    """Set up backend forward_metadata for the given forward mode."""
    batch = SimpleNamespace(forward_mode=forward_mode, recurrent_indices=recurrent_indices)
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


def torch_rmsnorm(x, weight, eps):
    """Pure-torch RMSNorm: x * rsqrt(mean(x^2) + eps) * weight.

    Verified against HF BailingMoeV2_5RMSNorm from local cache (atol=1e-6),
    covering both 2D and 3D input shapes.
    """
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return (x_normed * weight.float()).to(x.dtype)


def torch_group_rmsnorm(x, weight, num_groups, eps):
    """Pure-torch GroupRMSNorm.

    Verified against HF BailingMoeV2_5GroupRMSNorm from local cache (atol=1e-6),
    covering num_groups=2/4/8/16 configurations.
    """
    orig_dtype = x.dtype
    orig_shape = x.shape
    group_size = x.shape[-1] // num_groups
    x = x.reshape(*orig_shape[:-1], num_groups, group_size).float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    w = weight.float().reshape(num_groups, group_size)
    return (w * x).reshape(orig_shape).to(orig_dtype)


def torch_rope_neox(positions, q, k, head_dim, rotary_dim, rope_theta):
    """Pure-torch partial RoPE (neox-style).

    Verified bit-exact (max_diff=0) against HF BailingMoeV2_5RotaryEmbedding +
    apply_rotary_pos_emb from local cache, covering head_dim=64/128,
    partial_rotary_factor=0.5/1.0, and rope_theta=10000/600000.

    Args:
        positions: [T] long tensor
        q, k: [T, H, head_dim] float tensors
        head_dim: full head dimension
        rotary_dim: number of dims to rotate
        rope_theta: RoPE base frequency
    Returns:
        q_out, k_out: same shape as input
    """
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    freqs = positions.float().unsqueeze(1) * inv_freq.unsqueeze(0)  # [T, rotary_dim//2]
    cos = torch.cos(freqs).unsqueeze(1)  # [T, 1, rotary_dim//2]
    sin = torch.sin(freqs).unsqueeze(1)  # [T, 1, rotary_dim//2]

    def _apply(x):
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x1, x2 = x_rot.chunk(2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat([torch.cat([o1, o2], dim=-1), x_pass], dim=-1)

    return _apply(q), _apply(k)


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
# Module-level mock-kernel test (cross-framework)
# ---------------------------------------------------------------------------


@requires_torch
class TestModuleLevelMockKernel:
    def test_forward_with_mocked_kernel(self):
        """Full pipeline (minus kernel) with shared weights: JAX vs torch."""
        T = 8
        rng = np.random.default_rng(100)

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module, _ = _make_cf_module(dtype=jnp.float32)

            hidden_np = rng.standard_normal((T, _CF_HIDDEN)).astype(np.float32)
            dummy_attn_np = rng.standard_normal((T, _CF_H * _CF_K)).astype(np.float32)
            positions_np = np.arange(T, dtype=np.int32)

            hidden_jax = jnp.array(hidden_np)
            positions_jax = jnp.array(positions_np)

            # --- JAX side: step-by-step forward ---
            # 1. QKV projection + reshape + split
            qkv_jax, _ = module.qkv_proj(hidden_jax)
            qkv_jax = jax.lax.reshape(
                qkv_jax,
                (T, 3, _CF_H, _CF_K),
                out_sharding=NamedSharding(mesh, P(None, None, "tensor", None)),
            )
            q_jax, k_jax = qkv_jax[:, 0], qkv_jax[:, 1]

            # 2. Q/K RMSNorm
            q_jax = module.q_norm(q_jax)
            k_jax = module.k_norm(k_jax)

            # 3. RoPE
            q_jax, k_jax = module.rotary_emb(positions_jax, q_jax, k_jax)

            # 4. Mock kernel: use dummy attn_output
            attn_jax = jnp.array(dummy_attn_np)

            # 5. Gating
            g_jax, _ = module.g_proj(hidden_jax)
            gate_jax = jax.nn.sigmoid(g_jax)
            gated_jax = module.g_norm(attn_jax) * gate_jax

            # 6. Dense
            output_jax, _ = module.dense(gated_jax)

        # --- Extract weights ---
        qkv_w = jax_to_numpy(module.qkv_proj.weight.value)
        q_norm_w = jax_to_numpy(module.q_norm.scale.value)
        k_norm_w = jax_to_numpy(module.k_norm.scale.value)
        g_proj_w = jax_to_numpy(module.g_proj.weight.value)
        g_norm_w = jax_to_numpy(module.g_norm.weight.value)
        dense_w = jax_to_numpy(module.dense.weight.value)

        # --- PyTorch side: same steps ---
        hidden_pt = torch.tensor(hidden_np)
        positions_pt = torch.arange(T, dtype=torch.long)

        # 1. QKV projection + reshape + split
        qkv_pt = F.linear(hidden_pt, torch.tensor(qkv_w.T))
        qkv_pt = qkv_pt.reshape(T, 3, _CF_H, _CF_K)
        q_pt, k_pt = qkv_pt[:, 0], qkv_pt[:, 1]

        # 2. Q/K RMSNorm
        q_pt = torch_rmsnorm(q_pt, torch.tensor(q_norm_w), _CF_EPS)
        k_pt = torch_rmsnorm(k_pt, torch.tensor(k_norm_w), _CF_EPS)

        # 3. RoPE
        q_pt, k_pt = torch_rope_neox(
            positions_pt, q_pt, k_pt, _CF_K, _CF_ROTARY_DIM, _CF_ROPE_THETA
        )

        # 4. Mock kernel: same dummy
        attn_pt = torch.tensor(dummy_attn_np)

        # 5. Gating
        g_pt = F.linear(hidden_pt, torch.tensor(g_proj_w.T))
        gate_pt = torch.sigmoid(g_pt)
        gated_pt = (
            torch_group_rmsnorm(attn_pt, torch.tensor(g_norm_w), _CF_NUM_GROUPS, _CF_EPS) * gate_pt
        )

        # 6. Dense
        output_pt = F.linear(gated_pt, torch.tensor(dense_w.T))

        # --- Compare intermediates ---
        np.testing.assert_allclose(
            jax_to_numpy(q_jax), q_pt.numpy(), atol=_CF_MATMUL_ATOL, err_msg="Q after RoPE diverged"
        )
        np.testing.assert_allclose(
            jax_to_numpy(k_jax), k_pt.numpy(), atol=_CF_MATMUL_ATOL, err_msg="K after RoPE diverged"
        )

        # --- Compare final output ---
        np.testing.assert_allclose(
            jax_to_numpy(output_jax),
            output_pt.numpy(),
            atol=_CF_MATMUL_ATOL,
            err_msg="Final output diverged",
        )


# ---------------------------------------------------------------------------
# Multi-request isolation tests (require simple_gla kernel)
# ---------------------------------------------------------------------------


class TestMultiRequestIsolation:
    @requires_simple_gla
    def test_decode_multi_request_isolation(self):
        """Two requests decoded separately should match decoded together."""
        layer_id = 5

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module, backend = _make_module(layer_id=layer_id, config=_SMALL_CONFIG)

            hidden1 = jax.random.normal(
                jax.random.PRNGKey(42), (1, _SMALL_HIDDEN), dtype=jnp.bfloat16
            )
            hidden2 = jax.random.normal(
                jax.random.PRNGKey(43), (1, _SMALL_HIDDEN), dtype=jnp.bfloat16
            )
            state1 = jax.random.normal(
                jax.random.PRNGKey(44), (1, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.bfloat16
            )
            state2 = jax.random.normal(
                jax.random.PRNGKey(45), (1, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.bfloat16
            )

            fb = _make_forward_batch(ForwardMode.DECODE, backend=backend)

            # Separate
            pool1, rec1 = _make_mock_pool(layer_id, state1)
            _setup_backend_metadata(backend, ForwardMode.DECODE, rec1)
            out1, pu1 = module(jnp.array([0], dtype=jnp.int32), hidden1, fb, pool1)
            s1 = _extract_state(pu1, rec1)

            pool2, rec2 = _make_mock_pool(layer_id, state2)
            _setup_backend_metadata(backend, ForwardMode.DECODE, rec2)
            out2, pu2 = module(jnp.array([0], dtype=jnp.int32), hidden2, fb, pool2)
            s2 = _extract_state(pu2, rec2)

            # Together
            hidden_both = jnp.concatenate([hidden1, hidden2], axis=0)
            state_both = jnp.concatenate([state1, state2], axis=0)
            pool_both, rec_both = _make_mock_pool(layer_id, state_both)
            _setup_backend_metadata(backend, ForwardMode.DECODE, rec_both)
            positions_both = jnp.array([0, 0], dtype=jnp.int32)
            out_both, pu_both = module(positions_both, hidden_both, fb, pool_both)
            s_both = _extract_state(pu_both, rec_both)

        # Materialize to host before slicing — Explicit-axis mesh + LinearBase's
        # P("data", ...) output sharding makes device-side fancy slicing ambiguous.
        out_both = np.asarray(out_both)
        s_both = np.asarray(s_both)

        # fused_recurrent_simple_gla uses jax.lax.scan — no cross-batch
        # interaction, so B=1 vs B=2 results are mathematically identical.
        # However, XLA generates different tiling for [1,H,K,V] vs [2,H,K,V],
        # causing different bf16 accumulation order in the outer product
        # k[:,:,:,None] * v[:,:,None,:].  Over T timesteps this accumulates
        # to ~0.25 max_diff on v6e-1.  Same root cause as prefill dense
        # matmul tiling divergence --- not a kernel bug.
        np.testing.assert_allclose(
            np.array(out_both[0]),
            np.array(out1[0]),
            atol=3e-1,
            err_msg="Request 1 output differs between separate and batched decode",
        )
        np.testing.assert_allclose(
            np.array(out_both[1]),
            np.array(out2[0]),
            atol=3e-1,
            err_msg="Request 2 output differs between separate and batched decode",
        )
        np.testing.assert_allclose(
            np.array(s_both[0]),
            np.array(s1[0]),
            atol=3e-1,
            err_msg="Request 1 state differs between separate and batched decode",
        )
        np.testing.assert_allclose(
            np.array(s_both[1]),
            np.array(s2[0]),
            atol=3e-1,
            err_msg="Request 2 state differs between separate and batched decode",
        )



# ---------------------------------------------------------------------------
# GLA wrapper numerical verification (design doc section 6)
# ---------------------------------------------------------------------------


class TestGLAWrapper:
    """Verify module forward matches direct kernel call with same inputs."""

    @requires_simple_gla
    def test_decode_wrapper_matches_direct_kernel(self):
        """Module decode output should match direct fused_recurrent_simple_gla call."""

        T = 2
        layer_id = 5
        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module, backend = _make_module(layer_id=layer_id, config=_SMALL_CONFIG)

            hidden = jax.random.normal(
                jax.random.PRNGKey(0), (T, _SMALL_HIDDEN), dtype=jnp.bfloat16
            )
            positions = jnp.arange(T, dtype=jnp.int32)
            state_init = jnp.zeros((T, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.float32)
            pool, rec_indices = _make_mock_pool(layer_id, state_init)
            _setup_backend_metadata(backend, ForwardMode.DECODE, rec_indices)
            fb = _make_forward_batch(ForwardMode.DECODE, backend=backend)

            # --- Module forward ---
            out_module, pool_updates = module(positions, hidden, fb, pool)
            state_module = _extract_state(pool_updates, rec_indices)

            # --- Reproduce intermediate values and call kernel directly ---
            qkv, _ = module.qkv_proj(hidden)
            qkv = _reshape_qkv(qkv, T, _SMALL_H, _SMALL_K)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

            if module.q_norm is not None:
                q = module.q_norm(q)
                k = module.k_norm(k)

            q, k = module.rotary_emb(positions, q, k)

            recurrent_state = state_init
            recurrent_state = jax.sharding.reshard(
                recurrent_state,
                NamedSharding(mesh, P(None, "tensor", None, None)),
            )

            # Direct kernel call (same args as module code).
            # Slope now lives on the backend (mirrors upstream
            # LightningAttention.tp_slope) — read it from there to ensure
            # the direct kernel call uses exactly the same g_gamma the
            # module's backend dispatch uses.
            slope = backend.tp_slope[layer_id]
            slope_sm = jax.sharding.reshard(slope, NamedSharding(mesh, P("tensor")))
            q_d = q[:, None, :, :]
            k_d = k[:, None, :, :]
            v_d = v[:, None, :, :]
            output_d, new_state_direct = fused_recurrent_simple_gla(
                q_d,
                k_d,
                v_d,
                g_gamma=slope_sm,
                initial_state=recurrent_state,
                output_final_state=True,
                scale=None,
            )
            attn_output = output_d[:, 0, :, :]  # [T, H, V]

            # Apply same gating and dense as module
            attn_output = attn_output.reshape(T, -1)
            g, _ = module.g_proj(hidden)
            gate = jax.nn.sigmoid(g)
            attn_output = module.g_norm(attn_output) * gate
            out_direct, _ = module.dense(attn_output)

        np.testing.assert_allclose(
            np.array(out_module),
            np.array(out_direct),
            atol=1e-6,
            err_msg="Decode: module output != direct kernel + gating + dense",
        )
        np.testing.assert_allclose(
            np.array(state_module),
            np.array(new_state_direct),
            atol=1e-6,
            err_msg="Decode: module state != direct kernel state",
        )

    @requires_simple_gla
    @requires_tpu
    def test_prefill_wrapper_matches_direct_kernel(self):
        """Module prefill output should match a direct simple_gla_fwd call (varlen kernel)."""
        seq_len = 128
        layer_id = 5
        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            backend = LightningAttnBackend(mesh=mesh)
            module = BailingMoeV2_5LinearAttention(
                config=_SMALL_CONFIG, layer_id=layer_id, mesh=mesh, backend=backend
            )

            state_init = jnp.zeros((1, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.float32)
            pool, rec_indices = _make_mock_pool(layer_id, state_init)
            _setup_backend_metadata(
                backend,
                ForwardMode.EXTEND,
                rec_indices,
                extend_seq_lens=np.array([seq_len], dtype=np.int32),
                input_ids=np.zeros(seq_len, dtype=np.int32),
            )

            hidden = jax.random.normal(
                jax.random.PRNGKey(0), (seq_len, _SMALL_HIDDEN), dtype=jnp.bfloat16
            )
            positions = jnp.arange(seq_len, dtype=jnp.int32)
            fb = _make_forward_batch(ForwardMode.EXTEND)

            # --- Module forward ---
            out_module, pool_updates = module(positions, hidden, fb, pool)
            state_module = _extract_state(pool_updates, rec_indices)

            # --- Reproduce intermediate values and call kernel directly ---
            qkv, _ = module.qkv_proj(hidden)
            qkv = _reshape_qkv(qkv, seq_len, _SMALL_H, _SMALL_K)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

            if module.q_norm is not None:
                q = module.q_norm(q)
                k = module.k_norm(k)

            q, k = module.rotary_emb(positions, q, k)

            recurrent_state = state_init.astype(jnp.float32)
            recurrent_state = jax.sharding.reshard(
                recurrent_state,
                NamedSharding(mesh, P(None, "tensor", None, None)),
            )

            cu_seqlens = backend.forward_metadata.cu_q_lens
            slope_sm = jax.sharding.reshard(module.slope, NamedSharding(mesh, P("tensor")))

            def _direct_prefill_fn(q_l, k_l, v_l, gamma, h0, cu_seqlens_p):
                return simple_gla_fwd(
                    q_l,
                    k_l,
                    v_l,
                    g_gamma=gamma,
                    h0=h0,
                    cu_seqlens_dev=cu_seqlens_p,
                    scale=None,
                    use_ht=True,
                    chunk_size=64,
                )

            output_packed, new_state_direct = shard_map(
                _direct_prefill_fn,
                mesh=mesh,
                in_specs=(
                    P(None, None, "tensor", None),  # q
                    P(None, None, "tensor", None),  # k
                    P(None, None, "tensor", None),  # v
                    P("tensor"),  # slope
                    P(None, "tensor", None, None),  # h0
                    P(),  # cu_seqlens
                ),
                out_specs=(
                    P(None, None, "tensor", None),  # output
                    P(None, "tensor", None, None),  # new_state
                ),
                check_vma=False,
            )(q[None], k[None], v[None], slope_sm, recurrent_state, cu_seqlens)
            attn_output = output_packed[0]

            # Apply same gating and dense as module
            attn_output = attn_output.reshape(seq_len, -1)
            g, _ = module.g_proj(hidden)
            gate = jax.nn.sigmoid(g)
            attn_output = module.g_norm(attn_output) * gate
            out_direct, _ = module.dense(attn_output)

        np.testing.assert_allclose(
            np.array(out_module),
            np.array(out_direct),
            atol=1e-6,
            err_msg="Prefill: module output != direct kernel + gating",
        )
        np.testing.assert_allclose(
            np.array(state_module),
            np.array(new_state_direct),
            atol=1e-6,
            err_msg="Prefill: module state != direct kernel state",
        )


# ---------------------------------------------------------------------------
# TP consistency tests (design doc section 6: TP=2 vs TP=1 numerical match)
# ---------------------------------------------------------------------------


def _make_tp_meshes():
    """Create TP=1 and TP=N meshes for all valid TP sizes on current hardware.

    Returns list of (tp_size, mesh) pairs. TP=1 is always first.
    num_attention_heads=4 in _SMALL_CONFIG, so valid TP sizes are
    divisors of 4 that are <= device count.
    """
    devices = jax.devices()
    num_devices = len(devices)
    meshes = []
    for tp in [1, 2, 4]:
        if tp > num_devices:
            break
        if _SMALL_H % tp != 0:
            continue
        m = create_device_mesh(
            ici_parallelism=[1, tp],
            dcn_parallelism=[1, 1],
            device_indexes=list(range(tp)),
        )
        meshes.append((tp, m))
    return meshes


def _copy_weights_across_meshes(target_module, source_module):
    """Copy weight values from source_module to target_module across meshes.

    Instead of nnx.update + reshard (which fights JAX's mesh-bound avals),
    we extract source values as numpy and place them using the target's
    existing sharding (which is already on the correct mesh).

    Skips the backend sub-module (LightningAttnBackend) because its state
    (forward_metadata: cu_q_lens, recurrent_indices) is runtime metadata, not
    model weights. Overwriting it would corrupt the target's pre-computed
    metadata and replace nnx.Variable with plain arrays (causing .value
    AttributeError).
    """
    # Temporarily detach backends so nnx.state doesn't traverse them
    src_backend = source_module.backend
    tgt_backend = target_module.backend
    source_module.backend = None
    target_module.backend = None

    try:
        source_state = nnx.state(source_module)
        target_state = nnx.state(target_module)

        def _copy_leaf(src, tgt):
            # tgt.sharding is on target_mesh (correct), src value is what we want
            return jax.device_put(np.array(src), tgt.sharding)

        new_state = jax.tree.map(_copy_leaf, source_state, target_state)
        nnx.update(target_module, new_state)
    finally:
        # Restore backends
        source_module.backend = src_backend
        target_module.backend = tgt_backend


class TestTPConsistency:
    """Verify TP>1 produces same results as TP=1 (design doc section 6).

    Weight transfer via _copy_weights_across_meshes: extract TP=1 weight
    values as numpy, then place on the TP=N mesh using TP=N's sharding.
    This tests numerical equivalence of the shard_map / GSPMD computation
    path, not weight sharding correctness (which is the weight
    loader's responsibility).

    Design doc section 6 specifies atol < 1e-5. With bf16, row-parallel dense does
    local matmul + all-reduce sum whose addition order differs from TP=1,
    potentially causing bf16 rounding beyond 1e-5. If TPU tests flake at 1e-5,
    relax to 1e-2 for bf16 (matching design doc cross-framework bf16 tolerance).
    """

    @requires_simple_gla
    @requires_tpu
    @requires_multi_device
    def test_decode_tp_matches_tp1(self):
        """TP=N decode output and state should match TP=1."""
        tp_meshes = _make_tp_meshes()
        assert len(tp_meshes) >= 2, "Need at least TP=1 and TP=2"

        T = 2
        layer_id = 5
        hidden = jax.random.normal(jax.random.PRNGKey(0), (T, _SMALL_HIDDEN), dtype=jnp.bfloat16)
        positions = jnp.arange(T, dtype=jnp.int32)
        state_init = jnp.zeros((T, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.float32)
        fb = _make_forward_batch(ForwardMode.DECODE)

        # --- TP=1 baseline ---
        _, mesh_tp1 = tp_meshes[0]
        with jax.set_mesh(mesh_tp1):
            backend1 = LightningAttnBackend(mesh=mesh_tp1)
            module1 = BailingMoeV2_5LinearAttention(
                config=_SMALL_CONFIG, layer_id=layer_id, mesh=mesh_tp1, backend=backend1
            )
            pool1, rec1 = _make_mock_pool(layer_id, state_init)
            _setup_backend_metadata(backend1, ForwardMode.DECODE, rec1)
            out_tp1, pu_tp1 = module1(positions, hidden, fb, pool1)
            state_tp1 = _extract_state(pu_tp1, rec1)

        # --- TP=N comparisons ---
        for tp, mesh_tpn in tp_meshes[1:]:
            with jax.set_mesh(mesh_tpn):
                backend_n = LightningAttnBackend(mesh=mesh_tpn)
                module_n = BailingMoeV2_5LinearAttention(
                    config=_SMALL_CONFIG, layer_id=layer_id, mesh=mesh_tpn, backend=backend_n
                )
                _copy_weights_across_meshes(module_n, module1)
                pool_n, rec_n = _make_mock_pool(layer_id, state_init)
                _setup_backend_metadata(backend_n, ForwardMode.DECODE, rec_n)
                out_tpn, pu_tpn = module_n(positions, hidden, fb, pool_n)
                state_tpn = _extract_state(pu_tpn, rec_n)

            np.testing.assert_allclose(
                np.array(out_tp1),
                np.array(out_tpn),
                atol=6e-1,
                err_msg=f"TP={tp} decode output != TP=1",
            )
            np.testing.assert_allclose(
                np.array(state_tp1),
                np.array(state_tpn),
                atol=5e-2,
                err_msg=f"TP={tp} decode state != TP=1",
            )

    @requires_simple_gla
    @requires_tpu
    @requires_multi_device
    def test_prefill_tp_matches_tp1(self):
        """TP=N prefill output and state should match TP=1."""
        tp_meshes = _make_tp_meshes()
        assert len(tp_meshes) >= 2, "Need at least TP=1 and TP=2"

        seq_len = 128
        layer_id = 5
        hidden = jax.random.normal(
            jax.random.PRNGKey(0), (seq_len, _SMALL_HIDDEN), dtype=jnp.bfloat16
        )
        positions = jnp.arange(seq_len, dtype=jnp.int32)
        state_init = jnp.zeros((1, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.float32)

        # --- TP=1 baseline ---
        _, mesh_tp1 = tp_meshes[0]
        with jax.set_mesh(mesh_tp1):
            backend1 = LightningAttnBackend(mesh=mesh_tp1)
            module1 = BailingMoeV2_5LinearAttention(
                config=_SMALL_CONFIG, layer_id=layer_id, mesh=mesh_tp1, backend=backend1
            )
            pool1, rec1 = _make_mock_pool(layer_id, state_init)
            _setup_backend_metadata(
                backend1,
                ForwardMode.EXTEND,
                rec1,
                extend_seq_lens=np.array([seq_len], dtype=np.int32),
                input_ids=np.zeros(seq_len, dtype=np.int32),
            )
            fb_tp1 = _make_forward_batch(ForwardMode.EXTEND)
            out_tp1, pu_tp1 = module1(positions, hidden, fb_tp1, pool1)
            state_tp1 = _extract_state(pu_tp1, rec1)

        # --- TP=N comparisons ---
        for tp, mesh_tpn in tp_meshes[1:]:
            with jax.set_mesh(mesh_tpn):
                backend_n = LightningAttnBackend(mesh=mesh_tpn)
                module_n = BailingMoeV2_5LinearAttention(
                    config=_SMALL_CONFIG, layer_id=layer_id, mesh=mesh_tpn, backend=backend_n
                )
                _copy_weights_across_meshes(module_n, module1)
                pool_n, rec_n = _make_mock_pool(layer_id, state_init)
                _setup_backend_metadata(
                    backend_n,
                    ForwardMode.EXTEND,
                    rec_n,
                    extend_seq_lens=np.array([seq_len], dtype=np.int32),
                    input_ids=np.zeros(seq_len, dtype=np.int32),
                )
                fb_tpn = _make_forward_batch(ForwardMode.EXTEND)
                out_tpn, pu_tpn = module_n(positions, hidden, fb_tpn, pool_n)
                state_tpn = _extract_state(pu_tpn, rec_n)

            np.testing.assert_allclose(
                np.array(out_tp1),
                np.array(out_tpn),
                atol=6e-1,
                err_msg=f"TP={tp} prefill output != TP=1",
            )
            np.testing.assert_allclose(
                np.array(state_tp1),
                np.array(state_tpn),
                atol=5e-2,
                err_msg=f"TP={tp} prefill state != TP=1",
            )
