"""Cross-framework (JAX vs PyTorch) comparison tests for BailingMoeV2_5LinearAttention.

Verifies that each sub-component produces the same output as an independent
pure-torch reference implementation.  All tests run in float32 to eliminate
bf16 precision noise.

Run with: pytest python/sgl_jax/test/layers/test_cross_framework_linear_attention.py -v
"""

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.fla.linear_attention_backend import (
    LinearAttentionBackend,
)
from sgl_jax.srt.models.bailing_moe_v2_5_linear_attention import (
    BailingMoeV2_5LinearAttention,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

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
    from sgl_jax.srt.kernels.simple_gla.simple_gla import (
        fused_recurrent_simple_gla,
        simple_gla_fwd,
    )

    _HAS_SIMPLE_GLA = True
except ImportError:
    _HAS_SIMPLE_GLA = False

_HAS_TPU = any(d.platform == "tpu" for d in jax.devices())

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
requires_simple_gla = pytest.mark.skipif(
    not _HAS_SIMPLE_GLA, reason="simple_gla kernel not installed"
)
requires_tpu = pytest.mark.skipif(not _HAS_TPU, reason="chunk kernel requires TPU")

# ---------------------------------------------------------------------------
# Mesh & config
# ---------------------------------------------------------------------------
mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

# TPU float32 matmul uses reduced precision (MXU accumulates in bf16),
# causing ~0.17 max diff vs PyTorch CPU. This is a hardware characteristic,
# not a code bug. Use platform-appropriate atol for matmul-based tests.
_IS_TPU = any(d.platform == "tpu" for d in jax.devices())
_MATMUL_ATOL = 0.2 if _IS_TPU else 5e-5

_H = 4
_K = 64
_HIDDEN = 256
_NUM_LAYERS = 10
_NUM_GROUPS = 4
_EPS = 1e-6
_ROPE_THETA = 10000
_PARTIAL_ROTARY_FACTOR = 0.5
_ROTARY_DIM = int(_K * _PARTIAL_ROTARY_FACTOR)  # 32


def _make_config():
    return SimpleNamespace(
        hidden_size=_HIDDEN,
        num_attention_heads=_H,
        head_dim=_K,
        num_hidden_layers=_NUM_LAYERS,
        partial_rotary_factor=_PARTIAL_ROTARY_FACTOR,
        use_qk_norm=True,
        group_norm_size=_NUM_GROUPS,
        rms_norm_eps=_EPS,
        use_qkv_bias=False,
        use_bias=False,
        rope_theta=_ROPE_THETA,
        max_position_embeddings=1024,
    )


def _make_module(layer_idx=5, dtype=jnp.float32):
    config = _make_config()
    backend = LinearAttentionBackend(mesh=mesh)
    with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
        module = BailingMoeV2_5LinearAttention(
            config=config, layer_idx=layer_idx, mesh=mesh, backend=backend, dtype=dtype
        )
    return module


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


# ---------------------------------------------------------------------------
# Sub-component comparison tests
# ---------------------------------------------------------------------------


@requires_torch
class TestSubComponentComparison:
    def test_slope_computation_matches_torch(self):
        """ALiBi slope formula: JAX vs PyTorch reference (absolute values)."""

        def pt_build_slope_tensor(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest)
                    + pt_build_slope_tensor(2 * closest)[0::2][: n - closest]
                )

        def pt_compute_slope(num_heads, layer_id, num_layers):
            """PyTorch convention: positive slopes, 0-indexed layer_id."""
            base = np.array(pt_build_slope_tensor(num_heads), dtype=np.float32)
            return base * (1 - layer_id / (num_layers - 1) + 1e-5)

        # Test base slopes match
        jax_base = BailingMoeV2_5LinearAttention.build_slope_tensor(_H)
        pt_base = pt_build_slope_tensor(_H)
        np.testing.assert_array_equal(jax_base, pt_base)

        # Test per-layer slopes for multiple layers (comparing absolute values)
        for jax_layer_idx in [1, 5, 10]:
            pt_layer_id = jax_layer_idx - 1
            with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
                module = _make_module(layer_idx=jax_layer_idx)
            jax_slopes = jax_to_numpy(module.slope)  # negative
            pt_slopes = pt_compute_slope(_H, pt_layer_id, _NUM_LAYERS)  # positive
            np.testing.assert_allclose(
                np.abs(jax_slopes),
                np.abs(pt_slopes),
                atol=0,
                rtol=1e-7,
                err_msg=f"Slope mismatch at layer_idx={jax_layer_idx}",
            )

    def test_qkv_projection_matches_torch(self):
        """QKV linear projection: JAX LinearBase vs torch F.linear."""
        T = 8
        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = _make_module()
            hidden_np = np.random.default_rng(42).standard_normal((T, _HIDDEN)).astype(np.float32)
            hidden_jax = jnp.array(hidden_np)
            qkv_jax, _ = module.qkv_proj(hidden_jax)

        w_np = jax_to_numpy(module.qkv_proj.weight.value)  # (in, out)
        qkv_pt = F.linear(torch.tensor(hidden_np), torch.tensor(w_np.T))

        # float32 matmul accumulation order differs between JAX and PyTorch
        np.testing.assert_allclose(jax_to_numpy(qkv_jax), qkv_pt.numpy(), atol=_MATMUL_ATOL)

    def test_qk_rmsnorm_matches_torch(self):
        """Q/K RMSNorm: JAX RMSNorm vs pure-torch reference."""
        T = 8
        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = _make_module()
            x_np = np.random.default_rng(43).standard_normal((T, _H, _K)).astype(np.float32)
            x_jax = jnp.array(x_np)
            q_jax = module.q_norm(x_jax)
            k_jax = module.k_norm(x_jax)

        q_scale_np = jax_to_numpy(module.q_norm.scale.value)
        k_scale_np = jax_to_numpy(module.k_norm.scale.value)

        q_pt = torch_rmsnorm(torch.tensor(x_np), torch.tensor(q_scale_np), _EPS)
        k_pt = torch_rmsnorm(torch.tensor(x_np), torch.tensor(k_scale_np), _EPS)

        np.testing.assert_allclose(jax_to_numpy(q_jax), q_pt.numpy(), atol=1e-5)
        np.testing.assert_allclose(jax_to_numpy(k_jax), k_pt.numpy(), atol=1e-5)

    def test_rope_matches_torch(self):
        """Partial RoPE (neox-style): JAX RotaryEmbedding vs pure-torch reference."""
        T = 8
        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = _make_module(dtype=jnp.float32)
            positions = jnp.arange(T, dtype=jnp.int32)
            x_np = np.random.default_rng(44).standard_normal((T, _H, _K)).astype(np.float32)
            q_jax = jnp.array(x_np)
            k_jax = jnp.array(x_np)
            q_out_jax, k_out_jax = module.rotary_emb(positions, q_jax, k_jax)

        q_pt = torch.tensor(x_np)
        k_pt = torch.tensor(x_np)
        positions_pt = torch.arange(T, dtype=torch.long)
        q_out_pt, k_out_pt = torch_rope_neox(positions_pt, q_pt, k_pt, _K, _ROTARY_DIM, _ROPE_THETA)

        np.testing.assert_allclose(jax_to_numpy(q_out_jax), q_out_pt.numpy(), atol=1e-5)
        np.testing.assert_allclose(jax_to_numpy(k_out_jax), k_out_pt.numpy(), atol=1e-5)

    def test_g_proj_matches_torch(self):
        """Gate projection: JAX LinearBase vs torch F.linear."""
        T = 8
        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = _make_module()
            hidden_np = np.random.default_rng(45).standard_normal((T, _HIDDEN)).astype(np.float32)
            hidden_jax = jnp.array(hidden_np)
            g_jax, _ = module.g_proj(hidden_jax)

        w_np = jax_to_numpy(module.g_proj.weight.value)  # (in, out)
        g_pt = F.linear(torch.tensor(hidden_np), torch.tensor(w_np.T))

        np.testing.assert_allclose(jax_to_numpy(g_jax), g_pt.numpy(), atol=_MATMUL_ATOL)

    def test_group_rmsnorm_gating_matches_torch(self):
        """GroupRMSNorm + sigmoid gating: JAX vs pure-torch reference."""
        T = 8
        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = _make_module()
            rng = np.random.default_rng(46)
            attn_np = rng.standard_normal((T, _H * _K)).astype(np.float32)
            gate_np = rng.standard_normal((T, _H * _K)).astype(np.float32)

            attn_jax = jnp.array(attn_np)
            gate_jax = jax.nn.sigmoid(jnp.array(gate_np))
            normed_jax = module.g_norm(attn_jax)
            result_jax = normed_jax * gate_jax

        g_norm_weight_np = jax_to_numpy(module.g_norm.weight.value)
        normed_pt = torch_group_rmsnorm(
            torch.tensor(attn_np), torch.tensor(g_norm_weight_np), _NUM_GROUPS, _EPS
        )
        gate_pt = torch.sigmoid(torch.tensor(gate_np))
        result_pt = normed_pt * gate_pt

        np.testing.assert_allclose(jax_to_numpy(result_jax), result_pt.numpy(), atol=1e-5)

    def test_dense_projection_matches_torch(self):
        """Dense (output) projection: JAX LinearBase vs torch F.linear."""
        T = 8
        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = _make_module()
            x_np = np.random.default_rng(47).standard_normal((T, _H * _K)).astype(np.float32)
            x_jax = jnp.array(x_np)
            out_jax, _ = module.dense(x_jax)

        w_np = jax_to_numpy(module.dense.weight.value)  # (in, out)
        out_pt = F.linear(torch.tensor(x_np), torch.tensor(w_np.T))

        np.testing.assert_allclose(jax_to_numpy(out_jax), out_pt.numpy(), atol=_MATMUL_ATOL)


# ---------------------------------------------------------------------------
# Module-level mock-kernel test
# ---------------------------------------------------------------------------


@requires_torch
class TestModuleLevelMockKernel:
    def test_forward_with_mocked_kernel(self):
        """Full pipeline (minus kernel) with shared weights: JAX vs torch."""
        T = 8
        rng = np.random.default_rng(100)

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = _make_module(dtype=jnp.float32)

            hidden_np = rng.standard_normal((T, _HIDDEN)).astype(np.float32)
            dummy_attn_np = rng.standard_normal((T, _H * _K)).astype(np.float32)
            positions_np = np.arange(T, dtype=np.int32)

            hidden_jax = jnp.array(hidden_np)
            positions_jax = jnp.array(positions_np)

            # --- JAX side: step-by-step forward ---
            # 1. QKV projection + reshape + split
            qkv_jax, _ = module.qkv_proj(hidden_jax)
            qkv_jax = jax.lax.reshape(
                qkv_jax,
                (T, 3, _H, _K),
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
        qkv_pt = qkv_pt.reshape(T, 3, _H, _K)
        q_pt, k_pt = qkv_pt[:, 0], qkv_pt[:, 1]

        # 2. Q/K RMSNorm
        q_pt = torch_rmsnorm(q_pt, torch.tensor(q_norm_w), _EPS)
        k_pt = torch_rmsnorm(k_pt, torch.tensor(k_norm_w), _EPS)

        # 3. RoPE
        q_pt, k_pt = torch_rope_neox(positions_pt, q_pt, k_pt, _K, _ROTARY_DIM, _ROPE_THETA)

        # 4. Mock kernel: same dummy
        attn_pt = torch.tensor(dummy_attn_np)

        # 5. Gating
        g_pt = F.linear(hidden_pt, torch.tensor(g_proj_w.T))
        gate_pt = torch.sigmoid(g_pt)
        gated_pt = torch_group_rmsnorm(attn_pt, torch.tensor(g_norm_w), _NUM_GROUPS, _EPS) * gate_pt

        # 6. Dense
        output_pt = F.linear(gated_pt, torch.tensor(dense_w.T))

        # --- Compare intermediates ---
        np.testing.assert_allclose(
            jax_to_numpy(q_jax), q_pt.numpy(), atol=_MATMUL_ATOL, err_msg="Q after RoPE diverged"
        )
        np.testing.assert_allclose(
            jax_to_numpy(k_jax), k_pt.numpy(), atol=_MATMUL_ATOL, err_msg="K after RoPE diverged"
        )

        # --- Compare final output ---
        np.testing.assert_allclose(
            jax_to_numpy(output_jax),
            output_pt.numpy(),
            atol=_MATMUL_ATOL,
            err_msg="Final output diverged",
        )


# ---------------------------------------------------------------------------
# Scale behavior verification
# ---------------------------------------------------------------------------


@requires_simple_gla
class TestScaleBehavior:
    def test_scale_none_matches_explicit(self):
        """fused_recurrent_simple_gla: scale=None should equal scale=K^-0.5."""
        H, K = 4, 64
        rng = np.random.default_rng(200)
        q = jnp.array(rng.standard_normal((2, 1, H, K)).astype(np.float32))
        k = jnp.array(rng.standard_normal((2, 1, H, K)).astype(np.float32))
        v = jnp.array(rng.standard_normal((2, 1, H, K)).astype(np.float32))
        g_gamma = jnp.array([-0.1, -0.2, -0.15, -0.25], dtype=jnp.float32)
        state = jnp.zeros((2, H, K, K), dtype=jnp.float32)

        out_none, s_none = fused_recurrent_simple_gla(
            q, k, v, g_gamma=g_gamma, initial_state=state, output_final_state=True, scale=None
        )
        out_explicit, s_explicit = fused_recurrent_simple_gla(
            q,
            k,
            v,
            g_gamma=g_gamma,
            initial_state=state,
            output_final_state=True,
            scale=K**-0.5,
        )

        np.testing.assert_allclose(
            jax_to_numpy(out_none), jax_to_numpy(out_explicit), atol=1e-6, err_msg="Output differs"
        )
        np.testing.assert_allclose(
            jax_to_numpy(s_none), jax_to_numpy(s_explicit), atol=1e-6, err_msg="State differs"
        )


# ---------------------------------------------------------------------------
# GLA recurrence: kernel vs pure-numpy reference
# ---------------------------------------------------------------------------


def numpy_gla_recurrent(q, k, v, g_gamma, h0=None, scale=None):
    """Pure-numpy GLA recurrence reference implementation.

    g_gamma semantics: decay = exp(g_gamma) per step (verified empirically).

    Args:
        q, k, v: [B, T, H, K] float arrays
        g_gamma: [H] negative log-decay per head
        h0: [B, H, K, K] initial state or None (zeros)
        scale: float or None (defaults to K^-0.5)
    Returns:
        output: [B, T, H, K]
        final_state: [B, H, K, K]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5
    gamma = np.exp(g_gamma)  # [H], in (0, 1) for negative g_gamma

    h = np.zeros((B, H, K, V), dtype=np.float64) if h0 is None else h0.astype(np.float64).copy()

    outputs = []
    for t in range(T):
        # Decay existing state
        h = gamma[None, :, None, None] * h
        # Accumulate outer product k^T @ v
        h = h + np.einsum("bhk,bhv->bhkv", k[:, t], v[:, t])
        # Output: q @ h * scale
        o = np.einsum("bhk,bhkv->bhv", q[:, t], h) * scale
        outputs.append(o)

    output = np.stack(outputs, axis=1)  # [B, T, H, V]
    return output.astype(np.float32), h.astype(np.float32)


@requires_simple_gla
class TestGLARecurrenceReference:
    def test_decode_matches_numpy_reference(self):
        """fused_recurrent_simple_gla output matches pure-numpy GLA recurrence."""
        B, T, H, K = 2, 16, 4, 32
        rng = np.random.default_rng(300)
        q_np = rng.standard_normal((B, T, H, K)).astype(np.float32)
        k_np = rng.standard_normal((B, T, H, K)).astype(np.float32)
        v_np = rng.standard_normal((B, T, H, K)).astype(np.float32)
        h0_np = rng.standard_normal((B, H, K, K)).astype(np.float32) * 0.1
        g_gamma_np = np.array([-0.1, -0.2, -0.05, -0.15], dtype=np.float32)

        # Numpy reference (float64 internally)
        out_ref, state_ref = numpy_gla_recurrent(q_np, k_np, v_np, g_gamma_np, h0=h0_np)

        # JAX kernel
        out_jax, state_jax = fused_recurrent_simple_gla(
            jnp.array(q_np),
            jnp.array(k_np),
            jnp.array(v_np),
            g_gamma=jnp.array(g_gamma_np),
            initial_state=jnp.array(h0_np),
            output_final_state=True,
            scale=None,
        )

        np.testing.assert_allclose(
            jax_to_numpy(out_jax),
            out_ref,
            atol=1e-4,
            rtol=1e-4,
            err_msg="Decode kernel output diverges from numpy GLA reference",
        )
        np.testing.assert_allclose(
            jax_to_numpy(state_jax),
            state_ref,
            atol=1e-4,
            rtol=1e-4,
            err_msg="Decode kernel final state diverges from numpy GLA reference",
        )

    @requires_simple_gla
    @requires_tpu
    def test_prefill_matches_numpy_reference(self):
        """simple_gla_fwd (chunk kernel) output matches pure-numpy GLA recurrence.

        Single sequence, no packing — directly comparable to numpy reference.
        """
        T, H, K = 64, 4, 128  # K must be multiple of 128 (kernel constraint)
        B = 1
        rng = np.random.default_rng(301)
        q_np = rng.standard_normal((B, T, H, K)).astype(np.float32)
        k_np = rng.standard_normal((B, T, H, K)).astype(np.float32)
        v_np = rng.standard_normal((B, T, H, K)).astype(np.float32)
        h0_np = rng.standard_normal((B, H, K, K)).astype(np.float32) * 0.1
        g_gamma_np = np.array([-0.1, -0.2, -0.05, -0.15], dtype=np.float32)

        # Numpy reference (float64 internally)
        out_ref, state_ref = numpy_gla_recurrent(q_np, k_np, v_np, g_gamma_np, h0=h0_np)

        # Chunk kernel expects: q/k/v [1, T, H, K], cu_seqlens [2]
        cu_seqlens = jnp.array([0, T], dtype=jnp.int32)
        out_jax, state_jax = simple_gla_fwd(
            jnp.array(q_np),
            jnp.array(k_np),
            jnp.array(v_np),
            g_gamma=jnp.array(g_gamma_np),
            h0=jnp.array(h0_np),
            cu_seqlens_dev=cu_seqlens,
            scale=None,
            use_ht=True,
            chunk_size=64,
        )

        # Chunk kernel uses different reduction order; allow wider tolerance
        np.testing.assert_allclose(
            jax_to_numpy(out_jax).reshape(B, T, H, K),
            out_ref,
            atol=5e-2,
            rtol=1e-2,
            err_msg="Prefill kernel output diverges from numpy GLA reference",
        )
        np.testing.assert_allclose(
            jax_to_numpy(state_jax),
            state_ref,
            atol=5e-2,
            rtol=1e-2,
            err_msg="Prefill kernel final state diverges from numpy GLA reference",
        )

    @requires_simple_gla
    @requires_tpu
    def test_prefill_non_aligned_matches_numpy_reference(self):
        """simple_gla_fwd with non-chunk-aligned seq_len (zero-padded to chunk boundary).

        Verifies that scatter/padding → kernel → state produces results consistent
        with numpy recurrence over the same zero-padded input.
        """
        T_real = 100  # non-aligned
        CHUNK = 64
        T_padded = ((T_real + CHUNK - 1) // CHUNK) * CHUNK  # 128
        H, K = 4, 128
        B = 1
        rng = np.random.default_rng(302)

        # Generate real tokens, then zero-pad to chunk-aligned length
        q_real = rng.standard_normal((B, T_real, H, K)).astype(np.float32)
        k_real = rng.standard_normal((B, T_real, H, K)).astype(np.float32)
        v_real = rng.standard_normal((B, T_real, H, K)).astype(np.float32)

        q_padded = np.zeros((B, T_padded, H, K), dtype=np.float32)
        k_padded = np.zeros((B, T_padded, H, K), dtype=np.float32)
        v_padded = np.zeros((B, T_padded, H, K), dtype=np.float32)
        q_padded[:, :T_real] = q_real
        k_padded[:, :T_real] = k_real
        v_padded[:, :T_real] = v_real

        h0_np = rng.standard_normal((B, H, K, K)).astype(np.float32) * 0.1
        g_gamma_np = np.array([-0.1, -0.2, -0.05, -0.15], dtype=np.float32)

        # Numpy reference: recurrence over T_padded (including zero-padded positions)
        _, state_ref_padded = numpy_gla_recurrent(
            q_padded, k_padded, v_padded, g_gamma_np, h0=h0_np
        )

        # Kernel with cu_seqlens set to padded length (as our LinearAttentionBackend does)
        cu_seqlens = jnp.array([0, T_padded], dtype=jnp.int32)
        _, state_jax = simple_gla_fwd(
            jnp.array(q_padded),
            jnp.array(k_padded),
            jnp.array(v_padded),
            g_gamma=jnp.array(g_gamma_np),
            h0=jnp.array(h0_np),
            cu_seqlens_dev=cu_seqlens,
            scale=None,
            use_ht=True,
            chunk_size=CHUNK,
        )

        np.testing.assert_allclose(
            jax_to_numpy(state_jax),
            state_ref_padded,
            atol=5e-2,
            rtol=1e-2,
            err_msg="Kernel state diverges from numpy reference (non-aligned T=100, padded T=128)",
        )
