"""Unit tests for MLAAttention (Multi-head Latent Attention).

Tests the Q/K/V projection logic against a numpy fp32 reference
that follows the standard MLA computation flow.

With production-sized bf16 matmuls (N=8192), element-wise errors accumulate
to O(sqrt(N)*eps_bf16) ≈ 0.35 per step — too large for meaningful allclose.
Instead, we use cosine similarity, which measures directional agreement and
is invariant to per-element magnitude noise. A threshold of 0.99 means the
bf16 output vector deviates < 8° from the fp32 reference direction.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.mla import MLAAttention
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase

HIDDEN_SIZE = 8192
NUM_HEADS = 64
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
ROPE_THETA = 6000000.0
EPSILON = 1e-6
NUM_TOKENS = 3

# Cosine similarity threshold: empirical min across 100 samples is ~0.998.
# 0.99 provides comfortable margin while still being a meaningful assertion.
COSINE_THRESHOLD = 0.99


def cosine_similarity(a, b):
    """Cosine similarity between two arrays (flattened to 1-D)."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def run_jax_kv_decompress(mla, hidden):
    """KV path (without RoPE): kv_a_proj -> split -> norm -> kv_b_proj -> reshape."""
    kv_a_out, _ = mla.kv_a_proj(hidden)
    compressed = kv_a_out[:, : mla.kv_lora_rank]
    k_rope_raw = np.array(kv_a_out[:, mla.kv_lora_rank :])
    compressed = mla.kv_a_layernorm(compressed)
    kv_out, _ = mla.kv_b_proj(compressed)
    kv_out = kv_out.reshape(-1, mla.num_heads, mla.qk_nope_head_dim + mla.v_head_dim)
    k_nope = np.array(kv_out[:, :, : mla.qk_nope_head_dim])
    v = np.array(kv_out[:, :, mla.qk_nope_head_dim :])
    return k_nope, v, k_rope_raw


def run_jax_qkv(mla, hidden, positions):
    """Full MLA projection, concatenation in numpy to avoid sharding mismatch."""
    q_compressed, _ = mla.q_a_proj(hidden)
    q_compressed = mla.q_a_layernorm(q_compressed)
    q, _ = mla.q_b_proj(q_compressed)
    q = q.reshape(-1, mla.num_heads, mla.qk_head_dim)
    q_nope = np.array(q[:, :, : mla.qk_nope_head_dim])
    q_rope = q[:, :, mla.qk_nope_head_dim :]

    k_nope, v, k_rope_raw = run_jax_kv_decompress(mla, hidden)
    k_rope = jnp.array(k_rope_raw).reshape(-1, 1, mla.qk_rope_head_dim)

    q_rope, k_rope = mla.rotary_emb(positions, q_rope, k_rope)
    k_rope = jnp.broadcast_to(k_rope, (k_rope.shape[0], mla.num_heads, mla.qk_rope_head_dim))

    Q = np.concatenate([q_nope, np.array(q_rope)], axis=-1)
    K = np.concatenate([k_nope, np.array(k_rope)], axis=-1)
    V = v
    return Q, K, V


# Numpy fp32 reference implementations (ground-truth oracle)
# Follows BailingMoeV2_5MultiLatentAttention computation flow, adapted for
# sglang-jax weight layout (weight=(in,out), fwd=x@w).
# Reference: https://huggingface.co/inclusionAI/Ling-2.5-1T/blob/main/modeling_bailing_moe_v2_5.py
def numpy_rmsnorm_fp32(x, scale, eps=1e-6):
    x = x.astype(np.float32)
    scale = scale.astype(np.float32)
    variance = np.mean(x**2, axis=-1, keepdims=True)
    return x / np.sqrt(variance + eps) * scale


def numpy_linear_fp32(x, weight):
    return x.astype(np.float32) @ weight.astype(np.float32)


def numpy_rotary_emb_fp32(x, cos, sin):
    """Interleaved RoPE (is_neox_style=False): pairs (x[0],x[1]), (x[2],x[3]), ..."""
    x = x.astype(np.float32)
    cos = cos.astype(np.float32)[:, np.newaxis, :]
    sin = sin.astype(np.float32)[:, np.newaxis, :]
    x1, x2 = x[..., ::2], x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return np.stack((o1, o2), axis=-1).reshape(x.shape)


def numpy_mla_qkv_fp32(hidden, positions, weights, config):
    num_heads = config["num_heads"]
    qk_nope_head_dim = config["qk_nope_head_dim"]
    qk_rope_head_dim = config["qk_rope_head_dim"]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    v_head_dim = config["v_head_dim"]
    kv_lora_rank = config["kv_lora_rank"]
    rope_theta = config["rope_theta"]
    eps = config.get("epsilon", 1e-6)
    tokens = hidden.shape[0]

    # Q path
    q = numpy_linear_fp32(hidden, weights["q_a_proj"])
    q = numpy_rmsnorm_fp32(q, weights["q_a_layernorm_scale"], eps)
    q = numpy_linear_fp32(q, weights["q_b_proj"])
    q = q.reshape(tokens, num_heads, qk_head_dim)
    q_nope = q[:, :, :qk_nope_head_dim]
    q_rope = q[:, :, qk_nope_head_dim:]

    # KV path
    kv_a_out = numpy_linear_fp32(hidden, weights["kv_a_proj"])
    compressed = kv_a_out[:, :kv_lora_rank]
    k_rope_raw = kv_a_out[:, kv_lora_rank:]
    compressed = numpy_rmsnorm_fp32(compressed, weights["kv_a_layernorm_scale"], eps)
    kv_out = numpy_linear_fp32(compressed, weights["kv_b_proj"])
    kv_out = kv_out.reshape(tokens, num_heads, qk_nope_head_dim + v_head_dim)
    k_nope = kv_out[:, :, :qk_nope_head_dim]
    v = kv_out[:, :, qk_nope_head_dim:]
    k_rope = k_rope_raw.reshape(tokens, 1, qk_rope_head_dim)

    # RoPE
    inv_freq = 1.0 / (
        rope_theta ** (np.arange(0, qk_rope_head_dim, 2, dtype=np.float32) / qk_rope_head_dim)
    )
    freqs = np.einsum("n,d->nd", positions.astype(np.float32), inv_freq)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    q_rope = numpy_rotary_emb_fp32(q_rope, cos, sin)
    k_rope = numpy_rotary_emb_fp32(k_rope, cos, sin)
    k_rope = np.broadcast_to(k_rope, (tokens, num_heads, qk_rope_head_dim))

    Q = np.concatenate([q_nope, q_rope], axis=-1)
    K = np.concatenate([k_nope, k_rope.copy()], axis=-1)
    return Q, K, v


class TestMLAAttention(CustomTestCase):
    """Unit tests for MLAAttention Q/K/V projection pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
        jax.sharding.set_mesh(cls.mesh)
        cls.mla = MLAAttention(
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            q_lora_rank=Q_LORA_RANK,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            mesh=cls.mesh,
            layer_id=0,
            rope_theta=ROPE_THETA,
            rope_interleave=True,
            dtype=jnp.bfloat16,
        )
        cls.weights = {
            "q_a_proj": np.array(cls.mla.q_a_proj.weight[...]),
            "q_a_layernorm_scale": np.array(cls.mla.q_a_layernorm.scale[...]),
            "q_b_proj": np.array(cls.mla.q_b_proj.weight[...]),
            "kv_a_proj": np.array(cls.mla.kv_a_proj.weight[...]),
            "kv_a_layernorm_scale": np.array(cls.mla.kv_a_layernorm.scale[...]),
            "kv_b_proj": np.array(cls.mla.kv_b_proj.weight[...]),
        }
        cls.ref_config = {
            "num_heads": NUM_HEADS,
            "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
            "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
            "v_head_dim": V_HEAD_DIM,
            "kv_lora_rank": KV_LORA_RANK,
            "rope_theta": ROPE_THETA,
            "epsilon": EPSILON,
        }
        cls.hidden_np = (
            np.random.default_rng(42).standard_normal((NUM_TOKENS, HIDDEN_SIZE)).astype(np.float32)
        )

    def test_output_shapes(self):
        Q, K, V = run_jax_qkv(
            self.mla, jnp.array(self.hidden_np), jnp.arange(NUM_TOKENS, dtype=jnp.int32)
        )
        self.assertEqual(Q.shape, (NUM_TOKENS, NUM_HEADS, QK_HEAD_DIM))
        self.assertEqual(K.shape, (NUM_TOKENS, NUM_HEADS, QK_HEAD_DIM))
        self.assertEqual(V.shape, (NUM_TOKENS, NUM_HEADS, V_HEAD_DIM))

    def test_q_path(self):
        q_compressed, _ = self.mla.q_a_proj(jnp.array(self.hidden_np))
        q_compressed = self.mla.q_a_layernorm(q_compressed)
        q_jax, _ = self.mla.q_b_proj(q_compressed)
        q_jax = np.array(q_jax.reshape(-1, self.mla.num_heads, self.mla.qk_head_dim))

        q_ref = numpy_linear_fp32(self.hidden_np, self.weights["q_a_proj"])
        q_ref = numpy_rmsnorm_fp32(q_ref, self.weights["q_a_layernorm_scale"])
        q_ref = numpy_linear_fp32(q_ref, self.weights["q_b_proj"])
        q_ref = q_ref.reshape(NUM_TOKENS, NUM_HEADS, QK_HEAD_DIM)

        self.assertGreaterEqual(cosine_similarity(q_jax, q_ref), COSINE_THRESHOLD)

    def test_kv_path(self):
        k_nope_jax, v_jax, k_rope_raw_jax = run_jax_kv_decompress(
            self.mla, jnp.array(self.hidden_np)
        )

        kv_a_ref = numpy_linear_fp32(self.hidden_np, self.weights["kv_a_proj"])
        compressed_ref = numpy_rmsnorm_fp32(
            kv_a_ref[:, :KV_LORA_RANK], self.weights["kv_a_layernorm_scale"]
        )
        kv_ref = numpy_linear_fp32(compressed_ref, self.weights["kv_b_proj"])
        kv_ref = kv_ref.reshape(NUM_TOKENS, NUM_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM)

        self.assertGreaterEqual(
            cosine_similarity(k_nope_jax, kv_ref[:, :, :QK_NOPE_HEAD_DIM]), COSINE_THRESHOLD
        )
        self.assertGreaterEqual(
            cosine_similarity(v_jax, kv_ref[:, :, QK_NOPE_HEAD_DIM:]), COSINE_THRESHOLD
        )
        self.assertGreaterEqual(
            cosine_similarity(k_rope_raw_jax, kv_a_ref[:, KV_LORA_RANK:]), COSINE_THRESHOLD
        )

    def test_full_qkv(self):
        positions_np = np.arange(NUM_TOKENS, dtype=np.int32)

        Q_jax, K_jax, V_jax = run_jax_qkv(
            self.mla, jnp.array(self.hidden_np), jnp.array(positions_np)
        )
        Q_ref, K_ref, V_ref = numpy_mla_qkv_fp32(
            self.hidden_np, positions_np, self.weights, self.ref_config
        )

        self.assertGreaterEqual(cosine_similarity(Q_jax, Q_ref), COSINE_THRESHOLD)
        self.assertGreaterEqual(cosine_similarity(K_jax, K_ref), COSINE_THRESHOLD)
        self.assertGreaterEqual(cosine_similarity(V_jax, V_ref), COSINE_THRESHOLD)

    def test_k_rope_broadcast(self):
        _, K, _ = run_jax_qkv(
            self.mla, jnp.array(self.hidden_np), jnp.arange(NUM_TOKENS, dtype=jnp.int32)
        )
        k_rope = K[:, :, QK_NOPE_HEAD_DIM:]

        for h in range(1, NUM_HEADS):
            np.testing.assert_array_equal(k_rope[:, 0, :], k_rope[:, h, :])


if __name__ == "__main__":
    unittest.main()
