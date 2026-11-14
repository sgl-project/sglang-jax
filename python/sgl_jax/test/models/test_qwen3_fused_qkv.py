"""Qwen3 fused RMSNorm + QKV projection on a CPU (data=1, tensor=tp) mesh.

``QWen3Attention.forward_fused_rmsnorm_qkv`` replaces the q/k/v ``LinearBase``
calls with one ``MergedColumnParallelLinear.shard_local_merged_column_dot`` for
every token count and layer. It used to pick separate dots (``T > 64`` or
``layer_id < 6``), a shard-local merged dot (``T == 1``) or a
concatenated-weight dot (otherwise), so the cases still straddle those
thresholds. Each must produce the same q/k/v as ``QWen3Attention.__call__`` on
the normed input, including the attention bias, and its compiled tp=8 program
must not move q/k/v weights between devices.

The attention kernel is replaced with a stub that records the q/k/v it is fed.
"""

import os

# MUST precede any jax import.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.models import qwen3
from sgl_jax.srt.models.qwen3 import QWen3Attention, QWen3DecoderLayer

HIDDEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 256, 8, 4, 32
ROPE_THETA = 1e6
EPS = 1e-6


def _mesh(tp):
    devs = jax.devices()
    assert len(devs) >= tp, f"need {tp} CPU devices, got {len(devs)} (XLA_FLAGS not picked up?)"
    return jax.make_mesh(
        (1, tp),
        ("data", "tensor"),
        devices=devs[:tp],
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )


class _CaptureAttn:
    """Stands in for RadixAttention: records q/k/v, returns q flattened."""

    def __init__(self):
        self.qkv = None

    def __call__(self, q, k, v, forward_batch, token_to_kv_pool):
        self.qkv = tuple(np.asarray(x) for x in (q, k, v))
        return q.reshape(q.shape[0], -1), None


def _randomize_biases(mesh, attn, seed):
    rng = np.random.default_rng(seed)
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
        bias = proj.bias.get_value()
        proj.bias.set_value(
            jax.device_put(
                rng.uniform(-2.0, 2.0, bias.shape).astype(bias.dtype),
                NamedSharding(mesh, bias.sharding.spec),
            )
        )


def _attention(
    mesh,
    layer_id,
    bias,
    num_heads=NUM_HEADS,
    num_kv_heads=NUM_KV_HEADS,
    dtype=jnp.float32,
    capture=True,
):
    attn = QWen3Attention(
        HIDDEN,
        num_heads,
        num_kv_heads,
        4096,
        mesh,
        rope_theta=ROPE_THETA,
        head_dim=HEAD_DIM,
        rms_norm_eps=EPS,
        layer_id=layer_id,
        attention_bias=bias,
        dtype=dtype,
    )
    if bias:
        _randomize_biases(mesh, attn, seed=layer_id)
    if capture:
        attn.attn = _CaptureAttn()
    return attn


def _hidden(num_tokens, seed=0):
    return jax.random.normal(
        jax.random.PRNGKey(seed),
        (num_tokens, HIDDEN),
        jnp.float32,
        out_sharding=P("data", None),
    )


# (layer_id, num_tokens) pairs on each side of the former branch thresholds
# (layer_id < 6, T == 1, T > 64) of forward_fused_rmsnorm_qkv.
BRANCH_CASES = [
    pytest.param(0, 8, id="layer0-T8"),
    pytest.param(6, 65, id="layer6-T65"),
    pytest.param(6, 1, id="layer6-T1"),
    pytest.param(6, 2, id="layer6-T2"),
    pytest.param(6, 8, id="layer6-T8"),
    pytest.param(20, 64, id="layer20-T64"),
]


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize("layer_id,num_tokens", BRANCH_CASES)
def test_fused_qkv_matches_projection_path(tp, bias, layer_id, num_tokens):
    mesh = _mesh(tp)
    with jax.set_mesh(mesh):
        attn = _attention(mesh, layer_id, bias)
        ln = RMSNorm(HIDDEN, epsilon=EPS, param_dtype=jnp.float32, scope_name="ln")
        x = _hidden(num_tokens, seed=num_tokens)
        positions = jnp.arange(num_tokens, dtype=jnp.int32) + 7

        out_fused, _ = attn.forward_fused_rmsnorm_qkv(x, ln, positions, None, None)
        q_f, k_f, v_f = attn.attn.qkv
        out_ref, _ = attn(positions, ln(x), None, None)
        q_r, k_r, v_r = attn.attn.qkv

    tol = dict(rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(q_f, q_r, **tol, err_msg="q")
    np.testing.assert_allclose(k_f, k_r, **tol, err_msg="k")
    np.testing.assert_allclose(v_f, v_r, **tol, err_msg="v")
    np.testing.assert_allclose(np.asarray(out_fused), np.asarray(out_ref), **tol)


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize("layer_id,num_tokens", BRANCH_CASES)
def test_fused_qkv_moves_no_weights_between_devices(bias, layer_id, num_tokens):
    """At tp=8 the compiled step may only all-reduce the o_proj output.

    A concat of the column-sharded q/k/v weights along their sharded axis makes
    the SPMD partitioner all-to-all the weights and collective-permute the
    split outputs on every call.
    """
    mesh = _mesh(8)

    def backend(q, k, v, layer, forward_batch, token_to_kv_pool, **kwargs):
        # Keep k and v live; they are head-sharded like the result.
        return q.reshape(q.shape[0], -1), jnp.concatenate([k, v], axis=-1)

    forward_batch = SimpleNamespace(attn_backend=backend)
    with jax.set_mesh(mesh):
        attn = _attention(
            mesh,
            layer_id,
            bias,
            num_heads=16,
            num_kv_heads=8,
            dtype=jnp.bfloat16,
            capture=False,
        )
        ln = RMSNorm(HIDDEN, epsilon=EPS, param_dtype=jnp.bfloat16, scope_name="ln")
        x = _hidden(num_tokens).astype(jnp.bfloat16)
        positions = jnp.arange(num_tokens, dtype=jnp.int32)

        # The modules are jit arguments, so the weights stay runtime inputs.
        @nnx.jit
        def step(attn, ln, x, positions):
            return attn.forward_fused_rmsnorm_qkv(x, ln, positions, forward_batch, None)

        hlo = step.lower(attn, ln, x, positions).compile().as_text()

    found = [
        op
        for op in ("all-to-all", "all-gather", "collective-permute", "reduce-scatter")
        if op in hlo
    ]
    assert not found, f"compiled step moves data between devices: {found}"


@pytest.mark.parametrize("layer_id,num_tokens", BRANCH_CASES)
def test_hoisted_cos_sin_matches_rotary_emb(layer_id, num_tokens):
    """QWen3Model precomputes cos/sin once; the layer must apply it like rotary_emb."""
    mesh = _mesh(2)
    with jax.set_mesh(mesh):
        attn = _attention(mesh, layer_id, bias=True)
        ln = RMSNorm(HIDDEN, epsilon=EPS, param_dtype=jnp.float32, scope_name="ln")
        x = _hidden(num_tokens, seed=num_tokens + 1)
        positions = jnp.arange(num_tokens, dtype=jnp.int32) * 3 + 11
        cos, sin = attn.rotary_emb._compute_cos_sin(positions)
        cos_sin = (cos.astype(attn.rotary_emb.dtype), sin.astype(attn.rotary_emb.dtype))

        attn.forward_fused_rmsnorm_qkv(x, ln, positions, None, None, cos_sin=cos_sin)
        q_h, k_h, v_h = attn.attn.qkv
        attn.forward_fused_rmsnorm_qkv(x, ln, positions, None, None)
        q_r, k_r, v_r = attn.attn.qkv

    tol = dict(rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(q_h, q_r, **tol)
    np.testing.assert_allclose(k_h, k_r, **tol)
    np.testing.assert_allclose(v_h, v_r, **tol)


def test_decoder_layer_traces_normed_input(monkeypatch):
    """The precision tracer's INPUT_LAYERNORM record must hold the normed states."""
    mesh = _mesh(2)
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=EPS,
        attention_bias=False,
        intermediate_size=128,
        max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "rope_theta": ROPE_THETA},
    )
    records = {}

    def _record(tensor, name, stage, layer_id=None):
        records[name] = np.asarray(tensor)
        return None

    monkeypatch.setattr(qwen3.precision_tracer, "jit_pure_callback_record", _record)
    with jax.set_mesh(mesh):
        layer = QWen3DecoderLayer(config, mesh, layer_id=0, dtype=jnp.float32)
        scale = layer.input_layernorm.scale.get_value()
        layer.input_layernorm.scale.set_value(
            jax.device_put(
                np.linspace(0.5, 1.5, HIDDEN, dtype=np.float32),
                NamedSharding(mesh, scale.sharding.spec),
            )
        )
        layer.self_attn.attn = _CaptureAttn()
        x = _hidden(8, seed=3)
        positions = jnp.arange(8, dtype=jnp.int32)
        layer(positions, x, None, None)
        expected = np.asarray(layer.input_layernorm(x))

    np.testing.assert_allclose(records["input_layernorm_output"], expected, rtol=1e-6, atol=1e-6)


def test__token_parallel_prefill_enabled_gating(monkeypatch):
    """Verify token-parallel prefill gating (_token_parallel_prefill_enabled / _use_token_parallel)."""
    from sgl_jax.srt.environ import envs
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
    from sgl_jax.srt.models.qwen3 import TOKEN_PARALLEL_MIN_TOKENS, QWen3Model

    mesh = _mesh(2)
    config = SimpleNamespace(
        vocab_size=128,
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=EPS,
        attention_bias=False,
        intermediate_size=128,
        max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "rope_theta": ROPE_THETA},
        tie_word_embeddings=False,
    )
    with jax.set_mesh(mesh):
        model = QWen3Model(config, mesh=mesh, dtype=jnp.float32)
        assert model.supports_token_parallel()
        model.init_token_parallel_weights()
        assert model.token_parallel_ready

        num_tokens = TOKEN_PARALLEL_MIN_TOKENS
        fb_extend = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            input_ids=jnp.zeros((num_tokens,), dtype=jnp.int32),
            positions=jnp.arange(num_tokens, dtype=jnp.int32),
            spec_info=None,
            input_embedding=None,
            deepstack_visual_embedding=None,
        )
        assert model._use_token_parallel(fb_extend, fb_extend.positions)

        fb_decode = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            input_ids=jnp.zeros((num_tokens,), dtype=jnp.int32),
            positions=jnp.arange(num_tokens, dtype=jnp.int32),
            spec_info=None,
            input_embedding=None,
            deepstack_visual_embedding=None,
        )
        assert not model._use_token_parallel(fb_decode, fb_decode.positions)
        assert envs.SGLANG_JAX_TOKEN_PARALLEL_PREFILL.get() is True


def test__all_gather_tokens_and_qkv_tokens_to_heads():
    """Verify _replicated (_all_gather_tokens) and _qkv_tokens_to_heads / _heads_to_tokens."""
    from sgl_jax.srt.models.qwen3 import (
        _heads_to_tokens,
        _qkv_tokens_to_heads,
        _replicated,
        _token_sharding,
    )

    mesh = _mesh(2)
    with jax.set_mesh(mesh):
        w = jax.device_put(
            jnp.arange(32, dtype=jnp.float32).reshape(4, 8),
            NamedSharding(mesh, P(None, "tensor")),
        )
        rep = _replicated(w, mesh=mesh)
        np.testing.assert_array_equal(
            np.asarray(rep), np.arange(32, dtype=np.float32).reshape(4, 8)
        )

        tok3 = _token_sharding(mesh, 3)
        q = jax.device_put(
            jnp.arange(8 * 4 * 2, dtype=jnp.float32).reshape(8, 4, 2),
            tok3,
        )
        k = jax.device_put(
            jnp.arange(8 * 2 * 2, dtype=jnp.float32).reshape(8, 2, 2),
            tok3,
        )
        v = k + 1.0
        qh, kh, vh = _qkv_tokens_to_heads(q, k, v, mesh)
        np.testing.assert_array_equal(np.asarray(qh), np.asarray(q))
        np.testing.assert_array_equal(np.asarray(kh), np.asarray(k))
        np.testing.assert_array_equal(np.asarray(vh), np.asarray(v))

        flat_q = qh.reshape(8, 4 * 2)
        tok_q = _heads_to_tokens(flat_q, mesh)
        np.testing.assert_array_equal(np.asarray(tok_q), np.asarray(q.reshape(8, 8)))


def test__reduce_scatter_tokens_and_select_token_rows():
    """Verify _select_token_rows (_reduce_scatter_tokens / sampled-row gather) across shards."""
    from sgl_jax.srt.models.qwen3 import _select_token_rows, _token_sharding

    mesh = _mesh(4)
    with jax.set_mesh(mesh):
        x_np = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)
        x = jax.device_put(jnp.asarray(x_np), _token_sharding(mesh, 2))
        indices = jax.device_put(
            jnp.array([0, 3, 7, 11, 15], dtype=jnp.int32),
            NamedSharding(mesh, P("data")),
        )
        selected = _select_token_rows(x, indices, mesh)
        np.testing.assert_array_equal(np.asarray(selected), x_np[[0, 3, 7, 11, 15]])
