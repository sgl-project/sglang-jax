"""Qwen3 fused RMSNorm + QKV and merged MLP gate/up on a CPU (data, tensor) mesh.

``QWen3Attention.forward_fused_rmsnorm_qkv`` replaces the q/k/v ``LinearBase``
calls with one ``MergedColumnParallelLinear.shard_local_merged_column_dot`` for
every token count and layer. It used to pick separate dots (``T > 64`` or
``layer_id < 6``) or the same shard-local merged dot (otherwise), so the cases
still straddle those thresholds. Each must produce the same q/k/v as
``QWen3Attention.__call__`` on the normed input, including the attention bias,
and its compiled tp=8 program must not move q/k/v weights between devices.

``Qwen3MLP`` now computes gate/up with the same merged dot instead of separate
dots. Its output must match separate ``gate_proj``/``up_proj`` projections at
tp in {1, 4, 8} and dp2/tp4, from decode (``T == 1``) to prefill (``T > 64``),
in f32 and bf16. Neither the compiled MLP nor a compiled decoder layer may move
data between devices beyond the all-reduces that separate projections need.

The attention kernel is replaced with a stub that records the q/k/v it is fed.
"""

import os
import re

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
from sgl_jax.srt.models.qwen3 import QWen3Attention, QWen3DecoderLayer, Qwen3MLP

HIDDEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 256, 8, 4, 32
INTERMEDIATE = 512
ROPE_THETA = 1e6
EPS = 1e-6

COLLECTIVES = ("all-to-all", "all-gather", "collective-permute", "all-reduce", "reduce-scatter")


def _mesh(tp, dp=1):
    devs = jax.devices()
    n = dp * tp
    assert len(devs) >= n, f"need {n} CPU devices, got {len(devs)} (XLA_FLAGS not picked up?)"
    return jax.make_mesh(
        (dp, tp),
        ("data", "tensor"),
        devices=devs[:n],
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )


def _collective_counts(hlo):
    """Number of each cross-device collective (sync or async start) in compiled HLO."""
    return {op: len(re.findall(rf"\s{op}(-start)?\(", hlo)) for op in COLLECTIVES}


def _kv_live_backend(q, k, v, layer, forward_batch, token_to_kv_pool, **kwargs):
    # Keep k and v live; they are head-sharded like the result.
    return q.reshape(q.shape[0], -1), jnp.concatenate([k, v], axis=-1)


def _layer_config(num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, intermediate_size=128):
    return SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        rms_norm_eps=EPS,
        attention_bias=False,
        intermediate_size=intermediate_size,
        max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "rope_theta": ROPE_THETA},
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
# (layer_id < 6, T > 64) of forward_fused_rmsnorm_qkv, plus decode (T == 1).
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
    forward_batch = SimpleNamespace(attn_backend=_kv_live_backend)
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
    config = _layer_config()
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


def _mlp(mesh, dtype, seed):
    """Qwen3MLP with distinct random gate/up/down weights.

    LinearBase draws every weight from PRNGKey(0), so gate_proj and up_proj
    would start out identical and a gate/up mix-up would not change the output.
    Scaling by 1/sqrt(fan_in) keeps every activation O(1).
    """
    mlp = Qwen3MLP(HIDDEN, INTERMEDIATE, mesh, dtype=dtype)
    rng = np.random.default_rng(seed)
    for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        w = proj.weight.get_value()
        proj.weight.set_value(
            jax.device_put(
                (rng.standard_normal(w.shape) / np.sqrt(w.shape[0])).astype(w.dtype),
                NamedSharding(mesh, w.sharding.spec),
            )
        )
    return mlp


def _mlp_separate_projections(mlp, x):
    """Qwen3MLP as on main: separate gate_proj and up_proj LinearBase calls."""
    gate, _ = mlp.gate_proj(x)
    up, _ = mlp.up_proj(x)
    out, _ = mlp.down_proj(up * jax.nn.silu(gate))
    return out


def _mlp_cases(layouts):
    return [
        pytest.param(dp, tp, num_tokens, id=f"dp{dp}-tp{tp}-T{num_tokens}")
        for dp, tp in layouts
        for num_tokens in MLP_TOKENS
        if num_tokens % dp == 0
    ]


# Decode, small and mid batches, and prefill-sized T > 64. dp2 needs even T.
MLP_TOKENS = (1, 2, 8, 64, 65, 128)
# tp=1 takes the plain-concat fallback of shard_local_merged_column_dot; every
# tp > 1 layout here takes the shard-local path at every token count.
SHARDED_LAYOUTS = [(1, 8), (2, 4)]
MLP_LAYOUTS = [(1, 1), (1, 4), *SHARDED_LAYOUTS]

# f32: the two paths differ only in the order of f32 partial sums (per-shard
# dots, the down_proj all-reduce), about 1e-6 here; 1e-4 matches the q/k/v test.
# bf16 has 8 significant bits (eps 2**-7). The merged dot accumulates in f32 and
# rounds once, while the reference LinearBase dots ask for bf16 output, so a
# gate/up value, a down_proj partial sum or an output can round one ulp apart.
# The outputs here stay below 4, where a bf16 ulp is at most 2**-6; allow two
# (1 ulp is observed). A gate/up swap moves the outputs by about 0.4 on average.
MLP_TOL = {
    jnp.float32: dict(rtol=1e-4, atol=1e-4),
    jnp.bfloat16: dict(rtol=2**-5, atol=2**-5),
}


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("dp,tp,num_tokens", _mlp_cases(MLP_LAYOUTS))
def test_mlp_merged_gate_up_matches_separate_projections(dp, tp, num_tokens, dtype):
    mesh = _mesh(tp, dp=dp)
    with jax.set_mesh(mesh):
        mlp = _mlp(mesh, dtype, seed=num_tokens)
        x = _hidden(num_tokens, seed=num_tokens).astype(dtype)

        # jit as in serving; one compile per case is also faster than eager.
        out = nnx.jit(lambda mlp, x: mlp(x))(mlp, x)
        ref = nnx.jit(_mlp_separate_projections)(mlp, x)

    assert out.shape == ref.shape == (num_tokens, HIDDEN)
    assert out.dtype == ref.dtype == dtype
    np.testing.assert_allclose(
        np.asarray(out.astype(jnp.float32)),
        np.asarray(ref.astype(jnp.float32)),
        **MLP_TOL[dtype],
    )


@pytest.mark.parametrize("dp,tp,num_tokens", _mlp_cases(SHARDED_LAYOUTS))
def test_mlp_merged_gate_up_moves_no_weights_between_devices(dp, tp, num_tokens):
    """The compiled MLP has no collective beyond those of separate projections.

    Both only all-reduce the down_proj output. A global concat of the
    column-sharded gate/up weights makes the SPMD partitioner all-to-all the
    weights and collective-permute the split outputs on every call.
    """
    mesh = _mesh(tp, dp=dp)
    with jax.set_mesh(mesh):
        mlp = Qwen3MLP(HIDDEN, INTERMEDIATE, mesh, dtype=jnp.bfloat16)
        x = _hidden(num_tokens).astype(jnp.bfloat16)

        # The module is a jit argument, so the weights stay runtime inputs.
        merged = nnx.jit(lambda mlp, x: mlp(x)).lower(mlp, x).compile().as_text()
        separate = nnx.jit(_mlp_separate_projections).lower(mlp, x).compile().as_text()

    got, ref = _collective_counts(merged), _collective_counts(separate)
    extra = {op: got[op] - ref[op] for op in COLLECTIVES if got[op] > ref[op]}
    assert not extra, f"merged gate/up adds collectives {extra} (merged {got}, separate {ref})"


@pytest.mark.parametrize("dp,tp,num_tokens", _mlp_cases(SHARDED_LAYOUTS))
def test_decoder_layer_moves_no_weights_between_devices(dp, tp, num_tokens):
    """A compiled decoder layer may only all-reduce the o_proj and down_proj outputs."""
    mesh = _mesh(tp, dp=dp)
    forward_batch = SimpleNamespace(attn_backend=_kv_live_backend)
    config = _layer_config(num_heads=16, num_kv_heads=8, intermediate_size=INTERMEDIATE)
    with jax.set_mesh(mesh):
        layer = QWen3DecoderLayer(config, mesh, layer_id=6, dtype=jnp.bfloat16)
        x = _hidden(num_tokens).astype(jnp.bfloat16)
        positions = jnp.arange(num_tokens, dtype=jnp.int32)

        @nnx.jit
        def step(layer, x, positions):
            cos_sin = layer.self_attn.rotary_emb.get_cos_sin(positions)
            return layer(positions, x, forward_batch, None, cos_sin=cos_sin)

        hlo = step.lower(layer, x, positions).compile().as_text()

    counts = _collective_counts(hlo)
    found = {op: n for op, n in counts.items() if n and op != "all-reduce"}
    assert not found, f"compiled decoder layer moves data between devices: {counts}"
