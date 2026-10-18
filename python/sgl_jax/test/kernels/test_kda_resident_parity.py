"""Per-position numerical parity and regression tests for the KDA resident pipeline.

Compares ``resident_pipeline_kda_fwd`` (``pipeline_impl="fused_resident"``) against
the incumbent four-stage ``chunk_kda_fwd`` (``pipeline_impl="incumbent"``) and FP32
``naive_recurrent_kda`` across:
  - Single-chunk, multi-chunk, and unaligned single-sequence lengths.
  - Multi-segment packed varlen batches, including zero-length segments.
  - Non-default chunk sizes (``chunk_size=16`` and ``chunk_size=32``).
  - Bounded (``lower_bound=-5.0``) and unbounded (``lower_bound=None``) gates.
  - In-kernel gate activation (``A_log`` + ``dt_bias``) and pre-activated gates.
  - In-kernel Q/K L2 normalization and caller-normalized Q/K.
  - Padded single-sequence logical boundaries (``cu_seqlens[1] < T``).
  - Default selection-switch contract on ``chunk_kda_fwd`` and ``kda_forward_packed``.
  - TPU AOT compilation on v6e and v7x topologies (including ``H=32`` VMEM cap).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.kda import (
    chunk_kda,
    kda_forward_packed,
    naive_recurrent_kda,
    resident_pipeline_kda_fwd,
)
from sgl_jax.srt.kernels.kda.kda import chunk_kda_fwd

K = 128
V = 128
SCALE = K**-0.5


@pytest.fixture(autouse=True)
def _enable_pallas_interpret_on_cpu(monkeypatch):
    if jax.default_backend() == "cpu":
        monkeypatch.setenv("PALLAS_INTERPRET", "1")


def _l2_normalize(x: jax.Array) -> jax.Array:
    x_f32 = x.astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + 1e-6)
    return (x_f32 * rstd).astype(x.dtype)


def _activate_gate(
    raw_g: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array | None,
    lower_bound: float | None,
) -> jax.Array:
    g_f32 = raw_g.astype(jnp.float32)
    if dt_bias is not None:
        g_f32 = g_f32 + dt_bias.astype(jnp.float32)[None, None, :, :]
    a_scale = jnp.exp(jnp.minimum(a_log.astype(jnp.float32), 80.0))[None, None, :, None]
    if lower_bound is None:
        return -a_scale * jax.nn.softplus(g_f32)
    return lower_bound * jax.nn.sigmoid(a_scale * g_f32)


def _make_inputs(
    lengths: tuple[int, ...],
    heads: int,
    *,
    physical_t: int | None = None,
    seed: int = 2026,
    beta_dtype=jnp.bfloat16,
    with_initial_state: bool = True,
):
    rng = np.random.default_rng(seed)
    logical_t = sum(lengths)
    total_t = physical_t if physical_t is not None else max(1, logical_t)
    assert total_t >= logical_t

    shape_k = (1, total_t, heads, K)
    shape_v = (1, total_t, heads, V)

    def _bf16_normal(shape, std=1.0):
        return jnp.asarray(rng.standard_normal(shape, dtype=np.float32) * std, dtype=jnp.bfloat16)

    q = _bf16_normal(shape_k)
    k = _bf16_normal(shape_k)
    v = _bf16_normal(shape_v, std=1.5)
    raw_g = jnp.asarray(rng.uniform(-4.0, 4.0, shape_k).astype(np.float32), dtype=jnp.bfloat16)
    beta = jnp.asarray(
        rng.uniform(0.05, 0.95, (1, total_t, heads)).astype(np.float32), dtype=beta_dtype
    )
    a_log = jnp.asarray(rng.uniform(0.2, 2.5, (heads,)).astype(np.float32))
    dt_bias = jnp.asarray(rng.uniform(-6.0, -1.0, (heads, K)).astype(np.float32))
    cu_seqlens = jnp.asarray([0, *np.cumsum(lengths, dtype=np.int32)], dtype=jnp.int32)
    initial_state = (
        jnp.asarray(rng.standard_normal((len(lengths), heads, K, V), dtype=np.float32) * 0.1)
        if with_initial_state
        else None
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "raw_g": raw_g,
        "beta": beta,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "cu_seqlens": cu_seqlens,
        "initial_state": initial_state,
        "lengths": lengths,
        "logical_t": logical_t,
    }


def _run_naive_reference(
    case: dict,
    *,
    lower_bound: float | None,
    use_gate_in_kernel: bool,
    use_qk_l2norm_in_kernel: bool,
):
    q = _l2_normalize(case["q"]) if use_qk_l2norm_in_kernel else case["q"]
    k = _l2_normalize(case["k"]) if use_qk_l2norm_in_kernel else case["k"]
    beta = (
        jnp.clip(case["beta"].astype(jnp.float32), 0.0, 1.0)
        if use_qk_l2norm_in_kernel
        else case["beta"]
    )
    if use_gate_in_kernel:
        g_act = _activate_gate(case["raw_g"], case["a_log"], case["dt_bias"], lower_bound)
    else:
        g_act = case["raw_g"].astype(jnp.float32)

    outputs = []
    final_states = []
    offset = 0
    for seg_idx, length in enumerate(case["lengths"]):
        h0_i = (
            None if case["initial_state"] is None else case["initial_state"][seg_idx : seg_idx + 1]
        )
        if length == 0:
            heads = q.shape[2]
            outputs.append(jnp.zeros((1, 0, heads, V), dtype=q.dtype))
            final_states.append(
                jnp.zeros((heads, K, V), dtype=jnp.float32) if h0_i is None else h0_i[0]
            )
            continue
        out_i, state_i = naive_recurrent_kda(
            q[:, offset : offset + length],
            k[:, offset : offset + length],
            case["v"][:, offset : offset + length],
            g_act[:, offset : offset + length],
            beta[:, offset : offset + length],
            scale=SCALE,
            initial_state=h0_i,
            output_final_state=True,
        )
        outputs.append(out_i)
        final_states.append(state_i[0])
        offset += length

    out = (
        jnp.concatenate(outputs, axis=1)
        if outputs
        else jnp.zeros((1, 0, q.shape[2], V), dtype=q.dtype)
    )
    state = jnp.stack(final_states, axis=0)
    return out, state


@pytest.mark.parametrize(
    ("lengths", "heads", "chunk_size", "lower_bound", "safe_gate"),
    [
        ((64,), 8, 64, -5.0, True),
        ((64,), 2, 64, None, True),
        ((63,), 8, 64, -5.0, True),
        ((199,), 4, 64, None, True),
        ((256,), 8, 64, -5.0, True),
        ((128,), 16, 64, -5.0, False),
        ((128,), 32, 64, -5.0, True),
        ((37, 91), 8, 64, -5.0, True),
        ((32, 32), 4, 64, None, True),
        ((16, 16), 1, 16, -5.0, True),
        ((16, 16), 2, 32, -5.0, True),
    ],
)
def test_resident_matches_incumbent_and_naive_per_position(
    lengths: tuple[int, ...],
    heads: int,
    chunk_size: int,
    lower_bound: float | None,
    safe_gate: bool,
):
    """Per-position parity of resident vs. incumbent 4-stage and FP32 naive KDA."""
    case = _make_inputs(
        lengths,
        heads,
        seed=1700 + heads + sum(lengths) + chunk_size,
        beta_dtype=jnp.float32 if lower_bound is None else jnp.bfloat16,
    )
    q_norm = _l2_normalize(case["q"])
    k_norm = _l2_normalize(case["k"])

    inc_out, inc_state, *_ = chunk_kda_fwd(
        q_norm,
        k_norm,
        case["v"],
        case["raw_g"],
        case["beta"],
        SCALE,
        case["initial_state"],
        True,
        case["cu_seqlens"],
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        A_log=case["a_log"],
        dt_bias=case["dt_bias"],
        pipeline_impl="incumbent",
    )
    res_out, res_state, *_ = chunk_kda_fwd(
        q_norm,
        k_norm,
        case["v"],
        case["raw_g"],
        case["beta"],
        SCALE,
        case["initial_state"],
        True,
        case["cu_seqlens"],
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        A_log=case["a_log"],
        dt_bias=case["dt_bias"],
        pipeline_impl="fused_resident",
    )
    naive_out, naive_state = _run_naive_reference(
        case,
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
    )
    jax.block_until_ready((inc_out, inc_state, res_out, res_state, naive_out, naive_state))

    res_o_f32 = np.asarray(res_out, dtype=np.float32)
    inc_o_f32 = np.asarray(inc_out, dtype=np.float32)
    naive_o_f32 = np.asarray(naive_out, dtype=np.float32)
    res_s_f32 = np.asarray(res_state, dtype=np.float32)
    inc_s_f32 = np.asarray(inc_state, dtype=np.float32)
    naive_s_f32 = np.asarray(naive_state, dtype=np.float32)

    assert np.isfinite(res_o_f32).all()
    assert np.isfinite(res_s_f32).all()
    np.testing.assert_allclose(res_o_f32, inc_o_f32, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(res_o_f32, naive_o_f32, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(res_s_f32, inc_s_f32, rtol=3e-2, atol=3e-2)
    np.testing.assert_allclose(res_s_f32, naive_s_f32, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_padded_single_seq_preserves_logical_boundary(lower_bound: float | None):
    """Padded tail (cu_seqlens=[0, 63] inside T=128) must not decay final state."""
    heads = 2
    physical_t = 128
    logical_t = 63
    cu_seqlens = jnp.asarray([0, logical_t], dtype=jnp.int32)
    zeros_4d = jnp.zeros((1, physical_t, heads, K), dtype=jnp.bfloat16)
    zeros_beta = jnp.zeros(
        (1, physical_t, heads),
        dtype=jnp.float32 if lower_bound is None else jnp.bfloat16,
    )
    if lower_bound is None:
        a_log = jnp.full((heads,), math.log(0.01), dtype=jnp.float32)
        dt_bias = jnp.zeros((heads, K), dtype=jnp.float32)
    else:
        a_log = jnp.zeros((heads,), dtype=jnp.float32)
        dt_bias = jnp.full((heads, K), -5.0, dtype=jnp.float32)
    h0 = jnp.broadcast_to(jnp.eye(K, V, dtype=jnp.float32), (1, heads, K, V))

    res_out, res_state, *_ = resident_pipeline_kda_fwd(
        zeros_4d,
        zeros_4d,
        zeros_4d,
        zeros_4d,
        zeros_beta,
        SCALE,
        h0,
        True,
        cu_seqlens,
        chunk_size=64,
        safe_gate=True,
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        A_log=a_log,
        dt_bias=dt_bias,
    )
    g_act = _activate_gate(zeros_4d[:, :logical_t], a_log, dt_bias, lower_bound)
    ref_out, ref_state = naive_recurrent_kda(
        zeros_4d[:, :logical_t],
        zeros_4d[:, :logical_t],
        zeros_4d[:, :logical_t],
        g_act,
        zeros_beta[:, :logical_t],
        scale=SCALE,
        initial_state=h0,
        output_final_state=True,
    )
    jax.block_until_ready((res_out, res_state, ref_out, ref_state))

    np.testing.assert_allclose(np.asarray(res_out[:, :logical_t]), np.asarray(ref_out), atol=1e-6)
    np.testing.assert_allclose(np.asarray(res_state), np.asarray(ref_state), rtol=1e-4, atol=1e-4)
    expected_diag = 0.6461756 if lower_bound is None else 0.1214505
    assert abs(float(res_state[0, 0, 0, 0]) - expected_diag) < 2e-4


def test_zero_length_segments_and_single_chunk_state_elision():
    """Verify zero-length requests and single-chunk zero-initial-state elision."""
    # 1. Single-chunk zero-initial-state elision (N=1, T=64, h0=None, output_final_state=False)
    case_elide = _make_inputs((64,), 8, with_initial_state=False, seed=3101)
    q_norm = _l2_normalize(case_elide["q"])
    k_norm = _l2_normalize(case_elide["k"])
    res_o, res_ht, *_ = resident_pipeline_kda_fwd(
        q_norm,
        k_norm,
        case_elide["v"],
        case_elide["raw_g"],
        case_elide["beta"],
        SCALE,
        None,
        False,
        case_elide["cu_seqlens"],
        chunk_size=64,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        A_log=case_elide["a_log"],
        dt_bias=case_elide["dt_bias"],
    )
    naive_o, _ = _run_naive_reference(
        case_elide,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert res_ht is None
    np.testing.assert_allclose(
        np.asarray(res_o, dtype=np.float32),
        np.asarray(naive_o, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )

    # 2. Packed varlen with a zero-length segment: lengths = (0, 32, 32)
    case_zero_seg = _make_inputs((0, 32, 32), 4, seed=3102)
    res_o2, res_ht2, *_ = resident_pipeline_kda_fwd(
        case_zero_seg["q"],
        case_zero_seg["k"],
        case_zero_seg["v"],
        case_zero_seg["raw_g"],
        case_zero_seg["beta"],
        SCALE,
        case_zero_seg["initial_state"],
        True,
        case_zero_seg["cu_seqlens"],
        chunk_size=64,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        A_log=case_zero_seg["a_log"],
        dt_bias=case_zero_seg["dt_bias"],
    )
    naive_o2, naive_ht2 = _run_naive_reference(
        case_zero_seg,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
    )
    # Empty segment 0 must preserve its initial state unchanged.
    np.testing.assert_allclose(
        np.asarray(res_ht2[0]),
        np.asarray(case_zero_seg["initial_state"][0]),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(res_o2, dtype=np.float32),
        np.asarray(naive_o2, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(res_ht2, dtype=np.float32),
        np.asarray(naive_ht2, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


def test_selection_switch_contract_and_defaults():
    """Default pipeline_impl is 'incumbent' on chunk_kda and kda_forward_packed."""
    case = _make_inputs((64,), 8, seed=4001)
    q_norm = _l2_normalize(case["q"])
    k_norm = _l2_normalize(case["k"])

    # chunk_kda default vs explicit incumbent must be bit-identical
    default_out, default_state, *_ = chunk_kda(
        q_norm,
        k_norm,
        case["v"],
        case["raw_g"],
        case["beta"],
        SCALE,
        case["initial_state"],
        True,
        case["cu_seqlens"],
        use_gate_in_kernel=True,
        A_log=case["a_log"],
        dt_bias=case["dt_bias"],
        lower_bound=-5.0,
    )
    inc_out, inc_state, *_ = chunk_kda(
        q_norm,
        k_norm,
        case["v"],
        case["raw_g"],
        case["beta"],
        SCALE,
        case["initial_state"],
        True,
        case["cu_seqlens"],
        use_gate_in_kernel=True,
        A_log=case["a_log"],
        dt_bias=case["dt_bias"],
        lower_bound=-5.0,
        pipeline_impl="incumbent",
    )
    np.testing.assert_array_equal(np.asarray(default_out), np.asarray(inc_out))
    np.testing.assert_array_equal(np.asarray(default_state), np.asarray(inc_state))

    # kda_forward_packed with pipeline_impl='fused_resident' matches resident_pipeline_kda_fwd
    packed_res_out, packed_res_state = kda_forward_packed(
        case["q"],
        case["k"],
        case["v"],
        case["raw_g"],
        case["beta"],
        cu_seqlens=case["cu_seqlens"],
        A_log=case["a_log"],
        dt_bias=case["dt_bias"],
        scale=SCALE,
        initial_state=case["initial_state"],
        lower_bound=-5.0,
        pipeline_impl="fused_resident",
    )
    direct_res_out, direct_res_state, *_ = resident_pipeline_kda_fwd(
        case["q"],
        case["k"],
        case["v"],
        case["raw_g"],
        case["beta"],
        SCALE,
        case["initial_state"],
        True,
        case["cu_seqlens"],
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        A_log=case["a_log"],
        dt_bias=case["dt_bias"],
    )
    np.testing.assert_array_equal(np.asarray(packed_res_out), np.asarray(direct_res_out))
    np.testing.assert_array_equal(np.asarray(packed_res_state), np.asarray(direct_res_state))

    with pytest.raises(ValueError, match="pipeline_impl"):
        chunk_kda(
            q_norm,
            k_norm,
            case["v"],
            case["raw_g"],
            case["beta"],
            SCALE,
            case["initial_state"],
            True,
            case["cu_seqlens"],
            pipeline_impl="invalid_pipeline",
        )
    with pytest.raises(ValueError, match="pipeline_impl"):
        kda_forward_packed(
            case["q"],
            case["k"],
            case["v"],
            case["raw_g"],
            case["beta"],
            cu_seqlens=case["cu_seqlens"],
            A_log=case["a_log"],
            dt_bias=case["dt_bias"],
            scale=SCALE,
            initial_state=case["initial_state"],
            lower_bound=-5.0,
            pipeline_impl="invalid_pipeline",
        )


def test_resident_tpu_aot_compile_v6e_and_v7x(monkeypatch):
    """AOT-compile the resident pipeline for v6e and v7x (including H=32 VMEM cap)."""
    try:
        from jax.experimental.topologies import get_topology_desc

        from sgl_jax.srt.utils.jax_utils import compilation_target
    except Exception:
        pytest.skip("TPU topology descriptor unavailable")

    monkeypatch.setenv("PALLAS_INTERPRET", "0")

    for topo_name, chips_bounds, heads, tokens in [
        ("v6e:1x1", (1, 1, 1), 8, 64),
        ("v7x:2x2x1", (2, 2, 1), 32, 128),
    ]:
        try:
            devices = get_topology_desc(
                platform="tpu",
                topology_name=topo_name,
                chip_config_name="default",
                chips_per_host_bounds=chips_bounds,
                num_slices=1,
                wrap=(False, False, False),
            ).devices
        except Exception:
            continue

        mesh = jax.sharding.Mesh(np.asarray(devices[:1]), ("data",))
        q_spec = jax.ShapeDtypeStruct((1, tokens, heads, K), jnp.bfloat16)
        v_spec = jax.ShapeDtypeStruct((1, tokens, heads, V), jnp.bfloat16)
        b_spec = jax.ShapeDtypeStruct((1, tokens, heads), jnp.bfloat16)
        a_spec = jax.ShapeDtypeStruct((heads,), jnp.float32)
        db_spec = jax.ShapeDtypeStruct((heads, K), jnp.float32)
        h0_spec = jax.ShapeDtypeStruct((1, heads, K, V), jnp.float32)
        cu_spec = jax.ShapeDtypeStruct((2,), jnp.int32)

        def _fn(q, k, v, g, b, al, db, h0, cu):
            return resident_pipeline_kda_fwd(
                q,
                k,
                v,
                g,
                b,
                SCALE,
                h0,
                True,
                cu,
                chunk_size=64,
                safe_gate=True,
                lower_bound=-5.0,
                use_gate_in_kernel=True,
                use_qk_l2norm_in_kernel=True,
                A_log=al,
                dt_bias=db,
            )[:2]

        with compilation_target(mesh):
            lowered = (
                jax.jit(_fn)
                .trace(q_spec, q_spec, v_spec, q_spec, b_spec, a_spec, db_spec, h0_spec, cu_spec)
                .lower(lowering_platforms=("tpu",))
            )
            compiled = lowered.compile()
            assert compiled is not None
