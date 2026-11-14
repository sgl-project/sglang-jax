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
  - Padded single-sequence logical boundaries (``cu_seqlens[1] < T``), on both the
    fast kernel (K, V multiples of 128) and the generic fallback kernel (K or V = 64).
  - Default selection-switch contract on ``chunk_kda_fwd`` and ``kda_forward_packed``.
  - Per-position parity against the production-default Mega KDA kernel
    (``kda_forward_packed(pipeline_impl="incumbent")``).
  - TPU AOT compilation on v6e and v7x topologies (including ``H=32`` VMEM cap);
    skipped, not silently passed, when no TPU topology can be built.

Output parity is checked per position (one token of one head): the L2 error of every
position is normalized by the RMS per-position norm of the reference, so bf16 rounding
reads ~1e-3 while a systematic few-percent error fails.  Final states are checked the
same way per (sequence, head).
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
    k_dim: int = K,
    v_dim: int = V,
):
    rng = np.random.default_rng(seed)
    logical_t = sum(lengths)
    total_t = physical_t if physical_t is not None else max(1, logical_t)
    assert total_t >= logical_t

    shape_k = (1, total_t, heads, k_dim)
    shape_v = (1, total_t, heads, v_dim)

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
    dt_bias = jnp.asarray(rng.uniform(-6.0, -1.0, (heads, k_dim)).astype(np.float32))
    cu_seqlens = jnp.asarray([0, *np.cumsum(lengths, dtype=np.int32)], dtype=jnp.int32)
    initial_state = (
        jnp.asarray(
            rng.standard_normal((len(lengths), heads, k_dim, v_dim), dtype=np.float32) * 0.1
        )
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

    k_dim = q.shape[-1]
    v_dim = case["v"].shape[-1]
    outputs = []
    final_states = []
    offset = 0
    for seg_idx, length in enumerate(case["lengths"]):
        h0_i = (
            None if case["initial_state"] is None else case["initial_state"][seg_idx : seg_idx + 1]
        )
        if length == 0:
            heads = q.shape[2]
            outputs.append(jnp.zeros((1, 0, heads, v_dim), dtype=q.dtype))
            final_states.append(
                jnp.zeros((heads, k_dim, v_dim), dtype=jnp.float32) if h0_i is None else h0_i[0]
            )
            continue
        out_i, state_i = naive_recurrent_kda(
            q[:, offset : offset + length],
            k[:, offset : offset + length],
            case["v"][:, offset : offset + length],
            g_act[:, offset : offset + length],
            beta[:, offset : offset + length],
            scale=k_dim**-0.5,
            initial_state=h0_i,
            output_final_state=True,
        )
        outputs.append(out_i)
        final_states.append(state_i[0])
        offset += length

    out = (
        jnp.concatenate(outputs, axis=1)
        if outputs
        else jnp.zeros((1, 0, q.shape[2], v_dim), dtype=q.dtype)
    )
    state = jnp.stack(final_states, axis=0)
    return out, state


# Per-position tolerances as (max_rel, mean_rel), relative to the RMS per-position norm
# of the reference.  Against the FP32 naive recurrence and Mega KDA the resident kernel
# measures o max_rel <= 2e-3 / mean_rel <= 3e-5 and state max_rel <= 2e-5.  The
# incumbent four-stage kernel keeps bf16 intermediates, so resident vs. incumbent
# measures o max_rel <= 1.3e-2 / mean_rel <= 3.2e-3 and state max_rel <= 3.3e-3 /
# mean_rel <= 2.4e-3.  Scaling the inter-chunk output term by 0.9 (a 4-8% error)
# measures o max_rel >= 3.6e-2 and mean_rel >= 9e-3 in every multi-chunk or
# initial-state case below, against every reference.
O_TOL = (1e-2, 2e-3)
STATE_TOL = (2e-3, 1e-3)
O_TOL_INCUMBENT = (3e-2, 6e-3)
STATE_TOL_INCUMBENT = (1e-2, 6e-3)


def _per_position_stats(actual, expected, *, reduce_axes: int = 1) -> dict:
    """Error statistics of ``actual`` vs ``expected`` per position.

    A position is everything except the trailing ``reduce_axes`` axes: one token of one
    head for outputs ``[B, T, H, V]`` (``reduce_axes=1``), one (sequence, head) for
    states ``[N, H, K, V]`` (``reduce_axes=2``).
    """
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    assert a.shape == e.shape, (a.shape, e.shape)
    lead_shape = a.shape[: a.ndim - reduce_axes]
    a2 = a.reshape(int(np.prod(lead_shape)), -1)
    e2 = e.reshape(a2.shape)
    if a2.size == 0:
        return {"max_rel": 0.0, "mean_rel": 0.0, "max_abs": 0.0, "min_cos": 1.0}
    err_norm = np.linalg.norm(a2 - e2, axis=1)
    ref_norm = np.linalg.norm(e2, axis=1)
    ref_rms = max(float(np.sqrt(np.mean(ref_norm**2))), 1e-12)
    rel = err_norm / ref_rms
    act_norm = np.linalg.norm(a2, axis=1)
    denom = act_norm * ref_norm
    cos = np.where(denom > 0, np.sum(a2 * e2, axis=1) / np.maximum(denom, 1e-30), 1.0)
    worst = int(np.argmax(rel))
    return {
        "max_rel": float(rel.max()),
        "mean_rel": float(rel.mean()),
        "p99_rel": float(np.percentile(rel, 99)),
        "max_abs": float(np.abs(a2 - e2).max()),
        "ref_max_abs": float(np.abs(e2).max()),
        "min_cos": float(cos.min()),
        "worst_pos": tuple(int(i) for i in np.unravel_index(worst, lead_shape)),
    }


def _assert_per_position_close(
    actual,
    expected,
    *,
    label: str,
    reduce_axes: int = 1,
    tol: tuple[float, float] | None = None,
):
    if tol is None:
        tol = O_TOL if reduce_axes == 1 else STATE_TOL
    max_rel, mean_rel = tol
    actual_np = np.asarray(actual, dtype=np.float32)
    assert np.isfinite(actual_np).all(), f"{label}: non-finite values"
    stats = _per_position_stats(actual, expected, reduce_axes=reduce_axes)
    print(f"[per-position] {label}: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    assert stats["max_rel"] <= max_rel and stats["mean_rel"] <= mean_rel, (
        f"{label}: per-position error exceeds tolerance "
        f"(max_rel<={max_rel}, mean_rel<={mean_rel}): {stats}"
    )


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

    _assert_per_position_close(
        res_out, inc_out, label="o resident vs incumbent", tol=O_TOL_INCUMBENT
    )
    _assert_per_position_close(res_out, naive_out, label="o resident vs naive")
    _assert_per_position_close(
        res_state,
        inc_state,
        label="state resident vs incumbent",
        reduce_axes=2,
        tol=STATE_TOL_INCUMBENT,
    )
    _assert_per_position_close(
        res_state, naive_state, label="state resident vs naive", reduce_axes=2
    )


@pytest.mark.parametrize(
    ("lengths", "heads", "physical_t", "lower_bound"),
    [
        ((64,), 8, None, -5.0),
        ((199,), 4, 256, None),
        ((100,), 16, 128, -5.0),
        ((128,), 32, None, None),
        ((37, 91), 8, 192, -5.0),
    ],
)
def test_resident_matches_mega_per_position(
    lengths: tuple[int, ...],
    heads: int,
    physical_t: int | None,
    lower_bound: float | None,
):
    """Per-position parity of resident vs. the production-default Mega KDA kernel.

    Both run through ``kda_forward_packed`` with the arguments ``KDAAttnBackend`` uses
    (raw Q/K with in-kernel L2 norm and gate activation), on bucket-padded token layouts.
    """
    case = _make_inputs(
        lengths,
        heads,
        physical_t=physical_t,
        seed=6100 + heads + sum(lengths),
        beta_dtype=jnp.float32 if lower_bound is None else jnp.bfloat16,
    )
    kwargs = dict(
        cu_seqlens=case["cu_seqlens"],
        A_log=case["a_log"],
        dt_bias=case["dt_bias"],
        scale=SCALE,
        initial_state=case["initial_state"],
        lower_bound=lower_bound,
    )
    args = (case["q"], case["k"], case["v"], case["raw_g"], case["beta"])
    mega_out, mega_state = kda_forward_packed(*args, **kwargs, pipeline_impl="incumbent")
    res_out, res_state = kda_forward_packed(*args, **kwargs, pipeline_impl="fused_resident")
    naive_out, naive_state = _run_naive_reference(
        case,
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
    )
    jax.block_until_ready((mega_out, mega_state, res_out, res_state, naive_out, naive_state))

    logical_t = case["logical_t"]
    _assert_per_position_close(
        res_out[:, :logical_t], mega_out[:, :logical_t], label="o resident vs mega"
    )
    _assert_per_position_close(res_out[:, :logical_t], naive_out, label="o resident vs naive")
    _assert_per_position_close(res_state, mega_state, label="state resident vs mega", reduce_axes=2)
    _assert_per_position_close(
        res_state, naive_state, label="state resident vs naive", reduce_axes=2
    )


def _bf16_ulp_diff(actual: jax.Array, expected: jax.Array) -> np.ndarray:
    a_u16 = np.asarray(actual.astype(jnp.bfloat16)).view(np.uint16).astype(np.int32)
    e_u16 = np.asarray(expected.astype(jnp.bfloat16)).view(np.uint16).astype(np.int32)

    def _to_ordered(u: np.ndarray) -> np.ndarray:
        return np.where(u & 0x8000, 0x8000 - (u & 0x7FFF), u + 0x8000)

    return np.abs(_to_ordered(a_u16) - _to_ordered(e_u16))


@pytest.mark.parametrize(
    ("lengths", "heads", "lower_bound"),
    [
        ((1024,), 4, -5.0),
        ((1024,), 4, None),
        ((256, 256, 512), 4, -5.0),
    ],
)
def test_per_position_accuracy_vs_incumbent_and_naive(
    lengths: tuple[int, ...],
    heads: int,
    lower_bound: float | None,
):
    """Per-position accuracy profile across token windows [0..63, 64..255, 256..1023].

    Verifies per-window max_abs, mean_abs, and bfloat16 ULP differences against both
    the incumbent 4-stage kernel and FP32 naive recurrence, plus final_state parity.
    """
    case = _make_inputs(
        lengths,
        heads,
        seed=8200 + heads + len(lengths),
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
        chunk_size=64,
        safe_gate=True,
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
        chunk_size=64,
        safe_gate=True,
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

    windows = [(0, 64), (64, 256), (256, 1024)]
    for t_start, t_end in windows:
        r_slice = res_out[:, t_start:t_end]
        n_slice = naive_out[:, t_start:t_end]
        i_slice = inc_out[:, t_start:t_end]
        ulp_vs_naive = _bf16_ulp_diff(r_slice, n_slice)
        _assert_per_position_close(
            r_slice,
            n_slice,
            label=f"window[{t_start}:{t_end}] resident vs naive (max_ulp={int(ulp_vs_naive.max())})",
        )
        _assert_per_position_close(
            r_slice,
            i_slice,
            label=f"window[{t_start}:{t_end}] resident vs incumbent",
            tol=O_TOL_INCUMBENT,
        )
        assert (
            float(np.mean(ulp_vs_naive <= 1)) >= 0.98
        ), f"window[{t_start}:{t_end}]: <98% of elements within 1 BF16 ULP of FP32 naive"

    _assert_per_position_close(res_state, naive_state, label="final_state vs naive", reduce_axes=2)
    _assert_per_position_close(
        res_state,
        inc_state,
        label="final_state vs incumbent",
        reduce_axes=2,
        tol=STATE_TOL_INCUMBENT,
    )


# (K, V) = (128, 128) runs the fast multi-head kernel; any head dim that is not a
# multiple of 128 runs the generic fallback kernel, which must honor cu_seqlens[1] too.
_PADDED_HEAD_DIMS = [(128, 128), (64, 64), (64, 128), (128, 64)]


@pytest.mark.parametrize(("k_dim", "v_dim"), _PADDED_HEAD_DIMS)
@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_padded_single_seq_preserves_logical_boundary(
    lower_bound: float | None, k_dim: int, v_dim: int
):
    """Padded tail (cu_seqlens=[0, 63] inside T=128) must not decay final state."""
    heads = 2
    physical_t = 128
    logical_t = 63
    scale = k_dim**-0.5
    cu_seqlens = jnp.asarray([0, logical_t], dtype=jnp.int32)
    zeros_qk = jnp.zeros((1, physical_t, heads, k_dim), dtype=jnp.bfloat16)
    zeros_v = jnp.zeros((1, physical_t, heads, v_dim), dtype=jnp.bfloat16)
    zeros_beta = jnp.zeros(
        (1, physical_t, heads),
        dtype=jnp.float32 if lower_bound is None else jnp.bfloat16,
    )
    if lower_bound is None:
        a_log = jnp.full((heads,), math.log(0.01), dtype=jnp.float32)
        dt_bias = jnp.zeros((heads, k_dim), dtype=jnp.float32)
    else:
        a_log = jnp.zeros((heads,), dtype=jnp.float32)
        dt_bias = jnp.full((heads, k_dim), -5.0, dtype=jnp.float32)
    h0 = jnp.broadcast_to(jnp.eye(k_dim, v_dim, dtype=jnp.float32), (1, heads, k_dim, v_dim))

    res_out, res_state, *_ = resident_pipeline_kda_fwd(
        zeros_qk,
        zeros_qk,
        zeros_v,
        zeros_qk,
        zeros_beta,
        scale,
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
    g_act = _activate_gate(zeros_qk[:, :logical_t], a_log, dt_bias, lower_bound)
    ref_out, ref_state = naive_recurrent_kda(
        zeros_qk[:, :logical_t],
        zeros_qk[:, :logical_t],
        zeros_v[:, :logical_t],
        g_act,
        zeros_beta[:, :logical_t],
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    jax.block_until_ready((res_out, res_state, ref_out, ref_state))

    np.testing.assert_allclose(np.asarray(res_out[:, :logical_t]), np.asarray(ref_out), atol=1e-6)
    np.testing.assert_allclose(np.asarray(res_state), np.asarray(ref_state), rtol=1e-4, atol=1e-4)
    expected_diag = 0.6461756 if lower_bound is None else 0.1214505
    assert abs(float(res_state[0, 0, 0, 0]) - expected_diag) < 2e-4


@pytest.mark.parametrize(("k_dim", "v_dim"), _PADDED_HEAD_DIMS)
@pytest.mark.parametrize(
    ("lower_bound", "use_gate_in_kernel"), [(None, True), (-5.0, True), (-5.0, False)]
)
def test_padded_single_seq_random_tail_matches_naive(
    lower_bound: float | None, use_gate_in_kernel: bool, k_dim: int, v_dim: int
):
    """cu_seqlens=[0, 63] inside T=128 with random (non-zero) tail tokens.

    Only the 63 logical tokens may update the state; per-position outputs over the
    logical prefix and the final state must match the FP32 naive recurrence.
    """
    logical_t = 63
    case = _make_inputs(
        (logical_t,),
        2,
        physical_t=128,
        seed=5100 + k_dim + v_dim,
        beta_dtype=jnp.float32 if lower_bound is None else jnp.bfloat16,
        k_dim=k_dim,
        v_dim=v_dim,
    )
    if not use_gate_in_kernel:
        case["raw_g"] = _activate_gate(case["raw_g"], case["a_log"], case["dt_bias"], lower_bound)
    q_norm = _l2_normalize(case["q"])
    k_norm = _l2_normalize(case["k"])

    res_out, res_state, *_ = chunk_kda_fwd(
        q_norm,
        k_norm,
        case["v"],
        case["raw_g"],
        case["beta"],
        k_dim**-0.5,
        case["initial_state"],
        True,
        case["cu_seqlens"],
        chunk_size=64,
        safe_gate=True,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=case["a_log"] if use_gate_in_kernel else None,
        dt_bias=case["dt_bias"] if use_gate_in_kernel else None,
        pipeline_impl="fused_resident",
    )
    naive_out, naive_state = _run_naive_reference(
        case,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        use_qk_l2norm_in_kernel=True,
    )
    jax.block_until_ready((res_out, res_state, naive_out, naive_state))

    assert res_out.shape == (1, 128, 2, v_dim)
    _assert_per_position_close(
        res_out[:, :logical_t], naive_out, label=f"o padded K={k_dim} V={v_dim} vs naive"
    )
    _assert_per_position_close(
        res_state,
        naive_state,
        label=f"state padded K={k_dim} V={v_dim} vs naive",
        reduce_axes=2,
    )


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
    _assert_per_position_close(res_o, naive_o, label="o elided-state vs naive")

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
    _assert_per_position_close(res_o2, naive_o2, label="o zero-length varlen vs naive")
    _assert_per_position_close(
        res_ht2, naive_ht2, label="state zero-length varlen vs naive", reduce_axes=2
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


@pytest.mark.parametrize(
    ("topo_name", "chips_bounds", "heads", "tokens", "lower_bound"),
    [
        ("v6e:1x1", (1, 1, 1), 8, 64, -5.0),
        ("v7x:2x2x1", (2, 2, 1), 32, 128, -5.0),
        ("v7x:2x2x1", (2, 2, 1), 32, 128, None),
    ],
)
def test_resident_tpu_aot_compile_v6e_and_v7x(
    monkeypatch,
    topo_name: str,
    chips_bounds: tuple[int, int, int],
    heads: int,
    tokens: int,
    lower_bound: float | None,
):
    """AOT-compile the resident pipeline for v6e and v7x (including H=32 VMEM cap).

    Skipped (never silently passed) when the TPU topology cannot be built here.
    """
    try:
        from jax.experimental.topologies import get_topology_desc

        from sgl_jax.srt.utils.jax_utils import compilation_target
    except Exception as exc:
        pytest.skip(f"TPU topology descriptor unavailable: {exc}")

    try:
        devices = get_topology_desc(
            platform="tpu",
            topology_name=topo_name,
            chip_config_name="default",
            chips_per_host_bounds=chips_bounds,
            num_slices=1,
            wrap=(False, False, False),
        ).devices
    except Exception as exc:
        pytest.skip(f"TPU topology {topo_name} unavailable on this host: {exc}")

    monkeypatch.setenv("PALLAS_INTERPRET", "0")

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
            lower_bound=lower_bound,
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
