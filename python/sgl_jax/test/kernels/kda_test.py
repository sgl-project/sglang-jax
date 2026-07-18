"""TPU regressions for KDA Stage 1 chunk-local gate cumsum.

The 16K node is the compile/control case: it checks shape, finiteness, and all
logical tokens.  The 32K direct node additionally owns the direct-path
padding, invalid-chunk, and sentinel-zero contract because its baseline RED is
expected to occur first at Stage 1 Pallas VMEM allocation.  The custom-mapping
node independently characterizes those masking semantics at a small shape.

Run the exact acceptance nodes on TPU with::

    uv run --with pytest python -m pytest -q \
      python/sgl_jax/test/kernels/kda_test.py::test_kda_gate_chunk_cumsum_16k_tpu_control -vv
    uv run --with pytest python -m pytest -q \
      python/sgl_jax/test/kernels/kda_test.py::test_kda_gate_chunk_cumsum_32k_tpu_regression -vv
    uv run --with pytest python -m pytest -q \
      python/sgl_jax/test/kernels/kda_test.py::test_chunk_kda_32k_varlen_output_and_final_state_match_naive_recurrent_kda -vv
    uv run --with pytest python -m pytest -q \
      python/sgl_jax/test/kernels/kda_test.py::test_chunk_kda_32k_no_zero_length_output_and_final_state_match_naive_recurrent_kda -vv
    uv run --with pytest python -m pytest -q \
      python/sgl_jax/test/kernels/kda_test.py::test_chunk_local_cumsum_preserves_custom_chunk_order_and_masks_invalid_chunks -vv
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.kda import chunk_kda, naive_recurrent_kda
from sgl_jax.srt.kernels.kda.kda import (
    align_up,
    chunk_local_cumsum_vector,
    kda_gate_chunk_cumsum,
    prepare_chunk_indices,
)

_BT = 64
_H = 2
_K = 128
_V = 128
_GATE_SCALE = 1.0 / math.log(2)


def reference_kda_gate_chunk_cumsum(
    raw_g,
    A_log,
    dt_bias,
    cu_seqlens,
    chunk_size=64,
    scale=1.0 / math.log(2),
):
    """Independent JAX reference; deliberately does not call production cumsum."""
    activated = -jnp.exp(A_log.astype(jnp.float32))[None, None, :, None] * (
        jax.nn.softplus(raw_g.astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, None, :, :])
    )
    out = jnp.zeros_like(activated, dtype=jnp.float32)
    boundaries = np.asarray(cu_seqlens)
    for request in range(len(boundaries) - 1):
        begin, end = int(boundaries[request]), int(boundaries[request + 1])
        for start in range(begin, end, chunk_size):
            stop = min(start + chunk_size, end)
            out = out.at[:, start:stop].set(jnp.cumsum(activated[:, start:stop], axis=1) * scale)
    return out


def _reference_chunk_local_cumsum(
    g: jax.Array,
    *,
    chunk_size: int,
    reverse: bool,
    scale: float | None,
    cu_seqlens: jax.Array,
    head_first: bool,
    output_dtype,
    chunk_indices: jax.Array,
) -> jax.Array:
    """Independent mapping-aware reference for the characterization node."""
    if head_first:
        batch, heads, total_t, width = g.shape
        flat = g.reshape(batch * heads, total_t, width).astype(jnp.float32)
    else:
        batch, total_t, heads, width = g.shape
        flat = (
            jnp.transpose(g, (0, 2, 1, 3))
            .reshape(batch * heads, total_t, width)
            .astype(jnp.float32)
        )

    out = jnp.zeros_like(flat, dtype=jnp.float32)
    boundaries = np.asarray(cu_seqlens)
    mapping = np.asarray(chunk_indices)
    for seq_id_raw, local_chunk_raw in mapping:
        seq_id = int(seq_id_raw)
        local_chunk = int(local_chunk_raw)
        if seq_id < 0 or seq_id >= len(boundaries) - 1 or local_chunk < 0:
            continue
        begin, end = int(boundaries[seq_id]), int(boundaries[seq_id + 1])
        start = begin + local_chunk * chunk_size
        if start >= end or start >= total_t:
            continue
        stop = min(start + chunk_size, end, total_t)
        values = flat[:, start:stop]
        if reverse:
            values = jnp.flip(jnp.cumsum(jnp.flip(values, axis=1), axis=1), axis=1)
        else:
            values = jnp.cumsum(values, axis=1)
        if scale is not None:
            values = values * scale
        out = out.at[:, start:stop].set(values)

    out_dtype = output_dtype or g.dtype
    out = out.astype(out_dtype)
    if head_first:
        return out.reshape(batch, heads, total_t, width)
    return jnp.transpose(out.reshape(batch, heads, total_t, width), (0, 2, 1, 3))


def _make_direct_case(length: int):
    aligned_t = align_up(length + _BT - 1, _BT)
    raw_key, a_key, bias_key = jax.random.split(jax.random.PRNGKey(325), 3)
    raw_g = 0.25 + 0.2 * jax.random.normal(raw_key, (1, aligned_t, _H, _K), dtype=jnp.float32)
    A_log = -1.5 + 0.1 * jax.random.normal(a_key, (_H,), dtype=jnp.float32)
    dt_bias = 0.1 + 0.2 * jax.random.normal(bias_key, (_H, _K), dtype=jnp.float32)
    cu_seqlens = jnp.asarray([0, length], dtype=jnp.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size=_BT, max_T=aligned_t)

    assert raw_g.shape == (1, aligned_t, _H, _K)
    assert raw_g.dtype == jnp.float32
    assert np.count_nonzero(np.asarray(A_log)) > 0
    assert np.ptp(np.asarray(A_log)) > 0
    assert np.count_nonzero(np.asarray(dt_bias)) > 0
    assert np.ptp(np.asarray(dt_bias)) > 0
    np.testing.assert_array_equal(
        np.asarray(chunk_indices[-1]), np.asarray([0, length // _BT], dtype=np.int32)
    )
    # The invalid row gathers deliberately non-zero data unless production masks it.
    assert np.count_nonzero(np.asarray(raw_g[:, length:aligned_t])) > 0
    assert np.ptp(np.asarray(raw_g[:, length:aligned_t])) > 0
    return raw_g, A_log, dt_bias, cu_seqlens, chunk_indices, aligned_t


def _run_direct_case(length: int):
    raw_g, A_log, dt_bias, cu_seqlens, chunk_indices, aligned_t = _make_direct_case(length)
    optimized = jax.jit(
        lambda dynamic_raw_g: kda_gate_chunk_cumsum(
            dynamic_raw_g,
            A_log,
            chunk_size=_BT,
            scale=_GATE_SCALE,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            output_dtype=jnp.float32,
            chunk_indices=chunk_indices,
        )
    )(raw_g)
    # The 32K RED must surface at the optimized Stage 1 compile before the
    # independent eager reference performs one update per logical chunk.
    jax.block_until_ready(optimized)
    reference = reference_kda_gate_chunk_cumsum(
        raw_g, A_log, dt_bias, cu_seqlens, chunk_size=_BT, scale=_GATE_SCALE
    )
    assert optimized.shape == reference.shape == (1, aligned_t, _H, _K)
    assert np.isfinite(np.asarray(optimized)).all()
    assert np.isfinite(np.asarray(reference)).all()
    np.testing.assert_allclose(
        np.asarray(optimized[:, :length]),
        np.asarray(reference[:, :length]),
        rtol=1e-5,
        atol=1e-5,
    )
    return optimized, reference, aligned_t


def test_kda_gate_chunk_cumsum_16k_tpu_control():
    """16K control: logical/aligned/Stage-1 allocated T = 16384/16448/16512."""
    length = 16_384
    optimized, reference, aligned_t = _run_direct_case(length)
    assert aligned_t == 16_448
    assert aligned_t + _BT == 16_512
    # Materialize both results before the test returns so asynchronous TPU errors surface here.
    jax.block_until_ready((optimized, reference))


def test_kda_gate_chunk_cumsum_32k_tpu_regression():
    """32K blocker: logical/aligned/Stage-1 allocated T = 32768/32832/32896."""
    length = 32_768
    optimized, reference, aligned_t = _run_direct_case(length)
    assert aligned_t == 32_832
    assert aligned_t + _BT == 32_896

    np.testing.assert_array_equal(
        np.asarray(reference[:, length:aligned_t]),
        np.zeros((1, aligned_t - length, _H, _K), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(optimized[:, length:aligned_t]),
        np.zeros((1, aligned_t - length, _H, _K), dtype=np.float32),
    )
    # The sole extra mapping row is invalid.  Its gather and all partial lanes
    # map to sentinel=cu_seqlens[-1]=length and must contribute exactly zero.
    np.testing.assert_array_equal(
        np.asarray(optimized[:, length : length + _BT]),
        np.zeros((1, _BT, _H, _K), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(optimized[:, length]), np.zeros((1, _H, _K), dtype=np.float32)
    )


def _make_full_32k_case(*, include_zero_length: bool = True):
    base_seq_lens = [0, 1023, 1025] + [1024] * 30
    seq_lens = base_seq_lens if include_zero_length else base_seq_lens[1:]
    logical_t = sum(seq_lens)
    assert len(seq_lens) == (33 if include_zero_length else 32)
    assert logical_t == 32_768
    cu_seqlens = jnp.asarray(
        np.concatenate([[0], np.cumsum(seq_lens, dtype=np.int32)]), dtype=jnp.int32
    )

    keys = jax.random.split(jax.random.PRNGKey(325), 8)
    shape = (1, logical_t, _H, _K)
    q = (0.1 * jax.random.normal(keys[0], shape, dtype=jnp.float32)).astype(jnp.bfloat16)
    k = (0.1 * jax.random.normal(keys[1], shape, dtype=jnp.float32)).astype(jnp.bfloat16)
    v = (0.1 * jax.random.normal(keys[2], shape, dtype=jnp.float32)).astype(jnp.bfloat16)
    raw_g = (0.25 + 0.2 * jax.random.normal(keys[3], shape, dtype=jnp.float32)).astype(jnp.bfloat16)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (1, logical_t, _H), dtype=jnp.float32)).astype(
        jnp.bfloat16
    )
    A_log = -1.5 + 0.1 * jax.random.normal(keys[5], (_H,), dtype=jnp.float32)
    dt_bias = 0.1 + 0.2 * jax.random.normal(keys[6], (_H, _K), dtype=jnp.float32)
    base_initial_state = 0.01 * jax.random.normal(
        keys[7], (len(base_seq_lens), _H, _K, _V), dtype=jnp.float32
    )
    # Generate the 33-row base first so removing the zero-length request keeps
    # every remaining request's initial state bit-for-bit identical.
    initial_state = base_initial_state if include_zero_length else base_initial_state[1:]

    assert q.dtype == k.dtype == v.dtype == raw_g.dtype == jnp.bfloat16
    assert q.shape == k.shape == v.shape == raw_g.shape == shape
    assert np.count_nonzero(np.asarray(raw_g)) > 0
    assert np.ptp(np.asarray(raw_g, dtype=np.float32)) > 0
    assert initial_state.dtype == jnp.float32
    assert np.count_nonzero(np.asarray(initial_state)) > 0
    assert np.ptp(np.asarray(initial_state)) > 0
    return (
        seq_lens,
        cu_seqlens,
        q,
        k,
        v,
        raw_g,
        beta,
        A_log,
        dt_bias,
        initial_state,
    )


def _full_naive_reference(
    seq_lens,
    cu_seqlens,
    q,
    k,
    v,
    activated_g,
    beta,
    initial_state,
    scale,
):
    """Call ``naive_recurrent_kda`` exactly once for each logical request."""
    callable_by_length = {}
    outputs = []
    final_states = []
    call_count = 0
    boundaries = np.asarray(cu_seqlens)

    def build_callable():
        def run(q_i, k_i, v_i, g_i, beta_i, state_i):
            return naive_recurrent_kda(
                q_i,
                k_i,
                v_i,
                g_i,
                beta_i,
                scale=scale,
                initial_state=state_i,
                output_final_state=True,
            )

        return jax.jit(run)

    for request, expected_length in enumerate(seq_lens):
        begin, end = int(boundaries[request]), int(boundaries[request + 1])
        assert end - begin == expected_length
        if expected_length not in callable_by_length:
            callable_by_length[expected_length] = build_callable()
        run = callable_by_length[expected_length]
        output_i, final_state_i = run(
            q[:, begin:end],
            k[:, begin:end],
            v[:, begin:end],
            activated_g[:, begin:end],
            beta[:, begin:end],
            initial_state[request : request + 1],
        )
        call_count += 1
        outputs.append(output_i)
        final_states.append(final_state_i[0])

    assert call_count == len(seq_lens)
    assert set(callable_by_length) == set(seq_lens)
    return jnp.concatenate(outputs, axis=1), jnp.stack(final_states, axis=0)


def _run_full_32k_case(*, include_zero_length: bool):
    (
        seq_lens,
        cu_seqlens,
        q,
        k,
        v,
        raw_g,
        beta,
        A_log,
        dt_bias,
        initial_state,
    ) = _make_full_32k_case(include_zero_length=include_zero_length)

    scale = _K**-0.5
    optimized_output, optimized_final_state, *_ = chunk_kda(
        q,
        k,
        v,
        raw_g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_size=_BT,
        use_gate_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
    )
    # Materialize the optimized pipeline first so a baseline failure is
    # unambiguously Stage 1 VMEM, not naive-reference compile or host memory.
    jax.block_until_ready((optimized_output, optimized_final_state))

    activated_g = -jnp.exp(A_log.astype(jnp.float32))[None, None, :, None] * (
        jax.nn.softplus(raw_g.astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, None, :, :])
    )
    reference_output, reference_final_state = _full_naive_reference(
        seq_lens,
        cu_seqlens,
        q,
        k,
        v,
        activated_g,
        beta,
        initial_state,
        scale,
    )

    # _align_seqs allocates its static worst-case packed shape rather than only
    # padded_cu_seqlens[-1]. Keep logical, aligned, and Stage-1 allocated T
    # separate because the zero-length request changes only the latter two.
    aligned_t = align_up(len(seq_lens) * (_BT - 1) + q.shape[1], _BT)
    expected_aligned_t = 34_880 if include_zero_length else 34_816
    expected_allocated_t = expected_aligned_t + _BT
    assert q.shape[1] == 32_768
    assert aligned_t == expected_aligned_t
    assert aligned_t + _BT == expected_allocated_t
    assert optimized_output.shape == reference_output.shape == (1, 32_768, _H, _V)
    assert (
        optimized_final_state.shape
        == reference_final_state.shape
        == (
            len(seq_lens),
            _H,
            _K,
            _V,
        )
    )
    assert np.isfinite(np.asarray(optimized_output)).all()
    assert np.isfinite(np.asarray(reference_output)).all()
    assert np.isfinite(np.asarray(optimized_final_state)).all()
    assert np.isfinite(np.asarray(reference_final_state)).all()
    np.testing.assert_allclose(
        np.asarray(optimized_output),
        np.asarray(reference_output),
        rtol=2e-2,
        atol=1e-2,
    )
    return seq_lens, optimized_final_state, reference_final_state


def test_chunk_kda_32k_varlen_output_and_final_state_match_naive_recurrent_kda():
    """Zero-length full path: logical/aligned/allocated T = 32768/34880/34944."""
    seq_lens, optimized_final_state, reference_final_state = _run_full_32k_case(
        include_zero_length=True
    )

    nonempty_mask = np.asarray(seq_lens, dtype=np.int32) > 0
    assert nonempty_mask.shape == (33,)
    assert np.count_nonzero(nonempty_mask) == 32
    # Zero-length optimized final-state semantics are tracked by #326:
    # https://github.com/primatrix/projects/issues/326
    # All logical output remains checked above; only the explicitly nonempty
    # requests participate in this final-state acceptance boundary.
    np.testing.assert_allclose(
        np.asarray(optimized_final_state)[nonempty_mask],
        np.asarray(reference_final_state)[nonempty_mask],
        rtol=2e-2,
        atol=1e-2,
    )


def test_chunk_kda_32k_no_zero_length_output_and_final_state_match_naive_recurrent_kda():
    """No-zero full path: logical/aligned/allocated T = 32768/34816/34880."""
    seq_lens, optimized_final_state, reference_final_state = _run_full_32k_case(
        include_zero_length=False
    )

    nonempty_mask = np.asarray(seq_lens, dtype=np.int32) > 0
    assert nonempty_mask.shape == (32,)
    assert nonempty_mask.all()
    np.testing.assert_allclose(
        np.asarray(optimized_final_state),
        np.asarray(reference_final_state),
        rtol=2e-2,
        atol=1e-2,
    )


def test_chunk_local_cumsum_preserves_custom_chunk_order_and_masks_invalid_chunks():
    """Characterize caller mapping, both scan directions, dtype/layout, and masks."""
    chunk_size = 4
    total_t = 12
    # The duplicate boundary at 3 is a zero-length request. Keep this fixed-seed
    # fixture as the direct caller-mapping/mask characterization for #325.
    cu_seqlens = jnp.asarray([0, 3, 3, 8], dtype=jnp.int32)
    # [2, 1] is partial, the middle rows are deliberately reordered, and
    # [2, 2] is invalid; production must consume these caller-provided rows.
    chunk_indices = jnp.asarray([[2, 1], [0, 0], [2, 0], [2, 2]], dtype=jnp.int32)
    key = jax.random.PRNGKey(325)
    g = 0.25 + 0.2 * jax.random.normal(key, (1, total_t, _H, 7), dtype=jnp.float32)
    assert np.count_nonzero(np.asarray(g[:, 8:])) > 0
    assert np.ptp(np.asarray(g[:, 8:])) > 0

    cases = (
        (False, False, jnp.float32, 0.75, g, 2e-5, 2e-5),
        (
            True,
            True,
            jnp.bfloat16,
            1.25,
            jnp.transpose(g, (0, 2, 1, 3)),
            2e-2,
            2e-2,
        ),
    )
    for reverse, head_first, output_dtype, scale, case_g, rtol, atol in cases:
        optimized = chunk_local_cumsum_vector(
            case_g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
        reference = _reference_chunk_local_cumsum(
            case_g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
        assert optimized.shape == reference.shape == case_g.shape
        assert optimized.dtype == reference.dtype == output_dtype
        assert np.isfinite(np.asarray(optimized)).all()

        time_axis = 2 if head_first else 1
        logical_slice = [slice(None)] * optimized.ndim
        logical_slice[time_axis] = slice(0, 8)
        np.testing.assert_allclose(
            np.asarray(optimized[tuple(logical_slice)]),
            np.asarray(reference[tuple(logical_slice)]),
            rtol=rtol,
            atol=atol,
        )

        padding_slice = [slice(None)] * optimized.ndim
        padding_slice[time_axis] = slice(8, total_t)
        np.testing.assert_array_equal(
            np.asarray(reference[tuple(padding_slice)]),
            np.zeros_like(np.asarray(reference[tuple(padding_slice)])),
        )
        np.testing.assert_array_equal(
            np.asarray(optimized[tuple(padding_slice)]),
            np.zeros_like(np.asarray(optimized[tuple(padding_slice)])),
        )
        sentinel_slice = [slice(None)] * optimized.ndim
        sentinel_slice[time_axis] = 8
        np.testing.assert_array_equal(
            np.asarray(optimized[tuple(sentinel_slice)]),
            np.zeros_like(np.asarray(optimized[tuple(sentinel_slice)])),
        )

    # Mapping-sensitive omission probe: keep the approved mapping above as the
    # primary characterization, then omit [2, 0] here.  A kernel that silently
    # rebuilds canonical rows would populate positions 3:7 and fail this check.
    omission_mapping = jnp.asarray([[2, 1], [0, 0], [2, 2], [2, 3]], dtype=jnp.int32)
    omission_optimized = chunk_local_cumsum_vector(
        g,
        chunk_size=chunk_size,
        reverse=False,
        scale=0.75,
        cu_seqlens=cu_seqlens,
        head_first=False,
        output_dtype=jnp.float32,
        chunk_indices=omission_mapping,
    )
    omission_reference = _reference_chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        reverse=False,
        scale=0.75,
        cu_seqlens=cu_seqlens,
        head_first=False,
        output_dtype=jnp.float32,
        chunk_indices=omission_mapping,
    )
    np.testing.assert_allclose(
        np.asarray(omission_optimized),
        np.asarray(omission_reference),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_array_equal(
        np.asarray(omission_optimized[:, 3:7]),
        np.zeros((1, 4, _H, 7), dtype=np.float32),
    )
