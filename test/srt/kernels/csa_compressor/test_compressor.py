"""Independent compressor arithmetic, cache-output and state-update tests."""

from typing import NamedTuple
from unittest.mock import patch

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.csa_compressor import compressor as module
from sgl_jax.srt.kernels.csa_compressor.compressor import (
    CompressorBlock,
    CompressorMetadata,
    csa_compressor,
)
from sgl_jax.srt.kernels.csa_compressor.tune import get_compressor_schedule

from . import ref

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires TPU")


@jax.jit
def _observe_prefill(*args):
    """Expose pre-pack values without replacing the production arithmetic."""
    original_call = pl.pallas_call
    main_pack, index_pack = module._pack_main_cache_rows, module._pack_index_cache_rows
    captured = []

    def observe(kernel, **kwargs):
        spec, shapes = kwargs["grid_spec"], kwargs["out_shape"]
        batch, groups = shapes[0].shape[:2]
        tile = spec.out_specs[0].block_shape[1]
        extra_start = len(spec.in_specs) + len(spec.out_specs)

        def observed_kernel(*refs):
            def main(value):
                refs[extra_start][0] = value
                return main_pack(value)

            def index(value):
                refs[extra_start + 1][0] = value
                return index_pack(value)

            with (
                patch.object(module, "_pack_main_cache_rows", main),
                patch.object(module, "_pack_index_cache_rows", index),
            ):
                kernel(*refs[:extra_start], *refs[extra_start + 2 :])

        kwargs["grid_spec"] = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=spec.grid,
            in_specs=spec.in_specs,
            out_specs=(
                *spec.out_specs,
                *[
                    pl.BlockSpec((1, tile, width), lambda r, q, k: (r, q, 0))
                    for width in (ref.ATTENTION_DIM, ref.INDEX_DIM)
                ],
            ),
            scratch_shapes=spec.scratch_shapes,
        )
        kwargs["out_shape"] = (
            *shapes,
            *[
                jax.ShapeDtypeStruct((batch, groups, width), jnp.float32)
                for width in (ref.ATTENTION_DIM, ref.INDEX_DIM)
            ],
        )
        call = original_call(observed_kernel, **kwargs)

        def run(*operands):
            outputs = call(*operands)
            captured.extend(outputs[len(shapes) :])
            return outputs

        return run

    with patch.object(pl, "pallas_call", observe):
        outputs = module.csa_dual_uniform_prefill_pallas.__wrapped__(*args)
    groups = args[0].shape[1] // ref.COMPRESSION_RATIO
    return (*outputs, *[v[:, :groups] for v in captured])


@pytest.mark.parametrize(
    "batch,query,hidden", [(2, 128, 128), (1, 2048, 128), (1, 8, 4096), (1, 8, 7168)]
)
def test_compressor_layered_prefill(batch, query, hidden, record_property):
    operands, metadata = make_case(
        (query,) * batch, tuple(4 * r for r in range(batch)), hidden=hidden
    )
    check_layered_prefill(operands, metadata, record_property)


def check_layered_prefill(operands, metadata, record_property):
    x, w, ma, ia, mn, inn, cos, sin, ms, ins, *_ = operands
    lengths = np.diff(np.asarray(metadata.cu_q_lens))
    assert (lengths == lengths[0]).all()
    batch, query, hidden = len(lengths), int(lengths[0]), x.shape[1]
    ids = np.arange(batch * query).reshape(batch, query)
    slots = np.asarray(metadata.state_indices)
    starts = np.asarray(metadata.positions)[ids[:, :: ref.COMPRESSION_RATIO]]
    args = tuple(
        jnp.asarray(v)
        for v in (x[ids], ms[slots], ins[slots], w, ma, ia, mn, inn, cos[starts], sin[starts])
    )
    plain = jax.block_until_ready(module.csa_dual_uniform_prefill_pallas(*args))
    observed = jax.block_until_ready(_observe_prefill(*args))
    # Observation is valid only if every ordinary output remains unchanged.
    for a, b in zip(plain, observed[: len(plain)], strict=True):
        np.testing.assert_array_equal(a, b)
    main, index = (
        np.asarray(v).reshape(-1, width) for v, width in zip(observed[-2:], (512, 128), strict=True)
    )
    schedule = get_compressor_schedule(hidden, device_kind=jax.devices()[0].device_kind)
    projection = ref.compressor_projection(
        x, w, [ids], k_tile=schedule.projection_k_tile, query_tile=schedule.query_tile
    )
    expected = []
    for p, state, ape, norm in (
        (projection[:, :2048], ms, ma, mn),
        (projection[:, 2048:], ins, ia, inn),
    ):
        _, values, _ = ref.compressor_ragged(
            p[: batch * query],
            state[slots],
            ape,
            norm,
            cos,
            sin,
            np.asarray(metadata.positions)[: batch * query],
            (query,) * batch,
        )
        expected.append(np.concatenate(values))
    for name, actual, reference in zip(("main", "index"), (main, index), expected, strict=True):
        assert np.isfinite(actual).all() and np.isfinite(reference).all()
        np.testing.assert_allclose(actual, reference, rtol=2e-2, atol=1e-2, equal_nan=False)
        record_property(f"{name}_prepack_max_abs", float(np.max(np.abs(actual - reference))))
    encoded = (*ref.pack_main(main), ref.pack_index(index))
    for actual, reference in zip(plain[:3], encoded, strict=True):
        np.testing.assert_array_equal(np.asarray(actual).reshape(reference.shape), reference)
    # Record quantization differences separately from the numerical output gate.
    for name, actual, reference in zip(
        ("main", "index"),
        (encoded[0], encoded[2]),
        (ref.pack_main(expected[0])[0], ref.pack_index(expected[1])),
        strict=True,
    ):
        record_property(
            f"{name}_cross_cache_byte_differences", int(np.count_nonzero(actual != reference))
        )


def test_compressor_layered_encoding_midpoints():
    midpoint = np.float32(3.625)
    triplet = np.asarray(
        [
            np.nextafter(midpoint, np.float32(0)),
            midpoint,
            np.nextafter(midpoint, np.float32(np.inf)),
        ]
    )
    values = np.ones((8, 512), np.float32)
    values[:, 0] = np.concatenate((triplet, -triplet, [np.float32(0), np.float32(-0.0)]))
    values[:, 63] = np.float32(448)  # Anchor both encoders' relevant scale to one.

    def kernel(x, nope, rope, index):
        nope[...], rope[...] = module._pack_main_cache_rows(x[...])
        index[...] = module._pack_index_cache_rows(x[:, :128])

    actual = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((8, 4, 128), jnp.uint8),
            jax.ShapeDtypeStruct((8, 128), jnp.uint8),
            jax.ShapeDtypeStruct((8, 2, 128), jnp.uint8),
        ),
    )(jnp.asarray(values))
    for a, b in zip(actual, (*ref.pack_main(values), ref.pack_index(values[:, :128])), strict=True):
        np.testing.assert_array_equal(np.asarray(a).reshape(b.shape), b)


def test_compressor_encoding_scale_boundaries():
    # Amax just below/at/above power-of-two scale transitions, plus zero/floor.
    anchors = np.asarray([0, ref.FP8_AMAX_FLOOR, 224, 448, 896], np.float32)
    anchors = np.stack(
        (np.nextafter(anchors, np.float32(0)), anchors, np.nextafter(anchors, np.float32(np.inf))),
        axis=1,
    ).ravel()
    anchors = np.pad(anchors, (0, 1))
    values = np.broadcast_to(anchors[:, None], (16, 512)).copy()
    values[:, 1::2] *= -1

    # The amax floor bounds scales even for zero and tiny inputs.
    def kernel(x, nope, rope, index):
        nope[...], rope[...] = module._pack_main_cache_rows(x[...])
        index[...] = module._pack_index_cache_rows(x[:, :128])

    actual = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((16, 4, 128), jnp.uint8),
            jax.ShapeDtypeStruct((16, 128), jnp.uint8),
            jax.ShapeDtypeStruct((16, 2, 128), jnp.uint8),
        ),
    )(jnp.asarray(values))
    for a, b in zip(actual, (*ref.pack_main(values), ref.pack_index(values[:, :128])), strict=True):
        np.testing.assert_array_equal(np.asarray(a).reshape(b.shape), b)


def test_quantized_acceptance_rejects_corruption():
    reference = np.ones((1, 128), np.float32)
    reference[0, 0] = np.nextafter(np.float32(3.625), np.float32(np.inf))
    reference[0, -1] = 448  # Fixed scale; midpoints are not scale transitions.
    records = ref.pack_index(reference)
    decoded = ref.decode_index(records)
    scale = records[:, 128:129].view(ml_dtypes.float8_e8m0fnu).astype(np.float32)
    # Tiny arithmetic changes may legitimately land on either side of a midpoint.
    for value in (3.5, 3.75):
        alternate = decoded.copy()
        alternate[0, 0] = value
        ref.assert_quantized_output(alternate, reference, scale)
    bad = decoded.copy()
    bad[0, 1] = 1.125  # Even one adjacent code away is invalid away from a midpoint.
    with pytest.raises(AssertionError):
        ref.assert_quantized_output(bad, reference, scale)
    with pytest.raises(AssertionError):
        ref.assert_quantized_output(decoded, reference, scale * 4)
    with pytest.raises(AssertionError):
        ref.assert_quantized_output(np.full_like(decoded, np.nan), reference, scale)
    saturated = reference.copy()
    saturated[0, -1] = 896
    with pytest.raises(AssertionError):
        ref.assert_quantized_output(decoded, saturated, scale)


@pytest.mark.parametrize("records", [1, 16])
def test_attention_budget_propagation(records):
    rng = np.random.default_rng(43)
    cache = rng.normal(size=(records, 512)).astype(np.float32)
    query = rng.normal(size=(8, 512)).astype(np.float32)
    epsilon = ref.ARITHMETIC_ATOL + ref.ARITHMETIC_RTOL * np.abs(cache)
    budget = ref.attention_error_budget(query, cache, epsilon)
    assert np.isfinite(budget).all() and (budget >= 0).all()
    if records == 1:
        # Softmax over one key is exactly one: only the value budget remains.
        np.testing.assert_array_equal(budget, np.broadcast_to(epsilon, budget.shape))
    baseline = ref.attention_from_cache(query, cache)
    for direction in (-np.ones_like(cache), np.ones_like(cache), rng.uniform(-1, 1, cache.shape)):
        perturbed = (cache + direction * epsilon).astype(np.float32)
        difference = np.abs(ref.attention_from_cache(query, perturbed) - baseline)
        rounding = np.finfo(np.float32).eps * np.maximum(1, np.abs(baseline))
        assert (difference <= budget + rounding).all()


@pytest.mark.parametrize("k_tile", [32, 64, 128])
def test_reference_projection_rules(k_tile):
    lengths, prefixes = (2, 5, 0, 1), (1, 3, 0, 7)
    positions = np.concatenate(
        [np.arange(p, p + n) for p, n in zip(prefixes, lengths, strict=True)]
    )
    blocks = ref.projection_blocks(positions, lengths)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([b.ravel() for b in blocks])), np.arange(8)
    )
    # Integer products/sums are exactly representable: scheduling cannot change this answer.
    rng = np.random.default_rng(42)
    x = rng.integers(-2, 3, (8, 128)).astype(np.float32)
    w = rng.integers(-2, 3, (128, 2560)).astype(np.float32)
    blocks = [np.pad(b, ((0, 1), (0, 0)), constant_values=-1) for b in blocks]
    actual = ref.compressor_projection(x, w, blocks, k_tile=k_tile, query_tile=4)
    expected = np.einsum("tk,kd->td", x, w, optimize=False)
    np.testing.assert_array_equal(actual, expected)


def make_case(lengths, prefixes, *, hidden=128, seed=10, pad=3, zero=False):
    """Synthetic caller data; no indexer/attention/pool imports."""
    rng = np.random.default_rng(seed)
    batch, tokens = len(lengths), sum(lengths)
    cu = np.asarray((0, *np.cumsum(lengths)), np.int32)
    positions = np.concatenate(
        [np.arange(p, p + n, dtype=np.int32) for p, n in zip(prefixes, lengths, strict=True)]
    )
    positions = np.pad(positions, (0, pad), constant_values=-1)
    slots = np.arange(batch, 0, -1, dtype=np.int32)
    slots[np.asarray(lengths) == 0] = -1
    states = []
    for width in (1024, 256):
        state = np.zeros((batch + 2, 8, 2, width), np.float32)
        state[:, :, 1] = -np.inf
        for request, prefix in enumerate(prefixes):
            slot = int(slots[request])
            if slot < 0:
                continue
            if prefix >= 4:
                state[slot] = rng.normal(0, 0.1, (8, 2, width)).astype(np.float32)
            elif prefix:
                state[slot, 4 : 4 + prefix] = rng.normal(0, 0.1, (prefix, 2, width)).astype(
                    np.float32
                )
        states.append(state)

    def bf16(shape, scale):
        return rng.normal(0, scale, shape).astype(ml_dtypes.bfloat16)

    x = bf16((tokens + pad, hidden), 0.3)
    weight = bf16((hidden, 2560), 0 if zero else 0.02)
    ape = [rng.normal(0, 0.02, (4, d)).astype(np.float32) for d in (1024, 256)]
    norms = [np.ones(d, np.float32) for d in (512, 128)]
    maxpos = max((p + n for p, n in zip(prefixes, lengths, strict=True)), default=0) + 1
    angle = rng.normal(0, 0.2, (maxpos, 32)).astype(np.float32)
    cos, sin = np.cos(angle), np.sin(angle)
    capacity = max(128, ((maxpos // 4 * batch + 32 + 127) // 128) * 128)
    pages = capacity // 128
    caches = [
        np.full(shape, 0x25, np.uint8)
        for shape in ((pages, 128, 4, 128), (pages, 32, 4, 128), (pages, 32, 4, 256))
    ]
    locations = np.full(tokens + pad, -1, np.int32)
    # Consecutive records of different requests may share a four-record storage group.
    next_location = 3
    for request, n in enumerate(lengths):
        for token in range(cu[request], cu[request + 1]):
            if (positions[token] + 1) % 4 == 0:
                locations[token] = next_location
                next_location += 1
    blocks = []
    # Group requests only in this caller; production sees device arrays.
    for length, start_slot in sorted(set(zip(lengths, [p % 4 for p in prefixes], strict=True))):
        if not length:
            continue
        requests = np.asarray(
            [
                r
                for r, (n, p) in enumerate(zip(lengths, prefixes, strict=True))
                if (n, p % 4) == (length, start_slot)
            ],
            np.int32,
        )
        leading = min(length, (-start_slot) % 4)
        bulk_stop = leading + (length - leading) // 4 * 4
        intervals = [(i, i + 1) for i in range(leading)]
        if bulk_stop > leading:
            intervals.append((leading, bulk_stop))
        intervals.extend((i, i + 1) for i in range(bulk_stop, length))
        for start, stop in intervals:
            ids = cu[requests, None] + np.arange(start, stop, dtype=np.int32)[None]
            ids = np.pad(ids, ((0, 1), (0, 0)), constant_values=-1)
            reqs = np.pad(requests, (0, 1), constant_values=-1)
            blocks.append(CompressorBlock(jnp.asarray(ids), jnp.asarray(reqs)))
    metadata = CompressorMetadata(
        *[jnp.asarray(v) for v in (positions, cu, slots, locations)], tuple(blocks)
    )
    operands = [x, weight, *ape, *norms, cos, sin, *states, *caches]
    return operands, metadata


class OracleResult(NamedTuple):
    outputs: tuple
    prepack: tuple
    owners: np.ndarray


def oracle(operands, metadata):
    x, w, ma, ia, mn, inn, cos, sin, ms, ins, *caches = operands
    schedule = get_compressor_schedule(x.shape[1], device_kind=jax.devices()[0].device_kind)
    projection = ref.compressor_projection(
        x,
        w,
        [np.asarray(block.token_indices) for block in metadata.blocks],
        k_tile=schedule.projection_k_tile,
        query_tile=schedule.query_tile,
    )
    states = [ms.copy(), ins.copy()]
    expected = [v.copy() for v in caches]
    capacity = caches[0].size // 512
    prepack = tuple(np.full((capacity, d), np.nan, np.float32) for d in (512, 128))
    owners = np.full(capacity, -1, np.int32)
    cu, slots, pos, locations = map(
        np.asarray,
        (metadata.cu_q_lens, metadata.state_indices, metadata.positions, metadata.cache_locations),
    )
    for request, slot in enumerate(slots):
        if slot < 0:
            continue
        for token in range(cu[request], cu[request + 1]):
            values = []
            for i, (ape, norm) in enumerate(((ma, mn), (ia, inn))):
                value, emit, state = ref.compressor_step(
                    (
                        projection[token : token + 1, :2048]
                        if i == 0
                        else projection[token : token + 1, 2048:]
                    ),
                    states[i][slot : slot + 1],
                    ape,
                    norm,
                    cos,
                    sin,
                    pos[token : token + 1],
                )
                states[i][slot] = state[0]
                values.append(value)
            if emit[0] and locations[token] >= 0:
                owners[locations[token]] = request
                for target, value in zip(prepack, values, strict=True):
                    target[locations[token]] = value[0]
                records = (*ref.pack_main(values[0]), ref.pack_index(values[1]))
                for cache, record, width in zip(expected, records, (512, 128, 256), strict=True):
                    cache.reshape(-1, width)[locations[token]] = record[0]
    return OracleResult((*states, *expected), prepack, owners)


def check(actual, expected, initial, record_property=None):
    prepack, owners = expected.prepack, expected.owners
    expected = expected.outputs
    written = owners >= 0
    for a, b in zip(actual[:2], expected[:2], strict=True):
        # -inf is the specified empty-score sentinel, never an arithmetic result.
        np.testing.assert_array_equal(np.isneginf(a), np.isneginf(b))
        active = ~np.isneginf(b)
        assert np.isfinite(a[active]).all() and np.isfinite(b[active]).all()
        np.testing.assert_allclose(
            a[active], b[active], rtol=ref.ARITHMETIC_RTOL, atol=ref.ARITHMETIC_ATOL
        )
    for offset, width in ((2, 512), (3, 128), (4, 256)):
        before = initial[offset + 8].reshape(-1, width)
        np.testing.assert_array_equal(actual[offset].reshape(-1, width)[~written], before[~written])
    if not written.any():
        return
    nope, rope, index = [
        v.reshape(-1, width)[written] for v, width in zip(actual[2:], (512, 128, 256), strict=True)
    ]
    np.testing.assert_array_equal(nope[:, 455:], 0)
    np.testing.assert_array_equal(index[:, 129:], 0)
    main_values = ref.decode_main(nope, rope)
    nope_budget = ref.assert_quantized_output(
        main_values[:, :448],
        prepack[0][written, :448],
        nope[:, 448:455].view(ml_dtypes.float8_e8m0fnu).astype(np.float32),
    )
    rope_budget = ref.assert_quantized_output(main_values[:, 448:], prepack[0][written, 448:])
    ref.assert_quantized_output(
        ref.decode_index(index),
        prepack[1][written],
        index[:, 128:129].view(ml_dtypes.float8_e8m0fnu).astype(np.float32),
    )
    decoded = [
        ref.decode_main(v[2].reshape(-1, 512), v[3].reshape(-1, 128)) for v in (actual, expected)
    ]
    # Both caches originate from the same independent pre-quantization reference.
    # The reference cache's own encoding error is known without inspecting TPU output.
    cache_budget = np.concatenate((nope_budget, rope_budget), axis=-1)
    cache_budget += np.abs(decoded[1][written] - prepack[0][written])
    fixed_tolerance_violations, max_difference = 0, 0.0
    for request in np.unique(owners[written]):
        cache, reference = (v[owners == request] for v in decoded)
        # Identical queries: random plus cache-aligned probes that concentrate attention.
        queries = np.concatenate(
            (np.random.default_rng(0).normal(size=(8, 512)).astype(np.float32), reference)
        )
        outputs = [ref.attention_from_cache(queries, v) for v in (cache, reference)]
        assert all(np.isfinite(v).all() for v in outputs)
        difference = np.abs(outputs[0] - outputs[1])
        arithmetic = ref.ARITHMETIC_ATOL + ref.ARITHMETIC_RTOL * np.abs(outputs[1])
        propagated = ref.attention_error_budget(
            queries, reference, cache_budget[owners[written] == request]
        )
        assert np.isfinite(propagated).all()
        assert (
            difference <= propagated + arithmetic
        ).all(), "attention exceeds propagated cache budget"
        fixed_tolerance_violations += int(np.count_nonzero(difference > arithmetic))
        max_difference = max(max_difference, float(difference.max()))
    if record_property is not None:
        record_property("attention_fixed_tolerance_violations", fixed_tolerance_violations)
        record_property("attention_max_abs_difference", max_difference)


def execute(operands, metadata):
    return tuple(
        np.asarray(v)
        for v in jax.block_until_ready(
            csa_compressor(
                *[jnp.asarray(v) for v in operands],
                metadata,
                schedule=get_compressor_schedule(
                    operands[0].shape[1], device_kind=jax.devices()[0].device_kind
                ),
            )
        )
    )


@pytest.mark.parametrize(
    "lengths,prefixes",
    [
        ((1, 1, 1, 1), (0, 1, 2, 3)),
        ((128, 128), (0, 4)),
        ((1, 4, 7, 0), (7, 0, 2, 0)),
        ((2, 2), (1, 2)),
        ((3, 5), (3, 1)),
        ((0, 0), (0, 0)),
        ((132,), (0,)),
    ],
)
def test_compressor_operator(lengths, prefixes):
    operands, metadata = make_case(lengths, prefixes)
    check(execute(operands, metadata), oracle(operands, metadata), operands)


@pytest.mark.parametrize("hidden", [4096, 7168])
def test_compressor_widths(hidden):
    operands, metadata = make_case((8,), (0,), hidden=hidden)
    check(execute(operands, metadata), oracle(operands, metadata), operands)


@pytest.mark.parametrize("seed", [2026091702, 2026091808])
def test_compressor_projection_rounding_regression(seed):
    # Independent inputs catch FP8 boundary failures hidden by a single passing seed.
    operands, metadata = make_case((128, 128), (0, 4), seed=seed)
    check(execute(operands, metadata), oracle(operands, metadata), operands)


@pytest.mark.parametrize("hidden,seed", [(4096, 31003), (7168, 31002)])
def test_compressor_benchmark_input_regression(hidden, seed, record_property):
    # Reproduce numerical benchmark inputs independently of its timing harness.
    operands, metadata = make_case((512,), (0,), hidden=hidden, pad=0)
    rng = np.random.default_rng(seed)
    slot = int(np.asarray(metadata.state_indices)[0])
    for state, width in zip(operands[8:10], (1024, 256), strict=True):
        state[slot] = rng.normal(0, 0.05, (8, 2, width)).astype(np.float32)
        state[slot, :, 0] = 0
        state[slot, :, 1] = -np.inf
    operands[0] = rng.normal(0, 0.3, (512, hidden)).astype(ml_dtypes.bfloat16)
    operands[1] = rng.normal(0, 0.005, (hidden, 2560)).astype(ml_dtypes.bfloat16)
    for index, width in ((2, 1024), (3, 256)):
        operands[index] = rng.normal(0, 0.02, (4, width)).astype(np.float32)
    operands[6] = np.ones_like(operands[6])
    operands[7] = np.zeros_like(operands[7])
    check_layered_prefill(operands, metadata, record_property)
    check(execute(operands, metadata), oracle(operands, metadata), operands, record_property)


def test_compressor_mutations_are_detected():
    operands, metadata = make_case((4,), (0,), zero=True)
    # A missing zero write must differ numerically, not only in FP8 storage bits.
    records = (
        *ref.pack_main(np.ones((1, 512), np.float32)),
        ref.pack_index(np.ones((1, 128), np.float32)),
    )
    for cache, record in zip(operands[10:], records, strict=True):
        cache.reshape(-1, record.shape[-1])[:] = record
    expected = oracle(operands, metadata)
    actual = execute(operands, metadata)
    check(actual, expected, operands)
    missing_write = metadata._replace(cache_locations=jnp.full_like(metadata.cache_locations, -1))
    with pytest.raises(AssertionError):
        check(execute(operands, missing_write), expected, operands)
    corrupted = list(actual)
    corrupted[0] = corrupted[0].copy()
    corrupted[0][1, 0, 0, 0] += 1
    with pytest.raises(AssertionError):
        check(corrupted, expected, operands)
    for offset, column in ((2, 0), (2, 448), (4, 0), (4, 128)):
        corrupted = [v.copy() for v in actual]
        width = 512 if offset == 2 else 256
        destination = np.flatnonzero(expected.owners >= 0)[0]
        corrupted[offset].reshape(-1, width)[destination, column] = 0x7F
        with pytest.raises(AssertionError):
            check(corrupted, expected, operands)
    corrupted = [v.copy() for v in actual]
    corrupted[2].reshape(-1, 512)[0, 0] ^= 1
    with pytest.raises(AssertionError):
        check(corrupted, expected, operands)
    attention = ref.attention_from_cache
    calls = 0

    def corrupt_attention(query, cache):
        nonlocal calls
        calls += 1
        return attention(query, cache) + (1 if calls == 1 else 0)

    with (
        patch.object(ref, "attention_from_cache", corrupt_attention),
        pytest.raises(AssertionError),
    ):
        check(actual, expected, operands)


def test_compressor_metadata_values_do_not_recompile():
    operands, metadata = make_case((1, 1), (2, 3), seed=14)
    execute(operands, metadata)
    cache_size = csa_compressor._cache_size()
    # Same shapes, different positions, emit flags, state slots and destinations.
    other, changed = make_case((1, 1), (3, 2), seed=15)
    check(execute(other, changed), oracle(other, changed), other)
    assert csa_compressor._cache_size() == cache_size


def test_compressor_invalid_shape():
    operands, metadata = make_case((1,), (0,))
    operands[0] = operands[0].astype(np.float32)
    with pytest.raises(ValueError, match="BF16"):
        execute(operands, metadata)


@pytest.mark.parametrize("norm_eps", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0])
def test_compressor_invalid_norm_eps(norm_eps):
    operands, metadata = make_case((1,), (0,))
    with pytest.raises(ValueError, match="finite and positive"):
        csa_compressor(
            *[jnp.asarray(v) for v in operands],
            metadata,
            schedule=get_compressor_schedule(128, device_kind="TPU v6e"),
            norm_eps=norm_eps,
        )


@pytest.mark.parametrize("hidden", [4096, 7168])
@pytest.mark.parametrize(
    "lengths,prefixes",
    [
        ((1, 1, 1, 1), (0, 1, 2, 3)),
        ((1, 1, 1, 1), (4, 5, 6, 8)),
        ((1, 4, 7, 0), (7, 0, 2, 0)),
    ],
    ids=["decode_mixed", "decode_update_only", "ragged"],
)
def test_compressor_deployment_widths(hidden, lengths, prefixes):
    operands, metadata = make_case(lengths, prefixes, hidden=hidden)
    check(execute(operands, metadata), oracle(operands, metadata), operands)


@pytest.mark.parametrize("hidden", [128, 4096, 7168])
@pytest.mark.parametrize("chunks", [(3, 1, 4, 2, 5), (6, 1, 1, 1, 1)])
def test_compressor_stateful_chunk_chain(chunks, hidden):
    full, full_meta = make_case((sum(chunks),), (0,), hidden=hidden, seed=21)
    actual_state = [v.copy() for v in full[8:]]
    expected_state = [v.copy() for v in full[8:]]
    start = 0
    for length in chunks:
        chunk, metadata = make_case((length,), (start,), hidden=hidden, seed=22)
        chunk[:8] = [full[0][start : start + length].copy(), *full[1:8]]
        chunk[0] = np.pad(chunk[0], ((0, 3), (0, 0)))
        locations = np.pad(
            np.asarray(full_meta.cache_locations)[start : start + length],
            (0, 3),
            constant_values=-1,
        )
        metadata = metadata._replace(cache_locations=jnp.asarray(locations))
        actual_inputs = [*chunk[:8], *actual_state]
        expected_inputs = [*chunk[:8], *expected_state]
        expected = oracle(expected_inputs, metadata)
        expected_state = list(expected.outputs)
        actual_state = list(execute(actual_inputs, metadata))
        check(actual_state, expected, actual_inputs)
        start += length
    check(actual_state, oracle(full, full_meta), full)


def test_compressor_padding_cannot_write():
    operands, metadata = make_case((4, 0), (0, 0), seed=23)
    operands[0][-3:] = np.nan
    metadata = metadata._replace(
        cache_locations=metadata.cache_locations.at[-3:].set(0),
        positions=metadata.positions.at[-3:].set(3),
        blocks=(
            *metadata.blocks,
            CompressorBlock(jnp.asarray([[4]], jnp.int32), jnp.asarray([1], jnp.int32)),
        ),
    )
    check(execute(operands, metadata), oracle(operands, metadata), operands)


@pytest.mark.parametrize("batch", [4, 40])
def test_compressor_shared_packed_write_group(batch):
    operands, metadata = make_case((1,) * batch, (3,) * batch, seed=24)
    check(execute(operands, metadata), oracle(operands, metadata), operands)


@pytest.mark.parametrize(
    "page_size,placement",
    [
        (128, "aligned"),
        (128, "cross_page"),
        (128, "shuffled"),
        (128, "tail"),
        (16, "aligned"),
        (4, "aligned"),
    ],
)
def test_compressor_cache_write_layout(page_size, placement):
    operands, metadata = make_case((516 if placement == "tail" else 512,), (0,), seed=25)
    capacity = operands[10].size // 512
    operands[10] = operands[10].reshape(capacity // page_size, page_size, 4, 128)
    for i, width in ((11, 128), (12, 256)):
        operands[i] = operands[i].reshape(capacity // page_size, page_size // 4, 4, width)
    locations = np.asarray(metadata.cache_locations).copy()
    emitted = np.flatnonzero(locations >= 0)
    destinations = np.arange(len(emitted), dtype=np.int32)
    if placement == "cross_page":
        destinations += 113
    elif placement == "shuffled":
        np.random.default_rng(26).shuffle(destinations)
    locations[emitted] = destinations
    if placement == "shuffled":
        locations[emitted[::7]] = -1
    metadata = metadata._replace(cache_locations=jnp.asarray(locations))
    check(execute(operands, metadata), oracle(operands, metadata), operands)


@pytest.mark.parametrize("length", [1, 512])
@pytest.mark.parametrize("out_of_bounds", [False, True])
def test_compressor_suppressed_cache_writes(length, out_of_bounds):
    operands, metadata = make_case((length,), (3 if length == 1 else 0,), seed=27)
    no_writes = metadata._replace(cache_locations=jnp.full_like(metadata.cache_locations, -1))
    location = operands[10].size // 512 if out_of_bounds else -1
    metadata = metadata._replace(cache_locations=jnp.full_like(metadata.cache_locations, location))
    check(execute(operands, metadata), oracle(operands, no_writes), operands)


@pytest.mark.parametrize("placement", ["holes", "shuffled", "cross_page", "tail"])
def test_compressor_mixed_cache_writes(placement):
    length = {"tail": 1028, "shuffled": 1540}.get(placement, 1024)
    operands, metadata = make_case((length,), (0,), seed=28)
    locations = np.asarray(metadata.cache_locations).copy()
    emitted = np.flatnonzero(locations >= 0)
    destinations = np.arange(len(emitted), dtype=np.int32)
    if placement == "holes":
        destinations[128::7] = -1
    elif placement == "shuffled":
        # Row writes share packed groups within/across scatter blocks, but never DMA groups.
        np.random.default_rng(29).shuffle(destinations[128:])
    elif placement == "cross_page":
        destinations[128:] += 1
    locations[emitted] = destinations
    metadata = metadata._replace(cache_locations=jnp.asarray(locations))
    check(execute(operands, metadata), oracle(operands, metadata), operands)


def test_compressor_interpret_is_forwarded(monkeypatch):
    from sgl_jax.srt.kernels.csa_compressor import compressor as module

    observed = []

    def step(x, state, *args, **kwargs):
        observed.append((kwargs["record_kind"], kwargs["interpret"]))
        width = 640 if kwargs["record_kind"] == "main" else 256
        return jnp.zeros((x.shape[0], width), jnp.uint8), jnp.zeros(x.shape[0], bool), state

    def prefill(x, main, index, *args, **kwargs):
        observed.append(("prefill", kwargs["interpret"]))
        batch, groups = x.shape[0], x.shape[1] // 4
        return (
            jnp.zeros((batch, groups, 4, 128), jnp.uint8),
            jnp.zeros((batch, groups, 128), jnp.uint8),
            jnp.zeros((batch, groups, 2, 128), jnp.uint8),
            main,
            index,
        )

    def store(*args, **kwargs):
        observed.append(("store", kwargs["interpret"]))
        return args[-3:]

    monkeypatch.setattr(module, "csa_state_step_fused_pallas", step)
    monkeypatch.setattr(module, "csa_dual_uniform_prefill_pallas", prefill)
    monkeypatch.setattr(module, "_write_fp8_cache_rows", store)
    operands, metadata = make_case((1, 4), (3, 0))
    arrays = [jnp.asarray(v) for v in operands]
    jax.eval_shape(
        lambda: csa_compressor.__wrapped__(
            *arrays,
            metadata,
            schedule=get_compressor_schedule(128, device_kind="TPU v6e"),
            interpret=True,
        )
    )
    assert sorted(observed) == [("index", True), ("main", True), ("prefill", True), ("store", True)]


def test_compressor_unknown_platform_is_rejected():
    with pytest.raises(ValueError, match="no calibrated schedule"):
        get_compressor_schedule(4096, device_kind="unknown")
