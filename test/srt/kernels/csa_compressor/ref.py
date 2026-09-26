"""Independent NumPy compressor oracle: BF16 inputs, FP32 arithmetic, then cache encoding."""

from __future__ import annotations

import ml_dtypes
import numpy as np

COMPRESSION_RATIO = 4
STATE_SLOTS = 8
ATTENTION_DIM = 512
INDEX_DIM = 128
ROPE_DIM = 64
ROPE_FREQUENCY_DIM = 32
NOPE_DIM = 448
FP8_BLOCK_SIZE = 64
NOPE_SCALE_COUNT = 7
NOPE_PADDING_BYTES = 57
INDEX_PADDING_BYTES = 127
FP8_AMAX_FLOOR = 1e-4
NORM_EPS = 1e-6
VECTOR_LANES = 128
DECODE_ROWS = 8
# Existing arithmetic acceptance budget, independent of cache quantization.
ARITHMETIC_RTOL = 2e-2
ARITHMETIC_ATOL = 1e-2


def assert_quantized_output(decoded, reference, scales=None):
    """Check membership in rounding cells enlarged by the arithmetic budget.

    FP8 uses the supplied E8M0 block scales; otherwise the output is BF16.
    End cells stop at the finite format limit: saturation is not exempted.
    """
    decoded, reference = np.asarray(decoded, np.float32), np.asarray(reference, np.float32)
    assert decoded.shape == reference.shape
    assert np.isfinite(decoded).all() and np.isfinite(reference).all()
    budget = ARITHMETIC_ATOL + ARITHMETIC_RTOL * np.abs(reference)
    if scales is None:
        rounded = decoded.astype(ml_dtypes.bfloat16)
        np.testing.assert_array_equal(rounded.astype(np.float32), decoded)
        low = np.nextafter(rounded, ml_dtypes.bfloat16(-np.inf)).astype(np.float32)
        high = np.nextafter(rounded, ml_dtypes.bfloat16(np.inf)).astype(np.float32)
        low = np.where(np.isfinite(low), low, decoded)
        high = np.where(np.isfinite(high), high, decoded)
        lower = decoded + (low - decoded) / np.float32(2)
        upper = decoded + (high - decoded) / np.float32(2)
    else:
        scales = np.asarray(scales, np.float32)
        assert scales.ndim == 2 and scales.shape[0] == reference.shape[0]
        assert np.isfinite(scales).all() and (scales > 0).all()
        block = reference.shape[1] // scales.shape[1]
        values = reference.reshape(len(reference), -1, block)
        errors = budget.reshape(values.shape)
        # A corrupted scale must not enlarge its own permissible error arbitrarily.
        amax_low = np.maximum(
            np.max(np.maximum(np.abs(values) - errors, 0), axis=-1), FP8_AMAX_FLOOR
        )
        amax_high = np.maximum(np.max(np.abs(values) + errors, axis=-1), FP8_AMAX_FLOOR)
        min_scale = np.exp2(np.ceil(np.log2(amax_low / np.float32(448))))
        max_scale = np.exp2(np.ceil(np.log2(amax_high / np.float32(448))))
        assert ((scales >= min_scale) & (scales <= max_scale)).all(), "invalid FP8 block scale"
        np.testing.assert_array_equal(np.log2(scales), np.floor(np.log2(scales)))
        scale = np.repeat(scales, block, axis=-1)
        positive = np.arange(127, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        levels = np.unique(np.concatenate((-positive, positive)))
        code = np.searchsorted(levels, decoded / scale)
        assert (code < len(levels)).all()
        np.testing.assert_array_equal(levels[code] * scale, decoded)
        lower = (levels[np.maximum(code - 1, 0)] + levels[code]) * np.float32(0.5) * scale
        upper = (
            (levels[np.minimum(code + 1, len(levels) - 1)] + levels[code]) * np.float32(0.5) * scale
        )
    violation = np.maximum(lower - reference, reference - upper) - budget
    assert (violation <= 0).all(), (
        f"{np.count_nonzero(violation > 0)} values outside quantization + arithmetic budget; "
        f"max excess={np.max(violation)}"
    )
    return budget + np.maximum(decoded - lower, upper - decoded)


def attention_error_budget(query, reference, cache_error):
    """Propagate independently checked per-element cache budgets, not measured errors.

    |delta logits| <= |Q| @ cache_error / sqrt(D). Bound each softmax
    probability with opposing logit endpoints, then bound delta(PV).
    This is a conservative sensitivity bound, not a model-quality threshold.
    """
    assert reference.shape == cache_error.shape
    assert np.isfinite(cache_error).all() and (cache_error >= 0).all()
    divisor = np.float32(np.sqrt(reference.shape[-1]))
    logits = np.matmul(query, reference.T, dtype=np.float32) / divisor
    logit_error = np.matmul(np.abs(query), cache_error.T, dtype=np.float32) / divisor
    shifted = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    weights /= weights.sum(axis=-1, keepdims=True, dtype=np.float32)
    offset = np.max(logits + logit_error, axis=-1, keepdims=True)
    low_mass = np.exp(logits - logit_error - offset)
    high_mass = np.exp(logits + logit_error - offset)

    def other_mass(mass):
        # Exclusive sums retain tiny other-key masses next to a dominant key.
        prefix = np.cumsum(mass, axis=-1, dtype=np.float32)
        suffix = np.cumsum(mass[:, ::-1], axis=-1, dtype=np.float32)[:, ::-1]
        return np.pad(prefix[:, :-1], ((0, 0), (1, 0))) + np.pad(suffix[:, 1:], ((0, 0), (0, 1)))

    lower = low_mass / (low_mass + other_mass(high_mass))
    upper = high_mass / (high_mass + other_mass(low_mass))
    probability_error = np.maximum(np.abs(weights - lower), np.abs(upper - weights))
    # sum(delta P)=0 permits centering V, tightening the weight-error term.
    center = (reference.min(axis=0) + reference.max(axis=0)) * np.float32(0.5)
    key_error = probability_error @ np.abs(reference - center)
    value_error = np.minimum(upper @ cache_error, cache_error.max(axis=0))
    return key_error + value_error


def attention_from_cache(query, cache):
    """FP32 downstream sensitivity probe; keys and values share the compressed cache."""
    logits = np.matmul(query, cache.T, dtype=np.float32) / np.float32(np.sqrt(cache.shape[-1]))
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True, dtype=np.float32)
    return np.matmul(probabilities, cache, dtype=np.float32)


def projection_blocks(positions, query_lengths):
    """Independent host schedule: aligned groups between single-token edges."""
    offsets = np.asarray((0, *np.cumsum(query_lengths)), np.int32)
    partitions = {}
    for request, length in enumerate(query_lengths):
        if length:
            slot = int(positions[offsets[request]] % COMPRESSION_RATIO)
            partitions.setdefault((length, slot), []).append(request)
    blocks = []
    for (length, slot), requests in sorted(partitions.items()):
        leading = min(length, (-slot) % COMPRESSION_RATIO)
        stop = leading + (length - leading) // COMPRESSION_RATIO * COMPRESSION_RATIO
        intervals = [(i, i + 1) for i in range(leading)]
        if stop > leading:
            intervals.append((leading, stop))
        intervals.extend((i, i + 1) for i in range(stop, length))
        for start, end in intervals:
            blocks.append(offsets[np.asarray(requests), None] + np.arange(start, end)[None])
    return blocks


def compressor_projection(x, weight, blocks, *, k_tile, query_tile):
    """BF16 operands, FP32 tile products and ordered K-tile additions.

    Schedule sizes are supplied by the test caller, not imported from production.
    Every hidden size uses the same arithmetic; this is not a bit-exact TPU emulator.
    """
    x = np.asarray(x, ml_dtypes.bfloat16).astype(np.float32)
    weight = np.asarray(weight, ml_dtypes.bfloat16).astype(np.float32)
    if k_tile <= 0 or x.shape[1] % k_tile or query_tile <= 0:
        raise ValueError("invalid projection tiles")
    result = np.zeros((len(x), weight.shape[1]), np.float32)

    def project(ids, rows, columns):
        active = (ids >= 0) & (ids < len(x))
        inputs = np.zeros((rows, x.shape[1]), np.float32)
        inputs[np.flatnonzero(active)] = x[ids[active]]
        matrix = weight[:, columns]
        acc = np.zeros((rows, matrix.shape[1]), np.float32)
        for start in range(0, x.shape[1], k_tile):
            tile = np.matmul(
                inputs[:, start : start + k_tile], matrix[start : start + k_tile], dtype=np.float32
            )
            acc = np.add(acc, tile, dtype=np.float32)
        result[ids[active], columns] = acc[np.flatnonzero(active)]

    for block in blocks:
        block = np.asarray(block, np.int32)
        if block.shape[1] == 1:
            for start in range(0, len(block), DECODE_ROWS):
                ids = block[start : start + DECODE_ROWS, 0]
                # Decode projects main and index separately; prefill uses a fused weight.
                for columns in (slice(0, 4 * ATTENTION_DIM), slice(4 * ATTENTION_DIM, None)):
                    project(ids, DECODE_ROWS, columns)
        else:
            rows = min(block.shape[1], query_tile)
            for ids in block:
                for start in range(0, len(ids), rows):
                    project(ids[start : start + rows], rows, slice(None))
    return result


def _pool(state, norm, cos, sin):
    norm = np.asarray(norm, np.float32)
    cos, sin = np.asarray(cos, np.float32), np.asarray(sin, np.float32)
    head_dim = norm.shape[0]
    kv_halves = state[:, :, 0].reshape(state.shape[0], STATE_SLOTS, 2, head_dim)
    score_halves = state[:, :, 1].reshape(state.shape[0], STATE_SLOTS, 2, head_dim)
    values = np.concatenate(
        (
            kv_halves[:, :COMPRESSION_RATIO, 0],
            kv_halves[:, COMPRESSION_RATIO:, 1],
        ),
        axis=1,
    )
    logits = np.concatenate(
        (
            score_halves[:, :COMPRESSION_RATIO, 0],
            score_halves[:, COMPRESSION_RATIO:, 1],
        ),
        axis=1,
    )
    shape = (len(state), STATE_SLOTS, head_dim // VECTOR_LANES, VECTOR_LANES)
    values, logits = values.reshape(shape), logits.reshape(shape)
    # TPU lowering uses exp2(x * log2(e)) and reciprocal-then-multiply for division.
    shifted = np.subtract(logits, logits.max(axis=1, keepdims=True), dtype=np.float32)
    coefficients = np.exp2(np.multiply(shifted, np.float32(np.log2(np.e)), dtype=np.float32))
    inverse_sum = np.reciprocal(
        coefficients.sum(axis=1, keepdims=True, dtype=np.float32), dtype=np.float32
    )
    coefficients = np.multiply(coefficients, inverse_sum, dtype=np.float32)
    pooled = np.sum(values * coefficients, axis=1, dtype=np.float32)
    inverse_rms = np.float32(1) / np.sqrt(
        np.mean(np.square(pooled), axis=(1, 2), keepdims=True, dtype=np.float32)
        + np.float32(NORM_EPS)
    )
    pooled = (pooled * inverse_rms * norm.reshape(1, *shape[-2:])).reshape(len(state), head_dim)
    rope = pooled[:, -ROPE_DIM:].reshape(-1, ROPE_FREQUENCY_DIM, 2)
    real = rope[..., 0].copy()
    imaginary = rope[..., 1].copy()
    rope[..., 0] = real * cos - imaginary * sin
    rope[..., 1] = real * sin + imaginary * cos
    pooled[:, -ROPE_DIM:] = rope.reshape(-1, ROPE_DIM)
    return pooled


def compressor_step(projection, state, ape, norm, cos, sin, positions):
    """Consume the independently computed projection, update overlap state, and pool."""
    state = np.array(state, np.float32, copy=True)
    ape = np.asarray(ape, np.float32)
    projection = np.asarray(projection, np.float32)
    projected_dim = 2 * norm.shape[0]
    values = projection[:, :projected_dim]
    scores = projection[:, projected_dim:]
    emit = np.mod(positions + 1, COMPRESSION_RATIO) == 0
    for row, position in enumerate(positions):
        slot = int(position % COMPRESSION_RATIO) + COMPRESSION_RATIO
        state[row, slot, 0] = values[row]
        state[row, slot, 1] = scores[row] + ape[position % COMPRESSION_RATIO]
    source = np.maximum(positions + 1 - COMPRESSION_RATIO, 0)
    pooled = _pool(state, norm, cos[source], sin[source])
    for row in np.flatnonzero(emit):
        current = state[row, COMPRESSION_RATIO:].copy()
        state[row] = np.concatenate((current, current), axis=0)
    return pooled, emit, state


def compressor_ragged(
    projection,
    state,
    ape,
    norm,
    cos,
    sin,
    positions,
    query_lengths,
):
    """Sequential specification for a packed ragged compressor chunk."""
    state = np.asarray(state, np.float32).copy()
    emitted_positions = []
    emitted_values = []
    token_start = 0
    for request, query_length in enumerate(query_lengths):
        request_positions = []
        request_values = []
        request_state = state[request : request + 1]
        for token in range(token_start, token_start + query_length):
            pooled, emit, request_state = compressor_step(
                projection[token : token + 1],
                request_state,
                ape,
                norm,
                cos,
                sin,
                positions[token : token + 1],
            )
            if emit[0]:
                request_positions.append(int(positions[token]))
                request_values.append(pooled[0])
        state[request] = request_state[0]
        emitted_positions.append(np.asarray(request_positions, np.int32))
        emitted_values.append(np.asarray(request_values, np.float32).reshape(-1, norm.shape[0]))
        token_start += query_length
    return tuple(emitted_positions), tuple(emitted_values), state


def _encode_fp8(values):
    # Independent nearest-even lookup for the FP32 values supplied to the encoder.
    values = np.asarray(values, np.float32)
    levels = np.arange(127, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    magnitude = np.abs(values)
    assert np.isfinite(values).all() and (magnitude <= levels[-1]).all()
    high = np.minimum(np.searchsorted(levels, magnitude), len(levels) - 1)
    low = np.maximum(high - 1, 0)
    low_error, high_error = magnitude - levels[low], levels[high] - magnitude
    choose_high = (high_error < low_error) | ((high_error == low_error) & (high % 2 == 0))
    codes = np.where(choose_high, high, low).astype(np.uint8)
    return codes | (np.signbit(values).astype(np.uint8) << 7)


def pack_main(pooled):
    """Encode one CSA main record as FP8 NoPE plus big-endian BF16 RoPE."""
    pooled = np.asarray(pooled, np.float32)
    fp8_max = np.float32(ml_dtypes.finfo(ml_dtypes.float8_e4m3fn).max)
    values = []
    scales = []
    for block in range(NOPE_SCALE_COUNT):
        block_values = pooled[:, block * FP8_BLOCK_SIZE : (block + 1) * FP8_BLOCK_SIZE]
        amax = np.maximum(
            np.max(np.abs(block_values), axis=-1, keepdims=True),
            FP8_AMAX_FLOOR,
        )
        scale = np.exp2(np.ceil(np.log2(amax / fp8_max))).astype(np.float32)
        values.append(_encode_fp8(block_values / scale))
        scales.append(scale)
    scale_bytes = (np.concatenate(scales, axis=-1).view(np.uint32) >> np.uint32(23)).astype(
        np.uint8
    )
    nope = np.concatenate(
        (
            np.concatenate(values, axis=-1),
            scale_bytes,
            np.zeros((pooled.shape[0], NOPE_PADDING_BYTES), np.uint8),
        ),
        axis=-1,
    )
    rope_bits = pooled[:, -ROPE_DIM:].astype(ml_dtypes.bfloat16).view(np.uint16)
    rope = np.concatenate(
        (
            np.right_shift(rope_bits, 8).astype(np.uint8),
            np.bitwise_and(rope_bits, np.iinfo(np.uint8).max).astype(np.uint8),
        ),
        axis=-1,
    )
    return nope, rope


def pack_index(pooled):
    """Encode one Lightning-Indexer key and its E8M0 scale."""
    pooled = np.asarray(pooled, np.float32)
    fp8_max = np.float32(ml_dtypes.finfo(ml_dtypes.float8_e4m3fn).max)
    amax = np.maximum(
        np.max(np.abs(pooled), axis=-1, keepdims=True),
        FP8_AMAX_FLOOR,
    )
    scale = np.exp2(np.ceil(np.log2(amax / fp8_max))).astype(np.float32)
    values = _encode_fp8(pooled / scale)
    scale_byte = np.right_shift(scale.view(np.uint32), 23).astype(np.uint8)
    return np.concatenate(
        (
            values,
            scale_byte,
            np.zeros((pooled.shape[0], INDEX_PADDING_BYTES), np.uint8),
        ),
        axis=-1,
    )


def decode_main(nope, rope):
    values = nope[..., :NOPE_DIM].view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    scales = (
        nope[..., NOPE_DIM : NOPE_DIM + NOPE_SCALE_COUNT]
        .view(ml_dtypes.float8_e8m0fnu)
        .astype(np.float32)
    )
    nope_values = values * np.repeat(scales, FP8_BLOCK_SIZE, axis=-1)
    bits = np.left_shift(rope[..., :ROPE_DIM].astype(np.uint16), 8) | rope[..., ROPE_DIM:].astype(
        np.uint16
    )
    return np.concatenate(
        (nope_values, bits.view(ml_dtypes.bfloat16).astype(np.float32)),
        axis=-1,
    )


def decode_index(records):
    values = records[..., :INDEX_DIM].view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    scales = (
        records[..., INDEX_DIM : INDEX_DIM + 1].view(ml_dtypes.float8_e8m0fnu).astype(np.float32)
    )
    return values * scales
