from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.embeddings import (
    Llama3RotaryEmbedding,
    MRotaryEmbedding,
    ProportionalRotaryEmbedding,
    RotaryEmbedding,
    RotaryEmbeddingBase,
    YarnRotaryEmbedding,
    apply_interleaved_rope,
    apply_rotary_emb,
)
from sgl_jax.srt.models.grok import ScalingRotaryEmbedding, get_rope_scaling

_HEAD_SIZE = 128
_BASE = 1_000_000
_MROPE_SECTION = [16, 24, 24]


def _make_qk(num_tokens: int) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(42)
    query = rng.standard_normal((num_tokens, 3, _HEAD_SIZE), dtype=np.float32)
    key = rng.standard_normal((num_tokens, 2, _HEAD_SIZE), dtype=np.float32)
    return jnp.asarray(query, dtype=jnp.bfloat16), jnp.asarray(key, dtype=jnp.bfloat16)


def _reference_cos_sin(
    rotary_emb: RotaryEmbedding, positions: jax.Array
) -> tuple[jax.Array, jax.Array]:
    inv_freq = jnp.asarray(rotary_emb._inv_freq_np, dtype=jnp.float32)
    freqs = positions.astype(jnp.float32)[..., None] * inv_freq
    return jnp.cos(freqs), jnp.sin(freqs)


def _assert_array_equal(actual: jax.Array, expected: jax.Array) -> None:
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


# Independent float64 NumPy references, written from the published formulas
# (HF transformers / upstream sglang) rather than from the classes under test.
# get_cos_sin is compared against them with a tolerance, so a wrong
# implementation cannot agree with itself and host-dependent float32 pow
# rounding does not matter.
_COS_SIN_ATOL = 1e-2  # bf16 rounding of values in [-1, 1] is at most 4e-3
_QK_ATOL = 6e-2  # bf16 RoPE outputs for |q|, |k| up to ~5


def _np_default_inv_freq(base: float, dim: int) -> np.ndarray:
    return 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))


def _np_cos_sin(
    positions: jax.Array, inv_freq: np.ndarray, mscale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.asarray(positions, dtype=np.float64)[..., None] * inv_freq
    return np.cos(freqs) * mscale, np.sin(freqs) * mscale


def _np_proportional_inv_freq(base: float, head_size: int, partial_rotary_factor: float):
    # HF `_compute_proportional_rope_parameters` (factor=1).
    rope_angles = int(partial_rotary_factor * head_size // 2)
    inv_freq = 1.0 / (base ** (np.arange(0, 2 * rope_angles, 2, dtype=np.float64) / head_size))
    return np.concatenate([inv_freq, np.zeros(head_size // 2 - rope_angles)])


def _np_llama3_inv_freq(base, dim, factor, low_freq_factor, high_freq_factor, old_context_len):
    # HF `_compute_llama3_parameters`.
    inv_freq = _np_default_inv_freq(base, dim)
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * np.pi / inv_freq
    inv_freq_llama = np.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
    return np.where(is_medium_freq, smoothed, inv_freq_llama)


def _np_yarn_ramp(beta_fast, beta_slow, dim, base, max_position):
    def correction_dim(num_rotations):
        return (dim * np.log(max_position / (num_rotations * 2 * np.pi))) / (2 * np.log(base))

    low = max(np.floor(correction_dim(beta_fast)), 0)
    high = min(np.ceil(correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    return np.clip((np.arange(dim // 2, dtype=np.float64) - low) / (high - low), 0, 1)


def _np_yarn_inv_freq(base, dim, factor, original_max_position, beta_fast=32, beta_slow=1):
    # HF `_compute_yarn_parameters` (truncate=True); attention factor 0.1 * ln(factor) + 1.
    inv_freq_extrapolation = _np_default_inv_freq(base, dim)
    extrapolation_factor = 1 - _np_yarn_ramp(beta_fast, beta_slow, dim, base, original_max_position)
    inv_freq = (inv_freq_extrapolation / factor) * (
        1 - extrapolation_factor
    ) + inv_freq_extrapolation * extrapolation_factor
    return inv_freq, 0.1 * np.log(factor) + 1.0


def _np_grok_inv_freq(base, dim, scaling_factor, max_position, extra_method, beta_fast=32):
    # Upstream sglang grok.py ScalingRotaryEmbedding._compute_inv_freq. Its
    # _compute_cos_sin_cache leaves cos/sin unscaled (mscale is not applied).
    inv_freq_extrapolation = _np_default_inv_freq(base, dim)
    if extra_method == "original":
        return inv_freq_extrapolation
    assert extra_method == "yarn_log", extra_method
    mask = 1 - _np_yarn_ramp(beta_fast, 1, dim, base, max_position)
    return np.exp(
        np.log(inv_freq_extrapolation) * mask
        + np.log(inv_freq_extrapolation / scaling_factor) * (1.0 - mask)
    )


def _np_mrope_cos_sin(positions, inv_freq, mrope_section, interleaved):
    positions = np.asarray(positions).reshape(-1, np.shape(positions)[-1])
    cos_all, sin_all = _np_cos_sin(np.broadcast_to(positions, (3, positions.shape[-1])), inv_freq)
    dims = np.arange(cos_all.shape[-1])
    if interleaved:
        axis = np.zeros_like(dims)
        axis[(dims % 3 == 1) & (dims < 3 * mrope_section[1])] = 1
        axis[(dims % 3 == 2) & (dims < 3 * mrope_section[2])] = 2
    else:
        axis = np.repeat(np.arange(3), mrope_section)
    return cos_all[axis, :, dims].T, sin_all[axis, :, dims].T


def _np_apply_neox(x: jax.Array, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    half = cos.shape[-1]
    x1, x2 = x[..., :half], x[..., half : 2 * half]
    cos, sin = cos[:, None, :], sin[:, None, :]
    return np.concatenate((x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., 2 * half :]), axis=-1)


def _assert_close(actual: jax.Array, expected: np.ndarray, atol: float, err_msg: str = "") -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64), expected, rtol=0, atol=atol, err_msg=err_msg
    )


def test_rotary_embedding_computes_phase_in_float32():
    rotary_emb = RotaryEmbedding(
        head_size=_HEAD_SIZE,
        rotary_dim=_HEAD_SIZE,
        max_position_embeddings=32768,
        base=_BASE,
        is_neox_style=True,
        dtype=jnp.bfloat16,
    )
    positions = jnp.asarray([0, 127, 1023, 4095], dtype=jnp.int32)
    query, key = _make_qk(positions.size)

    actual_query, actual_key = rotary_emb(positions, query, key)
    cos, sin = _reference_cos_sin(rotary_emb, positions)
    expected_query = apply_rotary_emb(query, cos, sin, is_neox_style=True)
    expected_key = apply_rotary_emb(key, cos, sin, is_neox_style=True)

    _assert_array_equal(actual_query, expected_query)
    _assert_array_equal(actual_key, expected_key)

    low_precision_inv_freq = jnp.asarray(rotary_emb._inv_freq_np, dtype=jnp.bfloat16)
    low_precision_freqs = positions.astype(jnp.float32)[:, None] * low_precision_inv_freq
    low_precision_query = apply_rotary_emb(
        query,
        jnp.cos(low_precision_freqs).astype(jnp.bfloat16),
        jnp.sin(low_precision_freqs).astype(jnp.bfloat16),
        is_neox_style=True,
    )
    assert not np.array_equal(np.asarray(actual_query), np.asarray(low_precision_query))


@pytest.mark.parametrize("interleaved", [False, True])
def test_mrotary_embedding_computes_phase_in_float32(interleaved: bool):
    rotary_emb = MRotaryEmbedding(
        head_size=_HEAD_SIZE,
        rotary_dim=_HEAD_SIZE,
        max_position_embeddings=32768,
        base=_BASE,
        is_neox_style=True,
        dtype=jnp.bfloat16,
        mrope_section=_MROPE_SECTION,
        mrope_interleaved=interleaved,
    )
    positions = jnp.asarray(
        [
            [0, 127, 1023, 4095],
            [0, 63, 511, 2047],
            [0, 31, 255, 1023],
        ],
        dtype=jnp.int32,
    )
    query, key = _make_qk(positions.shape[-1])

    actual_query, actual_key = rotary_emb(positions, query, key)
    cos_all, sin_all = _reference_cos_sin(rotary_emb, positions)
    if interleaved:
        cos = apply_interleaved_rope(cos_all, _MROPE_SECTION)
        sin = apply_interleaved_rope(sin_all, _MROPE_SECTION)
    else:
        split_indices = np.cumsum(_MROPE_SECTION)[:-1].tolist()
        cos = jnp.concatenate(
            [part[axis] for axis, part in enumerate(jnp.split(cos_all, split_indices, axis=-1))],
            axis=-1,
        )
        sin = jnp.concatenate(
            [part[axis] for axis, part in enumerate(jnp.split(sin_all, split_indices, axis=-1))],
            axis=-1,
        )
    expected_query = apply_rotary_emb(query, cos, sin, is_neox_style=True)
    expected_key = apply_rotary_emb(key, cos, sin, is_neox_style=True)

    _assert_array_equal(actual_query, expected_query)
    _assert_array_equal(actual_key, expected_key)


def test_yarn_rotary_embedding_computes_phase_in_float32():
    rotary_emb = YarnRotaryEmbedding(
        head_size=_HEAD_SIZE,
        rotary_dim=_HEAD_SIZE,
        max_position_embeddings=32768,
        base=_BASE,
        is_neox_style=True,
        dtype=jnp.bfloat16,
        scaling_factor=4.0,
        original_max_position_embeddings=8192,
    )
    positions = jnp.asarray([0, 127, 1023, 4095], dtype=jnp.int32)
    query, key = _make_qk(positions.size)

    actual_query, actual_key = rotary_emb(positions, query, key)
    cos, sin = _reference_cos_sin(rotary_emb, positions)
    cos = (cos.astype(jnp.float32) * rotary_emb._rope_mscale).astype(jnp.bfloat16)
    sin = (sin.astype(jnp.float32) * rotary_emb._rope_mscale).astype(jnp.bfloat16)
    expected_query = apply_rotary_emb(query, cos, sin, is_neox_style=True)
    expected_key = apply_rotary_emb(key, cos, sin, is_neox_style=True)

    _assert_array_equal(actual_query, expected_query)
    _assert_array_equal(actual_key, expected_key)


@pytest.mark.parametrize("rotary_dim", [64, _HEAD_SIZE])
def test_get_cos_sin_matches_direct_call_for_standard_variants(rotary_dim: int):
    positions = jnp.asarray([0, 127, 1023, 4095], dtype=jnp.int32)
    query, key = _make_qk(positions.size)

    variants: list[RotaryEmbeddingBase] = [
        RotaryEmbedding(
            head_size=_HEAD_SIZE,
            rotary_dim=rotary_dim,
            max_position_embeddings=32768,
            base=_BASE,
            is_neox_style=True,
            dtype=jnp.bfloat16,
        ),
        ProportionalRotaryEmbedding(
            head_size=_HEAD_SIZE,
            max_position_embeddings=32768,
            base=_BASE,
            is_neox_style=True,
            dtype=jnp.bfloat16,
            partial_rotary_factor=rotary_dim / _HEAD_SIZE,
        ),
        Llama3RotaryEmbedding(
            head_size=_HEAD_SIZE,
            rotary_dim=rotary_dim,
            max_position_embeddings=32768,
            base=_BASE,
            is_neox_style=True,
            dtype=jnp.bfloat16,
            scaling_factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            orig_max_position=8192,
        ),
        YarnRotaryEmbedding(
            head_size=_HEAD_SIZE,
            rotary_dim=rotary_dim,
            max_position_embeddings=32768,
            base=_BASE,
            is_neox_style=True,
            dtype=jnp.bfloat16,
            scaling_factor=4.0,
            original_max_position_embeddings=8192,
        ),
        ScalingRotaryEmbedding(
            head_size=_HEAD_SIZE,
            rotary_dim=rotary_dim,
            max_position_embeddings=32768,
            base=_BASE,
            is_neox_style=True,
            scaling_factor=4.0,
            dtype=jnp.bfloat16,
            extra_method="yarn_log",
        ),
    ]

    # Independent reference (inv_freq, cos/sin magnitude scale) for each variant.
    references = [
        (_np_default_inv_freq(_BASE, rotary_dim), 1.0),
        (_np_proportional_inv_freq(_BASE, _HEAD_SIZE, rotary_dim / _HEAD_SIZE), 1.0),
        (_np_llama3_inv_freq(_BASE, rotary_dim, 8.0, 1.0, 4.0, 8192), 1.0),
        _np_yarn_inv_freq(_BASE, rotary_dim, 4.0, 8192),
        (_np_grok_inv_freq(_BASE, rotary_dim, 4.0, 32768, "yarn_log"), 1.0),
    ]

    for rope, (ref_inv_freq, ref_mscale) in zip(variants, references, strict=True):
        name = type(rope).__name__
        cos_sin = rope.get_cos_sin(positions)
        ref_cos, ref_sin = _np_cos_sin(positions, ref_inv_freq, ref_mscale)
        _assert_close(cos_sin[0], ref_cos, _COS_SIN_ATOL, name)
        _assert_close(cos_sin[1], ref_sin, _COS_SIN_ATOL, name)

        expected_q, expected_k = rope(positions, query, key)
        _assert_close(expected_q, _np_apply_neox(query, ref_cos, ref_sin), _QK_ATOL, name)
        _assert_close(expected_k, _np_apply_neox(key, ref_cos, ref_sin), _QK_ATOL, name)

        actual_q, actual_k = rope(positions, query, key, cos_sin=cos_sin)
        _assert_array_equal(actual_q, expected_q)
        _assert_array_equal(actual_k, expected_k)


def test_llama3_rotary_get_cos_sin_uses_scaled_frequencies():
    positions = jnp.asarray([1, 256, 2048, 8192], dtype=jnp.int32)
    base_rope = RotaryEmbedding(
        head_size=_HEAD_SIZE,
        rotary_dim=_HEAD_SIZE,
        max_position_embeddings=32768,
        base=500_000,
        is_neox_style=True,
        dtype=jnp.bfloat16,
    )
    llama3_rope = Llama3RotaryEmbedding(
        head_size=_HEAD_SIZE,
        rotary_dim=_HEAD_SIZE,
        max_position_embeddings=32768,
        base=500_000,
        is_neox_style=True,
        dtype=jnp.bfloat16,
        scaling_factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        orig_max_position=8192,
    )

    # Llama 3.1 rope_scaling, checked against the HF transformers llama3 formula.
    expected_inv_freq = _np_llama3_inv_freq(500_000, _HEAD_SIZE, 8.0, 1.0, 4.0, 8192)
    np.testing.assert_allclose(llama3_rope._inv_freq_np, expected_inv_freq, rtol=1e-5)

    cos, sin = llama3_rope.get_cos_sin(positions)
    expected_cos, expected_sin = _np_cos_sin(positions, expected_inv_freq)
    _assert_close(cos, expected_cos, _COS_SIN_ATOL)
    _assert_close(sin, expected_sin, _COS_SIN_ATOL)

    query, key = _make_qk(positions.size)
    actual_q, actual_k = llama3_rope(positions, query, key)
    _assert_close(actual_q, _np_apply_neox(query, expected_cos, expected_sin), _QK_ATOL)
    _assert_close(actual_k, _np_apply_neox(key, expected_cos, expected_sin), _QK_ATOL)

    base_cos, _ = base_rope.get_cos_sin(positions)
    assert not jnp.array_equal(cos, base_cos)


def test_grok2_scaling_rotary_does_not_apply_mscale():
    # RoPE fields of the public xai-org/grok-2 config.json, built the way
    # Grok1Attention builds its rotary embedding.
    grok2_config = SimpleNamespace(
        rope_type="original",
        original_max_position_embeddings=8192,
        scaling_factor=16.0,
        extrapolation_factor=1.0,
        attn_factor=1.0,
        beta_fast=8,
        beta_slow=1,
    )
    rope_theta = 208533496
    rope = ScalingRotaryEmbedding(
        _HEAD_SIZE,
        rotary_dim=_HEAD_SIZE,
        base=rope_theta,
        is_neox_style=True,
        **get_rope_scaling(grok2_config),
    )
    assert rope.mscale == 4.0  # sqrt(scaling_factor); must not scale cos/sin

    positions = jnp.asarray([0, 1, 4095, 8191, 131071], dtype=jnp.int32)
    cos, sin = rope.get_cos_sin(positions)
    np.testing.assert_array_equal(np.asarray(cos)[0], 1.0)
    np.testing.assert_array_equal(np.asarray(sin)[0], 0.0)
    expected_cos, expected_sin = _np_cos_sin(
        positions, _np_grok_inv_freq(rope_theta, _HEAD_SIZE, 16.0, 8192, "original")
    )
    _assert_close(cos, expected_cos, _COS_SIN_ATOL)
    _assert_close(sin, expected_sin, _COS_SIN_ATOL)

    query, key = _make_qk(positions.size)
    actual_q, actual_k = rope(positions, query, key)
    _assert_close(actual_q, _np_apply_neox(query, expected_cos, expected_sin), _QK_ATOL)
    _assert_close(actual_k, _np_apply_neox(key, expected_cos, expected_sin), _QK_ATOL)


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("pos_mode", ["1d", "2d_single", "3d"])
def test_mrotary_get_cos_sin_and_sharded_positions(interleaved: bool, pos_mode: str):
    rotary_emb = MRotaryEmbedding(
        head_size=_HEAD_SIZE,
        rotary_dim=_HEAD_SIZE,
        max_position_embeddings=32768,
        base=_BASE,
        is_neox_style=True,
        dtype=jnp.bfloat16,
        mrope_section=_MROPE_SECTION,
        mrope_interleaved=interleaved,
    )
    if pos_mode == "1d":
        positions = jnp.asarray([0, 127, 1023, 4095], dtype=jnp.int32)
    elif pos_mode == "2d_single":
        positions = jnp.asarray([[0, 127, 1023, 4095]], dtype=jnp.int32)
    else:
        positions = jnp.asarray(
            [
                [0, 127, 1023, 4095],
                [0, 63, 511, 2047],
                [0, 31, 255, 1023],
            ],
            dtype=jnp.int32,
        )
        # Two-way token sharding when the runner exposes >= 2 CPU devices.
        mesh = Mesh(np.array(jax.devices()[:2]), ("data",))
        positions = jax.device_put(positions, NamedSharding(mesh, P(None, "data")))

    query, key = _make_qk(positions.shape[-1])
    cos_sin = rotary_emb.get_cos_sin(positions)
    ref_cos, ref_sin = _np_mrope_cos_sin(
        positions, _np_default_inv_freq(_BASE, _HEAD_SIZE), _MROPE_SECTION, interleaved
    )
    _assert_close(cos_sin[0], ref_cos, _COS_SIN_ATOL)
    _assert_close(cos_sin[1], ref_sin, _COS_SIN_ATOL)

    expected_q, expected_k = rotary_emb(positions, query, key)
    _assert_close(expected_q, _np_apply_neox(query, ref_cos, ref_sin), _QK_ATOL)
    _assert_close(expected_k, _np_apply_neox(key, ref_cos, ref_sin), _QK_ATOL)
    actual_q, actual_k = rotary_emb(positions, query, key, cos_sin=cos_sin)

    _assert_array_equal(actual_q, expected_q)
    _assert_array_equal(actual_k, expected_k)
