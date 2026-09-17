import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.embeddings import (
    MRotaryEmbedding,
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_interleaved_rope,
    apply_rotary_emb,
)

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
