"""Unit tests for WeightLoader._split_qkv_weight scale handling.

Covers the dedicated 1-D per-channel scale branch (compressed-tensors W8A16
with HF shape [Q_out+K_out+V_out,]) and the 2-D non-block scale branch
(compressed-tensors per-channel FP8 on Ling-2.6-1T with HF shape
[Q_out+K_out+V_out, inner=1] and weight_block_size=None). Both must split
on the same Q/K/V offsets as the bias branch and apply head_dim_padding
the same way when set.
"""

import jax.numpy as jnp
import pytest

from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping


class _CaptureParam:
    """Stand-in for nnx.Param: exposes settable `value` so the loader's
    `model_param.value = ...` assignment lands here and we can read it back."""

    def __init__(self, dtype=jnp.float32):
        self._v = jnp.zeros((), dtype=dtype)

    @property
    def value(self):
        return self._v

    @value.setter
    def value(self, v):
        self._v = v


def _make_loader(
    *,
    num_heads: int = 4,
    num_kv_heads: int = 4,
    head_dim_original: int = 8,
    v_head_dim: int | None = None,
):
    """Construct a WeightLoader bypassing __init__, with just enough state to
    drive _split_qkv_weight; mock the assignment-side helpers to identity."""
    loader = object.__new__(WeightLoader)
    loader.num_heads = num_heads
    loader.num_kv_heads = num_kv_heads
    loader.head_dim_original = head_dim_original
    loader.head_dim_pad = (head_dim_original + 127) // 128 * 128 - head_dim_original
    loader.head_dim = head_dim_original + loader.head_dim_pad
    loader.v_head_dim = v_head_dim if v_head_dim is not None else head_dim_original
    loader.hidden_size = num_heads * head_dim_original
    loader.sharding_size = 1
    loader.dummy_mode = False

    captured: dict[str, _CaptureParam] = {}

    def _get_param(_params, path):
        p = _CaptureParam()
        captured[path] = p
        return p

    loader._shard_weight = lambda w, sharding: w
    loader._get_param = _get_param
    loader._maybe_expand_linear_block_scale = lambda w, _model_param, _path: w
    loader._apply_kv_head_padding = lambda w, _key: w

    return loader, captured


def _qkv_mapping(suffix: str, *, head_dim_padding: bool = False) -> WeightMapping:
    return WeightMapping(
        target_path=[
            f"model.layers.0.self_attn.q_proj.{suffix}",
            f"model.layers.0.self_attn.k_proj.{suffix}",
            f"model.layers.0.self_attn.v_proj.{suffix}",
        ],
        sharding=("tensor", None),
        transpose=False,
        head_dim_padding=head_dim_padding,
    )


def test_1d_per_channel_scale_splits_at_correct_offsets():
    """Most common case: GLA/MHA with num_kv_heads == num_heads and no padding.
    The pre-fix code IndexError'd here; the fix should split [0..32, 32..64, 64..96]."""
    loader, captured = _make_loader(num_heads=4, num_kv_heads=4, head_dim_original=8)
    mapping = _qkv_mapping("weight_scale")

    weight = jnp.arange(96, dtype=jnp.float32)  # 3 * 4 * 8

    loader._split_qkv_weight(
        params=None,
        hf_key="model.layers.0.attention.query_key_value.weight_scale",
        weight=weight,
        mapping=mapping,
    )

    q = captured["model.layers.0.self_attn.q_proj.weight_scale"].value
    k = captured["model.layers.0.self_attn.k_proj.weight_scale"].value
    v = captured["model.layers.0.self_attn.v_proj.weight_scale"].value

    assert q.shape == (32,)
    assert k.shape == (32,)
    assert v.shape == (32,)
    assert jnp.array_equal(q, jnp.arange(0, 32, dtype=jnp.float32))
    assert jnp.array_equal(k, jnp.arange(32, 64, dtype=jnp.float32))
    assert jnp.array_equal(v, jnp.arange(64, 96, dtype=jnp.float32))


def test_1d_per_channel_scale_gqa_offsets():
    """GQA: num_kv_heads < num_heads. Offsets become asymmetric."""
    loader, captured = _make_loader(num_heads=8, num_kv_heads=2, head_dim_original=4)
    mapping = _qkv_mapping("weight_scale")

    # Q: 8*4=32, K: 2*4=8, V: 2*4=8. Total: 48.
    weight = jnp.arange(48, dtype=jnp.float32)

    loader._split_qkv_weight(
        params=None,
        hf_key="model.layers.0.attention.query_key_value.weight_scale",
        weight=weight,
        mapping=mapping,
    )

    q = captured["model.layers.0.self_attn.q_proj.weight_scale"].value
    k = captured["model.layers.0.self_attn.k_proj.weight_scale"].value
    v = captured["model.layers.0.self_attn.v_proj.weight_scale"].value

    assert jnp.array_equal(q, jnp.arange(0, 32, dtype=jnp.float32))
    assert jnp.array_equal(k, jnp.arange(32, 40, dtype=jnp.float32))
    assert jnp.array_equal(v, jnp.arange(40, 48, dtype=jnp.float32))


def test_1d_per_channel_scale_head_dim_padding():
    """Qwen-style head_dim=96 case: each per-head slice gets padded to 128
    along the (only) axis. Validates that the 1-D scale branch mirrors the
    bias branch's reshape+pad+flatten when head_dim_padding=True."""
    loader, captured = _make_loader(num_heads=2, num_kv_heads=2, head_dim_original=96)
    mapping = _qkv_mapping("weight_scale", head_dim_padding=True)

    # Unpadded: 3 * 2 * 96 = 576
    weight = jnp.arange(576, dtype=jnp.float32)

    loader._split_qkv_weight(
        params=None,
        hf_key="model.layers.0.attention.query_key_value.weight_scale",
        weight=weight,
        mapping=mapping,
    )

    q = captured["model.layers.0.self_attn.q_proj.weight_scale"].value
    # Padded per-head 96 -> 128. Per-projection length: 2 * 128 = 256.
    assert q.shape == (256,)

    # Within each padded head, the first 96 elements are the original slice
    # and the last 32 are zero.
    q_reshaped = q.reshape(2, 128)
    assert jnp.array_equal(q_reshaped[0, :96], jnp.arange(0, 96, dtype=jnp.float32))
    assert jnp.all(q_reshaped[0, 96:] == 0)
    assert jnp.array_equal(q_reshaped[1, :96], jnp.arange(96, 192, dtype=jnp.float32))
    assert jnp.all(q_reshaped[1, 96:] == 0)


def test_pre_fix_else_branch_still_handles_2d_weight():
    """Regression guard: the 2-D weight else branch must keep working unchanged.
    A 2-D weight (transpose=False) with shape [Q+K+V, hidden] should split
    along axis 0."""
    loader, captured = _make_loader(num_heads=4, num_kv_heads=4, head_dim_original=8)
    mapping = WeightMapping(
        target_path=[
            "model.layers.0.self_attn.q_proj.weight_q",
            "model.layers.0.self_attn.k_proj.weight_q",
            "model.layers.0.self_attn.v_proj.weight_q",
        ],
        sharding=("tensor", None),
        transpose=False,
    )

    # 3 * 4 * 8 = 96 rows, hidden = 32 cols
    weight = jnp.arange(96 * 32, dtype=jnp.float32).reshape(96, 32)

    loader._split_qkv_weight(
        params=None,
        hf_key="model.layers.0.attention.query_key_value.weight_q",
        weight=weight,
        mapping=mapping,
    )

    q = captured["model.layers.0.self_attn.q_proj.weight_q"].value
    assert q.shape == (32, 32)
    assert jnp.array_equal(q, weight[:32])


# ---------------------------------------------------------------------------
# 2-D non-block scale branch (compressed-tensors per-channel FP8, Ling-2.6-1T)
# ---------------------------------------------------------------------------


class _DummyQuantCfg:
    """quantization_config with weight_block_size=None — selects the new
    element-wise fallback inside the 2-D scale branch."""

    weight_block_size = None


class _DummyModelConfig:
    quantization_config = _DummyQuantCfg()


def test_2d_non_block_scale_splits_at_correct_offsets():
    """Ling-2.6-1T compressed-tensors per-channel FP8 stores QKV weight_scale
    as 2-D shape [Q_out+K_out+V_out, 1] with weight_block_size=None. Should
    fall back to element-wise axis-0 split on the same Q/K/V offsets as the
    bias / 1-D branches, preserving the inner axis."""
    loader, captured = _make_loader(num_heads=4, num_kv_heads=4, head_dim_original=8)
    loader.model_config = _DummyModelConfig()
    mapping = _qkv_mapping("weight_scale")

    # 3 * 4 * 8 = 96 rows, inner = 1
    weight = jnp.arange(96, dtype=jnp.float32).reshape(96, 1)

    loader._split_qkv_weight(
        params=None,
        hf_key="model.layers.0.attention.query_key_value.weight_scale",
        weight=weight,
        mapping=mapping,
    )

    q = captured["model.layers.0.self_attn.q_proj.weight_scale"].value
    k = captured["model.layers.0.self_attn.k_proj.weight_scale"].value
    v = captured["model.layers.0.self_attn.v_proj.weight_scale"].value

    assert q.shape == (32, 1)
    assert k.shape == (32, 1)
    assert v.shape == (32, 1)
    assert jnp.array_equal(q, jnp.arange(0, 32, dtype=jnp.float32).reshape(32, 1))
    assert jnp.array_equal(k, jnp.arange(32, 64, dtype=jnp.float32).reshape(32, 1))
    assert jnp.array_equal(v, jnp.arange(64, 96, dtype=jnp.float32).reshape(32, 1))


def test_2d_non_block_scale_gqa_offsets():
    """GQA variant: num_kv_heads < num_heads. Same asymmetric offsets as the
    1-D GQA case, with the inner axis preserved."""
    loader, captured = _make_loader(num_heads=8, num_kv_heads=2, head_dim_original=4)
    loader.model_config = _DummyModelConfig()
    mapping = _qkv_mapping("weight_scale")

    # Q: 8*4=32, K: 2*4=8, V: 2*4=8. Total: 48. Inner: 1.
    weight = jnp.arange(48, dtype=jnp.float32).reshape(48, 1)

    loader._split_qkv_weight(
        params=None,
        hf_key="model.layers.0.attention.query_key_value.weight_scale",
        weight=weight,
        mapping=mapping,
    )

    q = captured["model.layers.0.self_attn.q_proj.weight_scale"].value
    k = captured["model.layers.0.self_attn.k_proj.weight_scale"].value
    v = captured["model.layers.0.self_attn.v_proj.weight_scale"].value

    assert q.shape == (32, 1)
    assert k.shape == (8, 1)
    assert v.shape == (8, 1)
    assert jnp.array_equal(q, jnp.arange(0, 32, dtype=jnp.float32).reshape(32, 1))
    assert jnp.array_equal(k, jnp.arange(32, 40, dtype=jnp.float32).reshape(8, 1))
    assert jnp.array_equal(v, jnp.arange(40, 48, dtype=jnp.float32).reshape(8, 1))


def test_2d_non_block_scale_head_dim_padding():
    """Qwen-style head_dim=96 case: each per-head slice gets padded to 128
    along axis 0 with the inner axis preserved. Validates that the 2-D
    branch's reshape + pad + flatten mirrors the 1-D branch with an extra
    inner dim."""
    loader, captured = _make_loader(num_heads=2, num_kv_heads=2, head_dim_original=96)
    loader.model_config = _DummyModelConfig()
    mapping = _qkv_mapping("weight_scale", head_dim_padding=True)

    # Unpadded: 3 * 2 * 96 = 576 rows, inner = 1
    weight = jnp.arange(576, dtype=jnp.float32).reshape(576, 1)

    loader._split_qkv_weight(
        params=None,
        hf_key="model.layers.0.attention.query_key_value.weight_scale",
        weight=weight,
        mapping=mapping,
    )

    q = captured["model.layers.0.self_attn.q_proj.weight_scale"].value
    # Padded per-head 96 -> 128, 2 heads, inner 1 → (256, 1).
    assert q.shape == (256, 1)

    # Within each padded head, the first 96 elements (along the per-head axis)
    # are the original slice and the last 32 are zero. Inner axis is preserved.
    q_reshaped = q.reshape(2, 128, 1)
    assert jnp.array_equal(q_reshaped[0, :96, 0], jnp.arange(0, 96, dtype=jnp.float32))
    assert jnp.all(q_reshaped[0, 96:, 0] == 0)
    assert jnp.array_equal(q_reshaped[1, :96, 0], jnp.arange(96, 192, dtype=jnp.float32))
    assert jnp.all(q_reshaped[1, 96:, 0] == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
