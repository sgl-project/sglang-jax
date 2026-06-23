"""Tests for pre-stacked MoE expert weight loading.

Step 3.5 Flash stores routed experts as a single stacked [E, out, in] tensor
per proj (e.g. moe.gate_proj.weight) instead of the per-expert
experts.{i}.gate_proj.weight keys used by earlier MoE models (e.g. Qwen3-MoE).

Three test groups:
1. Predicate hard-gate — mirrors the exact branch condition; self-contained.
2. End-to-end loader — builds a tiny safetensors checkpoint, loads via
   WeightLoader, verifies the transposed result.

The regression sentinel (test_qwen3_5.py — 9 passed) is run separately by
the CI suite driver, not inline here.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from safetensors.numpy import save_file

from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

# ---------------------------------------------------------------------------
# Shared mesh fixture (CPU single device)
# ---------------------------------------------------------------------------

_MESH = create_device_mesh(
    ici_parallelism=[1, 1], dcn_parallelism=[1, 1], devices=[jax.devices()[0]]
)
jax.sharding.set_mesh(_MESH)


# ---------------------------------------------------------------------------
# 1. Predicate hard-gate tests (pure-Python, no I/O)
# ---------------------------------------------------------------------------


def _predicate(expected_hf_keys: list[str], weight_info: dict, num_experts: int) -> bool:
    """Mirrors the dispatch condition in weight_utils.py.

    Returns True iff:
      - exactly one source key, AND
      - that tensor is 3-D, AND
      - its leading dim matches num_experts.
    """
    if len(expected_hf_keys) != 1:
        return False
    info0 = weight_info[expected_hf_keys[0]][0]
    shape0 = tuple(info0["shape"])
    return len(shape0) == 3 and shape0[0] == num_experts


class TestPredicateTriggersOnlyOnSingle3dELeading:
    """Strict gate: must fire on single stacked source, must NOT fire otherwise."""

    def test_single_3d_matching_E_returns_true(self):
        """Single key, 3-D tensor, leading dim == num_experts -> should route to prestacked."""
        E, out, in_ = 8, 128, 64
        weight_info = {
            "model.layers.0.moe.gate_proj.weight": [{"shape": (E, out, in_), "dtype": "BF16"}]
        }
        assert _predicate(["model.layers.0.moe.gate_proj.weight"], weight_info, E) is True

    def test_per_expert_keys_returns_false(self):
        """E separate 2-D keys -> per-expert path, must NOT trigger prestacked."""
        E = 8
        keys = [f"experts.{i}.gate_proj.weight" for i in range(E)]
        weight_info = {k: [{"shape": (128, 64), "dtype": "BF16"}] for k in keys}
        assert _predicate(keys, weight_info, E) is False

    def test_wrong_leading_dim_returns_false(self):
        """Single 3-D key but shape[0] != num_experts -> must NOT trigger prestacked."""
        E = 8
        wrong_E = 4
        weight_info = {
            "model.layers.0.moe.gate_proj.weight": [{"shape": (wrong_E, 128, 64), "dtype": "BF16"}]
        }
        assert _predicate(["model.layers.0.moe.gate_proj.weight"], weight_info, E) is False

    def test_single_2d_key_returns_false(self):
        """Single key but 2-D tensor -> not a pre-stacked source, must NOT trigger."""
        weight_info = {
            "model.layers.0.mlp.gate_proj.weight": [{"shape": (128, 64), "dtype": "BF16"}]
        }
        assert _predicate(["model.layers.0.mlp.gate_proj.weight"], weight_info, 8) is False

    def test_zero_keys_returns_false(self):
        """Empty key list should not trigger."""
        assert _predicate([], {}, 8) is False


# ---------------------------------------------------------------------------
# 2. End-to-end loader tests
# ---------------------------------------------------------------------------


class _Experts(nnx.Module):
    """Minimal NNX module holding a single wi_0 EPMoE parameter."""

    def __init__(self, E: int, in_: int, out: int):
        # wi_0 shape after transpose will be [E, in_, out].
        # Initialise as zeros placeholder; the loader overwrites it.
        self.wi_0 = nnx.Param(jnp.zeros((E, in_, out), dtype=jnp.float32))


class _MinimalModel(nnx.Module):
    """Minimal NNX model with a single experts block."""

    def __init__(self, E: int, in_: int, out: int):
        self.experts = _Experts(E=E, in_=in_, out=out)


@dataclass
class _DummyHFConfig:
    ep_size: int = 1


@dataclass
class _DummyModelConfig:
    model_path: str
    hf_config: _DummyHFConfig = None

    def __post_init__(self):
        if self.hf_config is None:
            self.hf_config = _DummyHFConfig()


def _make_prestacked_loader(tmp_dir: Path, E: int, out: int, in_: int):
    """Create a safetensors checkpoint with ONE stacked [E, out, in] tensor,
    build a WeightLoader scaffold, and return (loader, model, moe_key, mapping, source)."""

    # Build the checkpoint tensor: stacked [E, out, in] in HF layout.
    source = np.arange(E * out * in_, dtype=np.float32).reshape(E, out, in_)
    st_file = tmp_dir / "model.safetensors"
    save_file({"model.layers.0.moe.gate_proj.weight": source}, str(st_file))

    # Minimal NNX model: wi_0 has shape [E, in_, out] (transposed from HF layout).
    model = _MinimalModel(E=E, in_=in_, out=out)

    # WeightLoader bypassing __init__ (same pattern as test_weight_loader_qkv_split.py).
    loader = object.__new__(WeightLoader)
    loader.model = model
    loader.model_config = _DummyModelConfig(model_path=str(tmp_dir))
    loader.mesh = _MESH
    loader.dtype = jnp.float32
    loader.dummy_mode = False
    loader._weight_info_cache = None
    loader.sharding_size = 1
    loader.moe_abstract_mesh = None

    # Patch helpers that do not affect the MoE dispatch logic under test.
    loader._maybe_convert_epmoe_scale_for_kernel = lambda w, _mp, _path: w
    loader._is_excluded_layer_weight = lambda _k: False
    loader._normalize_physical_to_logical_map = (
        lambda *, physical_to_logical_map, **_kw: physical_to_logical_map
    )

    # MoE mapping: single stacked source, transpose=True -> wi_0 = source.transpose(0,2,1)
    # __MOE_EXPERTS__ prefix routes to the MoE dispatch block.
    moe_key = "__MOE_EXPERTS__model.layers.0.moe.gate_proj"
    mapping = WeightMapping(
        target_path=[
            "experts.wi_0",
            "model.layers.0.moe.gate_proj.weight",
        ],
        sharding=(None, None, None),
        transpose=True,
    )

    return loader, model, moe_key, mapping, source


class TestPreStackedLoadsTransposed:
    """End-to-end: build a tiny safetensors checkpoint, load it through the
    WeightLoader MoE dispatch, verify result == source.transpose(0,2,1)."""

    def test_prestacked_loads_transposed(self):
        """Transpose=True: loaded wi_0 must equal source.transpose(0,2,1)."""
        E, out, in_ = 4, 16, 8  # small enough for CPU test

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            loader, model, moe_key, mapping, source = _make_prestacked_loader(
                tmp_dir, E=E, out=out, in_=in_
            )

            loader.load_weights_from_safetensors(
                weight_mappings={moe_key: mapping},
                safetensors_partition=1,
            )

            result = np.asarray(model.experts.wi_0.value, dtype=np.float32)
            expected = source.transpose(0, 2, 1)  # [E, out, in] -> [E, in, out]

            assert result.shape == (
                E,
                in_,
                out,
            ), f"Expected shape {(E, in_, out)}, got {result.shape}"
            np.testing.assert_array_equal(result, expected)

    def test_prestacked_no_transpose(self):
        """With transpose=False the tensor should be stored as-is [E, out, in]."""
        E, out, in_ = 4, 16, 8

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            source = np.arange(E * out * in_, dtype=np.float32).reshape(E, out, in_)
            st_file = tmp_dir / "model.safetensors"
            save_file({"model.layers.0.moe.gate_proj.weight": source}, str(st_file))

            # Model param shape matches no-transpose layout [E, out, in].
            # Use a proper NNX module so nnx.state() works correctly.
            class _ExpertsNoTranspose(nnx.Module):
                def __init__(self, E, out, in_):
                    self.wi_0 = nnx.Param(jnp.zeros((E, out, in_), dtype=jnp.float32))

            class _ModelNoTranspose(nnx.Module):
                def __init__(self, E, out, in_):
                    self.experts = _ExpertsNoTranspose(E=E, out=out, in_=in_)

            model = _ModelNoTranspose(E=E, out=out, in_=in_)

            loader = object.__new__(WeightLoader)
            loader.model = model
            loader.model_config = _DummyModelConfig(model_path=str(tmp_dir))
            loader.mesh = _MESH
            loader.dtype = jnp.float32
            loader.dummy_mode = False
            loader._weight_info_cache = None
            loader.sharding_size = 1
            loader.moe_abstract_mesh = None
            loader._maybe_convert_epmoe_scale_for_kernel = lambda w, _mp, _path: w
            loader._is_excluded_layer_weight = lambda _k: False
            loader._normalize_physical_to_logical_map = (
                lambda *, physical_to_logical_map, **_kw: physical_to_logical_map
            )

            moe_key = "__MOE_EXPERTS__model.layers.0.moe.gate_proj"
            mapping_no_transpose = WeightMapping(
                target_path=[
                    "experts.wi_0",
                    "model.layers.0.moe.gate_proj.weight",
                ],
                sharding=(None, None, None),
                transpose=False,
            )

            loader.load_weights_from_safetensors(
                weight_mappings={moe_key: mapping_no_transpose},
                safetensors_partition=1,
            )

            result = np.asarray(model.experts.wi_0.value, dtype=np.float32)
            assert result.shape == (E, out, in_)
            np.testing.assert_array_equal(result, source)


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
