"""Unit tests for WeightLoader._load_dummy_weights fused-target handling.

Regression for the bug where dummy mode only materialized `target_path[0]` of a
list-valued regular mapping (fused `c_attn` -> q/k/v_proj, `gate_up` -> w1/w3),
leaving the remaining split params as abstract `ShapeDtypeStruct` from
`eval_shape` and crashing the first forward with "is not a valid JAX type".

These run on CPU without a model/checkpoint: WeightLoader is built via
`object.__new__` with just enough state to drive `_load_dummy_weights`, and
`_get_param` is mocked to hand back capture params whose `.value` starts as the
abstract placeholder and should be replaced with a concrete array.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from sgl_jax.srt.utils import weight_utils
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping


class _CaptureParam:
    """Stand-in for nnx.Param: `.value` starts abstract (as after eval_shape) and
    the loader's `model_param.value = ...` assignment lands here for read-back."""

    def __init__(self, shape, dtype=jnp.float32):
        self._v = jax.ShapeDtypeStruct(shape, dtype)

    @property
    def value(self):
        return self._v

    @value.setter
    def value(self, v):
        self._v = v


def _make_loader(captured, monkeypatch):
    """WeightLoader with only the state _load_dummy_weights needs; `_get_param`
    resolves against `captured` and raises (→ skipped) for anything else. The
    trailing `nnx.update(self.model, params)` is stubbed out."""
    loader = object.__new__(WeightLoader)
    loader.mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))
    loader.model = object()

    def _get_param(_params, path):
        if path in captured:
            return captured[path]
        raise KeyError(path)

    loader._get_param = _get_param
    monkeypatch.setattr(weight_utils.nnx, "update", lambda *a, **k: None)
    return loader


def _concrete(param):
    return isinstance(param.value, jax.Array)


def test_fused_mapping_materializes_every_target(monkeypatch):
    """The bug: a fused mapping (one HF key -> [q, k, v]) must fill ALL targets,
    not just target_path[0]. Pre-fix, k_proj/v_proj stayed abstract."""
    shape = (16, 16)
    captured = {
        "model.layers.0.self_attn.q_proj.weight": _CaptureParam(shape),
        "model.layers.0.self_attn.k_proj.weight": _CaptureParam(shape),
        "model.layers.0.self_attn.v_proj.weight": _CaptureParam(shape),
    }
    loader = _make_loader(captured, monkeypatch)
    mappings = {
        "transformer.h.0.attn.c_attn.weight": WeightMapping(
            target_path=list(captured), sharding=(None, None)
        )
    }

    loader._load_dummy_weights(params=None, weight_mappings=mappings)

    for path, param in captured.items():
        assert _concrete(param), f"{path} left abstract ({type(param.value).__name__})"


def test_str_mapping_still_materializes_single_target(monkeypatch):
    """Guard: the common non-fused case (target_path is a str) keeps working."""
    captured = {"model.embed_tokens.embedding": _CaptureParam((32, 16))}
    loader = _make_loader(captured, monkeypatch)
    mappings = {
        "transformer.wte.weight": WeightMapping(
            target_path="model.embed_tokens.embedding", sharding=(None, None)
        )
    }

    loader._load_dummy_weights(params=None, weight_mappings=mappings)

    assert _concrete(captured["model.embed_tokens.embedding"])


def test_missing_target_is_skipped(monkeypatch):
    """Guard: a target absent from the model is skipped, not fatal, and does not
    stop the remaining targets of the same mapping from being filled."""
    captured = {
        "model.layers.0.self_attn.q_proj.weight": _CaptureParam((16, 16)),
        "model.layers.0.self_attn.v_proj.weight": _CaptureParam((16, 16)),
    }
    loader = _make_loader(captured, monkeypatch)
    mappings = {
        "transformer.h.0.attn.c_attn.weight": WeightMapping(
            target_path=[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",  # not in captured -> skipped
                "model.layers.0.self_attn.v_proj.weight",
            ],
            sharding=(None, None),
        )
    }

    loader._load_dummy_weights(params=None, weight_mappings=mappings)

    assert _concrete(captured["model.layers.0.self_attn.q_proj.weight"])
    assert _concrete(captured["model.layers.0.self_attn.v_proj.weight"])


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
