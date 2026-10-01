"""Shared mappings tolerate variants without hiding incomplete model weights."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from flax import nnx
from safetensors.numpy import save_file

from sgl_jax.srt.model_loader.loader import validate_model_parameters
from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec
from sgl_jax.srt.model_loader.weights.source import LocalSource


@pytest.fixture
def mesh():
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]),
        ("tensor",),
        axis_types=(jax.sharding.AxisType.Explicit,),
    )
    with jax.set_mesh(mesh):
        yield mesh


def _model():
    model = nnx.Module()
    model.weight = nnx.Param(jax.ShapeDtypeStruct((4,), np.float32))
    return model


def _load(tmp_path, mesh, model, weights, mappings, **kwargs):
    save_file(weights, tmp_path / "model.safetensors")
    config = SimpleNamespace(model_path=str(tmp_path))
    with LocalSource(config) as source:
        return WeightLoader(model, config, mesh, source=source).load(mappings, **kwargs)


def test_missing_checkpoint_aliases_do_not_block_available_weight(tmp_path, mesh, caplog):
    model = _model()
    expected = np.arange(4, dtype=np.float32)
    report = _load(
        tmp_path,
        mesh,
        model,
        {"weight": expected, "unmapped": expected + 1},
        {
            "legacy.weight": "weight",
            "absent.group": WeightSpec("weight", sources=("old.a", "old.b")),
            "weight": "weight",
        },
    )

    validate_model_parameters(model)
    np.testing.assert_array_equal(model.weight[...], expected)
    assert report["loaded"] == ("weight",)
    assert report["skipped"] == ("legacy.weight", "old.a", "old.b")
    assert report["unexpected"] == ("unmapped",)
    assert "Skipped 2 weight mappings with missing checkpoint inputs" in caplog.text


def test_unused_model_targets_are_skipped_before_group_validation(tmp_path, mesh, caplog):
    model = _model()
    model.layers = nnx.List([])
    model.slots = nnx.Dict({})
    expected = np.arange(4, dtype=np.float32)
    ignored = {
        "disabled.weight": "disabled.weight",
        "extra.layer": "layers.1.weight",
        "extra.slot": "slots.missing.weight",
        "not.a.parameter": "slots",
    }
    report = _load(
        tmp_path,
        mesh,
        model,
        {"weight": expected, "old.a": expected, **{key: expected for key in ignored}},
        {
            **ignored,
            "old.group": WeightSpec("disabled.weight", sources=("old.a", "old.b")),
            "weight": "weight",
        },
        validate_checkpoint_coverage=True,
    )

    validate_model_parameters(model)
    np.testing.assert_array_equal(model.weight[...], expected)
    assert report["loaded"] == ("weight",)
    assert set(report["skipped"]) == {*ignored, "old.a", "old.b"}
    assert report["unexpected"] == ()
    assert "Skipped 5 weight mappings with missing model targets" in caplog.text


def test_missing_real_parameter_is_rejected_by_final_validation(tmp_path, mesh):
    model = _model()
    report = _load(
        tmp_path,
        mesh,
        model,
        {"unrelated": np.arange(4, dtype=np.float32)},
        {"required.weight": "weight"},
    )

    assert report["loaded"] == ()
    with pytest.raises(ValueError, match="Unloaded model parameters.*weight"):
        validate_model_parameters(model)


def test_partial_required_group_still_fails(tmp_path, mesh):
    with pytest.raises(ValueError, match="Missing checkpoint inputs.*missing"):
        _load(
            tmp_path,
            mesh,
            _model(),
            {"present": np.arange(4, dtype=np.float32)},
            {"group": WeightSpec("weight", sources=("present", "missing"))},
        )


def test_partial_optional_group_can_use_another_mapping(tmp_path, mesh, caplog):
    model = _model()
    expected = np.arange(4, dtype=np.float32)
    report = _load(
        tmp_path,
        mesh,
        model,
        {"weight": expected, "old.a": expected + 1},
        {
            "optional.group": WeightSpec("weight", sources=("old.a", "old.b"), optional=True),
            "weight": "weight",
        },
        validate_checkpoint_coverage=True,
    )

    validate_model_parameters(model)
    np.testing.assert_array_equal(model.weight[...], expected)
    assert report["skipped"] == ("old.a", "old.b")
    assert "Skipped" not in caplog.text


def test_shared_parameter_alias_is_a_valid_target(tmp_path, mesh):
    model = _model()
    model.alias = model.weight
    expected = np.arange(4, dtype=np.float32)
    _load(tmp_path, mesh, model, {"weight": expected}, {"weight": "alias"})

    validate_model_parameters(model)
    assert model.alias is model.weight
    np.testing.assert_array_equal(model.weight[...], expected)


@pytest.mark.parametrize("failure", ["shape", "conflicting_writers", "strict_coverage"])
def test_invalid_loads_still_fail(tmp_path, mesh, failure):
    model = _model()
    model.alias = model.weight
    weights = {"weight": np.arange(4, dtype=np.float32)}
    mappings = {"weight": "weight"}
    kwargs = {}
    if failure == "shape":
        weights["weight"] = np.arange(8, dtype=np.float32)
        message = "Target shape mismatch"
    elif failure == "conflicting_writers":
        weights["conflict"] = weights["weight"] + 1
        mappings["conflict"] = "alias"
        message = "Duplicate writer for shared parameter"
    else:
        weights["unmapped"] = weights["weight"] + 1
        kwargs["validate_checkpoint_coverage"] = True
        message = "Unmapped checkpoint tensors"

    with pytest.raises(ValueError, match=message):
        _load(tmp_path, mesh, model, weights, mappings, **kwargs)
