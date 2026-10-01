"""Preparation hooks must not be synthesized by graph nodes such as nnx.Rngs."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from flax import nnx
from safetensors.numpy import save_file

from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec


@pytest.fixture
def mesh():
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]),
        ("tensor",),
        axis_types=(jax.sharding.AxisType.Explicit,),
    )
    with jax.set_mesh(mesh):
        yield mesh


def test_rng_nodes_do_not_hide_inherited_preparation_hooks(tmp_path, mesh):
    calls = []

    class Layer(nnx.Module):
        def __init__(self):
            self.weight = nnx.Param(jax.ShapeDtypeStruct((4,), np.float32))
            self.rngs = nnx.Rngs(7)

        def prepare_weight_loading(self, loader, mappings, prefix):
            calls.append(prefix)
            assert self is loader.model.child
            assert "root" in mappings
            return {**mappings, "child": WeightSpec(f"{prefix}.weight")}

    class InheritedLayer(Layer):
        pass

    class Model(nnx.Module):
        def __init__(self):
            self.weight = nnx.Param(jax.ShapeDtypeStruct((4,), np.float32))
            self.child = InheritedLayer()
            self.rngs = nnx.Rngs(42)

        def prepare_weight_loading(self, loader, mappings):
            calls.append("root")
            assert self is loader.model
            return {**mappings, "root": WeightSpec("weight")}

    class InheritedModel(Model):
        pass

    expected = {"root": np.arange(4, dtype=np.float32), "child": np.full(4, 7, np.float32)}
    save_file(expected, tmp_path / "model.safetensors")
    model = InheritedModel()
    loader = WeightLoader(model, SimpleNamespace(model_path=str(tmp_path)), mesh)
    loader.load({})

    assert calls == ["root", "child"]
    np.testing.assert_array_equal(model.weight[...], expected["root"])
    np.testing.assert_array_equal(model.child.weight[...], expected["child"])
    assert int(model.rngs.default.count[...]) == 0
    assert int(model.child.rngs.default.count[...]) == 0


def test_flux_vae_weight_loading_ignores_rng_streams(tmp_path, mesh):
    from sgl_jax.srt.multimodal.configs.vaes.flux_vae_config import FluxVAEConfig
    from sgl_jax.srt.multimodal.models.vaes.autoencoder import AutoencoderKL
    from sgl_jax.srt.multimodal.models.vaes.flux_vae_weight_mappings import to_mappings

    config = FluxVAEConfig(
        model_path=str(tmp_path),
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,),
        layers_per_block=1,
        latent_channels=2,
        norm_num_groups=4,
        sample_size=8,
    )
    model = AutoencoderKL(config, mesh=mesh)
    key = "encoder.conv_in.weight"
    expected = np.arange(8 * 3 * 3 * 3, dtype=np.float32).reshape(8, 3, 3, 3)
    save_file({key: expected}, tmp_path / "model.safetensors")
    rng_count = np.asarray(model.rngs.default.count[...]).copy()
    WeightLoader(model, config, mesh).load({key: to_mappings(config)[key]})

    np.testing.assert_array_equal(model.encoder.conv_in.kernel[...], expected.transpose(2, 3, 1, 0))
    np.testing.assert_array_equal(model.rngs.default.count[...], rng_count)
