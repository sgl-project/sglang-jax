from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from safetensors.numpy import save_file

from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.model_loader.weights import JaxShardReader, LocalSource, WeightSpec


def test_deferred_moe_transpose_preserves_target_sharding(tmp_path):
    mesh = Mesh(
        np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1),
        axis_names=("expert", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    target_sharding = NamedSharding(mesh, P("expert", None, "tensor"))
    hf_weight = np.arange(6, dtype=np.float32).reshape(2, 3)
    key = "model.layers.0.mlp.experts.0.gate_proj.weight"

    save_file({key: hf_weight}, str(tmp_path / "model.safetensors"))
    with LocalSource(SimpleNamespace(model_path=str(tmp_path))) as source:
        result = JaxShardReader(mesh).read(
            source, key, WeightSpec("w", sources=(key,), transpose=True), target_sharding
        )

    assert result.shape == (1, 3, 2)
    np.testing.assert_array_equal(np.asarray(result)[0], hf_weight.T)
    assert result.sharding == target_sharding


def test_fp8_epmoe_scales_match_shard_map_contract():
    mesh = Mesh(
        np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    captured_inputs = {}

    def fake_shard_map(_fn, *, mesh, in_specs, out_specs, check_vma):
        del mesh, in_specs, out_specs, check_vma

        def invoke(*args):
            captured_inputs["wi_0_scale"] = args[6]
            captured_inputs["wi_1_scale"] = args[7]
            return args[0]

        return invoke

    with jax.set_mesh(mesh):
        layer = EPMoE(
            hidden_size=4,
            num_experts=1,
            num_experts_per_tok=1,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=4,
        )
        scale_sharding = NamedSharding(
            layer.moe_mesh,
            P("expert", None, None, None),
        )
        scale = jax.device_put(
            jnp.ones((1, 1, 1, 4), dtype=jnp.float32),
            scale_sharding,
        )
        del layer.wi_0_scale
        layer.wi_0_scale = nnx.Param(scale)
        del layer.wi_1_scale
        layer.wi_1_scale = nnx.Param(scale)

        with mock.patch("sgl_jax.srt.layers.moe.shard_map", side_effect=fake_shard_map):
            layer(
                jnp.ones((1, 4), dtype=jnp.bfloat16),
                jnp.ones((1, 1), dtype=jnp.float32),
                jnp.zeros((1, 1), dtype=jnp.int32),
            )

    expected = P("expert", None, None, "tensor")
    assert captured_inputs["wi_0_scale"].sharding.spec == expected
    assert captured_inputs["wi_1_scale"].sharding.spec == expected
