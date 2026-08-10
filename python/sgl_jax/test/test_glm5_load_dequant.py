from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.linear import LinearBase, QuantizedLinear
from sgl_jax.srt.models.glm5_moe import _dequantize_glm5_latency_sensitive_linears
from sgl_jax.srt.utils.weight_utils import WeightLoader


def _cpu_mesh():
    return jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


def test_dequant_fp8_linear_preserves_projection_scope():
    mesh = _cpu_mesh()
    with jax.set_mesh(mesh):
        quantized = QuantizedLinear(
            weight_q=jnp.ones((3, 4), dtype=jnp.float8_e4m3fn),
            weight_scale=jnp.full((1, 1, 3), 2.0, dtype=jnp.float32),
            bias=None,
            activation_dtype=jnp.float8_e4m3fn,
            mesh=mesh,
            kernel_axes=("tensor", None),
            params_dtype=jnp.bfloat16,
            weight_block_size=(128, 128),
            scope_name="quantized_o_proj",
        )

        linear = WeightLoader.dequant_fp8_linear(object.__new__(WeightLoader), quantized)

    assert isinstance(linear, LinearBase)
    assert linear.name == "o_proj"
    assert linear.weight.value.shape == (4, 3)
    assert linear.weight.value.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(linear.weight.value), np.full((4, 3), 2.0))


def test_glm5_load_time_dequant_targets_only_o_proj():
    calls = []
    layers = [object()]
    loader = SimpleNamespace(
        is_static_quant=True,
        dequant_fp8_layers=lambda actual_layers, specs: calls.append((actual_layers, specs)),
    )

    _dequantize_glm5_latency_sensitive_linears(loader, layers)

    assert calls == [(layers, [("self_attn.o_proj", None)])]


def test_glm5_load_time_dequant_skips_non_static_checkpoint():
    loader = SimpleNamespace(
        is_static_quant=False,
        dequant_fp8_layers=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError),
    )

    _dequantize_glm5_latency_sensitive_linears(loader, [])
