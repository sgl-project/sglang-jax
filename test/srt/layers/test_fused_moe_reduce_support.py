"""``fused_moe_reduce_supported``: when a GLM-5 MoE layer may defer the shared-expert and EPMoE
reduces and finish them with one all-reduce over the model ``tensor`` axis (CPU, stubs)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.layers.linear import LinearBase, QuantizedLinear
from sgl_jax.srt.models.glm5_moe import fused_moe_reduce_supported


def _mesh():
    return Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


def _qlinear(mesh, kernel_axes):
    w_q = jnp.zeros((256, 512), jnp.float8_e4m3fn)
    w_scale = jnp.ones((4, 1, 256), jnp.float32)
    return QuantizedLinear(
        weight_q=w_q,
        weight_scale=w_scale,
        bias=None,
        activation_dtype=None,
        mesh=mesh,
        kernel_axes=kernel_axes,
        weight_block_size=(128, 128),
        scope_name="down_proj",
    )


def _shape(**sizes):
    return SimpleNamespace(shape=sizes)


def test_row_parallel_quantized_shared_on_tensor_only_mesh():
    mesh = _mesh()
    with mesh:
        shared = SimpleNamespace(down_proj=_qlinear(mesh, ("tensor", None)))
    assert fused_moe_reduce_supported(_shape(data=1, tensor=8), shared, use_fused=False)
    assert fused_moe_reduce_supported(_shape(tensor=4), shared, use_fused=False)


def test_data_axis_larger_than_one_is_rejected():
    # data=2, tensor=2, EP=4: the deferred expert-axis psum spans all four devices, the
    # tensor all-reduce only two of them
    mesh = _mesh()
    with mesh:
        shared = SimpleNamespace(down_proj=_qlinear(mesh, ("tensor", None)))
    assert not fused_moe_reduce_supported(_shape(data=2, tensor=2), shared, use_fused=False)


def test_full_output_shared_experts_are_rejected():
    mesh = _mesh()
    with mesh, jax.set_mesh(mesh):
        replicated_reduce_axis = SimpleNamespace(down_proj=_qlinear(mesh, (None, None)))
        bf16 = SimpleNamespace(
            down_proj=LinearBase(
                input_size=512,
                output_size=256,
                kernel_axes=("tensor", None),
                use_bias=False,
                params_dtype=jnp.bfloat16,
                mesh=mesh,
                scope_name="down_proj",
            )
        )
    tensor_only = _shape(data=1, tensor=8)
    assert not fused_moe_reduce_supported(tensor_only, replicated_reduce_axis, use_fused=False)
    assert not fused_moe_reduce_supported(tensor_only, bf16, use_fused=False)


def test_in_kernel_or_absent_shared_experts_are_rejected():
    mesh = _mesh()
    with mesh:
        shared = SimpleNamespace(down_proj=_qlinear(mesh, ("tensor", None)))
    assert not fused_moe_reduce_supported(_shape(data=1, tensor=8), shared, use_fused=True)
    assert not fused_moe_reduce_supported(_shape(data=1, tensor=8), None, use_fused=False)
