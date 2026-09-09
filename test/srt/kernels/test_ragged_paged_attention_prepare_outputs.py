"""CPU regressions for the RPA output conversion's integer transport contract.

Run with JAX_PLATFORMS=cpu and PYTHONPATH=python. Tracing checks the actual
Pallas boundary without TPU lowering. The exhaustive interpreter check skips
the known JAX 0.11.1 strided-index limitation; it does not replace TPU validation.
"""

from functools import partial
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.ragged_paged_attention import ragged_paged_attention_v3 as rpa


def test_packed_output_call_uses_integer_transport():
    prepare = partial(rpa.prepare_outputs, actual_num_q_heads_per_kv_head=4, actual_head_dim=128)
    traced = jax.make_jaxpr(prepare)(jax.ShapeDtypeStruct((2, 4096, 2, 2, 128), jnp.bfloat16)).jaxpr
    assert traced.outvars[0].aval.shape == (4096, 8, 128)
    assert traced.outvars[0].aval.dtype == jnp.bfloat16

    (call,) = [eqn for eqn in traced.eqns if eqn.primitive.name == "pallas_call"]
    # A floating-point boundary can normalize subnormals and NaN payloads even
    # when the kernel itself only copies packed integers.
    for var in (*call.invars, *call.outvars):
        assert var.aval.dtype == jnp.uint16
    assert all(
        eqn.primitive.name == "bitcast_convert_type" for eqn in traced.eqns if eqn is not call
    )


def test_packed_output_all_bf16_bits_in_cpu_interpreter():
    shape = (2, 4096, 2, 2, 128)
    # Use integer host/device transport so BF16 transfer normalization cannot
    # hide a regression. Shuffle all patterns across both lanes of packed pairs.
    patterns = np.random.default_rng(0).permutation(np.arange(65536, dtype=np.uint16))
    bits = np.resize(patterns, shape)
    expected = bits.transpose(1, 0, 2, 3, 4).reshape(4096, 8, 128)

    @jax.jit
    def transport(x):
        out = rpa.prepare_outputs(jax.lax.bitcast_convert_type(x, jnp.bfloat16), 4, 128)
        return jax.lax.bitcast_convert_type(out, jnp.uint16)

    with (
        jax.default_device(jax.devices("cpu")[0]),
        patch.object(rpa.pl, "pallas_call", partial(rpa.pl.pallas_call, interpret=True)),
    ):
        try:
            actual = np.asarray(transport(jnp.asarray(bits)))
        except ValueError as exc:
            if str(exc) != "Invalid type of idxer: TypedInt":
                raise
            pytest.skip(f"JAX CPU Pallas interpreter cannot execute the strided refs: {exc}")
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "shape,dtype,heads,width",
    [
        ((2, 128, 2, 2, 128), jnp.bfloat16, 4, 128),
        ((2, 4096, 2, 2, 128), jnp.bfloat16, 3, 128),
        ((2, 4096, 2, 2, 128), jnp.bfloat16, 4, 127),
        ((2, 4096, 2, 2, 128), jnp.float32, 4, 128),
    ],
    ids=["tokens", "heads", "width", "dtype"],
)
def test_output_fallbacks(shape, dtype, heads, width):
    # Small integers are exactly representable in both tested floating dtypes.
    source = (np.arange(np.prod(shape), dtype=np.int32) % 251 - 125).reshape(shape)
    expected = source.transpose(1, 0, 2, 3, 4).reshape(shape[1], shape[0], 4, shape[-1])
    expected = expected[:, :, :heads, :width].reshape(shape[1], shape[0] * heads, width)
    prepare = partial(
        rpa.prepare_outputs, actual_num_q_heads_per_kv_head=heads, actual_head_dim=width
    )
    with jax.default_device(jax.devices("cpu")[0]):
        out = jnp.asarray(source, dtype=dtype)
        traced = jax.make_jaxpr(prepare)(out)
        assert all(eqn.primitive.name != "pallas_call" for eqn in traced.jaxpr.eqns)
        actual = jax.jit(prepare)(out)
        assert actual.dtype == dtype
        np.testing.assert_array_equal(np.asarray(actual), expected)
