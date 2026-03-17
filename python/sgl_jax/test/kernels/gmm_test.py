# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm import gmm
from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import TileSizes, gmm_v2

jax.config.parse_flags_with_absl()

GMM_VERSION_V1 = "v1"
GMM_VERSION_V2 = "v2"


def gmm_wrapper(
    lhs,
    rhs,
    group_sizes,
    version,
    rhs_scale=None,
    rhs_bias=None,
    group_offset=None,
    maybe_quantize_lhs=False,
    preferred_element_type=None,
):
    if version == GMM_VERSION_V1:
        out_dtype = preferred_element_type if preferred_element_type is not None else lhs.dtype
        return gmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=out_dtype,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            group_offset=group_offset,
        )
    else:
        return gmm_v2(
            lhs,
            rhs,
            group_sizes,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            group_offset=group_offset,
            maybe_quantize_lhs=maybe_quantize_lhs,
        )


def get_group_sizes(batch_size: int, num_groups: int) -> jax.Array:
    distribution = jax.random.uniform(jax.random.key(0), (num_groups - 1,), dtype=jnp.float32)
    distribution = distribution / jnp.sum(distribution)
    group_sizes = jnp.floor(distribution * batch_size).astype(jnp.int32)
    return jnp.append(group_sizes, batch_size - jnp.sum(group_sizes))


def quantize_tensor(x: jax.Array, dtype: jnp.dtype, axis: int = -1, block_size: int = 256):
    if jnp.issubdtype(dtype, jnp.integer):
        dtype_info = jnp.iinfo(dtype)
        max_val = int(dtype_info.max)
        min_val = int(dtype_info.min)
    else:
        dtype_info = jnp.finfo(dtype)
        max_val = float(dtype_info.max)
        min_val = float(dtype_info.min)

    orig_shape = x.shape
    blocked_shape = orig_shape[:axis] + (-1, block_size) + orig_shape[axis + 1 :]
    x_blocked = x.reshape(blocked_shape)

    x_blocked_abs_max = jnp.max(jnp.abs(x_blocked), axis=axis + 1, keepdims=True)
    scale = x_blocked_abs_max / max_val
    x_blocked_q = jnp.clip(x_blocked / scale, min_val, max_val).astype(dtype)

    x_q = x_blocked_q.reshape(orig_shape)
    scale = scale.squeeze(axis=axis + 1).astype(jnp.float32)
    return x_q, scale


def reference_gmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,
):
    num_tokens = lhs.shape[0]
    num_groups, in_size, out_size = rhs.shape
    assert lhs.shape[1] == in_size

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    elif jnp.isscalar(group_offset):
        assert group_offset.size == 1
        if jnp.isscalar(group_offset):
            group_offset = group_offset[None]

    num_blocks = rhs_scale.shape[1] if rhs_scale is not None else 1
    block_size = in_size // num_blocks

    start = 0
    gmm_out = []
    for global_group in range(group_sizes.size):
        group_size = group_sizes[global_group]

        group = global_group - group_offset[0]
        end = min(start + group_size, num_tokens)
        group_size = end - start
        if group >= 0 and group < num_groups:
            lhs_slice = lhs[start:end]
            rhs_slice = rhs[group]

            out = 0
            for block in range(num_blocks):
                block_start = block * block_size
                block_end = block_start + block_size
                lhs_block = lhs_slice[:, block_start:block_end].astype(jnp.float32)
                rhs_block = rhs_slice[block_start:block_end, :].astype(jnp.float32)

                acc = jnp.einsum("bd,dh->bh", lhs_block, rhs_block)
                if rhs_scale is not None:
                    acc *= rhs_scale[group][block]
                out += acc
            if rhs_bias is not None:
                out = out + rhs_bias[group]
        else:
            out = jnp.zeros((group_size, out_size), dtype=lhs.dtype)

        gmm_out.append(out.astype(lhs.dtype))
        start = end

    return jnp.concat(gmm_out, axis=0)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class GmmTest(jtu.JaxTestCase):

    @parameterized.product(
        batch_size=[128],
        in_size=[512, 1024],
        out_size=[512, 1024],
        num_groups=[16, 32],
        has_bias=[True, False],
        group_offset=[0, 2, 3],
        version=[GMM_VERSION_V1],
    )
    def test_gmm(self, batch_size, in_size, out_size, num_groups, has_bias, group_offset, version):
        num_local_groups = num_groups - group_offset
        key = jax.random.key(0)

        lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
        rhs = jax.random.normal(key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16)
        rhs_bias = None
        if has_bias:
            rhs_bias = jax.random.normal(key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16)

        group_sizes = get_group_sizes(batch_size, num_groups)
        group_offset = jnp.array(group_offset, dtype=jnp.int32)

        expected = reference_gmm(
            lhs, rhs, group_sizes, rhs_bias=rhs_bias, group_offset=group_offset
        )

        actual = gmm_wrapper(
            lhs,
            rhs,
            group_sizes,
            version=version,
            rhs_bias=rhs_bias,
            group_offset=group_offset,
        )

        self.assertArraysAllClose(actual, expected)

    @parameterized.product(
        batch_size=[128],
        in_size=[512, 1024],
        out_size=[512, 1024],
        num_groups=[16, 32],
        has_bias=[True, False],
        weight_dtype=[jnp.int8, jnp.float8_e4m3fn],
        block_size=[64, 128, 256, 512],
        group_offset=[0, 2, 3],
        version=[GMM_VERSION_V1],
    )
    def test_gmm_weight_quantized(
        self,
        batch_size,
        in_size,
        out_size,
        num_groups,
        has_bias,
        weight_dtype,
        block_size,
        group_offset,
        version,
    ):
        num_local_groups = num_groups - group_offset
        key = jax.random.key(0)

        lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
        rhs = jax.random.uniform(key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1)
        rhs_q, rhs_scale = quantize_tensor(rhs, weight_dtype, axis=1, block_size=block_size)
        rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

        rhs_bias = None
        if has_bias:
            rhs_bias = jax.random.normal(key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16)

        group_sizes = get_group_sizes(batch_size, num_groups)
        group_offset = jnp.array(group_offset, dtype=jnp.int32)

        expected = reference_gmm(
            lhs,
            rhs_q,
            group_sizes,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            group_offset=group_offset,
        )

        actual = gmm_wrapper(
            lhs,
            rhs_q,
            group_sizes,
            version=version,
            rhs_scale=rhs_scale,
            group_offset=group_offset,
            rhs_bias=rhs_bias,
            maybe_quantize_lhs=False,
        ).astype(lhs.dtype)

        self.assertArraysAllClose(actual, expected, atol=3e-1, rtol=3e-1)

    @parameterized.product(
        batch_size=[128],
        in_size=[1024],
        out_size=[512],
        num_groups=[16],
        weight_dtype=[jnp.int8, jnp.float8_e4m3fn],
        block_size=[1024],
        tile_k=[128, 256, 512],
        group_offset=[0],
    )
    def test_gmm_weight_quantized_block_larger_than_tile_k(
        self,
        batch_size,
        in_size,
        out_size,
        num_groups,
        weight_dtype,
        block_size,
        tile_k,
        group_offset,
    ):
        """Test that quant_block_size > tile_k is handled correctly."""
        num_local_groups = num_groups - group_offset
        key = jax.random.key(0)

        lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
        rhs = jax.random.uniform(key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1)
        rhs_q, rhs_scale = quantize_tensor(rhs, weight_dtype, axis=1, block_size=block_size)
        rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

        group_sizes = get_group_sizes(batch_size, num_groups)
        group_offset = jnp.array(group_offset, dtype=jnp.int32)

        expected = reference_gmm(
            lhs,
            rhs_q,
            group_sizes,
            rhs_scale=rhs_scale,
            group_offset=group_offset,
        )

        tile_info = TileSizes(tile_m=128, tile_k=tile_k, tile_n=out_size)
        actual = gmm_v2(
            lhs,
            rhs_q,
            group_sizes,
            rhs_scale=rhs_scale,
            group_offset=group_offset,
            tile_info=tile_info,
            maybe_quantize_lhs=False,
        ).astype(lhs.dtype)

        self.assertArraysAllClose(actual, expected, atol=3e-1, rtol=3e-1)

    @parameterized.product(
        batch_size=[128],
        in_size=[1024],
        out_size=[512],
        num_groups=[16],
        weight_dtype=[jnp.int8, jnp.float8_e4m3fn],
        block_size=[1024],
        tile_k=[128, 256, 512],
        group_offset=[0],
    )
    def test_gmm_activation_weight_quantized_block_larger_than_tile_k(
        self,
        batch_size,
        in_size,
        out_size,
        num_groups,
        weight_dtype,
        block_size,
        tile_k,
        group_offset,
    ):
        """Test activation+weight quantized path with quant_block_size > tile_k."""
        num_local_groups = num_groups - group_offset
        key = jax.random.key(0)

        lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
        rhs = jax.random.uniform(key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1)
        rhs_q, rhs_scale = quantize_tensor(rhs, weight_dtype, axis=1, block_size=block_size)
        rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

        group_sizes = get_group_sizes(batch_size, num_groups)
        group_offset = jnp.array(group_offset, dtype=jnp.int32)

        expected = reference_gmm(
            lhs,
            rhs_q,
            group_sizes,
            rhs_scale=rhs_scale,
            group_offset=group_offset,
        )

        tile_info = TileSizes(tile_m=128, tile_k=tile_k, tile_n=out_size)
        actual = gmm_v2(
            lhs,
            rhs_q,
            group_sizes,
            rhs_scale=rhs_scale,
            group_offset=group_offset,
            tile_info=tile_info,
            maybe_quantize_lhs=True,
        ).astype(lhs.dtype)

        self.assertArraysAllClose(actual, expected, atol=1.2, rtol=1.2)

    @parameterized.product(
        batch_size=[128],
        in_size=[512, 1024],
        out_size=[512, 1024],
        num_groups=[16, 32],
        weight_dtype=[jnp.int8, jnp.float8_e4m3fn],
        block_size=[512],
        group_offset=[0, 2, 3],
    )
    def test_gmm_v1_activation_weight_quantized(
        self,
        batch_size,
        in_size,
        out_size,
        num_groups,
        weight_dtype,
        block_size,
        group_offset,
    ):
        """V1 path: manually quantize LHS, call gmm, then rescale by x_scale."""
        num_local_groups = num_groups - group_offset
        key = jax.random.key(0)

        lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
        rhs = jax.random.uniform(key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1)
        rhs_q, rhs_scale = quantize_tensor(rhs, weight_dtype, axis=1, block_size=block_size)
        rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

        group_sizes = get_group_sizes(batch_size, num_groups)
        group_offset = jnp.array(group_offset, dtype=jnp.int32)

        expected = reference_gmm(
            lhs,
            rhs_q,
            group_sizes,
            rhs_scale=rhs_scale,
            group_offset=group_offset,
        )

        # Per-token activation quantization (mirrors moe.py _gmm_compute V1 path)
        lhs_abs_max = jnp.max(jnp.abs(lhs), axis=-1, keepdims=True)
        act_dtype_max = (
            float(jnp.finfo(weight_dtype).max)
            if jnp.issubdtype(weight_dtype, jnp.floating)
            else float(jnp.iinfo(weight_dtype).max)
        )
        x_scale = lhs_abs_max / act_dtype_max
        lhs_q = jnp.clip(lhs / x_scale, -act_dtype_max, act_dtype_max).astype(weight_dtype)

        actual = gmm_wrapper(
            lhs_q,
            rhs_q,
            group_sizes,
            version=GMM_VERSION_V1,
            rhs_scale=rhs_scale,
            group_offset=group_offset,
            preferred_element_type=jnp.bfloat16,
        )
        actual = actual * x_scale

        self.assertArraysAllClose(actual, expected, atol=1.1, rtol=1.1)

    @parameterized.product(
        batch_size=[128],
        in_size=[512, 1024],
        out_size=[512, 1024],
        num_groups=[16, 32],
        weight_dtype=[jnp.int8, jnp.float8_e4m3fn],
        block_size=[512, 1024],
        group_offset=[0, 2, 3],
    )
    def test_gmm_v2_activation_weight_quantized(
        self,
        batch_size,
        in_size,
        out_size,
        num_groups,
        weight_dtype,
        block_size,
        group_offset,
    ):
        if block_size > in_size:
            self.skipTest("block_size must be <= in_size")
        num_local_groups = num_groups - group_offset
        key = jax.random.key(0)

        lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
        rhs = jax.random.uniform(key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1)
        rhs_q, rhs_scale = quantize_tensor(rhs, weight_dtype, axis=1, block_size=block_size)
        rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
        group_sizes = get_group_sizes(batch_size, num_groups)
        group_offset = jnp.array(group_offset, dtype=jnp.int32)

        expected = reference_gmm(
            lhs,
            rhs_q,
            group_sizes,
            rhs_scale=rhs_scale,
            group_offset=group_offset,
        )

        actual = gmm_v2(
            lhs,
            rhs_q,
            group_sizes,
            rhs_scale=rhs_scale,
            group_offset=group_offset,
            maybe_quantize_lhs=True,
        ).astype(lhs.dtype)

        self.assertArraysAllClose(actual, expected, atol=1.1, rtol=1.1)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
