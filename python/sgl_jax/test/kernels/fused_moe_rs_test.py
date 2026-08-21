# Copyright 2026 The sgl-jax Authors. All rights reserved.
"""Correctness tests for the fused reduce-scatter EP-MoE kernel.

The optimized kernel is compared with an explicit MoE reference whose result
does not depend on the selected RS block configuration.  The FP8 reference
models the public per-channel W8A8 contract: full-row activation quantization
for GMM1, BF16 post-activation input for GMM2, FP32 routing accumulation, and a
BF16 backend boundary.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.fused_moe.fused_rs import fused_moe_func_rs
from sgl_jax.srt.kernels.fused_moe.fused_rs.gmm_fused_rs_nodedup import (
    set_fused_rs_block_sizes_override,
)
from sgl_jax.test.test_utils import create_device_mesh

jax.config.parse_flags_with_absl()

FP8 = jnp.float8_e4m3fn
_HIDDEN = 512
_INTERMEDIATE = 512
_NUM_EXPERTS = 16
_TOP_K = 8


def _pattern(shape: tuple[int, ...], *, offset: int) -> jax.Array:
    """Return signed, non-uniform values that distinguish every tensor axis."""
    value = jnp.zeros(shape, dtype=jnp.int32)
    for axis, size in enumerate(shape):
        axis_value = jnp.arange(size, dtype=jnp.int32)
        axis_value = axis_value.reshape(
            (1,) * axis + (size,) + (1,) * (len(shape) - axis - 1)
        )
        value = value + axis_value * (17 + offset + axis * 12)
    code = (value + offset) % 16
    sign = jnp.where(code < 8, -1.0, 1.0)
    magnitude = 0.25 + (code % 4).astype(jnp.float32) * 0.25
    return sign * magnitude


def _make_inputs(mesh: jax.sharding.Mesh, *, quantized: bool, num_tokens: int):
    expert_axis = ("data", "tensor")
    token_sharding = NamedSharding(mesh, P(expert_axis, None))
    weight_sharding = NamedSharding(mesh, P(expert_axis, None, None))
    scale_sharding = NamedSharding(mesh, P(expert_axis, None, None, None))
    replicated = NamedSharding(mesh, P())

    def make_replicated():
        tokens = _pattern((num_tokens, _HIDDEN), offset=1).astype(jnp.bfloat16)
        weight_dtype = FP8 if quantized else jnp.bfloat16
        w1 = _pattern((_NUM_EXPERTS, _HIDDEN, _INTERMEDIATE), offset=3).astype(
            weight_dtype
        )
        w3 = _pattern((_NUM_EXPERTS, _HIDDEN, _INTERMEDIATE), offset=7).astype(
            weight_dtype
        )
        w2 = _pattern((_NUM_EXPERTS, _INTERMEDIATE, _HIDDEN), offset=11).astype(
            weight_dtype
        )

        token = jnp.arange(num_tokens, dtype=jnp.int32)[:, None]
        slot = jnp.arange(_TOP_K, dtype=jnp.int32)[None, :]
        topk_ids = (token * 3 + slot * 5) % _NUM_EXPERTS
        logits = (((token * 7 + slot * 11) % 19).astype(jnp.float32) - 9.0) / 5.0
        topk_weights = jax.nn.softmax(logits, axis=-1)

        expert = jnp.arange(_NUM_EXPERTS, dtype=jnp.int32)[:, None, None, None]

        def scale(out_size: int, offset: int):
            output = jnp.arange(out_size, dtype=jnp.int32)[None, None, None, :]
            return (
                0.01
                + (expert % 5).astype(jnp.float32) * 0.001
                + ((expert * 3 + output * 5 + offset) % 11).astype(jnp.float32) * 0.0002
            )

        scales = (
            scale(_INTERMEDIATE, 1),
            scale(_INTERMEDIATE, 5),
            scale(_HIDDEN, 9),
        )
        return (
            tokens,
            w1,
            w3,
            w2,
            *scales,
            topk_weights,
            topk_ids,
        )

    replicated_inputs = jax.jit(
        make_replicated,
        out_shardings=(replicated,) * 9,
    )()
    tokens, w1, w3, w2, s1, s3, s2, topk_weights, topk_ids = replicated_inputs
    kernel_inputs = (
        jax.sharding.reshard(tokens, token_sharding),
        jax.sharding.reshard(w1, weight_sharding),
        jax.sharding.reshard(w3, weight_sharding),
        jax.sharding.reshard(w2, weight_sharding),
        None if not quantized else jax.sharding.reshard(s1, scale_sharding),
        None if not quantized else jax.sharding.reshard(s3, scale_sharding),
        None if not quantized else jax.sharding.reshard(s2, scale_sharding),
        jax.sharding.reshard(topk_weights, token_sharding),
        jax.sharding.reshard(topk_ids, token_sharding),
    )
    return replicated_inputs, kernel_inputs, token_sharding


def _explicit_reference(inputs, *, quantized: bool):
    """Compute the routed MoE result without the fused RS kernel."""
    tokens, w1, w3, w2, s1, s3, s2, topk_weights, topk_ids = inputs
    tokens_f32 = tokens.astype(jnp.float32)
    output = jnp.zeros(tokens.shape, dtype=jnp.float32)
    fp8_max = jnp.asarray(jnp.finfo(FP8).max, dtype=jnp.float32)

    if quantized:
        token_amax = jnp.max(jnp.abs(tokens_f32), axis=-1, keepdims=True)
        token_scale = jnp.maximum(token_amax, 1e-12) / fp8_max
        gmm1_lhs = jnp.clip(tokens_f32 / token_scale, -fp8_max, fp8_max).astype(FP8)
    else:
        token_scale = jnp.ones(tokens_f32.shape[:-1] + (1,), dtype=jnp.float32)
        gmm1_lhs = tokens

    for expert_id in range(_NUM_EXPERTS):
        if quantized:
            gate = jax.lax.dot_general(
                gmm1_lhs,
                w1[expert_id],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(jnp.float32)
            up = jax.lax.dot_general(
                gmm1_lhs,
                w3[expert_id],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(jnp.float32)
            gate *= token_scale * s1[expert_id, 0, 0, :][None, :]
            up *= token_scale * s3[expert_id, 0, 0, :][None, :]
        else:
            gate = tokens_f32 @ w1[expert_id].astype(jnp.float32)
            up = tokens_f32 @ w3[expert_id].astype(jnp.float32)

        intermediate = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
        if quantized:
            down = jax.lax.dot_general(
                intermediate,
                w2[expert_id].astype(jnp.bfloat16),
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(jnp.float32)
            down *= s2[expert_id, 0, 0, :][None, :]
        else:
            down = intermediate.astype(jnp.float32) @ w2[expert_id].astype(jnp.float32)

        route_weight = jnp.sum(
            jnp.where(topk_ids == expert_id, topk_weights, 0.0),
            axis=-1,
        )
        output += down.astype(jnp.bfloat16).astype(jnp.float32) * route_weight[:, None]

    return output.astype(tokens.dtype)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class MoERSKernelTest(jtu.JaxTestCase):
    def setUp(self):
        super().setUp()
        self.mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
        self.assertEqual(_NUM_EXPERTS % self.mesh.size, 0)

    def tearDown(self):
        set_fused_rs_block_sizes_override(None)
        super().tearDown()

    def _test_config(self, config, *, quantized: bool, num_tokens: int):
        reference_inputs, kernel_inputs, token_sharding = _make_inputs(
            self.mesh,
            quantized=quantized,
            num_tokens=num_tokens,
        )
        tokens, w1, w3, w2, s1, s3, s2, topk_weights, topk_ids = kernel_inputs

        expected = jax.jit(
            _explicit_reference,
            static_argnames=("quantized",),
            out_shardings=token_sharding,
        )(reference_inputs, quantized=quantized)

        set_fused_rs_block_sizes_override(config)
        actual = fused_moe_func_rs(
            hidden_states=tokens,
            w1=w1,
            w3=w3,
            w2=w2,
            w1_scale=s1,
            w3_scale=s3,
            w2_scale=s2,
            w1_bias=None,
            w2_bias=None,
            gating_output=None,
            topk=_TOP_K,
            renormalize=False,
            mesh=self.mesh,
            activation="silu",
            scoring_fn="softmax",
            topk_weights=topk_weights,
            topk_indices=topk_ids,
        )
        jax.block_until_ready((expected, actual))

        expected_host = np.asarray(jax.device_get(expected), dtype=np.float32)
        actual_host = np.asarray(jax.device_get(actual), dtype=np.float32)
        self.assertTrue(np.isfinite(actual_host).all())
        self.assertAllClose(
            actual_host,
            expected_host,
            atol=5e-2 if quantized else 2e-2,
            rtol=5e-2 if quantized else 2e-2,
        )

    def test_bf16_full_resident_matches_explicit_reference(self):
        self._test_config(
            (128, 512, 512, 512, 512, 1, 1),
            quantized=False,
            num_tokens=48,
        )

    @parameterized.named_parameters(
        ("m128_full_resident", (128, 512, 512, 512, 512, 1, 1), 48),
        ("m256_cache_w1", (256, 512, 256, 512, 128, 2, 2), 64),
        ("m256_cache_w2", (256, 512, 128, 512, 512, 2, 1), 80),
        ("m256_stream_both", (256, 512, 128, 512, 128, 2, 2), 96),
        ("m384_stream_both", (384, 512, 128, 512, 128, 2, 2), 144),
    )
    def test_fp8_per_channel_config_matches_explicit_reference(
        self, config, num_tokens
    ):
        self._test_config(config, quantized=True, num_tokens=num_tokens)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
