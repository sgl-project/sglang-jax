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
from sgl_jax.srt.kernels.fused_moe.fused_rs.fused_moe_rs import (
    _dequantize_hidden_per_rank,
    _quantize_hidden_per_tensor,
)
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
    """Return positive, non-uniform values that distinguish every tensor axis."""
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
    # The positive offset prevents the small explicit-oracle shapes from
    # degenerating to an almost-zero output through symmetric cancellation.
    return 0.5 + sign * magnitude * 0.25


def _make_inputs(
    mesh: jax.sharding.Mesh,
    *,
    quantized: bool,
    num_tokens: int,
    distinct_shard_scales: bool = False,
):
    expert_axis = ("data", "tensor")
    token_sharding = NamedSharding(mesh, P(expert_axis, None))
    weight_sharding = NamedSharding(mesh, P(expert_axis, None, None))
    scale_sharding = NamedSharding(mesh, P(expert_axis, None, None, None))
    replicated = NamedSharding(mesh, P())

    def make_replicated():
        tokens = _pattern((num_tokens, _HIDDEN), offset=1).astype(jnp.bfloat16)
        if distinct_shard_scales:
            local_tokens = num_tokens // mesh.size
            shard_id = jnp.arange(num_tokens, dtype=jnp.int32) // local_tokens
            shard_factor = (1.0 + shard_id.astype(jnp.float32) * 0.25)[:, None]
            tokens = (tokens.astype(jnp.float32) * shard_factor).astype(jnp.bfloat16)
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


def _explicit_reference(
    inputs,
    *,
    quantized: bool,
    fp8_hidden_all_gather: bool = False,
    ep_size: int = 1,
):
    """Compute the routed MoE result without the fused RS kernel."""
    tokens, w1, w3, w2, s1, s3, s2, topk_weights, topk_ids = inputs
    tokens_f32 = tokens.astype(jnp.float32)
    output = jnp.zeros(tokens.shape, dtype=jnp.float32)
    fp8_max = jnp.asarray(jnp.finfo(FP8).max, dtype=jnp.float32)

    if quantized:
        if fp8_hidden_all_gather:
            local_tokens = tokens.shape[0] // ep_size
            token_shards = tokens_f32.reshape(ep_size, local_tokens, tokens.shape[-1])
            shard_amax = jnp.max(
                jnp.abs(token_shards),
                axis=(1, 2),
                keepdims=True,
            )
            shard_scale = jnp.maximum(shard_amax, 1e-12) / fp8_max
            communicated_lhs = jnp.clip(
                token_shards / shard_scale,
                -fp8_max,
                fp8_max,
            ).astype(FP8)
            # The FP8 collective is dequantized locally to BF16 before the
            # established fused-RS W8A8 path performs its normal per-row GMM1
            # activation quantization.
            communicated_lhs = (
                communicated_lhs.astype(jnp.float32) * shard_scale
            ).astype(tokens.dtype)
            communicated_f32 = communicated_lhs.reshape(tokens.shape).astype(
                jnp.float32
            )
            token_amax = jnp.max(
                jnp.abs(communicated_f32), axis=-1, keepdims=True
            )
            token_scale = jnp.maximum(token_amax, 1e-12) / fp8_max
            gmm1_lhs = jnp.clip(
                communicated_f32 / token_scale, -fp8_max, fp8_max
            ).astype(FP8)
        else:
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


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    numerator = np.linalg.norm(actual.astype(np.float64) - expected.astype(np.float64))
    denominator = max(np.linalg.norm(expected.astype(np.float64)), 1e-12)
    return float(numerator / denominator)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class MoERSKernelTest(jtu.JaxTestCase):
    def setUp(self):
        super().setUp()
        self.mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
        self.assertEqual(_NUM_EXPERTS % self.mesh.size, 0)

    def tearDown(self):
        set_fused_rs_block_sizes_override(None)
        super().tearDown()

    def test_fp8_hidden_quantization_uses_one_physical_shard_scale(self):
        hidden = jnp.asarray(
            [
                [1.0, -2.0, 0.5, -0.25],
                [4.0, -1.0, 2.0, -3.0],
                [8.0, -8.0, 4.0, -4.0],
            ],
            dtype=jnp.bfloat16,
        )
        topk_ids = jnp.asarray(
            [[0, 1], [1, 0], [-1, -1]],
            dtype=jnp.int32,
        )

        quantized, scale = _quantize_hidden_per_tensor(hidden, topk_ids)
        quantized_host = np.asarray(quantized, dtype=np.float32)
        scale_host = float(np.asarray(scale))

        self.assertEqual(quantized.dtype, FP8)
        self.assertEqual(scale.shape, ())
        self.assertAlmostEqual(scale_host, 8.0 / float(jnp.finfo(FP8).max))
        dequantized = quantized.astype(jnp.float32) * scale
        self.assertAllClose(
            dequantized,
            hidden.astype(jnp.float32),
            atol=scale_host,
            rtol=0.05,
        )

    def test_fp8_hidden_quantization_is_invariant_to_routing_padding(self):
        hidden = jnp.asarray(
            [
                [1.0, -2.0, 0.5, -0.25],
                [4.0, -1.0, 2.0, -3.0],
                [8.0, -8.0, 4.0, -4.0],
            ],
            dtype=jnp.bfloat16,
        )
        all_active_ids = jnp.asarray(
            [[0, 1], [1, 0], [0, 1]],
            dtype=jnp.int32,
        )
        padded_ids = all_active_ids.at[2].set(-1)

        all_active_payload, all_active_scale = _quantize_hidden_per_tensor(
            hidden, all_active_ids
        )
        padded_payload, padded_scale = _quantize_hidden_per_tensor(
            hidden, padded_ids
        )

        self.assertAllClose(padded_scale, all_active_scale, atol=0.0, rtol=0.0)
        self.assertArraysEqual(padded_payload, all_active_payload)

    def test_fp8_hidden_dequantization_materializes_rank_scales_in_row_order(self):
        payload = jnp.asarray(
            [
                [1.0, -2.0],
                [3.0, -4.0],
                [5.0, -6.0],
                [7.0, -8.0],
            ],
            dtype=FP8,
        )
        rank_scales = jnp.asarray([0.25, 0.5], dtype=jnp.float32)

        actual = _dequantize_hidden_per_rank(
            payload,
            rank_scales,
            rows_per_rank=2,
            out_dtype=jnp.bfloat16,
        )
        expected = jnp.asarray(
            [
                [0.25, -0.5],
                [0.75, -1.0],
                [2.5, -3.0],
                [3.5, -4.0],
            ],
            dtype=jnp.bfloat16,
        )

        self.assertArraysEqual(actual, expected)

    def _test_config(
        self,
        config,
        *,
        quantized: bool,
        num_tokens: int,
        fp8_hidden_all_gather: bool = False,
    ):
        reference_inputs, kernel_inputs, token_sharding = _make_inputs(
            self.mesh,
            quantized=quantized,
            num_tokens=num_tokens,
            distinct_shard_scales=fp8_hidden_all_gather,
        )
        tokens, w1, w3, w2, s1, s3, s2, topk_weights, topk_ids = kernel_inputs

        expected = jax.jit(
            _explicit_reference,
            static_argnames=("quantized", "fp8_hidden_all_gather", "ep_size"),
            out_shardings=token_sharding,
        )(
            reference_inputs,
            quantized=quantized,
            fp8_hidden_all_gather=fp8_hidden_all_gather,
            ep_size=self.mesh.size,
        )

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
            fp8_hidden_all_gather=fp8_hidden_all_gather,
        )
        jax.block_until_ready((expected, actual))

        expected_host = np.asarray(jax.device_get(expected), dtype=np.float32)
        actual_host = np.asarray(jax.device_get(actual), dtype=np.float32)
        self.assertGreater(np.linalg.norm(expected_host), 1e-3)
        self.assertTrue(np.isfinite(actual_host).all())
        self.assertAllClose(
            actual_host,
            expected_host,
            atol=5e-2 if quantized else 2e-2,
            rtol=5e-2 if quantized else 2e-2,
        )

    def _test_padding_contract(
        self,
        config,
        *,
        num_tokens: int,
        active_tokens_per_device: int,
        fp8_hidden_all_gather: bool = False,
    ):
        reference_inputs, kernel_inputs, token_sharding = _make_inputs(
            self.mesh,
            quantized=True,
            num_tokens=num_tokens,
            distinct_shard_scales=fp8_hidden_all_gather,
        )
        tokens, w1, w3, w2, s1, s3, s2, topk_weights, topk_ids = kernel_inputs

        self.assertEqual(num_tokens % self.mesh.size, 0)
        local_tokens = num_tokens // self.mesh.size
        self.assertLess(active_tokens_per_device, local_tokens)
        valid_mask = (
            np.arange(num_tokens, dtype=np.int32) % local_tokens
        ) < active_tokens_per_device

        reference_topk_weights = reference_inputs[-2]
        reference_topk_ids = reference_inputs[-1]
        valid_mask_device = jnp.asarray(valid_mask)[:, None]
        padded_topk_weights = jnp.where(
            valid_mask_device,
            reference_topk_weights,
            jnp.asarray(0.0, dtype=reference_topk_weights.dtype),
        )
        padded_topk_ids = jnp.where(
            valid_mask_device,
            reference_topk_ids,
            jnp.asarray(-1, dtype=reference_topk_ids.dtype),
        )
        padded_reference_inputs = (
            *reference_inputs[:-2],
            padded_topk_weights,
            padded_topk_ids,
        )
        padded_topk_weights = jax.sharding.reshard(
            padded_topk_weights, token_sharding
        )
        padded_topk_ids = jax.sharding.reshard(padded_topk_ids, token_sharding)

        expected_padded = jax.jit(
            _explicit_reference,
            static_argnames=("quantized", "fp8_hidden_all_gather", "ep_size"),
            out_shardings=token_sharding,
        )(
            padded_reference_inputs,
            quantized=True,
            fp8_hidden_all_gather=fp8_hidden_all_gather,
            ep_size=self.mesh.size,
        )

        set_fused_rs_block_sizes_override(config)

        def run(weights, indices):
            return fused_moe_func_rs(
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
                topk_weights=weights,
                topk_indices=indices,
                fp8_hidden_all_gather=fp8_hidden_all_gather,
            )

        all_active = run(topk_weights, topk_ids)
        padded = run(padded_topk_weights, padded_topk_ids)
        jax.block_until_ready((expected_padded, all_active, padded))

        expected_padded_host = np.asarray(
            jax.device_get(expected_padded), dtype=np.float32
        )
        all_active_host = np.asarray(jax.device_get(all_active), dtype=np.float32)
        padded_host = np.asarray(jax.device_get(padded), dtype=np.float32)
        self.assertGreater(np.linalg.norm(expected_padded_host[valid_mask]), 1e-3)
        self.assertTrue(np.isfinite(padded_host).all())
        self.assertAllClose(
            padded_host,
            expected_padded_host,
            atol=5e-2,
            rtol=5e-2,
        )

        valid_rel_l2 = _relative_l2(
            padded_host[valid_mask], all_active_host[valid_mask]
        )
        oracle_rel_l2 = _relative_l2(padded_host, expected_padded_host)
        invalid_max_abs = float(np.max(np.abs(padded_host[~valid_mask])))
        self.assertLessEqual(
            valid_rel_l2,
            0.01,
            msg=(
                "valid output changed after padding: "
                f"rel_l2={valid_rel_l2}, oracle_rel_l2={oracle_rel_l2}, "
                f"invalid_max_abs={invalid_max_abs}"
            ),
        )
        self.assertEqual(
            invalid_max_abs,
            0.0,
            msg=f"invalid padded output was not zero: max_abs={invalid_max_abs}",
        )

    def test_bf16_full_resident_matches_explicit_reference(self):
        self._test_config(
            (128, 512, 512, 512, 512, 1, 1),
            quantized=False,
            num_tokens=48,
        )

    @parameterized.named_parameters(
        ("m128_full_resident", (128, 512, 512, 512, 512, 1, 1), 48),
        ("m256_full_resident", (256, 512, 512, 512, 512, 1, 1), 72),
        ("m384_full_resident", (384, 512, 512, 512, 512, 1, 1), 136),
    )
    def test_fp8_per_channel_config_matches_explicit_reference(
        self, config, num_tokens
    ):
        self._test_config(config, quantized=True, num_tokens=num_tokens)

    @parameterized.named_parameters(
        ("m128_full_resident", (128, 512, 512, 512, 512, 1, 1), 48),
        ("m256_full_resident", (256, 512, 512, 512, 512, 1, 1), 72),
        ("m384_full_resident", (384, 512, 512, 512, 512, 1, 1), 136),
    )
    def test_fp8_hidden_all_gather_matches_per_tensor_reference(
        self, config, num_tokens
    ):
        self._test_config(
            config,
            quantized=True,
            num_tokens=num_tokens,
            fp8_hidden_all_gather=True,
        )

    @parameterized.named_parameters(
        ("m128_full_resident", (128, 512, 512, 512, 512, 1, 1), 48, 3),
        ("m256_full_resident", (256, 512, 512, 512, 512, 1, 1), 72, 5),
        ("m384_full_resident", (384, 512, 512, 512, 512, 1, 1), 136, 8),
    )
    def test_fp8_per_channel_padding_contract(
        self,
        config,
        num_tokens,
        active_tokens_per_device,
    ):
        self._test_padding_contract(
            config,
            num_tokens=num_tokens,
            active_tokens_per_device=active_tokens_per_device,
        )

    @parameterized.named_parameters(
        ("m128_full_resident", (128, 512, 512, 512, 512, 1, 1), 48, 3),
        ("m256_full_resident", (256, 512, 512, 512, 512, 1, 1), 72, 5),
        ("m384_full_resident", (384, 512, 512, 512, 512, 1, 1), 136, 8),
    )
    def test_fp8_hidden_all_gather_padding_contract(
        self,
        config,
        num_tokens,
        active_tokens_per_device,
    ):
        self._test_padding_contract(
            config,
            num_tokens=num_tokens,
            active_tokens_per_device=active_tokens_per_device,
            fp8_hidden_all_gather=True,
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
