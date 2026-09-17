import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.hyperconnection import (
    HYPERCONNECTION_CLASS_DICT,
    GatedResidual,
    HyperConnectionBase,
    HyperConnectionConfig,
)
from sgl_jax.test.test_utils import CustomTestCase

HIDDEN_SIZE = 16
HC_COUNT = 4
HC_LOWRANK = 8
HYPER_SIZE = HC_COUNT * HIDDEN_SIZE
EPSILON = 1e-6
SEED = 42
TOKENS = 6

ATOL = 2e-5
RTOL = 2e-5


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _ref_normalize(hyper_input, norm_w, per_branch):
    """fp64 ground truth for hc_norm, one stream at a time.

    Loops over HC instead of broadcasting, so a reshape or grouping mistake in
    the vectorized implementation does not reproduce here. Each stream is
    normalized over its own HS elements either way; per_branch only changes
    whether the affine weight is per stream or shared.
    """
    x = hyper_input.astype(np.float64)
    streams = x.reshape(*x.shape[:-1], HC_COUNT, HIDDEN_SIZE)
    w = norm_w.astype(np.float64)

    normed = np.empty_like(streams)
    for j in range(HC_COUNT):
        s = streams[..., j, :]
        variance = np.mean(s**2, axis=-1, keepdims=True)
        w_j = w[j * HIDDEN_SIZE : (j + 1) * HIDDEN_SIZE] if per_branch else w
        normed[..., j, :] = (s / np.sqrt(variance + EPSILON)) * (1.0 + w_j)
    return normed.reshape(x.shape)


def _ref_mix(hyper_input, norm_w, down_w, up_w, per_branch):
    """fp64 ground truth for GatedResidual.mix; returns (mixed, normed_flat)."""
    x = hyper_input.astype(np.float64)
    normed_flat = _ref_normalize(hyper_input, norm_w, per_branch)

    gate = normed_flat @ down_w.astype(np.float64)
    gate = gate / HC_COUNT
    gate = gate * _sigmoid(gate)  # silu
    gate = _sigmoid(gate @ up_w.astype(np.float64))
    gate = gate.reshape(*x.shape[:-1], HC_COUNT, HIDDEN_SIZE)

    mixed = np.mean(gate * normed_flat.reshape(gate.shape), axis=-2)
    return mixed, normed_flat


def _ref_combine(hyper_input, normed_flat, block_output, inject_w):
    """fp64 ground truth for GatedResidual.combine, one stream at a time."""
    x = hyper_input.astype(np.float64)
    out_b = block_output.astype(np.float64)

    inject = normed_flat @ inject_w.astype(np.float64)
    inject = 2.0 * _sigmoid(inject / HC_COUNT)

    residual = x.reshape(*x.shape[:-1], HC_COUNT, HIDDEN_SIZE)
    out = np.empty_like(residual)
    for j in range(HC_COUNT):
        out[..., j, :] = residual[..., j, :] + out_b * inject[..., j : j + 1]
    return out.reshape(x.shape)


def _make_mesh():
    devices = np.array(jax.devices())
    return Mesh(
        devices[:1].reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _make_config(mesh, per_branch=True):
    return HyperConnectionConfig(
        hc_count=HC_COUNT,
        hidden_size=HIDDEN_SIZE,
        params_dtype=jnp.float32,
        hc_lowrank=HC_LOWRANK,
        rms_norm_eps=EPSILON,
        hc_per_branch_norm=per_branch,
        mesh=mesh,
    )


def _make_weights(rng, per_branch=True):
    norm_dim = HYPER_SIZE if per_branch else HIDDEN_SIZE
    return {
        "norm": rng.standard_normal(norm_dim).astype(np.float32),
        "down": rng.standard_normal((HYPER_SIZE, HC_LOWRANK)).astype(np.float32),
        "up": rng.standard_normal((HC_LOWRANK, HYPER_SIZE)).astype(np.float32),
        "inject": rng.standard_normal((HYPER_SIZE, HC_COUNT)).astype(np.float32),
    }


def _assign(param, value, mesh, spec):
    # Cast to the param's own dtype: assigning fp32 into a bf16 param is a
    # narrowing scatter, which JAX currently warns about and will later reject.
    value = jnp.array(value, param[...].dtype)
    param[...] = jax.device_put(value, NamedSharding(mesh, spec))


def _make_layer(weights, mesh, per_branch=True, use_mix=True, use_combine=True):
    with jax.set_mesh(mesh):
        layer = GatedResidual(
            _make_config(mesh, per_branch),
            use_mix=use_mix,
            use_combine=use_combine,
        )
    _assign(layer.hc_norm.weight, weights["norm"], mesh, P(None))
    if use_mix:
        _assign(layer.input_mix_weight_down.weight, weights["down"], mesh, P(None, None))
        _assign(layer.input_mix_weight_up.weight, weights["up"], mesh, P(None, None))
    if use_combine:
        _assign(layer.block_inject_weight.weight, weights["inject"], mesh, P(None, None))
    return layer


class TestHyperConnectionBase(CustomTestCase):
    """The ungated scheme: average to read, add to write."""

    def test_mix_averages_the_streams(self):
        mesh = _make_mesh()
        rng = np.random.default_rng(SEED)
        hyper_input = rng.standard_normal((TOKENS, HYPER_SIZE)).astype(np.float32)

        with jax.set_mesh(mesh):
            layer = HyperConnectionBase(_make_config(mesh))
            mixed, residuals = layer.mix(jnp.array(hyper_input))

        expected = hyper_input.reshape(TOKENS, HC_COUNT, HIDDEN_SIZE).mean(axis=-2)
        np.testing.assert_allclose(np.array(mixed), expected, atol=ATOL, rtol=ATOL)
        # 2-tuple like the gated subclass's, so both share one contract.
        self.assertEqual(len(residuals), 2)

    def test_combine_adds_to_every_stream(self):
        mesh = _make_mesh()
        rng = np.random.default_rng(SEED)
        hyper_input = rng.standard_normal((TOKENS, HYPER_SIZE)).astype(np.float32)
        block_output = rng.standard_normal((TOKENS, HIDDEN_SIZE)).astype(np.float32)

        with jax.set_mesh(mesh):
            layer = HyperConnectionBase(_make_config(mesh))
            _, residuals = layer.mix(jnp.array(hyper_input))
            combined = layer.combine(jnp.array(block_output), residuals)

        delta = np.array(combined).reshape(TOKENS, HC_COUNT, HIDDEN_SIZE) - hyper_input.reshape(
            TOKENS, HC_COUNT, HIDDEN_SIZE
        )
        for j in range(HC_COUNT):
            np.testing.assert_allclose(delta[:, j, :], block_output, atol=ATOL, rtol=ATOL)

    def test_carries_no_weights(self):
        mesh = _make_mesh()
        with jax.set_mesh(mesh):
            layer = HyperConnectionBase(_make_config(mesh))
        for attr in ("hc_norm", "input_mix_weight_down", "block_inject_weight"):
            self.assertFalse(hasattr(layer, attr), attr)

    def test_registry_exposes_both_variants(self):
        self.assertIs(HYPERCONNECTION_CLASS_DICT["hyperconnection_average"], HyperConnectionBase)
        self.assertIs(HYPERCONNECTION_CLASS_DICT["gated_residual_simple"], GatedResidual)


class TestGatedResidual(CustomTestCase):
    """mix/combine numerics and the invariants the backbone relies on."""

    def test_weight_shapes_match_the_checkpoint(self):
        """Qwen3.8-Flash-Next ships hc_norm.weight at [HC*HS], i.e. per branch;
        the mixer instance ships without block_inject_weight at all."""
        mesh = _make_mesh()
        rng = np.random.default_rng(SEED)
        layer = _make_layer(_make_weights(rng), mesh)

        self.assertEqual(layer.hc_norm.weight[...].shape, (HYPER_SIZE,))
        self.assertEqual(layer.input_mix_weight_down.weight[...].shape, (HYPER_SIZE, HC_LOWRANK))
        self.assertEqual(layer.input_mix_weight_up.weight[...].shape, (HC_LOWRANK, HYPER_SIZE))
        self.assertEqual(layer.block_inject_weight.weight[...].shape, (HYPER_SIZE, HC_COUNT))

        mixer = _make_layer(_make_weights(rng), mesh, use_combine=False)
        self.assertFalse(hasattr(mixer, "block_inject_weight"))

    def test_shared_norm_weight_is_one_stream_wide(self):
        mesh = _make_mesh()
        rng = np.random.default_rng(SEED)
        layer = _make_layer(_make_weights(rng, per_branch=False), mesh, per_branch=False)

        self.assertEqual(layer.hc_norm.weight[...].shape, (HIDDEN_SIZE,))

    def test_mix_matches_reference(self):
        for per_branch in (True, False):
            with self.subTest(hc_per_branch_norm=per_branch):
                mesh = _make_mesh()
                rng = np.random.default_rng(SEED)
                weights = _make_weights(rng, per_branch)
                hyper_input = rng.standard_normal((TOKENS, HYPER_SIZE)).astype(np.float32)

                layer = _make_layer(weights, mesh, per_branch)
                with jax.set_mesh(mesh):
                    mixed, _ = layer.mix(jnp.array(hyper_input))

                expected, _ = _ref_mix(
                    hyper_input, weights["norm"], weights["down"], weights["up"], per_branch
                )
                self.assertEqual(mixed.shape, (TOKENS, HIDDEN_SIZE))
                np.testing.assert_allclose(np.array(mixed), expected, atol=ATOL, rtol=RTOL)

    def test_combine_matches_reference(self):
        mesh = _make_mesh()
        rng = np.random.default_rng(SEED)
        weights = _make_weights(rng)
        hyper_input = rng.standard_normal((TOKENS, HYPER_SIZE)).astype(np.float32)
        block_output = rng.standard_normal((TOKENS, HIDDEN_SIZE)).astype(np.float32)

        layer = _make_layer(weights, mesh)
        with jax.set_mesh(mesh):
            _, residuals = layer.mix(jnp.array(hyper_input))
            combined = layer.combine(jnp.array(block_output), residuals)

        _, normed_flat = _ref_mix(
            hyper_input, weights["norm"], weights["down"], weights["up"], True
        )
        expected = _ref_combine(hyper_input, normed_flat, block_output, weights["inject"])
        self.assertEqual(combined.shape, (TOKENS, HYPER_SIZE))
        np.testing.assert_allclose(np.array(combined), expected, atol=ATOL, rtol=RTOL)

    def test_combine_leaves_streams_untouched_when_block_output_is_zero(self):
        """Streams are only ever added to, never rewritten."""
        mesh = _make_mesh()
        rng = np.random.default_rng(SEED)
        hyper_input = rng.standard_normal((TOKENS, HYPER_SIZE)).astype(np.float32)

        layer = _make_layer(_make_weights(rng), mesh)
        with jax.set_mesh(mesh):
            _, residuals = layer.mix(jnp.array(hyper_input))
            combined = layer.combine(jnp.zeros((TOKENS, HIDDEN_SIZE), jnp.float32), residuals)

        np.testing.assert_allclose(np.array(combined), hyper_input, atol=ATOL, rtol=RTOL)

    def test_streams_differ_only_by_a_scalar_per_stream(self):
        """Every stream receives the same block output, scaled by its own scalar.

        The whole backbone rests on this: the HC streams are HC differently
        weighted accumulations of one sequence of block outputs, not HC
        independent computations.
        """
        mesh = _make_mesh()
        rng = np.random.default_rng(SEED)
        hyper_input = rng.standard_normal((TOKENS, HYPER_SIZE)).astype(np.float32)
        block_output = rng.standard_normal((TOKENS, HIDDEN_SIZE)).astype(np.float32)

        layer = _make_layer(_make_weights(rng), mesh)
        with jax.set_mesh(mesh):
            _, residuals = layer.mix(jnp.array(hyper_input))
            combined = layer.combine(jnp.array(block_output), residuals)

        delta = (np.array(combined) - hyper_input).reshape(TOKENS, HC_COUNT, HIDDEN_SIZE)
        # delta[:, j, :] == scale[:, j] * block_output, so every stream's delta is
        # parallel to block_output; recover the scalar and check it reproduces delta.
        scale = delta[:, :, :1] / block_output[:, None, :1]
        np.testing.assert_allclose(delta, scale * block_output[:, None, :], atol=ATOL, rtol=1e-4)
        # The scalars are 2*sigmoid(.): they span (0, 2) and are not all equal.
        self.assertTrue(np.all(scale > 0.0))
        self.assertTrue(np.all(scale < 2.0))
        self.assertFalse(np.allclose(scale[:, 0], scale[:, 1]))

    def test_mixer_matches_a_full_instances_mix(self):
        """The model's final mixer is use_combine=False, not a separate class."""
        mesh = _make_mesh()
        rng = np.random.default_rng(SEED)
        weights = _make_weights(rng)
        hyper_input = rng.standard_normal((TOKENS, HYPER_SIZE)).astype(np.float32)

        full = _make_layer(weights, mesh)
        mixer = _make_layer(weights, mesh, use_combine=False)
        with jax.set_mesh(mesh):
            from_full, _ = full.mix(jnp.array(hyper_input))
            from_mixer, _ = mixer.mix(jnp.array(hyper_input))

        np.testing.assert_array_equal(np.array(from_mixer), np.array(from_full))
        with self.assertRaisesRegex(RuntimeError, "use_combine=False"), jax.set_mesh(mesh):
            mixer.combine(jnp.zeros((TOKENS, HIDDEN_SIZE), jnp.float32), (None, None))

    def test_leading_dims_are_preserved(self):
        mesh = _make_mesh()
        rng = np.random.default_rng(SEED)
        weights = _make_weights(rng)
        hyper_input = rng.standard_normal((2, TOKENS, HYPER_SIZE)).astype(np.float32)
        block_output = rng.standard_normal((2, TOKENS, HIDDEN_SIZE)).astype(np.float32)

        layer = _make_layer(weights, mesh)
        with jax.set_mesh(mesh):
            mixed, residuals = layer.mix(jnp.array(hyper_input))
            combined = layer.combine(jnp.array(block_output), residuals)

        self.assertEqual(mixed.shape, (2, TOKENS, HIDDEN_SIZE))
        self.assertEqual(combined.shape, (2, TOKENS, HYPER_SIZE))

        expected_mixed, normed_flat = _ref_mix(
            hyper_input, weights["norm"], weights["down"], weights["up"], True
        )
        expected_combined = _ref_combine(hyper_input, normed_flat, block_output, weights["inject"])
        np.testing.assert_allclose(np.array(mixed), expected_mixed, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(np.array(combined), expected_combined, atol=ATOL, rtol=RTOL)

    def test_bfloat16_roundtrips(self):
        """The serving config: bf16 weights and bf16 activations in, bf16 out.

        Nothing casts on the way out, so every step has to carry bf16 through
        for the residual stream to stay bf16.
        """
        mesh = _make_mesh()
        rng = np.random.default_rng(SEED)
        weights = _make_weights(rng)
        hyper_input = rng.standard_normal((TOKENS, HYPER_SIZE)).astype(np.float32)
        block_output = rng.standard_normal((TOKENS, HIDDEN_SIZE)).astype(np.float32)

        with jax.set_mesh(mesh):
            layer = GatedResidual(
                HyperConnectionConfig(
                    hc_count=HC_COUNT,
                    hidden_size=HIDDEN_SIZE,
                    params_dtype=jnp.bfloat16,
                    hc_lowrank=HC_LOWRANK,
                    rms_norm_eps=EPSILON,
                    hc_per_branch_norm=True,
                    mesh=mesh,
                )
            )
        _assign(layer.hc_norm.weight, weights["norm"], mesh, P(None))
        _assign(layer.input_mix_weight_down.weight, weights["down"], mesh, P(None, None))
        _assign(layer.input_mix_weight_up.weight, weights["up"], mesh, P(None, None))
        _assign(layer.block_inject_weight.weight, weights["inject"], mesh, P(None, None))

        with jax.set_mesh(mesh):
            mixed_bf16, residuals = layer.mix(jnp.array(hyper_input, jnp.bfloat16))
            combined_bf16 = layer.combine(jnp.array(block_output, jnp.bfloat16), residuals)

        self.assertEqual(mixed_bf16.dtype, jnp.bfloat16)
        self.assertEqual(combined_bf16.dtype, jnp.bfloat16)

        expected_mixed, normed_flat = _ref_mix(
            hyper_input, weights["norm"], weights["down"], weights["up"], True
        )
        expected_combined = _ref_combine(hyper_input, normed_flat, block_output, weights["inject"])
        np.testing.assert_allclose(
            np.array(mixed_bf16, dtype=np.float32), expected_mixed, atol=3e-2, rtol=3e-2
        )
        np.testing.assert_allclose(
            np.array(combined_bf16, dtype=np.float32),
            expected_combined,
            atol=3e-2,
            rtol=3e-2,
        )

    def test_rejects_mismatched_widths(self):
        mesh = _make_mesh()
        rng = np.random.default_rng(SEED)
        layer = _make_layer(_make_weights(rng), mesh)

        with jax.set_mesh(mesh):
            with self.assertRaisesRegex(ValueError, str(HYPER_SIZE)):
                layer.mix(jnp.zeros((TOKENS, HYPER_SIZE + HIDDEN_SIZE), jnp.float32))

            _, residuals = layer.mix(jnp.zeros((TOKENS, HYPER_SIZE), jnp.float32))
            with self.assertRaisesRegex(ValueError, "block output"):
                layer.combine(jnp.zeros((TOKENS, HYPER_SIZE), jnp.float32), residuals)

    def test_rejects_missing_mesh(self):
        with self.assertRaisesRegex(ValueError, "mesh"):
            GatedResidual(HyperConnectionConfig(hc_count=HC_COUNT, hidden_size=HIDDEN_SIZE))


if __name__ == "__main__":
    unittest.main()
